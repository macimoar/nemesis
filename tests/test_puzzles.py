import chess

from nemesis import puzzles
from nemesis.learn import record_game, train


# White to move; Black's queen on d4 is free for the knight on f3.
QUEEN_FREE = "4k3/8/8/8/3q4/5N2/8/4K3 w - - 0 20"


def test_make_puzzle_finds_the_free_queen(analyzer):
    board = chess.Board(QUEEN_FREE)
    lines = analyzer.lines(board, multipv=6, extra=["e1e2"])
    p = puzzles.make_puzzle(board, "e1e2", lines, analyzer, chance=0.2)
    assert p is not None
    assert p["solution_san"] == "Nxd4" and "material" in p["themes"]
    assert p["played_san"] == "Ke2" and p["loss"] >= 150
    assert p["line"] == ["f3d4"]


def test_no_puzzle_when_you_played_the_best_move_or_nothing_stands_out(analyzer):
    board = chess.Board(QUEEN_FREE)
    lines = analyzer.lines(board, multipv=6)
    assert puzzles.make_puzzle(board, "f3d4", lines, analyzer) is None
    quiet = chess.Board()
    assert puzzles.make_puzzle(quiet, "a2a3", analyzer.lines(quiet), analyzer) is None


def test_mate_theme(analyzer):
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    lines = analyzer.lines(board, multipv=6, extra=["g1f1"])
    p = puzzles.make_puzzle(board, "g1f1", lines, analyzer)
    assert p["solution_san"] == "Ra8#" and "mate" in p["themes"]


def _game_with_missed_queen(store):
    # 1.e4 e5 2.Nf3 Qg5 3.d3?? -- you are White and didn't take the queen with Nxg5.
    return record_game(store, "me", chess.WHITE, ["e2e4", "e7e5", "g1f3", "d8g5", "d2d3"], result="*")


def test_training_mines_puzzles_and_attempts_are_recorded(store, analyzer):
    _game_with_missed_queen(store)
    _, metrics = train(store, "me", analyzer)
    assert metrics["new_puzzles"] >= 1
    s = puzzles.summary(store, "me")
    assert s["total"] == metrics["new_puzzles"] and s["new"] == s["total"]
    p = puzzles.next_puzzle(store, "me")
    shown = puzzles.public(p)
    assert "solution" not in shown and "line" not in shown and shown["hint"]
    wrong = next(m.uci() for m in chess.Board(p["fen"]).legal_moves if m.uci() != p["solution"])
    res = puzzles.attempt(store, p["id"], wrong)
    assert not res["solved"] and res["explanation"][0].startswith("The move was")
    assert any("you played" in line for line in res["explanation"])
    assert puzzles.summary(store, "me")["missed"] == 1
    res = puzzles.attempt(store, p["id"], p["solution"])
    assert res["solved"]
    s = puzzles.summary(store, "me")
    assert s["solved"] == 1 and s["first_try"] == 0
    # Retraining doesn't duplicate puzzles.
    _, metrics = train(store, "me", analyzer)
    assert metrics["new_puzzles"] == 0


def test_order_new_first_then_missed_then_solved(store):
    for i, chance in enumerate([0.5, 0.1, 0.3]):
        store.put_puzzle(f"g#{i}", "me", "g", i, chess.STARTING_FEN, "e2e4",
                         {"themes": ["best"], "chance": chance, "color": "white", "move_number": 1,
                          "played_san": "a3"})
    assert puzzles.next_puzzle(store, "me")["id"] == "g#1"          # hardest new one
    store.add_puzzle_attempt("g#1", "a2a3", False, False)
    store.add_puzzle_attempt("g#2", "e2e4", True, False)
    assert puzzles.next_puzzle(store, "me")["id"] == "g#0"          # still new
    store.add_puzzle_attempt("g#0", "e2e4", True, False)
    assert puzzles.next_puzzle(store, "me")["id"] == "g#1"          # the missed one
    assert puzzles.next_puzzle(store, "me", skip="g#1")["id"] in {"g#0", "g#2"}


def test_puzzle_api(tmp_path, analyzer):
    from urllib.parse import quote
    from fastapi.testclient import TestClient
    from nemesis.store import Store
    from nemesis.web.app import create_app
    db = tmp_path / "t.db"
    store = Store(db)
    _game_with_missed_queen(store)
    train(store, "me", analyzer)
    store.close()
    c = TestClient(create_app(db))
    assert c.get("/api/puzzles/me").json()["new"] >= 1
    p = c.get("/api/puzzles/me/next").json()["puzzle"]
    assert "solution" not in p and p["legal"] and p["color"] == "white"
    res = c.post(f"/api/puzzle/{quote(p['id'], safe='')}/attempt", json={"uci": "f3g5"}).json()
    assert res["solved"] and res["fens"] and res["line"][0] == "f3g5"
    assert c.get("/api/puzzles/me").json()["solved"] == 1
    assert c.post("/api/puzzle/nope/attempt", json={"uci": "e2e4"}).status_code == 404
    assert c.get("/api/puzzles/nobody/next").json() == {"puzzle": None}
