import chess

from tests.test_explore import client  # noqa: F401  (the web app, with the test Stockfish)


def start(client, **kw):
    return client.post("/api/games", json={"player": "me", "color": "white", **kw}).json()


def test_a_hint_names_the_piece_not_the_move(client):
    g = start(client)
    assert g["hints_left"] == 3 and g["hint"] is None and g["moves"] == []
    h = client.post(f"/api/games/{g['id']}/hint").json()
    sq = h["hint"]["from"]
    assert h["hints_left"] == 2 and sq in h["legal"]
    text = h["log"][-1]["lines"][0]
    assert text.startswith("**Hint:** Look at your") and sq in text and "unrated" in text
    assert not any(u[2:4] in text for u in h["legal"][sq])     # where it goes stays yours to find
    again = client.post(f"/api/games/{g['id']}/hint").json()   # the same move: no charge
    assert again["hints_left"] == 2 and len(again["log"]) == len(h["log"])


def test_hints_run_out_and_unrate_the_game(client):
    g = start(client)
    for _ in range(3):
        g = client.post(f"/api/games/{g['id']}/hint").json()
        uci = next(u for us in g["legal"].values() for u in us)
        g = client.post(f"/api/games/{g['id']}/move", json={"uci": uci}).json()
    assert g["hints_left"] == 0 and len(g["moves"]) == 6
    assert client.post(f"/api/games/{g['id']}/hint").status_code == 409
    g = client.post(f"/api/games/{g['id']}/resign").json()
    assert "isn't rated" in g["log"][-1]["lines"][0] and "rating:" not in " ".join(g["log"][-1]["lines"])


def test_hinted_moves_are_left_out_of_training(tmp_path, analyzer):
    from nemesis.learn import record_game
    from nemesis.profile.move_model import build_samples
    from nemesis.profile.repertoire import OpeningTree
    from nemesis.store import Store
    store = Store(tmp_path / "h.db")
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]
    gid = record_game(store, "me", chess.WHITE, moves, result="1-0", hints=[2])
    plies = {s.ply for s in build_samples(store, "me", analyzer)}
    assert 2 not in plies and {0, 4} <= plies
    again = {s.ply for s in build_samples(store, "me", analyzer)}      # from the sample cache too
    assert again == plies
    tree = OpeningTree.from_store(store, "me")
    board = chess.Board(); board.push_uci("e2e4"); board.push_uci("e7e5")
    assert tree.stats(board).moves.get("g1f3", 0) == 0
    assert gid and store.hinted_plies(store.games("me")[0]) == {2}


def test_the_analysis_board_has_no_engine(client):
    res = client.post("/api/analysis", json={"fen": chess.STARTING_FEN}).json()
    assert res["step"] is None and res["turn"] == "white" and "e2" in res["legal"]
    res = client.post("/api/analysis", json={"fen": chess.STARTING_FEN, "uci": "e2e4"}).json()
    assert res["step"]["san"] == "e4" and res["turn"] == "black" and "e7" in res["legal"]   # both sides move
    assert not any(k in res for k in ("cp", "eval", "best"))
    mate = client.post("/api/analysis", json={"fen": "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
                                              "uci": "d8h4"}).json()
    assert mate["over"] and mate["result"] == "0-1" and mate["check"]
    assert client.post("/api/analysis", json={"fen": chess.STARTING_FEN, "uci": "e2e5"}).status_code == 400
