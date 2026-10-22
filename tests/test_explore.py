import chess
import pytest

from nemesis import commentary
from nemesis.engine import Nemesis

# White to move, Black's queen on d4 is free for the knight: Ke2 instead is a big mistake.
QUEEN_FREE = "4k3/8/8/8/3q4/5N2/8/4K3 w - - 0 20"


def test_xray_mistakes_get_the_better_move(store, analyzer):
    nem = Nemesis(store, "me", analyzer=analyzer, mode="xray")
    before = chess.Board("4k3/8/8/2q5/8/5N2/8/4K3 b - - 0 20")
    decision = nem.choose(before)
    board = before.copy()
    board.push(decision.move)
    losses = {m.uci(): max(analyzer.lines(board).values()) - analyzer.lines(board, extra=[m.uci()])[m.uci()]
              for m in board.legal_moves}
    worst = chess.Move.from_uci(max(losses, key=losses.get))
    if losses[worst.uci()] < 100:
        pytest.skip("no mistake available after this Nemesis move")
    lines = nem.react(board, decision, worst)
    assert any(l.startswith("**Better was:**") for l in lines)
    assert nem.drill and nem.drill["played"] == worst.uci() and nem.drill["fen"] == board.fen()
    assert nem.drill["best"] != worst.uci() and nem.drill["loss"] >= 100
    nem.mode = commentary.POKER                     # Poker Face gives nothing away
    assert len(nem.react(board, decision, worst)) == 1 and nem.drill is None


def test_better_move_wording():
    board = chess.Board(QUEEN_FREE)
    lines = commentary.better_move(board, chess.Move.from_uci("f3d4"), "it left your knight on f3 open to capture")
    assert lines[0] == "**Better was:** Nxd4 (knight takes the queen on d4)."
    assert lines[1].startswith("**What went wrong:** Your move: it left")
    assert len(commentary.better_move(board, chess.Move.from_uci("f3d4"), None)) == 1


@pytest.fixture
def client(tmp_path, analyzer, monkeypatch):
    from fastapi.testclient import TestClient
    from nemesis.web import app as webapp
    monkeypatch.setattr(webapp, "StockfishAnalyzer", lambda *a, **k: analyzer)
    monkeypatch.setattr(webapp, "ensure_stockfish", lambda *a, **k: "stockfish")
    return TestClient(webapp.create_app(tmp_path / "t.db"))


def test_explore_lines(client):
    res = client.post("/api/explore/lines", json={"fen": QUEEN_FREE, "played": "e1e2", "best": "f3d4"}).json()
    assert res["you"] == "white" and res["legal"]["f3"]
    assert res["played"]["steps"][0]["san"] == "Ke2" and res["played"]["steps"][0]["you"]
    assert res["best"]["steps"][0]["san"] == "Nxd4" and res["best"]["steps"][0]["fen"]
    assert res["best"]["cp"] > res["played"]["cp"] + 500
    bad = client.post("/api/explore/lines", json={"fen": QUEEN_FREE, "played": "e1e3", "best": "f3d4"})
    assert bad.status_code == 400


def test_explore_your_own_move(client):
    with_pawns = "4k3/pp6/8/8/3q4/5N2/PP6/4K3 w - - 0 20"
    res = client.post("/api/explore/move", json={"fen": with_pawns, "uci": "f3d4"}).json()
    you, reply = res["steps"]
    assert you["san"] == "Nxd4" and you["you"] and not reply["you"]
    assert res["cp"] > 0 and res["legal"]                      # you're up a queen, and it's your move
    after = client.post("/api/explore/move", json={"fen": with_pawns, "uci": "e1e2"}).json()
    assert after["cp"] < 0                                       # the queen stays, you're worse


def test_explore_a_move_that_ends_the_game(client):
    res = client.post("/api/explore/move", json={"fen": QUEEN_FREE, "uci": "f3d4"}).json()
    assert res["over"] and res["cp"] == 0 and len(res["steps"]) == 1     # knight vs king: a draw
