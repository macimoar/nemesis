import chess
import pytest

from nemesis.commentary import POKER, XRAY, draw_reply
from nemesis.engine import Nemesis
from nemesis.learn import build_pgn, record_game
from nemesis.timecontrol import TimeControl


def test_nemesis_declines_when_winning_and_accepts_when_losing(store, analyzer):
    nem = Nemesis(store, "me", analyzer=analyzer)
    # You (White) to move. Nemesis (Black) has an extra queen: it plays on.
    accept, expected, cp = nem.consider_draw(chess.Board("4k3/8/8/3q4/8/8/8/4K3 w - - 0 40"))
    assert not accept and expected > 0.9 and cp > 500
    # You have the extra queen: Nemesis takes the draw.
    accept, expected, cp = nem.consider_draw(chess.Board("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 40"))
    assert accept and expected < 0.1


def test_draw_replies():
    lines = draw_reply(False, 0.62, 140, XRAY)
    assert "Declined" in lines[0] and "62%" in lines[0] and "better" in lines[1]
    assert "Accepted" in draw_reply(True, 0.3, -200, XRAY)[0]
    (poker,) = draw_reply(True, 0.3, -200, POKER, seed="x")
    assert not any(ch.isdigit() for ch in poker)


def test_build_pgn_has_clocks_and_headers():
    game = build_pgn("me", chess.BLACK, ["e2e4", "e7e5"], "*", tc=TimeControl(180, 2), clocks=[178.0, 175.5])
    text = str(game)
    assert '[White "Nemesis"]' in text and '[Black "me"]' in text and '[TimeControl "180+2"]' in text
    assert "[%clk 0:02:58]" in text and "[%clk 0:02:55.5]" in text
    assert "[Date \"????" not in text


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient
    from nemesis.store import Store
    from nemesis.web.app import create_app
    db = tmp_path / "t.db"
    gid = record_game(Store(db), "me", chess.WHITE, ["e2e4", "e7e5", "g1f3"], result="*")
    return TestClient(create_app(db)), gid


def test_pgn_download(client):
    c, gid = client
    resp = c.get(f"/api/pgn/{gid}")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-chess-pgn")
    disposition = resp.headers["content-disposition"]
    assert disposition.startswith("attachment;") and "me-vs-Nemesis" in disposition and disposition.endswith('.pgn"')
    assert "1. e4 e5 2. Nf3" in resp.text
    assert c.get("/api/pgn/nemesis:nope").status_code == 404


@pytest.mark.skipif(not __import__("os").path.isfile(__import__("nemesis.analysis.stockfish", fromlist=["x"]).find_stockfish() or ""),
                    reason="needs Stockfish")
def test_draw_offer_flow(client, monkeypatch):
    c, _ = client
    g = c.post("/api/games", json={"player": "me", "color": "white"}).json()
    monkeypatch.setattr(Nemesis, "consider_draw", lambda self, board: (False, 0.7, 120))
    g = c.post(f"/api/games/{g['id']}/draw").json()
    assert not g["finished"] and g["draw_offer_in"] == 3 and "Declined" in g["log"][-1]["lines"][0]
    assert c.post(f"/api/games/{g['id']}/draw").status_code == 429      # too soon
    move = next(iter(g["legal"].values()))[0]
    g = c.post(f"/api/games/{g['id']}/move", json={"uci": move}).json()
    assert g["draw_offer_in"] == 2
    monkeypatch.setattr(Nemesis, "consider_draw", lambda self, board: (True, 0.3, -150))
    g["draw_offer_in"] = 0
    for _ in range(2):   # play on until the next offer is allowed
        move = next(iter(g["legal"].values()))[0]
        g = c.post(f"/api/games/{g['id']}/move", json={"uci": move}).json()
    g = c.post(f"/api/games/{g['id']}/draw").json()
    assert g["finished"] and g["result"] == "1/2-1/2" and g["reason"] == "Draw agreed."
    assert g["game_id"] and "Accepted" in g["log"][-2]["lines"][0]
