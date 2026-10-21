import random

import chess
import pytest

from nemesis.doppelganger import Doppelganger
from nemesis.learn import record_game, train

ITALIAN = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5"]
LONDON = ["d2d4", "d7d5", "c1f4", "g8f6", "e2e3", "e7e6", "g1f3", "f8d6"]


@pytest.fixture
def rival(store, analyzer):
    for _ in range(4):
        record_game(store, "rival", chess.WHITE, ITALIAN, result="1-0")
    record_game(store, "rival", chess.WHITE, LONDON, result="0-1")
    train(store, "rival", analyzer)
    return store


def test_plays_their_openings_as_often_as_they_did(rival, analyzer):
    d = Doppelganger(rival, "rival", analyzer, seed=1)
    firsts = [d.choose(chess.Board())[0].uci() for _ in range(300)]
    share = firsts.count("e2e4") / len(firsts)
    # 4 of 5 games started 1.e4, and with 5 games it follows its book 5/6 of the time.
    assert 0.6 < share < 0.85 and "d2d4" in firsts
    move, lines, choices = d.choose(chess.Board())
    assert choices[0] == {"san": "e4", "prob": 0.8}


def test_out_of_book_it_samples_from_their_model(rival, analyzer):
    d = Doppelganger(rival, "rival", analyzer, seed=2)
    board = chess.Board("4k3/8/8/3q4/8/5N2/8/4K3 w - - 0 30")
    moves = [d.choose(board)[0].uci() for _ in range(60)]
    assert len(set(moves)) > 1                       # it isn't a fixed engine
    _, lines, choices = d.choose(board)
    assert abs(sum(c["prob"] for c in choices) - 1) < 0.3 and "would play" in lines[0]


def test_draw_offers(rival, analyzer):
    d = Doppelganger(rival, "rival", analyzer)
    accept, _, cp = d.consider_draw(chess.Board("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 40"))   # you have a queen
    assert accept and cp < 0 and "Accepted" in d.draw_reply(accept, cp)[0]
    accept, _, _ = d.consider_draw(chess.Board("4k3/8/8/3q4/8/8/8/4K3 w - - 0 40"))    # it has one
    assert not accept


def test_doppelganger_game_in_the_app(tmp_path, analyzer, monkeypatch):
    from fastapi.testclient import TestClient
    from nemesis.store import Store
    from nemesis.web import app as webapp
    db = tmp_path / "t.db"
    store = Store(db)
    for _ in range(3):
        record_game(store, "rival", chess.WHITE, ITALIAN, result="1-0")
    record_game(store, "me", chess.BLACK, ITALIAN, result="0-1")
    train(store, "rival", analyzer)
    store.close()
    monkeypatch.setattr(webapp, "StockfishAnalyzer", lambda *a, **k: analyzer)
    monkeypatch.setattr(webapp, "ensure_stockfish", lambda *a, **k: "stockfish")
    c = TestClient(webapp.create_app(db))
    assert c.post("/api/games", json={"player": "me", "color": "black", "opponent": "nobody"}).status_code == 400
    g = c.post("/api/games", json={"player": "me", "color": "black", "opponent": "rival"}).json()
    assert g["opponent"] == "Doppelgänger of rival" and g["doppel"] == "rival"
    first = g["log"][0]
    assert first["lines"][0].startswith(("**Their opening:**", "**Their style:**")) and first["choices"]
    g = c.post(f"/api/games/{g['id']}/resign").json()
    assert g["finished"] and "aren't rated" in g["log"][-1]["lines"][0]
    gid = g["game_id"]
    assert c.get("/api/ratings/me").json() and Store(db).rating_change(gid) == {}
    assert "Doppelgänger of rival" in c.get(f"/api/pgn/{gid}").text
    listed = {x["id"]: x for x in c.get("/api/games/list/me").json()}
    assert listed[gid]["opponent"] == "Doppelgänger of rival"
