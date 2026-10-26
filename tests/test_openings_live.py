import sqlite3

import chess
import pytest

from nemesis import openings
from nemesis.learn import record_game
from nemesis.store import Store


def board(*ucis):
    b = chess.Board()
    for u in ucis:
        b.push_uci(u)
    return b


def test_names_by_position_including_transpositions():
    italian = openings.name_of(board("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"))
    assert italian == {"eco": "C50", "name": "Italian Game: Giuoco Piano", "ply": 6}
    assert openings.name_of(board("g1f3", "b8c6", "e2e4", "e7e5", "f1c4", "f8c5"))["name"] == italian["name"]
    # Leaving the known lines keeps the last name.
    later = openings.name_of(board("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "a2a3", "a7a6"))
    assert later["name"] == italian["name"] and later["ply"] == 6
    assert openings.name_of(chess.Board()) is None
    assert openings.family("Italian Game: Giuoco Piano") == "Italian Game"


def test_nemesis_games_get_opening_names(store):
    gid = record_game(store, "me", chess.WHITE, ["e2e4", "c7c5", "g1f3"], result="*")
    assert store.game(gid).opening.startswith("Sicilian Defense")


def test_migration_renames_old_games(tmp_path):
    db = tmp_path / "old.db"
    s = Store(db)
    gid = record_game(s, "me", chess.WHITE, ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"], result="*")
    s.conn.execute("UPDATE games SET opening = 'Kings Pawn Opening Kings Knight Variation' WHERE id = ?", (gid,))
    s.conn.execute("PRAGMA user_version = 1")
    s.conn.commit()
    s.close()
    s = Store(db)                                       # opening it migrates, after a backup
    assert s.game(gid).opening == "Italian Game: Giuoco Piano"
    assert (tmp_path / "old.db.backup-v1").exists()
    assert s.conn.execute("PRAGMA user_version").fetchone()[0] >= 2


@pytest.fixture
def client(tmp_path, analyzer, monkeypatch):
    from fastapi.testclient import TestClient
    from nemesis.web import app as webapp
    monkeypatch.setattr(webapp, "StockfishAnalyzer", lambda *a, **k: analyzer)
    monkeypatch.setattr(webapp, "ensure_stockfish", lambda *a, **k: "stockfish")
    return TestClient(webapp.create_app(tmp_path / "t.db"))


def test_game_state_has_an_eval_per_ply_and_the_opening(client):
    g = client.post("/api/games", json={"player": "me", "color": "white"}).json()
    assert g["evals"] == [] and g["opening"] is None
    g = client.post(f"/api/games/{g['id']}/move", json={"uci": "e2e4"}).json()
    assert len(g["evals"]) == 2                          # your move and Nemesis's reply
    assert all(isinstance(v, int) for v in g["evals"])
    assert g["opening"]["eco"][0] in "BC"                # 1.e4 openings are B or C
