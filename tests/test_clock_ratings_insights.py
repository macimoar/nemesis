import sqlite3

import chess
import pytest

from nemesis import ratings, timecontrol
from nemesis.engine import search_settings
from nemesis.ingest.pgn import import_pgn_text
from nemesis.insights import insights
from nemesis.learn import record_game
from nemesis.store import Store
from nemesis.timecontrol import TimeControl

BLITZ_PGN = """[Event "Rated Blitz game"]
[Site "https://lichess.org/blitz0001"]
[White "me"]
[Black "x"]
[WhiteElo "1510"]
[TimeControl "180+2"]
[Result "1-0"]
[UTCDate "2026.02.01"]

1. e4 e5 1-0

[Event "Rated Rapid game"]
[Site "https://lichess.org/rapid0001"]
[White "me"]
[Black "y"]
[WhiteElo "1620"]
[TimeControl "600+0"]
[Result "1-0"]
[UTCDate "2026.02.02"]

1. d4 d5 1-0
"""


def test_time_control_categories_and_parsing():
    assert timecontrol.parse("3+2") == TimeControl(180, 2)
    assert timecontrol.parse("180+2").category == "blitz"
    assert timecontrol.parse("60").category == "bullet"
    assert timecontrol.parse("600").category == "rapid"
    assert timecontrol.parse("1800+20").category == "classical"
    assert timecontrol.parse("1/86400") is None and timecontrol.parse("-") is None
    assert timecontrol.category(None) == "untimed"
    assert TimeControl(180, 2).label == "3+2" and TimeControl(180, 2).pgn() == "180+2"


def test_think_budget_never_risks_the_clock():
    assert timecontrol.think_budget(None) is None
    for left in (0.5, 5, 30, 60, 300):
        for inc in (0, 2, 10):
            b = timecontrol.think_budget(left, inc)
            assert 0 < b <= max(0.1, left / 4)
    assert search_settings(None, True) == (None, 6, 8, True)        # untimed: full search
    assert search_settings(0.5, True)[0] == 4                       # bullet: shallow and quick


def test_ratings_split_by_time_control(store):
    import_pgn_text(store, BLITZ_PGN, "me", source="lichess")
    assert ratings.player_rating(store, "me", "blitz") == (1510.0, 0)
    assert ratings.player_rating(store, "me", "rapid") == (1620.0, 0)
    assert ratings.player_rating(store, "me", "bullet") == (1620.0, 0)   # none there: latest rating
    assert ratings.target_elo(store, "me", tc="blitz") == 1560.0
    moves = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]
    record_game(store, "me", chess.WHITE, moves, risk=0.5, tc=TimeControl(180, 2))
    assert ratings.player_rating(store, "me", "blitz")[1] == 1
    assert ratings.player_rating(store, "me", "rapid")[1] == 0          # other time controls untouched
    assert ratings.parse_entity(ratings.nemesis_entity("me", "steering", "blitz")) == \
        {"kind": "nemesis", "tier": "steering", "tc": "blitz"}


def test_recorded_clocks_and_time_control(store):
    moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
    gid = record_game(store, "me", chess.WHITE, moves, result="*", tc=TimeControl(60, 0),
                      clocks=[58.0, 59.5, 41.2, 55.0])
    game = store.game(gid)
    assert '[TimeControl "60+0"]' in game.pgn and game.time_class == "Nemesis bullet"
    assert [m.clock for m in store.moves(gid)] == [58.0, 59.5, 41.2, 55.0]


def test_migration_moves_old_ratings_to_untimed(tmp_path):
    path = tmp_path / "old.db"
    old = Store(path)
    old.conn.execute("PRAGMA user_version = 0")
    old.conn.execute("INSERT INTO games VALUES ('g','me',1,1.0,'nemesis',NULL,NULL,NULL,NULL,'')")
    old.conn.executemany("INSERT INTO ratings VALUES (?,?,?)",
                         [("player:me", 1300, 4), ("nemesis:me:trapper", 1400, 4)])
    old.conn.execute("INSERT INTO rating_history VALUES ('g','player:me',1290,1300)")
    old.conn.commit()
    old.close()

    s = Store(path)  # opening runs the migration
    assert s.conn.execute("PRAGMA user_version").fetchone()[0] >= 1
    assert ratings.player_rating(s, "me") == (1300, 4)
    assert ratings.nemesis_rating(s, "me", "trapper") == (1400, 4)
    assert s.rating_change("g") == {"player:me:untimed": (1290, 1300)}
    backup = sqlite3.connect(tmp_path / "old.db.backup-v0")   # the pre-migration copy was kept
    assert backup.execute("SELECT entity FROM ratings ORDER BY entity").fetchall()[0][0] == "nemesis:me:trapper"
    backup.close()


def annotation(ply, top, probs, trap=None, loss=None):
    return {"ply": ply, "move": "", "risk": 0.5, "elo": 1400, "lapse": False, "top": top, "probs": probs,
            "trap": trap, **({"reply": {"uci": "", "loss": loss}} if loss is not None else {})}


def test_insights_counts_predictions_and_traps(store):
    assert insights(store, "me")["games"] == 0
    # Nemesis is Black; the player's replies are at plies 2 and 4.
    moves = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]
    trap = {"uci": "d1h5", "prob": 0.3, "loss": 150, "reasons": []}
    anns = [annotation(1, "d1h5", {"d1h5": 0.4}, trap=trap, loss=150),     # predicted, trap sprung
            annotation(3, "g1f3", {"f1c4": 0.2}, loss=0),                  # missed, no mistake
            annotation(5, "h5f7", {"h5f7": 0.9}, loss=0)]                  # predicted
    for _ in range(5):
        record_game(store, "me", chess.WHITE, moves, risk=0.5, annotations=anns)
    d = insights(store, "me")
    assert d["games"] == 5 and d["predictions"] == 15 and d["predicted"] == 10
    assert d["prediction_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert d["traps"] == 5 and d["sprung"] == 5 and d["trap_rate"] == 1.0
    assert d["mistake_rate"] == 0.0 and d["verdict"] == "traps"
    assert len(d["per_game"]) == 5 and d["per_game"][0]["predicted"] == 2
