import chess

from nemesis import repairs
from nemesis.learn import record_game, train

# 1.e4 d5 2.Qg4?? -- the bishop takes the queen. You keep doing it.
HABIT = ["e2e4", "d7d5", "d1g4", "c8g4"]


def _games(store, n=3):
    for _ in range(n):
        record_game(store, "me", chess.WHITE, HABIT, result="0-1")


def test_a_repeated_costly_move_becomes_a_spot(store, analyzer):
    _games(store)
    _, metrics = train(store, "me", analyzer)
    assert metrics["repair_spots"] == 1
    (spot,) = store.repairs("me")
    assert spot["habit_san"] == "Qg4" and spot["times"] == 3 and spot["visits"] == 3
    assert spot["line"] == ["e2e4", "d7d5"] and spot["color"] == "white" and spot["move_number"] == 2
    assert spot["habit"] not in spot["good"] and spot["loss"] >= 500
    assert spot["score"] == 0 and "queen" in spot["wrong"]


def test_a_one_off_mistake_is_not_a_habit(store, analyzer):
    _games(store, n=1)
    train(store, "me", analyzer)
    assert store.repairs("me") == []


def test_drill_until_learned(store, analyzer):
    _games(store)
    train(store, "me", analyzer)
    s = repairs.next_spot(store, "me")
    shown = repairs.public(store, "me", s)
    assert "good" not in shown and "best" not in shown
    assert shown["intro"]["line"] == ["e2e4", "d7d5"] and shown["fen"] == shown["intro"]["fens"][-1]
    res = repairs.attempt(store, "me", s["id"], "d1g4")
    assert not res["solved"] and res["explanation"][0].startswith("That's your usual move again")
    assert repairs.next_spot(store, "me")["id"] == s["id"]          # a miss comes straight back
    good = s["good"][0]
    res = repairs.attempt(store, "me", s["id"], good)
    assert res["solved"] and res["streak"] == 1 and "rests for an hour" in res["explanation"][-1]
    assert repairs.next_spot(store, "me") is None                     # resting
    assert repairs.summary(store, "me")["resting"] == 1
    res = repairs.attempt(store, "me", s["id"], good)                 # too soon: doesn't count
    assert res["solved"] and res["streak"] == 1 and "only count once" in res["explanation"][-1]


def test_spaced_answers_learn_and_a_miss_resets(store, analyzer):
    _games(store)
    train(store, "me", analyzer)
    key, good = store.repairs("me")[0]["id"], store.repairs("me")[0]["good"][0]
    for at in ("2026-01-01 10:00:00", "2026-01-01 10:20:00", "2026-01-01 11:05:00"):
        store.add_repair_attempt("me", key, good, True, at=at)
    assert repairs._streak(store.repair_attempts("me")[key]) == 2      # 10:20 was too soon
    store.add_repair_attempt("me", key, "d1g4", False, at="2026-01-01 13:00:00")
    assert repairs._streak(store.repair_attempts("me")[key]) == 0
    for at in ("2026-01-02 09:00:00", "2026-01-02 10:00:00", "2026-01-03 09:00:00"):
        store.add_repair_attempt("me", key, good, True, at=at)
    summ = repairs.summary(store, "me")
    assert summ["learned"] == 1 and summ["openings"][0]["spots"] == 1 and summ["resting"] == 0
    # Retraining keeps your progress: attempts are keyed by position.
    train(store, "me", analyzer)
    assert repairs.summary(store, "me")["learned"] == 1


def test_repair_api(tmp_path, analyzer):
    from urllib.parse import quote
    from fastapi.testclient import TestClient
    from nemesis.store import Store
    from nemesis.web.app import create_app
    db = tmp_path / "t.db"
    store = Store(db)
    _games(store)
    train(store, "me", analyzer)
    good = store.repairs("me")[0]["good"][0]
    store.close()
    c = TestClient(create_app(db))
    assert c.get("/api/repairs/me").json()["total"] == 1
    spot = c.get("/api/repairs/me/next").json()["spot"]
    assert spot["habit_san"] == "Qg4" and spot["legal"]
    res = c.post("/api/repairs/me/attempt", json={"id": spot["id"], "uci": good}).json()
    assert res["solved"] and res["streak"] == 1
    assert c.post("/api/repairs/me/attempt", json={"id": "nope", "uci": "e2e4"}).status_code == 404
    assert c.get("/api/repairs/nobody/next").json() == {"spot": None}
