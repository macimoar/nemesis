import random

import chess
import numpy as np

from nemesis import timecontrol
from nemesis.features import INDEX, N_FEATURES
from nemesis.learn import record_game
from nemesis.profile import timecontrols
from nemesis.profile.move_model import MoveModel, Sample, load_player_model, prior_vector, train_player
from nemesis.timecontrol import TimeControl

BLITZ = TimeControl(180, 2)


def make_samples(store, games_per_group=14, differ=True, seed=0):
    """Positions with 4 moves each: one capture (losing 2 pawns), three quiet moves. In
    blitz this player grabs the capture; in daily games they don't (unless differ=False)."""
    rng = random.Random(seed)
    samples = []
    for i in range(2 * games_per_group):
        blitz = i % 2 == 0
        gid = record_game(store, "me", chess.WHITE, ["e2e4", "e7e5"], result="1-0",
                          tc=BLITZ if blitz else None)
        for ply in range(20):
            X = np.zeros((4, N_FEATURES))
            X[0, INDEX["capture"]] = 1
            X[0, INDEX["cp_loss"]] = 2.0
            grabs = (blitz or not differ) and rng.random() < 0.8
            chosen = 0 if grabs else rng.randrange(1, 4)
            samples.append(Sample(X=X, chosen=chosen, game_id=gid, ply=ply,
                                  loss=200 if chosen == 0 else 0, moves=["a", "b", "c", "d"]))
    return samples


def test_groups():
    assert timecontrol.profile_group("classical") == "rapid"
    assert timecontrol.profile_group("untimed") == timecontrol.profile_group(None) == "untimed"
    assert timecontrol.profile_group("bullet") == "bullet"


def test_a_time_control_that_plays_differently_gets_its_own_profile(store):
    samples = make_samples(store)
    model, metrics = train_player(store, "me", samples)
    tcs = metrics["time_controls"]
    assert tcs["blitz"]["games"] == 14 and tcs["untimed"]["games"] == 14
    assert tcs["blitz"]["active"] and tcs["untimed"]["active"]
    h = tcs["blitz"]["holdout"]
    assert h["own"]["nll"] < h["main"]["nll"]
    assert not tcs["bullet"]["active"] and tcs["bullet"]["reason"].startswith("needs")
    # Playing blitz, Nemesis predicts this player with their blitz profile: they grab.
    blitz = load_player_model(store, "me", tc="blitz")
    daily = load_player_model(store, "me", tc="untimed")
    assert blitz.to_dict()["capture"] > model.to_dict()["capture"] > daily.to_dict()["capture"]
    assert load_player_model(store, "me", tc="bullet").to_dict() == model.to_dict()   # falls back


def test_no_profile_where_play_is_the_same(store):
    samples = make_samples(store, differ=False)
    _, metrics = train_player(store, "me", samples)
    assert not metrics["time_controls"]["blitz"]["active"]
    assert "didn't predict" in metrics["time_controls"]["blitz"]["reason"]
    assert timecontrols.load(store, "me", "blitz") is None


def test_differences_beyond_chance(store):
    samples = make_samples(store)
    diffs = timecontrols.differences(store, samples)
    assert set(diffs) == {"blitz", "untimed"}
    blitz = diffs["blitz"]
    measures = {n["measure"] for n in blitz["notes"]}
    assert {"capture", "loss"} <= measures
    loss = next(n for n in blitz["notes"] if n["measure"] == "loss")
    assert loss["here"] > loss["there"] and "in your daily & untimed games" in loss["text"]
    assert diffs["untimed"]["notes"] == []          # said once, not mirrored
    same = timecontrols.differences(store, make_samples(store, differ=False, seed=1))
    assert all(not d["notes"] for d in same.values())
