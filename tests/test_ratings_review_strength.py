import random

import chess
import pytest

from nemesis import ratings, strength
from nemesis.ingest.pgn import import_pgn_text
from nemesis.learn import record_game
from nemesis.review import game_accuracy, label_for, move_accuracy, review_game

RATED_PGN = """[Event "Rated Rapid game"]
[Site "https://lichess.org/rated001"]
[White "me"]
[Black "someone"]
[WhiteElo "1345"]
[BlackElo "1400"]
[Result "1-0"]
[UTCDate "2026.02.01"]

1. e4 e5 1-0
"""


# -- strength -----------------------------------------------------------------

def test_strength_params_monotonic():
    elos = range(strength.MIN_ELO, strength.MAX_ELO + 1, 100)
    acpl = [strength.params(e)[0] for e in elos]
    lapses = [strength.params(e)[1] for e in elos]
    assert acpl == sorted(acpl, reverse=True) and lapses == sorted(lapses, reverse=True)
    assert strength.params(strength.MAX_ELO) == (0.0, 0.0)
    assert not strength.limited(strength.MAX_ELO) and strength.limited(1500) and not strength.limited(None)


def test_lower_targets_lose_more_centipawns():
    losses = [0, 15, 40, 80, 150, 250, 350]
    scores = [0.6, 0.59, 0.57, 0.55, 0.5, 0.45, 0.4]

    def avg_loss(elo):
        rng = random.Random(7)
        return sum(losses[strength.pick(losses, scores, elo, rng)[0]] for _ in range(400)) / 400

    weak, mid, strong = avg_loss(800), avg_loss(1600), avg_loss(2400)
    assert weak > mid > strong == 0
    # the per-position temperature hits the target loss on average
    assert abs(avg_loss(1400) - strength.params(1400)[0]) < 25


def test_engine_plays_weaker_at_low_target(store, analyzer):
    from nemesis.search import Searcher
    from nemesis.profile.move_model import MoveModel
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    lines = analyzer.lines(board, multipv=12)
    best = max(lines.values())
    full = Searcher(analyzer, MoveModel(), risk=0.0, elo=2400).choose(board)
    assert full.elo is None and best - lines[full.move.uci()] <= 15
    rng = random.Random(3)
    picks = {Searcher(analyzer, MoveModel(), risk=0.0, elo=900, rng=rng).choose(board).move.uci()
             for _ in range(12)}
    assert len(picks) > 1  # a weak target varies its play


# -- ratings ------------------------------------------------------------------

def test_elo_maths_and_tiers():
    assert ratings.expected(1500, 1500) == pytest.approx(0.5)
    assert ratings.expected(1700, 1300) == pytest.approx(0.909, abs=0.001)
    assert ratings.k_factor(0) == 40 and ratings.k_factor(10) == 20
    assert [ratings.tier(r) for r in (0.0, 0.19, 0.2, 0.59, 0.6, 0.84, 0.85, 1.0)] == [
        "sound", "sound", "steering", "steering", "trapper", "trapper", "hustler", "hustler"]


def test_seed_from_imports_and_match_me(store):
    assert ratings.player_rating(store, "me") == (1200.0, 0)
    import_pgn_text(store, RATED_PGN, "me", source="lichess")
    assert ratings.player_rating(store, "me") == (1345.0, 0)
    assert ratings.target_elo(store, "me") == 1395.0
    assert ratings.target_elo(store, "me", fixed=1100) == 1100.0
    assert ratings.nemesis_rating(store, "me", "trapper") == (1395.0, 0)


def test_rated_game_updates_both_once(store):
    import_pgn_text(store, RATED_PGN, "me", source="lichess")
    moves = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]  # the player mates Nemesis
    gid = record_game(store, "me", chess.WHITE, moves, risk=0.7)
    me, games = ratings.player_rating(store, "me")
    gain = 40 * (1 - ratings.expected(1345, 1395))
    assert games == 1 and me == pytest.approx(1345 + gain)
    assert ratings.nemesis_rating(store, "me", "trapper")[0] == pytest.approx(1395 - gain)
    assert ratings.update_after_game(store, gid, "me", 0.7, 1.0) is None  # idempotent
    assert ratings.player_rating(store, "me")[1] == 1
    # an unfinished game is stored but not rated
    record_game(store, "me", chess.WHITE, ["e2e4", "e7e5"], result="*", risk=0.7)
    assert ratings.player_rating(store, "me")[1] == 1


# -- review -------------------------------------------------------------------

def test_labels_and_accuracy():
    assert label_for(0, 0, True, False, False, False) == "best"
    assert label_for(0, 0, True, True, False, False) == "brilliant"
    assert label_for(0, 0, True, True, True, False) == "best"  # no brilliancy once it's decided
    assert label_for(30, 1.5, False, False, False, False) == "excellent"
    assert label_for(80, 8, False, False, False, False) == "inaccuracy"
    assert label_for(200, 15, False, False, False, True) == "miss"
    assert label_for(600, 40, False, False, False, False) == "blunder"
    assert move_accuracy(0) == pytest.approx(100, abs=0.01) and move_accuracy(100) == 0
    assert game_accuracy([100, 100]) == 100 and game_accuracy([]) is None


def test_review_finds_the_blunder_and_the_trap(store, analyzer):
    # The player (White) hangs the queen: 1.e4 d5 2.Qg4?? Bxg4.
    moves = ["e2e4", "d7d5", "d1g4", "c8g4"]
    annotations = [{"ply": 1, "move": "d7d5", "risk": 0.7, "elo": 1400, "lapse": False,
                    "trap": {"uci": "d1g4", "prob": 0.4, "loss": 900, "reasons": ["bring the queen out early"]},
                    "probs": {"d1g4": 0.4, "e4d5": 0.35}, "top": "d1g4"},
                   {"ply": 3, "move": "c8g4", "risk": 0.7, "elo": 1400, "lapse": False,
                    "trap": None, "probs": {}, "top": None}]
    gid = record_game(store, "me", chess.WHITE, moves, result="0-1", risk=0.7, annotations=annotations)
    rev = review_game(store, analyzer, gid)
    labels = [m["label"] for m in rev["moves"]]
    assert labels[2] == "blunder" and rev["moves"][2]["side"] == "you"
    assert rev["key_moments"][0]["ply"] == 2 and "queen to g4" in rev["key_moments"][0]["text"]
    assert len(rev["graph"]) == len(moves)
    assert 0 <= rev["accuracy"]["you"] < rev["accuracy"]["opponent"] <= 100
    nem = rev["nemesis"]
    assert nem["traps_set"] == 1 and nem["traps_sprung"] == 1
    assert "bring the queen out early" in nem["traps"][0]["text"]
    assert nem["predicted_top"] == 1 and nem["predictions"] == 1
    assert rev["ratings"]["you"]["change"] < 0 < rev["ratings"]["nemesis"]["change"]
    assert store.get_json("reviews", gid) is not None  # cached
