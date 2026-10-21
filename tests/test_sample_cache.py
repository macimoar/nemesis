import chess
import numpy as np
import pytest

from nemesis.features import INDEX
from nemesis.learn import record_game
from nemesis.profile import move_model
from nemesis.profile.move_model import build_samples, pack_samples, unpack_samples

ITALIAN = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5"]


def same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert (x.game_id, x.ply, x.chosen, x.moves, x.phase, x.loss, x.key) == \
               (y.game_id, y.ply, y.chosen, y.moves, y.phase, y.loss, y.key)
        assert np.allclose(x.X, y.X, atol=1e-6)


def test_pack_round_trip(store, analyzer):
    record_game(store, "me", chess.WHITE, ITALIAN, result="*")
    samples = build_samples(store, "me", analyzer)
    same(unpack_samples(*pack_samples(samples), game_id=samples[0].game_id), samples)
    assert unpack_samples(*pack_samples([]), game_id="x") == []


def test_cached_samples_match_fresh_ones(store, analyzer, monkeypatch):
    for _ in range(3):
        record_game(store, "me", chess.WHITE, ITALIAN, result="*")
    first = build_samples(store, "me", analyzer)              # computes and fills the cache
    calls = []
    real = move_model.position_features
    monkeypatch.setattr(move_model, "position_features", lambda *a, **k: calls.append(1) or real(*a, **k))
    cached = build_samples(store, "me", analyzer)             # served from the cache
    assert calls == []
    same(cached, first)


def test_book_features_follow_the_growing_repertoire(store, analyzer):
    record_game(store, "me", chess.WHITE, ITALIAN, result="*")
    record_game(store, "me", chess.WHITE, ITALIAN, result="*")
    before = build_samples(store, "me", analyzer)
    # A third identical game makes the line "known": from the cache, game 1's book features
    # must now count two other games, not one.
    record_game(store, "me", chess.WHITE, ITALIAN, result="*")
    after = build_samples(store, "me", analyzer)
    s0 = before[0]
    i = s0.moves.index("e2e4")
    assert s0.X[i, INDEX["book_freq"]] == pytest.approx(np.log1p(1))
    assert after[0].X[i, INDEX["book_freq"]] == pytest.approx(np.log1p(2))
    fresh = build_samples(store, "me", analyzer)
    same(after, fresh)


def test_feature_version_change_invalidates_the_cache(store, analyzer, monkeypatch):
    record_game(store, "me", chess.WHITE, ITALIAN, result="*")
    build_samples(store, "me", analyzer)
    monkeypatch.setattr(move_model, "SAMPLES_VERSION", move_model.SAMPLES_VERSION + 1)
    calls = []
    real = move_model.position_features
    monkeypatch.setattr(move_model, "position_features", lambda *a, **k: calls.append(1) or real(*a, **k))
    build_samples(store, "me", analyzer)
    assert calls  # recomputed under the new version
