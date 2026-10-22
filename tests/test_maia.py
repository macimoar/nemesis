import chess
import numpy as np
import pytest

from nemesis import maia
from nemesis.features import INDEX, Context, position_features
from nemesis.learn import record_game
from nemesis.profile import move_model
from nemesis.profile.move_model import MAIA_PRIOR, build_samples, load_player_model, train_player


class FakeMaia:
    """Stands in for Maia-2: likes one particular move, everything else is unlikely."""

    def __init__(self, favourite="e2e4"):
        self.favourite, self.calls = favourite, []

    def policies(self, boards, elo_self, elo_oppo):
        self.calls.append((len(boards), elo_self, elo_oppo))
        return [self.policy(b, elo_self, elo_oppo) for b in boards]

    def policy(self, board, elo_self, elo_oppo):
        legal = [m.uci() for m in board.legal_moves]
        rest = 0.2 / max(1, len(legal) - 1)
        return {u: (0.8 if u == self.favourite else rest) for u in legal}


def test_buckets_match_maia2():
    assert [maia.bucket(e) for e in (800, 1099, 1100, 1199, 1250, 1999, 2000, 2700)] == [0, 0, 1, 1, 2, 9, 10, 10]
    assert maia.bucket(None) == maia.bucket(1500)
    assert maia.kind_for("blitz") == maia.kind_for("bullet") == "blitz"
    assert maia.kind_for("rapid") == maia.kind_for("untimed") == maia.kind_for(None) == "rapid"


def test_encoding_is_from_the_side_to_move():
    x, seen = maia.encode(chess.Board())
    assert x.shape == (18, 8, 8) and x[0, 1].sum() == 8 and x[6, 6].sum() == 8   # pawns
    assert x[12].all() and x[13:17].all() and not x[17].any()                    # turn, castling, no e.p.
    # After 1.e4 it's Black to move: Maia-2 sees the board mirrored, so it's "White" to move again.
    board = chess.Board()
    board.push_uci("e2e4")
    x2, seen = maia.encode(board)
    assert seen.turn == chess.WHITE and x2[12].all() and x2[6, 4, 4] == 1   # the e4 pawn, now an enemy on e5
    assert not x2[17].any()        # no en-passant square: no capture is possible (as in a FEN)
    board = chess.Board("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
    board.push_uci("c2c4")         # now ...dxc3 e.p. is legal
    assert maia.encode(board)[0][17].sum() == 1
    assert maia._mirror("e7e8q") == "e2e1q"


def test_feature_is_the_log_probability():
    middlegame = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 6")
    losses = {m.uci(): 0 for m in middlegame.legal_moves}
    fake = FakeMaia(favourite="f3g5")
    moves, X = position_features(middlegame, losses, ctx=Context(maia=fake.policy(middlegame, 1500, 1500)))
    ucis = [m.uci() for m in moves]
    n = len(ucis)
    assert X[ucis.index("f3g5"), INDEX["maia"]] == pytest.approx(np.log(0.8))
    assert X[ucis.index("a2a3"), INDEX["maia"]] == pytest.approx(np.log(0.2 / (n - 1)))
    assert not X[:, INDEX["maia_early"]].any()
    # In the first 10 plies, which Maia-2 never trained on, it's a separate feature.
    start = chess.Board()
    moves, X = position_features(start, {m.uci(): 0 for m in start.legal_moves},
                                 ctx=Context(maia=FakeMaia().policy(start, 1500, 1500)))
    assert X[[m.uci() for m in moves].index("e2e4"), INDEX["maia_early"]] == pytest.approx(np.log(0.8))
    assert not X[:, INDEX["maia"]].any()
    _, X0 = position_features(middlegame, losses)
    assert not X0[:, [INDEX["maia"], INDEX["maia_early"]]].any()   # without Maia-2 the features are off


def test_training_with_maia(store, analyzer):
    record_game(store, "me", chess.WHITE, ["e2e4", "e7e5", "g1f3", "b8c6"], result="*")
    fake = FakeMaia()
    samples = build_samples(store, "me", analyzer, maia_for=lambda kind: fake)
    assert fake.calls and all(n == 2 for n, *_ in fake.calls)      # one batch per game
    first = samples[0]
    # Opening positions (Maia-2 never trained on the first 10 plies) use their own feature.
    assert first.X[first.moves.index("e2e4"), INDEX["maia_early"]] == pytest.approx(np.log(0.8))
    assert not first.X[:, INDEX["maia"]].any()
    # Cached separately from samples built without Maia-2.
    plain = build_samples(store, "me", analyzer)
    assert not plain[0].X[:, INDEX["maia"]].any()
    model, _ = train_player(store, "me", samples)
    assert model.to_dict()["maia"] > 0.3                              # starts from "trust Maia-2"


def test_new_players_start_from_maia(store):
    assert load_player_model(store, "nobody", with_maia=True).to_dict()["maia"] == MAIA_PRIOR["maia"]
    assert load_player_model(store, "nobody").to_dict()["maia"] == 0


def test_rating_offset_follows_where_games_come_from(store):
    record_game(store, "me", chess.WHITE, ["e2e4"], result="*")          # Nemesis games don't count
    assert maia.rating_offset(store, "me") == 0
    from nemesis.ingest.pgn import import_pgn_text
    pgn = """[Site "Chess.com"]
[White "me"]
[Black "x"]
[WhiteElo "900"]
[BlackElo "950"]
[Result "1-0"]

1. e4 e5 1-0
"""
    import_pgn_text(store, pgn, "me")
    assert maia.rating_offset(store, "me") == maia.CHESSCOM_TO_LICHESS
    imported = [g for g in store.games("me") if g.source == "pgn"][0]
    assert maia.game_elos(imported) == (900.0, 950.0)


@pytest.mark.skipif(maia.installed("rapid") is None, reason="Maia-2 models not downloaded")
def test_real_model_predicts_sensible_moves():
    m = maia.installed("rapid")
    # Black's pawn just took on d5: a human recaptures.
    probs = m.policy(chess.Board("rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 13"), 1500, 1500)
    assert abs(sum(probs.values()) - 1) < 1e-4 and max(probs, key=probs.get) == "e4d5"
    # The queen came out to h4: a human chases it.
    probs = m.policy(chess.Board("rnb1kbnr/pppp1ppp/8/4p3/4P2q/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 13"), 1500, 1500)
    assert max(probs, key=probs.get) == "g1f3"
