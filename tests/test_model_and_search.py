import chess
import numpy as np

from nemesis.features import INDEX, N_FEATURES
from nemesis.profile.move_model import MoveModel, Sample, evaluate, prior_vector
from nemesis.search import Searcher, window_cp


def synthetic_samples(n=300, seed=0):
    """A player who always captures when a capture is available."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        X = np.zeros((10, N_FEATURES))
        X[:, INDEX["cp_loss"]] = rng.uniform(0, 3, 10)
        X[:, INDEX["capture"]] = rng.random(10) < 0.2
        caps = np.flatnonzero(X[:, INDEX["capture"]])
        chosen = int(caps[0]) if len(caps) else int(np.argmin(X[:, INDEX["cp_loss"]]))
        samples.append(Sample(X=X, chosen=chosen, game_id=str(i // 30)))
    return samples


def test_model_learns_capture_habit():
    samples = synthetic_samples()
    model = MoveModel().fit(samples, l2=1.0)
    assert model.w[INDEX["capture"]] > prior_vector()[INDEX["capture"]] + 1.5
    assert evaluate(model, samples)["nll"] < evaluate(MoveModel(), samples)["nll"]


def test_model_round_trip():
    m = MoveModel(np.arange(N_FEATURES, dtype=float))
    assert np.allclose(MoveModel.from_dict(m.to_dict()).w, m.w)


def test_risk_zero_stays_in_window(analyzer):
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    lines = analyzer.lines(board, multipv=8)
    best = max(lines.values())
    for risk in (0.0, 0.5, 1.0):
        d = Searcher(analyzer, MoveModel(), risk=risk).choose(board)
        assert best - lines[d.move.uci()] <= window_cp(risk, best)


def test_exploits_a_grabber(analyzer):
    # Black to move. ...Qxb2-style poison is modelled by a player who always captures:
    # Nemesis should prefer a move whose likely reply is a losing capture.
    grabber = MoveModel(prior_vector())
    grabber.w[INDEX["capture"]] = 6.0
    grabber.w[INDEX["cp_loss"]] = -0.2
    board = chess.Board("4k3/8/8/8/8/8/3q4/R3K3 b - - 0 1")
    d = Searcher(analyzer, grabber, risk=1.0).choose(board)
    assert d.best.exploit >= d.best.objective
    assert d.trap is not None or d.best.replies


def test_window_shrinks_when_winning():
    assert window_cp(1.0, 0) == 300
    assert window_cp(1.0, 600) < 60
    assert window_cp(0.0, 600) == 15
