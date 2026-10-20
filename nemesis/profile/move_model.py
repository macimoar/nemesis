"""Per-player move prediction: P(move | position, player).

A conditional-logit model: every legal move gets the score w·x (x = its feature
vector), and the probabilities are a softmax over the legal moves. Personal weights
are pulled toward a population prior, so a player with few games still gets sensible
predictions and the model specialises as games come in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import chess
import numpy as np

from nemesis.analysis.stockfish import Analyzer, impute_losses
from nemesis.features import FEATURES, INDEX, N_FEATURES, Context, _phase, position_features, position_key
from nemesis.profile.repertoire import OpeningTree
from nemesis.store import Store

# A generic club player: mostly plays good moves, follows the book, likes forcing moves a bit.
DEFAULT_PRIOR = {
    "cp_loss": -1.3, "is_best": 0.9, "top3": 0.7, "mistake": -0.5, "blunder": -0.6,
    "capture": 0.4, "check": 0.3, "book_freq": 1.2, "book_deviation": -0.3,
    "time_pressure_x_loss": 0.5, "endgame_x_loss": 0.2, "hangs_piece": -0.3,
}

POPULATION = "__population__"


def prior_vector(prior: dict[str, float] | None = None) -> np.ndarray:
    prior = DEFAULT_PRIOR if prior is None else prior
    return np.array([prior.get(name, 0.0) for name in FEATURES])


@dataclass
class Sample:
    X: np.ndarray            # [n_moves, F]
    chosen: int              # index of the move actually played
    weight: float = 1.0
    game_id: str = ""
    ply: int = 0
    phase: str = ""
    loss: int = 0            # centipawn loss of the move played
    time_pressure: bool = False
    moves: list[str] = field(default_factory=list)


class LeaveOneOut:
    """Opening-tree view with one game's own moves removed, so book features don't leak the answer."""

    def __init__(self, tree: OpeningTree, own: set[tuple[str, str]]):
        self.tree, self.own = tree, own
        self.own_keys = {k for k, _ in own}

    def freq(self, key: str, uci: str) -> int:
        return self.tree.freq(key, uci) - ((key, uci) in self.own)

    def total(self, key: str) -> int:
        return self.tree.total(key) - (key in self.own_keys)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_samples(
    store: Store,
    player: str,
    analyzer: Analyzer,
    tree: OpeningTree | None = None,
    multipv: int = 6,
    max_games: int | None = None,
    halflife_games: float = 150.0,
    progress: Callable[[int, int], None] | None = None,
) -> list[Sample]:
    games = store.games(player)
    if max_games:
        games = games[-max_games:]
    tree = tree or OpeningTree.from_store(store, player)
    samples: list[Sample] = []
    n = len(games)
    for gi, game in enumerate(games):
        recency = 0.5 ** ((n - 1 - gi) / halflife_games)
        if game.source == "nemesis":
            recency *= 1.5
        moves = store.moves(game.id)
        own = {(position_key(chess.Board(m.fen)), m.uci) for m in moves[:24] if m.is_player}
        book = LeaveOneOut(tree, own)
        last_clock: float | None = None
        for m in moves:
            if not m.is_player:
                continue
            board = chess.Board(m.fen)
            clock_before, last_clock = last_clock, m.clock
            if board.legal_moves.count() < 2:
                continue
            lines = analyzer.lines(board, multipv=multipv, extra=[m.uci])
            losses = impute_losses(board, lines)
            best = max(lines.values()) if lines else 0
            ctx = Context(clock=clock_before, best_cp=best)
            legal, X = position_features(board, losses, book, ctx)
            ucis = [mv.uci() for mv in legal]
            if m.uci not in ucis:
                continue
            samples.append(Sample(
                X=X, chosen=ucis.index(m.uci), weight=recency, game_id=game.id, ply=m.ply,
                phase=_phase(board), loss=losses[m.uci],
                time_pressure=clock_before is not None and clock_before < 30, moves=ucis,
            ))
        if progress:
            progress(gi + 1, n)
    return samples


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _softmax(scores: np.ndarray) -> np.ndarray:
    z = scores - scores.max()
    e = np.exp(z)
    return e / e.sum()


class MoveModel:
    def __init__(self, weights: np.ndarray | None = None):
        self.w = prior_vector() if weights is None else np.asarray(weights, dtype=np.float64)

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> dict[str, float]:
        return {name: float(v) for name, v in zip(FEATURES, self.w)}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "MoveModel":
        return cls(np.array([d.get(name, 0.0) for name in FEATURES]))

    # -- inference ----------------------------------------------------------
    def probs(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        return _softmax(X @ self.w)

    def contributions(self, x: np.ndarray) -> dict[str, float]:
        return {name: float(c) for name, c in zip(FEATURES, self.w * x) if c}

    # -- training -----------------------------------------------------------
    def fit(self, samples: list[Sample], prior: np.ndarray | None = None, l2: float = 4.0,
            iters: int = 30, tol: float = 1e-6) -> "MoveModel":
        """Damped Newton's method on weighted NLL + l2·||w - prior||². The objective is convex."""
        prior = prior_vector() if prior is None else prior
        w = prior.copy()
        if not samples:
            self.w = w
            return self
        Xs = [s.X for s in samples]
        chosen = [s.chosen for s in samples]
        wts = np.array([s.weight for s in samples])

        def objective(w: np.ndarray) -> float:
            total = 0.0
            for X, c, wt in zip(Xs, chosen, wts):
                sc = X @ w
                m = sc.max()
                total += wt * (m + np.log(np.exp(sc - m).sum()) - sc[c])
            return total + l2 * float(((w - prior) ** 2).sum())

        f_old = objective(w)
        for _ in range(iters):
            g = 2 * l2 * (w - prior)
            H = 2 * l2 * np.eye(N_FEATURES)
            for X, c, wt in zip(Xs, chosen, wts):
                p = _softmax(X @ w)
                mean = p @ X
                g += wt * (mean - X[c])
                Xc = X - mean
                H += wt * (Xc.T * p) @ Xc
            step = np.linalg.solve(H, g)
            t = 1.0
            while t > 1e-4:
                w_new = w - t * step
                f_new = objective(w_new)
                if f_new <= f_old:
                    break
                t *= 0.5
            if f_old - f_new < tol * max(1.0, abs(f_old)):
                w, f_old = w_new, f_new
                break
            w, f_old = w_new, f_new
        self.w = w
        return self


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: MoveModel, samples: list[Sample]) -> dict[str, float]:
    if not samples:
        return {"n": 0}
    top1 = top3 = sf_top1 = 0
    nll = 0.0
    uni_nll = 0.0
    for s in samples:
        p = model.probs(s.X)
        order = np.argsort(-p)
        top1 += order[0] == s.chosen
        top3 += s.chosen in order[:3]
        nll -= np.log(max(p[s.chosen], 1e-12))
        uni_nll += np.log(len(p))
        sf_top1 += s.X[s.chosen, INDEX["is_best"]] > 0 and s.X[:, INDEX["is_best"]].sum() == 1
    n = len(samples)
    return {
        "n": n,
        "top1": top1 / n,
        "top3": top3 / n,
        "nll": nll / n,
        "stockfish_top1": sf_top1 / n,
        "uniform_nll": uni_nll / n,
    }


def split_by_game(samples: list[Sample], test_frac: float = 0.2) -> tuple[list[Sample], list[Sample]]:
    """Chronological split: the most recent games are held out (samples come in game order)."""
    game_ids = list(dict.fromkeys(s.game_id for s in samples))
    n_test = max(1, int(len(game_ids) * test_frac)) if len(game_ids) > 1 else 0
    test_ids = set(game_ids[len(game_ids) - n_test:])
    return [s for s in samples if s.game_id not in test_ids], [s for s in samples if s.game_id in test_ids]


def train_player(store: Store, player: str, samples: list[Sample], l2: float = 4.0) -> tuple[MoveModel, dict]:
    """Fit on older games, report accuracy on the newest, then refit on everything and save."""
    player = player.lower()
    pop = store.load_model(POPULATION)
    prior = MoveModel.from_dict(pop[0]).w if pop else prior_vector()

    train, test = split_by_game(samples)
    metrics: dict = {"samples": len(samples), "games": len({s.game_id for s in samples})}
    if test and train:
        held = MoveModel().fit(train, prior=prior, l2=l2)
        metrics["holdout"] = evaluate(held, test)
        metrics["holdout_prior"] = evaluate(MoveModel(prior), test)
    model = MoveModel().fit(samples, prior=prior, l2=l2)
    store.save_model(player, model.to_dict(), metrics)
    return model, metrics


def train_population(store: Store, samples: list[Sample]) -> MoveModel:
    """Pooled model across all players: the starting point for each personal model."""
    model = MoveModel().fit(samples, prior=prior_vector(), l2=1.0)
    store.save_model(POPULATION, model.to_dict(), {"samples": len(samples)})
    return model


def load_player_model(store: Store, player: str) -> MoveModel:
    found = store.load_model(player.lower()) or store.load_model(POPULATION)
    return MoveModel.from_dict(found[0]) if found else MoveModel()
