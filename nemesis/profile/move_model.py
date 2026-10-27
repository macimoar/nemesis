"""Per-player move prediction: P(move | position, player).

A conditional-logit model: every legal move gets the score w·x (x = its feature
vector), and the probabilities are a softmax over the legal moves. Personal weights
are pulled toward a population prior, so a player with few games still gets sensible
predictions and the model specialises as games come in.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field
from typing import Callable

import chess
import numpy as np

from nemesis import maia
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

# With Maia-2: start from "plays like a human at your rating". Its predictions already
# account for move quality, so the generic weights are halved rather than counted twice;
# its opening predictions (positions it never trained on) get half its weight. Chosen on
# real games (scripts/evaluate_maia.py): the best fit on the training games, confirmed on
# held-out ones.
MAIA_PRIOR = {**{k: v / 2 for k, v in DEFAULT_PRIOR.items()}, "maia": 1.0, "maia_early": 0.5}

POPULATION = "__population__"


def prior_vector(prior: dict[str, float] | None = None) -> np.ndarray:
    prior = DEFAULT_PRIOR if prior is None else prior
    return np.array([prior.get(name, 0.0) for name in FEATURES])


def uses_maia(samples: list["Sample"]) -> bool:
    return any(s.X[:, [INDEX["maia"], INDEX["maia_early"]]].any() for s in samples[:50])


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
    key: str = ""            # position key, to refresh the opening-book features


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

SAMPLES_VERSION = 2  # bump when features change, so cached samples are recomputed
MAIA_VERSION = 3     # samples built with Maia-2's predictions are cached under their own key
_BOOK_FREQ, _BOOK_DEV = INDEX["book_freq"], INDEX["book_deviation"]


def pack_samples(samples: list[Sample]) -> tuple[bytes, str]:
    """A game's samples as (npz bytes, JSON) for the samples cache."""
    buf = io.BytesIO()
    if samples:
        np.savez_compressed(
            buf, X=np.concatenate([s.X for s in samples]).astype(np.float32),
            sizes=np.array([len(s.X) for s in samples]), chosen=np.array([s.chosen for s in samples]),
            ply=np.array([s.ply for s in samples]), loss=np.array([s.loss for s in samples]),
            tp=np.array([s.time_pressure for s in samples]))
    meta = [{"phase": s.phase, "key": s.key, "moves": s.moves} for s in samples]
    return buf.getvalue(), json.dumps(meta)


def unpack_samples(data: bytes, meta: str, game_id: str) -> list[Sample]:
    info = json.loads(meta)
    if not info:
        return []
    arr = np.load(io.BytesIO(data))
    bounds = np.cumsum(arr["sizes"])[:-1]
    out = []
    for i, X in enumerate(np.split(arr["X"].astype(np.float64), bounds)):
        out.append(Sample(X=X, chosen=int(arr["chosen"][i]), game_id=game_id, ply=int(arr["ply"][i]),
                          phase=info[i]["phase"], loss=int(arr["loss"][i]), time_pressure=bool(arr["tp"][i]),
                          moves=info[i]["moves"], key=info[i]["key"]))
    return out


def refresh_book(sample: Sample, book) -> None:
    """Recompute the two opening-book features from the current repertoire."""
    total = book.total(sample.key)
    for i, uci in enumerate(sample.moves):
        freq = book.freq(sample.key, uci) if total else 0
        sample.X[i, _BOOK_FREQ] = math.log1p(freq) if total else 0.0
        sample.X[i, _BOOK_DEV] = 1.0 if total >= 3 and freq == 0 else 0.0


def build_samples(
    store: Store,
    player: str,
    analyzer: Analyzer,
    tree: OpeningTree | None = None,
    multipv: int = 6,
    max_games: int | None = None,
    halflife_games: float = 150.0,
    progress: Callable[[int, int], None] | None = None,
    maia_for: Callable[[str], object] | None = None,
) -> list[Sample]:
    """Every position the player faced, with features for each legal move. `maia_for(kind)`
    returns the Maia-2 model for "rapid" or "blitz" (or None); with it, each move also gets
    how likely a human at the player's rating is to play it."""
    games = store.games(player)
    if max_games:
        games = games[-max_games:]
    tree = tree or OpeningTree.from_store(store, player)
    depth = getattr(analyzer, "depth", 0)
    version = SAMPLES_VERSION * 100 + MAIA_VERSION if maia_for else SAMPLES_VERSION
    offset = maia.rating_offset(store, player) if maia_for else 0
    seed_elo = None
    samples: list[Sample] = []
    n = len(games)
    for gi, game in enumerate(games):
        recency = 0.5 ** ((n - 1 - gi) / halflife_games)
        if game.source == "nemesis":
            recency *= 1.5
        moves = store.moves(game.id)
        hinted = store.hinted_plies(game)   # a hint's move was Stockfish's idea, not yours
        own = {(position_key(chess.Board(m.fen)), m.uci) for m in moves[:24] if m.is_player and m.ply not in hinted}
        book = LeaveOneOut(tree, own)
        cached = store.get_samples(game.id, version, depth)
        if cached is not None:
            # A game's features never change, except the opening-book ones (your repertoire
            # grows with every game), which are cheap to refresh.
            for smp in unpack_samples(*cached, game_id=game.id):
                if smp.ply in hinted:
                    continue
                smp.weight = recency
                refresh_book(smp, book)
                samples.append(smp)
            if progress:
                progress(gi + 1, n)
            continue
        game_samples: list[Sample] = []
        last_clock: float | None = None
        policies: dict[int, dict] = {}
        model = maia_for(maia.game_kind(game)) if maia_for else None
        if model is not None:
            mine, theirs = maia.game_elos(game)
            if mine is None:   # games against Nemesis: the player's rating then
                if seed_elo is None:
                    from nemesis import ratings
                    seed_elo = ratings.player_rating(store, player)[0]
                mine = seed_elo
            theirs = theirs if theirs is not None else mine
            mover = [m for m in moves if m.is_player]
            for m, pol in zip(mover, model.policies([chess.Board(m.fen) for m in mover],
                                                    mine + offset, theirs + offset)):
                policies[m.ply] = pol
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
            ctx = Context(clock=clock_before, best_cp=best, maia=policies.get(m.ply))
            legal, X = position_features(board, losses, book, ctx)
            ucis = [mv.uci() for mv in legal]
            if m.uci not in ucis:
                continue
            game_samples.append(Sample(
                X=X, chosen=ucis.index(m.uci), weight=recency, game_id=game.id, ply=m.ply,
                phase=_phase(board), loss=losses[m.uci],
                time_pressure=clock_before is not None and clock_before < 30, moves=ucis,
                key=position_key(board),
            ))
        store.put_samples(game.id, version, depth, *pack_samples(game_samples))
        samples.extend(s for s in game_samples if s.ply not in hinted)
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
    if uses_maia(samples):
        prior = prior_vector(MAIA_PRIOR)
    else:
        prior = MoveModel.from_dict(pop[0]).w if pop else prior_vector()

    train, test = split_by_game(samples)
    metrics: dict = {"samples": len(samples), "games": len({s.game_id for s in samples}),
                     "maia": uses_maia(samples)}
    if test and train:
        held = MoveModel().fit(train, prior=prior, l2=l2)
        metrics["holdout"] = evaluate(held, test)
        metrics["holdout_prior"] = evaluate(MoveModel(prior), test)
        store.add_model_history(player, metrics["games"], metrics["holdout"]["top1"],
                                metrics["holdout"]["top3"], metrics["holdout_prior"]["top1"])
    model = MoveModel().fit(samples, prior=prior, l2=l2)
    from nemesis.profile import timecontrols
    metrics["time_controls"] = timecontrols.train_groups(store, player, samples, model, prior, l2)
    store.save_model(player, model.to_dict(), metrics)
    return model, metrics


def train_population(store: Store, samples: list[Sample]) -> MoveModel:
    """Pooled model across all players: the starting point for each personal model."""
    model = MoveModel().fit(samples, prior=prior_vector(), l2=1.0)
    store.save_model(POPULATION, model.to_dict(), {"samples": len(samples)})
    return model


def load_player_model(store: Store, player: str, with_maia: bool = False, tc: str | None = None) -> MoveModel:
    """The player's trained model: their profile for this time control (`tc`, a category)
    if it earned its place, else their main one. Someone not studied yet gets the starting
    model: Maia-2's "a human at your rating" when it's available, the generic club player
    otherwise."""
    if tc:
        from nemesis.profile import timecontrols
        own = timecontrols.load(store, player, tc)
        if own is not None:
            return own
    found = store.load_model(player.lower())
    if found:
        return MoveModel.from_dict(found[0])
    if with_maia:
        return MoveModel(prior_vector(MAIA_PRIOR))
    found = store.load_model(POPULATION)
    return MoveModel.from_dict(found[0]) if found else MoveModel()
