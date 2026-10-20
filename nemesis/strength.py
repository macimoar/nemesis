"""Playing-strength limit: make Nemesis a challenge, not a wall.

Each target Elo maps to a target average centipawn loss (ACPL), the standard
measure of how much a player gives away per move. On every move, Nemesis scores
its candidates with the exploit search, then samples one:

    p(move) ∝ exp(SHARPNESS · (score − best_score) − loss / T)

where `loss` is the move's centipawn cost against perfect play. The temperature T
is solved per position so that the expected loss matches the target. That keeps the
error rate steady whether the position offers many reasonable moves or few. Rarely,
it also has a human-style lapse: a move costing 150-500cp. At MAX_ELO the limit is off.

TABLE is fitted by scripts/calibrate_strength.py against Stockfish's UCI_Elo.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

MAX_ELO = 2400
MIN_ELO = 800
SHARPNESS = 8.0     # preference for moves that work against you, next to their centipawn cost
LOSS_CAP = 500      # losses beyond this count as this much when matching the target
WINDOW = 500        # how far below the best move a strength-limited Nemesis looks

# (elo, target average centipawn loss per move, lapse probability per move)
# 1400-2000 fitted against Stockfish UCI_Elo (scripts/calibrate_strength.py): the
# loss at which Nemesis scores ~50%. Stockfish can't go below 1320, so the 800 and
# 1100 rows are extrapolated toward typical beginner play.
TABLE = [
    (800, 85.0, 0.030),
    (1100, 60.0, 0.020),
    (1400, 42.0, 0.012),
    (1700, 37.0, 0.009),
    (2000, 31.0, 0.006),
    (2200, 20.0, 0.003),
    (MAX_ELO, 0.0, 0.0),
]


def params(elo: float) -> tuple[float, float]:
    """(target ACPL, lapse rate) for a target Elo, linearly interpolated."""
    elo = max(MIN_ELO, min(MAX_ELO, elo))
    for (e0, a0, l0), (e1, a1, l1) in zip(TABLE, TABLE[1:]):
        if elo <= e1:
            f = (elo - e0) / (e1 - e0)
            return a0 + f * (a1 - a0), l0 + f * (l1 - l0)
    return 0.0, 0.0


def limited(elo: float | None) -> bool:
    return elo is not None and elo < MAX_ELO


def _weights(losses: Sequence[float], scores: Sequence[float], temp: float) -> list[float]:
    top = max(scores)
    logits = [SHARPNESS * (s - top) - loss / temp for s, loss in zip(scores, losses)]
    m = max(logits)
    return [math.exp(x - m) for x in logits]


def _expected_loss(losses: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    return sum(min(loss, LOSS_CAP) * w for loss, w in zip(losses, weights)) / total


def temperature(losses: Sequence[float], scores: Sequence[float], target: float) -> float:
    """The temperature at which the expected loss over these candidates equals `target`."""
    lo, hi = 1.0, 5000.0
    if _expected_loss(losses, _weights(losses, scores, hi)) <= target:
        return hi  # even near-uniform play loses less than the target here
    for _ in range(40):  # expected loss rises with temperature: bisect (in log space)
        mid = math.sqrt(lo * hi)
        if _expected_loss(losses, _weights(losses, scores, mid)) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def pick(losses: Sequence[float], scores: Sequence[float], elo: float,
         rng: random.Random) -> tuple[int, bool]:
    """Choose a candidate index. Returns (index, was_a_lapse)."""
    target, lapse = params(elo)
    best = max(range(len(scores)), key=lambda i: (scores[i], -losses[i]))
    if target <= 0 or len(losses) == 1:
        return best, False
    lapses = [i for i, loss in enumerate(losses) if 150 <= loss <= LOSS_CAP]
    if lapses and rng.random() < lapse:
        return rng.choice(lapses), True
    weights = _weights(losses, scores, temperature(losses, scores, target))
    r = rng.random() * sum(weights)
    for i, w in enumerate(weights):
        r -= w
        if r <= 0:
            return i, False
    return len(weights) - 1, False
