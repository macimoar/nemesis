"""Playing-strength limit: make Nemesis a challenge, not a wall.

At a target Elo, Nemesis still scores its candidates with the exploit search, but
instead of always taking the top one it samples:

    p(move) ∝ exp(SHARPNESS · (score − best_score) − loss / T)

where `loss` is the centipawn cost against perfect play and the temperature T grows
as the target drops. Separately, with a small rating-dependent probability, it has a
human-style lapse and plays a move costing 100-400cp. At MAX_ELO the limit is off.

TABLE is fitted by scripts/calibrate_strength.py against Stockfish's UCI_Elo.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

MAX_ELO = 2400
MIN_ELO = 800
SHARPNESS = 40.0  # how much a 1% difference in expected score matters next to centipawns

# (elo, temperature in centipawns, lapse probability per move)
TABLE = [
    (800, 130.0, 0.14),
    (1100, 95.0, 0.09),
    (1400, 65.0, 0.055),
    (1700, 42.0, 0.03),
    (2000, 24.0, 0.012),
    (2200, 12.0, 0.004),
    (MAX_ELO, 0.0, 0.0),
]


def params(elo: float) -> tuple[float, float]:
    """(temperature, lapse rate) for a target Elo, linearly interpolated."""
    elo = max(MIN_ELO, min(MAX_ELO, elo))
    for (e0, t0, l0), (e1, t1, l1) in zip(TABLE, TABLE[1:]):
        if elo <= e1:
            f = (elo - e0) / (e1 - e0)
            return t0 + f * (t1 - t0), l0 + f * (l1 - l0)
    return 0.0, 0.0


def limited(elo: float | None) -> bool:
    return elo is not None and elo < MAX_ELO


def search_window(elo: float) -> float:
    """How far below the best move a strength-limited Nemesis may look."""
    temp, _ = params(elo)
    return 100 + 3 * temp


def pick(losses: Sequence[float], scores: Sequence[float], elo: float,
         rng: random.Random) -> tuple[int, bool]:
    """Choose a candidate index. Returns (index, was_a_lapse)."""
    temp, lapse = params(elo)
    best = max(range(len(scores)), key=lambda i: (scores[i], -losses[i]))
    if temp <= 0:
        return best, False
    lapses = [i for i, loss in enumerate(losses) if 100 <= loss <= 400]
    if lapses and rng.random() < lapse:
        return rng.choice(lapses), True
    top = scores[best]
    logits = [SHARPNESS * (s - top) - loss / temp for s, loss in zip(scores, losses)]
    m = max(logits)
    weights = [math.exp(x - m) for x in logits]
    r = rng.random() * sum(weights)
    for i, w in enumerate(weights):
        r -= w
        if r <= 0:
            return i, False
    return len(weights) - 1, False
