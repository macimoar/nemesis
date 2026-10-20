"""Calibrate Nemesis's strength limit against Stockfish's own UCI_Elo.

    python scripts/calibrate_strength.py --targets 1400 1700 2000 --games 16

Each target plays `--games` games (alternating colours) as Nemesis at that Elo
against Stockfish with UCI_LimitStrength at the same Elo. A well-calibrated target
scores about 50%. With --fit, the temperature and lapse rate in strength.TABLE are
scaled after each round (too strong -> hotter, too weak -> cooler) and the fitted
table is printed for pasting into nemesis/strength.py.

Stockfish's Elo scale is engine-calibrated, so it maps only roughly onto online
ratings; the results-based ratings correct for that over time.
"""

from __future__ import annotations

import argparse
import math
import random
from concurrent.futures import ProcessPoolExecutor

import chess
import chess.engine

from nemesis import strength
from nemesis.analysis.stockfish import StockfishAnalyzer, find_stockfish
from nemesis.profile.move_model import MoveModel
from nemesis.search import Searcher


def play_one(job: tuple) -> float:
    """One game; returns Nemesis's score."""
    target, seed, nemesis_white, depth, table, movetime, max_ply = job
    strength.TABLE[:] = table
    rng = random.Random(seed)
    with StockfishAnalyzer(None, depth=depth) as analyzer, \
            chess.engine.SimpleEngine.popen_uci(find_stockfish()) as sf:
        sf.configure({"UCI_LimitStrength": True, "UCI_Elo": max(1320, int(target))})
        board = chess.Board()
        # A couple of random opening moves so games differ.
        for _ in range(2):
            board.push(rng.choice(list(board.legal_moves)))
        model = MoveModel()  # the generic human prior: calibration is about strength, not exploitation
        while not board.is_game_over(claim_draw=True) and board.ply() < max_ply:
            if board.turn == nemesis_white:
                move = Searcher(analyzer, model, risk=0.3, elo=target, rng=rng).choose(board).move
            else:
                move = sf.play(board, chess.engine.Limit(time=movetime)).move
            board.push(move)
        outcome = board.outcome(claim_draw=True)
        if outcome is None:  # ply cap: adjudicate by eval
            info = sf.analyse(board, chess.engine.Limit(depth=10))
            cp = info["score"].pov(nemesis_white).score(mate_score=10000)
            return 1.0 if cp > 200 else 0.0 if cp < -200 else 0.5
        if outcome.winner is None:
            return 0.5
        return 1.0 if outcome.winner == nemesis_white else 0.0


def measure(target: float, games: int, depth: int, table: list, movetime: float, workers: int,
            max_ply: int) -> float:
    jobs = [(target, 1000 * int(target) + g, g % 2 == 0, depth, table, movetime, max_ply) for g in range(games)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        scores = list(pool.map(play_one, jobs))
    return sum(scores) / len(scores)


def elo_gap(score: float) -> float:
    """Elo difference implied by a score (positive: Nemesis stronger than the target)."""
    score = min(max(score, 0.03), 0.97)
    return -400 * math.log10(1 / score - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=float, nargs="+", default=[1400, 1700, 2000])
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--movetime", type=float, default=0.1, help="Stockfish seconds per move")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-ply", type=int, default=160)
    ap.add_argument("--fit", type=int, default=0, help="rounds of table fitting")
    args = ap.parse_args()

    table = [list(row) for row in strength.TABLE]
    for rnd in range(args.fit + 1):
        print(f"round {rnd}:")
        for target in args.targets:
            score = measure(target, args.games, args.depth, [tuple(r) for r in table], args.movetime,
                            args.workers, args.max_ply)
            gap = elo_gap(score)
            print(f"  target {target:.0f}: Nemesis scored {score:.0%} vs Stockfish UCI_Elo {max(1320, target):.0f}"
                  f"  (~{gap:+.0f} Elo)")
            if rnd < args.fit:
                # Too strong (gap > 0) -> raise temperature and lapses near this target, and vice versa.
                factor = min(2.0, max(0.5, 2 ** (gap / 300)))
                lowest = target == min(args.targets)  # rows below the lowest target follow it
                for row in table:
                    near = abs(row[0] - target) <= 200 or (lowest and row[0] < target)
                    if near and row[0] < strength.MAX_ELO:
                        row[1] *= factor
                        row[2] = min(0.3, row[2] * factor)
    print("\nTABLE = [")
    for e, t, l in table:
        print(f"    ({e:.0f}, {t:.1f}, {l:.3f}),")
    print("]")


if __name__ == "__main__":
    main()
