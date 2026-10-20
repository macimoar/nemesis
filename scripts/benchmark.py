"""Does exploitation pay? Play Nemesis and plain Stockfish against the same simulated player.

    python scripts/benchmark.py --db test.db --player synthetic_grabber --style grabber --games 10

Both sides search at the same depth. "Plain" is Stockfish's own best move. Reported:
score, and how many moves it took to win.
"""

from __future__ import annotations

import argparse
import random
import statistics

import chess
import chess.engine

from nemesis.analysis.stockfish import StockfishAnalyzer, find_stockfish
from nemesis.engine import Nemesis
from nemesis.store import Store
from simulate import pick


def play(chooser, style: str, sf: chess.engine.SimpleEngine, seed: int, engine_white: bool, max_ply: int):
    rng = random.Random(seed)
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and board.ply() < max_ply:
        if board.turn == engine_white:
            move = chooser(board)
        else:
            infos = sf.analyse(board, chess.engine.Limit(depth=6), multipv=3)
            move = pick(style, board, [i["pv"][0] for i in infos if "pv" in i], rng)
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5, board.ply()
    return (1.0 if outcome.winner == engine_white else 0.0), board.ply()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--player", required=True)
    ap.add_argument("--style", default="grabber")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--risk", type=float, default=1.0)
    ap.add_argument("--max-ply", type=int, default=200)
    args = ap.parse_args()

    store = Store(args.db)
    analyzer = StockfishAnalyzer(store, depth=args.depth)
    nem = Nemesis(store, args.player, risk=args.risk, depth=args.depth, analyzer=analyzer)
    plain = analyzer.engine
    contenders = {
        "stockfish": lambda b: plain.play(b, chess.engine.Limit(depth=args.depth)).move,
        "nemesis": lambda b: nem.choose(b).move,
    }
    with chess.engine.SimpleEngine.popen_uci(find_stockfish()) as sim:
        sim.configure({"Skill Level": 3})
        for name, chooser in contenders.items():
            results = [play(chooser, args.style, sim, seed=g, engine_white=g % 2 == 0, max_ply=args.max_ply)
                       for g in range(args.games)]
            print(f"{name:<10} game lengths (plies): {[p for _, p in results]}")
            score = sum(r for r, _ in results) / len(results)
            win_len = [ply for r, ply in results if r == 1.0]
            avg = f"{statistics.mean(win_len) / 2:.1f}" if win_len else "-"
            print(f"{name:<10} score {score:.0%}  wins {len(win_len)}/{args.games}  avg moves to win {avg}")
    analyzer.close()


if __name__ == "__main__":
    main()
