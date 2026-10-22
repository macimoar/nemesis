"""Generate games for a synthetic player with a known style, to check that Nemesis learns it.

    python scripts/simulate.py grabber --games 40 --out grabber.pgn

Styles:
  grabber  - takes material whenever it can, even when it's poisoned
  coward   - prefers retreats and avoids trades
  solid    - plays one of Stockfish's top 2 moves
  pinblind - plays well, but often walks into pins and skewers
"""

from __future__ import annotations

import argparse
import random

import chess
import chess.engine
import chess.pgn

from nemesis.analysis.stockfish import find_stockfish
from nemesis.features import VALUE, opponent_tactics


def pick(style: str, board: chess.Board, top: list[chess.Move], rng: random.Random) -> chess.Move:
    legal = list(board.legal_moves)
    if style == "grabber":
        caps = [m for m in legal if board.is_capture(m)]
        if caps and rng.random() < 0.8:
            return max(caps, key=lambda m: VALUE[board.piece_type_at(m.to_square) or chess.PAWN])
    elif style == "coward":
        forward = 1 if board.turn == chess.WHITE else -1
        backs = [m for m in legal if (chess.square_rank(m.to_square) - chess.square_rank(m.from_square)) * forward < 0
                 and board.piece_type_at(m.from_square) != chess.KING]
        if backs and rng.random() < 0.5:
            return rng.choice(backs)
    elif style == "pinblind" and rng.random() < 0.5:
        blind = []
        for m in legal:
            board.push(m)
            if opponent_tactics(board, not board.turn) & {"pin", "skewer"}:
                blind.append(m)
            board.pop()
        if blind:
            return rng.choice(blind)
    return rng.choice(top[:2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("style", choices=["grabber", "coward", "solid", "pinblind"])
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    name = f"synthetic_{args.style}"
    with chess.engine.SimpleEngine.popen_uci(find_stockfish()) as sf, open(args.out, "w") as out:
        sf.configure({"Skill Level": 3})
        for g in range(args.games):
            board = chess.Board()
            human = chess.WHITE if g % 2 == 0 else chess.BLACK
            game = chess.pgn.Game()
            game.headers.update({"Event": "simulated", "Site": f"sim/{args.style}/{args.seed}/{g}",
                                 "White": name if human else "sf", "Black": "sf" if human else name,
                                 "UTCDate": f"2026.01.{g % 28 + 1:02d}", "UTCTime": f"{g:02d}:00:00"})
            node = game
            while not board.is_game_over(claim_draw=True) and board.ply() < 160:
                if board.turn == human:
                    infos = sf.analyse(board, chess.engine.Limit(depth=6), multipv=3)
                    move = pick(args.style, board, [i["pv"][0] for i in infos if "pv" in i], rng)
                else:
                    move = sf.play(board, chess.engine.Limit(depth=4)).move
                node = node.add_variation(move)
                board.push(move)
            game.headers["Result"] = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
            print(game, file=out, end="\n\n")
    print(f"wrote {args.games} games for {name} to {args.out}")


if __name__ == "__main__":
    main()
