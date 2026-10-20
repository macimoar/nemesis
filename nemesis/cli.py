"""Command line: nemesis import | train | profile | play | uci | serve."""

from __future__ import annotations

import argparse
import json
import sys

import chess

from nemesis.store import DEFAULT_DB, Store


def _progress(done: int, total: int) -> None:
    print(f"\r  analysed {done}/{total} games", end="", file=sys.stderr, flush=True)
    if done == total:
        print(file=sys.stderr)


def cmd_import(args: argparse.Namespace) -> None:
    store = Store(args.db)
    if args.source == "lichess":
        from nemesis.ingest.lichess import import_user
        n = import_user(store, args.username, args.max)
    elif args.source == "chesscom":
        from nemesis.ingest.chesscom import import_user
        n = import_user(store, args.username, args.max)
    else:
        from nemesis.ingest.pgn import import_pgn_text
        if not args.file:
            sys.exit("--file is required for pgn imports")
        with open(args.file, encoding="utf-8", errors="replace") as fh:
            n = import_pgn_text(store, fh.read(), args.username)
    print(f"imported {n} new games for {args.username}")


def cmd_train(args: argparse.Namespace) -> None:
    from nemesis.analysis.stockfish import StockfishAnalyzer
    from nemesis.learn import train, train_pooled

    store = Store(args.db)
    with StockfishAnalyzer(store, depth=args.depth) as analyzer:
        if args.population:
            train_pooled(store, analyzer, args.max_games)
            print("population prior trained")
            return
        _, metrics = train(store, args.username, analyzer, args.max_games, progress=_progress)
    print_metrics(metrics)


def print_metrics(metrics: dict) -> None:
    print(f"trained on {metrics['samples']} moves from {metrics['games']} games")
    if "holdout" in metrics:
        h, p = metrics["holdout"], metrics["holdout_prior"]
        print(f"held-out newest games ({h['n']} moves):")
        print(f"  personal model   top-1 {h['top1']:.1%}  top-3 {h['top3']:.1%}  nll {h['nll']:.3f}")
        print(f"  generic prior    top-1 {p['top1']:.1%}  top-3 {p['top3']:.1%}  nll {p['nll']:.3f}")
        print(f"  stockfish best   top-1 {h['stockfish_top1']:.1%}")
        print(f"  uniform          nll {h['uniform_nll']:.3f}")


def cmd_profile(args: argparse.Namespace) -> None:
    found = Store(args.db).load_model(args.username.lower())
    if not found or "profile" not in found[1]:
        sys.exit(f"no profile for {args.username}: run `nemesis train {args.username}` first")
    prof = found[1]["profile"]
    if args.json:
        print(json.dumps(prof, indent=2))
        return
    print(f"{prof['player']}: {prof['games']} games, score {prof['score']}")
    print("\nWhat makes you you (vs a generic player):")
    for t in prof["traits"][:6]:
        sign = "more" if t["delta"] > 0 else "less"
        print(f"  you {t['text']:<45} {sign} than most ({t['delta']:+.2f})")
    print("\nStyle:", ", ".join(f"{k} {v:.0%}" for k, v in prof["style"].items() if v is not None))
    print("Blind spots:", ", ".join(f"{k} {v:.1%}" for k, v in prof["blind_spots"].items() if v is not None))
    for phase, d in prof["by_phase"].items():
        print(f"  {phase:<10} avg loss {d['avg_cp_loss']:>5}cp  blunders {d['blunder_rate']:.1%}")
    c = prof["clock"]
    if c["moves_in_time_pressure"]:
        print(f"Time trouble: blunders {c['blunder_rate_time_pressure']:.1%} vs {c['blunder_rate_normal']:.1%} normally")
    if prof["weakest_openings"]:
        print("\nWhere Nemesis will take you:")
        for o in prof["weakest_openings"]:
            print(f"  {o['opening']:<40} {o['score']:.0%} in {o['games']} games")


def cmd_play(args: argparse.Namespace) -> None:
    from nemesis.engine import Nemesis
    from nemesis.learn import record_game

    store = Store(args.db)
    nem = Nemesis(store, args.username, risk=args.risk, depth=args.depth, deep=args.deep,
                  mode=args.commentary)
    human = chess.WHITE if args.color == "white" else chess.BLACK
    board = chess.Board()
    played: list[str] = []
    decision = None
    try:
        while not board.is_game_over(claim_draw=True):
            print(board.unicode(invert_color=True, borders=True, orientation=human))
            if board.turn == human:
                text = input("your move (SAN/UCI, 'resign'): ").strip()
                if text == "resign":
                    break
                try:
                    move = board.parse_san(text)
                except ValueError:
                    try:
                        move = chess.Move.from_uci(text)
                        if move not in board.legal_moves:
                            raise ValueError
                    except ValueError:
                        print("illegal move")
                        continue
                if decision is not None:
                    say(nem.react(board, decision, move))
            else:
                decision = nem.choose(board)
                move = decision.move
                print(f"\nNemesis plays {board.san(move)}")
                say(decision.lines)
            played.append(move.uci())
            board.push(move)
        result = board.result(claim_draw=True)
        if result == "*":
            result = "0-1" if human == chess.WHITE else "1-0"
        print("result:", result)
        record_game(store, args.username, human, played, result=result)
        print("game saved; run `nemesis train` to learn from it")
    finally:
        nem.close()


def say(lines: list[str]) -> None:
    for line in lines:
        print("  - " + line.replace("**", ""))
    print()


def cmd_uci(args: argparse.Namespace) -> None:
    from nemesis.uci import main
    main(db=args.db)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn
    from nemesis.web.app import create_app
    uvicorn.run(create_app(args.db), host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nemesis", description="The chess engine that learns to beat you.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database path")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("import", help="import a player's games")
    p.add_argument("source", choices=["lichess", "chesscom", "pgn"])
    p.add_argument("username")
    p.add_argument("--max", type=int, default=200)
    p.add_argument("--file", help="PGN file (for source=pgn)")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("train", help="analyse games and fit the player's model")
    p.add_argument("username", nargs="?")
    p.add_argument("--max-games", type=int)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--population", action="store_true", help="train the shared prior on all players")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("profile", help="show what Nemesis knows about a player")
    p.add_argument("username")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("play", help="play Nemesis in the terminal")
    p.add_argument("username")
    p.add_argument("--color", choices=["white", "black"], default="white")
    p.add_argument("--risk", type=float, default=0.5)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--deep", action="store_true")
    p.add_argument("--commentary", choices=["xray", "poker"], default="xray",
                   help="xray: full breakdown, no spoilers; poker: in-character one-liners")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("uci", help="speak UCI on stdin/stdout")
    p.set_defaults(func=cmd_uci)

    p = sub.add_parser("serve", help="run the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    if args.cmd == "train" and not args.population and not args.username:
        parser.error("train needs a username (or --population)")
    args.func(args)


if __name__ == "__main__":
    main()
