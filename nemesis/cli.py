"""Command line: nemesis import | train | profile | play | games | review | ratings | uci | serve."""

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
    print(f"trained on {metrics['samples']} moves from {metrics['games']} games; "
          f"{metrics.get('new_puzzles', 0)} new puzzles")
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
                  mode=args.commentary, elo=args.elo or None)
    print(f"Nemesis plays at about {nem.target_elo():.0f}.")
    annotations: list[dict] = []
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
                    annotations[-1]["reply"] = decision.reply
            else:
                decision = nem.choose(board)
                move = decision.move
                annotations.append(decision.annotation(len(board.move_stack), args.risk))
                print(f"\nNemesis plays {board.san(move)}")
                say(decision.lines)
            played.append(move.uci())
            board.push(move)
        result = board.result(claim_draw=True)
        if result == "*":
            result = "0-1" if human == chess.WHITE else "1-0"
        print("result:", result)
        gid = record_game(store, args.username, human, played, result=result, risk=args.risk,
                          annotations=annotations)
        print(f"game saved as {gid}; run `nemesis train` to learn from it, `nemesis review` to review it")
        for entity, (before, after) in store.rating_change(gid).items():
            who = "you" if entity.startswith("player:") else "Nemesis"
            print(f"  rating ({who}): {before:.0f} -> {after:.0f} ({after - before:+.0f})")
    finally:
        nem.close()


def cmd_games(args: argparse.Namespace) -> None:
    store = Store(args.db)
    for g in reversed(store.games(args.username)[-args.limit:]):
        opp = "Nemesis" if g.source == "nemesis" else g.opponent
        res = {1.0: "won", 0.0: "lost", 0.5: "draw"}.get(g.result, "?")
        print(f"{g.id:<28} {(g.played_at or '')[:10]:<10} vs {opp:<20} {res:<5} {g.opening or ''}")


def cmd_ratings(args: argparse.Namespace) -> None:
    from nemesis import ratings
    r = ratings.summary(Store(args.db), args.username)
    q = lambda p: "?" if p else ""
    me = r["player"]
    print(f"You: {me['rating']}{q(me['provisional'])}"
          + ("  (seeded from your imported games)" if me["seeded_from_imports"] else f"  ({me['games']} rated game{'' if me['games'] == 1 else 's'})"))
    for t in r["nemesis"]:
        print(f"  Nemesis {t['label']:<9} {t['rating']}{q(t['provisional'])}  "
              f"({t['games']} game{'' if t['games'] == 1 else 's'})")
    for h in r["history"][:10]:
        print(f"  {(h['played_at'] or '')[:10]}  {h['before']:.0f} -> {h['after']:.0f} ({h['after'] - h['before']:+.0f})")


def cmd_review(args: argparse.Namespace) -> None:
    from nemesis.analysis.stockfish import StockfishAnalyzer
    from nemesis.review import LABELS, get_review

    store = Store(args.db)
    games = store.games(args.username)
    if not games:
        sys.exit(f"no games for {args.username}")
    gid = args.game or games[-1].id
    with StockfishAnalyzer(store, depth=args.depth) as analyzer:
        r = get_review(store, analyzer, gid)
    g = r["game"]
    res = {1.0: "you won", 0.0: "you lost", 0.5: "draw"}.get(g["result"], "unfinished")
    print(f"Review of {gid}: you ({g['you']}) vs {g['opponent']}, {res}")
    print(f"  Accuracy: you {r['accuracy']['you']}  {g['opponent']} {r['accuracy']['opponent']}")
    for side, name in (("you", "You"), ("opponent", g["opponent"])):
        c = r["counts"][side]
        print(f"  {name:<10} " + ", ".join(f"{c[l]} {l}" for l in LABELS if c.get(l)))
    if r["ratings"]:
        y = r["ratings"]["you"]
        print(f"  Rating: {y['before']} -> {y['after']} ({y['change']:+})")
    print("\nKey moments:")
    for k in r["key_moments"] or [{"text": "No big swings. A clean game."}]:
        print("  - " + k["text"])
    n = r["nemesis"]
    if n:
        plural = "" if n["traps_set"] == 1 else "s"
        print(f"\nNemesis's side: {n['traps_set']} trap{plural} set, you fell for {n['traps_sprung']}.")
        for t in n["traps"]:
            print("  - " + t["text"])
        if n["predictions"]:
            print(f"  It predicted your exact move {n['predicted_top']} of {n['predictions']} times.")
    print("\nMoves:")
    for m in r["moves"]:
        num = f"{m['number']}{'.' if m['white'] else '...'}"
        note = "" if m["label"] in ("best", "brilliant", "excellent", "good") else f"  (best: {m['best_san']})"
        print(f"  {num:<6} {m['san']:<8} {m['label']}{note}")


def say(lines: list[str]) -> None:
    for line in lines:
        print("  - " + line.replace("**", ""))
    print()


def cmd_uci(args: argparse.Namespace) -> None:
    from nemesis.uci import main
    main(db=args.db)


def cmd_serve(args: argparse.Namespace) -> None:
    from nemesis.launcher import serve
    serve(args.db, host=args.host, port=args.port, open_browser=args.open)


def cmd_install_stockfish(args: argparse.Namespace) -> None:
    from nemesis.analysis.install import ensure_stockfish

    def show(msg: str, frac: float) -> None:
        print(f"\r{msg}: {frac:4.0%}", end="", flush=True)

    path = ensure_stockfish(progress=show)
    print(f"\nStockfish ready: {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nemesis", description="The chess engine that learns to beat you.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database path")
    sub = parser.add_subparsers(dest="cmd")

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
    p.add_argument("--elo", type=int, default=0, help="Nemesis's strength (default: your rating + 50)")
    p.add_argument("--commentary", choices=["xray", "poker"], default="xray",
                   help="xray: full breakdown, no spoilers; poker: in-character one-liners")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("games", help="list a player's games")
    p.add_argument("username")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_games)

    p = sub.add_parser("review", help="review a game: accuracy, mistakes, key moments")
    p.add_argument("username")
    p.add_argument("--game", help="game id (see `nemesis games`); default: the latest")
    p.add_argument("--last", action="store_true", help="review the latest game (the default)")
    p.add_argument("--depth", type=int, default=10)
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("ratings", help="your rating and Nemesis's, per risk tier")
    p.add_argument("username")
    p.set_defaults(func=cmd_ratings)

    p = sub.add_parser("uci", help="speak UCI on stdin/stdout")
    p.set_defaults(func=cmd_uci)

    p = sub.add_parser("serve", help="run the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, help="default: 8000, or the next free port")
    p.add_argument("--open", action="store_true", help="open the browser too")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("install-stockfish", help="download Stockfish for this computer")
    p.set_defaults(func=cmd_install_stockfish)

    args = parser.parse_args(argv)
    if args.cmd is None:  # plain `nemesis`: start the app and open the browser
        from nemesis.launcher import serve
        serve(args.db)
        return
    if args.cmd == "train" and not args.population and not args.username:
        parser.error("train needs a username (or --population)")
    args.func(args)


if __name__ == "__main__":
    main()
