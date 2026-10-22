"""Before/after: does Maia-2 help Nemesis predict a real player's moves?

    python scripts/evaluate_maia.py --player macimoar white.pgn black.pgn

Imports the games into a scratch database, analyses them with Stockfish, then trains
on the older games and tests on the newest quarter (never seen in training). It compares:

  before  - Nemesis's generic starting model, and the personal model trained on your games
  Maia-2  - Maia-2 alone, at your rating
  after   - the starting model and the personal model with Maia-2 as a feature

and how each improves as more of your games come in. It also picks the Chess.com ->
Lichess rating offset that best fits your training games.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import chess
import numpy as np

from nemesis import maia
from nemesis.analysis.stockfish import StockfishAnalyzer
from nemesis.features import INDEX
from nemesis.ingest.pgn import import_pgn_text
from nemesis.profile.move_model import MAIA_PRIOR, MoveModel, build_samples, evaluate, prior_vector
from nemesis.store import Store

OFFSETS = (0, 100, 200, 300, 400, 500, 600)


def models_from(directory: Path):
    loaded = {}

    def get(kind: str):
        if kind not in loaded:
            fp16 = directory / f"maia2-{kind}-fp16.onnx"
            path = fp16 if fp16.exists() else directory / f"maia2-{kind}.onnx"
            loaded[kind] = maia.Maia(path, directory / "moves.json")
        return loaded[kind]
    return get


def split(store: Store, player: str, test_frac: float = 0.25) -> tuple[set[str], set[str]]:
    games = sorted(store.games(player), key=lambda g: g.played_at or "")
    cut = int(len(games) * (1 - test_frac))
    return {g.id for g in games[:cut]}, {g.id for g in games[cut:]}


def maia_only(samples) -> dict:
    """Maia-2's own predictions: only its features, weight 1 (the softmax gives back its probabilities)."""
    w = np.zeros(len(prior_vector()))
    w[INDEX["maia"]] = w[INDEX["maia_early"]] = 1.0
    return evaluate(MoveModel(w), samples)


def tune_offset(store: Store, player: str, train_ids: set[str], get) -> tuple[int, dict]:
    """Average log-likelihood of your actual moves under Maia-2, per rating offset."""
    positions = []
    for g in store.games(player):
        if g.id not in train_ids:
            continue
        mine, theirs = maia.game_elos(g)
        kind = maia.game_kind(g)
        for m in store.moves(g.id):
            if m.is_player:
                positions.append((kind, chess.Board(m.fen), m.uci, mine or 1200, theirs or mine or 1200))
    scores = {}
    for off in OFFSETS:
        ll = 0.0
        for kind, board, uci, mine, theirs in positions:
            ll += math.log(max(get(kind).policy(board, mine + off, theirs + off).get(uci, 0), 1e-9))
        scores[off] = ll / len(positions)
    return max(scores, key=scores.get), scores


def row(name: str, m: dict) -> str:
    return f"  {name:44} top-1 {m['top1']:6.1%}   top-3 {m['top3']:6.1%}   log loss {m['nll']:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgns", nargs="+", type=Path)
    ap.add_argument("--player", required=True)
    ap.add_argument("--maia-dir", type=Path, default=maia.DIR)
    ap.add_argument("--db", type=Path, help="reuse this scratch database (keeps the Stockfish analysis)")
    ap.add_argument("--depth", type=int, default=10)
    args = ap.parse_args()
    player = args.player.lower()
    db = args.db or Path(tempfile.mkdtemp()) / "eval.db"
    store = Store(db)
    for pgn in args.pgns:
        n = import_pgn_text(store, pgn.read_text(errors="replace"), player, source="pgn")
        print(f"{pgn.name}: imported {n} new games")
    get = models_from(args.maia_dir)
    train_ids, test_ids = split(store, player)
    print(f"{len(train_ids)} games to train on, {len(test_ids)} newest games to test on")

    best, scores = tune_offset(store, player, train_ids, get)
    print("Chess.com -> Lichess offset, fit on the training games (higher is better):")
    print("  " + "  ".join(f"+{o}: {s:.3f}" for o, s in scores.items()) + f"   -> using +{best}")
    maia.CHESSCOM_TO_LICHESS = best
    store.__dict__.pop("_maia_offset", None)

    with StockfishAnalyzer(store, depth=args.depth) as sf:
        def progress(done, total):
            print(f"\r  analysing game {done}/{total}", end="", flush=True)
        plain = build_samples(store, player, sf, progress=progress)
        print()
        withm = build_samples(store, player, sf, maia_for=get)
    for s in plain + withm:
        s.weight = 1.0
    tr = lambda ss: [s for s in ss if s.game_id in train_ids]   # noqa: E731
    te = lambda ss: [s for s in ss if s.game_id in test_ids]    # noqa: E731
    print(f"{len(tr(plain))} of your moves to train on, {len(te(plain))} to test on\n")

    maia_prior = prior_vector(MAIA_PRIOR)
    results = {
        "Stockfish's best move": None,
        "BEFORE  generic starting model": evaluate(MoveModel(prior_vector()), te(plain)),
        "BEFORE  personal model (Nemesis today)": evaluate(MoveModel().fit(tr(plain)), te(plain)),
        "Maia-2 alone, at your rating": maia_only(te(withm)),
        "AFTER   starting model with Maia-2": evaluate(MoveModel(maia_prior), te(withm)),
        "AFTER   personal model with Maia-2": evaluate(MoveModel().fit(tr(withm), prior=maia_prior), te(withm)),
    }
    base = results["BEFORE  generic starting model"]
    print(f"Predicting your exact move in your {base['n']} newest positions:")
    print(f"  {'Stockfish best move':44} top-1 {base['stockfish_top1']:6.1%}")
    for name, m in results.items():
        if m:
            print(row(name, m))

    print("\nAs your games come in (personal model, trained on your oldest N games):")
    order = sorted({s.game_id for s in tr(plain)},
                   key=lambda gid: next(g.played_at or "" for g in store.games(player) if g.id == gid))
    for n in (0, 5, 10, 20, 40, len(order)):
        if n > len(order):
            continue
        ids = set(order[:n])
        before = evaluate(MoveModel().fit([s for s in plain if s.game_id in ids]), te(plain))
        after = evaluate(MoveModel().fit([s for s in withm if s.game_id in ids], prior=maia_prior), te(withm))
        print(f"  {n:3} games   before: top-1 {before['top1']:6.1%}, log loss {before['nll']:.3f}"
              f"   after: top-1 {after['top1']:6.1%}, log loss {after['nll']:.3f}")
    print(f"\nScratch database: {db}")


if __name__ == "__main__":
    main()
