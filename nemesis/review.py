"""Game Review: accuracy, move labels, eval graph, key moments, and Nemesis's side.

Works for any stored game: games against Nemesis and imported ones. Labels follow
the drop in the mover's expected score (win%), like Lichess and Chess.com.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable

import chess

from nemesis import ratings
from nemesis.analysis.stockfish import Analyzer, win_prob
from nemesis.commentary import _pawns, describe_move
from nemesis.features import VALUE, en_prise
from nemesis.store import Store

REVIEW_VERSION = 3  # bump when review logic changes, so cached reviews are recomputed

LABELS = ["brilliant", "best", "excellent", "good", "inaccuracy", "mistake", "miss", "blunder"]
LABEL_TEXT = {
    "brilliant": "a brilliant move", "best": "the best move", "excellent": "an excellent move",
    "good": "a good move", "inaccuracy": "an inaccuracy", "mistake": "a mistake",
    "miss": "a missed chance", "blunder": "a blunder",
}


def label_for(loss: int, drop: float, is_best: bool, sacrifice: bool, decided: bool,
              punished_before: bool) -> str:
    """Classify a move. `drop` is the fall in the mover's win%, 0-100.

    `decided`: the game was already won or lost, where a sacrifice is no feat.
    """
    if is_best or loss <= 10:
        return "brilliant" if sacrifice and not decided else "best"
    if drop > 20:
        return "blunder"
    if punished_before and drop >= 10:
        return "miss"
    if drop <= 2:
        return "excellent"
    if drop <= 5:
        return "good"
    if drop <= 10:
        return "inaccuracy"
    return "mistake"


def move_accuracy(drop: float) -> float:
    """Lichess's per-move accuracy from the win% drop."""
    return max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * max(0.0, drop)) - 3.1669))


def game_accuracy(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    harmonic = len(values) / sum(1 / max(v, 1.0) for v in values)
    return round((mean + harmonic) / 2, 1)


def is_sacrifice(board: chess.Board, move: chess.Move) -> bool:
    """Gives up at least a minor piece's worth: the moved piece can be taken for less."""
    piece = board.piece_type_at(move.from_square)
    captured = board.piece_type_at(move.to_square) or (chess.PAWN if board.is_en_passant(move) else None)
    if piece in (chess.PAWN, chess.KING):
        return False
    given = VALUE[piece] - (VALUE[captured] if captured else 0)
    if given < 3:
        return False
    board.push(move)
    try:
        return en_prise(board, move.to_square)
    finally:
        board.pop()


def _after_text(cp: int) -> str:
    if cp > 150:
        return "which kept you clearly ahead"
    if cp > 40:
        return "which kept you on top"
    if cp >= -40:
        return "which kept the game level"
    if cp >= -150:
        return "which kept you in the game"
    return "which would have limited the damage"


def _after(board: chess.Board, move: chess.Move) -> str:
    board.push(move)
    try:
        return board.fen()
    finally:
        board.pop()


def review_game(store: Store, analyzer: Analyzer, game_id: str,
                progress: Callable[[int, int], None] | None = None) -> dict:
    game = store.game(game_id)
    if game is None:
        raise ValueError(f"no such game: {game_id}")
    moves = store.moves(game_id)
    annotations = store.get_json("annotations", game_id)
    vs_nemesis = game.source == "nemesis"
    opponent = "Nemesis" if vs_nemesis else (game.opponent or "your opponent")

    rows: list[dict] = []
    graph: list[float] = []
    acc: dict[str, list[float]] = {"you": [], "opponent": []}
    for i, m in enumerate(moves):
        board = chess.Board(m.fen)
        move = chess.Move.from_uci(m.uci)
        lines = analyzer.lines(board, multipv=3, extra=[m.uci])
        best_uci = max(lines, key=lines.get) if lines else m.uci
        best_cp = lines.get(best_uci, 0)
        played_cp = lines.get(m.uci, best_cp)
        loss = max(0, best_cp - played_cp)
        drop = max(0.0, (win_prob(best_cp) - win_prob(played_cp)) * 100)
        side = "you" if m.is_player else "opponent"
        prev = rows[-1] if rows else None
        label = label_for(
            loss, drop, is_best=m.uci == best_uci, sacrifice=is_sacrifice(board, move),
            decided=abs(best_cp) > 400, punished_before=bool(prev and prev["label"] in ("mistake", "blunder")),
        )
        acc[side].append(move_accuracy(drop))
        white_cp = played_cp if board.turn == chess.WHITE else -played_cp
        graph.append(round(win_prob(white_cp) * 100, 1))
        best_move = chess.Move.from_uci(best_uci)
        rows.append({
            "ply": i, "number": board.fullmove_number, "white": board.turn == chess.WHITE,
            "side": side, "san": board.san(move), "uci": m.uci, "text": describe_move(board, move),
            "best_san": board.san(best_move), "best_uci": best_uci, "best_text": describe_move(board, best_move),
            "loss": loss, "drop": round(drop, 1), "label": label, "eval_white": white_cp,
            "best_cp": best_cp, "fen_before": m.fen, "fen_after": _after(board, move),
        })
        if progress:
            progress(i + 1, len(moves))

    counts = {side: dict(Counter(r["label"] for r in rows if r["side"] == side)) for side in ("you", "opponent")}
    result = {
        "game": {
            "id": game.id, "source": game.source, "date": game.played_at, "opening": game.opening,
            "you": "white" if game.color else "black", "opponent": opponent, "result": game.result,
            "start_fen": moves[0].fen if moves else chess.STARTING_FEN,
        },
        "moves": rows,
        "accuracy": {"you": game_accuracy(acc["you"]), "opponent": game_accuracy(acc["opponent"])},
        "counts": counts,
        "graph": graph,
        "key_moments": key_moments(rows, opponent),
        "nemesis": nemesis_side(rows, annotations) if annotations else None,
        "ratings": rating_changes(store, game_id),
        "depth": getattr(analyzer, "depth", 0),
        "version": REVIEW_VERSION,
    }
    store.put_json("reviews", game_id, result, depth=result["depth"])
    return result


def key_moments(rows: list[dict], opponent: str, limit: int = 5) -> list[dict]:
    weight = {"blunder": 1.0, "miss": 0.9, "mistake": 0.8}

    def importance(r: dict) -> float:
        return 10.0 if r["label"] == "brilliant" else (r["drop"] + 15) * weight[r["label"]]

    picked = sorted((r for r in rows if r["label"] in weight or r["label"] == "brilliant"),
                    key=importance, reverse=True)[:limit]
    out = []
    for r in sorted(picked, key=lambda r: r["ply"]):
        who = "you" if r["side"] == "you" else opponent
        Who = who[0].upper() + who[1:]
        num = f"{r['number']}{'' if r['white'] else ' (Black)'}"
        if r["label"] == "brilliant":
            text = (f"Move {num}: {Who} played {r['text']}. Brilliant: it gives up material, "
                    "and it's still the best move on the board.")
        else:
            what = LABEL_TEXT[r["label"]]
            if r["label"] == "miss":
                lead = f"Move {num}: {Who} had a chance to punish the previous mistake but played {r['text']}."
            else:
                lead = f"Move {num}: {Who} played {r['text']}."
            text = (f"{lead} That's {what}: it gives away about {_pawns(r['loss'])}. "
                    f"Better was {r['best_text']}, {_after_text(r['best_cp'])}.")
            if who != "you":
                text = text.replace("kept you", "kept them")
        out.append({"ply": r["ply"], "label": r["label"], "side": r["side"], "text": text})
    return out


def nemesis_side(rows: list[dict], annotations: dict) -> dict:
    """Traps set and sprung, and how predictable you were, from what Nemesis saved at play time."""
    by_ply = {r["ply"]: r for r in rows}
    traps: list[dict] = []
    predictions, slips = [], 0
    for a in annotations.get("moves", []):
        reply = by_ply.get(a["ply"] + 1)
        nem_move = by_ply.get(a["ply"])
        slips += bool(a.get("lapse"))
        if reply is None:
            continue
        prob = a["probs"].get(reply["uci"], 0.0)
        predictions.append({"ply": reply["ply"], "san": reply["san"], "prob": prob,
                            "top": a.get("top") == reply["uci"]})
        trap = a.get("trap")
        if trap:
            sprung = trap["uci"] == reply["uci"]
            board = chess.Board(reply["fen_before"])
            trap_text = describe_move(board, chess.Move.from_uci(trap["uci"]))
            why = f" because you {' and '.join(trap['reasons'])}" if trap.get("reasons") else ""
            nem_san = nem_move["san"] if nem_move else "its move"
            if sprung:
                text = (f"After {nem_san}, Nemesis hoped you'd play {trap_text}{why}, and you did. "
                        f"It cost you about {_pawns(trap['loss'])}.")
            else:
                text = f"After {nem_san}, Nemesis hoped you'd play {trap_text}{why}. You didn't fall for it."
            # Same bait (described the same, even if the piece moved meanwhile) left out again: one entry.
            same = traps[-1] if traps and traps[-1]["desc"] == trap_text and not traps[-1]["sprung"] else None
            if same:
                same["moves"] += 1
                same["sprung"], same["text"] = sprung, text
                if not sprung:
                    same["text"] = (f"For {same['moves']} moves, starting with {same['first_san']}, Nemesis "
                                    f"left out the same bait, hoping you'd play {trap_text}{why}. "
                                    "You never fell for it.")
            else:
                traps.append({"ply": reply["ply"], "desc": trap_text, "sprung": sprung, "prob": trap["prob"],
                              "text": text, "moves": 1, "first_san": nem_san})
    n = len(predictions)
    hits = sum(p["top"] for p in predictions)
    ordered = sorted(predictions, key=lambda p: p["prob"])
    elos = [a["elo"] for a in annotations.get("moves", []) if a.get("elo")]
    return {
        "traps_set": len(traps),
        "traps_sprung": sum(t["sprung"] for t in traps),
        "traps": traps,
        "predicted_top": hits,
        "predictions": n,
        "predictability": round(hits / n, 3) if n else None,
        "most_predictable": ordered[-1] if ordered else None,
        "most_surprising": ordered[0] if ordered else None,
        "elo": round(sum(elos) / len(elos)) if elos else None,
        "slips": slips,
        "risk": annotations.get("risk"),
    }


def rating_changes(store: Store, game_id: str) -> dict | None:
    changes = store.rating_change(game_id)
    if not changes:
        return None
    out = {}
    for entity, (before, after) in changes.items():
        info = ratings.parse_entity(entity)
        out[info["kind"] if info["kind"] == "nemesis" else "you"] = {
            "before": round(before), "after": round(after), "change": round(after - before),
            "tc": info["tc"], **({"tier": info["tier"]} if "tier" in info else {}),
        }
    return out


def cached_review(store: Store, game_id: str, depth: int) -> dict | None:
    """A stored review, if it is current: same review logic and at least this depth."""
    cached = store.get_json("reviews", game_id, min_depth=depth)
    return cached if cached and cached.get("version") == REVIEW_VERSION else None


def get_review(store: Store, analyzer: Analyzer, game_id: str) -> dict:
    """Cached review, recomputed if missing, outdated, or analysed at a lower depth than now."""
    cached = cached_review(store, game_id, getattr(analyzer, "depth", 0))
    if cached is not None:
        cached["ratings"] = rating_changes(store, game_id)
        return cached
    return review_game(store, analyzer, game_id)
