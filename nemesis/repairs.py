"""Opening repair: the moves you keep playing in your openings that keep costing you.

A "spot" is a position in the first 12 moves that you've reached in several games,
where your usual move loses at least half a pawn. Spots are grouped by opening, worst-
scoring openings first, and drilled until you find a good move three times in a row.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict

import chess

from nemesis.analysis.stockfish import Analyzer
from nemesis.features import position_key
from nemesis.profile.move_model import Sample
from nemesis.puzzles import THEMES, _san_line, themes_for, what_went_wrong
from nemesis.store import Store

MAX_PLY = 24         # the opening: your first 12 moves
MIN_TIMES = 2        # a habit: you played it in at least this many games
MIN_LOSS = 50        # ...and it costs at least this much on average
GOOD = 50            # moves within this of the best count as a repair
LEARNED = 3          # right this many times in a row...
SPACING = dt.timedelta(hours=1)   # ...at least this far apart, so it's memory, not repetition
LOST = -300          # past this, the opening has already gone wrong earlier
LINE_PLIES = 5


def _opening_name(store: Store, game_ids: list[str], line: list[str], start_fen: str) -> str:
    names = Counter()
    for gid in set(game_ids):
        g = store.game(gid)
        if g and g.opening:
            names[g.opening.split(":")[0].strip()] += 1
    if names:
        return names.most_common(1)[0][0]
    board = chess.Board(start_fen)
    return board.variation_san([chess.Move.from_uci(u) for u in line[:4]]) or "Starting position"


def build(store: Store, player: str, analyzer: Analyzer, samples: list[Sample]) -> list[dict]:
    """Find the opening habits worth repairing and save them. Returns the spots."""
    player = player.lower()
    visits: dict[str, list[Sample]] = defaultdict(list)
    for smp in samples:                    # oldest game first
        if smp.ply < MAX_PLY:
            visits[smp.key].append(smp)
    moves_cache: dict[str, list] = {}
    results = {g.id: g.result for g in store.games(player)}
    spots = []
    for key, seen in visits.items():
        if len({s.game_id for s in seen}) < MIN_TIMES:
            continue
        counts = Counter(s.moves[s.chosen] for s in seen)
        habit, times = counts.most_common(1)[0]
        losses = [s.loss for s in seen if s.moves[s.chosen] == habit]
        if times < MIN_TIMES or sum(losses) / len(losses) < MIN_LOSS:
            continue
        last = seen[-1]
        if last.game_id not in moves_cache:
            moves_cache[last.game_id] = store.moves(last.game_id)
        game_moves = moves_cache[last.game_id]
        here = next((m for m in game_moves if m.ply == last.ply), None)
        if here is None:
            continue
        board = chess.Board(here.fen)
        lines = analyzer.lines(board, multipv=6, extra=[habit])
        if not lines or habit not in lines:
            continue
        best_uci = max(lines, key=lines.get)
        best = lines[best_uci]
        if best < LOST:
            continue
        good = [u for u, cp in sorted(lines.items(), key=lambda kv: -kv[1]) if cp >= best - GOOD]
        if habit in good:
            continue
        line = [m.uci for m in sorted(game_moves, key=lambda m: m.ply) if m.ply < last.ply]
        start_fen = game_moves[0].fen if game_moves else chess.STARTING_FEN
        scores = [results.get(s.game_id) for s in seen]
        score = sum(0.5 if r is None else r for r in scores) / len(scores)
        pv = analyzer.pv(board, first=best_uci, plies=LINE_PLIES) if hasattr(analyzer, "pv") else []
        idea = pv if pv and pv[0] == best_uci else [best_uci]
        loss = round(sum(losses) / len(losses))
        recent = last.moves[last.chosen]
        path = set()
        b = chess.Board(start_fen)
        for u in line:
            path.add(position_key(b))
            b.push(chess.Move.from_uci(u))
        spots.append({"_path": path, "_ply": last.ply,
            "id": key, "fen": here.fen, "start_fen": start_fen, "line": line,
            "opening": _opening_name(store, [s.game_id for s in seen], line, start_fen),
            "color": "white" if board.turn == chess.WHITE else "black",
            "move_number": board.fullmove_number,
            "habit": habit, "habit_san": board.san(chess.Move.from_uci(habit)),
            "times": times, "visits": len(seen), "loss": loss, "score": round(score, 3),
            "wrong": what_went_wrong(board, chess.Move.from_uci(habit)),
            "best": best_uci, "best_san": board.san(chess.Move.from_uci(best_uci)),
            "good": good, "lines": lines, "idea": idea, "idea_san": _san_line(board, idea),
            "themes": themes_for(board, chess.Move.from_uci(best_uci), best),
            "recent_good": recent in good,
            "priority": round(times * min(loss, 300) / 100 * (1.5 - score), 3),
        })
    # Fix the first habit in a line first: positions further down only happen because of it.
    kept, keys = [], set()
    for spot in sorted(spots, key=lambda s: s["_ply"]):
        if not spot.pop("_path") & keys:
            keys.add(spot["id"])
            kept.append(spot)
        spot.pop("_ply")
    spots = sorted(kept, key=lambda s: -s["priority"])
    store.put_repairs(player, spots)
    return spots


def _time(at: str) -> dt.datetime:
    return dt.datetime.fromisoformat(at)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def _progress(attempts: list[tuple[bool, str]]) -> tuple[int, dt.datetime | None]:
    """(streak, time of the last correct answer that counted). A miss resets the streak;
    a correct answer within SPACING of the last one that counted doesn't add to it."""
    streak, counted = 0, None
    for ok, at in attempts:
        if not ok:
            streak, counted = 0, None
        elif counted is None or _time(at) - counted >= SPACING:
            streak, counted = streak + 1, _time(at)
    return streak, counted


def _streak(attempts: list[tuple[bool, str]]) -> int:
    return _progress(attempts)[0]


def resting(attempts: list[tuple[bool, str]], now: dt.datetime | None = None) -> bool:
    """Answered correctly within the last SPACING: come back to it later."""
    streak, counted = _progress(attempts)
    return bool(streak) and attempts[-1][0] and (now or _now()) - counted < SPACING


def summary(store: Store, player: str) -> dict:
    spots = store.repairs(player)
    tries = store.repair_attempts(player)
    openings: dict[str, dict] = {}
    for s in spots:
        o = openings.setdefault(s["opening"], {"opening": s["opening"], "spots": 0, "learned": 0,
                                               "visits": 0, "score_sum": 0.0, "fixed_in_games": 0})
        o["spots"] += 1
        o["learned"] += _streak(tries.get(s["id"], [])) >= LEARNED
        o["visits"] += s["visits"]
        o["score_sum"] += s["score"] * s["visits"]
        o["fixed_in_games"] += s["recent_good"]
    rows = []
    for o in openings.values():
        o["score"] = round(o.pop("score_sum") / o["visits"], 3) if o["visits"] else None
        rows.append(o)
    rows.sort(key=lambda o: (o["score"] if o["score"] is not None else 0.5, -o["spots"]))
    for o in rows:
        o["resting"] = sum(resting(tries.get(s["id"], [])) for s in spots if s["opening"] == o["opening"])
    return {"total": len(spots), "learned": sum(o["learned"] for o in rows),
            "resting": sum(o["resting"] for o in rows), "openings": rows}


def next_spot(store: Store, player: str, opening: str | None = None, skip: str | None = None) -> dict | None:
    """Spots you haven't learned yet, in your worst openings first; your biggest habits first.
    Spots you just got right rest for a while; ones you missed come back straight away."""
    spots = [s for s in store.repairs(player) if opening in (None, "", s["opening"])]
    tries = store.repair_attempts(player)
    spots = [s for s in spots if not resting(tries.get(s["id"], []))]
    if not spots:
        return None
    opening_rank = {o["opening"]: i for i, o in enumerate(summary(store, player)["openings"])}
    pool = [s for s in spots if s["id"] != skip] or spots

    def key(s):
        streak = _streak(tries.get(s["id"], []))
        return (streak >= LEARNED, opening_rank.get(s["opening"], 0), streak, -s["priority"])

    return min(pool, key=key)


def public(store: Store, player: str, s: dict) -> dict:
    """What the board needs before you answer: the line to play through and the position."""
    board = chess.Board(s["start_fen"])
    fens = []
    for u in s["line"]:
        board.push(chess.Move.from_uci(u))
        fens.append(board.fen())
    legal: dict[str, list[str]] = {}
    for m in board.legal_moves:
        legal.setdefault(chess.square_name(m.from_square), []).append(m.uci())
    tries = store.repair_attempts(player).get(s["id"], [])
    return {"id": s["id"], "fen": s["fen"], "start_fen": s["start_fen"], "color": s["color"],
            "move_number": s["move_number"], "opening": s["opening"], "legal": legal,
            "check": board.is_check(), "intro": {"line": s["line"], "fens": fens},
            "habit_san": s["habit_san"], "times": s["times"], "visits": s["visits"],
            "hint": THEMES[s["themes"][0]], "streak": _streak(tries), "tries": len(tries)}


def explain(s: dict, uci: str, correct: bool, streak: int, counted: bool = True) -> list[str]:
    board = chess.Board(s["fen"])
    san = board.san(chess.Move.from_uci(uci)) if uci else ""
    out = []
    if correct:
        top = "" if uci == s["best"] else f" The engine's top choice is {s['best_san']}."
        out.append(f"Good: {san}.{top}")
    elif not uci:
        out.append(f"A good move here is {s['best_san']}.")
    elif uci == s["habit"]:
        out.append(f"That's your usual move again. Better is {s['best_san']}.")
    elif uci in s["lines"]:
        cost = max(s["lines"].values()) - s["lines"][uci]
        out.append(f"{san} costs about {cost / 100:.1f} pawns too. Better is {s['best_san']}.")
    else:
        out.append(f"{san} isn't one of the good moves here. Better is {s['best_san']}.")
    wrong = f", and {s['wrong']}" if s.get("wrong") else ""
    out.append(f"You've played {s['habit_san']} here in {s['times']} of {s['visits']} games. "
               f"It costs about {s['loss'] / 100:.1f} pawns{wrong}.")
    if len(s.get("idea_san", [])) > 1:
        out.append("The idea: " + " ".join(s["idea_san"]))
    if correct and streak >= LEARNED:
        out.append("Learned: you've found a good move three times running.")
    elif correct and counted:
        out.append(f"{streak} of {LEARNED} in a row to learn this one. It rests for an hour, "
                   "so the next one tests your memory.")
    elif correct:
        out.append(f"Still {streak} of {LEARNED}: answers less than an hour apart only count once.")
    if s.get("recent_good"):
        out.append("In your most recent game here you got it right. Keep it up.")
    return out


def attempt(store: Store, player: str, key: str, uci: str) -> dict:
    s = next((x for x in store.repairs(player) if x["id"] == key), None)
    if s is None:
        raise KeyError(key)
    correct = uci in s["good"]
    before = _streak(store.repair_attempts(player).get(key, []))
    store.add_repair_attempt(player, key, uci, correct)
    streak = _streak(store.repair_attempts(player).get(key, []))
    line = [uci] if correct and uci != s["best"] else s["idea"]
    board = chess.Board(s["fen"])
    fens = []
    for u in line:
        board.push(chess.Move.from_uci(u))
        fens.append(board.fen())
    return {"solved": correct, "line": line, "fens": fens, "streak": streak, "learned": streak >= LEARNED,
            "also_good": correct and uci != s["best"], "explanation": explain(s, uci, correct, streak, counted=streak > before)}
