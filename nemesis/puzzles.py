"""Puzzles from your own mistakes.

Every position where you lost at least 1.5 pawns, and where one move was clearly
best, becomes a "find the move" exercise. Difficulty is personal: it's the chance
your move model gives that *you* would find the move, so the puzzles you'd most
likely miss come first.
"""

from __future__ import annotations

import chess

from nemesis.analysis.stockfish import MATE_CP, Analyzer
from nemesis.commentary import PIECE, describe_move
from nemesis.features import VALUE, created_tactics, en_prise, hanging_squares, opponent_tactics
from nemesis.profile.move_model import MoveModel, Sample
from nemesis.store import Store

MIN_LOSS = 150       # your move cost at least this much
MIN_GAP = 100        # the solution beats every other move by at least this much
ALSO_GOOD = 50       # moves within this of the best are also accepted
LOST = -300          # skip positions that were already lost whatever you played
LINE_PLIES = 5
MAX_TRIES = 3        # wrong moves allowed before the solution is shown
THEMES_VERSION = 2   # bump when themes or explanations change: existing puzzles are re-described

THEMES = {  # the first hint: the idea, not the move
    "back_rank": "Their king is stuck on its back rank.",
    "mate": "There's a checkmate.",
    "fork": "One piece can attack two things at once.",
    "pin": "One of their pieces is stuck in front of a more valuable one.",
    "skewer": "Two of their pieces stand on one line.",
    "discovered": "Moving one piece can uncover an attack by another.",
    "material": "You can win material.",
    "defend": "One of your pieces is in danger.",
    "threat": "Your opponent is threatening something.",
    "best": "Look for the move that keeps your advantage.",
}


def themes_for(board: chess.Board, solution: chess.Move, best_cp: int) -> list[str]:
    me = board.turn
    board.push(solution)
    mate_now = board.is_checkmate()
    tactics = created_tactics(board, solution)
    board.pop()
    themes = ["back_rank"] if "back_rank" in tactics else []
    if (mate_now or best_cp >= MATE_CP) and not themes:
        themes.append("mate")
    themes += [t for t in ("fork", "pin", "skewer", "discovered") if t in tactics]
    if board.is_capture(solution) and not board.is_en_passant(solution) and en_prise(board, solution.to_square):
        themes.append("material")
    in_danger = hanging_squares(board, me)
    if in_danger:
        board.push(solution)
        saved = not hanging_squares(board, me)
        board.pop()
        if saved:
            themes.append("defend")
    if not themes:
        # Would doing nothing have let the opponent win something?
        null = board.copy(stack=False)
        null.push(chess.Move.null())
        if not null.is_check() and (opponent_tactics(null, me) or hanging_squares(null, me) or
                                    any(null.gives_check(m) for m in null.legal_moves if null.is_capture(m))):
            themes.append("threat")
    return themes or ["best"]


ALLOWED = {"back_rank": "it allowed a back-rank mate", "fork": "it allowed a fork",
           "pin": "it allowed a pin", "skewer": "it allowed a skewer",
           "discovered": "it allowed a discovered attack"}


def what_went_wrong(board: chess.Board, played: chess.Move) -> str | None:
    """A short reason your move was a mistake, when there's an obvious one."""
    me = board.turn
    after = board.copy(stack=False)
    after.push(played)
    hanging = hanging_squares(after, me)
    taken = board.piece_type_at(played.to_square)
    if taken and VALUE[taken] >= VALUE[board.piece_type_at(played.from_square)]:
        hanging.discard(played.to_square)   # a fair trade, not a piece left hanging
    hanging = sorted(hanging, key=lambda sq: -VALUE[after.piece_type_at(sq)])
    if hanging:
        sq = hanging[0]
        piece = f"your {PIECE[after.piece_type_at(sq)]} on {chess.square_name(sq)}"
        # Was that piece already in trouble before your move (and didn't move)?
        if sq != played.to_square and sq in hanging_squares(board, me):
            return f"it didn't deal with the attack on {piece}"
        return f"it left {piece} open to capture"
    tactics = opponent_tactics(after, me)
    for t in ("back_rank", "fork", "pin", "skewer", "discovered"):
        if t in tactics:
            return ALLOWED[t]
    for reply in after.legal_moves:
        after.push(reply)
        mate = after.is_checkmate()
        after.pop()
        if mate:
            return "it allowed checkmate"
    return None


def _san_line(board: chess.Board, ucis: list[str]) -> list[str]:
    b = board.copy(stack=False)
    out = []
    for u in ucis:
        move = chess.Move.from_uci(u)
        if move not in b.legal_moves:
            break
        out.append(b.san(move))
        b.push(move)
    return out


def make_puzzle(board: chess.Board, played: str, lines: dict[str, int], analyzer: Analyzer,
                chance: float | None = None) -> dict | None:
    """The puzzle for this position, or None if it doesn't make a good one."""
    if len(lines) < 2 or played not in {m.uci() for m in board.legal_moves}:
        return None
    ranked = sorted(lines.items(), key=lambda kv: -kv[1])
    best_uci, best = ranked[0]
    if best < LOST:
        return None
    good = [u for u, cp in ranked if cp >= best - ALSO_GOOD]
    rest = [cp for u, cp in ranked if cp < best - ALSO_GOOD]
    if len(good) > 2 or not rest or best - max(rest) < MIN_GAP or played in good:
        return None
    if best - lines.get(played, min(rest)) < MIN_LOSS:
        return None
    solution = chess.Move.from_uci(best_uci)
    pv = analyzer.pv(board, first=best_uci, plies=LINE_PLIES) if hasattr(analyzer, "pv") else []
    line = pv if pv and pv[0] == best_uci else [best_uci]
    played_move = chess.Move.from_uci(played)
    return {
        "solution_san": board.san(solution),
        "also": [u for u in good if u != best_uci],
        "themes": themes_for(board, solution, best),
        "played": played,
        "played_san": board.san(played_move),
        "played_text": describe_move(board, played_move),
        "wrong": what_went_wrong(board, played_move),
        "loss": best - lines.get(played, min(rest)),
        "eval": best,
        "line": line,
        "line_san": _san_line(board, line),
        "chance": chance,
        "v": THEMES_VERSION,
    }


def mine(store: Store, player: str, analyzer: Analyzer, samples: list[Sample],
         model: MoveModel | None = None) -> int:
    """Turn new mistakes into puzzles. Returns how many were added."""
    player = player.lower()
    redescribe(store, player)
    have = store.puzzle_ids(player)
    fens: dict[str, dict[int, str]] = {}
    added = 0
    for smp in samples:
        pid = f"{smp.game_id}#{smp.ply}"
        if smp.loss < MIN_LOSS or pid in have:
            continue
        if smp.game_id not in fens:
            fens[smp.game_id] = {m.ply: m.fen for m in store.moves(smp.game_id)}
        fen = fens[smp.game_id].get(smp.ply)
        if fen is None:
            continue
        board = chess.Board(fen)
        played = smp.moves[smp.chosen]
        lines = analyzer.lines(board, multipv=6, extra=[played])
        best_uci = max(lines, key=lines.get) if lines else None
        chance = None
        if model is not None and best_uci in smp.moves:
            chance = float(model.probs(smp.X)[smp.moves.index(best_uci)])
        puzzle = make_puzzle(board, played, lines, analyzer, chance)
        if puzzle is None:
            continue
        game = store.game(smp.game_id)
        puzzle.update({
            "color": "white" if board.turn == chess.WHITE else "black",
            "move_number": board.fullmove_number,
            "opponent": ("Nemesis" if game.source == "nemesis" else game.opponent) if game else None,
            "date": (game.played_at or "")[:10].replace(".", "-") if game else "",
        })
        store.put_puzzle(pid, player, smp.game_id, smp.ply, fen, best_uci, puzzle)
        added += 1
    return added


def redescribe(store: Store, player: str) -> None:
    """Bring puzzles mined by an older version up to date: new themes and explanations."""
    keep = ("id", "player", "game_id", "ply", "fen", "solution", "attempts")
    for p in store.puzzles(player):
        if p.get("v", 1) >= THEMES_VERSION:
            continue
        board = chess.Board(p["fen"])
        data = {k: v for k, v in p.items() if k not in keep}
        data["themes"] = themes_for(board, chess.Move.from_uci(p["solution"]), p["eval"])
        data["wrong"] = what_went_wrong(board, chess.Move.from_uci(p["played"]))
        data["v"] = THEMES_VERSION
        store.put_puzzle(p["id"], player, p["game_id"], p["ply"], p["fen"], p["solution"], data)


def status(p: dict) -> str:
    tries = p.get("attempts", [])
    if not tries:
        return "new"
    return "solved" if tries[-1]["solved"] else "missed"


def summary(store: Store, player: str) -> dict:
    puzzles = store.puzzles(player)
    counts = {"new": 0, "solved": 0, "missed": 0}
    for p in puzzles:
        counts[status(p)] += 1
    first_try = [p["attempts"][0]["solved"] and p["attempts"][0]["tries"] == 1
                 for p in puzzles if p["attempts"]]
    themes: dict[str, int] = {}
    for p in puzzles:
        for t in p["themes"]:
            themes[t] = themes.get(t, 0) + 1
    return {"total": len(puzzles), **counts,
            "first_try": round(100 * sum(first_try) / len(first_try)) if first_try else None,
            "themes": dict(sorted(themes.items(), key=lambda kv: -kv[1]))}


def next_puzzle(store: Store, player: str, skip: str | None = None) -> dict | None:
    """New puzzles first (the ones you're least likely to solve first), then ones you
    missed, then solved ones for review. `skip` avoids repeating the one just played."""
    puzzles = [p for p in store.puzzles(player) if p["id"] != skip] or store.puzzles(player)
    if not puzzles:
        return None
    order = {"new": 0, "missed": 1, "solved": 2}

    def key(p):
        chance = p.get("chance")
        return (order[status(p)], len(p["attempts"]), chance if chance is not None else 1.0, p["id"])

    return min(puzzles, key=key)


def public(p: dict) -> dict:
    """What the board needs before you've answered: no solution, no line, not even your old move."""
    board = chess.Board(p["fen"])
    legal: dict[str, list[str]] = {}
    for m in board.legal_moves:
        legal.setdefault(chess.square_name(m.from_square), []).append(m.uci())
    return {"id": p["id"], "fen": p["fen"], "color": p["color"], "move_number": p["move_number"],
            "opponent": p.get("opponent"), "date": p.get("date"), "status": status(p),
            "chance": p.get("chance"), "hint": THEMES[p["themes"][0]], "hint2": second_hint(p),
            "legal": legal, "check": board.is_check(), "max_tries": MAX_TRIES}


def second_hint(p: dict) -> str:
    """What you played in the real game, and what was wrong with it."""
    why = p["wrong"] if p.get("wrong") else f"it cost about {p['loss'] / 100:.1f} pawns"
    return f"In the game you played {p['played_san']} here. It's not that: {why}."


ORDINAL = {1: "", 2: " on your second try", 3: " on your third try"}


def explain(p: dict, solved: bool, uci: str = "", tries: int = 1) -> list[str]:
    """Plain-English lines shown after you answer."""
    who = f"against {p['opponent']}" if p.get("opponent") else "in one of your games"
    when = f" on {p['date']}" if p.get("date") else ""
    lines = []
    if solved and uci != p["solution"]:
        san = chess.Board(p["fen"]).san(chess.Move.from_uci(uci))
        lines.append(f"Correct{ORDINAL.get(tries, '')}: {san} works too. "
                     f"The engine's top choice was {p['solution_san']}.")
    elif solved:
        lines.append(f"Correct{ORDINAL.get(tries, '')}: {p['solution_san']}.")
    else:
        lines.append(f"The move was {p['solution_san']}.")
    wrong = f", and {p['wrong']}" if p.get("wrong") else ""
    lines.append(f"{who.capitalize()}{when} you played {p['played_san']} here. "
                 f"That cost about {p['loss'] / 100:.1f} pawns{wrong}.")
    if len(p.get("line_san", [])) > 1:
        lines.append("The idea: " + " ".join(p["line_san"]))
    if p.get("chance") is not None:
        lines.append(f"Nemesis gave you a {round(100 * p['chance'])}% chance of finding it.")
    return lines


def attempt(store: Store, pid: str, uci: str, hints: int = 0, tried: list[str] | None = None) -> dict:
    """Check one try. `tried` are your earlier wrong moves at this showing of the puzzle;
    an empty `uci` means you gave up. Until it's solved or the tries run out, nothing
    about the solution is revealed and nothing is recorded."""
    p = store.puzzle(pid)
    if p is None:
        raise KeyError(pid)
    tried = [u for u in (tried or []) if u][:MAX_TRIES - 1]
    solved = bool(uci) and (uci == p["solution"] or uci in p.get("also", []))
    moves = tried + ([uci] if uci else [])
    if uci and not solved and len(moves) < MAX_TRIES:
        left = MAX_TRIES - len(moves)
        board = chess.Board(p["fen"])
        move = chess.Move.from_uci(uci)
        san = board.san(move) if move in board.legal_moves else uci
        return {"done": False, "solved": False, "tries_left": left,
                "message": f"Not {san}. {left} {'tries' if left > 1 else 'try'} left."}
    store.add_puzzle_attempt(pid, ",".join(moves), solved, hints)
    # The line to replay on the board: your move if it was an alternative that also works,
    # otherwise the solution and what follows.
    line = [uci] if solved and uci in p.get("also", []) else p["line"]
    board = chess.Board(p["fen"])
    fens = []
    for u in line:
        board.push(chess.Move.from_uci(u))
        fens.append(board.fen())
    return {"done": True, "solved": solved, "solution": p["solution"], "line": line, "fens": fens,
            "line_san": p["line_san"], "themes": p["themes"],
            "explanation": explain(p, solved, uci, len(moves)),
            "also_good": solved and uci in p.get("also", [])}
