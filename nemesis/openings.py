"""Opening names, from the Lichess opening list (CC0, public domain).

Looked up by position, not by move order, so a transposition gets the right name. A
game's opening is the last named position it passed through.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import chess

DATA = Path(__file__).parent / "data" / "openings.json"


@lru_cache(maxsize=1)
def _table() -> dict[str, list[str]]:
    try:
        return json.loads(DATA.read_text())
    except OSError:
        return {}


def name_of(board: chess.Board) -> dict | None:
    """{"eco", "name", "ply"}: the last named position in this game so far, or None."""
    table = _table()
    if not table:
        return None
    b = board.root()
    found = None
    if b.epd() in table:
        found = (table[b.epd()], 0)
    for ply, move in enumerate(board.move_stack, 1):
        b.push(move)
        hit = table.get(b.epd())
        if hit:
            found = (hit, ply)
    if found is None:
        return None
    (eco, name), ply = found
    return {"eco": eco, "name": name, "ply": ply}


def family(name: str) -> str:
    """"Italian Game: Giuoco Piano" -> "Italian Game"."""
    return name.split(":")[0].strip()
