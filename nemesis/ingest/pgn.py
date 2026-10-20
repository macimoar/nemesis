"""Parse PGN text into store rows for one modelled player."""

from __future__ import annotations

import hashlib
import io
from typing import Iterator

import chess
import chess.pgn

from nemesis.store import GameRow, MoveRow, Store

RESULT_WHITE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def iter_games(pgn_text: str) -> Iterator[chess.pgn.Game]:
    stream = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            return
        yield game


def game_id(game: chess.pgn.Game, source: str) -> str:
    h = game.headers
    site = h.get("Link") or h.get("Site", "")
    if site.startswith("http"):
        return f"{source}:{site.rstrip('/').rsplit('/', 1)[-1]}"
    digest = hashlib.sha1(str(game).encode()).hexdigest()[:16]
    return f"{source}:{digest}"


def opening_name(game: chess.pgn.Game) -> str | None:
    h = game.headers
    if h.get("Opening") and h["Opening"] != "?":
        return h["Opening"]
    url = h.get("ECOUrl")  # chess.com
    if url:
        return url.rsplit("/", 1)[-1].replace("-", " ")
    return h.get("ECO")


def to_rows(game: chess.pgn.Game, player: str, source: str) -> tuple[GameRow, list[MoveRow]] | None:
    """Convert a parsed game to rows, or None if the player isn't in it / it isn't standard chess."""
    h = game.headers
    player_l = player.lower()
    if h.get("Variant", "Standard").lower() not in ("standard", "from position"):
        return None
    if h.get("White", "").lower() == player_l:
        color = 1
    elif h.get("Black", "").lower() == player_l:
        color = 0
    else:
        return None

    white_score = RESULT_WHITE.get(h.get("Result", "*"))
    result = None if white_score is None else (white_score if color else 1 - white_score)
    gid = game_id(game, source)

    moves: list[MoveRow] = []
    board = game.board()
    for ply, node in enumerate(game.mainline()):
        mover_white = board.turn == chess.WHITE
        moves.append(MoveRow(
            game_id=gid, ply=ply, fen=board.fen(), uci=node.move.uci(),
            is_player=int(mover_white == bool(color)), clock=node.clock(),
        ))
        board.push(node.move)
    if not moves:
        return None

    date = h.get("UTCDate") or h.get("Date") or ""
    time = h.get("UTCTime") or ""
    row = GameRow(
        id=gid, player=player_l, color=color, result=result, source=source,
        opponent=h.get("Black" if color else "White"), opening=opening_name(game),
        time_class=h.get("Event"), played_at=f"{date} {time}".strip() or None, pgn=str(game),
    )
    return row, moves


def import_pgn_text(store: Store, pgn_text: str, player: str, source: str = "pgn") -> int:
    """Store every game of `player` found in `pgn_text`. Returns the number of new games."""
    added = 0
    for game in iter_games(pgn_text):
        rows = to_rows(game, player, source)
        if rows and store.add_game(*rows):
            added += 1
    return added
