"""Fetch a user's games from the Chess.com public API (newest months first)."""

from __future__ import annotations

import httpx

from nemesis.ingest.pgn import iter_games, to_rows
from nemesis.store import Store

ARCHIVES = "https://api.chess.com/pub/player/{user}/games/archives"
HEADERS = {"User-Agent": "nemesis-chess/0.1 (personal opponent-modelling engine)"}


def import_user(store: Store, username: str, max_games: int = 200, timeout: float = 60.0) -> int:
    added = seen = 0
    with httpx.Client(headers=HEADERS, timeout=timeout) as client:
        resp = client.get(ARCHIVES.format(user=username.lower()))
        resp.raise_for_status()
        for month_url in reversed(resp.json().get("archives", [])):
            pgn = client.get(month_url + "/pgn")
            pgn.raise_for_status()
            # Monthly files are oldest-first; walk newest-first so max_games keeps recent play.
            for game in reversed(list(iter_games(pgn.text))):
                rows = to_rows(game, username, source="chesscom")
                if rows is None:
                    continue
                seen += 1
                if store.add_game(*rows):
                    added += 1
                if seen >= max_games:
                    return added
    return added
