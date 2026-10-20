"""Fetch a user's games from the Lichess API."""

from __future__ import annotations

import httpx

from nemesis.ingest.pgn import import_pgn_text
from nemesis.store import Store

API = "https://lichess.org/api/games/user/{user}"


def fetch_pgn(username: str, max_games: int = 200, timeout: float = 120.0) -> str:
    params = {"max": max_games, "clocks": "true", "opening": "true", "evals": "false"}
    headers = {"Accept": "application/x-chess-pgn"}
    resp = httpx.get(API.format(user=username), params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def import_user(store: Store, username: str, max_games: int = 200) -> int:
    return import_pgn_text(store, fetch_pgn(username, max_games), username, source="lichess")
