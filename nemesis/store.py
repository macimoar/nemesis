"""SQLite persistence: games, the positions a player faced, cached engine evals, trained models."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DB = Path(os.environ.get("NEMESIS_DB", Path.home() / ".nemesis" / "nemesis.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id         TEXT PRIMARY KEY,
    player     TEXT NOT NULL,          -- normalised (lower-case) username we are modelling
    color      INTEGER NOT NULL,       -- 1 = player had white, 0 = black
    result     REAL,                   -- player's score: 1, 0.5, 0, NULL if unknown
    source     TEXT NOT NULL,          -- lichess | chesscom | pgn | nemesis
    opponent   TEXT,
    opening    TEXT,
    time_class TEXT,
    played_at  TEXT,
    pgn        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS games_player ON games(player);

CREATE TABLE IF NOT EXISTS moves (
    game_id   TEXT NOT NULL,
    ply       INTEGER NOT NULL,        -- 0-based half-move index
    fen       TEXT NOT NULL,           -- position before the move
    uci       TEXT NOT NULL,
    is_player INTEGER NOT NULL,        -- 1 if the modelled player made this move
    clock     REAL,                    -- mover's remaining seconds after the move, if known
    PRIMARY KEY (game_id, ply)
);

CREATE TABLE IF NOT EXISTS evals (
    fen   TEXT PRIMARY KEY,
    depth INTEGER NOT NULL,
    lines TEXT NOT NULL                -- JSON {uci: cp from side-to-move's view}
);

CREATE TABLE IF NOT EXISTS models (
    player     TEXT PRIMARY KEY,
    weights    TEXT NOT NULL,
    metrics    TEXT NOT NULL,
    trained_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


@dataclass
class GameRow:
    id: str
    player: str
    color: int
    result: float | None
    source: str
    opponent: str | None
    opening: str | None
    time_class: str | None
    played_at: str | None
    pgn: str


@dataclass
class MoveRow:
    game_id: str
    ply: int
    fen: str
    uci: str
    is_player: int
    clock: float | None


class Store:
    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # -- games -------------------------------------------------------------
    def has_game(self, game_id: str) -> bool:
        with self._lock:
            return self.conn.execute("SELECT 1 FROM games WHERE id=?", (game_id,)).fetchone() is not None

    def add_game(self, game: GameRow, moves: list[MoveRow]) -> bool:
        """Insert a game and its moves. Returns False if it was already stored."""
        with self._lock:
            if self.has_game(game.id):
                return False
            self.conn.execute(
                "INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?,?)",
                (game.id, game.player, game.color, game.result, game.source, game.opponent,
                 game.opening, game.time_class, game.played_at, game.pgn),
            )
            self.conn.executemany(
                "INSERT INTO moves VALUES (?,?,?,?,?,?)",
                [(m.game_id, m.ply, m.fen, m.uci, m.is_player, m.clock) for m in moves],
            )
            self.conn.commit()
            return True

    def games(self, player: str) -> list[GameRow]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM games WHERE player=? ORDER BY played_at, id", (player.lower(),)
            ).fetchall()
        return [GameRow(**dict(r)) for r in rows]

    def moves(self, game_id: str) -> list[MoveRow]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM moves WHERE game_id=? ORDER BY ply", (game_id,)
            ).fetchall()
        return [MoveRow(**dict(r)) for r in rows]

    def players(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT g.player, COUNT(*) AS games, m.trained_at, m.metrics FROM games g "
                "LEFT JOIN models m ON m.player=g.player GROUP BY g.player ORDER BY g.player"
            ).fetchall()
        return [
            {"player": r["player"], "games": r["games"], "trained_at": r["trained_at"],
             "metrics": json.loads(r["metrics"]) if r["metrics"] else None}
            for r in rows
        ]

    # -- evals -------------------------------------------------------------
    def get_eval(self, fen: str, min_depth: int) -> dict[str, int] | None:
        with self._lock:
            row = self.conn.execute("SELECT depth, lines FROM evals WHERE fen=?", (fen,)).fetchone()
        if row is None or row["depth"] < min_depth:
            return None
        return json.loads(row["lines"])

    def put_eval(self, fen: str, depth: int, lines: dict[str, int]) -> None:
        with self._lock:
            row = self.conn.execute("SELECT depth, lines FROM evals WHERE fen=?", (fen,)).fetchone()
            if row is not None and row["depth"] == depth:
                # Same depth: merge so extra root moves analysed later are kept.
                merged = json.loads(row["lines"])
                merged.update(lines)
                lines = merged
            elif row is not None and row["depth"] > depth:
                return
            self.conn.execute(
                "INSERT OR REPLACE INTO evals VALUES (?,?,?)", (fen, depth, json.dumps(lines))
            )
            self.conn.commit()

    # -- models ------------------------------------------------------------
    def save_model(self, player: str, weights: dict, metrics: dict) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO models(player, weights, metrics, trained_at) "
                "VALUES (?,?,?,CURRENT_TIMESTAMP)",
                (player.lower(), json.dumps(weights), json.dumps(metrics)),
            )
            self.conn.commit()

    def load_model(self, player: str) -> tuple[dict, dict] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT weights, metrics FROM models WHERE player=?", (player.lower(),)
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["weights"]), json.loads(row["metrics"])
