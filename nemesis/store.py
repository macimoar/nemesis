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

CREATE TABLE IF NOT EXISTS ratings (
    entity TEXT PRIMARY KEY,           -- player:<name> | nemesis:<tier>
    rating REAL NOT NULL,
    games  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS rating_history (
    game_id TEXT NOT NULL,
    entity  TEXT NOT NULL,
    before  REAL NOT NULL,
    after   REAL NOT NULL,
    PRIMARY KEY (game_id, entity)
);

CREATE TABLE IF NOT EXISTS annotations (  -- what Nemesis was thinking, saved at play time
    game_id TEXT PRIMARY KEY,
    data    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    game_id TEXT PRIMARY KEY,
    depth   INTEGER NOT NULL,
    data    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_history (  -- how well the model predicted you, each time it was trained
    player     TEXT NOT NULL,
    trained_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    games      INTEGER,
    top1       REAL,
    top3       REAL,
    baseline   REAL                     -- the generic model's top-1 on the same held-out moves
);

CREATE TABLE IF NOT EXISTS samples (  -- cached training features per game (see move_model)
    game_id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    depth   INTEGER NOT NULL,
    data    BLOB NOT NULL,
    meta    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS puzzles (  -- positions from your games where you went wrong
    id       TEXT PRIMARY KEY,            -- game_id#ply
    player   TEXT NOT NULL,
    game_id  TEXT NOT NULL,
    ply      INTEGER NOT NULL,
    fen      TEXT NOT NULL,
    solution TEXT NOT NULL,               -- best move (uci)
    data     TEXT NOT NULL                -- JSON: line, themes, difficulty, your move, loss, ...
);
CREATE INDEX IF NOT EXISTS puzzles_player ON puzzles(player);

CREATE TABLE IF NOT EXISTS puzzle_attempts (
    puzzle_id TEXT NOT NULL,
    at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    move      TEXT NOT NULL,
    solved    INTEGER NOT NULL,
    hinted    INTEGER NOT NULL DEFAULT 0
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
        self._migrate()

    # -- schema versions ---------------------------------------------------
    def _migrate(self) -> None:
        """Bring an older database up to date. A file database is backed up first."""
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        pending = [(v, fn) for v, fn in MIGRATIONS if v > version]
        if not pending:
            return
        has_data = self.conn.execute("SELECT 1 FROM games LIMIT 1").fetchone() is not None
        if has_data and self.path != ":memory:":
            backup = Path(f"{self.path}.backup-v{version}")
            if not backup.exists():
                dest = sqlite3.connect(backup)
                self.conn.backup(dest)
                dest.close()
        for v, fn in pending:
            with self.conn:
                fn(self.conn)
                self.conn.execute(f"PRAGMA user_version = {v}")

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

    def game(self, game_id: str) -> GameRow | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM games WHERE id=?", (game_id,)).fetchone()
        return GameRow(**dict(row)) if row else None

    # -- ratings -----------------------------------------------------------
    def get_rating(self, entity: str) -> tuple[float, int] | None:
        with self._lock:
            row = self.conn.execute("SELECT rating, games FROM ratings WHERE entity=?", (entity,)).fetchone()
        return (row["rating"], row["games"]) if row else None

    def set_rating(self, entity: str, rating: float, games: int) -> None:
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO ratings VALUES (?,?,?)", (entity, rating, games))
            self.conn.commit()

    def rating_change(self, game_id: str) -> dict[str, tuple[float, float]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT entity, before, after FROM rating_history WHERE game_id=?", (game_id,)).fetchall()
        return {r["entity"]: (r["before"], r["after"]) for r in rows}

    def apply_rating_changes(self, game_id: str, changes: dict[str, tuple[float, float, int]]) -> bool:
        """Atomically record {entity: (before, after, games_after)} for a game. False if already rated."""
        with self._lock:
            if self.conn.execute("SELECT 1 FROM rating_history WHERE game_id=?", (game_id,)).fetchone():
                return False
            for entity, (before, after, games) in changes.items():
                self.conn.execute("INSERT INTO rating_history VALUES (?,?,?,?)", (game_id, entity, before, after))
                self.conn.execute("INSERT OR REPLACE INTO ratings VALUES (?,?,?)", (entity, after, games))
            self.conn.commit()
            return True

    def rating_history(self, entity: str, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT h.game_id, h.before, h.after, g.played_at FROM rating_history h "
                "LEFT JOIN games g ON g.id=h.game_id WHERE h.entity=? ORDER BY g.played_at DESC, h.rowid DESC "
                "LIMIT ?", (entity, limit)).fetchall()
        return [dict(r) for r in rows]

    # -- annotations and reviews -------------------------------------------
    def put_json(self, table: str, game_id: str, data, depth: int | None = None) -> None:
        assert table in ("annotations", "reviews")
        with self._lock:
            if table == "reviews":
                self.conn.execute("INSERT OR REPLACE INTO reviews VALUES (?,?,?)", (game_id, depth or 0, json.dumps(data)))
            else:
                self.conn.execute("INSERT OR REPLACE INTO annotations VALUES (?,?)", (game_id, json.dumps(data)))
            self.conn.commit()

    def get_json(self, table: str, game_id: str, min_depth: int = 0):
        assert table in ("annotations", "reviews")
        with self._lock:
            row = self.conn.execute(f"SELECT * FROM {table} WHERE game_id=?", (game_id,)).fetchone()
        if row is None or (table == "reviews" and row["depth"] < min_depth):
            return None
        return json.loads(row["data"])

    def delete_review(self, game_id: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM reviews WHERE game_id=?", (game_id,))
            self.conn.commit()

    # -- cached training samples -------------------------------------------
    def get_samples(self, game_id: str, version: int, depth: int) -> tuple[bytes, str] | None:
        with self._lock:
            row = self.conn.execute("SELECT version, depth, data, meta FROM samples WHERE game_id=?",
                                    (game_id,)).fetchone()
        if row is None or row["version"] != version or row["depth"] < depth:
            return None
        return row["data"], row["meta"]

    def put_samples(self, game_id: str, version: int, depth: int, data: bytes, meta: str) -> None:
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO samples VALUES (?,?,?,?,?)",
                              (game_id, version, depth, data, meta))
            self.conn.commit()

    # -- puzzles -----------------------------------------------------------
    def put_puzzle(self, pid: str, player: str, game_id: str, ply: int, fen: str, solution: str,
                   data: dict) -> None:
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO puzzles VALUES (?,?,?,?,?,?,?)",
                              (pid, player.lower(), game_id, ply, fen, solution, json.dumps(data)))
            self.conn.commit()

    def puzzle_ids(self, player: str) -> set[str]:
        with self._lock:
            return {r[0] for r in self.conn.execute("SELECT id FROM puzzles WHERE player=?", (player.lower(),))}

    def puzzles(self, player: str) -> list[dict]:
        """All of a player's puzzles, each with its attempt history (oldest first)."""
        with self._lock:
            rows = self.conn.execute("SELECT * FROM puzzles WHERE player=?", (player.lower(),)).fetchall()
            attempts = self.conn.execute(
                "SELECT a.puzzle_id, a.solved, a.hinted FROM puzzle_attempts a JOIN puzzles p "
                "ON p.id=a.puzzle_id WHERE p.player=? ORDER BY a.rowid", (player.lower(),)).fetchall()
        tries: dict[str, list[dict]] = {}
        for a in attempts:
            tries.setdefault(a["puzzle_id"], []).append({"solved": bool(a["solved"]), "hinted": bool(a["hinted"])})
        return [{**{k: r[k] for k in ("id", "game_id", "ply", "fen", "solution")}, **json.loads(r["data"]),
                 "attempts": tries.get(r["id"], [])} for r in rows]

    def puzzle(self, pid: str) -> dict | None:
        with self._lock:
            r = self.conn.execute("SELECT * FROM puzzles WHERE id=?", (pid,)).fetchone()
        return None if r is None else {**{k: r[k] for k in ("id", "player", "game_id", "ply", "fen", "solution")},
                                       **json.loads(r["data"])}

    def add_puzzle_attempt(self, pid: str, move: str, solved: bool, hinted: bool) -> None:
        with self._lock:
            self.conn.execute("INSERT INTO puzzle_attempts(puzzle_id, move, solved, hinted) VALUES (?,?,?,?)",
                              (pid, move, int(solved), int(hinted)))
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

    def add_model_history(self, player: str, games: int, top1: float, top3: float, baseline: float) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO model_history(player, games, top1, top3, baseline) VALUES (?,?,?,?,?)",
                (player.lower(), games, top1, top3, baseline))
            self.conn.commit()

    def model_history(self, player: str) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT trained_at, games, top1, top3, baseline FROM model_history "
                "WHERE player=? ORDER BY rowid", (player.lower(),)).fetchall()
        return [dict(r) for r in rows]

    def load_model(self, player: str) -> tuple[dict, dict] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT weights, metrics FROM models WHERE player=?", (player.lower(),)
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["weights"]), json.loads(row["metrics"])


# ---------------------------------------------------------------------------
# Migrations: (version, function). Never edit a released one; add a new one.
# ---------------------------------------------------------------------------

def _m1_ratings_per_time_control(conn: sqlite3.Connection) -> None:
    """v0.2: ratings are kept per time control. Everything before was untimed."""
    for table in ("ratings", "rating_history"):
        conn.execute(f"UPDATE {table} SET entity = entity || ':untimed' WHERE "
                     "(entity LIKE 'player:%' AND length(entity) - length(replace(entity, ':', '')) = 1) OR "
                     "(entity LIKE 'nemesis:%' AND length(entity) - length(replace(entity, ':', '')) = 2)")


MIGRATIONS = [
    (1, _m1_ratings_per_time_control),
]
