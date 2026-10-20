"""Stockfish wrapper with a persistent eval cache.

`lines(board, multipv, extra)` returns {uci: centipawns} from the side to move's
point of view for the top `multipv` moves plus any `extra` moves requested.
Moves outside that set are unknown; callers impute them (see `impute_losses`).
"""

from __future__ import annotations

import math
import os
import shutil
import threading
from typing import Iterable, Protocol

import chess
import chess.engine

from nemesis.store import Store

MATE_CP = 2000  # mates are clamped to this many centipawns


def find_stockfish() -> str | None:
    env = os.environ.get("STOCKFISH_PATH")
    if env:
        return env
    for cand in ("stockfish", "/usr/games/stockfish", "/usr/local/bin/stockfish", "/opt/homebrew/bin/stockfish"):
        path = shutil.which(cand) or (cand if os.path.exists(cand) else None)
        if path:
            return path
    return None


def win_prob(cp: float) -> float:
    """Expected score for the side with `cp` (Lichess's logistic fit)."""
    return 1.0 / (1.0 + math.exp(-0.00368208 * cp))


def score_cp(score: chess.engine.PovScore, turn: chess.Color) -> int:
    cp = score.pov(turn).score(mate_score=MATE_CP + 100)
    return max(-MATE_CP, min(MATE_CP, cp))


def terminal_cp(board: chess.Board) -> int | None:
    """Eval from the side to move if the game is over, else None."""
    if board.is_checkmate():
        return -MATE_CP
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0
    return None


def impute_losses(board: chess.Board, lines: dict[str, int]) -> dict[str, int]:
    """Centipawn loss for every legal move. Unanalysed moves are assumed worse than the worst analysed one."""
    if not lines:
        return {m.uci(): 0 for m in board.legal_moves}
    best = max(lines.values())
    floor_loss = best - min(lines.values())
    unknown = max(floor_loss + 60, 120)
    return {m.uci(): best - lines[m.uci()] if m.uci() in lines else unknown for m in board.legal_moves}


class Analyzer(Protocol):
    def lines(self, board: chess.Board, multipv: int = 6, extra: Iterable[str] = ()) -> dict[str, int]: ...


class StockfishAnalyzer:
    def __init__(self, store: Store | None = None, path: str | None = None, depth: int = 10,
                 threads: int = 1, hash_mb: int = 64):
        path = path or find_stockfish()
        if not path:
            raise RuntimeError("Stockfish not found: install it or set STOCKFISH_PATH")
        self.depth = depth
        self.store = store
        self._lock = threading.Lock()
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"Threads": threads, "Hash": hash_mb})

    def close(self) -> None:
        try:
            self.engine.quit()
        except chess.engine.EngineTerminatedError:
            pass

    def __enter__(self) -> "StockfishAnalyzer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _run(self, board: chess.Board, multipv: int, root_moves: list[chess.Move] | None) -> dict[str, int]:
        with self._lock:
            infos = self.engine.analyse(
                board, chess.engine.Limit(depth=self.depth), multipv=multipv, root_moves=root_moves,
            )
        out: dict[str, int] = {}
        for info in infos:
            if "pv" in info and info["pv"]:
                out[info["pv"][0].uci()] = score_cp(info["score"], board.turn)
        return out

    def lines(self, board: chess.Board, multipv: int = 6, extra: Iterable[str] = ()) -> dict[str, int]:
        if board.is_game_over():
            return {}
        key = board.fen()
        legal = {m.uci() for m in board.legal_moves}
        multipv = min(multipv, len(legal))
        cached = self.store.get_eval(key, self.depth) if self.store else None
        # "_mpv" records how many unrestricted lines are in the entry, so that extra moves
        # analysed on their own never pass for a full MultiPV search.
        if cached is None or cached.get("_mpv", 0) < multipv:
            cached = {**(cached or {}), **self._run(board, multipv, None), "_mpv": multipv}
            if self.store:
                self.store.put_eval(key, self.depth, cached)
        missing = [u for u in extra if u in legal and u not in cached]
        if missing:
            more = self._run(board, len(missing), [chess.Move.from_uci(u) for u in missing])
            cached = {**cached, **more}
            if self.store:
                self.store.put_eval(key, self.depth, cached)
        return {u: cp for u, cp in cached.items() if u != "_mpv"}
