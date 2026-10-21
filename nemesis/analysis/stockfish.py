"""Stockfish wrapper with a persistent eval cache.

`lines(board, multipv, extra)` returns {uci: centipawns} from the side to move's
point of view for the top `multipv` moves plus any `extra` moves requested.
Moves outside that set are unknown; callers impute them (see `impute_losses`).
"""

from __future__ import annotations

import glob
import math
import os
import shutil
import sys
import threading
from typing import Iterable, Protocol

import chess
import chess.engine

from nemesis.store import Store

MATE_CP = 2000  # mates are clamped to this many centipawns


def _engine_in(directory: str) -> str | None:
    """A Stockfish binary inside `directory` (Windows downloads unzip to a folder of that name)."""
    for pattern in ("stockfish*.exe", "stockfish*", "*/stockfish*.exe", "*/stockfish*"):
        for cand in sorted(glob.glob(os.path.join(directory, pattern))):
            if os.path.isfile(cand) and not cand.lower().endswith((".txt", ".md", ".zip", ".tar")):
                return cand
    return None


def normalise_path(raw: str) -> str | None:
    """Clean up a user-supplied Stockfish path: strip quotes, accept a folder. None if no file is there."""
    path = os.path.expanduser(raw.strip().strip('"').strip("'").strip())
    if os.path.isdir(path):
        return _engine_in(path)
    if os.path.isfile(path):
        return path
    if os.path.isfile(path + ".exe"):
        return path + ".exe"
    return None


def bundled_stockfish() -> str | None:
    """The copy shipped inside the packaged app (PyInstaller unpacks it to sys._MEIPASS)."""
    base = getattr(sys, "_MEIPASS", None)
    path = _engine_in(os.path.join(base, "stockfish")) if base else None
    if path and not os.access(path, os.X_OK):  # unpacked without the executable bit
        os.chmod(path, os.stat(path).st_mode | 0o111)
    return path


def find_stockfish() -> str | None:
    """Look in order: STOCKFISH_PATH, the app bundle, PATH and usual places, Nemesis's own download."""
    env = os.environ.get("STOCKFISH_PATH")
    if env:
        # A bad STOCKFISH_PATH is reported by stockfish_path_problem(), not silently replaced.
        return normalise_path(env)
    if path := bundled_stockfish():
        return path
    for cand in ("stockfish", "stockfish.exe"):
        if path := shutil.which(cand):
            return path
    # Windows builds are named like stockfish-windows-x86-64-avx2.exe, which `which` won't match.
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if directory:
            for cand in sorted(glob.glob(os.path.join(directory, "stockfish*.exe"))):
                return cand
    for cand in ("/usr/games/stockfish", "/usr/local/bin/stockfish", "/opt/homebrew/bin/stockfish"):
        if os.path.isfile(cand):
            return cand
    from nemesis.analysis.install import downloaded
    return downloaded()


def stockfish_path_problem() -> str:
    env = os.environ.get("STOCKFISH_PATH")
    if env:
        return (f"STOCKFISH_PATH={env!r} is not a Stockfish executable. "
                "Set it to the .exe file itself, without quotes.")
    return ("Stockfish not found. Run `nemesis install-stockfish` to download it, "
            "or set STOCKFISH_PATH to the Stockfish executable")


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
    def lines(self, board: chess.Board, multipv: int = 6, extra: Iterable[str] = (),
              depth: int | None = None) -> dict[str, int]: ...


class StockfishAnalyzer:
    def __init__(self, store: Store | None = None, path: str | None = None, depth: int = 10,
                 threads: int = 1, hash_mb: int = 64):
        path = normalise_path(path) if path else find_stockfish()
        if not path:
            raise RuntimeError(stockfish_path_problem())
        self.depth = depth
        self.store = store
        self._lock = threading.Lock()
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
            self.engine.configure({"Threads": threads, "Hash": hash_mb})
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError, OSError) as exc:
            raise RuntimeError(
                f"Stockfish at {path} failed to start ({exc}). If it crashes as soon as it starts, "
                "your CPU may not support this build: try the 'compatible' / non-AVX2 download."
            ) from exc

    def close(self) -> None:
        try:
            self.engine.quit()
        except chess.engine.EngineTerminatedError:
            pass

    def __enter__(self) -> "StockfishAnalyzer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _run(self, board: chess.Board, multipv: int, root_moves: list[chess.Move] | None,
             depth: int) -> dict[str, int]:
        with self._lock:
            infos = self.engine.analyse(
                board, chess.engine.Limit(depth=depth), multipv=multipv, root_moves=root_moves,
            )
        out: dict[str, int] = {}
        for info in infos:
            if "pv" in info and info["pv"]:
                out[info["pv"][0].uci()] = score_cp(info["score"], board.turn)
        return out

    def lines(self, board: chess.Board, multipv: int = 6, extra: Iterable[str] = (),
              depth: int | None = None) -> dict[str, int]:
        """`depth` overrides the default for this call (shallower when short on time).
        A deeper cached analysis always satisfies a shallower request."""
        if board.is_game_over():
            return {}
        depth = depth or self.depth
        key = board.fen()
        legal = {m.uci() for m in board.legal_moves}
        multipv = min(multipv, len(legal))
        cached = self.store.get_eval(key, depth) if self.store else None
        # "_mpv" records how many unrestricted lines are in the entry, so that extra moves
        # analysed on their own never pass for a full MultiPV search.
        if cached is None or cached.get("_mpv", 0) < multipv:
            cached = {**(cached or {}), **self._run(board, multipv, None, depth), "_mpv": multipv}
            if self.store:
                self.store.put_eval(key, depth, cached)
        missing = [u for u in extra if u in legal and u not in cached]
        if missing:
            more = self._run(board, len(missing), [chess.Move.from_uci(u) for u in missing], depth)
            cached = {**cached, **more}
            if self.store:
                self.store.put_eval(key, depth, cached)
        return {u: cp for u, cp in cached.items() if u != "_mpv"}
