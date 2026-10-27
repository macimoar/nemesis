"""Maia-2: how likely a human of a given rating is to play each move.

Maia-2 (https://github.com/CSSLab/maia2, MIT licence) is a neural network trained on
millions of Lichess games. Given a position and both players' ratings, it predicts the
move a human would actually play. Nemesis uses it as the starting point of each
player's move model: one more feature, "how likely is this move for someone at your
level", that your personal weights then adjust.

The models are converted to ONNX (scripts/convert_maia2.py) and run with onnxruntime,
so no PyTorch is needed. They are downloaded on first use into ~/.nemesis/maia2/.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import chess
import numpy as np

REPO = "macimoar/nemesis"
TAG = "maia2-onnx-v2"
DIR = Path(os.environ.get("NEMESIS_HOME", Path.home() / ".nemesis")) / "maia2"
HALF = True          # fp16 weights: half the download (43 MB), the same predictions
# Maia-2 groups ratings in 100-point buckets from 1100 to 2000 (Lichess ratings).
BUCKET_START, BUCKET_END, BUCKET_STEP = 1100, 2000, 100
# Maia-2 was trained without each game's first 10 plies (its first_n_moves setting), so its
# opening predictions are rough: the move model gives them their own, smaller weight.
FIRST_PLY = 10
# Chess.com ratings run lower than Lichess ones; this is added before bucketing.
# Fitted on real games with scripts/evaluate_maia.py (it matters little below 1100,
# where Maia-2 puts everyone in one bucket).
CHESSCOM_TO_LICHESS = 100

Progress = Callable[[str, float], None]


def kind_for(category: str | None) -> str:
    """Which model fits a time-control category: blitz for fast games, rapid otherwise."""
    return "blitz" if category in ("bullet", "blitz") else "rapid"


def bucket(elo: float | None) -> int:
    """Maia-2's rating bucket: 0 below 1100, then one per 100 points, 10 from 2000."""
    if elo is None:
        elo = 1500
    if elo < BUCKET_START:
        return 0
    if elo >= BUCKET_END:
        return (BUCKET_END - BUCKET_START) // BUCKET_STEP + 1
    return int((elo - BUCKET_START) // BUCKET_STEP) + 1


def encode(board: chess.Board) -> tuple[np.ndarray, chess.Board]:
    """Maia-2's 18x8x8 input, always from the side to move (Black's positions are mirrored).
    Returns the planes and the board as Maia-2 sees it."""
    b = board if board.turn == chess.WHITE else board.mirror()
    x = np.zeros((18, 8, 8), dtype=np.float32)
    for i, piece_type in enumerate(chess.PIECE_TYPES):
        for color, offset in ((chess.WHITE, 0), (chess.BLACK, 6)):
            for sq in b.pieces(piece_type, color):
                x[i + offset, sq // 8, sq % 8] = 1.0
    if b.turn == chess.WHITE:
        x[12] = 1.0
    rights = (b.has_kingside_castling_rights(chess.WHITE), b.has_queenside_castling_rights(chess.WHITE),
              b.has_kingside_castling_rights(chess.BLACK), b.has_queenside_castling_rights(chess.BLACK))
    for i, has in enumerate(rights):
        if has:
            x[13 + i] = 1.0
    # Maia-2 trained on positions saved as FEN, which only keeps an en-passant square when
    # the capture is actually possible (a board fresh from a double pawn push always has one).
    if b.ep_square is not None and b.has_legal_en_passant():
        x[17, b.ep_square // 8, b.ep_square % 8] = 1.0
    return x, b


def _mirror(uci: str) -> str:
    flip = lambda sq: sq[0] + str(9 - int(sq[1]))   # noqa: E731
    return flip(uci[:2]) + flip(uci[2:4]) + uci[4:]


class Maia:
    """One Maia-2 model (rapid or blitz). `policy` gives P(move) over the legal moves."""

    def __init__(self, model_path: str | Path, moves_path: str | Path, cache_size: int = 20000):
        import onnxruntime as ort   # telemetry is already off: see nemesis/__init__.py
        if hasattr(ort, "disable_telemetry_events"):
            ort.disable_telemetry_events()   # and again, in case it was imported elsewhere first
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 2) // 2))
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.moves = json.loads(Path(moves_path).read_text())
        self.index = {m: i for i, m in enumerate(self.moves)}
        self._cache: OrderedDict = OrderedDict()
        self.cache_size = cache_size

    def policy(self, board: chess.Board, elo_self: float | None, elo_oppo: float | None) -> dict[str, float]:
        """{uci: probability} for every legal move, for a player rated `elo_self` (Lichess
        scale) facing `elo_oppo`."""
        return self.policies([board], elo_self, elo_oppo)[0]

    def policies(self, boards: list[chess.Board], elo_self: float | None,
                 elo_oppo: float | None) -> list[dict[str, float]]:
        es, eo = bucket(elo_self), bucket(elo_oppo)
        out: list[dict[str, float] | None] = []
        todo = []
        for i, board in enumerate(boards):
            key = (board.fen(), es, eo)
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
            out.append(hit)
            if hit is None:
                todo.append(i)
        if todo:
            encoded = [encode(boards[i]) for i in todo]
            logits, _ = self.session.run(None, {
                "boards": np.stack([x for x, _ in encoded]),
                "elo_self": np.full(len(todo), es, dtype=np.int64),
                "elo_oppo": np.full(len(todo), eo, dtype=np.int64),
            })
            for row, i, (_, seen) in zip(logits, todo, encoded):
                legal = [m.uci() for m in seen.legal_moves]
                if not legal:
                    out[i] = {}
                    continue
                z = row[[self.index[u] for u in legal]].astype(np.float64)
                p = np.exp(z - z.max())
                p /= p.sum()
                flip = boards[i].turn == chess.BLACK
                probs = {(_mirror(u) if flip else u): float(q) for u, q in zip(legal, p)}
                out[i] = probs
                self._cache[(boards[i].fen(), es, eo)] = probs
                if len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        return out  # type: ignore[return-value]


def log_prob(probs: dict[str, float] | None, uci: str) -> float:
    """The feature the move model uses: log P(move), floored so one rare move can't dominate."""
    if not probs:
        return 0.0
    return max(math.log(max(probs.get(uci, 0.0), 1e-9)), -9.0)


# ---------------------------------------------------------------------------
# Ratings: Maia-2 thinks in Lichess ratings
# ---------------------------------------------------------------------------
def _headers(pgn: str) -> dict:
    import io
    import chess.pgn
    return dict(chess.pgn.read_headers(io.StringIO(pgn)) or {})


def rating_offset(store, player: str) -> int:
    """What to add to this player's ratings to put them on Lichess's scale: their
    Chess.com ratings run lower. Decided by where most of their imported games are from."""
    cache = store.__dict__.setdefault("_maia_offset", {})
    games = store.games(player)
    key = (player.lower(), len(games))
    if key not in cache:
        chesscom = lichess = 0
        for g in games:
            site = g.source if g.source in ("chesscom", "lichess") else _headers(g.pgn).get("Site", "").lower()
            chesscom += "chess" in site and "com" in site
            lichess += "lichess" in site
        cache[key] = CHESSCOM_TO_LICHESS if chesscom > lichess else 0
    return cache[key]


def game_kind(game) -> str:
    """rapid or blitz model for a stored game, from its PGN TimeControl header."""
    from nemesis import timecontrol
    return kind_for(timecontrol.pgn_category(_headers(game.pgn).get("TimeControl")))


def game_elos(game) -> tuple[float | None, float | None]:
    """(player's rating, opponent's rating) from a stored game's PGN headers, if recorded."""
    h = _headers(game.pgn)

    def num(tag: str) -> float | None:
        try:
            return float(h.get(tag, ""))
        except ValueError:
            return None
    white, black = num("WhiteElo"), num("BlackElo")
    return (white, black) if game.color == chess.WHITE else (black, white)


# ---------------------------------------------------------------------------
# Download on first use
# ---------------------------------------------------------------------------
def _files(kind: str, half: bool = HALF) -> tuple[str, str]:
    return f"maia2-{kind}{'-fp16' if half else ''}.onnx", "moves.json"


def installed(kind: str, dest: Path = DIR) -> Maia | None:
    model, moves = _files(kind)
    if (dest / model).exists() and (dest / moves).exists():
        return Maia(dest / model, dest / moves)
    return None


def download(kind: str, dest: Path = DIR, progress: Progress | None = None) -> Path:
    """Fetch one model (and the move list) from the GitHub release, checking checksums."""
    import httpx
    dest.mkdir(parents=True, exist_ok=True)
    base = f"https://github.com/{REPO}/releases/download/{TAG}/"
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        sums = {}
        resp = client.get(base + "SHA256SUMS")
        resp.raise_for_status()
        for line in resp.text.splitlines():
            digest, _, name = line.partition("  ")
            sums[name.strip()] = digest
        for name in (*_files(kind), "LICENSE-maia2.txt"):
            target = dest / name
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == sums.get(name):
                continue
            h, tmp = hashlib.sha256(), target.with_suffix(".part")
            with client.stream("GET", base + name) as r, open(tmp, "wb") as f:
                r.raise_for_status()
                total, done = int(r.headers.get("content-length") or 0), 0
                for chunk in r.iter_bytes(1 << 16):
                    f.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if progress and total:
                        progress("Downloading the Maia human-move model", done / total)
            if sums.get(name) and h.hexdigest() != sums[name]:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"{name}: checksum mismatch, download corrupted")
            tmp.replace(target)
    return dest


_loaded: dict[str, Maia | None] = {}
_lock = __import__("threading").Lock()   # one download at a time


def get(kind: str = "rapid", allow_download: bool = True, progress: Progress | None = None) -> Maia | None:
    """The Maia model for `kind`, downloading it the first time. None if it can't be had
    (offline, or onnxruntime missing): Nemesis then works as before, without it."""
    if os.environ.get("NEMESIS_NO_MAIA"):
        return None
    if _loaded.get(kind) is not None:
        return _loaded[kind]
    if not allow_download:
        try:
            model = installed(kind)
        except Exception:
            model = None
        if model is not None:
            _loaded[kind] = model
        return model
    with _lock:
        if _loaded.get(kind) is not None:
            return _loaded[kind]
        try:
            model = installed(kind)
            if model is None:
                download(kind, progress=progress)
                model = installed(kind)
        except Exception:  # never let the human-move model stop Nemesis from working
            model = None
        _loaded[kind] = model
        return model
