"""Describe every legal move in a position as a feature vector.

The opponent model learns, per player, how much each feature attracts or repels
them. Some features describe objective quality (centipawn loss), some describe style
(captures, king attacks, retreats), and some describe tactical blind spots: does
the move hang a piece, ignore a threat, or walk into a fork?
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import chess
import numpy as np

VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}

FEATURES = [
    # objective quality
    "cp_loss", "is_best", "top3", "mistake", "blunder",
    # style
    "capture", "capture_up", "trade", "check", "sacrifice", "pawn_move", "pawn_storm",
    "near_king", "retreat", "advance", "castle", "king_move", "develop", "queen_move",
    # tactical blind spots
    "hangs_piece", "ignores_threat", "allows_fork",
    # repertoire
    "book_freq", "book_deviation",
    # context interactions
    "time_pressure_x_loss", "time_pressure_x_best", "endgame_x_loss", "losing_x_loss",
    "losing_x_sacrifice", "winning_x_trade",
]
INDEX = {name: i for i, name in enumerate(FEATURES)}
N_FEATURES = len(FEATURES)

# Features that describe *how* someone plays (used for explanations and the dashboard).
STYLE_TEXT = {  # phrased to follow "you ..."
    "capture": "grab material when offered",
    "capture_up": "can't resist winning material",
    "trade": "like to trade pieces",
    "check": "play checks reflexively",
    "sacrifice": "love a speculative sacrifice",
    "pawn_storm": "throw pawns at the king",
    "near_king": "go straight for the king",
    "retreat": "back off under pressure",
    "advance": "push forward at every chance",
    "castle": "castle early",
    "king_move": "walk the king around",
    "develop": "develop by the book",
    "queen_move": "bring the queen out early",
    "hangs_piece": "leave pieces hanging",
    "ignores_threat": "miss the opponent's threats",
    "allows_fork": "walk into forks",
    "book_freq": "stick to your repertoire",
    "book_deviation": "leave the repertoire in unfamiliar lines",
    "time_pressure_x_loss": "crack in time trouble",
    "losing_x_sacrifice": "gamble when behind",
    "winning_x_trade": "trade down when ahead",
}


class Repertoire(Protocol):
    def freq(self, key: str, uci: str) -> int: ...
    def total(self, key: str) -> int: ...


def position_key(board: chess.Board) -> str:
    return board.epd()


@dataclass
class Context:
    clock: float | None = None      # mover's seconds remaining before the move
    best_cp: int = 0                # eval of the position for the mover


def _attackers_min(board: chess.Board, color: chess.Color, sq: chess.Square) -> int | None:
    vals = [VALUE[board.piece_type_at(a)] for a in board.attackers(color, sq)]
    return min(vals) if vals else None


def en_prise(board: chess.Board, sq: chess.Square) -> bool:
    """A piece is en prise if attacked and undefended, or attacked by something cheaper."""
    piece = board.piece_at(sq)
    if piece is None or piece.piece_type == chess.KING:
        return False
    attacker = _attackers_min(board, not piece.color, sq)
    if attacker is None:
        return False
    if attacker < VALUE[piece.piece_type]:
        return True
    return not board.is_attacked_by(piece.color, sq)


def hanging_squares(board: chess.Board, color: chess.Color, min_value: int = 3) -> set[chess.Square]:
    return {
        sq for sq in chess.SquareSet(board.occupied_co[color])
        if VALUE[board.piece_type_at(sq)] >= min_value and en_prise(board, sq)
    }


def allows_fork(after: chess.Board, victim: chess.Color) -> bool:
    """Can the side to move in `after` fork two valuable `victim` pieces (or king + piece)?"""
    for reply in after.legal_moves:
        piece = after.piece_type_at(reply.from_square)
        if piece == chess.KING:
            continue
        after.push(reply)
        try:
            sq = reply.to_square
            defenders = after.attackers(victim, sq)
            # The forking piece must survive: nobody can take it, or only the king can and it's protected.
            king_only = defenders == chess.SquareSet(after.kings & after.occupied_co[victim])
            if defenders and not (king_only and after.is_attacked_by(not victim, sq)):
                continue
            fval = VALUE[piece]
            targets = 0
            for t in after.attacks(sq) & chess.SquareSet(after.occupied_co[victim]):
                tv = VALUE[after.piece_type_at(t)]
                if tv > fval or (tv >= 3 and not after.is_attacked_by(victim, t)):
                    targets += 1
            if targets >= 2:
                return True
        finally:
            after.pop()
    return False


def _phase(board: chess.Board) -> str:
    material = sum(VALUE[p.piece_type] for p in board.piece_map().values()
                   if p.piece_type not in (chess.PAWN, chess.KING))
    if material <= 20:
        return "endgame"
    if board.fullmove_number <= 10:
        return "opening"
    return "middlegame"


def position_features(
    board: chess.Board,
    losses: dict[str, int],
    repertoire: Repertoire | None = None,
    ctx: Context | None = None,
    check_forks: bool = True,
) -> tuple[list[chess.Move], np.ndarray]:
    """Return the legal moves and an [n_moves, N_FEATURES] matrix describing them."""
    ctx = ctx or Context()
    me = board.turn
    moves = list(board.legal_moves)
    X = np.zeros((len(moves), N_FEATURES), dtype=np.float64)
    if not moves:
        return moves, X

    ranked = sorted(losses.values())
    third = ranked[min(2, len(ranked) - 1)]
    phase = _phase(board)
    enemy_king = board.king(not me)
    threatened = hanging_squares(board, me)
    time_pressure = ctx.clock is not None and ctx.clock < 30
    losing = ctx.best_cp < -200
    winning = ctx.best_cp > 200
    key = position_key(board) if repertoire else None
    book_total = repertoire.total(key) if repertoire else 0
    forward = 1 if me == chess.WHITE else -1

    for i, move in enumerate(moves):
        f = X[i]
        uci = move.uci()
        loss = losses.get(uci, 0)
        piece = board.piece_type_at(move.from_square)
        captured = board.piece_type_at(move.to_square)
        if board.is_en_passant(move):
            captured = chess.PAWN
        castle = board.is_castling(move)

        f[INDEX["cp_loss"]] = min(loss, 800) / 100.0
        f[INDEX["is_best"]] = loss <= 0
        f[INDEX["top3"]] = loss <= third
        f[INDEX["mistake"]] = loss >= 100
        f[INDEX["blunder"]] = loss >= 300

        if captured:
            f[INDEX["capture"]] = 1
            f[INDEX["capture_up"]] = VALUE[captured] > VALUE[piece]
            f[INDEX["trade"]] = VALUE[captured] == VALUE[piece]
        f[INDEX["pawn_move"]] = piece == chess.PAWN
        to_rank = chess.square_rank(move.to_square)
        if enemy_king is not None:
            dist = chess.square_distance(move.to_square, enemy_king)
            f[INDEX["near_king"]] = dist <= 2 and piece != chess.KING
            if piece == chess.PAWN:
                rel_rank = to_rank if me == chess.WHITE else 7 - to_rank
                king_file = chess.square_file(enemy_king)
                f[INDEX["pawn_storm"]] = rel_rank >= 3 and abs(chess.square_file(move.to_square) - king_file) <= 1
        delta = (to_rank - chess.square_rank(move.from_square)) * forward
        f[INDEX["advance"]] = delta / 7.0
        f[INDEX["retreat"]] = delta < 0 and piece != chess.KING
        f[INDEX["castle"]] = castle
        f[INDEX["king_move"]] = piece == chess.KING and not castle
        back = 0 if me == chess.WHITE else 7
        f[INDEX["develop"]] = (phase == "opening" and piece in (chess.KNIGHT, chess.BISHOP)
                               and chess.square_rank(move.from_square) == back)
        f[INDEX["queen_move"]] = piece == chess.QUEEN

        board.push(move)
        try:
            f[INDEX["check"]] = board.is_check()
            moved_hangs = en_prise(board, move.to_square)
            sac = moved_hangs and VALUE[piece] > (VALUE[captured] if captured else 0)
            f[INDEX["sacrifice"]] = sac
            still = hanging_squares(board, me) - {move.to_square}
            f[INDEX["hangs_piece"]] = bool(still - threatened)
            f[INDEX["ignores_threat"]] = bool(threatened & still) and not captured
            if check_forks and not board.is_check() and loss >= 100:
                # Only worth the cost for moves Stockfish already dislikes.
                f[INDEX["allows_fork"]] = allows_fork(board, me)
        finally:
            board.pop()

        if repertoire and book_total:
            freq = repertoire.freq(key, uci)
            f[INDEX["book_freq"]] = math.log1p(freq)
            f[INDEX["book_deviation"]] = book_total >= 3 and freq == 0

        if time_pressure:
            f[INDEX["time_pressure_x_loss"]] = f[INDEX["cp_loss"]]
            f[INDEX["time_pressure_x_best"]] = f[INDEX["is_best"]]
        if phase == "endgame":
            f[INDEX["endgame_x_loss"]] = f[INDEX["cp_loss"]]
        if losing:
            f[INDEX["losing_x_loss"]] = f[INDEX["cp_loss"]]
            f[INDEX["losing_x_sacrifice"]] = f[INDEX["sacrifice"]]
        if winning:
            f[INDEX["winning_x_trade"]] = f[INDEX["trade"]]

    return moves, X
