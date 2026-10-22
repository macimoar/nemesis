"""Describe every legal move in a position as a feature vector.

The opponent model learns, per player, how much each feature attracts or repels
them. Some features describe objective quality (centipawn loss), some describe style
(captures, king attacks, retreats), and some describe tactical blind spots: does
the move hang a piece, ignore a threat, or walk into a fork, pin, skewer, discovered
attack or back-rank mate?
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
    "allows_pin", "allows_skewer", "allows_discovered", "allows_back_rank",
    # repertoire
    "book_freq", "book_deviation",
    # how likely a human at this rating is to play the move (Maia-2's log-probability)
    "maia",
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
    "allows_pin": "walk into pins",
    "allows_skewer": "get skewered",
    "allows_discovered": "miss discovered attacks",
    "allows_back_rank": "leave your back rank weak",
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
    maia: dict | None = None        # Maia-2's {uci: probability} for this player, if available


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


# ---------------------------------------------------------------------------
# Tactics: what the move just played sets up against the side now to move
# ---------------------------------------------------------------------------
TACTICS = ("fork", "pin", "skewer", "discovered", "back_rank")
SLIDERS = {chess.BISHOP: ((1, 1), (1, -1), (-1, 1), (-1, -1)),
           chess.ROOK: ((1, 0), (-1, 0), (0, 1), (0, -1))}
SLIDERS[chess.QUEEN] = SLIDERS[chess.BISHOP] + SLIDERS[chess.ROOK]


def _survives(board: chess.Board, sq: chess.Square, attacker: chess.Color) -> bool:
    """Can the piece on `sq` stay there: nobody can take it, or only at a cost (the king
    into protection, or a piece at least as valuable into a recapture)?"""
    cheapest = _attackers_min(board, not attacker, sq)
    if cheapest is None:
        return True
    protected = board.is_attacked_by(attacker, sq)
    if cheapest == VALUE[chess.KING]:
        return protected
    return protected and cheapest >= VALUE[board.piece_type_at(sq)]


def _wins(board: chess.Board, target: chess.Square, by_value: int) -> bool:
    """Is the piece on `target` worth taking: more valuable than the attacker, or undefended?"""
    piece = board.piece_type_at(target)
    if piece == chess.KING:
        return True
    return VALUE[piece] > by_value or (VALUE[piece] >= 3 and not board.is_attacked_by(board.color_at(target), target))


def _behind(board: chess.Board, sq: chess.Square, target: chess.Square) -> chess.Square | None:
    """The first occupied square beyond `target` on the line from `sq` through it."""
    df = chess.square_file(target) - chess.square_file(sq)
    dr = chess.square_rank(target) - chess.square_rank(sq)
    step = ((df > 0) - (df < 0), (dr > 0) - (dr < 0))
    f, r = chess.square_file(target), chess.square_rank(target)
    while True:
        f, r = f + step[0], r + step[1]
        if not (0 <= f < 8 and 0 <= r < 8):
            return None
        s2 = chess.square(f, r)
        if board.piece_at(s2):
            return s2


def created_tactics(board: chess.Board, move: chess.Move) -> set[str]:
    """Tactics the move just played (`board` is the position after it) sets up against the
    side to move. Only ones that win something count: a pin of a pawn isn't a tactic."""
    attacker, victim = not board.turn, board.turn
    sq = move.to_square
    piece = board.piece_type_at(sq)
    if piece is None:
        return set()
    found = set()
    home = 0 if victim == chess.WHITE else 7
    if piece in (chess.ROOK, chess.QUEEN) and chess.square_rank(sq) == home and board.is_check():
        king = board.king(victim)
        if king is not None and chess.square_rank(king) == home and board.is_checkmate():
            found.add("back_rank")
            return found
    mine = chess.SquareSet(board.occupied_co[victim])
    hit = board.attacks(sq) & mine
    safe = None
    # Fork: one piece, two worthwhile targets.
    if piece != chess.KING and len(hit) >= 2:
        safe = _survives(board, sq, attacker)
        if safe and sum(_wins(board, t, VALUE[piece]) for t in hit) >= 2:
            found.add("fork")
    # Pin and skewer: two of your pieces on one line, the moved piece looking at the first.
    if piece in SLIDERS and hit:
        for front_sq in hit:
            back_sq = _behind(board, sq, front_sq)
            if back_sq is None or board.color_at(back_sq) != victim:
                continue
            front, back = board.piece_type_at(front_sq), board.piece_type_at(back_sq)
            pin = front != chess.KING and VALUE[front] >= 3 and VALUE[back] > VALUE[front] \
                and _wins(board, front_sq, VALUE[piece])
            skewer = not pin and VALUE[front] > VALUE[back] >= 3 and _wins(board, back_sq, VALUE[piece])
            if pin or skewer:
                safe = _survives(board, sq, attacker) if safe is None else safe
                if safe:
                    found.add("pin" if pin else "skewer")
    # Discovered attack: moving away uncovers another piece's attack, and the moved piece
    # threatens something too, so you can't answer both.
    revealed = []
    frm, own_bb = move.from_square, board.occupied_co[attacker] & ~chess.BB_SQUARES[sq]
    # Only pieces that share a line with the square just vacated can have been uncovered.
    lined = ((board.rooks | board.queens) & own_bb & (chess.BB_RANK_MASKS[frm] | chess.BB_FILE_MASKS[frm])) \
        | ((board.bishops | board.queens) & own_bb & chess.BB_DIAG_MASKS[frm])
    for slider in chess.scan_forward(lined):
        s_value = VALUE[board.piece_type_at(slider)]
        for t in board.attacks(slider) & mine:
            if chess.BB_SQUARES[frm] & chess.between(slider, t) and _wins(board, t, s_value):
                revealed.append(t)
    if revealed:
        own = [t for t in hit if t not in revealed and _wins(board, t, VALUE[piece])]
        if own or board.is_check() and board.king(victim) not in revealed:
            found.add("discovered")
    return found


def opponent_tactics(after: chess.Board, victim: chess.Color) -> set[str]:
    """Every tactic the side to move in `after` could play against `victim` right now."""
    found: set[str] = set()
    for reply in list(after.legal_moves):
        after.push(reply)
        try:
            found |= created_tactics(after, reply)
        finally:
            after.pop()
        if len(found) == len(TACTICS):
            break
    return found


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
            if check_forks and loss >= 100:
                # Only worth the cost for moves Stockfish already dislikes.
                tactics = opponent_tactics(board, me)
                for t in TACTICS:
                    f[INDEX["allows_" + t]] = t in tactics
        finally:
            board.pop()

        if ctx.maia:
            f[INDEX["maia"]] = max(math.log(max(ctx.maia.get(uci, 0.0), 1e-9)), -9.0)

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
