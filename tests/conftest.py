import chess
import pytest

from nemesis.features import VALUE
from nemesis.store import Store


class MaterialAnalyzer:
    """A tiny stand-in for Stockfish: 2-ply material count, in centipawns, from the mover's view."""

    def _material(self, board: chess.Board, color: chess.Color) -> int:
        total = 0
        for p in board.piece_map().values():
            if p.piece_type != chess.KING:
                total += VALUE[p.piece_type] * (1 if p.color == color else -1)
        return total * 100

    def lines(self, board, multipv=6, extra=()):
        me = board.turn
        scores = {}
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                worst = 2000
            else:
                worst = None
                for reply in board.legal_moves:
                    board.push(reply)
                    v = -2000 if board.is_checkmate() else self._material(board, me)
                    board.pop()
                    worst = v if worst is None else min(worst, v)
                worst = self._material(board, me) if worst is None else worst
            board.pop()
            scores[move.uci()] = worst
        top = dict(sorted(scores.items(), key=lambda kv: -kv[1])[:multipv])
        for u in extra:
            if u in scores:
                top[u] = scores[u]
        return top


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture
def analyzer():
    return MaterialAnalyzer()
