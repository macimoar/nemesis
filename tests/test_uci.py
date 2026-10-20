import sys

import chess
import chess.engine
import pytest

from nemesis.analysis.stockfish import find_stockfish

pytestmark = pytest.mark.skipif(find_stockfish() is None, reason="Stockfish not installed")


def test_uci_plays_legal_moves(tmp_path):
    cmd = [sys.executable, "-m", "nemesis.cli", "--db", str(tmp_path / "t.db"), "uci"]
    with chess.engine.SimpleEngine.popen_uci(cmd) as eng:
        assert eng.id["name"] == "Nemesis"
        eng.configure({"Player": "tester", "Risk": 80, "Depth": 6})
        board = chess.Board()
        for _ in range(4):
            result = eng.play(board, chess.engine.Limit(time=1))
            assert result.move in board.legal_moves
            board.push(result.move)
