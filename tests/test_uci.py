import io
import os
import sys

import chess
import chess.engine
import pytest

from nemesis.analysis.stockfish import find_stockfish, normalise_path
from nemesis.uci import UCI

HAVE_STOCKFISH = find_stockfish() is not None and os.path.isfile(find_stockfish())


@pytest.mark.skipif(not HAVE_STOCKFISH, reason="Stockfish not found (install it or set STOCKFISH_PATH)")
def test_uci_plays_legal_moves(tmp_path):
    cmd = [sys.executable, "-m", "nemesis.cli", "--db", str(tmp_path / "t.db"), "uci"]
    with chess.engine.SimpleEngine.popen_uci(cmd) as eng:
        assert eng.id["name"] == "Nemesis"
        eng.configure({"Player": "tester", "Risk": 80, "Depth": 6})
        board = chess.Board()
        for _ in range(4):
            result = eng.play(board, chess.engine.Limit(time=1), info=chess.engine.INFO_ALL)
            assert result.move in board.legal_moves, result.info.get("string")
            assert "error" not in (result.info.get("string") or ""), result.info["string"]
            board.push(result.move)


def test_uci_reports_errors_instead_of_dying(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKFISH_PATH", str(tmp_path / "missing.exe"))
    out = io.StringIO()
    uci = UCI(out=out, db=str(tmp_path / "t.db"))
    for line in ["uci", "isready", "position startpos moves e2e4", "go movetime 100"]:
        assert uci.handle(line)
    text = out.getvalue()
    assert "info string error: STOCKFISH_PATH=" in text
    move = text.strip().splitlines()[-1].split()[1]
    board = chess.Board()
    board.push_uci("e2e4")
    assert chess.Move.from_uci(move) in board.legal_moves


def test_normalise_path(tmp_path):
    exe = tmp_path / "stockfish" / "stockfish-windows-x86-64-avx2.exe"
    exe.parent.mkdir()
    exe.write_text("")
    (exe.parent / "README.md").write_text("")
    assert normalise_path(f'"{exe}"') == str(exe)            # quoted, as `set X="..."` leaves it
    assert normalise_path(f"  {exe} ") == str(exe)
    assert normalise_path(str(exe.parent)) == str(exe)       # the unzipped folder
    assert normalise_path(str(tmp_path)) == str(exe)         # folder containing that folder
    assert normalise_path(str(exe)[:-4]) == str(exe)         # .exe left off
    assert normalise_path(str(tmp_path / "nope.exe")) is None


def test_bad_env_is_not_replaced(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCKFISH_PATH", str(tmp_path / "nope"))
    assert find_stockfish() is None
