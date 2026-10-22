import chess
import pytest

from nemesis.features import INDEX, created_tactics, opponent_tactics, position_features


def after(fen, uci):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    board.push(move)
    return created_tactics(board, move)


@pytest.mark.parametrize("name, fen, uci, expected", [
    ("pin: knight to king, undefended", "4k3/8/2n5/8/8/8/8/4KB2 w - - 0 1", "f1b5", {"pin"}),
    ("skewer: king, then rook", "r3k3/8/8/8/8/8/8/4K2R w - - 0 1", "h1h8", {"skewer"}),
    ("fork: knight on king and rook", "r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1", "b5c7", {"fork"}),
    ("discovered: knight hits queen, bishop hits rook", "4k2r/8/3q4/8/8/2N5/1B6/4K3 w - - 0 1", "c3e4", {"discovered"}),
    ("back-rank mate", "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", "e1e8", {"back_rank"}),
    ("quiet move", chess.STARTING_FEN, "e2e4", set()),
])
def test_created_tactics(name, fen, uci, expected):
    assert after(fen, uci) == expected, name


@pytest.mark.parametrize("name, fen, uci", [
    # The pinned knight is defended by its queen and the pinner is a rook: nothing is won.
    ("pin that wins nothing", "3q4/8/3n4/8/8/8/8/R3K2k w - - 0 1", "a1d1"),
    # Uncovering an attack on a rook while threatening nothing else: the rook just moves.
    ("discovered attack with one threat", "4k2r/8/8/8/8/2N5/1B6/4K3 w - - 0 1", "c3e4"),
    # A knight "forking" king and rook, but the bishop just takes it.
    ("fork by a piece that's taken for free", "r3k3/8/8/bN6/8/8/8/4K3 w - - 0 1", "b5c7"),
])
def test_no_false_alarms(name, fen, uci):
    assert after(fen, uci) == set(), name


def test_back_rank_blind_spot_feature():
    # White to move. Black threatens Rd1#: Kh1 ignores it, h3 makes room for the king.
    board = chess.Board("3r2k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1")
    ucis = [m.uci() for m in board.legal_moves]
    losses = {u: 900 for u in ucis} | {"h2h3": 0, "g2g3": 0}
    moves, X = position_features(board, losses)
    assert [m.uci() for m in moves] == ucis
    assert X[ucis.index("g1h1"), INDEX["allows_back_rank"]] == 1
    assert X[ucis.index("h2h3"), INDEX["allows_back_rank"]] == 0      # good moves aren't scanned
    # Only moves Stockfish already dislikes are scanned, to keep it fast.
    losses["g1h1"] = 0
    _, X = position_features(board, losses)
    assert X[ucis.index("g1h1"), INDEX["allows_back_rank"]] == 0
