import chess

from nemesis.features import INDEX, position_features


def features_of(board, uci, losses=None):
    losses = losses or {m.uci(): 200 for m in board.legal_moves}
    moves, X = position_features(board, losses)
    return X[[m.uci() for m in moves].index(uci)]


def test_allows_knight_fork():
    # The queen on d1 guards c2. Moving it away lets ...Nc2+ fork king and rook.
    board = chess.Board("6k1/8/8/8/1n6/8/8/R2QK3 w - - 0 1")
    assert features_of(board, "d1h5")[INDEX["allows_fork"]] == 1
    assert features_of(board, "d1d3")[INDEX["allows_fork"]] == 0  # still guards c2
    assert features_of(board, "d1d2")[INDEX["allows_fork"]] == 0


def test_capture_trade_and_hanging():
    board = chess.Board("4k3/8/8/3p4/4N3/8/8/4K3 w - - 0 1")
    assert features_of(board, "e4d6")[INDEX["capture"]] == 0
    x = features_of(chess.Board("4k3/8/8/3n4/4N3/8/8/4K3 w - - 0 1"), "e4c5")
    assert x[INDEX["capture"]] == 0
    trade = features_of(chess.Board("4k3/8/3n4/8/4N3/8/8/4K3 w - - 0 1"), "e4d6")
    assert trade[INDEX["capture"]] == 1 and trade[INDEX["trade"]] == 1
    # A knight dropped where a pawn takes it is a sacrifice.
    sac = features_of(chess.Board("4k3/4p3/8/8/4N3/8/8/4K3 w - - 0 1"), "e4d6")
    assert sac[INDEX["sacrifice"]] == 1


def test_objective_features():
    board = chess.Board()
    losses = {m.uci(): 50 for m in board.legal_moves}
    losses["e2e4"] = 0
    losses["g2g4"] = 350
    assert features_of(board, "e2e4", losses)[INDEX["is_best"]] == 1
    x = features_of(board, "g2g4", losses)
    assert x[INDEX["blunder"]] == 1 and x[INDEX["mistake"]] == 1 and x[INDEX["cp_loss"]] == 3.5
