import chess

from nemesis.ingest.pgn import import_pgn_text
from nemesis.learn import record_game

PGN = """[Event "Rated Blitz game"]
[Site "https://lichess.org/abcd1234"]
[White "Someone"]
[Black "Me"]
[Result "0-1"]
[UTCDate "2026.01.02"]
[UTCTime "10:00:00"]
[Opening "Italian Game"]

1. e4 { [%clk 0:03:00] } e5 { [%clk 0:02:59] } 2. Nf3 { [%clk 0:02:58] } Nc6 { [%clk 0:02:55] }
3. Bc4 { [%clk 0:02:57] } Nf6 { [%clk 0:02:50] } 0-1

[Event "Other"]
[Site "https://lichess.org/zzzz9999"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. d4 d5 1-0
"""


def test_import_pgn(store):
    assert import_pgn_text(store, PGN, "me", source="lichess") == 1
    assert import_pgn_text(store, PGN, "me", source="lichess") == 0  # deduplicated
    (game,) = store.games("ME")
    assert game.id == "lichess:abcd1234"
    assert game.color == 0 and game.result == 1.0 and game.opening == "Italian Game"
    moves = store.moves(game.id)
    assert [m.uci for m in moves] == ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]
    assert [m.is_player for m in moves] == [0, 1, 0, 1, 0, 1]
    assert moves[1].clock == 179
    assert chess.Board(moves[2].fen).turn == chess.WHITE


def test_record_game(store):
    gid = record_game(store, "me", chess.BLACK, ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"])
    (game,) = store.games("me")
    assert game.id == gid and game.source == "nemesis"
    assert game.result == 0.0  # Scholar's mate against the player
    assert [m.is_player for m in store.moves(gid)][:2] == [0, 1]
