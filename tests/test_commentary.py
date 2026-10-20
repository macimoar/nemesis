import re

import chess
import pytest

from nemesis.commentary import (POKER, XRAY, before, describe_move, eval_words, normalise_mode,
                                quality_words, react)
from nemesis.search import Candidate, Decision, Reply


def make_decision(with_trap=True):
    """After 1.e4, Nemesis (black) plays ...a6 hoping White grabs it with the bishop."""
    board = chess.Board()
    board.push_uci("e2e4")
    replies = [
        Reply("f1a6", 0.5 if with_trap else 0.05, cp=-450, loss=480 if with_trap else 0,
              reasons=["grab material when offered"]),
        Reply("d2d4", 0.25, cp=30, loss=0),
        Reply("g1f3", 0.2, cp=25, loss=5),
    ]
    if not with_trap:
        replies.sort(key=lambda r: -r.prob)
    best = Candidate("a7a6", "a6", cp=-40, objective=0.46, exploit=0.69, score=0.6, replies=replies)
    other = Candidate("e7e5", "e5", cp=-25, objective=0.48, exploit=0.5, score=0.49)
    return board, Decision(move=chess.Move.from_uci("a7a6"), candidates=[best, other])


def after(board, decision):
    b = board.copy()
    b.push(decision.move)
    return b


@pytest.mark.parametrize("mode", [XRAY, POKER])
def test_before_never_names_a_predicted_reply(mode):
    board, d = make_decision()
    text = " ".join(before(board, d, mode))
    pos = after(board, d)
    for r in d.best.replies:
        move = chess.Move.from_uci(r.uci)
        assert pos.san(move) not in text
        assert describe_move(pos, move) not in text


def test_xray_before_lists_the_reasoning():
    board, d = make_decision()
    lines = before(board, d, XRAY)
    labels = [re.match(r"\*\*(.+?):\*\*", l).group(1) for l in lines]
    assert labels[:4] == ["My move", "The position", "Why this move", "My chances"]
    assert "Hint" in labels
    assert "pawn to a6" in lines[0]
    assert "69% against you" in " ".join(lines)
    assert not any("expected against" in l or "Objectively" in l for l in lines)


def test_trap_sprung_is_revealed():
    board, d = make_decision()
    (line,) = react(after(board, d), d, chess.Move.from_uci("f1a6"), loss=480)
    assert "That was the trap" in line and "grab material when offered" in line
    assert "bishop takes the pawn on a6" in line


def test_trap_dodged_and_prediction_tiers():
    board, d = make_decision()
    pos = after(board, d)
    lines = " ".join(react(pos, d, chess.Move.from_uci("d2d4"), loss=0))
    assert "I thought you might" in lines and "1 in 4" in lines
    assert "Good eye" in lines and "bishop takes the pawn on a6" in lines
    lines = " ".join(react(pos, d, chess.Move.from_uci("h2h4"), loss=150))
    assert "didn't predict" in lines and "That's a mistake" in lines
    assert "Good eye" not in lines and "stepped around the trap" in lines
    assert "Why I think" not in lines  # a surprise move has no "why I expected it"


def test_expected_reaction_without_trap():
    board, d = make_decision(with_trap=False)
    assert d.trap is None
    lines = react(after(board, d), d, chess.Move.from_uci("d2d4"), loss=0)
    assert "Just what I expected" in lines[0]
    assert not any("Trap" in l for l in lines)
    assert not any("Hint" in l for l in before(board, d, XRAY))


def test_poker_face_gives_nothing_away():
    board, d = make_decision()
    pos = after(board, d)
    said = before(board, d, POKER)
    for reply, loss in [("f1a6", 480), ("d2d4", 0), ("h2h4", 150)]:
        said += react(pos, d, chess.Move.from_uci(reply), loss=loss, mode=POKER)
    assert react(pos, d, chess.Move.from_uci("h2h4"), loss=500, mode=POKER)[0] in (
        "Oh, thank you.", "Are you sure about that?", "I'll take that.")
    for line in said:
        assert not re.search(r"\d", line)
        assert not re.search(r"\b(pawn|knight|bishop|rook|queen|king)\b", line, re.I)
        assert "**" not in line


def test_plain_language_helpers():
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert describe_move(b, chess.Move.from_uci("e1g1")) == "castles kingside"
    b = chess.Board("4k3/8/8/3n4/8/8/8/3QK3 w - - 0 1")
    assert describe_move(b, chess.Move.from_uci("d1d5")) == "queen takes the knight on d5"
    assert describe_move(b, chess.Move.from_uci("d1a4")) == "queen to a4 with check"
    assert quality_words(0) == "your best move"
    assert quality_words(450).startswith("a blunder: it gives away about 4.5 pawns")
    assert eval_words(10) == "The position is level."
    assert eval_words(350).startswith("I'm clearly winning")
    assert eval_words(-150).startswith("You're better")
    assert normalise_mode("Poker Face") == POKER and normalise_mode("X-Ray") == XRAY
