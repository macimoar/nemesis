"""What Nemesis says: before its move, and in reaction to yours.

Two modes:
  * X-Ray: a plain-English breakdown of its thinking. It never names the move it
    expects from you before you play. Traps are only hinted at, then revealed once
    sprung or dodged.
  * Poker Face: short, in-character lines with no numbers and no move names.
"""

from __future__ import annotations

import random
import zlib
from typing import TYPE_CHECKING

import chess

from nemesis.analysis.stockfish import MATE_CP

if TYPE_CHECKING:
    from nemesis.search import Decision, Reply

XRAY = "xray"
POKER = "poker"
MODES = {XRAY: "X-Ray", POKER: "Poker Face"}

PIECE = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
         chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}


def normalise_mode(mode: str | None) -> str:
    m = (mode or XRAY).lower().replace("-", "").replace(" ", "").replace("_", "")
    return POKER if m.startswith("poker") else XRAY


# ---------------------------------------------------------------------------
# Plain-language helpers
# ---------------------------------------------------------------------------

def describe_move(board: chess.Board, move: chess.Move) -> str:
    """'bishop to f4', 'knight takes the pawn on e4 with check', 'castles kingside'."""
    if board.is_castling(move):
        text = "castles " + ("kingside" if chess.square_file(move.to_square) == 6 else "queenside")
    else:
        piece = PIECE[board.piece_type_at(move.from_square)]
        to = chess.square_name(move.to_square)
        if board.is_en_passant(move):
            text = f"pawn takes the pawn en passant on {to}"
        elif board.is_capture(move):
            text = f"{piece} takes the {PIECE[board.piece_type_at(move.to_square)]} on {to}"
        else:
            text = f"{piece} to {to}"
        if move.promotion:
            text += f", promoting to a {PIECE[move.promotion]}"
    board.push(move)
    try:
        if board.is_checkmate():
            text += ", checkmate"
        elif board.is_check():
            text += " with check"
    finally:
        board.pop()
    return text


def _pawns(cp: float) -> str:
    p = abs(cp) / 100
    return f"{p:.1f} pawn" + ("" if p == 1 else "s")


def eval_words(cp: int) -> str:
    """Nemesis's view of the position, `cp` from its own side."""
    if cp >= MATE_CP - 50:
        return "I have a forced mate."
    if cp <= -(MATE_CP - 50):
        return "You have a forced mate. Let's see if you find it."
    mag = abs(cp)
    if mag < 30:
        return "The position is level."
    who = "I'm" if cp > 0 else "You're"
    if mag < 100:
        level = "slightly better"
    elif mag < 250:
        level = "better"
    elif mag < 600:
        level = "clearly winning"
    else:
        level = "winning decisively"
    side = "me" if cp > 0 else "you"
    return f"{who} {level} (about {_pawns(cp)} ahead for {side})."


def quality_words(loss: int) -> str:
    if loss <= 20:
        return "your best move"
    if loss <= 60:
        return "a good move"
    if loss <= 120:
        return f"a small slip (about {_pawns(loss)})"
    if loss <= 300:
        return f"a mistake: it gives away about {_pawns(loss)}"
    return f"a blunder: it gives away about {_pawns(loss)}"


def odds_words(p: float) -> str:
    if p >= 0.8:
        return "almost certain"
    if p >= 0.55:
        return "likely"
    if p >= 0.4:
        return "about a coin flip"
    if p >= 0.08:
        return f"about a 1 in {max(2, round(1 / p))} chance"
    return "a long shot"


def _label(label: str, text: str) -> str:
    return f"**{label}:** {text}"


def _pick(seed: str, options: list[str]) -> str:
    return random.Random(zlib.crc32(seed.encode())).choice(options)


# ---------------------------------------------------------------------------
# Before the human moves
# ---------------------------------------------------------------------------

def before(board: chess.Board, decision: "Decision", mode: str = XRAY) -> list[str]:
    """Commentary on Nemesis's move. `board` is the position *before* that move."""
    mode = normalise_mode(mode)
    c = decision.best
    if mode == POKER:
        return [_poker_before(board, decision)]

    lines = [_label("My move", f"{c.san} ({describe_move(board, decision.move)}).")]
    lines.append(_label("The position", eval_words(c.cp)))

    top = max(decision.candidates, key=lambda x: x.cp)
    suits = c.exploit > c.objective + 0.03
    gap = top.cp - c.cp
    if decision.elo and decision.top_scored != c.uci and gap > 30:
        # Honest about strength-limited slips rather than dressing them up as cunning.
        why = (f"Not my strongest option: {top.san} was better by {_pawns(gap)}. "
               f"I'm playing at about {decision.elo:.0f} strength, so I don't always find the best move.")
    elif top.uci == c.uci or gap <= 0:
        why = "It's the engine's top choice" + (", and it suits how you play." if suits else ".")
    elif gap <= 30:
        why = (f"It's as good as the engine's top choice. {top.san} is {gap / 100:.2f} pawns better, "
               "too small to matter" + (", and this one suits how you play." if suits else "."))
    elif gap <= 100:
        but = ("I expect it to work better against you." if suits
               else "my chances come out about the same either way.")
        why = f"It's slightly weaker than {top.san} objectively (by {_pawns(gap)}), but {but}"
    else:
        but = "I think it pays off against you." if suits else "I judged the risk worth taking."
        why = f"It's a deliberate gamble: {_pawns(gap)} worse than {top.san} against perfect play, but {but}"
    lines.append(_label("Why this move", why))

    if abs(c.exploit - c.objective) < 0.03:
        chances = f"About {c.exploit:.0%}. Nothing tricky here, just solid chess."
    else:
        chances = f"About {c.exploit:.0%} against you, compared with {c.objective:.0%} if you play perfectly."
    lines.append(_label("My chances", chances))

    if c.book:
        games, score = c.book
        lines.append(_label("Opening plan",
                            f"Heading for a position you've reached in {games} games and scored {score:.0%} from."))
    if decision.trap:
        lines.append(_label("Hint", "I've set something up. Look carefully before you move."))
    return lines


POKER_BEFORE = {
    "mate": ["This ends soon.", "I can see the end from here.", "Checkmate is coming. Can you stop it?"],
    "trap": ["Your move. Take your time.", "Go on, then.", "Careful now.", "I wonder what you'll do."],
    "gamble": ["Let's make this interesting.", "Let's see what you're made of.", "Feeling brave today."],
    "crushing": ["This is going well.", "You might want to dig in.", "I like where this is going."],
    "better": ["Things are looking up.", "I'm comfortable here.", "Steady."],
    "level": ["Your move.", "Hmm.", "All to play for.", "Let's see."],
    "worse": ["You've been playing well.", "I'm not done yet.", "Hm. Annoying."],
}


def _poker_before(board: chess.Board, decision: "Decision") -> str:
    c = decision.best
    top = max(decision.candidates, key=lambda x: x.cp)
    if c.cp >= MATE_CP - 50:
        kind = "mate"
    elif decision.trap:
        kind = "trap"
    elif top.cp - c.cp > 100:
        kind = "gamble"
    elif c.cp > 300:
        kind = "crushing"
    elif c.cp > 80:
        kind = "better"
    elif c.cp > -80:
        kind = "level"
    else:
        kind = "worse"
    return _pick(board.fen() + kind, POKER_BEFORE[kind])


# ---------------------------------------------------------------------------
# After the human moves
# ---------------------------------------------------------------------------

def react(board: chess.Board, decision: "Decision", reply: chess.Move, loss: int,
          mode: str = XRAY) -> list[str]:
    """Nemesis's reaction to your move. `board` is the position after Nemesis's move, before yours."""
    mode = normalise_mode(mode)
    replies = decision.best.replies
    found = next((r for r in replies if r.uci == reply.uci()), None)
    prob = found.prob if found else 0.0
    top = bool(replies) and replies[0].uci == reply.uci()
    tier = "expected" if top or prob >= 0.3 else "plausible" if prob >= 0.08 else "surprise"
    trap = decision.trap
    sprung = trap is not None and trap.uci == reply.uci()

    if mode == POKER:
        if sprung:
            kind = "sprung"
        elif loss >= 300:
            kind = "blunder"
        elif trap is not None and loss < 100:
            kind = "dodged"
        elif tier == "surprise" and loss <= 20:
            kind = "sharp"
        else:
            kind = tier
        return [_pick(board.fen() + reply.uci(), POKER_REACT[kind])]

    lines: list[str] = []
    if sprung:
        why = f", because you {' and '.join(found.reasons)}" if found and found.reasons else ""
        lines.append(_label("Gotcha", (
            f"That was the trap. I played {decision.best.san} hoping you'd play "
            f"{describe_move(board, reply)}{why}. It costs you about {_pawns(loss)}.")))
        return lines

    reaction = {
        "expected": "Just what I expected. I had that as your most likely move.",
        "plausible": f"I thought you might. I gave that {odds_words(prob)}.",
        "surprise": "Interesting, I didn't predict that.",
    }[tier]
    lines.append(_label("Reaction", reaction))
    lines.append(_label("Your move", f"{_capitalise(describe_move(board, reply))}. That's {quality_words(loss)}."))
    if found and found.reasons and loss >= 100 and tier != "surprise":
        lines.append(_label("Why I think you played it", f"You {' and '.join(found.reasons)}."))
    if trap is not None:
        hoped = f"I was hoping you'd play {describe_move(board, chess.Move.from_uci(trap.uci))}"
        if loss < 100:
            text = f"Good eye. {hoped}, which would have cost you about {_pawns(trap.loss)}."
        else:
            text = f"You stepped around the trap ({hoped}), but this move costs you too."
        lines.append(_label("Trap avoided", text))
    return lines


def better_move(board: chess.Board, best: chess.Move, wrong: str | None) -> list[str]:
    """X-Ray, after a mistake: what to play instead and, when it's clear, what went wrong.
    `board` is the position before your move."""
    out = [_label("Better was", f"{board.san(best)} ({describe_move(board, best)}).")]
    if wrong:
        out.append(_label("What went wrong", f"Your move: {wrong}."))
    return out


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:]


POKER_REACT = {
    "expected": ["As I thought.", "Predictable.", "Mm-hm.", "I knew you'd say that."],
    "plausible": ["I thought you might.", "Okay.", "Noted."],
    "surprise": ["Hm. Didn't see that coming.", "Interesting.", "Well, well.", "That's new."],
    "sharp": ["Oh. That's good, actually.", "Hm. Strong move.", "Didn't see that coming. Well played."],
    "sprung": ["Gotcha.", "And there it is.", "I was hoping you'd do that."],
    "dodged": ["...Well spotted.", "You saw it. Shame.", "Hm. You're paying attention."],
    "blunder": ["Oh, thank you.", "Are you sure about that?", "I'll take that."],
}


POKER_DRAW = {
    True: ["Fine. A draw.", "I'll take it.", "Agreed. Another time."],
    False: ["No thanks.", "Keep playing.", "Not a chance.", "We're not done."],
}


def draw_reply(accept: bool, expected: float, cp: int, mode: str = XRAY, seed: str = "") -> list[str]:
    """Nemesis's answer to your draw offer."""
    if normalise_mode(mode) == POKER:
        return [_pick(seed + str(accept), POKER_DRAW[accept])]
    if accept:
        why = (f"I give myself only about {expected:.0%} against you from here."
               if expected <= 0.45 else "There's nothing left to play for. It's level.")
        return [_label("Draw offer", f"Accepted. {why}")]
    return [_label("Draw offer", f"Declined. I give myself about {expected:.0%} against you here, so let's play on."),
            _label("The position", eval_words(cp))]
