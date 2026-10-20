"""Exploit search: choose the move that does best against *this* opponent.

Nemesis's turns are max nodes. The human's turns are chance nodes: we average over
the moves the opponent model predicts, instead of assuming the best reply. Leaves are
Stockfish evals converted to expected score.

The risk dial r in [0, 1] controls two things:
  * window(r): how far below Stockfish's best a root move may be (15cp at r=0, 300cp at r=1)
  * lam(r):    the blend between objective expected score and expected score against you
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

from nemesis.analysis.stockfish import Analyzer, impute_losses, terminal_cp, win_prob
from nemesis.features import STYLE_TEXT, Context, position_features
from nemesis.profile.move_model import MoveModel
from nemesis.profile.repertoire import OpeningTree


def window_cp(risk: float, best_cp: float = 0) -> float:
    """How far below Stockfish's best a move may be. Shrinks when already winning: convert, don't hustle."""
    ahead = max(0.0, win_prob(best_cp) - 0.5) * 2   # 0 when equal or worse, ->1 when completely winning
    return 15 + 285 * risk * (1 - ahead) ** 2


def blend(risk: float) -> float:
    return 0.5 + 0.5 * risk


@dataclass
class Reply:
    uci: str
    prob: float
    cp: int            # eval after the reply, from the human's point of view
    loss: int          # how much worse than their best reply it is
    reasons: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    uci: str
    san: str
    cp: int            # Stockfish eval for Nemesis after this move
    objective: float   # expected score if the human plays perfectly
    exploit: float     # expected score against the modelled human
    score: float
    replies: list[Reply] = field(default_factory=list)
    book_note: str | None = None


@dataclass
class Decision:
    move: chess.Move
    candidates: list[Candidate]
    explanation: str

    @property
    def best(self) -> Candidate:
        return self.candidates[0]


class Searcher:
    def __init__(self, analyzer: Analyzer, model: MoveModel, tree: OpeningTree | None = None,
                 risk: float = 0.5, multipv: int = 6, root_multipv: int = 8, deep: bool = False):
        self.analyzer = analyzer
        self.model = model
        self.tree = tree
        self.risk = max(0.0, min(1.0, risk))
        self.multipv = multipv
        self.root_multipv = root_multipv
        self.deep = deep

    # -- the human's turn ---------------------------------------------------
    def human_replies(self, board: chess.Board, clock: float | None = None) -> list[Reply]:
        """Predicted distribution over the human's moves, with an eval for each."""
        lines = self.analyzer.lines(board, multipv=self.multipv)
        best = max(lines.values()) if lines else 0
        ctx = Context(clock=clock, best_cp=best)
        losses = impute_losses(board, lines)
        moves, X = position_features(board, losses, self.tree, ctx)
        p = self.model.probs(X)
        # Likely moves Stockfish didn't rank are where blunders hide: evaluate them properly.
        unknown = [moves[i].uci() for i in p.argsort()[::-1][:6]
                   if p[i] >= 0.03 and moves[i].uci() not in lines]
        if unknown:
            lines = self.analyzer.lines(board, multipv=self.multipv, extra=unknown)
            best = max(lines.values())
            losses = impute_losses(board, lines)
            moves, X = position_features(board, losses, self.tree, ctx)
            p = self.model.probs(X)
        replies = []
        for i, mv in enumerate(moves):
            u = mv.uci()
            replies.append(Reply(uci=u, prob=float(p[i]), cp=best - losses[u], loss=losses[u],
                                 reasons=self._reasons(X[i])))
        replies.sort(key=lambda r: -r.prob)
        return replies

    def _reasons(self, x) -> list[str]:
        contrib = self.model.contributions(x)
        top = sorted(((c, n) for n, c in contrib.items() if n in STYLE_TEXT and c > 0.15), reverse=True)
        return [STYLE_TEXT[n] for _, n in top[:2]]

    def human_value(self, board: chess.Board, depth: int, clock: float | None = None) -> tuple[float, list[Reply]]:
        """Nemesis's expected score with the human to move."""
        term = terminal_cp(board)
        if term is not None:
            return win_prob(-term), []
        replies = self.human_replies(board, clock)
        value = 0.0
        for r in replies:
            v = win_prob(-r.cp)
            if depth > 1 and r.prob >= 0.12:
                board.push(chess.Move.from_uci(r.uci))
                v = self.nemesis_value(board, depth - 1)
                board.pop()
            value += r.prob * v
        return self._steer(board, value), replies

    def nemesis_value(self, board: chess.Board, depth: int) -> float:
        term = terminal_cp(board)
        if term is not None:
            return win_prob(term)
        lines = self.analyzer.lines(board, multipv=3)
        best = max(lines.values())
        out = 0.0
        for u, cp in lines.items():
            if best - cp > 50:
                continue
            board.push(chess.Move.from_uci(u))
            v, _ = self.human_value(board, depth)
            board.pop()
            out = max(out, v)
        return out

    def _steer(self, board: chess.Board, value: float) -> float:
        """In the opening, lean toward positions where the player has historically scored badly."""
        if not self.tree:
            return value
        node = self.tree.stats(board)
        if node is None or node.games < 3:
            return value
        trust = min(node.games / 20, 0.4)
        return (1 - trust) * value + trust * (1 - node.avg)

    # -- Nemesis's turn (root) ----------------------------------------------
    def choose(self, board: chess.Board, human_clock: float | None = None) -> Decision:
        lines = self.analyzer.lines(board, multipv=self.root_multipv)
        if not lines:
            raise ValueError("no legal moves")
        best = max(lines.values())
        window, lam = window_cp(self.risk, best), blend(self.risk)
        depth = 2 if self.deep else 1
        cands: list[Candidate] = []
        for u, cp in lines.items():
            if best - cp > window:
                continue
            move = chess.Move.from_uci(u)
            san = board.san(move)
            board.push(move)
            obj = win_prob(cp)
            exploit, replies = self.human_value(board, depth, human_clock)
            note = None
            if self.tree and (node := self.tree.stats(board)) and node.games >= 3:
                note = f"you score {node.avg:.0%} from here in {node.games} games"
            board.pop()
            cands.append(Candidate(uci=u, san=san, cp=cp, objective=obj, exploit=exploit,
                                   score=(1 - lam) * obj + lam * exploit, replies=replies[:5], book_note=note))
        # Expected score saturates near 1 in won positions, so the raw eval breaks near-ties.
        cands.sort(key=lambda c: (-round(c.score, 3), -c.cp))
        decision = Decision(move=chess.Move.from_uci(cands[0].uci), candidates=cands, explanation="")
        decision.explanation = explain(board, decision)
        return decision


def explain(board: chess.Board, decision: Decision) -> str:
    c = decision.best
    after = board.copy()
    after.push(decision.move)
    parts = [f"{c.san}: {c.exploit:.0%} expected against you ({c.objective:.0%} against perfect play)."]
    traps = [r for r in c.replies if r.loss >= 100 and r.prob >= 0.08]
    if traps:
        r = traps[0]
        san = after.san(chess.Move.from_uci(r.uci))
        why = f" because you {' and '.join(r.reasons)}" if r.reasons else ""
        parts.append(f"Trap: I expect {san} ({r.prob:.0%}){why}; it loses {r.loss / 100:.1f} pawns.")
    elif c.replies:
        r = c.replies[0]
        parts.append(f"Most likely reply: {after.san(chess.Move.from_uci(r.uci))} ({r.prob:.0%}).")
    if c.book_note:
        parts.append(f"Steering: {c.book_note}.")
    best_obj = max(decision.candidates, key=lambda x: x.objective)
    if best_obj.uci != c.uci:
        parts.append(f"(Objectively {best_obj.san} is stronger, by {(best_obj.cp - c.cp) / 100:.2f}.)")
    return " ".join(parts)
