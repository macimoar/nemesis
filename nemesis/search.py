"""Exploit search: choose the move that does best against *this* opponent.

Nemesis's turns are max nodes. The human's turns are chance nodes: we average over
the moves the opponent model predicts, instead of assuming the best reply. Leaves are
Stockfish evals converted to expected score.

The risk dial r in [0, 1] controls two things:
  * window(r): how far below Stockfish's best a root move may be (15cp at r=0, 300cp at r=1)
  * lam(r):    the blend between objective expected score and expected score against you

With a strength target (Elo), the final choice is sampled instead (see strength.py).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import chess

from nemesis import strength
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
    replies: list[Reply] = field(default_factory=list)   # full predicted distribution, most likely first
    book: tuple[int, float] | None = None                  # (games, player's score) if a known opening node


@dataclass
class Decision:
    move: chess.Move
    candidates: list[Candidate]
    lines: list[str] = field(default_factory=list)   # filled in by the engine's commentary
    elo: float | None = None      # strength target it played at (None = unlimited)
    top_scored: str = ""          # the move the unlimited search would have played
    lapse: bool = False           # a deliberate human-style slip at this strength
    reply: dict | None = None     # the human's answer once known: {"uci", "loss"}

    @property
    def best(self) -> Candidate:
        return self.candidates[0]

    @property
    def trap(self) -> Reply | None:
        """The reply Nemesis is hoping for: likely enough, and costly for the human."""
        return next((r for r in self.best.replies if r.loss >= 100 and r.prob >= 0.08), None)

    def annotation(self, ply: int, risk: float) -> dict:
        """What Nemesis was thinking at this move, saved with the game for its review."""
        trap = self.trap
        return {
            "ply": ply, "move": self.move.uci(), "risk": risk, "elo": self.elo,
            "best_effort": self.top_scored == self.move.uci(), "lapse": self.lapse,
            "trap": {"uci": trap.uci, "prob": round(trap.prob, 3), "loss": trap.loss,
                     "reasons": trap.reasons} if trap else None,
            "probs": {r.uci: round(r.prob, 3) for r in self.best.replies if r.prob >= 0.01},
            "top": self.best.replies[0].uci if self.best.replies else None,
        }


class Searcher:
    def __init__(self, analyzer: Analyzer, model: MoveModel, tree: OpeningTree | None = None,
                 risk: float = 0.5, multipv: int = 6, root_multipv: int = 8, deep: bool = False,
                 elo: float | None = None, rng: random.Random | None = None,
                 depth: int | None = None, deadline: float | None = None):
        self.analyzer = analyzer
        self.model = model
        self.tree = tree
        self.risk = max(0.0, min(1.0, risk))
        self.multipv = multipv
        self.root_multipv = root_multipv
        self.deep = deep
        self.elo = elo if strength.limited(elo) else None
        self.rng = rng or random.Random()
        self.depth = depth          # Stockfish depth override (shallower when short on time)
        self.deadline = deadline    # time.monotonic() by which a move must be chosen

    def _lines(self, board: chess.Board, multipv: int, extra=()) -> dict[str, int]:
        if self.depth is None:
            return self.analyzer.lines(board, multipv=multipv, extra=extra)
        return self.analyzer.lines(board, multipv=multipv, extra=extra, depth=self.depth)

    def out_of_time(self) -> bool:
        return self.deadline is not None and time.monotonic() > self.deadline

    # -- the human's turn ---------------------------------------------------
    def human_replies(self, board: chess.Board, clock: float | None = None) -> list[Reply]:
        """Predicted distribution over the human's moves, with an eval for each."""
        lines = self._lines(board, self.multipv)
        best = max(lines.values()) if lines else 0
        ctx = Context(clock=clock, best_cp=best)
        losses = impute_losses(board, lines)
        moves, X = position_features(board, losses, self.tree, ctx)
        p = self.model.probs(X)
        # Likely moves Stockfish didn't rank are where blunders hide: evaluate them properly.
        unknown = [moves[i].uci() for i in p.argsort()[::-1][:6]
                   if p[i] >= 0.03 and moves[i].uci() not in lines]
        if unknown and not self.out_of_time():
            lines = self._lines(board, self.multipv, extra=unknown)
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
        lines = self._lines(board, 3)
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
        multipv = 12 if self.elo else self.root_multipv
        lines = self._lines(board, multipv)
        if not lines:
            raise ValueError("no legal moves")
        best = max(lines.values())
        window, lam = window_cp(self.risk, best), blend(self.risk)
        if self.elo:
            window = max(window, strength.WINDOW)
        depth = 2 if self.deep else 1
        cands: list[Candidate] = []
        for u, cp in sorted(lines.items(), key=lambda kv: -kv[1]):  # best first, in case time runs out
            if best - cp > window:
                continue
            if cands and self.out_of_time():
                break
            move = chess.Move.from_uci(u)
            san = board.san(move)
            board.push(move)
            obj = win_prob(cp)
            exploit, replies = self.human_value(board, depth, human_clock)
            book = None
            if self.tree and (node := self.tree.stats(board)) and node.games >= 3:
                book = (node.games, node.avg)
            board.pop()
            cands.append(Candidate(uci=u, san=san, cp=cp, objective=obj, exploit=exploit,
                                   score=(1 - lam) * obj + lam * exploit, replies=replies, book=book))
        # Expected score saturates near 1 in won positions, so the raw eval breaks near-ties.
        cands.sort(key=lambda c: (-round(c.score, 3), -c.cp))
        top_scored, lapse = cands[0].uci, False
        if self.elo:
            i, lapse = strength.pick([best - c.cp for c in cands], [c.score for c in cands], self.elo, self.rng)
            cands.insert(0, cands.pop(i))  # the move actually played always comes first
        return Decision(move=chess.Move.from_uci(cands[0].uci), candidates=cands, elo=self.elo,
                        top_scored=top_scored, lapse=lapse)

