"""Doppelgänger: play like any imported player.

The move model that predicts a player's moves can also play them: in each position it
samples a move with the probability that player would choose it. In the opening it
follows the player's own repertoire, picking each move as often as they did. No
search, no traps; just their habits, good and bad.
"""

from __future__ import annotations

import random

import chess

from nemesis import commentary, maia, ratings
from nemesis.analysis.stockfish import Analyzer, impute_losses, win_prob
from nemesis.engine import search_settings
from nemesis.features import Context, position_features
from nemesis.profile.move_model import load_player_model
from nemesis.profile.repertoire import MAX_PLY, OpeningTree
from nemesis.store import Store

TOP = 5   # choices shown in X-Ray


def label(name: str) -> str:
    return f"Doppelgänger of {name}"


class Doppelganger:
    def __init__(self, store: Store, name: str, analyzer: Analyzer, mode: str = commentary.XRAY,
                 seed: int | None = None, tc: str = "untimed", opponent_elo: float | None = None):
        self.store, self.analyzer = store, analyzer
        self.name = name.lower()
        self.model = load_player_model(store, self.name, with_maia=maia.get("rapid", allow_download=False) is not None,
                                       tc=tc)
        self.tree = OpeningTree.from_store(store, self.name)
        self.rng = random.Random(seed)
        self.mode = commentary.normalise_mode(mode)
        self.tc = tc                      # time-control category: picks Maia's rapid or blitz model
        self.opponent_elo = opponent_elo  # your rating, which Maia-2 also takes into account

    def _book(self, board: chess.Board) -> tuple[dict[str, int], int]:
        """Their moves from this position in the opening, and how many games they had here."""
        node = self.tree.stats(board) if len(board.move_stack) < MAX_PLY else None
        if node is None:
            return {}, 0
        legal = {m.uci() for m in board.legal_moves}
        moves = {u: n for u, n in node.moves.items() if u in legal and n}
        return moves, sum(moves.values())

    def choose(self, board: chess.Board, own_clock: float | None = None,
               budget: float | None = None) -> tuple[chess.Move, list[str], list[dict]]:
        """(move, what X-Ray says about it, their top choices with probabilities)."""
        book, total = self._book(board)
        # The more often they've been here, the more surely they repeat themselves.
        if total and self.rng.random() < total / (total + 1):
            ucis = sorted(book, key=lambda u: -book[u])
            uci = self.rng.choices(ucis, weights=[book[u] for u in ucis])[0]
            move = chess.Move.from_uci(uci)
            choices = [{"san": board.san(chess.Move.from_uci(u)), "prob": round(book[u] / total, 3)}
                       for u in ucis[:TOP]]
            times = f"{book[uci]} of {total} game{'s' if total > 1 else ''}"
            lines = [commentary._label("Their opening", f"{self.name} played {board.san(move)} here in {times}.")]
            return move, self._say(lines), choices
        depth, multipv, _, _ = search_settings(budget, False)
        lines = self.analyzer.lines(board, multipv=multipv, depth=depth)
        losses = impute_losses(board, lines)
        best = max(lines.values()) if lines else 0
        human = None
        model = maia.get(maia.kind_for(self.tc), allow_download=False)
        if model is not None:
            offset = maia.rating_offset(self.store, self.name)
            me = ratings.player_rating(self.store, self.name, self.tc)[0] + offset
            human = model.policy(board, me, self.opponent_elo + offset if self.opponent_elo else me)
        legal, X = position_features(board, losses, self.tree, Context(clock=own_clock, best_cp=best, maia=human))
        probs = self.model.probs(X)
        i = self.rng.choices(range(len(legal)), weights=probs)[0]
        move = legal[i]
        order = sorted(range(len(legal)), key=lambda j: -probs[j])
        choices = [{"san": board.san(legal[j]), "prob": round(float(probs[j]), 3)} for j in order[:TOP]]
        # Whether it was a slip stays unsaid: spotting it is your job, as against the real thing.
        said = [commentary._label("Their style", f"{self.name} would play {board.san(move)} here about "
                                                 f"{probs[i]:.0%} of the time.")]
        return move, self._say(said), choices

    def _say(self, lines: list[str]) -> list[str]:
        return lines if self.mode == commentary.XRAY else []

    def consider_draw(self, board: chess.Board) -> tuple[bool, float, int]:
        """Like most players: take a draw when worse, play on when not. `board` has you to move."""
        lines = self.analyzer.lines(board, multipv=1)
        cp = -max(lines.values(), default=0)          # from the Doppelgänger's side
        major = sum(len(board.pieces(t, c)) for t in (chess.QUEEN, chess.ROOK) for c in chess.COLORS)
        dead_level = abs(cp) <= 25 and major <= 2 and board.fullmove_number >= 30
        return cp <= -100 or dead_level, win_prob(cp), cp

    def draw_reply(self, accept: bool, cp: int) -> list[str]:
        if self.mode != commentary.XRAY:
            return ["Agreed." if accept else "No thanks."]
        if accept:
            return [commentary._label("Draw offer", f"Accepted. Like most players, {self.name} takes a draw "
                                                    "when worse or when nothing is left to play for.")]
        return [commentary._label("Draw offer", f"Declined. {self.name} doesn't think they're worse here."),
                commentary._label("The position", commentary.eval_words(cp))]
