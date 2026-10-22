"""Nemesis: the opponent-specific engine, tying the model, opening tree and search together."""

from __future__ import annotations

import random
import time

import chess

from nemesis import commentary, maia, ratings
from nemesis.analysis.stockfish import Analyzer, StockfishAnalyzer, impute_losses
from nemesis.profile.move_model import load_player_model
from nemesis.profile.repertoire import OpeningTree
from nemesis.search import Decision, Searcher
from nemesis.store import Store


def search_settings(budget: float | None, deep: bool) -> tuple[int | None, int, int, bool]:
    """(Stockfish depth, MultiPV for your replies, MultiPV at the root, deep) for a time budget.
    Less time means a shallower, narrower search: faster and a little weaker, never a loss on time."""
    if budget is None or budget >= 5:
        return None, 6, 8, deep
    if budget >= 2.5:
        return 8, 6, 8, False
    if budget >= 1:
        return 6, 5, 6, False
    return 4, 4, 4, False


class Nemesis:
    def __init__(self, store: Store, player: str, risk: float = 0.5, depth: int = 10,
                 deep: bool = False, analyzer: Analyzer | None = None, stockfish_path: str | None = None,
                 mode: str = commentary.XRAY, elo: float | None = None, seed: int | None = None,
                 tc: str = "untimed"):
        self.store = store
        self.tc = tc                  # time-control category: which rating "match me" uses
        self.fixed_elo = elo          # None or 0: "match me" (your rating + 50)
        self.rng = random.Random(seed)
        self.mode = commentary.normalise_mode(mode)
        self.depth = depth
        self.deep = deep
        self.risk = risk
        self._own_analyzer = analyzer is None
        self.analyzer = analyzer or StockfishAnalyzer(store, path=stockfish_path, depth=depth)
        self.set_player(player)

    def set_player(self, player: str) -> None:
        self.player = player.lower()
        has_maia = maia.get(maia.kind_for(self.tc), allow_download=False) is not None
        self.model = load_player_model(self.store, self.player, with_maia=has_maia)
        self.tree = OpeningTree.from_store(self.store, self.player)

    def reload(self) -> None:
        self.set_player(self.player)

    def choose(self, board: chess.Board, human_clock: float | None = None,
               budget: float | None = None) -> Decision:
        """Pick a move. `budget`: seconds Nemesis may think (from its own clock), None = no limit."""
        depth, multipv, root_multipv, deep = search_settings(budget, self.deep)
        searcher = Searcher(self.analyzer, self.model, self.tree, risk=self.risk, deep=deep,
                            elo=self.target_elo(), rng=self.rng, depth=depth,
                            multipv=multipv, root_multipv=root_multipv,
                            deadline=None if budget is None else time.monotonic() + 0.8 * budget,
                            **self._maia())
        decision = searcher.choose(board, human_clock)
        decision.lines = commentary.before(board, decision, self.mode)
        return decision

    def consider_draw(self, board: chess.Board) -> tuple[bool, float, int]:
        """Should Nemesis accept your draw offer? (accept, its expected score against you,
        its eval in centipawns). `board` has you to move.

        It judges as itself: by its chances against *you* (your predicted replies), not
        against perfect play. It declines while it expects to come out ahead, and takes the
        draw when it's worse, or when the game is a dead-level endgame.
        """
        searcher = Searcher(self.analyzer, self.model, self.tree, risk=self.risk, elo=None, rng=self.rng,
                            **self._maia())
        expected, replies = searcher.human_value(board, 1)
        cp = -max((r.cp for r in replies), default=0)   # its eval: your best reply, from its side
        major = sum(len(board.pieces(t, c)) for t in (chess.QUEEN, chess.ROOK) for c in chess.COLORS)
        dead_level = abs(cp) <= 25 and major <= 2 and board.fullmove_number >= 30
        return expected <= 0.45 or dead_level, expected, cp

    def _maia(self) -> dict:
        """Maia-2 for predicting your moves, if it's been downloaded (never waits for it)."""
        model = maia.get(maia.kind_for(self.tc), allow_download=False)
        if model is None:
            return {}
        offset = maia.rating_offset(self.store, self.player)
        you = ratings.player_rating(self.store, self.player, self.tc)[0]
        return {"maia": model, "maia_elos": (you + offset, self.target_elo() + offset)}

    def target_elo(self) -> float:
        return ratings.target_elo(self.store, self.player, self.fixed_elo, self.tc)

    def react(self, board: chess.Board, decision: Decision, reply: chess.Move) -> list[str]:
        """Comment on the human's reply. `board` is the position after Nemesis's move."""
        lines = self.analyzer.lines(board, extra=[reply.uci()])
        loss = impute_losses(board, lines).get(reply.uci(), 0)
        decision.reply = {"uci": reply.uci(), "loss": loss}   # kept for "does it work on you?"
        return commentary.react(board, decision, reply, loss, self.mode)

    def close(self) -> None:
        if self._own_analyzer and hasattr(self.analyzer, "close"):
            self.analyzer.close()
