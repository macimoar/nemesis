"""Nemesis: the opponent-specific engine, tying the model, opening tree and search together."""

from __future__ import annotations

import random

import chess

from nemesis import commentary, ratings
from nemesis.analysis.stockfish import Analyzer, StockfishAnalyzer, impute_losses
from nemesis.profile.move_model import load_player_model
from nemesis.profile.repertoire import OpeningTree
from nemesis.search import Decision, Searcher
from nemesis.store import Store


class Nemesis:
    def __init__(self, store: Store, player: str, risk: float = 0.5, depth: int = 10,
                 deep: bool = False, analyzer: Analyzer | None = None, stockfish_path: str | None = None,
                 mode: str = commentary.XRAY, elo: float | None = None, seed: int | None = None):
        self.store = store
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
        self.model = load_player_model(self.store, self.player)
        self.tree = OpeningTree.from_store(self.store, self.player)

    def reload(self) -> None:
        self.set_player(self.player)

    def choose(self, board: chess.Board, human_clock: float | None = None) -> Decision:
        searcher = Searcher(self.analyzer, self.model, self.tree, risk=self.risk, deep=self.deep,
                            elo=self.target_elo(), rng=self.rng)
        decision = searcher.choose(board, human_clock)
        decision.lines = commentary.before(board, decision, self.mode)
        return decision

    def target_elo(self) -> float:
        return ratings.target_elo(self.store, self.player, self.fixed_elo)

    def react(self, board: chess.Board, decision: Decision, reply: chess.Move) -> list[str]:
        """Comment on the human's reply. `board` is the position after Nemesis's move."""
        lines = self.analyzer.lines(board, extra=[reply.uci()])
        loss = impute_losses(board, lines).get(reply.uci(), 0)
        return commentary.react(board, decision, reply, loss, self.mode)

    def close(self) -> None:
        if self._own_analyzer and hasattr(self.analyzer, "close"):
            self.analyzer.close()
