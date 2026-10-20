"""Nemesis: the opponent-specific engine, tying the model, opening tree and search together."""

from __future__ import annotations

import chess

from nemesis.analysis.stockfish import Analyzer, StockfishAnalyzer
from nemesis.profile.move_model import load_player_model
from nemesis.profile.repertoire import OpeningTree
from nemesis.search import Decision, Searcher
from nemesis.store import Store


class Nemesis:
    def __init__(self, store: Store, player: str, risk: float = 0.5, depth: int = 10,
                 deep: bool = False, analyzer: Analyzer | None = None, stockfish_path: str | None = None):
        self.store = store
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
        searcher = Searcher(self.analyzer, self.model, self.tree, risk=self.risk, deep=self.deep)
        return searcher.choose(board, human_clock)

    def close(self) -> None:
        if self._own_analyzer and hasattr(self.analyzer, "close"):
            self.analyzer.close()
