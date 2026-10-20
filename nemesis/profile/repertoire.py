"""The player's opening tree: what they play in each position and how they score from it."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import chess

from nemesis.features import position_key
from nemesis.store import Store

MAX_PLY = 24


@dataclass
class Node:
    moves: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    games: int = 0
    score: float = 0.0  # player's total score from games that reached this position

    @property
    def avg(self) -> float:
        return self.score / self.games if self.games else 0.5


class OpeningTree:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = defaultdict(Node)

    def add_game(self, fens_and_moves: list[tuple[str, str, bool]], result: float | None) -> None:
        """fens_and_moves: (fen before move, uci, made by player)."""
        for ply, (fen, uci, by_player) in enumerate(fens_and_moves[:MAX_PLY]):
            node = self.nodes[position_key(chess.Board(fen))]
            node.games += 1
            node.score += 0.5 if result is None else result
            if by_player:
                node.moves[uci] += 1

    def freq(self, key: str, uci: str) -> int:
        node = self.nodes.get(key)
        return node.moves.get(uci, 0) if node else 0

    def total(self, key: str) -> int:
        node = self.nodes.get(key)
        return sum(node.moves.values()) if node else 0

    def stats(self, board: chess.Board) -> Node | None:
        return self.nodes.get(position_key(board))

    @classmethod
    def from_store(cls, store: Store, player: str) -> "OpeningTree":
        tree = cls()
        for game in store.games(player):
            moves = store.moves(game.id)
            tree.add_game([(m.fen, m.uci, bool(m.is_player)) for m in moves], game.result)
        return tree
