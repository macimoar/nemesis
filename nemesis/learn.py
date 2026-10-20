"""Training pipeline, plus learning from games played against Nemesis."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Callable

import chess
import chess.pgn

from nemesis import ratings
from nemesis.analysis.stockfish import Analyzer
from nemesis.ingest.pgn import to_rows
from nemesis.profile.move_model import MoveModel, build_samples, train_player, train_population
from nemesis.profile.repertoire import OpeningTree
from nemesis.profile.style import style_profile
from nemesis.store import Store


def train(store: Store, player: str, analyzer: Analyzer, max_games: int | None = None,
          progress: Callable[[int, int], None] | None = None) -> tuple[MoveModel, dict]:
    """Analyse the player's games, fit their model, and store the model with its profile."""
    player = player.lower()
    tree = OpeningTree.from_store(store, player)
    samples = build_samples(store, player, analyzer, tree=tree, max_games=max_games, progress=progress)
    model, metrics = train_player(store, player, samples)
    metrics["profile"] = style_profile(store, player, model, samples)
    store.save_model(player, model.to_dict(), metrics)
    return model, metrics


def train_pooled(store: Store, analyzer: Analyzer, max_games_each: int | None = None) -> MoveModel:
    """Fit the population prior on every imported player's games."""
    samples = []
    for row in store.players():
        samples += build_samples(store, row["player"], analyzer, max_games=max_games_each)
    return train_population(store, samples)


def record_game(store: Store, player: str, player_color: chess.Color, moves: list[str],
                result: str | None = None, start_fen: str = chess.STARTING_FEN,
                risk: float | None = None, annotations: list[dict] | None = None) -> str | None:
    """Store a game played against Nemesis so the next retrain learns from it.

    `result` is a PGN result ("1-0", "0-1", "1/2-1/2"). If it is None, it is taken from the board.
    With `risk`, a decided game is also rated; `annotations` (Decision.annotation per Nemesis
    move) are kept for the game review.
    """
    if not moves:
        return None
    board = chess.Board(start_fen)
    game = chess.pgn.Game()
    if start_fen != chess.STARTING_FEN:
        game.setup(board)
    node = game
    for u in moves:
        move = chess.Move.from_uci(u)
        node = node.add_variation(move)
        board.push(move)
    if result is None:
        result = board.result(claim_draw=True)
    now = dt.datetime.now(dt.timezone.utc)
    game.headers.update({
        "Event": "Nemesis", "Site": f"nemesis/{uuid.uuid4().hex[:12]}",
        "White": player if player_color == chess.WHITE else "Nemesis",
        "Black": player if player_color == chess.BLACK else "Nemesis",
        "Result": result, "UTCDate": now.strftime("%Y.%m.%d"), "UTCTime": now.strftime("%H:%M:%S"),
    })
    rows = to_rows(game, player, source="nemesis")
    if rows is None:
        return None
    row, move_rows = rows
    row.id = f"nemesis:{uuid.uuid4().hex[:12]}"
    for m in move_rows:
        m.game_id = row.id
    store.add_game(row, move_rows)
    if annotations:
        store.put_json("annotations", row.id, {"risk": risk, "moves": annotations})
    if risk is not None and row.result is not None:
        ratings.update_after_game(store, row.id, player, risk, row.result)
    return row.id
