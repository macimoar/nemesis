"""Training pipeline, plus learning from games played against Nemesis."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Callable

import chess
import chess.pgn

from nemesis import puzzles, ratings, timecontrol
from nemesis.analysis.stockfish import Analyzer
from nemesis.ingest.pgn import to_rows
from nemesis.profile.move_model import MoveModel, build_samples, train_player, train_population
from nemesis.profile.repertoire import OpeningTree
from nemesis.profile.style import style_profile
from nemesis.store import Store
from nemesis.timecontrol import TimeControl


def train(store: Store, player: str, analyzer: Analyzer, max_games: int | None = None,
          progress: Callable[[int, int], None] | None = None) -> tuple[MoveModel, dict]:
    """Analyse the player's games, fit their model, and store the model with its profile."""
    player = player.lower()
    tree = OpeningTree.from_store(store, player)
    samples = build_samples(store, player, analyzer, tree=tree, max_games=max_games, progress=progress)
    model, metrics = train_player(store, player, samples)
    metrics["profile"] = style_profile(store, player, model, samples)
    store.save_model(player, model.to_dict(), metrics)
    metrics["new_puzzles"] = puzzles.mine(store, player, analyzer, samples, model)
    return model, metrics


def train_pooled(store: Store, analyzer: Analyzer, max_games_each: int | None = None) -> MoveModel:
    """Fit the population prior on every imported player's games."""
    samples = []
    for row in store.players():
        samples += build_samples(store, row["player"], analyzer, max_games=max_games_each)
    return train_population(store, samples)


def build_pgn(player: str, player_color: chess.Color, moves: list[str], result: str | None = None,
              start_fen: str = chess.STARTING_FEN, tc: TimeControl | None = None,
              clocks: list[float | None] | None = None) -> chess.pgn.Game:
    """A game against Nemesis as PGN, with clock times when known. `result` None = from the board."""
    board = chess.Board(start_fen)
    game = chess.pgn.Game()
    if start_fen != chess.STARTING_FEN:
        game.setup(board)
    node = game
    for i, u in enumerate(moves):
        move = chess.Move.from_uci(u)
        node = node.add_variation(move)
        if clocks and i < len(clocks) and clocks[i] is not None:
            node.set_clock(max(0.0, clocks[i]))
        board.push(move)
    if result is None:
        result = board.result(claim_draw=True)
    now = dt.datetime.now(dt.timezone.utc)
    game.headers.update({
        "Event": f"Nemesis {timecontrol.category(tc)}", "TimeControl": tc.pgn() if tc else "-",
        "Site": f"nemesis/{uuid.uuid4().hex[:12]}",
        "White": player if player_color == chess.WHITE else "Nemesis",
        "Black": player if player_color == chess.BLACK else "Nemesis",
        "Result": result, "Date": now.strftime("%Y.%m.%d"),
        "UTCDate": now.strftime("%Y.%m.%d"), "UTCTime": now.strftime("%H:%M:%S"),
    })
    return game


def record_game(store: Store, player: str, player_color: chess.Color, moves: list[str],
                result: str | None = None, start_fen: str = chess.STARTING_FEN,
                risk: float | None = None, annotations: list[dict] | None = None,
                tc: TimeControl | None = None, clocks: list[float] | None = None) -> str | None:
    """Store a game played against Nemesis so the next retrain learns from it.

    `result` is a PGN result ("1-0", "0-1", "1/2-1/2"). If it is None, it is taken from the board.
    With `risk`, a decided game is also rated (in its time control's rating); `annotations`
    (Decision.annotation per Nemesis move) are kept for the game review. `clocks` holds each
    mover's remaining seconds after each move, so training sees how you play under time pressure.
    """
    if not moves:
        return None
    game = build_pgn(player, player_color, moves, result, start_fen, tc, clocks)
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
        ratings.update_after_game(store, row.id, player, risk, row.result, timecontrol.category(tc))
    return row.id
