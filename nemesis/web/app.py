"""Web app: import a player, inspect their profile, and play Nemesis with a risk dial."""

from __future__ import annotations

import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import chess
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from nemesis.analysis.stockfish import StockfishAnalyzer
from nemesis import commentary
from nemesis.engine import Nemesis
from nemesis.search import Decision
from nemesis.features import STYLE_TEXT
from nemesis.learn import record_game, train
from nemesis.store import DEFAULT_DB, Store

STATIC = Path(__file__).parent / "static"


class ImportReq(BaseModel):
    source: str               # lichess | chesscom | pgn
    username: str
    max_games: int = 200
    pgn: str | None = None


class NewGameReq(BaseModel):
    player: str
    color: str = "white"      # the human's colour
    risk: float = 0.5
    deep: bool = False
    commentary: str = "xray"  # xray | poker


class MoveReq(BaseModel):
    uci: str


class RiskReq(BaseModel):
    risk: float


class CommentaryReq(BaseModel):
    mode: str


@dataclass
class Job:
    id: str
    kind: str
    player: str
    status: str = "running"   # running | done | error
    message: str = ""
    progress: float = 0.0


@dataclass
class Game:
    id: str
    player: str
    human: chess.Color
    engine: Nemesis
    board: chess.Board = field(default_factory=chess.Board)
    log: list[dict] = field(default_factory=list)
    finished: bool = False
    result: str | None = None
    pending: Decision | None = None   # Nemesis's last decision, kept to react to the human's reply


def create_app(db: str | Path = DEFAULT_DB, depth: int = 10) -> FastAPI:
    store = Store(db)
    analyzer = StockfishAnalyzer(store, depth=depth)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        analyzer.close()

    app = FastAPI(title="Nemesis", lifespan=lifespan)
    jobs: dict[str, Job] = {}
    games: dict[str, Game] = {}

    def run_job(job: Job, fn) -> None:
        def target():
            try:
                fn(job)
                job.status, job.progress = "done", 1.0
            except Exception as exc:  # surfaced to the UI
                traceback.print_exc()
                job.status, job.message = "error", str(exc)
        threading.Thread(target=target, daemon=True).start()

    def train_job(job: Job) -> None:
        def progress(done: int, total: int) -> None:
            job.progress = 0.1 + 0.9 * done / max(total, 1)
            job.message = f"analysing game {done}/{total}"
        _, metrics = train(store, job.player, analyzer, progress=progress)
        job.message = f"trained on {metrics['samples']} moves from {metrics['games']} games"
        for g in games.values():
            if g.player == job.player:
                g.engine.reload()

    def state(g: Game) -> dict:
        legal: dict[str, list[str]] = {}
        if not g.finished and g.board.turn == g.human:
            for m in g.board.legal_moves:
                legal.setdefault(chess.square_name(m.from_square), []).append(m.uci())
        return {
            "id": g.id, "player": g.player, "fen": g.board.fen(),
            "human": "white" if g.human == chess.WHITE else "black",
            "turn": "white" if g.board.turn == chess.WHITE else "black",
            "last": g.board.peek().uci() if g.board.move_stack else None,
            "legal": legal, "log": g.log, "finished": g.finished, "result": g.result,
            "risk": g.engine.risk, "check": g.board.is_check(), "commentary": g.engine.mode,
        }

    def finish(g: Game, result: str) -> None:
        g.finished, g.result = True, result
        record_game(store, g.player, g.human, [m.uci() for m in g.board.move_stack], result=result)
        job = Job(id=uuid.uuid4().hex[:8], kind="learn", player=g.player)
        jobs[job.id] = job
        g.log.append({"who": "nemesis", "kind": "info", "lines": [f"Game over ({result}). I'm studying this one."]})
        run_job(job, train_job)

    def nemesis_turn(g: Game) -> None:
        if g.board.is_game_over(claim_draw=True):
            finish(g, g.board.result(claim_draw=True))
            return
        decision = g.engine.choose(g.board)
        san = g.board.san(decision.move)
        g.board.push(decision.move)
        g.pending = decision
        entry = {"who": "nemesis", "kind": "move", "san": san, "lines": decision.lines}
        if g.engine.mode == commentary.XRAY:  # Poker Face keeps its numbers to itself
            entry["candidates"] = [{"san": c.san, "objective": round(c.objective, 3),
                                    "exploit": round(c.exploit, 3)} for c in decision.candidates[:6]]
        g.log.append(entry)
        if g.board.is_game_over(claim_draw=True):
            finish(g, g.board.result(claim_draw=True))

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/players")
    def players() -> list[dict]:
        return [{k: v for k, v in p.items() if k != "metrics"} for p in store.players()]

    @app.post("/api/import")
    def import_games(req: ImportReq) -> dict:
        player = req.username.strip()
        if not player:
            raise HTTPException(400, "username required")
        job = Job(id=uuid.uuid4().hex[:8], kind="import", player=player.lower())
        jobs[job.id] = job

        def work(job: Job) -> None:
            job.message = "downloading games"
            if req.source == "lichess":
                from nemesis.ingest.lichess import import_user
                n = import_user(store, player, req.max_games)
            elif req.source == "chesscom":
                from nemesis.ingest.chesscom import import_user
                n = import_user(store, player, req.max_games)
            elif req.source == "pgn" and req.pgn:
                from nemesis.ingest.pgn import import_pgn_text
                n = import_pgn_text(store, req.pgn, player)
            else:
                raise ValueError("unknown source or empty PGN")
            if not store.games(player):
                raise ValueError(f"no games found for {player}")
            job.message, job.progress = f"imported {n} new games", 0.1
            train_job(job)

        run_job(job, work)
        return {"job": job.id}

    @app.post("/api/train/{player}")
    def retrain(player: str) -> dict:
        job = Job(id=uuid.uuid4().hex[:8], kind="train", player=player.lower())
        jobs[job.id] = job
        run_job(job, train_job)
        return {"job": job.id}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.__dict__

    @app.get("/api/profile/{player}")
    def profile(player: str) -> dict:
        found = store.load_model(player.lower())
        if not found or "profile" not in found[1]:
            raise HTTPException(404, "no profile yet: import and train first")
        prof = found[1]["profile"]
        for t in prof["traits"]:  # profiles saved by older versions may carry stale wording
            t["text"] = STYLE_TEXT.get(t["feature"], t["text"])
        prof["weakest_openings"] = [o for o in prof["weakest_openings"] if o["opening"] != "Unknown"]
        return prof | {"metrics": {k: v for k, v in found[1].items() if k != "profile"}}

    @app.post("/api/games")
    def new_game(req: NewGameReq) -> dict:
        engine = Nemesis(store, req.player, risk=req.risk, deep=req.deep, analyzer=analyzer,
                         mode=req.commentary)
        human = chess.WHITE if req.color == "white" else chess.BLACK
        g = Game(id=uuid.uuid4().hex[:10], player=req.player.lower(), human=human, engine=engine)
        games[g.id] = g
        if g.board.turn != human:
            nemesis_turn(g)
        return state(g)

    def get_game(game_id: str) -> Game:
        g = games.get(game_id)
        if g is None:
            raise HTTPException(404, "no such game")
        return g

    @app.get("/api/games/{game_id}")
    def game_state(game_id: str) -> dict:
        return state(get_game(game_id))

    @app.post("/api/games/{game_id}/move")
    def play_move(game_id: str, req: MoveReq) -> dict:
        g = get_game(game_id)
        if g.finished or g.board.turn != g.human:
            raise HTTPException(409, "not your turn")
        try:
            move = chess.Move.from_uci(req.uci)
        except ValueError:
            raise HTTPException(400, "bad move") from None
        if move not in g.board.legal_moves:
            raise HTTPException(400, "illegal move")
        g.log.append({"who": "you", "san": g.board.san(move)})
        if g.pending is not None:
            g.log.append({"who": "nemesis", "kind": "reaction",
                          "lines": g.engine.react(g.board, g.pending, move)})
        g.board.push(move)
        nemesis_turn(g)
        return state(g)

    @app.post("/api/games/{game_id}/risk")
    def set_risk(game_id: str, req: RiskReq) -> dict:
        g = get_game(game_id)
        g.engine.risk = max(0.0, min(1.0, req.risk))
        return state(g)

    @app.post("/api/games/{game_id}/commentary")
    def set_commentary(game_id: str, req: CommentaryReq) -> dict:
        g = get_game(game_id)
        g.engine.mode = commentary.normalise_mode(req.mode)
        return state(g)

    @app.post("/api/games/{game_id}/resign")
    def resign(game_id: str) -> dict:
        g = get_game(game_id)
        if not g.finished:
            finish(g, "0-1" if g.human == chess.WHITE else "1-0")
        return state(g)

    return app
