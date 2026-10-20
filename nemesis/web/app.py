"""Web app: import a player, inspect their profile, and play Nemesis with a risk dial."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import chess
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from nemesis.analysis.install import ensure_stockfish
from nemesis.analysis.stockfish import StockfishAnalyzer
from nemesis import commentary, ratings
from nemesis.engine import Nemesis
from nemesis.search import Decision
from nemesis.features import STYLE_TEXT
from nemesis.learn import record_game, train
from nemesis.review import cached_review, rating_changes, review_game
from nemesis.store import DEFAULT_DB, Store

STATIC = Path(__file__).parent / "static"


def _time_left(started: float, done: int, total: int) -> str:
    """' about 3 minutes left', from the pace so far."""
    if done < 3:
        return ""
    left = (time.monotonic() - started) / done * (total - done)
    if left < 60:
        return " less than a minute left"
    minutes = round(left / 60)
    return f" about {minutes} minute{'s' if minutes != 1 else ''} left"


def _quietly(fn) -> None:
    """Run a warm-up step; its failure is reported through /api/setup instead."""
    try:
        fn()
    except Exception:
        pass


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
    elo: int | None = None    # Nemesis's strength; None/0 = "match me" (your rating + 50)


class MoveReq(BaseModel):
    uci: str


class RiskReq(BaseModel):
    risk: float


class CommentaryReq(BaseModel):
    mode: str


class EloReq(BaseModel):
    elo: int | None = None


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
    annotations: list[dict] = field(default_factory=list)   # what Nemesis was thinking, for review
    game_id: str | None = None        # id in the store once finished


def create_app(db: str | Path = DEFAULT_DB, depth: int = 10) -> FastAPI:
    store = Store(db)
    # Stockfish starts lazily: on a fresh computer it may need downloading first.
    setup = {"stockfish": "checking", "message": "", "progress": 0.0}
    holder: dict[str, StockfishAnalyzer] = {}
    ready = threading.Lock()

    def get_analyzer() -> StockfishAnalyzer:
        with ready:
            if "a" not in holder:
                def progress(msg: str, frac: float) -> None:
                    setup.update(stockfish="downloading", message=msg, progress=frac)
                try:
                    path = ensure_stockfish(progress)
                    holder["a"] = StockfishAnalyzer(store, path=path, depth=depth)
                except Exception as exc:
                    setup.update(stockfish="error", message=str(exc))
                    raise
                setup.update(stockfish="ready", message="", progress=1.0, engine=path)
            return holder["a"]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        threading.Thread(target=lambda: _quietly(get_analyzer), daemon=True).start()
        yield
        if "a" in holder:
            holder["a"].close()

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
        started = time.monotonic()

        def progress(done: int, total: int) -> None:
            job.progress = 0.1 + 0.9 * done / max(total, 1)
            job.message = f"Studying game {done} of {total}…{_time_left(started, done, total)}"
        if setup["stockfish"] != "ready":
            job.message = "Getting Stockfish ready (first launch only)…"
        analyzer = get_analyzer()
        started = time.monotonic()
        _, metrics = train(store, job.player, analyzer, progress=progress)
        job.message = f"Done: studied {metrics['samples']} of your moves from {metrics['games']} games."
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
            "tier": ratings.tier(g.engine.risk), "elo": round(g.engine.target_elo()),
            "fixed_elo": g.engine.fixed_elo, "ratings": ratings.summary(store, g.player),
            "game_id": g.game_id,
        }

    def finish(g: Game, result: str) -> None:
        g.finished, g.result = True, result
        # Rate against the tier Nemesis mostly played at (the risk can change mid-game).
        risks = [a["risk"] for a in g.annotations] or [g.engine.risk]
        g.game_id = record_game(store, g.player, g.human, [m.uci() for m in g.board.move_stack],
                                result=result, risk=sum(risks) / len(risks), annotations=g.annotations)
        job = Job(id=uuid.uuid4().hex[:8], kind="learn", player=g.player)
        jobs[job.id] = job
        lines = [f"Game over ({result}). I'm studying this one."]
        change = store.rating_change(g.game_id) if g.game_id else {}
        for entity, (before, after) in change.items():
            name = "Your rating" if entity.startswith("player:") else \
                f"My rating ({ratings.TIER_LABEL[entity.rsplit(':', 1)[-1]]})"
            lines.append(f"{name}: {before:.0f} → {after:.0f} ({after - before:+.0f}).")
        g.log.append({"who": "nemesis", "kind": "info", "lines": lines})
        run_job(job, train_job)

    def nemesis_turn(g: Game) -> None:
        if g.board.is_game_over(claim_draw=True):
            finish(g, g.board.result(claim_draw=True))
            return
        decision = g.engine.choose(g.board)
        san = g.board.san(decision.move)
        g.annotations.append(decision.annotation(len(g.board.move_stack), g.engine.risk))
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

    @app.get("/api/health")
    def health() -> dict:
        return {"app": "nemesis"}

    @app.get("/api/setup")
    def setup_state() -> dict:
        return setup | {"players": len(store.players())}

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
            site = {"lichess": "Lichess", "chesscom": "Chess.com"}.get(req.source, "the PGN")
            job.message = f"Downloading your games from {site}…"
            try:
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
                    raise ValueError("Paste some PGN text first.")
            except httpx.HTTPStatusError as exc:  # explain in plain words, not HTTP jargon
                if exc.response.status_code == 404:
                    raise ValueError(f"We couldn't find a {site} player called “{player}”. "
                                     "Check the spelling.") from None
                if exc.response.status_code == 429:
                    raise ValueError(f"{site} asked us to slow down. Wait a minute and try again.") from None
                raise ValueError(f"{site} had a problem ({exc.response.status_code}). Try again later.") from None
            except httpx.TransportError:
                raise ValueError(f"Couldn't reach {site}. Are you connected to the internet?") from None
            if not store.games(player):
                raise ValueError(f"{site} has no standard chess games for “{player}” yet.")
            job.message, job.progress = f"Got {n} new games. Getting ready to study them…", 0.1
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
        engine = Nemesis(store, req.player, risk=req.risk, deep=req.deep, analyzer=get_analyzer(),
                         mode=req.commentary, elo=req.elo or None)
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

    @app.post("/api/games/{game_id}/elo")
    def set_elo(game_id: str, req: EloReq) -> dict:
        g = get_game(game_id)
        g.engine.fixed_elo = req.elo or None
        return state(g)

    @app.post("/api/games/{game_id}/resign")
    def resign(game_id: str) -> dict:
        g = get_game(game_id)
        if not g.finished:
            finish(g, "0-1" if g.human == chess.WHITE else "1-0")
        return state(g)

    @app.get("/api/ratings/{player}")
    def player_ratings(player: str) -> dict:
        return ratings.summary(store, player.lower())

    @app.get("/api/games/list/{player}")
    def list_games(player: str) -> list[dict]:
        out = []
        for gr in reversed(store.games(player.lower())):
            change = store.rating_change(gr.id).get(ratings.player_entity(player))
            out.append({
                "id": gr.id, "date": gr.played_at, "source": gr.source, "opening": gr.opening,
                "opponent": "Nemesis" if gr.source == "nemesis" else gr.opponent,
                "color": "white" if gr.color else "black", "result": gr.result,
                "rating_change": round(change[1] - change[0]) if change else None,
                "reviewed": cached_review(store, gr.id, depth) is not None,
            })
        return out

    @app.post("/api/review/{game_id:path}")
    def start_review(game_id: str) -> dict:
        game = store.game(game_id)
        if game is None:
            raise HTTPException(404, "no such game")
        job = Job(id=uuid.uuid4().hex[:8], kind="review", player=game.player)
        jobs[job.id] = job

        def work(job: Job) -> None:
            def progress(done: int, total: int) -> None:
                job.progress = done / max(total, 1)
                job.message = f"analysing move {done}/{total}"
            if cached_review(store, game_id, depth) is None:
                review_game(store, get_analyzer(), game_id, progress=progress)
            job.message = "review ready"

        run_job(job, work)
        return {"job": job.id}

    @app.get("/api/review/{game_id:path}")
    def get_review(game_id: str) -> dict:
        rev = cached_review(store, game_id, depth)
        if rev is None:
            return {"ready": False}
        rev["ratings"] = rating_changes(store, game_id)
        return rev

    return app
