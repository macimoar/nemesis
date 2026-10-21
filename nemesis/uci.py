"""UCI front-end, so Nemesis can be loaded into any chess GUI (Arena, Cute Chess, En Croissant...)."""

from __future__ import annotations

import sys
import traceback
from typing import TextIO

import chess

from nemesis import commentary, timecontrol
from nemesis.engine import Nemesis
from nemesis.search import Decision
from nemesis.learn import record_game
from nemesis.store import DEFAULT_DB, Store
from nemesis.timecontrol import TimeControl

OPTIONS = [
    "option name Player type string default <empty>",
    "option name Risk type spin default 50 min 0 max 100",
    "option name Depth type spin default 10 min 4 max 30",
    "option name Deep type check default false",
    "option name Learn type check default true",
    "option name StockfishPath type string default <empty>",
    "option name Commentary type combo default X-Ray var X-Ray var Poker Face",
    "option name Elo type spin default 0 min 0 max 2400",
]


class UCI:
    def __init__(self, out: TextIO = sys.stdout, db: str | None = None):
        self.out = out
        self.db = db or str(DEFAULT_DB)
        self.opts = {"player": "", "risk": "50", "depth": "10", "deep": "false", "learn": "true",
                     "stockfishpath": "", "commentary": "X-Ray", "elo": "0"}
        self.engine: Nemesis | None = None
        self.store: Store | None = None
        self.start_fen = chess.STARTING_FEN
        self.moves: list[str] = []
        self.board = chess.Board()
        self.nemesis_color: chess.Color | None = None
        self.last: tuple[list[str], Decision] | None = None   # moves after Nemesis's last move, and why
        self.annotations: list[dict] = []                      # what Nemesis was thinking, for review
        self.game_tc: TimeControl | None = None                # from the GUI's clocks, if timed

    def send(self, line: str) -> None:
        self.out.write(line + "\n")
        self.out.flush()

    def ensure_engine(self) -> Nemesis:
        if self.engine is None:
            self.store = Store(self.db)
            self.engine = Nemesis(
                self.store, self.opts["player"] or "anonymous", risk=int(self.opts["risk"]) / 100,
                depth=int(self.opts["depth"]), deep=self.opts["deep"] == "true",
                stockfish_path=self.opts["stockfishpath"] or None, mode=self.opts["commentary"],
                elo=int(self.opts["elo"] or 0) or None,
            )
        return self.engine

    def set_option(self, tokens: list[str]) -> None:
        text = " ".join(tokens)
        if " value " in text:
            name, value = text.split(" value ", 1)
        else:
            name, value = text, ""
        name = name.replace("name", "", 1).strip().lower()
        value = "" if value.strip() == "<empty>" else value.strip()
        if name not in self.opts:
            return
        self.opts[name] = value.lower() if name in ("deep", "learn") else value
        if self.engine is None:
            return
        if name == "player":
            self.engine.set_player(value or "anonymous")
        elif name == "risk":
            self.engine.risk = int(value) / 100
        elif name == "deep":
            self.engine.deep = value.lower() == "true"
        elif name == "commentary":
            self.engine.mode = commentary.normalise_mode(value)
        elif name == "elo":
            self.engine.fixed_elo = int(value or 0) or None
        elif name in ("depth", "stockfishpath"):
            self.engine.close()
            self.engine = None

    def finish_game(self) -> None:
        """Save the game just played so the player's model can learn from it."""
        if (self.opts["learn"] == "true" and self.opts["player"] and self.nemesis_color is not None
                and len(self.moves) >= 10 and self.store is not None):
            result = self.board.result(claim_draw=True)
            record_game(self.store, self.opts["player"], not self.nemesis_color, self.moves,
                        result=None if result == "*" else result, start_fen=self.start_fen,
                        risk=int(self.opts["risk"]) / 100, annotations=self.annotations,
                        tc=self.game_tc)
            self.send("info string game recorded; run `nemesis train` to update the model")
        self.moves, self.nemesis_color = [], None
        self.last = None
        self.annotations = []
        self.game_tc = None
        if self.engine:
            self.engine.tc = "untimed"

    def position(self, tokens: list[str]) -> None:
        if tokens[0] == "startpos":
            fen, rest = chess.STARTING_FEN, tokens[1:]
        else:
            fen, rest = " ".join(tokens[1:7]), tokens[7:]
        moves = rest[1:] if rest and rest[0] == "moves" else []
        # A position that doesn't continue the current game means a new game started.
        if fen != self.start_fen or moves[:len(self.moves)] != self.moves:
            self.finish_game()
        self.start_fen, self.moves = fen, moves
        self.board = chess.Board(fen)
        for u in moves:
            self.board.push_uci(u)

    def go(self, tokens: list[str]) -> None:
        try:
            self.search(tokens)
        except Exception as exc:
            # Never die mid-game: tell the GUI why and still answer with a legal move.
            traceback.print_exc(file=sys.stderr)
            self.send(f"info string error: {exc}")
            fallback = next(iter(self.board.legal_moves), None)
            self.send(f"bestmove {fallback.uci() if fallback else '0000'}")

    def time_budget(self, args: dict) -> float | None:
        """How long to think, from the GUI's clock info. Also notes the game's time control
        (from the first timed `go`), so the result goes to the right rating."""
        def secs(key: str) -> float | None:
            v = args.get(key)
            return int(v) / 1000 if v and v.lstrip("-").isdigit() else None
        if (movetime := secs("movetime")) is not None:
            return movetime
        white = self.board.turn == chess.WHITE
        own, inc = secs("wtime" if white else "btime"), secs("winc" if white else "binc") or 0.0
        if own is None:
            return None
        if self.game_tc is None:
            self.game_tc = TimeControl(round(own), inc)
            if self.engine:
                self.engine.tc = self.game_tc.category
        return timecontrol.think_budget(own, inc)

    def search(self, tokens: list[str]) -> None:
        engine = self.ensure_engine()
        self.nemesis_color = self.board.turn
        args = dict(zip(tokens[::2], tokens[1::2]))
        human_ms = args.get("btime" if self.board.turn == chess.WHITE else "wtime")
        human_clock = int(human_ms) / 1000 if human_ms and human_ms.lstrip("-").isdigit() else None
        budget = self.time_budget(args)
        if self.last and self.moves[:-1] == self.last[0] and len(self.moves) == len(self.last[0]) + 1:
            before = chess.Board(self.start_fen)
            for u in self.last[0]:
                before.push_uci(u)
            for line in engine.react(before, self.last[1], chess.Move.from_uci(self.moves[-1])):
                self.send(f"info string {line.replace('**', '')}")
            if self.annotations and self.last[1].reply:
                self.annotations[-1]["reply"] = self.last[1].reply
        decision = engine.choose(self.board, human_clock, budget=budget)
        best = decision.best
        self.send(f"info depth {engine.depth} score cp {best.cp} pv {best.uci}")
        for line in decision.lines:  # UCI info strings can't contain newlines: one per line
            self.send(f"info string {line.replace('**', '')}")
        self.last = (self.moves + [decision.move.uci()], decision)
        self.annotations.append(decision.annotation(len(self.moves), engine.risk))
        self.send(f"bestmove {decision.move.uci()}")

    def handle(self, line: str) -> bool:
        tokens = line.strip().split()
        if not tokens:
            return True
        cmd, args = tokens[0], tokens[1:]
        if cmd == "uci":
            self.send("id name Nemesis")
            self.send("id author nemesis")
            for opt in OPTIONS:
                self.send(opt)
            self.send("uciok")
        elif cmd == "isready":
            self.send("readyok")
        elif cmd == "setoption":
            self.set_option(args)
        elif cmd == "ucinewgame":
            self.finish_game()
            if self.engine:
                self.engine.reload()
        elif cmd == "position" and args:
            self.position(args)
        elif cmd == "go":
            self.go(args)
        elif cmd == "quit":
            self.finish_game()
            if self.engine:
                self.engine.close()
            return False
        return True


def main(db: str | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):  # Windows pipes default to a legacy code page
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    uci = UCI(db=db)
    for line in sys.stdin:
        if not uci.handle(line):
            break
