"""A profile per time control, used only where it proves itself.

Most people play differently in bullet and in daily games, but a separate model per time
control only helps if it predicts that time control's games better than your main model
does. So for each time control with enough games, Nemesis fits a profile that starts from
your main one, tests both on that time control's newest games, and uses the time-control
profile only if it wins. Separately, it lists the ways your play really differs by time
control: differences bigger than chance, by a permutation test over your games.
"""

from __future__ import annotations

import io
import random

import chess.pgn
import numpy as np

from nemesis import timecontrol
from nemesis.features import INDEX
from nemesis.profile.move_model import MoveModel, Sample, evaluate
from nemesis.store import Store

MIN_GAMES = 10       # a time control needs this many of your games to get its own profile
TEST_FRAC = 0.25     # its newest games test it against your main profile
L2 = 64.0            # stays close to your main profile unless its games clearly differ
MARGIN = 0.01        # must beat the main profile's log loss by this much to be used
PERMUTATIONS = 300
SIGNIFICANCE = 0.01  # several comparisons are made, so only strong differences count


def key(player: str, group: str) -> str:
    return f"{player.lower()}@{group}"


def game_groups(store: Store, samples: list[Sample]) -> dict[str, str]:
    """{game_id: time-control group} for the games behind these samples."""
    out = {}
    for gid in dict.fromkeys(s.game_id for s in samples):
        game = store.game(gid)
        headers = chess.pgn.read_headers(io.StringIO(game.pgn)) if game else None
        cat = timecontrol.pgn_category(headers.get("TimeControl") if headers else None)
        out[gid] = timecontrol.profile_group(cat)
    return out


def load(store: Store, player: str, category: str | None) -> MoveModel | None:
    """The time-control profile for this kind of game, if it earned its place."""
    found = store.load_model(key(player, timecontrol.profile_group(category)))
    if found and found[1].get("active"):
        return MoveModel.from_dict(found[0])
    return None


def train_groups(store: Store, player: str, samples: list[Sample], base: MoveModel,
                 prior: np.ndarray, l2: float = 4.0) -> dict:
    """Fit, test and save a profile per time control. Returns what happened, per group."""
    groups = game_groups(store, samples)
    report = {}
    for group in timecontrol.PROFILE_GROUPS:
        ids = [gid for gid in dict.fromkeys(s.game_id for s in samples) if groups[gid] == group]
        entry: dict = {"games": len(ids), "label": timecontrol.PROFILE_LABELS[group], "active": False}
        if len(ids) < MIN_GAMES:
            entry["reason"] = f"needs {MIN_GAMES} games"
            store.save_model(key(player, group), base.to_dict(), entry)   # inactive placeholder
            report[group] = entry
            continue
        test_ids = set(ids[len(ids) - max(3, int(len(ids) * TEST_FRAC)):])
        test = [s for s in samples if s.game_id in test_ids]
        # Your main profile, fitted without these test games, against one specialised on
        # this time control's other games.
        main = MoveModel().fit([s for s in samples if s.game_id not in test_ids], prior=prior, l2=l2)
        own = MoveModel().fit([s for s in samples if groups[s.game_id] == group and s.game_id not in test_ids],
                              prior=main.w, l2=L2)
        m_main, m_own = evaluate(main, test), evaluate(own, test)
        entry["holdout"] = {"main": {k: round(m_main[k], 4) for k in ("top1", "top3", "nll")},
                            "own": {k: round(m_own[k], 4) for k in ("top1", "top3", "nll")},
                            "moves": m_main["n"], "test_games": len(test_ids)}
        entry["active"] = bool(m_own["nll"] < m_main["nll"] - MARGIN)
        if not entry["active"]:
            entry["reason"] = "a separate one didn't predict these games better"
        final = MoveModel().fit([s for s in samples if groups[s.game_id] == group], prior=base.w, l2=L2)
        store.save_model(key(player, group), final.to_dict(), entry)
        report[group] = entry
    return report


# ---------------------------------------------------------------------------
# How you play differently by time control
# ---------------------------------------------------------------------------
MEASURES = [  # (name, value of your move, how to say it, format)
    ("best", lambda s: s.X[s.chosen, INDEX["is_best"]], "play the engine's best move", "pct"),
    ("loss", lambda s: min(s.loss, 800) / 100, "lose on average {} a move", "pawns"),
    ("blunder", lambda s: float(s.loss >= 300), "blunder", "pct"),
    ("capture", lambda s: s.X[s.chosen, INDEX["capture"]], "capture", "pct"),
    ("trade", lambda s: s.X[s.chosen, INDEX["trade"]], "trade pieces", "pct"),
    ("check", lambda s: s.X[s.chosen, INDEX["check"]], "give check", "pct"),
    ("retreat", lambda s: s.X[s.chosen, INDEX["retreat"]], "retreat", "pct"),
    ("hangs_piece", lambda s: s.X[s.chosen, INDEX["hangs_piece"]], "leave a piece hanging", "pct"),
    ("ignores_threat", lambda s: s.X[s.chosen, INDEX["ignores_threat"]], "miss a threat", "pct"),
]


def _fmt(kind: str, value: float) -> str:
    return f"{value:.0%}" if kind == "pct" else f"{value:.2f} pawns"


def differences(store: Store, samples: list[Sample], seed: int = 0) -> dict:
    """Per time control: how your moves there differ from your other games, where the
    difference is bigger than chance (shuffling which games are which time control)."""
    groups = game_groups(store, samples)
    gids = list(dict.fromkeys(s.game_id for s in samples))
    # Per game: (number of moves, sum of each measure) - the permutation shuffles whole games.
    sums = {g: np.zeros(len(MEASURES)) for g in gids}
    counts = {g: 0 for g in gids}
    for s in samples:
        sums[s.game_id] += [f(s) for _, f, _, _ in MEASURES]
        counts[s.game_id] += 1
    S = np.array([sums[g] for g in gids])
    N = np.array([counts[g] for g in gids], dtype=float)
    labels = np.array([groups[g] for g in gids])
    rng = random.Random(seed)
    out = {}
    for group in timecontrol.PROFILE_GROUPS:
        inside = labels == group
        if inside.sum() < 8 or (~inside).sum() < 8:
            continue

        def diff(mask):
            return S[mask].sum(0) / N[mask].sum() - S[~mask].sum(0) / N[~mask].sum()
        observed = diff(inside)
        exceed = np.zeros(len(MEASURES))
        idx = list(range(len(gids)))
        for _ in range(PERMUTATIONS):
            rng.shuffle(idx)
            mask = np.zeros(len(gids), dtype=bool)
            mask[idx[:inside.sum()]] = True
            exceed += np.abs(diff(mask)) >= np.abs(observed)
        p = (exceed + 1) / (PERMUTATIONS + 1)
        here = S[inside].sum(0) / N[inside].sum()
        there = S[~inside].sum(0) / N[~inside].sum()
        notes = []
        for i, (name, _, text, kind) in enumerate(MEASURES):
            if p[i] < SIGNIFICANCE:
                if kind == "pct":
                    said = f"you {text} {_fmt(kind, here[i])} of the time ({_fmt(kind, there[i])} in your other games)"
                else:
                    said = f"you {text.format(_fmt(kind, here[i]))} ({_fmt(kind, there[i])} in your other games)"
                notes.append({"measure": name, "text": said, "p": round(float(p[i]), 4),
                              "here": round(float(here[i]), 4), "there": round(float(there[i]), 4)})
        out[group] = {"label": timecontrol.PROFILE_LABELS[group], "games": int(inside.sum()),
                      "best": round(float(here[0]), 3), "loss": round(float(here[1]), 3),
                      "blunder": round(float(here[2]), 3), "notes": notes}
    # With two time controls to compare, each difference shows up twice, mirrored: say it
    # once, for the one with fewer games, naming the other when it's all the other games.
    if len(out) == 2:
        small, big = sorted(out, key=lambda g: out[g]["games"])
        if len({str(x) for x in labels}) == 2:
            other = timecontrol.PROFILE_LABELS[big].lower()
            for note in out[small]["notes"]:
                note["text"] = note["text"].replace("in your other games", f"in your {other} games")
        out[big]["notes"] = []
    return out
