"""Human-readable style profile: what the model learned, in words and numbers."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from nemesis.features import INDEX, STYLE_TEXT
from nemesis.profile.move_model import MoveModel, Sample, prior_vector
from nemesis.store import Store


def _rate(samples: list[Sample], pred) -> float | None:
    hits = [pred(s) for s in samples]
    return round(float(np.mean(hits)), 3) if hits else None


def _chosen(s: Sample, name: str) -> float:
    return float(s.X[s.chosen, INDEX[name]])


def style_profile(store: Store, player: str, model: MoveModel, samples: list[Sample]) -> dict:
    player = player.lower()
    games = store.games(player)
    prior = prior_vector()
    pop = store.load_model("__population__")
    if pop:
        prior = MoveModel.from_dict(pop[0]).w

    # Traits: where this player's weights differ most from the population.
    traits = []
    for name, text in STYLE_TEXT.items():
        delta = float(model.w[INDEX[name]] - prior[INDEX[name]])
        traits.append({"feature": name, "text": text, "delta": round(delta, 3),
                       "weight": round(float(model.w[INDEX[name]]), 3)})
    traits.sort(key=lambda t: -abs(t["delta"]))

    by_phase = {}
    for phase in ("opening", "middlegame", "endgame"):
        ps = [s for s in samples if s.phase == phase]
        if ps:
            by_phase[phase] = {
                "moves": len(ps),
                "avg_cp_loss": round(float(np.mean([min(s.loss, 800) for s in ps])), 1),
                "blunder_rate": _rate(ps, lambda s: s.loss >= 300),
            }
    tp = [s for s in samples if s.time_pressure]
    calm = [s for s in samples if not s.time_pressure]

    style = {
        "aggression": _rate(samples, lambda s: max(_chosen(s, "capture"), _chosen(s, "check"),
                                                    _chosen(s, "near_king"), _chosen(s, "pawn_storm"))),
        "trade_appetite": _rate(samples, lambda s: _chosen(s, "trade")),
        "sacrifice_rate": _rate(samples, lambda s: _chosen(s, "sacrifice")),
        "retreat_rate": _rate(samples, lambda s: _chosen(s, "retreat")),
        "best_move_rate": _rate(samples, lambda s: _chosen(s, "is_best")),
    }
    blind_spots = {
        "hangs_piece": _rate(samples, lambda s: _chosen(s, "hangs_piece")),
        "ignores_threat": _rate(samples, lambda s: _chosen(s, "ignores_threat")),
        "allows_fork": _rate(samples, lambda s: _chosen(s, "allows_fork")),
    }
    clock = {
        "blunder_rate_time_pressure": _rate(tp, lambda s: s.loss >= 300),
        "blunder_rate_normal": _rate(calm, lambda s: s.loss >= 300),
        "moves_in_time_pressure": len(tp),
    }

    openings: dict[str, list[float]] = defaultdict(list)
    for g in games:
        if g.result is not None:
            openings[(g.opening or "Unknown").split(":")[0]].append(g.result)
    opening_rows = sorted(
        ({"opening": k, "games": len(v), "score": round(sum(v) / len(v), 3)} for k, v in openings.items()),
        key=lambda r: (-r["games"], r["score"]),
    )
    named = (r for r in opening_rows if r["games"] >= 3 and r["opening"] != "Unknown")
    weak = sorted(named, key=lambda r: r["score"])[:5]

    results = [g.result for g in games if g.result is not None]
    found = store.load_model(player)
    return {
        "player": player,
        "games": len(games),
        "score": round(sum(results) / len(results), 3) if results else None,
        "moves_analysed": len(samples),
        "traits": traits[:8],
        "style": style,
        "blind_spots": blind_spots,
        "by_phase": by_phase,
        "clock": clock,
        "openings": opening_rows[:15],
        "weakest_openings": weak,
        "metrics": found[1] if found else None,
        "weights": model.to_dict(),
    }
