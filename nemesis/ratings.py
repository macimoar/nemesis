"""Elo ratings: yours, and Nemesis's for each risk tier (Nemesis-vs-you specifically).

Your rating is seeded from the rating recorded in your imported games; only games
against Nemesis move it. Each Nemesis tier starts at your rating + 50, the same
target its play is limited to, and from then on is earned from results.
"""

from __future__ import annotations

import io

import chess.pgn

from nemesis.store import Store

DEFAULT_RATING = 1200
EDGE = 50            # "match me": Nemesis plays this much stronger than you
PROVISIONAL_GAMES = 10

TIERS = [  # (upper risk bound, key, label): matches the risk slider's wording
    (0.2, "sound", "Sound"),
    (0.6, "steering", "Steering"),
    (0.85, "trapper", "Trapper"),
    (1.01, "hustler", "Hustler"),
]
TIER_LABEL = {key: label for _, key, label in TIERS}


def tier(risk: float) -> str:
    for bound, key, _ in TIERS:
        if risk < bound:
            return key
    return TIERS[-1][1]


def expected(a: float, b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((b - a) / 400))


def k_factor(games: int) -> float:
    return 40.0 if games < PROVISIONAL_GAMES else 20.0


def player_entity(player: str) -> str:
    return f"player:{player.lower()}"


def nemesis_entity(player: str, tier_key: str) -> str:
    return f"nemesis:{player.lower()}:{tier_key}"


def seed_rating(store: Store, player: str) -> float:
    """The player's rating from their most recent imported game that records one."""
    for game in reversed(store.games(player)):
        if game.source == "nemesis":
            continue
        headers = chess.pgn.read_headers(io.StringIO(game.pgn))
        value = headers.get("WhiteElo" if game.color else "BlackElo", "") if headers else ""
        if value.isdigit() and int(value) > 0:
            return float(value)
    return float(DEFAULT_RATING)


def player_rating(store: Store, player: str) -> tuple[float, int]:
    """(rating, rated games). Until you've played Nemesis, the seed follows your imports."""
    found = store.get_rating(player_entity(player))
    if found and found[1] > 0:
        return found
    return seed_rating(store, player), 0


def nemesis_rating(store: Store, player: str, tier_key: str) -> tuple[float, int]:
    found = store.get_rating(nemesis_entity(player, tier_key))
    if found:
        return found
    return player_rating(store, player)[0] + EDGE, 0


def target_elo(store: Store, player: str, fixed: float | None = None) -> float:
    """The strength Nemesis plays at: fixed, or "match me" (your rating + EDGE)."""
    if fixed:
        return float(fixed)
    return player_rating(store, player)[0] + EDGE


def update_after_game(store: Store, game_id: str, player: str, risk: float, score: float) -> dict | None:
    """Rate a finished game. `score` is the player's (1, 0.5, 0). Idempotent per game."""
    t = tier(risk)
    me, me_games = player_rating(store, player)
    nem, nem_games = nemesis_rating(store, player, t)
    exp = expected(me, nem)
    me_after = me + k_factor(me_games) * (score - exp)
    nem_after = nem + k_factor(nem_games) * ((1 - score) - (1 - exp))
    changes = {
        player_entity(player): (me, me_after, me_games + 1),
        nemesis_entity(player, t): (nem, nem_after, nem_games + 1),
    }
    if not store.apply_rating_changes(game_id, changes):
        return None
    return {"tier": t, "player": (me, me_after), "nemesis": (nem, nem_after)}


def summary(store: Store, player: str) -> dict:
    """Everything the UI shows: you, each tier, recent history."""
    me, games = player_rating(store, player)
    tiers = []
    for _, key, label in TIERS:
        r, g = nemesis_rating(store, player, key)
        tiers.append({"tier": key, "label": label, "rating": round(r), "games": g,
                      "provisional": g < PROVISIONAL_GAMES})
    return {
        "player": {"rating": round(me), "games": games, "provisional": games < PROVISIONAL_GAMES,
                   "seeded_from_imports": games == 0},
        "nemesis": tiers,
        "history": store.rating_history(player_entity(player), limit=20),
    }
