"""Does it work on you? How well Nemesis predicts you, and whether its traps beat you
more often than your ordinary mistakes would.

Everything comes from what Nemesis saved while playing (annotations), so the numbers
describe the model as it was at the time, not as retrained since.
"""

from __future__ import annotations

from nemesis.store import Store

MIN_TRAPS = 5       # below this, trap effectiveness isn't worth a claim
MISTAKE_CP = 100    # a reply losing at least a pawn counts as a mistake


def insights(store: Store, player: str) -> dict:
    per_game = []
    traps = sprung = 0
    other_replies = other_mistakes = 0
    total_n = total_hits = 0
    total_prob = 0.0
    for game in store.games(player):
        if game.source != "nemesis":
            continue
        ann = store.get_json("annotations", game.id)
        if not ann:
            continue
        moves = store.moves(game.id)
        n = hits = 0
        prob_sum = 0.0
        g_traps = g_sprung = 0
        hinted = set(ann.get("hints", []))
        for a in ann.get("moves", []):
            if a["ply"] + 1 >= len(moves) or a["ply"] + 1 in hinted:
                continue  # the game ended on Nemesis's move, or a hint chose your reply
            reply = moves[a["ply"] + 1].uci
            n += 1
            hits += a.get("top") == reply
            prob_sum += a.get("probs", {}).get(reply, 0.0)
            trap = a.get("trap")
            loss = (a.get("reply") or {}).get("loss")
            if trap:
                g_traps += 1
                g_sprung += trap["uci"] == reply
            elif loss is not None:
                other_replies += 1
                other_mistakes += loss >= MISTAKE_CP
        if not n:
            continue
        per_game.append({
            "game_id": game.id, "date": (game.played_at or "")[:10], "result": game.result,
            "moves": n, "predicted": hits, "rate": round(hits / n, 3), "avg_prob": round(prob_sum / n, 3),
            "traps": g_traps, "sprung": g_sprung,
        })
        total_n += n
        total_hits += hits
        total_prob += prob_sum
        traps += g_traps
        sprung += g_sprung

    trap_rate = sprung / traps if traps else None
    base_rate = other_mistakes / other_replies if other_replies else None
    verdict = None
    if traps >= MIN_TRAPS and base_rate is not None:
        if base_rate == 0:
            verdict = "traps" if sprung else "even"
        else:
            ratio = trap_rate / base_rate
            verdict = "traps" if ratio >= 1.25 else "even" if ratio >= 0.8 else "worse"

    return {
        "games": len(per_game),
        "predictions": total_n,
        "predicted": total_hits,
        "prediction_rate": round(total_hits / total_n, 3) if total_n else None,
        "avg_prob": round(total_prob / total_n, 3) if total_n else None,
        "traps": traps,
        "sprung": sprung,
        "trap_rate": None if trap_rate is None else round(trap_rate, 3),
        "mistake_rate": None if base_rate is None else round(base_rate, 3),
        "verdict": verdict,          # traps (work) | even | worse | None (not enough data)
        "per_game": per_game,
        "training": store.model_history(player),
    }
