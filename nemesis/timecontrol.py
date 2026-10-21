"""Time controls: presets, categories (bullet/blitz/rapid/classical/untimed), and how
long Nemesis may think on its own clock."""

from __future__ import annotations

from dataclasses import dataclass

CATEGORIES = ["bullet", "blitz", "rapid", "classical", "untimed"]
LABELS = {"bullet": "Bullet", "blitz": "Blitz", "rapid": "Rapid", "classical": "Classical", "untimed": "Untimed"}


@dataclass(frozen=True)
class TimeControl:
    base: float        # seconds per side
    increment: float   # seconds added after each move

    @property
    def category(self) -> str:
        # Lichess's rule: estimated game length = base + 40 * increment.
        est = self.base + 40 * self.increment
        if est < 180:
            return "bullet"
        if est < 480:
            return "blitz"
        if est < 1500:
            return "rapid"
        return "classical"

    @property
    def label(self) -> str:
        minutes = self.base / 60
        m = f"{minutes:g}" if minutes >= 1 else f"{minutes:.2g}"
        return f"{m}+{self.increment:g}"

    def pgn(self) -> str:
        return f"{self.base:g}+{self.increment:g}"


PRESETS = {
    "1+0": TimeControl(60, 0), "2+1": TimeControl(120, 1),
    "3+0": TimeControl(180, 0), "3+2": TimeControl(180, 2), "5+0": TimeControl(300, 0),
    "10+0": TimeControl(600, 0), "15+10": TimeControl(900, 10),
}


def parse(value: str | None) -> TimeControl | None:
    """A preset name ("3+2") or a PGN TimeControl header ("180+2", "600"). None = untimed."""
    if not value or value in ("-", "?", "untimed"):
        return None
    if value in PRESETS:
        return PRESETS[value]
    if "/" in value:  # correspondence/daily, e.g. Chess.com "1/86400"
        return None
    base, _, inc = value.partition("+")
    try:
        return TimeControl(float(base), float(inc or 0))
    except ValueError:
        return None


def category(tc: TimeControl | None) -> str:
    return tc.category if tc else "untimed"


def pgn_category(header: str | None) -> str:
    """Category of an imported game, from its PGN TimeControl header."""
    return category(parse(header))


def think_budget(remaining: float | None, increment: float = 0.0) -> float | None:
    """Seconds Nemesis may spend on this move, or None for no limit.

    A standard engine rule: a slice of what's left plus most of the increment, never
    more than a quarter of the clock, so it can't lose on time from thinking.
    """
    if remaining is None:
        return None
    budget = remaining / 30 + increment * 0.8
    return max(0.1, min(budget, remaining / 4))
