"""Build nemesis/data/openings.json from the Lichess opening list.

    git clone https://github.com/lichess-org/chess-openings
    python scripts/build_openings.py chess-openings

The list (CC0, public domain) names about 3,500 openings by their moves. Nemesis looks
openings up by position, so transpositions get the right name: the table maps each
named position (FEN without the move counters) to [ECO code, name].
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import chess.pgn

OUT = Path(__file__).resolve().parent.parent / "nemesis" / "data" / "openings.json"


def main(src: str) -> None:
    table: dict[str, tuple[str, str, int]] = {}
    for tsv in sorted(Path(src).glob("[a-e].tsv")):
        with open(tsv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                game = chess.pgn.read_game(io.StringIO(row["pgn"]))
                board = game.end().board()
                key = board.epd()
                plies = board.ply()
                # Several move orders can name the same position: keep the most direct one.
                if key not in table or plies < table[key][2]:
                    table[key] = (row["eco"], row["name"], plies)
    out = {k: [eco, name] for k, (eco, name, _) in sorted(table.items())}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{len(out)} named positions -> {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main(sys.argv[1])
