"""Write the licence files that ship in the app's zip:  python packaging/notices.py OUT_DIR

  OUT_DIR/LICENSE.txt                       Nemesis's own licence (MIT)
  OUT_DIR/licenses/COPYING-Stockfish.txt    Stockfish's licence (GPL v3), its authors, and
  OUT_DIR/licenses/AUTHORS-Stockfish.txt    where its exact source code is, as saved by
  OUT_DIR/licenses/SOURCE-Stockfish.txt     `python -m nemesis.analysis.install`
  OUT_DIR/licenses/THIRD-PARTY-NOTICES.txt  every Python package inside the app, with its licence

Run it in the environment the app was built from, so the versions are the ones bundled.
"""

from __future__ import annotations

import shutil
import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent
STOCKFISH_FILES = ("COPYING-Stockfish.txt", "AUTHORS-Stockfish.txt", "SOURCE-Stockfish.txt")

HEADER = """\
THIRD-PARTY SOFTWARE IN NEMESIS

Nemesis's own code is under the MIT licence (LICENSE.txt). The app also contains the
software below, each under its own licence. Its source code is public:

  Nemesis      https://github.com/macimoar/nemesis
  Stockfish    see SOURCE-Stockfish.txt (GNU GPL v3, COPYING-Stockfish.txt). Stockfish runs as
               a separate program that Nemesis talks to.
  python-chess and the Python packages listed below: on https://pypi.org, at the versions shown.

python-chess (the "chess" package below) is under the GNU GPL v3, and this program is built
with it, so the program as a whole is distributed under the GNU GPL v3
(https://www.gnu.org/licenses/gpl-3.0.html; full text in COPYING-Stockfish.txt). Nemesis's own
source code is still MIT on its own.

Downloaded on first use, not included in this zip:
  Maia-2 human-move models (MIT, Copyright (c) 2024 CSSLab, https://github.com/CSSLab/maia2),
  saved with their licence in your .nemesis folder.

Also included:
  Opening names: the Lichess chess-openings list (CC0, public domain),
    https://github.com/lichess-org/chess-openings
  Python (Python Software Foundation License), https://docs.python.org/3/license.html
  The PyInstaller bootloader (GPL v2 with an exception that allows bundling any program),
    https://github.com/pyinstaller/pyinstaller/blob/develop/COPYING.txt
"""


def bundled(root: str = "nemesis") -> list[metadata.Distribution]:
    """The Python packages the app needs, found by following requirements from `root`."""
    seen: dict[str, metadata.Distribution] = {}
    todo = [root]
    while todo:
        name = todo.pop()
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        key = dist.metadata["Name"].lower().replace("_", "-")
        if key in seen:
            continue
        seen[key] = dist
        for spec in dist.requires or []:
            req = Requirement(spec)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                todo.append(req.name)
    seen.pop(root, None)
    return sorted(seen.values(), key=lambda d: d.metadata["Name"].lower())


def licence_of(dist: metadata.Distribution) -> str:
    md = dist.metadata
    expr = md.get("License-Expression")
    if expr:
        return expr
    classifiers = [c.split(" :: ")[-1] for c in md.get_all("Classifier") or [] if c.startswith("License ::")]
    short = (md.get("License") or "").strip()
    if classifiers:
        return ", ".join(classifiers)
    return short if short and len(short) < 80 else "see the licence text below"


def licence_texts(dist: metadata.Distribution) -> list[tuple[str, str]]:
    out = []
    for f in dist.files or []:
        name = f.name.upper()
        # In the package's metadata, or at the top of the package itself (onnxruntime keeps them there).
        if name.startswith(("LICENSE", "LICENCE", "COPYING", "NOTICE", "AUTHORS", "THIRDPARTYNOTICES")) and (
                ".dist-info" in str(f) or len(f.parts) == 2):
            try:
                out.append((str(f), Path(f.locate()).read_text(encoding="utf-8", errors="replace")))
            except OSError:
                pass
    if not out and (md_text := dist.metadata.get("License")) and len(md_text) >= 80:
        out.append(("licence from the package metadata", md_text))
    return out


def main(out: Path) -> None:
    lic = out / "licenses"
    lic.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "LICENSE", out / "LICENSE.txt")
    for name in STOCKFISH_FILES:
        src = ROOT / "packaging" / "stockfish" / name
        if not src.exists():
            raise SystemExit(f"missing {src}: fetch Stockfish with `python -m nemesis.analysis.install --dest packaging/stockfish`")
        shutil.copy(src, lic / name)
    parts = [HEADER, "\nPYTHON PACKAGES IN THIS APP\n"]
    dists = bundled()
    for d in dists:
        parts.append(f"  {d.metadata['Name']} {d.version}: {licence_of(d)}")
    for d in dists:
        parts.append("\n" + "=" * 78 + f"\n{d.metadata['Name']} {d.version}: {licence_of(d)}\n" + "=" * 78)
        texts = licence_texts(d)
        if not texts and "Apache" in licence_of(d):
            texts = [("Apache License 2.0", "The package ships no licence file. The licence: "
                      "https://www.apache.org/licenses/LICENSE-2.0.txt")]
        if not texts:
            parts.append(f"(No licence file in the package; see https://pypi.org/project/{d.metadata['Name']}/{d.version}/)")
        for where, text in texts:
            parts.append(f"--- {where} ---\n{text.strip()}\n")
    (lic / "THIRD-PARTY-NOTICES.txt").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote licences for Nemesis, Stockfish and {len(dists)} Python packages to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
