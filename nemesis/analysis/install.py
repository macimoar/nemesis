"""Get Stockfish without any manual setup.

`ensure_stockfish()` returns a working Stockfish binary. It uses one that's already
available (see `find_stockfish`), otherwise it downloads the right official build for
this computer from GitHub into ~/.nemesis/stockfish/. If the fast (AVX2) build
can't run on this CPU, it falls back to a more compatible build automatically.

    python -m nemesis.analysis.install [--dest DIR]
"""

from __future__ import annotations

import argparse
import io
import os
import platform
import re
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Callable

import httpx

PINNED_TAG = "sf_17.1"
API = "https://api.github.com/repos/official-stockfish/Stockfish/releases"
HEADERS = {"User-Agent": "nemesis-chess", "Accept": "application/vnd.github+json"}
INSTALL_DIR = Path(os.environ.get("NEMESIS_HOME", Path.home() / ".nemesis")) / "stockfish"

Progress = Callable[[str, float], None]   # (message, fraction 0..1)


def build_preferences(system: str | None = None, machine: str | None = None) -> list[str]:
    """Release build names to try, fastest first, for this OS and CPU."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    arm = machine in ("arm64", "aarch64", "armv8", "arm64e")
    if system == "darwin":
        return ["macos-m1-apple-silicon"] if arm else \
            ["macos-x86-64-avx2", "macos-x86-64-sse41-popcnt", "macos-x86-64"]
    if system == "windows":
        return ["windows-x86-64-avx2", "windows-x86-64-sse41-popcnt", "windows-x86-64"]
    if arm:
        return ["ubuntu-arm64", "android-armv8"]
    return ["ubuntu-x86-64-avx2", "ubuntu-x86-64-sse41-popcnt", "ubuntu-x86-64"]


def pick_assets(assets: list[dict], preferences: list[str]) -> list[dict]:
    """Release assets matching the preferences, in preference order."""
    out = []
    for build in preferences:
        pattern = re.compile(rf"^stockfish-{re.escape(build)}\.(zip|tar|tar\.gz|tgz)$")
        out += [a for a in assets if pattern.match(a["name"])]
    return out


def release(client: httpx.Client, tag: str = PINNED_TAG) -> dict:
    resp = client.get(f"{API}/tags/{tag}")
    if resp.status_code == 404:  # pinned tag gone: use the latest release
        resp = client.get(f"{API}/latest")
    resp.raise_for_status()
    return resp.json()


def release_assets(client: httpx.Client, tag: str = PINNED_TAG) -> list[dict]:
    return release(client, tag).get("assets", [])


def _download(client: httpx.Client, url: str, progress: Progress | None) -> bytes:
    buf = io.BytesIO()
    with client.stream("GET", url) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        for chunk in resp.iter_bytes():
            buf.write(chunk)
            if progress and total:
                progress("Downloading Stockfish", buf.tell() / total)
    return buf.getvalue()


def _is_engine(name: str) -> bool:
    base = os.path.basename(name).lower()
    return base.startswith("stockfish") and not base.endswith(
        (".txt", ".md", ".nnue", ".h", ".cpp", ".py", ".sh", ".yml", ".zip", ".tar"))


# Stockfish is GPLv3: whoever passes the engine on must pass its licence on with it.
NOTICES = {"copying.txt": "COPYING-Stockfish.txt", "authors": "AUTHORS-Stockfish.txt"}


def extract_binary(data: bytes, name: str, dest: Path) -> Path:
    """Pull the engine executable out of a release archive into `dest`, with its licence
    and authors list (as COPYING-Stockfish.txt and AUTHORS-Stockfish.txt)."""
    dest.mkdir(parents=True, exist_ok=True)
    found: tuple[str, bytes] | None = None
    notices: dict[str, bytes] = {}
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if os.path.basename(info.filename).lower() in NOTICES:
                    notices[NOTICES[os.path.basename(info.filename).lower()]] = zf.read(info)
                elif _is_engine(info.filename) and info.file_size > 100_000:
                    if found is None or info.file_size > len(found[1]):
                        found = (info.filename, zf.read(info))
    else:
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                if os.path.basename(member.name).lower() in NOTICES:
                    notices[NOTICES[os.path.basename(member.name).lower()]] = tf.extractfile(member).read()
                elif _is_engine(member.name) and member.size > 100_000:
                    if found is None or member.size > len(found[1]):
                        found = (member.name, tf.extractfile(member).read())
    if found is None:
        raise RuntimeError(f"no Stockfish executable inside {name}")
    target = dest / os.path.basename(found[0])
    target.write_bytes(found[1])
    for fname, content in notices.items():
        (dest / fname).write_bytes(content)
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return target


def works(path: str | Path, timeout: float = 15.0) -> bool:
    """Does this binary start and speak UCI? (An AVX2 build crashes on CPUs without AVX2.)"""
    try:
        out = subprocess.run([str(path)], input="uci\nquit\n", capture_output=True, text=True,
                             timeout=timeout, **_no_window())
    except (OSError, subprocess.SubprocessError):
        return False
    return "uciok" in out.stdout


def _no_window() -> dict:
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def downloaded(dest: Path = INSTALL_DIR) -> str | None:
    """A Stockfish previously downloaded by Nemesis, if one is there and works."""
    marker = dest / "ENGINE"
    if marker.exists():
        path = dest / marker.read_text().strip()
        if path.is_file():
            return str(path)
    return None


def download_stockfish(dest: Path = INSTALL_DIR, progress: Progress | None = None,
                       client: httpx.Client | None = None, preferences: list[str] | None = None) -> str:
    """Download the best official build that runs on this computer. Returns its path."""
    own = client is None
    headers = dict(HEADERS)
    if token := os.environ.get("GITHUB_TOKEN"):  # optional: avoids GitHub's anonymous rate limit (CI)
        headers["Authorization"] = f"Bearer {token}"
    client = client or httpx.Client(headers=headers, timeout=60, follow_redirects=True)
    try:
        if progress:
            progress("Finding the right Stockfish for this computer", 0.0)
        rel = release(client)
        candidates = pick_assets(rel.get("assets", []), preferences or build_preferences())
        if not candidates:
            raise RuntimeError(f"no Stockfish build for {platform.system()} {platform.machine()}")
        for asset in candidates:
            data = _download(client, asset["browser_download_url"], progress)
            if progress:
                progress("Unpacking Stockfish", 1.0)
            path = extract_binary(data, asset["name"], dest)
            if works(path):
                (dest / "ENGINE").write_text(path.name)
                tag = rel.get("tag_name", PINNED_TAG)
                (dest / "SOURCE-Stockfish.txt").write_text(
                    f"Stockfish {tag} ({asset['name']}), free software under the GNU GPL v3 (see COPYING-Stockfish.txt).\n"
                    f"Its complete source code: https://github.com/official-stockfish/Stockfish/tree/{tag}\n"
                    f"The same source is inside the official download: {asset['browser_download_url']}\n")
                return str(path)
            path.unlink(missing_ok=True)  # this CPU can't run that build: try the next one
        raise RuntimeError("none of the Stockfish builds run on this computer")
    finally:
        if own:
            client.close()


def ensure_stockfish(progress: Progress | None = None) -> str:
    """A working Stockfish: the one already available, or a fresh download."""
    from nemesis.analysis.stockfish import find_stockfish
    path = find_stockfish()
    if path and works(path):
        return path
    if os.environ.get("STOCKFISH_PATH"):
        # The user pointed us somewhere specific: don't silently replace it.
        raise RuntimeError(f"STOCKFISH_PATH={os.environ['STOCKFISH_PATH']!r} doesn't run as Stockfish")
    return download_stockfish(progress=progress)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Stockfish for this computer.")
    ap.add_argument("--dest", type=Path, default=INSTALL_DIR)
    ap.add_argument("--build", action="append", help="build name to prefer, e.g. windows-x86-64-avx2")
    args = ap.parse_args()

    def show(msg: str, frac: float) -> None:
        print(f"\r{msg}: {frac:4.0%}", end="", flush=True)

    path = download_stockfish(args.dest, progress=show, preferences=args.build)
    print(f"\nStockfish installed: {path}")


if __name__ == "__main__":
    main()
