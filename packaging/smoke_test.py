"""Smoke-test a built app: start it, wait for the server and the Maia-2 model, play one
Nemesis move, stop it.

    python packaging/smoke_test.py dist/Nemesis(.exe)
"""

import os
import subprocess
import sys
import tempfile
import time

import httpx


def main(exe: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # Output goes to a file, not a pipe: a one-file app is a launcher plus a child
        # process, and on Windows a leftover child would keep a pipe open forever.
        log = open(f"{tmp}/app.log", "w")
        env = {**os.environ, "NEMESIS_HOME": tmp}   # a fresh install: downloads Stockfish if needed, and Maia-2
        proc = subprocess.Popen([exe, "--no-browser", "--db", f"{tmp}/smoke.db"],
                                stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            url = None
            deadline = time.time() + 120
            while time.time() < deadline and url is None:
                for port in range(8000, 8011):
                    try:
                        if httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=0.5).json()["app"] == "nemesis":
                            url = f"http://127.0.0.1:{port}"
                            break
                    except Exception:
                        pass
                time.sleep(1)
            assert url, "the app never started serving"
            page = httpx.get(url, timeout=10)
            assert page.status_code == 200 and "Welcome to Nemesis" in page.text, "web page missing"
            # The human-move model downloads in the background and must load in the packaged app.
            deadline = time.time() + 300
            while time.time() < deadline:
                maia = httpx.get(f"{url}/api/setup", timeout=10).json().get("maia")
                if maia in ("ready", "unavailable"):
                    break
                time.sleep(2)
            assert maia == "ready", f"Maia-2 didn't load in the built app: {maia}"
            game = httpx.post(f"{url}/api/games", json={"player": "smoke", "color": "black"}, timeout=180).json()
            assert game["last"], f"Nemesis didn't move: {game}"
            assert game.get("opening"), f"no opening name after {game['last']}: is nemesis/data in the bundle?"
            assert len(game.get("evals", [])) == 1, f"no evaluation recorded: {game.get('evals')}"
            setup = httpx.get(f"{url}/api/setup", timeout=10).json()
            print(f"OK: {url}, Nemesis played {game['last']} ({game['opening']['name']}), "
                  f"engine {setup.get('engine')}, Maia-2 {setup.get('maia')}")
        finally:
            stop(proc)
            log.close()
            print(open(f"{tmp}/app.log").read())


def stop(proc: subprocess.Popen) -> None:
    """Stop the app and the child process it unpacked itself into."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=15)


if __name__ == "__main__":
    main(sys.argv[1])
