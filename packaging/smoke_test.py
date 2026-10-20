"""Smoke-test a built app: start it, wait for the server, play one Nemesis move, stop it.

    python packaging/smoke_test.py dist/Nemesis(.exe)
"""

import subprocess
import sys
import tempfile
import time

import httpx


def main(exe: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.Popen([exe, "--no-browser", "--db", f"{tmp}/smoke.db"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
            game = httpx.post(f"{url}/api/games", json={"player": "smoke", "color": "black"}, timeout=180).json()
            assert game["last"], f"Nemesis didn't move: {game}"
            setup = httpx.get(f"{url}/api/setup", timeout=10).json()
            print(f"OK: {url}, Nemesis played {game['last']}, engine {setup.get('engine')}")
        finally:
            proc.terminate()
            try:
                out, _ = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, _ = proc.communicate()
            print(out)


if __name__ == "__main__":
    main(sys.argv[1])
