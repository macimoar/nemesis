"""Double-click entry point: start Nemesis and open it in the browser.

If Nemesis is already running, just opens the browser there. Otherwise picks a free
port (8000-8010), starts the server, and opens the page. Used by the packaged app,
by `nemesis` with no arguments, and by `nemesis serve` for its port handling.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import traceback
import webbrowser

import httpx

PORTS = range(8000, 8011)


def running_nemesis(host: str = "127.0.0.1", ports=PORTS) -> str | None:
    """URL of a Nemesis server already running on this computer, if any."""
    for port in ports:
        url = f"http://{host}:{port}"
        try:
            if httpx.get(f"{url}/api/health", timeout=0.4).json().get("app") == "nemesis":
                return url
        except Exception:
            continue
    return None


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def free_port(host: str = "127.0.0.1", ports=PORTS) -> int:
    for port in ports:
        if port_free(host, port):
            return port
    raise RuntimeError(f"ports {ports.start}-{ports.stop - 1} are all in use; close some programs and try again")


def serve(db: str | None = None, host: str = "127.0.0.1", port: int | None = None,
          open_browser: bool = True, banner: bool = True) -> None:
    import uvicorn

    from nemesis.store import DEFAULT_DB
    from nemesis.web.app import create_app

    existing = running_nemesis(host)
    if existing and port is None:
        print(f"Nemesis is already running at {existing}. Opening it in your browser.")
        if open_browser:
            webbrowser.open(existing)
        return
    if port is None or not port_free(host, port):
        if port is not None:
            print(f"Port {port} is busy, so Nemesis will use another one.")
        port = free_port(host)
    url = f"http://{host}:{port}"

    server = uvicorn.Server(uvicorn.Config(create_app(db or DEFAULT_DB), host=host, port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started and thread.is_alive():
        time.sleep(0.05)
    if not thread.is_alive():
        raise RuntimeError("the Nemesis server couldn't start")
    if banner:
        print("=" * 60)
        print(f"  Nemesis is running at {url}")
        print("  Keep this window open while you play. Close it to quit.")
        print("=" * 60, flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        while thread.is_alive():
            thread.join(0.5)
    except KeyboardInterrupt:
        server.should_exit = True
        thread.join(5)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="nemesis-app", description="Start Nemesis and open it in your browser.")
    ap.add_argument("--db", help="database path (default: ~/.nemesis/nemesis.db)")
    ap.add_argument("--port", type=int)
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser (for testing)")
    args = ap.parse_args(argv)
    try:
        serve(args.db, port=args.port, open_browser=not args.no_browser)
    except Exception:
        traceback.print_exc()
        if getattr(sys, "frozen", False):  # double-clicked app: keep the window up so the error is readable
            input("\nSomething went wrong (details above). Press Enter to close.")
        sys.exit(1)


if __name__ == "__main__":
    main()
