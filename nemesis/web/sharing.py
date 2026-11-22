"""Play from another device on your Wi-Fi (an iPad, a phone, another computer).

Nemesis normally answers only this computer (127.0.0.1). Sharing adds a second listener on
this computer's address on the local network, on the same port, so the same app and games
are reachable from, say, an iPad's browser. It's off until you turn it on, and remembered.

Other devices must enter a PIN, shown on this computer, before they get anything: a cookie
then keeps them in. Only this computer can turn sharing on or off or see the PIN.
"""

from __future__ import annotations

import json
import secrets
import socket
import threading
import time
from pathlib import Path

COOKIE = "nemesis_pin"
MAX_TRIES = 10        # wrong PINs per device before it has to wait
LOCKOUT = 60.0        # seconds


def is_local(host: str | None) -> bool:
    return host in ("127.0.0.1", "::1", "localhost", "testclient")


def lan_address() -> str | None:
    """This computer's address on the local network (the one its default route uses).
    A UDP 'connect' sends nothing; it just makes the system choose the outgoing interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        except OSError:
            return None
    return None if ip.startswith("127.") or ip == "0.0.0.0" else ip


class Sharing:
    def __init__(self, settings_path: Path, app=None):
        self.path = Path(settings_path)
        self.app = app
        self.port: int | None = None
        self.server = None          # the uvicorn server on the network address, while sharing
        self.thread: threading.Thread | None = None
        self.address: str | None = None
        self.error = ""
        self._fails: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self.settings = self._load()

    # -- settings ------------------------------------------------------------
    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            data = {}
        if not str(data.get("pin", "")).isdigit() or len(str(data.get("pin"))) != 6:
            data["pin"] = f"{secrets.randbelow(10 ** 6):06d}"
        data.setdefault("share", False)
        return data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.settings))

    @property
    def pin(self) -> str:
        return self.settings["pin"]

    @property
    def on(self) -> bool:
        return self.server is not None

    def status(self) -> dict:
        return {"on": self.on, "wanted": self.settings["share"], "pin": self.pin, "error": self.error,
                "url": f"http://{self.address}:{self.port}" if self.on else None}

    # -- the second listener -------------------------------------------------
    def started(self, port: int) -> None:
        """Called once the main server is up: share again if it was on last time."""
        self.port = port
        if self.settings["share"]:
            self.start()

    def start(self) -> dict:
        with self._lock:
            if self.on:
                return self.status()
            self.error = ""
            address = lan_address()
            if address is None:
                self.error = "This computer doesn't seem to be on a network. Connect to Wi-Fi and try again."
                return self.status()
            if self.port is None or self.app is None:
                self.error = "Nemesis isn't fully started yet. Try again in a moment."
                return self.status()
            import uvicorn
            server = uvicorn.Server(uvicorn.Config(self.app, host=address, port=self.port, log_level="warning",
                                                   lifespan="off"))   # the main server owns startup and shutdown
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            deadline = time.monotonic() + 5
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.05)
            if not server.started:
                server.should_exit = True
                self.error = (f"Couldn't open {address}:{self.port} to the network. Another program may be "
                              "using it, or a firewall blocked it.")
                return self.status()
            self.server, self.thread, self.address = server, thread, address
            self.settings["share"] = True
            self._save()
            return self.status()

    def stop(self) -> dict:
        with self._lock:
            if self.server is not None:
                self.server.should_exit = True
                self.thread.join(5)   # let go of the port, so turning it back on works straight away
            self.server, self.thread, self.address, self.error = None, None, None, ""
            self.settings["share"] = False
            self._save()
            return self.status()

    def new_pin(self) -> dict:
        """A fresh PIN: every other device has to enter it again."""
        self.settings["pin"] = f"{secrets.randbelow(10 ** 6):06d}"
        self._save()
        return self.status()

    # -- who gets in ---------------------------------------------------------
    def allowed(self, cookie: str | None) -> bool:
        return bool(cookie) and secrets.compare_digest(cookie, self.pin)

    def try_pin(self, client: str, pin: str) -> tuple[bool, str]:
        now = time.monotonic()
        fails = [t for t in self._fails.get(client, []) if now - t < LOCKOUT]
        if len(fails) >= MAX_TRIES:
            return False, "Too many wrong PINs. Wait a minute and try again."
        if secrets.compare_digest(pin.strip(), self.pin):
            self._fails.pop(client, None)
            return True, ""
        self._fails[client] = fails + [now]
        return False, "That's not the PIN shown on your computer."


PIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Nemesis</title>
<style>
  :root { color-scheme: light dark; --bg: #f6f3ee; --ink: #1d1b19; --accent: #a3192b; --muted: #6b655f; }
  @media (prefers-color-scheme: dark) { :root { --bg: #1d1c1b; --ink: #eee; --accent: #e0485c; --muted: #a59f99; } }
  body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: var(--bg); color: var(--ink);
         font: 17px/1.4 system-ui, sans-serif; padding: 16px; box-sizing: border-box; }
  form { max-width: 360px; width: 100%; text-align: center; }
  h1 { color: var(--accent); margin: 0 0 6px; }
  p { color: var(--muted); }
  input { font: 600 32px/1 ui-monospace, monospace; letter-spacing: .3em; text-align: center; width: 100%;
          box-sizing: border-box; padding: 12px; border-radius: 10px; border: 1px solid var(--muted); background: transparent; color: var(--ink); }
  button { margin-top: 14px; width: 100%; font: 600 18px system-ui; padding: 12px; border: 0; border-radius: 10px;
           background: var(--accent); color: #fff; }
  .err { color: var(--accent); min-height: 1.4em; }
</style></head><body>
<form method="post" action="/pin">
  <h1>Nemesis</h1>
  <p>Enter the PIN shown on your computer, under <b>Play on another device</b>.</p>
  <input name="pin" inputmode="numeric" autocomplete="one-time-code" maxlength="6" autofocus required>
  <div class="err">{error}</div>
  <button>Continue</button>
</form></body></html>"""
