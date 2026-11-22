import httpx
import pytest
from fastapi.testclient import TestClient

from nemesis.launcher import free_port
from nemesis.web import app as webapp
from nemesis.web import sharing as share


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp, "ensure_stockfish", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no engine in this test")))
    return webapp.create_app(tmp_path / "t.db")


def test_this_computer_needs_no_pin(app):
    with TestClient(app) as c:
        assert c.get("/api/health").json() == {"app": "nemesis"}
        st = c.get("/api/share").json()
        assert st["on"] is False and len(st["pin"]) == 6 and st["pin"].isdigit()


def test_other_devices_need_the_pin(app):
    pin = app.state.sharing.pin
    ipad = TestClient(app, client=("192.168.1.50", 50000), follow_redirects=False)
    page = ipad.get("/")
    assert page.status_code == 401 and "Enter the PIN" in page.text
    assert ipad.get("/api/health").status_code == 401
    wrong = ipad.post("/pin", content="pin=000000" if pin != "000000" else "pin=111111",
                      headers={"content-type": "application/x-www-form-urlencoded"})
    assert wrong.status_code == 401 and "not the PIN" in wrong.text
    ok = ipad.post("/pin", content=f"pin={pin}", headers={"content-type": "application/x-www-form-urlencoded"})
    assert ok.status_code == 303 and share.COOKIE in ok.headers["set-cookie"]
    assert ipad.get("/api/health").json() == {"app": "nemesis"}          # the cookie keeps it in
    assert ipad.get("/api/share").status_code == 403                    # but it can't see or change sharing
    assert ipad.post("/api/share", json={"on": False}).status_code == 403
    app.state.sharing.new_pin()                                          # a new PIN locks everyone out again
    assert ipad.get("/api/health").status_code == 401


def test_guessing_the_pin_gets_locked_out(app):
    s = app.state.sharing
    wrong = "000000" if s.pin != "000000" else "111111"
    for _ in range(share.MAX_TRIES):
        assert s.try_pin("10.0.0.9", wrong)[0] is False
    ok, why = s.try_pin("10.0.0.9", s.pin)
    assert not ok and "Wait a minute" in why
    assert s.try_pin("10.0.0.10", s.pin)[0]                            # other devices aren't affected


def test_sharing_opens_a_second_listener_and_is_remembered(app, tmp_path, monkeypatch):
    monkeypatch.setattr(share, "lan_address", lambda: "127.0.0.2")       # stands in for the Wi-Fi address
    s = app.state.sharing
    s.started(free_port("127.0.0.2"))
    st = s.start()
    try:
        assert st["on"] and st["url"] == f"http://127.0.0.2:{s.port}"
        # From another address, as an iPad would connect: reachable, and PIN-protected.
        ipad = dict(transport=httpx.HTTPTransport(local_address="127.0.0.3"), trust_env=False)
        with httpx.Client(**ipad) as c:
            assert c.get(f"{st['url']}/api/health").status_code == 401
        with httpx.Client(cookies={share.COOKIE: s.pin}, **ipad) as c:
            assert c.get(f"{st['url']}/api/health").json() == {"app": "nemesis"}
    finally:
        s.stop()
    assert not s.on and not s.status()["wanted"]
    s.start(); again = share.Sharing(tmp_path / "settings.json"); s.stop()
    assert again.settings["share"] is True and again.pin == s.pin      # the choice and the PIN survive a restart


def test_no_network_says_so(app, monkeypatch):
    monkeypatch.setattr(share, "lan_address", lambda: None)
    s = app.state.sharing
    s.started(8000)
    st = s.start()
    assert not st["on"] and "network" in st["error"]
