import io
import socket
import sys
import tarfile
import zipfile

import httpx
import pytest

from nemesis import launcher
from nemesis.analysis import install

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="fake engines are shell scripts")

PAD = "#" + "x" * 120_000 + "\n"   # real engines are big; the extractor ignores tiny files
GOOD = "#!/bin/sh\n" + PAD + "read line; echo 'id name Stockfish'; echo uciok\n"
BAD = "#!/bin/sh\n" + PAD + "exit 132\n"   # like an AVX2 build on a CPU without AVX2


def zip_of(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return buf.getvalue()


def tar_of(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def fake_github(archives: dict[str, bytes]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if "/releases/" in request.url.path:
            assets = [{"name": n, "browser_download_url": f"https://dl.example/{n}"} for n in archives]
            return httpx.Response(200, json={"assets": assets})
        name = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, content=archives[name])
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_build_preferences_by_platform():
    assert install.build_preferences("Windows", "AMD64")[0] == "windows-x86-64-avx2"
    assert install.build_preferences("Darwin", "arm64") == ["macos-m1-apple-silicon"]
    assert install.build_preferences("Darwin", "x86_64")[0] == "macos-x86-64-avx2"
    assert install.build_preferences("Linux", "x86_64")[-1] == "ubuntu-x86-64"


def test_pick_assets_matches_exact_builds_in_order():
    assets = [{"name": n} for n in ["stockfish-windows-x86-64.zip", "stockfish-windows-x86-64-avx2.zip",
                                     "stockfish-ubuntu-x86-64-avx2.tar", "stockfish-windows-x86-64-avx2.zip.sha256"]]
    picked = [a["name"] for a in install.pick_assets(assets, install.build_preferences("Windows", "AMD64"))]
    assert picked == ["stockfish-windows-x86-64-avx2.zip", "stockfish-windows-x86-64.zip"]


def test_download_falls_back_when_fast_build_cannot_run(tmp_path):
    archives = {
        "stockfish-ubuntu-x86-64-avx2.tar": tar_of({"stockfish/stockfish-ubuntu-x86-64-avx2": BAD,
                                                    "stockfish/README.md": "docs"}),
        "stockfish-ubuntu-x86-64.zip": zip_of({"stockfish/stockfish-ubuntu-x86-64": GOOD,
                                               "stockfish/AUTHORS.txt": "people"}),
    }
    steps = []
    path = install.download_stockfish(tmp_path, progress=lambda m, f: steps.append(m),
                                      client=fake_github(archives),
                                      preferences=["ubuntu-x86-64-avx2", "ubuntu-x86-64"])
    assert path.endswith("stockfish-ubuntu-x86-64") and install.works(path)
    assert not (tmp_path / "stockfish-ubuntu-x86-64-avx2").exists()   # the broken one is removed
    assert install.downloaded(tmp_path) == path
    assert "Downloading Stockfish" in steps


def test_stockfish_licence_travels_with_it(tmp_path):
    # Stockfish is GPLv3: its licence, authors and where its source is go wherever the engine goes.
    archives = {"stockfish-ubuntu-x86-64.tar": tar_of({"stockfish/stockfish-ubuntu-x86-64": GOOD,
                                                       "stockfish/Copying.txt": "GNU GENERAL PUBLIC LICENSE",
                                                       "stockfish/AUTHORS": "Tord Romstad"})}
    path = install.download_stockfish(tmp_path, client=fake_github(archives), preferences=["ubuntu-x86-64"])
    assert (tmp_path / "COPYING-Stockfish.txt").read_text() == "GNU GENERAL PUBLIC LICENSE"
    assert (tmp_path / "AUTHORS-Stockfish.txt").read_text() == "Tord Romstad"
    source = (tmp_path / "SOURCE-Stockfish.txt").read_text()
    assert f"tree/{install.PINNED_TAG}" in source and "GPL" in source
    assert install.downloaded(tmp_path) == path        # the extra files don't confuse the lookup


def test_download_reports_when_nothing_runs(tmp_path):
    archives = {"stockfish-ubuntu-x86-64.tar": tar_of({"stockfish/stockfish-ubuntu-x86-64": BAD})}
    with pytest.raises(RuntimeError, match="none of the Stockfish builds run"):
        install.download_stockfish(tmp_path, client=fake_github(archives), preferences=["ubuntu-x86-64"])


def test_lookup_order(tmp_path, monkeypatch):
    from nemesis.analysis import stockfish as sf
    bundle = tmp_path / "bundle" / "stockfish"
    bundle.mkdir(parents=True)
    (bundle / "stockfish-ubuntu-x86-64").write_text(GOOD)
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    assert sf.find_stockfish() == str(bundle / "stockfish-ubuntu-x86-64")   # bundled beats PATH
    monkeypatch.setenv("STOCKFISH_PATH", str(tmp_path / "missing"))
    assert sf.find_stockfish() is None                                       # explicit setting wins


# -- launcher -------------------------------------------------------------------

def test_busy_port_moves_to_the_next_free_one():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        assert not launcher.port_free("127.0.0.1", busy)
        assert launcher.free_port("127.0.0.1", range(busy, busy + 5)) != busy


def test_detects_a_running_nemesis(tmp_path):
    import threading
    import time
    import uvicorn
    from nemesis.web.app import create_app

    port = launcher.free_port("127.0.0.1", range(8650, 8700))
    server = uvicorn.Server(uvicorn.Config(create_app(tmp_path / "t.db"), host="127.0.0.1", port=port,
                                           log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    while not server.started:
        time.sleep(0.05)
    try:
        assert launcher.running_nemesis(ports=range(port, port + 1)) == f"http://127.0.0.1:{port}"
        setup = httpx.get(f"http://127.0.0.1:{port}/api/setup").json()
        assert setup["players"] == 0 and setup["stockfish"] in ("checking", "downloading", "ready", "error")
        assert launcher.running_nemesis(ports=range(port + 1, port + 2)) is None
    finally:
        server.should_exit = True
        t.join(5)
