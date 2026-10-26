# PyInstaller spec for the download-and-run app.
#
#   python -m nemesis.analysis.install --dest packaging/stockfish   # fetch Stockfish for this OS
#   pyinstaller packaging/nemesis.spec                               # -> dist/Nemesis(.exe)
#
# One self-contained program per OS. It opens a small console window ("keep this
# open while you play") and the game in the browser.
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

root = Path(SPECPATH).parent
stockfish = [(str(p), "stockfish") for p in (root / "packaging" / "stockfish").glob("stockfish*") if p.is_file()]
if not stockfish:
    raise SystemExit("No Stockfish in packaging/stockfish: run `python -m nemesis.analysis.install --dest packaging/stockfish`")

a = Analysis(
    [str(root / "packaging" / "nemesis_app.py")],
    pathex=[str(root)],
    binaries=stockfish,
    datas=[(str(root / "nemesis" / "web" / "static"), "nemesis/web/static"),
           (str(root / "nemesis" / "data"), "nemesis/data")],
    hiddenimports=collect_submodules("uvicorn"),
    excludes=["tkinter", "pytest", "playwright"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="Nemesis",
    console=True,
    upx=False,
)
