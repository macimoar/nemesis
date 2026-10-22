"""Convert the official Maia-2 models to ONNX, so Nemesis can run them without PyTorch.

Run where Google Drive is reachable (the "Maia-2 models" GitHub workflow does this):

    pip install maia2 onnx onnxruntime onnxconverter-common
    python scripts/convert_maia2.py --out maia2-onnx

For each of the rapid and blitz models it writes a full-precision .onnx file and a
half-size fp16 one, checks both against the PyTorch original on a few hundred positions,
and writes the move vocabulary (moves.json), the Maia-2 licence and checksums.
`--random-init` builds the same network with random weights instead of downloading,
to test the export itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import chess
import numpy as np
import torch

from maia2 import model as maia_model
from maia2.main import MAIA2Model
from maia2.utils import board_to_tensor, create_elo_dict, get_all_possible_moves, parse_args

LICENSE = """Maia-2 (https://github.com/CSSLab/maia2) - MIT License

Copyright (c) 2024 CSSLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


class Policy(torch.nn.Module):
    """Only what Nemesis uses: move logits and the value head (expected score)."""

    def __init__(self, net: MAIA2Model):
        super().__init__()
        self.net = net

    def forward(self, boards, elos_self, elos_oppo):
        logits, _, value = self.net(boards, elos_self, elos_oppo)
        return logits, value


def sample_positions(n: int, seed: int = 0) -> list[chess.Board]:
    """Positions from random-ish games, side to move always White (as Maia-2 sees them)."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = chess.Board()
        for _ in range(rng.randint(0, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over():
            continue
        out.append(board if board.turn == chess.WHITE else board.mirror())
    return out


def check(onnx_path: Path, policy: Policy, boards: list[chess.Board], moves: list[str]) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.stack([board_to_tensor(b) for b in boards])
    rng = np.random.default_rng(0)
    es = torch.tensor(rng.integers(0, 11, len(boards)))
    eo = torch.tensor(rng.integers(0, 11, len(boards)))
    with torch.no_grad():
        ref_logits, ref_value = policy(x, es, eo)
    got_logits, got_value = sess.run(None, {"boards": x.numpy(), "elo_self": es.numpy(), "elo_oppo": eo.numpy()})
    index = {m: i for i, m in enumerate(moves)}
    agree, max_diff = 0, 0.0
    for k, board in enumerate(boards):
        legal = [index[m.uci()] for m in board.legal_moves]
        ref = torch.softmax(ref_logits[k, legal], 0).numpy()
        got = np.exp(got_logits[k, legal] - got_logits[k, legal].max())
        got /= got.sum()
        agree += int(ref.argmax() == got.argmax())
        max_diff = max(max_diff, float(np.abs(ref - got).max()))
    return {"positions": len(boards), "top1_agreement": agree / len(boards), "max_prob_diff": max_diff,
            "max_value_diff": float(np.abs(ref_value.numpy() - got_value).max())}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="maia2-onnx")
    ap.add_argument("--types", default="rapid,blitz")
    ap.add_argument("--random-init", metavar="CONFIG_YAML",
                    help="test the export with random weights and this architecture config")
    ap.add_argument("--positions", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    moves = get_all_possible_moves()
    (out / "moves.json").write_text(json.dumps(moves))
    (out / "LICENSE-maia2.txt").write_text(LICENSE)
    boards = sample_positions(args.positions)
    report = {"elo_buckets": list(create_elo_dict()), "models": {}}

    for kind in args.types.split(","):
        if args.random_init:
            net = MAIA2Model(len(moves), create_elo_dict(), parse_args(args.random_init))
        else:
            net = maia_model.from_pretrained(type=kind, device="cpu", save_root=str(out / "pt"))
        policy = Policy(net).eval()
        path = out / f"maia2-{kind}.onnx"
        example = (torch.stack([board_to_tensor(b) for b in boards[:2]]),
                   torch.tensor([3, 4]), torch.tensor([4, 3]))
        torch.onnx.export(policy, example, str(path), input_names=["boards", "elo_self", "elo_oppo"],
                          output_names=["logits", "value"], opset_version=17, dynamo=False,
                          dynamic_axes={"boards": {0: "batch"}, "elo_self": {0: "batch"},
                                        "elo_oppo": {0: "batch"}, "logits": {0: "batch"}, "value": {0: "batch"}})
        # Half-precision weights: half the download, same answers (int8 quantisation was
        # tried and changes the top move in about 1 position in 6).
        import onnx
        from onnxconverter_common import float16
        small = out / f"maia2-{kind}-fp16.onnx"
        onnx.save(float16.convert_float_to_float16(onnx.load(str(path)), keep_io_types=True), str(small))
        entry = {}
        for p in (path, small):
            entry[p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p), **check(p, policy, boards, moves)}
        source = out / "pt" / f"{kind}_model.pt"
        if source.exists():
            entry["source_sha256"] = sha256(source)
        report["models"][kind] = entry
        print(kind, json.dumps(entry, indent=2))
        for name, (min_agree, max_diff) in ((path.name, (0.99, 1e-3)), (small.name, (0.97, 0.05))):
            got = entry[name]
            if got["top1_agreement"] < min_agree or got["max_prob_diff"] > max_diff:
                raise SystemExit(f"{name} doesn't match the PyTorch model: {got}")

    (out / "report.json").write_text(json.dumps(report, indent=2))
    sums = [f"{sha256(p)}  {p.name}" for p in sorted(out.glob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    main()
