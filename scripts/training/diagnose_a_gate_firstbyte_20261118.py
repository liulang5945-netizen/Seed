"""Why is the first byte E4 (the character 中) never emitted?

Prints, for every fit world whose target is the middle class, the probability mass
on the three first-byte candidates E4/E5/E6, whether E4 is masked, and the argmax
symbol.  A masked E4 is a decoder bug; a finite-but-dominated E4 is class
competition in the emission head.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5 import (
    generate_examples,
    load_learner,
    output,
    prepare,
)  # noqa: E402

OUT = ROOT / "reports/taiji_u2_firstbyte_probe_20261118.json"
CANDIDATES = {"E4-\u4e2d": 0xE4, "E5-\u5de6\u53f3": 0xE5, "E6-\u672a": 0xE6}


def main() -> int:
    checkpoint = sorted(glob.glob(str(ROOT / "output/a_pixel_parallel_*/segment.bin")))[-1]
    model, _, _ = load_learner(Path(checkpoint))
    rows = []
    with torch.no_grad():
        for example in generate_examples("fit", 32):
            target = example["target"].decode()
            state = prepare(model, example["public"], example["id"])
            probability, _, _ = output(model, state)
            entry = {
                "target": target,
                "argmax_symbol": int(probability.argmax()),
                "argmax_hex": hex(int(probability.argmax())),
                "argmax_prob": round(float(probability.max()), 4),
            }
            for name, symbol in CANDIDATES.items():
                entry[name] = round(float(probability[symbol]), 4)
            rows.append(entry)
    middle = [row for row in rows if row["target"] == "中间"]
    payload = {
        "schema": "taiji-u2-firstbyte-probe-v1",
        "card": "22.5.0.A",
        "checkpoint": Path(checkpoint).relative_to(ROOT).as_posix(),
        "middle_rows": middle,
        "e4_finite_anywhere": all(all(row[name] > -1e3 for name in CANDIDATES) for row in rows),
        "argmax_histogram": {
            hex(symbol): sum(1 for row in rows if row["argmax_symbol"] == symbol)
            for symbol in sorted({row["argmax_symbol"] for row in rows})
        },
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    for row in middle:
        print(
            row["target"],
            "argmax",
            row["argmax_hex"],
            row["argmax_prob"],
            "E4",
            next(v for k, v in row.items() if k.startswith("E4")),
            "E5",
            next(v for k, v in row.items() if k.startswith("E5")),
            "E6",
            next(v for k, v in row.items() if k.startswith("E6")),
        )
    print("hist", payload["argmax_histogram"], "all_finite", payload["e4_finite_anywhere"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
