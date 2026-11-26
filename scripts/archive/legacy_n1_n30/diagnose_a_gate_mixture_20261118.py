"""Is the middle class lost in the byte head, or in the emission mixture?

For every fit world whose target is the middle class, print the argmax of the byte
head alone, the argmax after the emission mixture, and the mixture's copy weights.
If the head already picks E4 and the mixture overrides it, the defect is the copy
path; if the head also picks E5, the head needs the delta as a separate input.
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

OUT = ROOT / "reports/taiji_u2_mixture_probe_20261118.json"


def main() -> int:
    checkpoint = sorted(glob.glob(str(ROOT / "output/a_pixel_parallel_*/segment.bin")))[-1]
    model, _, _ = load_learner(Path(checkpoint))
    rows = []
    with torch.no_grad():
        for example in generate_examples("fit", 32):
            state = prepare(model, example["public"], example["id"])
            final, context, details = output(model, state)
            previous = model.previous_embedding(256)[None].expand(model.c.branches, -1)
            head = model.byte_head(torch.cat([context, previous], -1)).log_softmax(-1)
            mixture = details["mixture"]
            rows.append(
                {
                    "target": example["target"].decode(),
                    "head_argmax": hex(int(head.argmax())),
                    "final_argmax": hex(int(final.argmax())),
                    "head_e4": round(float(head[0, 0xE4]), 3),
                    "head_e5": round(float(head[0, 0xE5]), 3),
                    "final_e4": round(float(final[0xE4]), 3),
                    "final_e5": round(float(final[0xE5]), 3),
                    "mixture_keys": sorted(mixture) if isinstance(mixture, dict) else "not-a-dict",
                }
            )
    middle = [row for row in rows if row["target"] == "中间"]
    agree = sum(1 for row in rows if row["head_argmax"] == row["final_argmax"])
    payload = {
        "schema": "taiji-u2-mixture-probe-v1",
        "card": "22.5.0.A",
        "checkpoint": Path(checkpoint).relative_to(ROOT).as_posix(),
        "head_equals_final": agree,
        "total": len(rows),
        "middle_rows": middle,
        "middle_head_argmax": [row["head_argmax"] for row in middle],
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    for row in middle:
        print(
            row["target"],
            "head",
            row["head_argmax"],
            "final",
            row["final_argmax"],
            "head_e4/e5",
            row["head_e4"],
            row["head_e5"],
            "final_e4/e5",
            row["final_e4"],
            row["final_e5"],
        )
    print("head_equals_final", agree, "/", len(rows), "mixture_keys", rows[0]["mixture_keys"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
