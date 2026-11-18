"""Decode with the byte head alone: is the mixture eating the correct answers?

`head argmax` and the mixed `final argmax` agree on only 19 of 32 fit worlds.  This
decodes the same 32 worlds using the head's own logits (no copy/generation
mixture, EOS masked at the first byte) and reports the confusion.  If the head-only
decode reaches the gate while the mixed one does not, the mixture is the defect;
otherwise the head is.
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
    LABELS,
    generate_examples,
    load_learner,
    prepare,
)  # noqa: E402
from taiji.utf8_state import advance_utf8  # noqa: E402

OUT = ROOT / "reports/taiji_u2_head_decode_20261118.json"


def head_decode(model, public, identity) -> bytes:
    state = prepare(model, public, identity)
    data = bytearray()
    with torch.no_grad():
        for _ in range(65):
            context = model.branch_context(
                state, model.language_goal(state), state["prediction_action"]
            )
            previous = model.previous_embedding(state["previous_byte"])[None].expand(
                model.c.branches, -1
            )
            logits = model.byte_head(torch.cat([context, previous], -1))[0]
            remaining = model.c.max_output_bytes - state["data_bytes"]
            if remaining <= 0:
                symbol = 256
            else:
                allowed = torch.ones_like(logits, dtype=torch.bool)
                if state["utf8_remaining"] == 0:
                    allowed[0x80:0xC0] = False
                else:
                    allowed[:0x80] = False
                    allowed[0xC0:] = False
                if not data:
                    allowed[256] = False
                symbol = int(torch.where(allowed, logits, torch.full_like(logits, -1e9)).argmax())
            child = dict(state)
            if symbol == 256:
                return bytes(data)
            data.append(symbol)
            child["previous_byte"] = symbol
            child["data_bytes"] = state["data_bytes"] + 1
            child["utf8_remaining"], child["utf8_lead"] = advance_utf8(
                state["utf8_remaining"], state["utf8_lead"], symbol
            )
            state = child
    return bytes(data)


def main() -> int:
    checkpoint = sorted(glob.glob(str(ROOT / "output/a_pixel_parallel_*/segment.bin")))[-1]
    model, _, _ = load_learner(Path(checkpoint))
    table = [[0] * 4 for _ in range(4)]
    correct = 0
    outputs = []
    for example in generate_examples("fit", 32):
        content = head_decode(model, example["public"], example["id"])
        text = content.decode("utf-8", "replace")
        outputs.append(text)
        truth = LABELS.index(example["target"].decode())
        guess = LABELS.index(text) if text in LABELS else 0
        table[truth][guess] += 1
        correct += int(content == example["target"])
    payload = {
        "schema": "taiji-u2-head-decode-v1",
        "card": "22.5.0.A",
        "checkpoint": Path(checkpoint).relative_to(ROOT).as_posix(),
        "head_only_correct": correct,
        "head_only_denominator": 32,
        "head_only_confusion": table,
        "outputs": outputs,
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        json.dumps(
            {key: payload[key] for key in ("head_only_correct", "head_only_confusion")},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
