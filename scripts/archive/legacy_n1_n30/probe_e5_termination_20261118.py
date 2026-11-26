"""Why does emission never stop after the copied span?

Walks one task-B episode with a released record and prints, per emitted byte, the
number of live cursor positions, the copy distribution's extent and mass, and the
EOS margin against the best non-EOS symbol.  The question is whether the copy span
ever marks itself exhausted and, if it does, whether EOS becomes available.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import run_a2_byte_20261115 as a2  # noqa: E402
from scripts.training import run_a5_22_5 as a5  # noqa: E402
from taiji.u2_history import value_digest  # noqa: E402

OUT = ROOT / "reports/taiji_u2_e5_termination_20261118.json"
CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
STEPS = 14


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    data = a5.atomic_load(ROOT / contract["data_root"])
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    items, _ = a5.task_b_examples(data, contract)
    template_of = lambda index: index % len(contract["task_b"]["templates"])  # noqa: E731
    model = a5.build_subject(contract)
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    a2.code_by_id(model, head, prior, data["train"] + data["fit"], contract)

    item = items[1]
    public = a5.p4_2.column_public(item["example"])
    record = a5.record_of(item["records"][template_of(1)], item["example"]["id"])
    state = a5.episode(model, public, item["example"]["id"], [record], True, (record.record_id,))
    rows = []
    with torch.no_grad():
        for step in range(STEPS):
            probability, _, details = model.output(
                state, model.language_goal(state), state["prediction_action"]
            )
            mixture = details["mixture"]
            log_copy = mixture["log_copy"]
            valid = details["valid"]
            eos = float(probability[256])
            best_other = float(torch.cat([probability[:256], probability[257:]]).max())
            symbol = int(probability.argmax())
            rows.append(
                {
                    "step": step,
                    "symbol": symbol,
                    "char": bytes([symbol]).decode("latin-1") if symbol < 256 else "EOS",
                    "valid_positions": int(valid.sum()),
                    "copy_slots": int(log_copy.numel()),
                    "copy_mass": (
                        round(float(torch.exp(log_copy).sum()), 4) if log_copy.numel() else 0.0
                    ),
                    "eos_prob": round(eos, 3),
                    "best_non_eos": round(best_other, 3),
                    "eos_margin": round(eos - best_other, 3),
                }
            )
            if symbol == 256:
                break
            state = model.accept_byte(state, symbol, details)
    payload = {
        "schema": "taiji-u2-e5-termination-v1",
        "card": "E5",
        "item": item["example"]["id"],
        "target": item["target"].decode("utf-8", "replace"),
        "rows": rows,
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    for row in rows:
        print(
            row["step"],
            row["char"],
            "valid",
            row["valid_positions"],
            "copyslots",
            row["copy_slots"],
            "copymass",
            row["copy_mass"],
            "eos",
            row["eos_prob"],
            "margin",
            row["eos_margin"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
