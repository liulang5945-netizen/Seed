"""Dose-response for A.5's correction: accumulated updates per error, 8/16/32/64.

A.5's zero is protocol-caused (one update, immediate retest, restore on failure).
Its A.4 arm accumulated eight updates per error and reached 0.333.  This probe
sweeps the dose on the same first pass, and measures old-ability damage by hand
because the frozen retention gate cannot fail by construction.

Read-only over the frozen contract: the dose is a parameter here, the manifest is
untouched.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import run_a2_byte_20261115 as a2  # noqa: E402
from scripts.training import run_a5_22_5 as a5  # noqa: E402
from taiji.u2_history import value_digest  # noqa: E402

OUT = ROOT / "reports/taiji_u2_e5_dose_20261118.json"
DOSES = (8, 16, 32, 64)
CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"


def main(argv: list[str] | None = None) -> int:
    selected = [int(value) for value in argv] if argv else list(DOSES)
    if OUT.exists():
        raise FileExistsError(OUT)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    data = a5.atomic_load(ROOT / contract["data_root"])
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    items, _ = a5.task_b_examples(data, contract)
    template_of = lambda index: index % len(contract["task_b"]["templates"])  # noqa: E731
    base_model = a5.build_subject(contract)
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    codes = a2.code_by_id(base_model, head, prior, data["train"] + data["fit"], contract)

    rows = []
    for dose in selected:
        started = time.process_time()
        model = a5.build_subject(contract)
        names = a5.set_trainable(model, contract)
        before = [a5.a_round(model, example, codes[example["id"]]) for example in data["fit"]]
        first = a5.b_pass(model, items, codes, template_of)
        errors = [item for item, row in zip(items, first, strict=True) if not row["correct"]]
        optimizer = a5.optimizer_for(
            [p for p in model.parameters() if p.requires_grad], contract["feedback_optimizer"]
        )
        for item in errors:
            public = a5.p4_2.column_public(item["example"])
            code = codes[item["example"]["id"]]
            template = template_of(items.index(item))
            for _ in range(dose):
                optimizer.zero_grad(set_to_none=True)
                loss = a5.record_loss(
                    model,
                    public,
                    item["example"]["id"],
                    item["records"][template],
                    item["target"],
                    code,
                )
                loss.backward()
                optimizer.step()
                model.parameter_revision += 1
        after = a5.b_pass(model, items, codes, template_of)
        corrected = sum(
            1
            for item, row in zip(items, after, strict=True)
            if row["correct"] is not False and item in errors
        )
        correct_after = [
            a5.a_round(model, example, codes[example["id"]]) for example in data["fit"]
        ]
        before_accuracy = a5.accuracy(before)
        after_accuracy = a5.accuracy(correct_after)
        rows.append(
            {
                "dose": dose,
                "errors": len(errors),
                "updates": dose * len(errors),
                "corrected_first_period": corrected,
                "correction_rate": round(corrected / max(len(errors), 1), 4),
                "fit_before": round(before_accuracy, 4),
                "fit_after": round(after_accuracy, 4),
                "damage": round(before_accuracy - after_accuracy, 4),
                "cpu": round(time.process_time() - started, 2),
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False))
    payload = {
        "schema": "taiji-u2-e5-dose-v1",
        "card": "E5",
        "doses": rows,
        "verdict": (
            "dose_solves"
            if max(row["correction_rate"] for row in rows) >= 0.80
            else "dose_insufficient"
        ),
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print("verdict", payload["verdict"])
    del names
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
