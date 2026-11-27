"""Definitive EOS contrast on the trained A.5 subject.

Two earlier probes measured an untrained subject and were void.  This one loads the
bundle A.5 itself saved at the end of A training (`bundle-pre-feedback.pt`), first
reproduces A.5's published reading (task A fit 32/32), and aborts without drawing
any conclusion if that reproduction fails.

Only then does it compare termination on task A with termination in the record
state, which decides between "the stopping rule exists but does not transfer" and
"no stopping rule was learned".
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

OUT = ROOT / "reports/taiji_u2_e5_eos_contrast_v2_20261118.json"
CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
BUNDLE = ROOT / "data/experiments/taiji_u2/a5/20261116T0520Z/seed-20261110/bundle-pre-feedback.pt"
REPRODUCTION_FLOOR = 0.90
B_ITEMS = 8


def decode(model, public, identity, records, release, ids, code):
    state = a5.episode(model, public, identity, records, release, ids)
    return a5.answer_bytes(model, state, code)


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
    payload = torch.load(BUNDLE, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    a5.set_trainable(model, contract)
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    codes = a2.code_by_id(model, head, prior, data["train"] + data["fit"], contract)

    a_rows = []
    for example in data["fit"]:
        got = decode(
            model,
            a5.p4_2.column_public(example),
            example["id"],
            None,
            False,
            None,
            codes[example["id"]],
        )
        a_rows.append(
            {
                "id": example["id"],
                "exact": got["raw"] == example["target"],
                "eos": got["eos"],
                "len": len(got["raw"]),
            }
        )
    a_accuracy = sum(row["exact"] for row in a_rows) / len(a_rows)
    payload_out = {
        "schema": "taiji-u2-e5-eos-contrast-v2",
        "card": "E5",
        "bundle": BUNDLE.relative_to(ROOT).as_posix(),
        "bundle_update": int(payload.get("update", -1)),
        "a_accuracy": round(a_accuracy, 4),
        "a_eos_rate": round(sum(row["eos"] for row in a_rows) / len(a_rows), 4),
        "a_rows": a_rows,
    }
    if a_accuracy < REPRODUCTION_FLOOR:
        payload_out["status"] = "bundle_does_not_reproduce"
        payload_out["reading"] = None
        OUT.write_bytes(json.dumps(payload_out, ensure_ascii=False, indent=2).encode() + b"\n")
        print("REPRODUCTION FAILED", round(a_accuracy, 4))
        return 1

    b_rows = []
    for index, item in enumerate(items[:B_ITEMS]):
        record = a5.record_of(item["records"][template_of(index)], item["example"]["id"])
        got = decode(
            model,
            a5.p4_2.column_public(item["example"]),
            item["example"]["id"],
            [record],
            True,
            (record.record_id,),
            codes[item["example"]["id"]],
        )
        b_rows.append(
            {
                "id": item["example"]["id"],
                "exact": got["raw"] == item["target"],
                "eos": got["eos"],
                "len": len(got["raw"]),
                "text": got["raw"].decode("utf-8", "replace")[:24],
            }
        )
    b_eos = sum(row["eos"] for row in b_rows) / len(b_rows)
    b_exact = sum(row["exact"] for row in b_rows) / len(b_rows)
    payload_out.update(
        {
            "status": "measured",
            "b_items": len(b_rows),
            "b_exact_rate": round(b_exact, 4),
            "b_eos_rate": round(b_eos, 4),
            "b_rows": b_rows,
            "reading": (
                "terminates_but_ignores_record"
                if b_eos >= 0.9 and b_exact < 0.1
                else (
                    "stopping_rule_does_not_transfer"
                    if b_eos < payload_out["a_eos_rate"]
                    else "inconclusive"
                )
            ),
        }
    )
    OUT.write_bytes(json.dumps(payload_out, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "A_acc",
        payload_out["a_accuracy"],
        "A_eos",
        payload_out["a_eos_rate"],
        "B_exact",
        payload_out["b_exact_rate"],
        "B_eos",
        payload_out["b_eos_rate"],
        "reading",
        payload_out["reading"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
