"""Does the subject terminate anywhere, or only in the record state?

Before claiming "no stopping rule was learned", test the same subject on task A,
where it is reported to answer 32/32 exactly.  If EOS appears there and never with
a released record, the stopping rule exists and simply does not transfer to the
record state; if EOS never appears anywhere, the subject never learned to stop.
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

OUT = ROOT / "reports/taiji_u2_e5_eos_contrast_20261118.json"
CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
COUNT = 4


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
    codes = a2.code_by_id(model, head, prior, data["train"] + data["fit"], contract)

    a_rows, b_rows = [], []
    for example in data["fit"][:COUNT]:
        public = a5.p4_2.column_public(example)
        state = a5.episode(model, public, example["id"], None, False)
        got = a5.answer_bytes(model, state, codes[example["id"]])
        a_rows.append(
            {
                "id": example["id"],
                "eos": got["eos"],
                "len": len(got["raw"]),
                "target": example.get("label", ""),
                "exact": got["raw"] == example["target"],
            }
        )
    for index, item in enumerate(items[:COUNT]):
        public = a5.p4_2.column_public(item["example"])
        record = a5.record_of(item["records"][template_of(index)], item["example"]["id"])
        state = a5.episode(
            model, public, item["example"]["id"], [record], True, (record.record_id,)
        )
        got = a5.answer_bytes(model, state, codes[item["example"]["id"]])
        b_rows.append(
            {
                "id": item["example"]["id"],
                "eos": got["eos"],
                "len": len(got["raw"]),
                "exact": got["raw"] == item["target"],
            }
        )
    payload = {
        "schema": "taiji-u2-e5-eos-contrast-v1",
        "card": "E5",
        "task_a": a_rows,
        "task_b_with_record": b_rows,
        "a_eos_rate": round(sum(row["eos"] for row in a_rows) / max(len(a_rows), 1), 3),
        "b_eos_rate": round(sum(row["eos"] for row in b_rows) / max(len(b_rows), 1), 3),
        "reading": (
            "stopping_rule_exists_but_does_not_transfer"
            if sum(row["eos"] for row in a_rows) > 0 and sum(row["eos"] for row in b_rows) == 0
            else "no_stopping_rule_anywhere"
        ),
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        json.dumps(
            {key: payload[key] for key in ("a_eos_rate", "b_eos_rate", "reading")},
            ensure_ascii=False,
        )
    )
    for row in a_rows:
        print("A", row["id"], "eos", row["eos"], "len", row["len"], "exact", row["exact"])
    for row in b_rows:
        print("B", row["id"], "eos", row["eos"], "len", row["len"], "exact", row["exact"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
