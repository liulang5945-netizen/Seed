"""Is the copy path inert? Zero-training probe on task B's released record.

All three E5 negatives share one shape: the only way to use a released fact is to
overwrite shared weights, and that damages the old ability monotonically.  If the
copy path carried any mass, the released record could change the answer with no
update at all.  This measures, on the same episode, the copy share of the emission
mixture with and without the released record, and whether the answer changes.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import run_a2_byte_20261115 as a2  # noqa: E402
from scripts.training import run_a5_22_5 as a5  # noqa: E402
from taiji.u2_history import value_digest  # noqa: E402

OUT = ROOT / "reports/taiji_u2_e5_copy_share_20261118.json"
CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
ITEMS = 8


def first_step(model, state):
    with torch.no_grad():
        probability, _, details = model.output(
            state, model.language_goal(state), state["prediction_action"]
        )
    mixture = details["mixture"]
    log_copy = mixture["log_copy"]
    copy_mass = 0.0 if log_copy.numel() == 0 else float(torch.exp(log_copy).sum())
    gen_mass = float(torch.exp(mixture["log_gen"]).sum())
    return {
        "argmax": int(probability.argmax()),
        "copy_slots": int(log_copy.numel()),
        "log_copy_max": None if log_copy.numel() == 0 else round(float(log_copy.max()), 4),
        "log_gen_max": round(float(mixture["log_gen"].max()), 4),
        "copy_mass": round(copy_mass, 6),
        "gen_mass": round(gen_mass, 6),
    }


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

    rows = []
    for index, item in enumerate(items[:ITEMS]):
        public = a5.p4_2.column_public(item["example"])
        template = template_of(index)
        record = a5.record_of(item["records"][template], item["example"]["id"])
        code = codes[item["example"]["id"]]
        without = a5.episode(model, public, item["example"]["id"], None, False)
        with_record = a5.episode(
            model,
            public,
            item["example"]["id"],
            [record],
            True,
            (record.record_id,),
        )
        first = first_step(model, without)
        second = first_step(model, with_record)
        answered = a5.answer_bytes(model, with_record, code)
        correct = answered["raw"] == item["target"]
        rows.append(
            {
                "id": item["example"]["id"],
                "target": item["target"].decode("utf-8", "replace"),
                "no_record": first,
                "with_record": second,
                "answer_changed": first["argmax"] != second["argmax"],
                "answer_correct": correct,
                "answer_text": answered["raw"].decode("utf-8", "replace"),
                "copy_gain": round(second["copy_mass"] - first["copy_mass"], 6),
                "record_id": record.record_id,
                "code_dim": int(code.numel()) if hasattr(code, "numel") else 0,
            }
        )
    changed = sum(1 for row in rows if row["answer_changed"])
    correct_count = sum(1 for row in rows if row["answer_correct"])
    copy_alive = sum(
        1
        for row in rows
        if math.isfinite(row["with_record"]["log_copy_max"])
        and row["with_record"]["copy_mass"] > 1e-6
    )
    payload = {
        "schema": "taiji-u2-e5-copy-share-v1",
        "card": "E5",
        "items": len(rows),
        "answers_changed_at_zero_updates": changed,
        "answers_correct_at_zero_updates": correct_count,
        "items_with_copy_mass": copy_alive,
        "rows": rows,
        "verdict": ("copy_path_live" if copy_alive and changed else "copy_path_inert"),
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "changed",
        changed,
        "/",
        len(rows),
        "copy_mass_items",
        copy_alive,
        "verdict",
        payload["verdict"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
