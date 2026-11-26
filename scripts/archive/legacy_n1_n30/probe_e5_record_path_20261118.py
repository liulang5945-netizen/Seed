"""Where does the released record stop: cursor, copy mass, or the mixture?

Measured on the trained A.5 subject with a calibration gate: it first reproduces
the published 32/32 on task A and aborts otherwise.  For each task-B item the same
episode is built twice, without and with the legally released record, and the first
emission's copy/gen/branch distributions are compared.

Decision rules, fixed before running:
  * copy slots 0 or copy mass 0 with the record present -> the record never reaches
    the cursor (wiring);
  * copy mass present and its argmax equals the copied symbol, yet the final argmax
    follows generation -> the mixture or the branch score suppresses copy;
  * final follows copy but the symbol is wrong -> the record content is misaligned.
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
from scripts.training.subject_registry import load_subject  # noqa: E402
from taiji.u2_history import value_digest  # noqa: E402

OUT = ROOT / "reports/taiji_u2_e5_record_path_20261118.json"
CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
SUBJECT = "a5-pre-seed20261110"
FLOOR = 0.90
ITEMS = 4


def snapshot(model, state, code):
    with torch.no_grad(), a2.reading_code(model, code):
        probability, _, details = model.output(
            state, model.language_goal(state), state["prediction_action"]
        )
    mixture = details["mixture"]
    log_copy, log_gen, log_branch = (
        mixture["log_copy"],
        mixture["log_gen"],
        mixture["log_branch"],
    )
    copy_mass = float(torch.exp(log_copy).sum()) if log_copy.numel() else 0.0
    copy_arg = int(log_copy.argmax()) if log_copy.numel() else -1
    return {
        "copy_slots": int(log_copy.numel()),
        "copy_mass": round(copy_mass, 6),
        "copy_argmax": copy_arg,
        "gen_argmax": int(log_gen.argmax()),
        "branch_argmax": int(log_branch.argmax()),
        "final_argmax": int(probability.argmax()),
        "valid_positions": int(details["valid"].sum()),
        "record_slots": int(mixture["log_copy"].shape[-1]) if log_copy.numel() else 0,
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

    model, subject = load_subject(SUBJECT, contract)
    a5.set_trainable(model, contract)
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    codes = a2.code_by_id(model, head, prior, data["train"] + data["fit"], contract)

    calibration = 0
    for example in data["fit"]:
        state = a5.episode(model, a5.p4_2.column_public(example), example["id"], None, False)
        got = a5.answer_bytes(model, state, codes[example["id"]])
        calibration += int(got["raw"] == example["target"])
    accuracy = calibration / len(data["fit"])
    result = {
        "schema": "taiji-u2-e5-record-path-v1",
        "card": "E5",
        "subject": SUBJECT,
        "bundle": subject["path"],
        "bundle_sha256": subject["sha256"],
        "bundle_update": subject["update"],
        "calibration_a_accuracy": round(accuracy, 4),
    }
    if accuracy < FLOOR:
        result["status"] = "bundle_does_not_reproduce"
        OUT.write_bytes(json.dumps(result, ensure_ascii=False, indent=2).encode() + b"\n")
        print("REPRODUCTION FAILED", round(accuracy, 4))
        return 1

    rows = []
    for index, item in enumerate(items[:ITEMS]):
        public = a5.p4_2.column_public(item["example"])
        record = a5.record_of(item["records"][template_of(index)], item["example"]["id"])
        code = codes[item["example"]["id"]]
        without = snapshot(
            model,
            a5.episode(model, public, item["example"]["id"], None, False),
            code,
        )
        with_record = snapshot(
            model,
            a5.episode(model, public, item["example"]["id"], [record], True, (record.record_id,)),
            code,
        )
        rows.append(
            {
                "id": item["example"]["id"],
                "target_first_byte": int(item["target"][0]),
                "record_len": len(record.raw) if hasattr(record, "raw") else None,
                "without_record": without,
                "with_record": with_record,
            }
        )
    result.update({"status": "measured", "items": ITEMS, "rows": rows})
    OUT.write_bytes(json.dumps(result, ensure_ascii=False, indent=2).encode() + b"\n")
    for row in rows:
        w, wo = row["with_record"], row["without_record"]
        print(
            row["id"],
            "target_byte",
            row["target_first_byte"],
            "| WITHOUT copy_slots",
            wo["copy_slots"],
            "mass",
            wo["copy_mass"],
            "final",
            wo["final_argmax"],
            "| WITH copy_slots",
            w["copy_slots"],
            "mass",
            w["copy_mass"],
            "copy_arg",
            w["copy_argmax"],
            "gen_arg",
            w["gen_argmax"],
            "final",
            w["final_argmax"],
            "valid",
            w["valid_positions"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
