"""Control: does task A survive more training on task A itself?

E5.2, E5.3 and the rehearsal arm all ended with task-A accuracy 0.25, which is the
constant-answer level.  Before attributing that drop to interference from task B,
train the same subject on task A alone and measure.  If A collapses here too, the
"damage" is not caused by B at all and the earlier attributions must be withdrawn.
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

OUT = ROOT / "reports/taiji_u2_e5_a_only_control_20261118.json"
CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
SUBJECT = "a5-pre-seed20261110"
MARKS = (0, 5, 20, 50)


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    data = a5.atomic_load(ROOT / contract["data_root"])
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    model, subject = load_subject(SUBJECT, contract)
    a5.set_trainable(model, contract)
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    codes = a2.code_by_id(model, head, prior, data["train"] + data["fit"], contract)
    positions = a2.cache_positions(model, data["train"], codes)
    optimizer = a5.optimizer_for(
        [p for p in model.parameters() if p.requires_grad], contract["optimizer"]
    )

    def accuracy() -> float:
        ok = 0
        for example in data["fit"]:
            state = a5.episode(model, a5.p4_2.column_public(example), example["id"], None, False)
            ok += int(
                a5.answer_bytes(model, state, codes[example["id"]])["raw"] == example["target"]
            )
        return ok / len(data["fit"])

    marks, step = [], 0
    for target in MARKS:
        while step < target:
            optimizer.zero_grad(set_to_none=True)
            a2.emission_loss(model, [positions[step % len(positions)]], use_code=True).backward()
            optimizer.step()
            model.parameter_revision += 1
            step += 1
        marks.append({"steps": step, "a_accuracy": round(accuracy(), 4)})
    payload = {
        "schema": "taiji-u2-e5-a-only-control-v1",
        "card": "E5",
        "subject": SUBJECT,
        "bundle_sha256": subject["sha256"],
        "marks": marks,
        "verdict": (
            "a_collapses_without_any_task_b" if marks[-1]["a_accuracy"] < 0.9 else "a_is_stable"
        ),
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps(payload["marks"], ensure_ascii=False), payload["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
