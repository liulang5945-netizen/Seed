"""E5.2: train only the emission/record families on task B and watch both ends.

Frozen pre-registration: PREREG-TAIJI_U2_E5_2_EMISSION_ONLY_20261118.md.
Pass means B exact >= 20/24 with A damage <= 0.02; anything else is a recorded exit,
not a reason to add rounds.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import run_a2_byte_20261115 as a2  # noqa: E402
from scripts.training import run_a5_22_5 as a5  # noqa: E402
from scripts.training.subject_registry import load_subject  # noqa: E402
from taiji.u2_history import value_digest  # noqa: E402

REHEARSAL = os.environ.get("TAIJI_E5_REHEARSAL") == "1"
_ONLY = os.environ.get("TAIJI_E5_ONLY_GROUP", "").strip().strip(".") or "none"
_SUFFIX = "rehearsal" if REHEARSAL else _ONLY
OUT = (
    ROOT / f"reports/taiji_u2_e5_{_SUFFIX}_20261118.json"
    if _SUFFIX != "none"
    else (
        ROOT / "reports/taiji_u2_e5_3_frozen_head_20261118.json"
        if os.environ.get("TAIJI_E5_FREEZE_BYTE_HEAD") == "1"
        else ROOT / "reports/taiji_u2_e5_2_emission_only_20261118.json"
    )
)
CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
SUBJECT = "a5-pre-seed20261110"
EPOCHS = 12 if REHEARSAL else 25
B_PASS = 20


def measure(model, data, items, codes, template_of) -> dict:
    a_ok = 0
    for example in data["fit"]:
        state = a5.episode(model, a5.p4_2.column_public(example), example["id"], None, False)
        a_ok += int(a5.answer_bytes(model, state, codes[example["id"]])["raw"] == example["target"])
    b_ok = 0
    b_text = []
    for index, item in enumerate(items):
        public = a5.p4_2.column_public(item["example"])
        record = a5.record_of(item["records"][template_of(index)], item["example"]["id"])
        state = a5.episode(
            model, public, item["example"]["id"], [record], True, (record.record_id,)
        )
        got = a5.answer_bytes(model, state, codes[item["example"]["id"]])
        b_ok += int(got["raw"] == item["target"])
        b_text.append(got["raw"].decode("utf-8", "replace")[:8])
    return {
        "a_correct": a_ok,
        "a_denominator": len(data["fit"]),
        "a_accuracy": round(a_ok / len(data["fit"]), 4),
        "b_correct": b_ok,
        "b_denominator": len(items),
        "b_accuracy": round(b_ok / len(items), 4),
        "b_sample": b_text[:6],
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
    names = a5.set_trainable(model, contract)
    freeze_head = os.environ.get("TAIJI_E5_FREEZE_BYTE_HEAD") == "1"
    only_group = os.environ.get("TAIJI_E5_ONLY_GROUP", "").strip()
    if freeze_head:
        for name, parameter in model.named_parameters():
            if name.startswith("byte_head."):
                parameter.requires_grad_(False)
    if only_group:
        for name, parameter in model.named_parameters():
            if not name.startswith(only_group):
                parameter.requires_grad_(False)
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    codes = a2.code_by_id(model, head, prior, data["train"] + data["fit"], contract)

    started = time.process_time()
    before = measure(model, data, items, codes, template_of)
    a_items = a2.cache_positions(model, data["train"], codes) if REHEARSAL else []
    trainable = [p for p in model.parameters() if p.requires_grad]
    feedback_spec = dict(contract["feedback_optimizer"])
    if os.environ.get("TAIJI_E5_LR"):
        feedback_spec["lr"] = float(os.environ["TAIJI_E5_LR"])
    optimizer = a5.optimizer_for(trainable, feedback_spec)
    updates = 0
    for _epoch in range(EPOCHS):
        for index, item in enumerate(items):
            public = a5.p4_2.column_public(item["example"])
            template = template_of(index)
            optimizer.zero_grad(set_to_none=True)
            loss = a5.record_loss(
                model,
                public,
                item["example"]["id"],
                item["records"][template],
                item["target"],
                codes[item["example"]["id"]],
            )
            loss.backward()
            optimizer.step()
            model.parameter_revision += 1
            updates += 1
            if REHEARSAL:
                example = a_items[updates % len(a_items)]
                optimizer.zero_grad(set_to_none=True)
                a2.emission_loss(model, [example], use_code=True).backward()
                optimizer.step()
                model.parameter_revision += 1
                updates += 1
    after = measure(model, data, items, codes, template_of)

    damage = round(before["a_accuracy"] - after["a_accuracy"], 4)
    if after["b_correct"] >= B_PASS and damage <= 0.02:
        verdict = "passes"
    elif after["b_correct"] >= B_PASS:
        verdict = "tradeoff_with_emission_only"
    else:
        verdict = "pointer_training_insufficient"
    payload = {
        "schema": "taiji-u2-e5-2-emission-only-v1",
        "card": "E5.2",
        "prereg": "plans/reference/PREREG-TAIJI_U2_E5_2_EMISSION_ONLY_20261118.md",
        "subject": SUBJECT,
        "bundle": subject["path"],
        "bundle_sha256": subject["sha256"],
        "bundle_update": subject["update"],
        "trainable_tensors": len(trainable),
        "trainable_names": names,
        "frozen_byte_head": freeze_head,
        "only_group": only_group,
        "rehearsal": REHEARSAL,
        "lr": feedback_spec["lr"],
        "updates": updates,
        "before": before,
        "after": after,
        "damage": damage,
        "verdict": verdict,
        "cpu": round(time.process_time() - started, 2),
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "before A",
        before["a_accuracy"],
        "B",
        before["b_accuracy"],
        "| after A",
        after["a_accuracy"],
        "B",
        after["b_correct"],
        "/",
        after["b_denominator"],
        "| damage",
        damage,
        "| verdict",
        verdict,
        "| cpu",
        payload["cpu"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
