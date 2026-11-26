"""E5.10: the organ must also decline an irrelevant record.

E5.9 routed every record to the organ.  A record that does not concern the asked
entity must not override the observation, so this adds a relevance head and a second
item family:

* conflict  - record about the queried entity, position differs -> answer the record;
* irrelevant - record about another entity                      -> answer the observation.

Criteria: A 32/32, B (conflict) >= 20/24, D (irrelevant) >= 20/24, parent digest
unchanged over the parent's key set.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import run_a2_byte_20261115 as a2  # noqa: E402
from scripts.training import run_a5_22_5 as a5  # noqa: E402
from scripts.training.probe_e5_6_char_channel_20261118 import items_of  # noqa: E402
from scripts.training.probe_e5_9_organ_20261118 import _organ_config  # noqa: E402
from scripts.training.subject_registry import resolve  # noqa: E402
from taiji.u2_history import value_digest  # noqa: E402
from taiji.u2_record_organ import windows_of  # noqa: E402
from taiji.u2_subject import U2Subject  # noqa: E402

OUT = ROOT / "reports/taiji_u2_e5_11_sentinel_20261118.json"
CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
SUBJECT = "a5-pre-seed20261110"
STEPS = 6000


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    data = a5.atomic_load(ROOT / contract["data_root"])
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    templates = contract["task_b"]["templates"]
    train_items = items_of(data["train"], templates)
    fit_items = items_of(data["fit"], templates)

    reference = a5.build_subject(contract)
    reference.load_state_dict(
        torch.load(ROOT / resolve(SUBJECT)["path"], weights_only=False)["model"], strict=False
    )
    parent_keys = set(reference.state_dict())
    before = value_digest({k: v for k, v in reference.state_dict().items()})

    model = U2Subject(_organ_config(contract, True)).metadata_candidate().request_candidate()
    model.load_state_dict(
        torch.load(ROOT / resolve(SUBJECT)["path"], weights_only=False)["model"], strict=False
    )
    model.eval()
    organ = model.record_organ
    optimizer = torch.optim.Adam(organ.parameters(), lr=0.01, weight_decay=1e-4)
    started = time.process_time()
    rng = random.Random(20261110)
    for _ in range(STEPS):
        item = rng.choice(train_items)
        other = rng.choice(train_items)
        index = rng.randrange(len(templates))
        record = item["records"][index]
        windows = windows_of(record)
        hits = [i for i, window in enumerate(windows) if window == item["target"]]
        if not hits or len(windows) < 1:
            continue
        attention = organ.attention(item["entity"].encode(), record)
        # Pointer-sum loss: the sentinel or any matching window may carry the target.
        # The conflict case must be answered by the pointer: the gate may not carry it.
        loss = -torch.log(attention[hits].sum().clamp_min(1e-9))
        negative_record = (
            templates[index].format(entity=other["entity"], label=other["target"].decode()).encode()
        )
        attention_negative = organ.attention(item["entity"].encode(), negative_record)
        # No window matches, so the only way to satisfy the target is the sentinel.
        loss_negative = -torch.log(attention_negative[-1].clamp_min(1e-9))
        optimizer.zero_grad(set_to_none=True)
        (loss + loss_negative).backward()
        optimizer.step()

    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    codes = a2.code_by_id(model, head, prior, data["train"] + data["fit"], contract)

    a_correct = 0
    for example in data["fit"]:
        state = a5.episode(model, a5.p4_2.column_public(example), example["id"], None, False)
        a_correct += int(
            a5.answer_bytes(model, state, codes[example["id"]])["raw"] == example["target"]
        )

    b_correct, d_correct = 0, 0
    b_rows, d_rows = [], []
    for index, item in enumerate(fit_items):
        example = next(entry for entry in data["fit"] if entry["id"] == item["id"])
        record = a5.record_of(item["records"][index % len(templates)], item["id"])
        state = a5.episode(
            model,
            a5.p4_2.column_public(example),
            example["id"],
            [record],
            True,
            (record.record_id,),
        )
        raw = a5.answer_bytes(model, state, codes[example["id"]])["raw"]
        b_correct += int(raw == item["target"])
        b_rows.append(
            {
                "id": item["id"],
                "got": raw.decode("utf-8", "replace"),
                "correct": raw == item["target"],
            }
        )

        other = fit_items[(index + 1) % len(fit_items)]
        distractor = (
            templates[index % len(templates)]
            .format(entity=other["entity"], label=other["target"].decode())
            .encode()
        )
        off = a5.record_of(distractor, item["id"] + "-other")
        state_d = a5.episode(
            model, a5.p4_2.column_public(example), example["id"], [off], True, (off.record_id,)
        )
        raw_d = a5.answer_bytes(model, state_d, codes[example["id"]])["raw"]
        d_correct += int(raw_d == example["target"])
        d_rows.append(
            {
                "id": item["id"],
                "got": raw_d.decode("utf-8", "replace"),
                "correct": raw_d == example["target"],
            }
        )

    after = value_digest({k: v for k, v in model.state_dict().items() if k in parent_keys})
    passed = (
        before == after and a_correct == len(data["fit"]) and b_correct >= 20 and d_correct >= 20
    )
    payload = {
        "schema": "taiji-u2-e5-11-sentinel-v1",
        "card": "E5.11",
        "subject": SUBJECT,
        "a_correct": a_correct,
        "a_denominator": len(data["fit"]),
        "b_correct": b_correct,
        "b_denominator": len(fit_items),
        "d_correct": d_correct,
        "d_denominator": len(fit_items),
        "parent_unchanged": before == after,
        "verdict": "sentinel_passes" if passed else "sentinel_fails",
        "cpu": round(time.process_time() - started, 2),
        "b_rows": b_rows,
        "d_rows": d_rows,
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "A",
        a_correct,
        "/",
        len(data["fit"]),
        "B",
        b_correct,
        "/",
        len(fit_items),
        "D",
        d_correct,
        "/",
        len(fit_items),
        "parent",
        before == after,
        payload["verdict"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
