"""E5.9: the record organ lives inside the subject; parent stays bit-identical.

Builds a subject with ``record_organ=True``, loads the registered parent weights on
top of it, trains only the organ's three tensors on train worlds, then measures both
ends through ``model.output`` itself:

* A - no record in the state -> the parent head answers (must be 32/32);
* B - a released record -> the organ routes and answers (must be >= 20/24).

The parent digest is taken over the parent's key set only, so attaching the organ
cannot hide a change to the parent.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import run_a2_byte_20261115 as a2  # noqa: E402
from scripts.training import run_a5_22_5 as a5  # noqa: E402
from scripts.training.probe_e5_6_char_channel_20261118 import items_of  # noqa: E402
from scripts.training.subject_registry import resolve  # noqa: E402
from taiji.u2_history import value_digest  # noqa: E402
from taiji.u2_record_organ import windows_of  # noqa: E402
from taiji.u2_subject import U2Config, U2Subject  # noqa: E402

OUT = ROOT / "reports/taiji_u2_e5_9_organ_20261118.json"
CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
SUBJECT = "a5-pre-seed20261110"
STEPS = 4000


def _organ_config(contract, with_organ: bool):
    from dataclasses import replace

    return replace(
        U2Config.small(),
        seed=int(contract["candidate"]["seed"]),
        record_organ=with_organ,
    )


def organ_scores(organ, request: bytes, record: bytes):
    windows = windows_of(record)
    pieces = [torch.tensor(list(window), dtype=torch.long) for window in windows]
    keys = organ.key(torch.stack([organ.byte(piece).mean(0) for piece in pieces]))
    text = organ.byte(torch.tensor(list(request), dtype=torch.long))
    pooled = torch.cat([text.mean(0), text.amax(0)], -1)
    query = organ.query(pooled)
    return keys @ query / (query.shape[-1] ** 0.5)


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
    parent_keys = {name: value for name, value in reference.state_dict().items()}
    before = value_digest({k: v for k, v in parent_keys.items()})

    model = U2Subject(_organ_config(contract, True)).metadata_candidate().request_candidate()
    loaded = torch.load(ROOT / resolve(SUBJECT)["path"], weights_only=False)["model"]
    model.load_state_dict(loaded, strict=False)
    model.eval()
    attached = value_digest({k: v for k, v in model.state_dict().items() if k in parent_keys})

    started = time.process_time()
    organ = model.record_organ
    optimizer = torch.optim.Adam(organ.parameters(), lr=0.01, weight_decay=1e-4)
    rng = random.Random(20261110)
    for _ in range(STEPS):
        item = rng.choice(train_items)
        index = rng.randrange(len(templates))
        record = item["records"][index]
        windows = windows_of(record)
        labels = [i for i, window in enumerate(windows) if window == item["target"]]
        if not labels:
            continue
        scores = organ_scores(organ, item["entity"].encode(), record)
        loss = F.cross_entropy(scores[None], torch.tensor([labels[0]]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    codes = a2.code_by_id(model, head, prior, data["train"] + data["fit"], contract)

    a_correct, a_rows = 0, []
    for example in data["fit"]:
        state = a5.episode(model, a5.p4_2.column_public(example), example["id"], None, False)
        raw = a5.answer_bytes(model, state, codes[example["id"]])["raw"]
        ok = raw == example["target"]
        a_correct += int(ok)
        a_rows.append({"id": example["id"], "correct": ok})

    b_correct, b_rows = 0, []
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
        ok = raw == item["target"]
        b_correct += int(ok)
        b_rows.append(
            {
                "id": item["id"],
                "target": item["target"].decode(),
                "got": raw.decode("utf-8", "replace"),
                "correct": ok,
                "organ_window": state.get("organ_window", b"").decode("utf-8", "replace"),
            }
        )

    after = value_digest({k: v for k, v in model.state_dict().items() if k in parent_keys})
    payload = {
        "schema": "taiji-u2-e5-9-organ-v1",
        "card": "E5.9",
        "subject": SUBJECT,
        "parent_digest_before": before,
        "parent_digest_after_load": attached,
        "parent_digest_after_training": after,
        "parent_bit_identical": before == attached == after,
        "a_correct": a_correct,
        "a_denominator": len(a_rows),
        "b_correct": b_correct,
        "b_denominator": len(fit_items),
        "organ_tensors": [
            name for name, _ in model.named_parameters() if name.startswith("record_organ")
        ],
        "verdict": (
            "organ_inside_subject_passes"
            if before == attached == after and a_correct == len(a_rows) and b_correct >= 20
            else "organ_inside_subject_fails"
        ),
        "cpu": round(time.process_time() - started, 2),
        "b_rows": b_rows,
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "A",
        a_correct,
        "/",
        len(a_rows),
        "B",
        b_correct,
        "/",
        len(fit_items),
        "parent_identical",
        before == attached == after,
        payload["verdict"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
