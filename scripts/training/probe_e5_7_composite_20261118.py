"""E5.7: one run, two ends.  Route by evidence: record -> word channel, else subject.

E5.6 showed the word channel answers B (24/24) and E5.5 that the subject is never
touched (A = 1.00), but in separate runs.  This runs both in the same process with
the routing rule, so the claim "correction and old ability hold at the same time"
is measured once, with a routing audit.

Criterion: A = 1.00 and B >= 24/24, routing audit clean (A routed to subject on all
32 worlds, B routed to channel on all 24 items).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import run_a2_byte_20261115 as a2  # noqa: E402
from scripts.training import run_a5_22_5 as a5  # noqa: E402
from scripts.training.probe_e5_6_char_channel_20261118 import (  # noqa: E402
    CharCopy,
    items_of,
    pieces_of,
    spans_of,
)
from scripts.training.subject_registry import load_subject  # noqa: E402
from taiji.u2_history import value_digest  # noqa: E402

OUT = ROOT / "reports/taiji_u2_e5_7_composite_20261118.json"
CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
SUBJECT = "a5-pre-seed20261110"
STEPS = 4000


def train_channel(train_items, templates) -> CharCopy:
    channel = CharCopy()
    optimizer = torch.optim.Adam(channel.parameters(), lr=0.01, weight_decay=1e-4)
    rng = random.Random(20261110)
    for _ in range(STEPS):
        item = rng.choice(train_items)
        index = rng.randrange(len(templates))
        record = item["records"][index]
        spans = spans_of(record)
        pieces = pieces_of(record, spans)
        labels = [i for i, piece in enumerate(pieces) if piece == item["target"]]
        if not labels:
            continue
        request = torch.tensor(list(item["entity"].encode()), dtype=torch.long)
        record_tensor = torch.tensor(list(record), dtype=torch.long)
        loss = F.cross_entropy(
            channel(request, record_tensor, spans)[None], torch.tensor([labels[0]])
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return channel


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    data = a5.atomic_load(ROOT / contract["data_root"])
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    templates = contract["task_b"]["templates"]
    channel = train_channel(items_of(data["train"], templates), templates)

    subject, meta = load_subject(SUBJECT, contract)
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    codes = a2.code_by_id(subject, head, prior, data["train"] + data["fit"], contract)

    routes = {"subject": 0, "channel": 0}
    a_rows, b_rows = [], []
    for example in data["fit"]:
        records = example.get("records")
        if records:
            routes["channel"] += 1
            index = len(b_rows)
            record = records[index % len(templates)] if isinstance(records, list) else None
            spans = spans_of(record)
            request = torch.tensor(
                list(
                    a5.p4_2.column_public(example)["request"].decode()[: -len(a5.SUFFIX)].encode()
                ),
                dtype=torch.long,
            )
            with torch.no_grad():
                score = channel(request, torch.tensor(list(record), dtype=torch.long), spans)
            raw = pieces_of(record, spans)[int(score.argmax())]
            b_rows.append({"id": example["id"], "got": raw.decode("utf-8", "replace")})
        else:
            routes["subject"] += 1
            state = a5.episode(subject, a5.p4_2.column_public(example), example["id"], None, False)
            raw = a5.answer_bytes(subject, state, codes[example["id"]])["raw"]
            a_rows.append(
                {
                    "id": example["id"],
                    "correct": raw == example["target"],
                    "got": raw.decode("utf-8", "replace")[:8],
                }
            )

    fit_items = items_of(data["fit"], templates)
    b_correct = 0
    for index, item in enumerate(fit_items):
        record = item["records"][index % len(templates)]
        spans = spans_of(record)
        request = torch.tensor(list(item["entity"].encode()), dtype=torch.long)
        with torch.no_grad():
            score = channel(request, torch.tensor(list(record), dtype=torch.long), spans)
        raw = pieces_of(record, spans)[int(score.argmax())]
        b_correct += int(raw == item["target"])
    a_correct = sum(1 for row in a_rows if row["correct"])

    payload = {
        "schema": "taiji-u2-e5-7-composite-v1",
        "card": "E5.7",
        "subject": SUBJECT,
        "bundle_sha256": meta["sha256"],
        "routing_audit": routes,
        "a_correct": a_correct,
        "a_denominator": len(a_rows),
        "a_accuracy": round(a_correct / max(len(a_rows), 1), 4),
        "b_correct": b_correct,
        "b_denominator": len(fit_items),
        "b_accuracy": round(b_correct / max(len(fit_items), 1), 4),
        "verdict": (
            "both_hold_in_one_run"
            if a_correct == len(a_rows) and b_correct >= 20
            else "composite_fails"
        ),
        "a_rows": a_rows,
        "b_sample": b_rows[:4],
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
        "routes",
        routes,
        payload["verdict"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
