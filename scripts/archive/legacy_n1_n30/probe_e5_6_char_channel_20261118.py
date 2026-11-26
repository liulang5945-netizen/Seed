"""E5.6: character-aligned copying removes the first-byte drift.

The byte-level channel failed 5/24 by copying the right last two bytes and the
wrong first one (中 E4B8AD emitted as 席 E5B8AD): independent per-byte addressing
lets the pointer land in a neighbouring glyph.  This version attends over whole
characters and emits the chosen character's own bytes, so a copy is one binding to
one object rather than three independent byte picks.

Criterion unchanged: B >= 20/24 with A = 1.00 (the subject is never touched).
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import run_a2_byte_20261115 as a2  # noqa: E402
from scripts.training import run_a5_22_5 as a5  # noqa: E402
from scripts.training.subject_registry import load_subject  # noqa: E402
from taiji.u2_history import value_digest  # noqa: E402

OUT = ROOT / "reports/taiji_u2_e5_6_char_channel_20261118.json"
CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
SUBJECT = "a5-pre-seed20261110"
ORDER = ("左边", "中间", "右边")
STEPS = 4000


def spans_of(record: bytes) -> list[int]:
    """Byte offsets of character starts."""
    out, index = [], 0
    while index < len(record):
        out.append(index)
        lead = record[index]
        index += 1 if lead < 0x80 else 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4
    return out


def pieces_of(record: bytes, spans: list[int]) -> list[bytes]:
    """Two-character windows: the answer is a two-character word."""
    out = []
    for i in range(len(spans) - 1):
        end = spans[i + 2] if i + 2 < len(spans) else len(record)
        out.append(record[spans[i] : end])
    return out


class CharCopy(nn.Module):
    """Bind one character from the record.  No subject weights."""

    def __init__(self, emb: int = 16, d: int = 32) -> None:
        super().__init__()
        self.byte = nn.Embedding(256, emb)
        self.query = nn.Linear(2 * emb, d)
        self.key = nn.Linear(emb, d)

    def forward(self, request: torch.Tensor, record: torch.Tensor, spans: list[int]):
        text = self.byte(request)
        pooled = torch.cat([text.mean(0), text.amax(0)], -1)
        query = self.query(pooled)
        keys = torch.stack(
            [
                self.byte(
                    record[start : (spans[i + 1] if i + 1 < len(spans) else len(record))]
                ).mean(0)
                for i, start in enumerate(spans)
            ]
        )
        keys = self.key(keys)
        return keys @ query / math.sqrt(query.shape[-1])


def items_of(examples, templates):
    items = []
    for example in examples:
        truth = example["target"].decode()
        if truth not in ORDER:
            continue
        revised = ORDER[(ORDER.index(truth) + 1) % 3]
        entity = a5.p4_2.column_public(example)["request"].decode()[: -len(a5.SUFFIX)]
        items.append(
            {
                "id": example["id"],
                "entity": entity,
                "target": revised.encode(),
                "records": [
                    template.format(entity=entity, label=revised).encode() for template in templates
                ],
            }
        )
    return items


def pick(channel, item, index, templates) -> bytes:
    record = item["records"][index % len(templates)]
    spans = spans_of(record)
    request = torch.tensor(list(item["entity"].encode()), dtype=torch.long)
    record_tensor = torch.tensor(list(record), dtype=torch.long)
    with torch.no_grad():
        score = channel(request, record_tensor, spans)
    return pieces_of(record, spans)[int(score.argmax())]


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

    channel = CharCopy()
    optimizer = torch.optim.Adam(channel.parameters(), lr=0.01, weight_decay=1e-4)
    started = time.process_time()
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
        score = channel(request, record_tensor, spans)
        loss = F.cross_entropy(score[None], torch.tensor([labels[0]]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    correct, rows = 0, []
    for index, item in enumerate(fit_items):
        raw = pick(channel, item, index, templates)
        ok = raw == item["target"]
        correct += int(ok)
        rows.append(
            {
                "id": item["id"],
                "target": item["target"].decode(),
                "got": raw.decode("utf-8", "replace"),
                "correct": ok,
            }
        )

    subject, meta = load_subject(SUBJECT, contract)
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    codes = a2.code_by_id(subject, head, prior, data["train"] + data["fit"], contract)
    a_ok = 0
    for example in data["fit"]:
        state = a5.episode(subject, a5.p4_2.column_public(example), example["id"], None, False)
        a_ok += int(
            a5.answer_bytes(subject, state, codes[example["id"]])["raw"] == example["target"]
        )

    payload = {
        "schema": "taiji-u2-e5-6-char-channel-v1",
        "card": "E5.6",
        "subject": SUBJECT,
        "bundle_sha256": meta["sha256"],
        "steps": STEPS,
        "train_items": len(train_items),
        "fit_items": len(fit_items),
        "b_correct": correct,
        "b_accuracy": round(correct / len(fit_items), 4),
        "a_accuracy": round(a_ok / len(data["fit"]), 4),
        "rows": rows,
        "verdict": "char_aligned_passes" if correct >= 20 else "char_aligned_fails",
        "cpu": round(time.process_time() - started, 2),
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "B",
        correct,
        "/",
        len(fit_items),
        "A",
        payload["a_accuracy"],
        payload["verdict"],
        "cpu",
        payload["cpu"],
    )
    for row in rows:
        if not row["correct"]:
            print("  miss", row["id"], row["target"], "->", repr(row["got"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
