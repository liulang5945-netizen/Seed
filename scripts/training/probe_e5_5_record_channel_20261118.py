"""WIP/INVALID: do not cite.  As of 2026-11-18 the harness fails its own calibration (task-A accuracy 0.0 on the registered subject, whereas probe_e5_eos_contrast_v2 measured 1.0), so the B reading cannot be interpreted.

E5.5: an isolated record-only channel answers B while A is untouched.

Task B is genuine correction (the record contradicts the observation).  Every
shared-weight arm either failed to learn it or destroyed A.  This probe routes by
evidence type: with a released record, an extractive copy channel answers from the
record alone; without one, the frozen subject answers.  A stays at 1.0 by
construction, and B measures whether the channel generalises to unseen worlds.

Pre-registered criterion: B exact >= 20/24 on the 24 fit worlds, trained only on
train worlds.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import run_a5_22_5 as a5  # noqa: E402
from scripts.training.subject_registry import load_subject  # noqa: E402
from taiji.u2_history import value_digest  # noqa: E402

OUT = ROOT / "reports/taiji_u2_e5_5_record_channel_20261118.json"
CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
SUBJECT = "a5-pre-seed20261110"
ORDER = ("左边", "中间", "右边")
STEPS = 6000


class RecordCopy(nn.Module):
    """Extractive copy with a learned EOS position.  No subject weights."""

    def __init__(self, emb: int = 16, d: int = 32) -> None:
        super().__init__()
        self.byte = nn.Embedding(256, emb)
        self.query = nn.Linear(2 * emb + 1, d)
        self.key = nn.Linear(emb, d)
        self.stop = nn.Linear(2 * emb + 1, 1)

    def forward(
        self, request: torch.Tensor, record: torch.Tensor, mask: torch.Tensor, emitted: int
    ):
        text = self.byte(request)
        pooled = torch.cat([text.mean(0), text.amax(0)], -1)
        progress = torch.tensor([emitted / 24.0])
        query = self.query(torch.cat([pooled, progress]))
        key = self.key(self.byte(record))
        score = (key @ query) / math.sqrt(key.shape[-1])
        score = score.masked_fill(~mask, -1e4)
        weights = torch.cat([score, self.stop(torch.cat([pooled, progress])).reshape(1)]).softmax(
            -1
        )
        record_weights, stop_weight = weights[:-1], weights[-1]
        distribution = torch.zeros(257)
        distribution = distribution.scatter_add(0, record, record_weights)
        distribution[256] = stop_weight
        return distribution


def build_items(examples: list[dict], templates: list[str]) -> list[dict]:
    items = []
    for example in examples:
        truth = example["target"].decode()
        if truth not in ORDER:
            continue
        revised = ORDER[(ORDER.index(truth) + 1) % 3]
        public = a5.p4_2.column_public(example)
        entity = public["request"].decode()[: -len(a5.SUFFIX)]
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


def tensors(item: dict, template_index: int):
    record = item["records"][template_index]
    return (
        torch.tensor(list(item["entity"].encode()), dtype=torch.long),
        torch.tensor(list(record), dtype=torch.long),
        torch.ones(len(record), dtype=torch.bool),
    )


def decode(channel: RecordCopy, item: dict, template_index: int) -> bytes:
    request, record, mask = tensors(item, template_index)
    emitted = []
    for _ in range(24):
        symbol = int(channel(request, record, mask, len(emitted)).argmax())
        if symbol == 256:
            return bytes(emitted)
        emitted.append(symbol)
    return bytes(emitted)


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    data = a5.atomic_load(ROOT / contract["data_root"])
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    templates = contract["task_b"]["templates"]
    train_items = build_items(data["train"], templates)
    fit_items = build_items(data["fit"], templates)

    channel = RecordCopy()
    optimizer = torch.optim.Adam(channel.parameters(), lr=0.01, weight_decay=1e-4)
    started = time.process_time()
    rng = random.Random(20261110)
    for _ in range(STEPS):
        item = rng.choice(train_items)
        template_index = rng.randrange(len(templates))
        request, record, mask = tensors(item, template_index)
        optimizer.zero_grad(set_to_none=True)
        terms = []
        for position, symbol in enumerate([*item["target"], 256]):
            distribution = channel(request, record, mask, position).clamp_min(1e-12)
            terms.append(-torch.log(distribution[symbol]))
        torch.stack(terms).mean().backward()
        optimizer.step()

    correct, rows = 0, []
    for index, item in enumerate(fit_items):
        raw = decode(channel, item, index % len(templates))
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

    subject, subject_meta = load_subject(SUBJECT, contract)
    head = __import__("scripts.training.run_a2_byte_20261115", fromlist=["load_head"]).load_head(
        contract
    )
    prior = __import__("scripts.training.run_a2_byte_20261115", fromlist=["prior_of"]).prior_of(
        data
    )
    codes = __import__("scripts.training.run_a2_byte_20261115", fromlist=["code_by_id"]).code_by_id(
        subject, head, prior, data["train"] + data["fit"], contract
    )
    a_ok = 0
    for example in data["fit"]:
        state = a5.episode(subject, a5.p4_2.column_public(example), example["id"], None, False)
        a_ok += int(
            a5.answer_bytes(subject, state, codes[example["id"]])["raw"] == example["target"]
        )
    a_accuracy = a_ok / len(data["fit"])

    payload = {
        "schema": "taiji-u2-e5-5-record-channel-v1",
        "card": "E5.5",
        "subject": SUBJECT,
        "bundle_sha256": subject_meta["sha256"],
        "train_items": len(train_items),
        "fit_items": len(fit_items),
        "steps": STEPS,
        "channel_parameters": sum(p.numel() for p in channel.parameters()),
        "b_correct": correct,
        "b_denominator": len(fit_items),
        "b_accuracy": round(correct / len(fit_items), 4),
        "a_accuracy": round(a_accuracy, 4),
        "a_note": "A is measured on the untouched registered subject",
        "rows": rows,
        "verdict": "isolated_channel_passes" if correct >= 20 else "isolated_channel_fails",
        "cpu": round(time.process_time() - started, 2),
    }
    OUT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "B",
        correct,
        "/",
        len(fit_items),
        "A",
        round(a_accuracy, 4),
        "verdict",
        payload["verdict"],
        "cpu",
        payload["cpu"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
