"""E5.8: one interface, real routing, parent bit-identical.

E5.7 measured A and B in one process but the routing lived in the harness and never
fired (the A list had no records).  This builds a single object that owns the frozen
subject and the word channel and answers through exactly one path per call, deciding
by whether the episode carries a legally released record.

The parent must stay bit-identical: the runner records a digest of the subject's
state dict before and after, and fails the run if it moved.

Criterion: A = 32/32 via the subject path, B >= 20/24 via the channel path, the
routing audit matches the two denominators, and the parent digest is unchanged.
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
from scripts.training.probe_e5_6_char_channel_20261118 import (  # noqa: E402
    pieces_of,
    spans_of,
)
from scripts.training.probe_e5_7_composite_20261118 import train_channel  # noqa: E402
from scripts.training.subject_registry import load_subject  # noqa: E402
from taiji.u2_history import value_digest  # noqa: E402

OUT = ROOT / "reports/taiji_u2_e5_8_routed_20261118.json"
CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
SUBJECT = "a5-pre-seed20261110"


class RoutedSubject:
    """One interface: records -> word channel, no records -> frozen subject."""

    def __init__(self, subject, channel, templates: list[str], codes: dict) -> None:
        self.subject = subject
        self.channel = channel
        self.templates = templates
        self.codes = codes
        self.audit = {"subject": 0, "channel": 0}

    def answer(self, example: dict, record: bytes | None = None, template_index: int = 0) -> bytes:
        if record is not None:
            self.audit["channel"] += 1
            spans = spans_of(record)
            entity = a5.p4_2.column_public(example)["request"].decode()[: -len(a5.SUFFIX)].encode()
            with torch.no_grad():
                score = self.channel(
                    torch.tensor(list(entity), dtype=torch.long),
                    torch.tensor(list(record), dtype=torch.long),
                    spans,
                )
            return pieces_of(record, spans)[int(score.argmax())]
        self.audit["subject"] += 1
        state = a5.episode(self.subject, a5.p4_2.column_public(example), example["id"], None, False)
        return a5.answer_bytes(self.subject, state, self.codes[example["id"]])["raw"]


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    data = a5.atomic_load(ROOT / contract["data_root"])
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    templates = contract["task_b"]["templates"]

    from scripts.training.probe_e5_6_char_channel_20261118 import items_of

    channel = train_channel(items_of(data["train"], templates), templates)
    subject, meta = load_subject(SUBJECT, contract)
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    codes = a2.code_by_id(subject, head, prior, data["train"] + data["fit"], contract)

    before = value_digest({k: v for k, v in subject.state_dict().items()})
    agent = RoutedSubject(subject, channel, templates, codes)

    a_correct, a_rows = 0, []
    for example in data["fit"]:
        raw = agent.answer(example)
        ok = raw == example["target"]
        a_correct += int(ok)
        a_rows.append({"id": example["id"], "correct": ok})

    fit_items = items_of(data["fit"], templates)
    b_correct, b_rows = 0, []
    for index, item in enumerate(fit_items):
        example = next(e for e in data["fit"] if e["id"] == item["id"])
        record = item["records"][index % len(templates)]
        raw = agent.answer(example, record, index % len(templates))
        ok = raw == item["target"]
        b_correct += int(ok)
        b_rows.append(
            {
                "id": item["id"],
                "target": item["target"].decode(),
                "got": raw.decode("utf-8", "replace"),
                "correct": ok,
            }
        )

    after = value_digest({k: v for k, v in subject.state_dict().items()})
    parent_unchanged = before == after
    verdict = (
        "routed_both_hold"
        if parent_unchanged
        and a_correct == len(a_rows)
        and b_correct >= 20
        and agent.audit == {"subject": len(a_rows), "channel": len(fit_items)}
        else "routed_fails"
    )
    payload = {
        "schema": "taiji-u2-e5-8-routed-v1",
        "card": "E5.8",
        "subject": SUBJECT,
        "bundle_sha256": meta["sha256"],
        "a_correct": a_correct,
        "a_denominator": len(a_rows),
        "b_correct": b_correct,
        "b_denominator": len(fit_items),
        "audit": agent.audit,
        "parent_state_digest_before": before,
        "parent_state_digest_after": after,
        "parent_unchanged": parent_unchanged,
        "verdict": verdict,
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
        "audit",
        agent.audit,
        "parent_unchanged",
        parent_unchanged,
        verdict,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
