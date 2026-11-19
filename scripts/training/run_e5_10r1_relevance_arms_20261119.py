"""22.5.1/E5.10r1 relevance-gate arms (D-20261119-05, owner-authorized, cap 100 CPU).

Three modes:
* --diagnosis  zero optimizer steps: on the E5.9 frozen organ, dot vs cosine
  separability of match/mismatch when the query is the full request
  (entity + suffix) versus the bare entity used at training time.
* --arm A      instance-level patch: attention scores become consistent
  cosine (window scores and sentinel term both normalized; no extra scale -
  the literal prereg "/scale" would flatten the softmax into the always-route
  regime, disclosed).  Training loss inherits E5.11 verbatim, computed on the
  patched attention; the eval runs through real episodes on the fixed wiring.
* --arm B      score formula untouched; loss becomes direct match supervision
  per item - cross entropy toward the entity-bearing window (match) or the
  sentinel (mismatch), replacing the pointer-sum.

Gates (E5.10 frozen): A 32/32, B >= 20/24, D >= 20/24, parent digest unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from types import MethodType

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import run_a2_byte_20261115 as a2  # noqa: E402
from scripts.training import run_a5_22_5 as a5  # noqa: E402
from scripts.training.probe_e5_6_char_channel_20261118 import items_of  # noqa: E402
from scripts.training.probe_e5_9_organ_20261118 import _organ_config  # noqa: E402
from scripts.training.subject_registry import resolve  # noqa: E402
from taiji.u2_history import value_digest  # noqa: E402
from taiji.u2_record_organ import spans_of, windows_of  # noqa: E402
from taiji.u2_subject import U2Subject  # noqa: E402

CONTRACT = ROOT / "plans/manifests/taiji_u2_a5_protocol_20261116.json"
SUBJECT = "a5-pre-seed20261110"
BUNDLE = ROOT / "data/experiments/taiji_u2/e5_9/bundle-parent-plus-organ.pt"
STEPS = 6000
SUFFIX = "现在在哪里？"


def _cosine_attention(self, request: bytes, record: bytes) -> torch.Tensor:
    windows = windows_of(record)
    if not windows:
        return torch.tensor([1.0])
    text = self.byte(torch.tensor(list(request), dtype=torch.long))
    pooled = torch.cat([text.mean(0), text.amax(0)], -1)
    query = self.query(pooled)
    keys = self.key(
        torch.stack(
            [self.byte(torch.tensor(list(window), dtype=torch.long)).mean(0) for window in windows]
        )
    )
    unit_keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    unit_query = query / query.norm().clamp_min(1e-9)
    scores = unit_keys @ unit_query
    sentinel = (query @ self.sentinel) / (
        query.norm().clamp_min(1e-9) * self.sentinel.norm().clamp_min(1e-9)
    ) + self.gain * scores.max()
    return torch.cat([scores, sentinel.reshape(1)]).softmax(-1)


def _load_bundle_organ(contract):
    saved = torch.load(BUNDLE, weights_only=True)["model"]
    organ_state = {
        key.removeprefix("record_organ."): value
        for key, value in saved.items()
        if key.startswith("record_organ.")
    }
    config = _organ_config(contract, True)
    model = U2Subject(config).metadata_candidate().request_candidate()
    organ = model.record_organ
    # The E5.9-era organ predates the sentinel/gain parameters (added by the
    # E5.10/E5.11 iterations): byte/query/key restore, the two later tensors
    # keep their fresh defaults, which the window-score diagnosis never reads.
    organ.load_state_dict(organ_state, strict=False)
    return organ


def _entity_window_indices(record: bytes, entity: bytes) -> list[int]:
    spans = spans_of(record)
    start = record.find(entity)
    if start < 0:
        return []
    end = start + len(entity)
    hits = []
    for index in range(len(spans) - 1):
        window_end = spans[index + 2] if index + 2 < len(spans) else len(record)
        if spans[index] < end and window_end > start:
            hits.append(index)
    return hits


def _cosine_scores(organ, request: bytes, record: bytes) -> torch.Tensor:
    windows = windows_of(record)
    text = organ.byte(torch.tensor(list(request), dtype=torch.long))
    pooled = torch.cat([text.mean(0), text.amax(0)], -1)
    query = organ.query(pooled)
    keys = organ.key(
        torch.stack(
            [organ.byte(torch.tensor(list(window), dtype=torch.long)).mean(0) for window in windows]
        )
    )
    unit_keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    unit_query = query / query.norm().clamp_min(1e-9)
    return unit_keys @ unit_query


def _stats(rows: list[float]) -> dict:
    if not rows:
        return {"n": 0}
    tensor = torch.tensor(rows)
    return {
        "n": len(rows),
        "mean": round(float(tensor.mean()), 4),
        "std": round(float(tensor.std()), 4),
    }


def diagnosis(contract) -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    data = a5.atomic_load(ROOT / contract["data_root"])
    templates = contract["task_b"]["templates"]
    fit_items = items_of(data["fit"], templates)
    organ = _load_bundle_organ(contract)
    dot_match, dot_mismatch, cos_match, cos_mismatch, bare_cos_match, bare_cos_mismatch = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for index, item in enumerate(fit_items):
        other = fit_items[(index + 1) % len(fit_items)]
        record = item["records"][index % len(templates)]
        negative = (
            templates[index % len(templates)]
            .format(entity=other["entity"], label=other["target"].decode())
            .encode()
        )
        entity = item["entity"].encode()
        full = f"{item['entity']}{SUFFIX}".encode()
        for rows, cos_rows, request, rec in (
            (dot_match, cos_match, full, record),
            (dot_mismatch, cos_mismatch, full, negative),
            (bare_cos_match, None, entity, record),
            (bare_cos_mismatch, None, entity, negative),
        ):
            windows = windows_of(rec)
            keys = organ.key(
                torch.stack(
                    [organ.byte(torch.tensor(list(w), dtype=torch.long)).mean(0) for w in windows]
                )
            )
            text = organ.byte(torch.tensor(list(request), dtype=torch.long))
            query = organ.query(torch.cat([text.mean(0), text.amax(0)], -1))
            scale = math.sqrt(query.shape[-1])
            rows.append(float((keys @ query / scale).max()))
            if cos_rows is not None:
                hits = _entity_window_indices(rec, entity)
                scores = _cosine_scores(organ, request, rec)
                cos_rows.append(float(scores[hits].max()) if hits else float(scores.max()))
    payload = {
        "schema": "taiji-u2-e5-10r1-diagnosis-v1",
        "card": "22.5.1/E5.10r1",
        "organ": "E5.9 frozen (bundle sha in taiji_u2_e5_9_organ_20261118.json)",
        "dot_full_request": {"match": _stats(dot_match), "mismatch": _stats(dot_mismatch)},
        "cosine_full_request": {"match": _stats(cos_match), "mismatch": _stats(cos_mismatch)},
        "cosine_bare_entity": {
            "match": _stats(bare_cos_match),
            "mismatch": _stats(bare_cos_mismatch),
        },
        "cpu": round(time.process_time() - started, 4),
    }
    out = ROOT / "reports/taiji_u2_e5_10r1_diagnosis_20261119.json"
    out.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def run_arm(contract, which: str) -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    data = a5.atomic_load(ROOT / contract["data_root"])
    if value_digest(data) != contract["data_digest"]:
        raise ValueError("data_digest_mismatch")
    templates = contract["task_b"]["templates"]
    train_items = items_of(data["train"], templates)
    fit_items = items_of(data["fit"], templates)
    parent_state = torch.load(ROOT / resolve(SUBJECT)["path"], weights_only=True)["model"]
    model = U2Subject(_organ_config(contract, True)).metadata_candidate().request_candidate()
    model.load_state_dict(parent_state, strict=False)
    model.eval()
    parent_keys = {k for k in model.state_dict() if not k.startswith("record_organ.")}
    before = value_digest({k: v for k, v in model.state_dict().items() if k in parent_keys})
    organ = model.record_organ
    if which == "A":
        organ.attention = MethodType(_cosine_attention, organ)
    optimizer = torch.optim.Adam(organ.parameters(), lr=0.01, weight_decay=1e-4)
    rng = random.Random(20261110)
    for _ in range(STEPS):
        item = rng.choice(train_items)
        other = rng.choice(train_items)
        index = rng.randrange(len(templates))
        record = item["records"][index]
        entity = item["entity"].encode()
        windows = windows_of(record)
        attention = organ.attention(entity, record)
        if which == "A":
            hits = [i for i, w in enumerate(windows) if w == item["target"]]
            if not hits or other["entity"] == item["entity"]:
                continue
            loss = -torch.log(attention[hits].sum().clamp_min(1e-9))
        else:
            entity_hits = _entity_window_indices(record, entity)
            if entity_hits:
                loss = -torch.log(attention[entity_hits].sum().clamp_min(1e-9))
            else:
                loss = -torch.log(attention[-1].clamp_min(1e-9))
        negative = (
            templates[index].format(entity=other["entity"], label=other["target"].decode()).encode()
        )
        if other["entity"] != item["entity"]:
            attention_negative = organ.attention(entity, negative)
            loss = loss - torch.log(attention_negative[-1].clamp_min(1e-9))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
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
    b_correct, d_correct, b_rows, d_rows = 0, 0, [], []
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
        b_rows.append(raw.decode("utf-8", "replace"))
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
        d_rows.append(raw_d.decode("utf-8", "replace"))
    after = value_digest({k: v for k, v in model.state_dict().items() if k in parent_keys})
    passed = (
        before == after and a_correct == len(data["fit"]) and b_correct >= 20 and d_correct >= 20
    )
    payload = {
        "schema": f"taiji-u2-e5-10r1-arm-{which.lower()}-v1",
        "card": "22.5.1/E5.10r1",
        "arm": which,
        "subject": SUBJECT,
        "mechanism": (
            "consistent-cosine attention (instance patch, disclosed no-extra-scale deviation)"
            if which == "A"
            else "direct match supervision (entity-window log-sum CE + sentinel on true mismatch)"
        ),
        "wiring": "route reads current request (defect fix a7c992e95)",
        "steps": STEPS,
        "a_correct": a_correct,
        "a_denominator": len(data["fit"]),
        "b_correct": b_correct,
        "b_denominator": len(fit_items),
        "d_correct": d_correct,
        "d_denominator": len(fit_items),
        "parent_unchanged": before == after,
        "b_outputs": b_rows,
        "d_outputs": d_rows,
        "verdict": f"arm_{which.lower()}_passes" if passed else f"arm_{which.lower()}_fails",
        "cpu": round(time.process_time() - started, 2),
    }
    out = ROOT / f"reports/taiji_u2_e5_10r1_arm_{which.lower()}_20261119.json"
    out.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        json.dumps(
            {
                k: payload[k]
                for k in (
                    "arm",
                    "a_correct",
                    "b_correct",
                    "d_correct",
                    "parent_unchanged",
                    "verdict",
                    "cpu",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mutex = parser.add_mutually_exclusive_group(required=True)
    mutex.add_argument("--diagnosis", action="store_true")
    mutex.add_argument("--arm", choices=("A", "B"))
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if args.diagnosis:
        return diagnosis(contract)
    return run_arm(contract, args.arm)


if __name__ == "__main__":
    raise SystemExit(main())
