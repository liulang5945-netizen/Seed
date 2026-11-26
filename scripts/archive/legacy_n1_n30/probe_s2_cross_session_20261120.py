"""Pre-registered cross-session accumulation probe (22.5.1 children, zero check).

Session 1 = the already-saved joint-gate child (organ trained 256 steps on
``train[:64]``).  Session 2 = continue training the SAME child's organ for 256
steps on a DISJOINT batch ``train[64:128]`` with a fresh optimizer, exactly the
kind of session-boundary a persistent subject must survive.  Evaluation uses two
disjoint 24-item B faces built from the two batches, before and after session 2:

* retention: session-1 face B success must not drop (one-sided 95% UB <= 0.02);
* acquisition: session-2 face B success must rise (mean gain > 0, 95% LB > 0);
* identity: inherited non-organ parameters stay bit-identical to the parent;
* cost cap 220 CPU; zero check/final reads; no capability signing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import generate_examples as generate_v2  # noqa: E402
from scripts.training.s0_parent_loader_20261119 import (  # noqa: E402
    CHECKPOINTS,
    load_historical,
)
from taiji.u2_22_5_pipeline import (  # noqa: E402
    DEFAULT_TEMPLATES,
    build_b_items,
    cosine_warmup_lr,  # noqa: E402
    decode_episode,
    make_record,
)
from taiji.u2_record_organ import mix_log_probabilities  # noqa: E402
from taiji.u2_subject import U2Config, U2Subject  # noqa: E402

OUT = ROOT / "reports/taiji_u2_22_5_1_cross_session_20261120.json"
ARTIFACTS = ROOT / "output/s2_joint_gate_20261120"
PARENT_SEED = 20261112
SEEDS = (20261110, 20261111, 20261112)
STEPS = 256
BATCH = 2
LR = 0.015
MIN_LR = 0.001
CAP_CPU = 220.0
SEEN = DEFAULT_TEMPLATES[:2]
FACE_ITEMS = 24


def bound(values: list[float], which: float, resamples: int = 10000, seed: int = 20261120) -> float:
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    draws = rng.choice(data, size=(resamples, data.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, which))


def _distractor(item: dict, other: dict, template_index: int, tag: str):
    raw = SEEN[template_index].format(entity=other["entity"], label=other["revised"]).encode()
    return make_record(raw, tag, version=1)


def cache_parent_logp(parent: U2Subject, items: list[dict]) -> dict:
    from taiji.u2_22_5_pipeline import prepare_episode
    from taiji.u2_operators import utf8_mask

    cache: dict[tuple[str, bytes], list[tuple[torch.Tensor, torch.Tensor]]] = {}
    with torch.no_grad():
        for item in items:
            ex = item["example"]
            for target in (ex["target"], item["target"]):
                key = (ex["id"], target)
                if key in cache:
                    continue
                state = prepare_episode(parent, ex["public"], ex["id"], records=None)
                goal = parent.language_goal(state)
                action = state["prediction_action"]
                rows = []
                for sym in [*target, 256]:
                    mask = utf8_mask(
                        state["utf8_remaining"],
                        state["utf8_lead"],
                        parent.c.max_output_bytes - state["data_bytes"],
                        device=parent.native_w.device,
                    )
                    logp, _, details = parent.output(state, goal, action)
                    rows.append((logp.detach(), mask))
                    state = parent.accept_byte(state, sym, details)
                cache[key] = rows
    return cache


def train_session(
    model: U2Subject,
    parent: U2Subject,
    examples: list[dict],
    seed: int,
    templates: tuple[str, ...] = SEEN,
) -> dict:
    """Train the existing organ for one session (fresh optimizer, session boundary)."""
    parent_keys = {k for k in model.state_dict() if not k.startswith("record_organ.")}
    before = {k: v.detach().clone() for k, v in model.state_dict().items() if k in parent_keys}
    items = build_b_items(examples, templates=templates)
    cache = cache_parent_logp(parent, items)
    gate = [
        p
        for n, p in model.record_organ.named_parameters()
        if n in {"organ_mix", "recency_logit"} or n.startswith("route_head.")
    ]
    other = [
        p
        for n, p in model.record_organ.named_parameters()
        if not (n in {"organ_mix", "recency_logit"} or n.startswith("route_head."))
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": other, "lr": LR, "weight_decay": 1e-4},
            {"params": gate, "lr": LR * 3.0, "weight_decay": 0.0},
        ]
    )
    rng = random.Random(seed + 31)
    bce = torch.nn.functional.binary_cross_entropy
    for step in range(STEPS):
        lr = cosine_warmup_lr(step, STEPS, LR, MIN_LR, warmup_steps=20)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[1]["lr"] = lr * 3.0
        losses = []
        for _ in range(BATCH):
            item = rng.choice(items)
            other_item = rng.choice(items)
            while other_item["entity"] == item["entity"]:
                other_item = rng.choice(items)
            other_item2 = rng.choice(items)
            while other_item2["entity"] == item["entity"]:
                other_item2 = rng.choice(items)
            template_index = rng.randrange(len(templates))
            mode = rng.randrange(6)
            ex = item["example"]
            if mode == 0:
                recs = [make_record(item["records"][template_index], f"s-{step}", version=1)]
                target = item["target"]
                should_route = 1.0
            elif mode in (1, 4):
                recs = [_distractor(item, other_item, template_index, f"d-{mode}-{step}")]
                target = ex["target"]
                should_route = 0.0
            elif mode == 2:
                recs = [
                    _distractor(item, other_item, rng.randrange(len(templates)), f"m1-{step}"),
                    make_record(item["records"][template_index], f"m2-{step}", version=1),
                    _distractor(item, other_item2, rng.randrange(len(templates)), f"m3-{step}"),
                ]
                rng.shuffle(recs)
                target = item["target"]
                should_route = 1.0
            elif mode == 3:
                old_rec = make_record(
                    item["counter_records"][template_index], f"o-{step}", version=1
                )
                new_rec = make_record(
                    item["records"][(template_index + 1) % len(templates)],
                    f"n-{step}",
                    version=2,
                )
                recs = [old_rec, new_rec]
                target = item["target"]
                should_route = 1.0
            else:
                recs = [
                    _distractor(item, other_item, rng.randrange(len(templates)), f"q1-{step}"),
                    _distractor(item, other_item2, rng.randrange(len(templates)), f"q2-{step}"),
                ]
                target = ex["target"]
                should_route = 0.0
            plan = model.record_organ.plan(ex["public"]["request"], recs)
            rows = cache[(ex["id"], target)]
            terms = []
            span_terms = []
            for offset, (sym, (parent_logp, mask)) in enumerate(
                zip([*target, 256], rows, strict=True)
            ):
                route_prob, organ_probs = model.record_organ.emit(plan, offset, mask=mask)
                mixed = mix_log_probabilities(parent_logp, route_prob, organ_probs, mask)
                weight = 2.0 if offset == 0 else 1.0
                terms.append(-mixed[sym] * weight)
                if should_route > 0.5:
                    span_terms.append(-organ_probs[sym].clamp_min(1e-9).log() * weight)
            seq_loss = torch.stack(terms).mean()
            route_target = torch.tensor(should_route, dtype=plan["match_prob"].dtype)
            route_aux = bce(plan["match_prob"].clamp(1e-6, 1.0 - 1e-6), route_target)
            loss = seq_loss + 1.0 * route_aux
            if span_terms:
                loss = loss + 0.5 * torch.stack(span_terms).mean()
            losses.append(loss)
        optimizer.zero_grad(set_to_none=True)
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(list(model.record_organ.parameters()), 1.0)
        optimizer.step()
    after = {k: v for k, v in model.state_dict().items() if k in parent_keys}
    inherited_ok = all(torch.equal(before[k], after[k]) for k in before)
    return {
        "parent_bit_identical": inherited_ok,
        "organ_mix": round(float(model.record_organ.organ_mix.detach()), 4),
        "steps": STEPS,
    }


def evaluate_face(
    model: U2Subject, examples: list[dict], templates: tuple[str, ...] = SEEN
) -> dict:
    items = build_b_items(examples, templates=templates)[:FACE_ITEMS]
    rows = []
    with torch.no_grad():
        for index, item in enumerate(items):
            ex = item["example"]
            template_index = index % len(templates)
            rec = make_record(item["records"][template_index], f"eval-{index}", version=1)
            out, eos = decode_episode(model, ex["public"], ex["id"], records=[rec])
            rows.append(1 if bool(eos and out == item["target"]) else 0)
    return {"n": len(items), "correct": sum(rows), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    parent, _payload, _audit = load_historical(CHECKPOINTS[PARENT_SEED])
    train = generate_v2("train", 256)
    s1_examples = train[:64]
    s2_examples = train[64:128]
    rows = []
    for seed in SEEDS:
        artifact = ARTIFACTS / f"child-{seed}.pt"
        payload_data = torch.load(artifact, map_location="cpu", weights_only=False)
        child = U2Subject(U2Config(**payload_data["config"]))
        child.load_state_dict(payload_data["state_dict"])
        before = {
            "s1": evaluate_face(child, s1_examples),
            "s2": evaluate_face(child, s2_examples),
        }
        info = train_session(child, parent, s2_examples, seed + 1)
        after = {
            "s1": evaluate_face(child, s1_examples),
            "s2": evaluate_face(child, s2_examples),
        }
        retention_pairs = [
            float(after["s1"]["rows"][i]) - float(before["s1"]["rows"][i])
            for i in range(before["s1"]["n"])
        ]
        acquisition_pairs = [
            float(after["s2"]["rows"][i]) - float(before["s2"]["rows"][i])
            for i in range(before["s2"]["n"])
        ]
        rows.append(
            {
                "seed": seed,
                "before": {face: data["correct"] for face, data in before.items()},
                "after": {face: data["correct"] for face, data in after.items()},
                "n": before["s1"]["n"],
                "retention_drop_upper_bound": round(bound(retention_pairs, 0.95), 4),
                "acquisition_gain": round(float(np.mean(acquisition_pairs)), 4),
                "acquisition_gain_lower_bound": round(bound(acquisition_pairs, 0.05), 4),
                "parent_bit_identical": info["parent_bit_identical"],
                "organ_mix": info["organ_mix"],
            }
        )
        if time.process_time() - started > CAP_CPU:
            break
    budget_ok = time.process_time() - started <= CAP_CPU
    retention_ok = all(row["retention_drop_upper_bound"] <= 0.02 for row in rows)
    acquisition_ok = all(
        row["acquisition_gain"] > 0 and row["acquisition_gain_lower_bound"] > 0 for row in rows
    )
    identity_ok = all(row["parent_bit_identical"] for row in rows)
    if not budget_ok:
        verdict = "budget_stop"
    elif not identity_ok:
        verdict = "engineering_failure"
    elif not retention_ok:
        verdict = "cross_session_retention_breach"
    elif not acquisition_ok:
        verdict = "cross_session_no_learning"
    elif len(rows) < len(SEEDS):
        verdict = "engineering_failure"
    else:
        verdict = "cross_session_accumulation_pass"
    payload = {
        "schema": "taiji-u2-22-5-1-cross-session-v1",
        "card": "22.5.1 children cross-session accumulation probe",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "prereg": {
            "session1": "saved joint-gate child (organ trained 256 steps on train[:64])",
            "session2": "continue SAME child 256 steps on disjoint train[64:128], fresh optimizer",
            "faces": {
                "s1": "first 24 non-unknown items of train[:64]",
                "s2": "first 24 non-unknown items of train[64:128]",
            },
            "thresholds": {
                "retention_drop_upper_bound_max": 0.02,
                "acquisition_gain_min": 0.0,
                "acquisition_lower_bound_min": 0.0,
            },
            "exits": [
                "budget_stop",
                "engineering_failure",
                "cross_session_retention_breach",
                "cross_session_no_learning",
                "cross_session_accumulation_pass",
            ],
            "cap_cpu": CAP_CPU,
            "check_reads": 0,
            "final_reads": 0,
        },
        "rows": rows,
        "verdict": verdict,
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "optimizer_steps": STEPS * len(rows),
            "scope": "fit/train faces only; no capability signing",
        },
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "rows": rows, "cpu": payload["cost"]["cpu"]}, ensure_ascii=False
        )
    )
    return 0 if verdict == "cross_session_accumulation_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
