#!/usr/bin/env python
"""Bounded attribution probe for 15.1.1 (instrument check only).

Answers three questions without re-running the 300-update grid:

1. Are the reader and self-prefix switches actually wired? (arm -> config,
   gradient effect, loss divergence).
2. Is the greedy self-prefix equal to the teacher prefix for R1 (which would
   make the P1 curriculum inert), while differing for R0?
3. Is the reported ``source_top1`` estimator well-posed? It argmaxes the
   byte-level copy distribution at the first prediction, whose target byte is a
   template character absent from every record, so its copy mass is ~0.

Cost: a handful of Adam steps. Booked separately from the 2400 formal calls.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji.binding_copy import (  # noqa: E402
    BindingCopyConfig,
    answer_step,
    build_binding_candidate,
    generate_bytes,
    initial_binding_state,
    make_optimizer,
    observe_goal,
    prepare,
    score_example,
)
from taiji.living_core import EOS, RawSource  # noqa: E402
from taiji.record_relation import RecordRelationConfig  # noqa: E402
from taiji.record_relation_core import (  # noqa: E402
    ARM_FULL,
    CLIP_NORM,
    RecordRelationCandidate,
    build_subject_from_candidate_bundle,
)
from taiji.record_relation_core import generate_bytes as base_generate_bytes  # noqa: E402
from taiji.record_relation_core import make_optimizer as base_make_optimizer  # noqa: E402
from taiji.record_relation_core import score_example as base_score_example  # noqa: E402

PARENT = "output/taiji_core_record_relation_14_6/27031_A_full_step300.pt"
DATA = "output/taiji_core_binding_copy_15_1/data_v2.json"
REPORT = "reports/taiji_binding_copy_15_1_attribution_20261103.json"
BATCH = 8
STEPS = 5


def fixture(example: dict) -> dict:
    return {
        "index": int(example["index"]),
        "goal": example["goal"],
        "records": tuple(
            RawSource(source_id=r["source_id"], raw=r["raw"].encode("utf-8"))
            for r in example["records"]
        ),
        "target_bytes": tuple(example["action"].encode("utf-8")) + (EOS,),
        "raw_example": example,
    }


def base_from_parent(payload: dict) -> RecordRelationCandidate:
    config = RecordRelationConfig(**payload["relation_config"])
    subject = build_subject_from_candidate_bundle(payload)
    candidate = RecordRelationCandidate(subject, config, ARM_FULL, list(payload["growth_gates"]))
    named = dict(candidate.named_parameters())
    for name, value in payload["parameters"].items():
        if name.startswith("relation."):
            named[name].data.copy_(value)
    candidate.to(torch.float64)
    candidate.configure_trainable()
    return candidate


def build(payload: dict, r1: bool):
    base = base_from_parent(payload)
    if not r1:
        return base, base_make_optimizer(base)
    candidate = build_binding_candidate(
        base, config=BindingCopyConfig(), profile="R1-f64", dtype=torch.float64
    )
    return candidate, make_optimizer(candidate)


def step_loss(candidate, batch, *, r1: bool, flags) -> torch.Tensor:
    score_fn = score_example if r1 else base_score_example
    gen_fn = generate_bytes if r1 else base_generate_bytes
    losses = []
    for idx, item in enumerate(batch):
        if flags and flags[idx]:
            with torch.no_grad():
                generated = gen_fn(
                    candidate,
                    item["goal"].encode(),
                    item["records"],
                    subject_id=f"probe-{item['index']}",
                )
            target_len = len(item["target_bytes"]) - 1
            prefix = tuple(generated["raw_bytes"][:target_len])
            if len(prefix) < target_len:
                prefix = prefix + (EOS,) * (target_len - len(prefix))
        else:
            prefix = item["target_bytes"][:-1]
        losses.append(score_fn(candidate, item["goal"].encode(), item["records"], prefix).nll)
    return sum(losses) / len(losses)


def stage_bundles() -> dict:
    """0-Adam checks on the saved step300 bundles.

    (a) Does the greedy self-prefix equal the teacher prefix on train (which
        would make the P1 curriculum inert for R1)?
    (b) Record-level selection accuracy from ``read.alpha`` (the intended
        source top-1), versus the byte-level proxy used in the 1.1 report.
    """
    from taiji.binding_copy import load_bundle

    data = json.loads(Path(DATA).read_text(encoding="utf-8"))
    train = [fixture(e) for e in data["examples"] if e["split"] == "train"]
    dev = [fixture(e) for e in data["examples"] if e["split"] == "dev"]
    rows = []
    for seed in (28031, 28032):
        for arm in ("R1P0", "R1P1"):
            path = Path(f"output/taiji_core_binding_copy_15_1_compare/{seed}-{arm}-step300.pt")
            candidate, _state, _opt, _profile, _idx, _cursor, _payload = load_bundle(
                path, torch.float64
            )
            prefix_equal = 0
            for item in train[:8]:
                generated = generate_bytes(
                    candidate,
                    item["goal"].encode(),
                    item["records"],
                    subject_id=f"chk-{item['index']}",
                )
                target = item["target_bytes"][:-1]
                prefix = tuple(generated["raw_bytes"][: len(target)])
                if prefix == target:
                    prefix_equal += 1
            alpha_hits = 0
            byte_hits = 0
            for item in dev:
                state = initial_binding_state(candidate, f"chk-top1-{item['index']}")
                prepared = prepare(candidate, state, item["goal"].encode(), item["records"])
                state, gate = observe_goal(candidate, prepared)
                step = answer_step(candidate, prepared, state, EOS, gate_index=gate)
                object_index = next(
                    (
                        i
                        for i, record in enumerate(item["raw_example"]["records"])
                        if record["kind"] == "object"
                        and record["description"] == item["raw_example"]["object_description"]
                    ),
                    None,
                )
                alpha_hit = int(torch.argmax(step.read.alpha[:-1]).item()) == object_index
                weights = step.read.copy_weights
                offsets = prepared.base.view.record_offsets
                chosen = int(torch.argmax(weights).item())
                chosen_record = next(
                    (i for i, (s, e) in enumerate(offsets) if s <= chosen < e), None
                )
                alpha_hits += int(alpha_hit)
                byte_hits += int(chosen_record == object_index)
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "train_prefix_equal": prefix_equal,
                    "train_prefix_probes": 8,
                    "alpha_record_top1": alpha_hits / len(dev),
                    "byte_level_top1": byte_hits / len(dev),
                    "dev_examples": len(dev),
                }
            )
    return {
        "schema": "taiji-binding-copy-attribution-v1",
        "step": "15.1.1-review-bundles",
        "rows": rows,
        "note": "0 Adam. Record-level alpha accuracy is the intended source top-1.",
    }


def main() -> int:
    if "--stage" in sys.argv and sys.argv[sys.argv.index("--stage") + 1] == "bundles":
        out = stage_bundles()
        Path("reports/taiji_binding_copy_15_1_attribution_bundles_20261103.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    payload = torch.load(ROOT / PARENT, map_location="cpu", weights_only=False)
    data = json.loads(Path(DATA).read_text(encoding="utf-8"))
    train = [fixture(e) for e in data["examples"] if e["split"] == "train"]
    dev = [fixture(e) for e in data["examples"] if e["split"] == "dev"]
    batch = tuple(train[:BATCH])
    out: dict = {"schema": "taiji-binding-copy-attribution-v1", "step": "15.1.1-review"}
    start_cpu = time.process_time()
    adam = 0

    # A. Reader switch: same parent, different graph -> same params, different outputs?
    r0, _ = build(payload, r1=False)
    r1, _ = build(payload, r1=True)
    r0_params = {k: v.detach().clone() for k, v in r0.named_parameters()}
    r1_params = {k: v.detach().clone() for k, v in r1.named_parameters()}
    shared = sorted(set(r0_params) & set(r1_params))
    param_equal = all(torch.equal(r0_params[k], r1_params[k]) for k in shared)
    sample = dev[0]
    r0_gen = base_generate_bytes(r0, sample["goal"].encode(), sample["records"], subject_id="p0")
    r1_gen = generate_bytes(r1, sample["goal"].encode(), sample["records"], subject_id="p1")
    out["reader_switch"] = {
        "shared_parameters_bit_equal_at_init": bool(param_equal),
        "shared_parameter_count": len(shared),
        "r0_raw_hex": bytes(r0_gen["raw_bytes"]).hex(),
        "r1_raw_hex": bytes(r1_gen["raw_bytes"]).hex(),
        "outputs_differ": bytes(r0_gen["raw_bytes"]) != bytes(r1_gen["raw_bytes"]),
        "r1_has_reader_module": hasattr(r1, "reader"),
    }

    # B. Self-prefix equal to teacher prefix? (explains why P1 is inert for R1)
    rows = []
    for item in batch[:4]:
        gen = generate_bytes(
            r1, item["goal"].encode(), item["records"], subject_id=f"sp-{item['index']}"
        )
        prefix = tuple(gen["raw_bytes"])
        target = item["target_bytes"][:-1]
        rows.append(
            {
                "index": item["index"],
                "generated_len": len(prefix),
                "target_len": len(target),
                "prefix_equals_target": bool(prefix == target),
                "terminated": bool(gen["terminated"]),
            }
        )
    out["self_prefix_vs_target_r1_init"] = rows
    out["self_prefix_equal_count"] = sum(1 for r in rows if r["prefix_equals_target"])

    # C. Curriculum effect after a few real updates: R1 vs R0, TF vs all-self.
    def divergence(r1_flag: bool) -> dict:
        cand, opt = build(payload, r1=r1_flag)
        torch.manual_seed(28031)
        tf_losses = []
        self_losses = []
        non_adam = 0
        for _ in range(STEPS):
            loss_tf = step_loss(cand, batch, r1=r1_flag, flags=None)
            loss_self = step_loss(cand, batch, r1=r1_flag, flags=tuple(True for _ in batch))
            tf_losses.append(float(loss_tf.detach()))
            self_losses.append(float(loss_self.detach()))
            opt.zero_grad()
            loss_self.backward()
            torch.nn.utils.clip_grad_norm_(cand.trainable_parameters(), CLIP_NORM)
            opt.step()
            non_adam += 1
        return {
            "steps": STEPS,
            "tf_losses": tf_losses,
            "self_losses": self_losses,
            "max_abs_gap": max(abs(a - b) for a, b in zip(tf_losses, self_losses, strict=True)),
            "adam_calls": non_adam,
        }

    out["curriculum"] = {"R0": divergence(False), "R1": divergence(True)}
    adam = out["curriculum"]["R0"]["adam_calls"] + out["curriculum"]["R1"]["adam_calls"]

    # D. source_top1 estimator: byte-level argmax vs record-level alpha.
    state = initial_binding_state(r1, "probe-top1")
    prepared = prepare(r1, state, sample["goal"].encode(), sample["records"])
    state, gate = observe_goal(r1, prepared)
    step = answer_step(r1, prepared, state, EOS, gate_index=gate)
    weights = step.read.copy_weights
    offsets = prepared.base.view.record_offsets
    chosen = int(torch.argmax(weights).item())
    chosen_record = next(
        (i for i, (start, end) in enumerate(offsets) if start <= chosen < end), None
    )
    object_index = next(
        (
            i
            for i, record in enumerate(sample["raw_example"]["records"])
            if record["kind"] == "object"
            and record["description"] == sample["raw_example"]["object_description"]
        ),
        None,
    )
    template_byte = sample["target_bytes"][0]
    template_in_records = any(template_byte in record.raw for record in sample["records"])
    alpha_argmax = int(torch.argmax(step.read.alpha).item())
    alpha_null = float(step.read.alpha[-1])
    out["source_top1_estimator"] = {
        "first_target_byte": int(template_byte),
        "first_target_byte_in_any_record": bool(template_in_records),
        "copy_weights_total_mass": float(weights.sum().detach()),
        "copy_weights_argmax_position": chosen,
        "copy_weights_argmax_record": chosen_record,
        "object_record_index": object_index,
        "byte_level_hit": bool(chosen_record == object_index),
        "alpha_argmax_record": alpha_argmax,
        "alpha_null": alpha_null,
        "record_level_hit": bool(alpha_argmax == object_index),
        "record_count": int(prepared.base.view.record_count),
    }

    out["cost"] = {
        "cpu_seconds": time.process_time() - start_cpu,
        "adam_calls": int(adam),
    }
    out["limits"] = (
        "Instrument review only. It does not re-run the 300-update grid and makes no "
        "capability claim."
    )
    Path(REPORT).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
