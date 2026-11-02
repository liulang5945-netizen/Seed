#!/usr/bin/env python
"""15.1.1 bounded formal comparison for the span-bound binding-copy package.

Two seeds x four arms x 300 updates (batch 8), 288 train / 240 dev, fixed
step-0 and step-300 endpoints. Task gate, mechanism gate, rescue definition and
exit names follow SPEC-N15-01 rev2 sections 6 and 8; the 2026-11-02 owner
budget ruling replaces the CPU/wall caps with a sufficient budget plus an
observation line at 20,000 CPU / 30,000 wall that reports before continuing.

Serial per arm: never run two arms at once. Pricing measurements are not rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji.binding_copy import (  # noqa: E402
    BindingCopyConfig,
    advance_copy,
    answer_step,
    build_binding_candidate,
    forced_read,
    generate_bytes,
    initial_binding_state,
    make_optimizer,
    observe_goal,
    prepare,
    save_bundle,
    score_example,
)
from taiji.binding_copy_scoring import score_generation  # noqa: E402
from taiji.living_core import EOS, RawSource  # noqa: E402
from taiji.record_relation import RecordRelationConfig  # noqa: E402
from taiji.record_relation_core import (  # noqa: E402
    ARM_FULL,
    CLIP_NORM,
    CONTENT_REVISION,
    MAX_OUTPUT_BYTES,
    MAX_PREDICTIONS,
    RecordRelationCandidate,
    build_subject_from_candidate_bundle,
    decode_endpoint,
)
from taiji.record_relation_core import generate_bytes as base_generate_bytes  # noqa: E402
from taiji.record_relation_core import make_optimizer as base_make_optimizer  # noqa: E402
from taiji.record_relation_core import save_bundle as base_save_bundle  # noqa: E402
from taiji.record_relation_core import score_example as base_score_example  # noqa: E402

PARENT_BUNDLE = "output/taiji_core_record_relation_14_6/27031_A_full_step300.pt"
PARENT_SHA256 = "b75a83a0a474c041c2a61bed8c4e0a51de11197d67a9b196262e7e039a9a4cef"
DATA_PAYLOAD = "output/taiji_core_binding_copy_15_1/data_v2.json"
REPORT = "reports/taiji_binding_copy_15_1_compare_20261102.json"
MANIFEST = "plans/manifests/taiji_binding_copy_15_1_compare_20261102.json"
OUTPUT_DIR = Path("output/taiji_core_binding_copy_15_1_compare")
PARTIAL = "reports/taiji_binding_copy_15_1_compare_partial_20261102.json"
SMOKE_REPORT = "reports/taiji_binding_copy_15_1_compare_smoke_20261102.json"

SEEDS = (28031, 28032)
ORDER_SEEDS = {28031: 38031, 28032: 38032}
ARMS = ("R0P0", "R1P0", "R0P1", "R1P1")
ARM_READER = {"R0P0": "R0", "R1P0": "R1", "R0P1": "R0", "R1P1": "R1"}
ARM_PREFIX = {"R0P0": "TF", "R1P0": "TF", "R0P1": "self", "R1P1": "self"}

BATCH = 8
UPDATES = 300
CHECKPOINT_UPDATE = 150
SELF_PREFIX_PROB = {101: 0.5, 201: 0.75}
SELF_PREFIX_START = 101
SELF_PREFIX_TIGHTEN = 201

ADAM_CAP = 2400
PRODUCT_CAP = 4 * 1024**3
WATCH_CPU_SECONDS = 20000.0
WATCH_WALL_SECONDS = 30000.0

# Pre-registered gates (SPEC rev2 section 6 / section 8).
GATE_OVERALL = 0.80
GATE_SUBSET = 0.60
GATE_BOTH_CORRECT = 0.60
GATE_REVERSED_PAIRED = 0.95
GATE_UTF8 = 0.98
GATE_EOS = 0.95
GATE_SOURCE_TOP1 = 0.60
GATE_PACKAGE_GAIN = 0.10
GATE_INTERVENTION_DROP = 0.10

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 58031

SOURCE_FILES = (
    "taiji/binding_copy.py",
    "taiji/binding_copy_data.py",
    "taiji/binding_copy_scoring.py",
    "taiji/record_relation.py",
    "taiji/record_relation_core.py",
    "scripts/training/verify_taiji_binding_copy_15_1_compare.py",
)

STARTED_WALL = time.perf_counter()
STARTED_CPU = time.process_time()


class CompareStop(RuntimeError):
    """Named stop for the comparison stage."""


class WatchLine(RuntimeError):
    """Observation line reached: report progress instead of running through."""


class Budget:
    def __init__(self) -> None:
        self.wall = 0.0
        self.adam_calls = 0
        self.live_bytes = 0
        self.max_live_bytes = 0
        self.cumulative_bytes = 0
        self.watch_reported = False

    def cpu(self) -> float:
        return time.process_time() - STARTED_CPU

    def wall_now(self) -> float:
        return time.perf_counter() - STARTED_WALL

    def add_product(self, info: dict) -> None:
        size = int(info.get("bytes", 0))
        self.live_bytes += size
        self.cumulative_bytes += size
        self.max_live_bytes = max(self.max_live_bytes, self.live_bytes)

    def drop_product(self, info: dict) -> None:
        self.live_bytes -= int(info.get("bytes", 0))

    def ensure(self, *, context: dict | None = None) -> None:
        cpu = self.cpu()
        wall = self.wall_now()
        if self.adam_calls > ADAM_CAP:
            raise CompareStop(f"adam cap exceeded: {self.adam_calls}")
        if self.max_live_bytes > PRODUCT_CAP:
            raise CompareStop(f"product cap exceeded: {self.max_live_bytes}")
        if cpu >= WATCH_CPU_SECONDS or wall >= WATCH_WALL_SECONDS:
            self.watch_reported = True
            raise WatchLine(
                json.dumps(
                    {
                        "reason": "observation_line",
                        "cpu_seconds": cpu,
                        "wall_seconds": wall,
                        "adam_calls": self.adam_calls,
                        "context": context or {},
                    },
                    ensure_ascii=False,
                )
            )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_base_from_parent(payload: dict) -> RecordRelationCandidate:
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


def example_to_fixture(example: dict) -> dict:
    records = tuple(
        RawSource(source_id=r["source_id"], raw=r["raw"].encode("utf-8"))
        for r in example["records"]
    )
    return {
        "index": int(example["index"]),
        "world": int(example["world"]),
        "goal": example["goal"],
        "records": records,
        "target_bytes": tuple(example["action"].encode("utf-8")) + (EOS,),
        "target_action": example["action"],
        "flags": {
            "unseen_output_name": bool(example["unseen_output_name"]),
            "both_new": bool(example["both_new"]),
            "new_description": bool(example["new_description"]),
            "unseen_combination": bool(example["unseen_combination"]),
        },
        "object_description": example["object_description"],
        "place_description": example["place_description"],
        "object_name": example["object_name"],
        "place_name": example["place_name"],
        "raw_example": example,
    }


def load_data() -> tuple[tuple[dict, ...], tuple[dict, ...], dict]:
    payload = json.loads(Path(DATA_PAYLOAD).read_text(encoding="utf-8"))
    train = tuple(example_to_fixture(e) for e in payload["examples"] if e["split"] == "train")
    dev = tuple(example_to_fixture(e) for e in payload["examples"] if e["split"] == "dev")
    return train, dev, payload


def record_template(style_name: str, description: str, name: str) -> str:
    if style_name == "stated":
        return f"{description}是{name}。"
    if style_name == "aka":
        return f"{name}又叫{description}。"
    if style_name == "mapped":
        return f"{description}对应{name}。"
    raise CompareStop(f"unknown record style {style_name}")


def object_records(example: dict) -> list[dict]:
    return [r for r in example["raw_example"]["records"] if r["kind"] == "object"]


def swap_object_name(example: dict) -> dict | None:
    """In-domain mapping counterfactual: replace the target name with another
    object record's name, keeping every description and record style fixed."""
    objects = object_records(example)
    target = next((r for r in objects if r["description"] == example["object_description"]), None)
    if target is None:
        return None
    alternate = next((r for r in objects if r["name"] != target["name"]), None)
    if alternate is None:
        return None
    new_name = alternate["name"]
    records = []
    for record in example["raw_example"]["records"]:
        name = new_name if record["source_id"] == target["source_id"] else record["name"]
        raw = record_template(record["style_name"], record["description"], name)
        records.append(RawSource(source_id=record["source_id"], raw=raw.encode("utf-8")))
    expected = f"把{new_name}放到{example['place_name']}"
    return {
        "records": tuple(records),
        "target_bytes": tuple(expected.encode("utf-8")) + (EOS,),
        "target_action": expected,
    }


def reversed_records(example: dict) -> tuple[RawSource, ...]:
    return tuple(reversed(example["records"]))


def build_config(payload: dict, arm: str):
    r1 = ARM_READER[arm] == "R1"
    if r1:
        base = build_base_from_parent(payload)
        candidate = build_binding_candidate(
            base, config=BindingCopyConfig(), profile=f"{arm}-f64", dtype=torch.float64
        )
        return candidate, make_optimizer(candidate), r1
    candidate = build_base_from_parent(payload)
    return candidate, base_make_optimizer(candidate), r1


def schedule(seed: int, train: tuple[dict, ...], updates: int) -> list[tuple[dict, ...]]:
    """Deterministic schedule with a fresh permutation per epoch."""
    batches: list[tuple[dict, ...]] = []
    epoch = 0
    generator = torch.Generator().manual_seed(ORDER_SEEDS[seed])
    permutation = torch.randperm(len(train) - len(train) % BATCH, generator=generator).tolist()
    cursor = 0
    while len(batches) < updates:
        if cursor + BATCH > len(permutation):
            epoch += 1
            generator = torch.Generator().manual_seed(ORDER_SEEDS[seed] + epoch)
            permutation = torch.randperm(
                len(train) - len(train) % BATCH, generator=generator
            ).tolist()
            cursor = 0
        batches.append(tuple(train[i] for i in permutation[cursor : cursor + BATCH]))
        cursor += BATCH
    return batches


def self_probability(update: int) -> float:
    if update < SELF_PREFIX_START:
        return 0.0
    if update < SELF_PREFIX_TIGHTEN:
        return SELF_PREFIX_PROB[SELF_PREFIX_START]
    return SELF_PREFIX_PROB[SELF_PREFIX_TIGHTEN]


def batch_loss(candidate, fixtures, *, r1: bool, self_flags) -> tuple[torch.Tensor, int]:
    score_fn = score_example if r1 else base_score_example
    gen_fn = generate_bytes if r1 else base_generate_bytes
    losses = []
    self_count = 0
    for idx, fixture in enumerate(fixtures):
        if self_flags and self_flags[idx]:
            self_count += 1
            with torch.no_grad():
                generated = gen_fn(
                    candidate,
                    fixture["goal"].encode(),
                    fixture["records"],
                    subject_id=f"cmp-self-{fixture['index']}",
                )
            target_len = len(fixture["target_bytes"]) - 1
            prefix = tuple(generated["raw_bytes"][:target_len])
            if len(prefix) < target_len:
                prefix = prefix + (EOS,) * (target_len - len(prefix))
        else:
            prefix = fixture["target_bytes"][:-1]
        losses.append(score_fn(candidate, fixture["goal"].encode(), fixture["records"], prefix).nll)
    return sum(losses) / len(losses), self_count


def train_arm(
    payload: dict,
    train: tuple[dict, ...],
    dev: tuple[dict, ...],
    *,
    seed: int,
    arm: str,
    updates: int,
    budget: Budget,
    progress: dict,
) -> dict:
    r1 = ARM_READER[arm] == "R1"
    prefix_mode = ARM_PREFIX[arm]
    candidate, optimizer, _r1 = build_config(payload, arm)
    torch.manual_seed(seed)

    step0 = evaluate(candidate, dev, r1=r1, full=False, budget=budget)
    progress["endpoints"] = {"step0": step0}

    batches = schedule(seed, train, updates)
    rows = []
    for update in range(1, updates + 1):
        batch = batches[update - 1]
        prob = self_probability(update) if prefix_mode == "self" else 0.0
        flags = (
            tuple(bool(torch.rand(1).item() < prob) for _ in range(BATCH))
            if prob > 0
            else tuple(False for _ in range(BATCH))
        )
        loss, self_count = batch_loss(candidate, batch, r1=r1, self_flags=flags)
        optimizer.zero_grad()
        loss.backward()
        clip = float(torch.nn.utils.clip_grad_norm_(candidate.trainable_parameters(), CLIP_NORM))
        optimizer.step()
        budget.adam_calls += 1
        rows.append(
            {
                "update": update,
                "loss": float(loss.detach()),
                "clip_scale": clip,
                "self_count": self_count,
                "prob": prob,
            }
        )
        progress["updates_done"] = update
        progress["loss_ema"] = (
            float(loss.detach())
            if not rows[:-1]
            else 0.9 * progress.get("loss_ema", 0.0) + 0.1 * float(loss.detach())
        )
        if update == CHECKPOINT_UPDATE:
            path = OUTPUT_DIR / f"{seed}-{arm}-step{update}.pt"
            state = (
                initial_binding_state(candidate, f"cmp-{seed}-{arm}")
                if r1
                else candidate.subject.initial_state(f"cmp-{seed}-{arm}", revision=CONTENT_REVISION)
            )
            save_fn = save_bundle if r1 else base_save_bundle
            info = save_fn(
                path,
                profile=f"{arm}-f64",
                candidate=candidate,
                state=state,
                optimizer=optimizer,
                update_index=update,
                cursor={"stage": "15.1.1", "seed": seed, "arm": arm, "update": update},
                parent_identity={"sha256": PARENT_SHA256},
            )
            budget.add_product(info)
            progress["checkpoint_150"] = {"bytes": int(info["bytes"]), "sha256": info["sha256"]}
        try:
            budget.ensure(context={"seed": seed, "arm": arm, "update": update, "phase": "train"})
        except WatchLine:
            progress["stopped_at_update"] = update
            raise

    final = evaluate(candidate, dev, r1=r1, full=True, budget=budget)
    progress["endpoints"]["step300"] = final

    final_path = OUTPUT_DIR / f"{seed}-{arm}-step300.pt"
    final_state = (
        initial_binding_state(candidate, f"cmp-final-{seed}-{arm}")
        if r1
        else candidate.subject.initial_state(f"cmp-final-{seed}-{arm}", revision=CONTENT_REVISION)
    )
    save_fn = save_bundle if r1 else base_save_bundle
    final_info = save_fn(
        final_path,
        profile=f"{arm}-f64",
        candidate=candidate,
        state=final_state,
        optimizer=optimizer,
        update_index=updates,
        cursor={"stage": "15.1.1", "seed": seed, "arm": arm, "update": updates},
        parent_identity={"sha256": PARENT_SHA256},
    )
    budget.add_product(final_info)

    return {
        "seed": seed,
        "arm": arm,
        "reader": ARM_READER[arm],
        "prefix_mode": prefix_mode,
        "updates": updates,
        "adam_calls": updates,
        "step0": step0,
        "step300": final,
        "train": {
            "first_loss": rows[0]["loss"],
            "last_loss": rows[-1]["loss"],
            "loss_last20_mean": statistics.fmean(r["loss"] for r in rows[-20:]),
            "self_count_total": sum(r["self_count"] for r in rows),
            "clip_min": min(r["clip_scale"] for r in rows),
            "clip_max": max(r["clip_scale"] for r in rows),
            "rows": rows,
        },
    }


def generate_one(candidate, fixture, *, r1: bool, records=None, subject_id: str) -> dict:
    gen_fn = generate_bytes if r1 else base_generate_bytes
    result = gen_fn(
        candidate,
        fixture["goal"].encode(),
        records if records is not None else fixture["records"],
        subject_id=subject_id,
    )
    return result


def evaluate(candidate, dev, *, r1: bool, full: bool, budget: Budget) -> dict:
    rows = []
    true_raw = {}
    for fixture in dev:
        generated = generate_one(
            candidate, fixture, r1=r1, subject_id=f"cmp-dev-{fixture['index']}"
        )
        true_raw[fixture["index"]] = generated["raw_bytes"]
        scored = score_generation(
            generated["raw_bytes"], fixture["target_bytes"][:-1], generated["terminated"]
        )
        rows.append(
            {
                "index": fixture["index"],
                "world": fixture["world"],
                "flags": fixture["flags"],
                "score": scored,
            }
        )
        budget.ensure(context={"phase": "dev_generate"})

    summary = summarize(rows)
    out: dict = {"overall": summary, "rows": rows if full else None}

    if not full:
        out["rows_count"] = len(rows)
        return out

    out["reversed"] = evaluate_reversed(candidate, dev, r1=r1, budget=budget, true_raw=true_raw)
    out["counterfactual"] = evaluate_counterfactual(candidate, dev, r1=r1, budget=budget)
    out["source_top1"] = evaluate_source_top1(candidate, dev, r1=r1, budget=budget)
    out["identity_rescue"] = evaluate_identity_rescue(candidate, dev, r1=r1, budget=budget)
    out["content_interventions"] = evaluate_interventions(candidate, dev, r1=r1, budget=budget)
    out["nll"] = evaluate_nll(candidate, dev, r1=r1, budget=budget)
    return out


def evaluate_reversed(candidate, dev, *, r1: bool, budget: Budget, true_raw: dict) -> dict:
    rows = []
    for fixture in dev:
        generated = generate_one(
            candidate,
            fixture,
            r1=r1,
            records=reversed_records(fixture),
            subject_id=f"cmp-rev-{fixture['index']}",
        )
        scored = score_generation(
            generated["raw_bytes"], fixture["target_bytes"][:-1], generated["terminated"]
        )
        paired = generated["raw_bytes"] == true_raw.get(fixture["index"], b"")
        rows.append(
            {
                "index": fixture["index"],
                "world": fixture["world"],
                "score": scored,
                "paired_with_true": bool(paired),
            }
        )
        budget.ensure(context={"phase": "reversed"})
    accuracy = sum(1 for r in rows if r["score"]["completed"]) / len(rows)
    paired_rate = sum(1 for r in rows if r["paired_with_true"]) / len(rows)
    return {"accuracy": accuracy, "paired_rate": paired_rate, "rows": rows}


def evaluate_counterfactual(candidate, dev, *, r1: bool, budget: Budget) -> dict:
    rows = []
    for fixture in dev:
        swapped = swap_object_name(fixture)
        if swapped is None:
            continue
        generated = generate_one(
            candidate,
            fixture,
            r1=r1,
            records=swapped["records"],
            subject_id=f"cmp-map-{fixture['index']}",
        )
        scored = score_generation(
            generated["raw_bytes"], swapped["target_bytes"][:-1], generated["terminated"]
        )
        true_generated = generate_one(
            candidate, fixture, r1=r1, subject_id=f"cmp-true-{fixture['index']}"
        )
        true_scored = score_generation(
            true_generated["raw_bytes"], fixture["target_bytes"][:-1], true_generated["terminated"]
        )
        rows.append(
            {
                "index": fixture["index"],
                "world": fixture["world"],
                "counterfactual_completed": scored["completed"],
                "true_completed": true_scored["completed"],
            }
        )
        budget.ensure(context={"phase": "counterfactual"})
    both = sum(1 for r in rows if r["counterfactual_completed"] and r["true_completed"])
    return {
        "applicable": len(rows),
        "both_correct": both / len(rows) if rows else 0.0,
        "rows": rows,
    }


def evaluate_source_top1(candidate, dev, *, r1: bool, budget: Budget) -> dict:
    if not r1:
        return {"applicable": False, "reason": "R0 has no span reader to attribute"}
    rows = []
    for fixture in dev:
        gen_state = initial_binding_state(candidate, f"cmp-src-{fixture['index']}")
        prepared = prepare(candidate, gen_state, fixture["goal"].encode(), fixture["records"])
        state, gate_index = observe_goal(candidate, prepared)
        step = answer_step(candidate, prepared, state, EOS, gate_index=gate_index)
        weights = step.read.copy_weights
        offsets = prepared.base.view.record_offsets
        object_index = next(
            (
                index
                for index, record in enumerate(fixture["raw_example"]["records"])
                if record["kind"] == "object"
                and record["description"] == fixture["object_description"]
            ),
            None,
        )
        chosen = int(torch.argmax(weights).item()) if weights.numel() else -1
        chosen_record = None
        for index, (start, end) in enumerate(offsets):
            if start <= chosen < end:
                chosen_record = index
                break
        correct = bool(object_index is not None and chosen_record == object_index)
        rows.append(
            {
                "index": fixture["index"],
                "world": fixture["world"],
                "chosen_record": chosen_record,
                "object_record": object_index,
                "correct": correct,
            }
        )
        budget.ensure(context={"phase": "source_top1"})
    return {"top1": sum(1 for r in rows if r["correct"]) / len(rows), "rows": rows}


def evaluate_identity_rescue(candidate, dev, *, r1: bool, budget: Budget) -> dict:
    """Identity rescue: applying and then removing an intervention must return
    the distribution bit-for-bit to the untouched one."""
    if not r1:
        return {"applicable": False, "reason": "R0 keeps the 14 identity path"}
    rows = []
    for fixture in dev[:8]:
        gen_state = initial_binding_state(candidate, f"cmp-rescue-{fixture['index']}")
        prepared = prepare(candidate, gen_state, fixture["goal"].encode(), fixture["records"])
        state, gate_index = observe_goal(candidate, prepared)
        baseline = answer_step(candidate, prepared, state, EOS, gate_index=gate_index)
        intervened = answer_step(
            candidate, prepared, state, EOS, gate_index=gate_index, gamma_override=0.0
        )
        restored = answer_step(candidate, prepared, state, EOS, gate_index=gate_index)
        rows.append(
            {
                "index": fixture["index"],
                "intervention_diff": float(
                    (intervened.probabilities.detach() - baseline.probabilities.detach())
                    .abs()
                    .max()
                ),
                "rescue_equal": bool(torch.equal(restored.probabilities, baseline.probabilities)),
                "rescue_max_abs_diff": float(
                    (restored.probabilities.detach() - baseline.probabilities.detach()).abs().max()
                ),
            }
        )
        budget.ensure(context={"phase": "identity_rescue"})
    return {
        "equal": sum(1 for r in rows if r["rescue_equal"]),
        "denominator": len(rows),
        "rows": rows,
    }


def generate_policy(
    candidate,
    fixture,
    *,
    r1: bool,
    records=None,
    gamma_override=None,
    force_record: int | None = None,
    force_offset: int | None = None,
    subject_id: str,
) -> dict:
    """Generate one answer, forcing the copy path on the first step if asked."""
    if not r1:
        return generate_one(candidate, fixture, r1=False, records=records, subject_id=subject_id)
    state = initial_binding_state(candidate, subject_id)
    prepared = prepare(
        candidate,
        state,
        fixture["goal"].encode(),
        records if records is not None else fixture["records"],
    )
    state, gate_index = observe_goal(candidate, prepared)
    previous = EOS
    raw: list[int] = []
    terminated = False
    for _prediction in range(MAX_PREDICTIONS):
        if force_record is not None or force_offset is not None:
            step = forced_read(
                candidate,
                prepared,
                state,
                previous,
                record_index=force_record,
                offset=force_offset if force_offset is not None else 0,
                gate_index=gate_index,
                gamma_override=gamma_override,
            )
        else:
            step = answer_step(
                candidate,
                prepared,
                state,
                previous,
                gate_index=gate_index,
                gamma_override=gamma_override,
            )
        endpoint = decode_endpoint(step.probabilities)
        if endpoint["eos"]:
            terminated = True
            break
        raw.append(int(endpoint["byte"]))
        if len(raw) > MAX_OUTPUT_BYTES:
            break
        m_next, coverage_next = advance_copy(
            candidate.reader,
            state,
            step.read,
            step.probabilities,
            step.gamma,
            endpoint["byte"],
            prepared.base.view,
        )
        state = replace(
            step.state,
            m=m_next,
            coverage=coverage_next,
            prefix_cursor=step.state.prefix_cursor + 1,
        )
        gate_index = step.gate_index
        previous = int(endpoint["byte"])
        # Only the first step is forced; later steps follow the model.
        force_record = None
        force_offset = None
    return {"raw_bytes": bytes(raw), "terminated": terminated}


def object_record_index(fixture: dict) -> int | None:
    return next(
        (
            index
            for index, record in enumerate(fixture["raw_example"]["records"])
            if record["kind"] == "object" and record["description"] == fixture["object_description"]
        ),
        None,
    )


def alternate_object_index(fixture: dict) -> int | None:
    correct = object_record_index(fixture)
    if correct is None:
        return None
    names = {record["source_id"]: record["name"] for record in object_records(fixture)}
    target_name = names.get(fixture["raw_example"]["records"][correct]["source_id"])
    for index, record in enumerate(fixture["raw_example"]["records"]):
        if index == correct or record["kind"] != "object":
            continue
        if names.get(record["source_id"]) != target_name:
            return index
    return None


def evaluate_interventions(candidate, dev, *, r1: bool, budget: Budget) -> dict:
    """Content necessity: no-source / wrong-record / wrong-position / gamma=0."""
    if not r1:
        return {"applicable": False, "reason": "R0 keeps the 14 content path"}
    targets = [d for d in dev if int(d["index"]) % 15 == 0]
    rows = []
    for fixture in targets:
        tag = fixture["index"]
        true_generated = generate_one(candidate, fixture, r1=r1, subject_id=f"cmp-int-true-{tag}")
        true_scored = score_generation(
            true_generated["raw_bytes"], fixture["target_bytes"][:-1], true_generated["terminated"]
        )
        row = {
            "index": tag,
            "world": fixture["world"],
            "true_completed": true_scored["completed"],
        }
        no_source = generate_one(
            candidate, fixture, r1=r1, records=(), subject_id=f"cmp-int-nosrc-{tag}"
        )
        row["no_source_completed"] = score_generation(
            no_source["raw_bytes"], fixture["target_bytes"][:-1], no_source["terminated"]
        )["completed"]
        alternate = alternate_object_index(fixture)
        wrong_record = generate_policy(
            candidate,
            fixture,
            r1=r1,
            force_record=alternate,
            force_offset=0,
            subject_id=f"cmp-int-wrec-{tag}",
        )
        row["wrong_record_completed"] = score_generation(
            wrong_record["raw_bytes"], fixture["target_bytes"][:-1], wrong_record["terminated"]
        )["completed"]
        correct = object_record_index(fixture)
        wrong_position = generate_policy(
            candidate,
            fixture,
            r1=r1,
            force_record=correct,
            force_offset=0,
            subject_id=f"cmp-int-wpos-{tag}",
        )
        row["wrong_position_completed"] = score_generation(
            wrong_position["raw_bytes"], fixture["target_bytes"][:-1], wrong_position["terminated"]
        )["completed"]
        gamma0 = generate_policy(
            candidate, fixture, r1=r1, gamma_override=0.0, subject_id=f"cmp-int-g0-{tag}"
        )
        row["gamma0_completed"] = score_generation(
            gamma0["raw_bytes"], fixture["target_bytes"][:-1], gamma0["terminated"]
        )["completed"]
        oracle = generate_policy(
            candidate,
            fixture,
            r1=r1,
            force_record=correct,
            force_offset=0,
            subject_id=f"cmp-int-oracle-{tag}",
        )
        row["forced_record_oracle_completed"] = score_generation(
            oracle["raw_bytes"], fixture["target_bytes"][:-1], oracle["terminated"]
        )["completed"]
        rows.append(row)
        budget.ensure(context={"phase": "interventions"})
    drops = {}
    for key in ("no_source", "wrong_record", "wrong_position", "gamma0"):
        rate = sum(1 for r in rows if r[f"{key}_completed"]) / len(rows) if rows else 0.0
        true_rate = sum(1 for r in rows if r["true_completed"]) / len(rows) if rows else 0.0
        drops[key] = {"completed_rate": rate, "drop": true_rate - rate}
    oracle_rate = (
        sum(1 for r in rows if r["forced_record_oracle_completed"]) / len(rows) if rows else 0.0
    )
    return {
        "examples": len(rows),
        "true_completed_rate": (
            sum(1 for r in rows if r["true_completed"]) / len(rows) if rows else 0.0
        ),
        "drops": drops,
        "forced_record_oracle": {
            "completed_rate": oracle_rate,
            "note": "upper bound and reachability only; no false gate",
        },
        "rows": rows,
    }


def evaluate_nll(candidate, dev, *, r1: bool, budget: Budget) -> dict:
    score_fn = score_example if r1 else base_score_example
    total = 0.0
    for fixture in dev:
        with torch.no_grad():
            total += float(
                score_fn(
                    candidate,
                    fixture["goal"].encode(),
                    fixture["records"],
                    fixture["target_bytes"][:-1],
                    subject_id=f"cmp-nll-{fixture['index']}",
                ).nll
            )
        budget.ensure(context={"phase": "nll"})
    return {"mean_nll": total / len(dev), "examples": len(dev)}


def summarize(rows) -> dict:
    def rate(pred) -> float:
        return sum(1 for r in rows if pred(r)) / len(rows) if rows else 0.0

    flags = {
        "unseen_combination": rate(
            lambda r: r["flags"]["unseen_combination"] and r["score"]["completed"]
        ),
        "unseen_output_name": rate(
            lambda r: r["flags"]["unseen_output_name"] and r["score"]["completed"]
        ),
        "new_description": rate(
            lambda r: r["flags"]["new_description"] and r["score"]["completed"]
        ),
        "both_new": rate(lambda r: r["flags"]["both_new"] and r["score"]["completed"]),
    }
    denominators = {
        "unseen_combination": sum(1 for r in rows if r["flags"]["unseen_combination"]),
        "unseen_output_name": sum(1 for r in rows if r["flags"]["unseen_output_name"]),
        "new_description": sum(1 for r in rows if r["flags"]["new_description"]),
        "both_new": sum(1 for r in rows if r["flags"]["both_new"]),
    }
    by_world = {}
    for row in rows:
        entry = by_world.setdefault(str(row["world"]), {"n": 0, "completed": 0})
        entry["n"] += 1
        entry["completed"] += int(row["score"]["completed"])
    return {
        "completed": rate(lambda r: r["score"]["completed"]),
        "subsets": flags,
        "subset_denominators": denominators,
        "utf8": rate(lambda r: r["score"]["utf8"]),
        "eos": rate(lambda r: r["score"]["terminated"]),
        "full_sentence": rate(lambda r: r["score"]["full_sentence"]),
        "by_world": by_world,
        "examples": len(rows),
    }


def bootstrap(deltas: list[float]) -> dict:
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    means = []
    n = len(deltas)
    for _ in range(BOOTSTRAP_REPS):
        idx = torch.randint(0, n, (n,), generator=generator).tolist()
        means.append(sum(deltas[i] for i in idx) / n)
    means.sort()
    return {
        "mean": sum(deltas) / n,
        "lower": means[int(0.025 * BOOTSTRAP_REPS)],
        "upper": means[int(0.975 * BOOTSTRAP_REPS)],
        "reps": BOOTSTRAP_REPS,
        "seed": BOOTSTRAP_SEED,
    }


def write_partial(payload: dict) -> None:
    Path(PARTIAL).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def task_gate(entry: dict) -> dict:
    overall = entry["step300"]["overall"]
    checks = {
        "overall": overall["completed"] >= GATE_OVERALL,
        "unseen_combination": overall["subsets"]["unseen_combination"] >= GATE_SUBSET,
        "unseen_output_name": overall["subsets"]["unseen_output_name"] >= GATE_SUBSET,
        "new_description": overall["subsets"]["new_description"] >= GATE_SUBSET,
        "both_correct_counterfactual": entry["step300"]["counterfactual"]["both_correct"]
        >= GATE_BOTH_CORRECT,
        "reversed_paired": entry["step300"]["reversed"]["paired_rate"] >= GATE_REVERSED_PAIRED,
        "utf8": overall["utf8"] >= GATE_UTF8,
        "eos": overall["eos"] >= GATE_EOS,
    }
    return {"checks": checks, "passed": all(checks.values())}


def world_rates(entry: dict) -> dict[str, float]:
    by_world = entry["step300"]["overall"]["by_world"]
    return {world: stats["completed"] / stats["n"] for world, stats in by_world.items()}


def contrast(entries: dict, base_arm: str, other_arm: str) -> dict:
    per_seed = {}
    for seed in sorted({key[0] for key in entries if key[1] in (base_arm, other_arm)}):
        base = world_rates(entries[(seed, base_arm)])
        other = world_rates(entries[(seed, other_arm)])
        deltas = [other[w] - base[w] for w in sorted(base)]
        per_seed[str(seed)] = {
            "delta_mean": sum(deltas) / len(deltas),
            "bootstrap": bootstrap(deltas),
            "world_deltas": deltas,
        }
    pooled = [stat["delta_mean"] for stat in per_seed.values()]
    return {
        "base_arm": base_arm,
        "other_arm": other_arm,
        "per_seed": per_seed,
        "pooled_mean": sum(pooled) / len(pooled),
    }


def build_report(configs, budget, progress, payload, args) -> dict:
    entries = {(c["seed"], c["arm"]): c for c in configs}
    gates = {f"{seed}-{arm}": task_gate(entry) for (seed, arm), entry in sorted(entries.items())}
    contrasts = {
        "package_R1P1_minus_R0P0": contrast(entries, "R0P0", "R1P1"),
        "reader_R1P0_minus_R0P0": contrast(entries, "R0P0", "R1P0"),
        "prefix_R0P1_minus_R0P0": contrast(entries, "R0P0", "R0P1"),
        "reader_under_self_R1P1_minus_R0P1": contrast(entries, "R0P1", "R1P1"),
        "prefix_under_reader_R1P1_minus_R1P0": contrast(entries, "R1P0", "R1P1"),
    }
    primary = contrasts["package_R1P1_minus_R0P0"]
    mechanism = {
        "package_gain": primary["pooled_mean"],
        "package_gain_ok": primary["pooled_mean"] >= GATE_PACKAGE_GAIN,
        "package_bootstrap_ok": all(
            stat["bootstrap"]["lower"] > 0 for stat in primary["per_seed"].values()
        ),
        "source_top1": {
            f"{key[0]}-{key[1]}": entry["step300"]["source_top1"].get("top1")
            for key, entry in entries.items()
            if ARM_READER[key[1]] == "R1"
        },
        "source_top1_ok": all(
            entry["step300"]["source_top1"].get("top1", 0.0) >= GATE_SOURCE_TOP1
            for key, entry in entries.items()
            if ARM_READER[key[1]] == "R1"
        ),
        "identity_rescue": {
            f"{key[0]}-{key[1]}": entry["step300"]["identity_rescue"].get("equal")
            for key, entry in entries.items()
            if ARM_READER[key[1]] == "R1"
        },
        "identity_rescue_ok": all(
            entry["step300"]["identity_rescue"].get("equal", 0)
            == entry["step300"]["identity_rescue"].get("denominator", -1)
            for key, entry in entries.items()
            if ARM_READER[key[1]] == "R1"
        ),
        "content_drops": {
            f"{key[0]}-{key[1]}": entry["step300"]["content_interventions"]["drops"]
            for key, entry in entries.items()
            if ARM_READER[key[1]] == "R1"
        },
        "content_drops_ok": all(
            entry["step300"]["content_interventions"]["drops"][name]["drop"]
            >= GATE_INTERVENTION_DROP
            for key, entry in entries.items()
            if ARM_READER[key[1]] == "R1"
            for name in ("no_source", "wrong_record", "wrong_position")
        ),
    }
    all_task = all(gate["passed"] for gate in gates.values())
    mech_ok = all(
        [
            mechanism["package_gain_ok"],
            mechanism["package_bootstrap_ok"],
            mechanism["source_top1_ok"],
            mechanism["identity_rescue_ok"],
            mechanism["content_drops_ok"],
        ]
    )
    if all_task and mech_ok:
        verdict = "binding_copy_supported_on_declared_task"
    elif all_task and not mech_ok:
        verdict = "task_supported_component_gain_unresolved"
    elif not all_task:
        verdict = "native_byte_task_failed"
    else:
        verdict = "engineering_failure"

    source_hashes = {f: sha256_file(ROOT / f) for f in SOURCE_FILES}
    return {
        "schema": "taiji-binding-copy-compare-v1",
        "step": "15.1.1",
        "date": "2026-11-02",
        "status": verdict,
        "verdict": verdict,
        "gates": {"task": gates, "mechanism": mechanism},
        "contrasts": contrasts,
        "configs": configs,
        "budget_ruling": {
            "date": "2026-11-02",
            "scope": "15.1 only",
            "cpu_cap": None,
            "wall_cap": None,
            "observation_line_cpu": WATCH_CPU_SECONDS,
            "observation_line_wall": WATCH_WALL_SECONDS,
            "dose_unchanged": "2 seeds x 4 arms x 300 updates, batch 8, 288 train / 240 dev",
        },
        "accounting": {
            "actual_adam_calls": budget.adam_calls,
            "effective_adam_calls": budget.adam_calls,
            "consumed_train_examples": budget.adam_calls * BATCH,
            "consumed_dev_examples": len(configs) * len(payload.get("examples", [])) * 0,
            "consumed_confirm_examples": 0,
        },
        "cost": {
            "cpu_seconds": budget.cpu(),
            "wall_seconds": budget.wall_now(),
            "adam_calls": budget.adam_calls,
            "peak_live_product_bytes": budget.max_live_bytes,
            "cumulative_product_bytes": budget.cumulative_bytes,
        },
        "source_hashes": source_hashes,
        "limits": (
            "Formal bounded comparison on the declared native-byte task only. No confirm read, "
            "no N15.2, no step 12/13, no 14.7 release, no capability claim beyond this task."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dev-limit", type=int, default=0)
    parser.add_argument("--force-report", action="store_true")
    args = parser.parse_args()
    arms = tuple(args.arms.split(","))
    seeds = tuple(int(s) for s in args.seeds.split(","))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    budget = Budget()
    progress = {"configs_done": 0, "configs_total": len(seeds) * len(arms)}

    try:
        parent_path = ROOT / PARENT_BUNDLE
        if sha256_file(parent_path) != PARENT_SHA256:
            raise SystemExit("parent bundle sha drifted")
        payload = torch.load(parent_path, map_location="cpu", weights_only=False)
        train, dev, data_payload = load_data()
        if args.dev_limit:
            dev = dev[: args.dev_limit]

        configs = []
        for seed in seeds:
            for arm in arms:
                progress["current"] = {"seed": seed, "arm": arm}
                result = train_arm(
                    payload,
                    train,
                    dev,
                    seed=seed,
                    arm=arm,
                    updates=args.updates,
                    budget=budget,
                    progress=progress,
                )
                configs.append(result)
                progress["configs_done"] += 1
                if args.smoke:
                    break

        formal = (
            len(configs) == len(seeds) * len(arms)
            and len(configs) == len(SEEDS) * len(ARMS)
            and (args.updates == UPDATES or args.force_report)
            and (not args.dev_limit or args.force_report)
        )
        if formal:
            smoke_run = bool(args.force_report) and (
                args.updates != UPDATES or bool(args.dev_limit)
            )
            report_path = SMOKE_REPORT if smoke_run else REPORT
            report = build_report(configs, budget, progress, data_payload, args)
            Path(report_path).write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            if not smoke_run:
                manifest = {
                    "schema": "taiji-binding-copy-manifest-v1",
                    "step": "15.1.1",
                    "status": report["status"],
                    "report": REPORT,
                    "parent_bundle": PARENT_BUNDLE,
                    "parent_sha256": PARENT_SHA256,
                    "seeds": list(SEEDS),
                    "arms": list(ARMS),
                    "updates": UPDATES,
                    "batch": BATCH,
                    "dev_examples": len(dev),
                    "source_hashes": report["source_hashes"],
                    "gates": report["gates"],
                    "cost": report["cost"],
                    "accounting": report["accounting"],
                    "products": sorted(str(p) for p in OUTPUT_DIR.glob("*.pt")),
                }
                Path(MANIFEST).write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            print(f"1.1 status: {report['status']}")
            print(f"  package gain: {report['gates']['mechanism']['package_gain']:.4f}")
            print(f"  cost: {report['cost']}")

        write_partial(
            {
                "schema": "taiji-binding-copy-compare-partial-v1",
                "step": "15.1.1",
                "smoke": bool(args.smoke),
                "updates": int(args.updates),
                "configs": configs,
                "progress": progress,
                "cost": {
                    "cpu_seconds": budget.cpu(),
                    "wall_seconds": budget.wall_now(),
                    "adam_calls": budget.adam_calls,
                    "peak_live_product_bytes": budget.max_live_bytes,
                },
            }
        )
        print(f"compare smoke/partial written: {PARTIAL}")
        print(f"  cpu {budget.cpu():.2f}, wall {budget.wall_now():.2f}, adam {budget.adam_calls}")
        return 0

    except WatchLine as exc:
        info = json.loads(str(exc))
        write_partial(
            {
                "schema": "taiji-binding-copy-compare-partial-v1",
                "step": "15.1.1",
                "status": "watch_line_reported",
                "detail": info,
                "progress": progress,
            }
        )
        print(f"WATCH LINE reached: {info}")
        return 2
    except CompareStop as exc:
        print(f"compare stop: {exc}")
        return 1
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
