"""20.4a: decompose the unit cost of one outer update and trial optimizations.

The 20.4 pricing priced the shipped protocol at 167.50 CPU per batch-2 update,
i.e. 144,648.7 CPU for 200 updates x 4 runs.  That is 1.6 CPU per turn of a
3.3M-parameter graph, two or three orders of magnitude above the arithmetic.
This probe attributes the cost instead of shrinking the experiment: it times the
phases of a real update, names the top consumers, and measures the candidate
optimizations that can be run without changing R021 numerics.

No formal training step is taken: the only updates here are calibration samples.
No confirm world is read, the graph is not edited, ``addressed_store`` stays at
its frozen value (off), and no threshold is altered.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from taiji import rule_learning_20_1 as fixture  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.rule_learning_20_1_net import (  # noqa: E402
    ANSWER_BYTES,
    DECISION_POSITIONS,
    TEMPLATE_BYTES,
    RuleLearningGraph,
    _group_of,
    copy_subject,
)
from taiji.selection_routing_checkpoint import load_routing_bundle  # noqa: E402

REPORT = ROOT / "reports/taiji_package20_4a_unit_cost_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package20_4a_unit_cost_20261105.json"
PROGRESS = ROOT / "reports/taiji_package20_4a_unit_cost_20261105_progress.jsonl"
BUNDLE = ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt"

CPU_CAP = 800.0
CPU_GUARD = 800.0
WALL_CAP = 1800.0
FROZEN_BUDGET_CPU = 24000.0
MAGNITUDE_CEILING_CPU = 2 * FROZEN_BUDGET_CPU
UPDATES_PER_SEED = 200
SEEDS = (20261105, 20261106)
ARMS = ("CF1", "B_EH")
BATCH = 2
LEARNING_RATE = 1e-3
GRAD_CLIP = 1.0
TRAIN_INDICES = tuple(range(0, 160))
DEV_INDICES = tuple(range(160, 240))
ADDRESSED_STORE = False
EARLY_STOP_PROBES_PER_RUN = len((25, 50, 100)) * 5
THREADS = 1
BASE_SAMPLES = 5
TRUNCATED_SAMPLES = 5
T3_SAMPLES = 5
READOUT_TURN_INDEX = 5
EQUIVALENCE_TOLERANCE = 1e-6
TOTAL_TURNS = fixture.TOTAL_TURNS

START_WALL = time.time()
START_CPU = time.process_time()

PHASE = {"observe": 0.0, "decode": 0.0}


class BudgetStop(RuntimeError):
    pass


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def guard(stage: str) -> None:
    cpu, wall = cost()
    if cpu > CPU_GUARD or wall > WALL_CAP:
        append_progress({"stage": "budget_stop", "at": stage, "cpu": round(cpu, 2)})
        raise BudgetStop(stage)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def append_progress(row: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_progress() -> list[dict]:
    if not PROGRESS.exists():
        return []
    rows = []
    for line in PROGRESS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_graph(candidate, gate, arm: str) -> RuleLearningGraph:
    return RuleLearningGraph(
        copy_subject(candidate.subject),
        arm=arm,
        growth_gates=gate,
        addressed_store_enabled=ADDRESSED_STORE,
    )


def unfreeze_outer(subject) -> None:
    for name, parameter in subject.named_parameters():
        parameter.requires_grad_(_group_of(name) != "world")


def gold_bytes(world: dict, number: int) -> bytes:
    return world["scoring"]["turns"][str(number)]["expected_action"].encode("utf-8")


def legal_vocab(released: list[dict]) -> tuple[int, ...]:
    ids = set(TEMPLATE_BYTES.values())
    for item in released:
        ids.update(item["feedback"].encode("utf-8"))
    return tuple(sorted(ids))


def train_turn(graph, state, turn: dict, released: list[dict], now: int):
    query = turn["query"].encode("utf-8")
    records = graph._records(released, now)
    started = time.process_time()
    living = graph.subject.observe(
        state.living,
        query,
        goal=query,
        records=records,
        allowed=graph._allowed(False),
        now=now,
        **graph.step_kwargs,
    )
    PHASE["observe"] += time.process_time() - started
    logits = []
    emitted = []
    previous = EOS
    started = time.process_time()
    for _ in range(ANSWER_BYTES):
        event = graph._patch(graph.subject, previous)
        output = graph.subject.step(
            living,
            event=event,
            goal=query,
            records=records,
            previous_byte=previous,
            allowed=graph._allowed(False),
            now=now,
            **graph.step_kwargs,
        )
        logits.append(output.probabilities)
        choice = int(torch.argmax(output.probabilities[:256].detach()))
        emitted.append(choice)
        living = graph.subject.consume_output(output, choice)
        previous = choice
    PHASE["decode"] += time.process_time() - started
    return torch.stack(logits), tuple(emitted), type(state)(living, state.journal, state.revision)


def train_release(graph, state, release: dict, released: list[dict], now: int, write: bool):
    records = graph._records(released, now)
    living = state.living
    written = 0
    for item in release["items"]:
        feedback = item["feedback"].encode("utf-8")
        query = item["query"].encode("utf-8")
        before = living.fast.detach().clone()
        started = time.process_time()
        living = graph.subject.observe(
            living,
            feedback,
            goal=query,
            records=records,
            allowed=graph._allowed(write),
            now=now,
            **graph.step_kwargs,
        )
        PHASE["observe"] += time.process_time() - started
        written += int(not torch.equal(before, living.fast.detach()))
    return type(state)(living, state.journal, state.revision), written


def detach_state(state):
    living = state.living
    return type(state)(
        replace(
            living,
            h=living.h.detach(),
            fast=living.fast.detach(),
            coverage=living.coverage.detach(),
            prior=living.prior.detach(),
            nu_ema=living.nu_ema.detach(),
        ),
        state.journal,
        state.revision,
    )


def run_world(graph, world: dict, truncate: bool) -> dict:
    response = world["response"]
    state = graph.initial_state(response["world_id"])
    released: list[dict] = []
    losses = []
    decision_bits = 0
    template_bits = 0
    full_bytes = 0
    writes = 0
    surfaces = 0
    template_positions = [p for p in range(ANSWER_BYTES) if p not in DECISION_POSITIONS]
    for turn in response["turns"]:
        number = int(turn["turn"])
        logits, emitted, state = train_turn(graph, state, turn, released, number)
        if truncate:
            state = detach_state(state)
        gold = gold_bytes(world, number)
        decision_bits += sum(int(emitted[p] == gold[p]) for p in DECISION_POSITIONS)
        template_bits += sum(int(emitted[p] == gold[p]) for p in template_positions)
        full_bytes += int(tuple(emitted) == tuple(gold))
        if turn["correction_surface"] and released:
            surfaces += 1
            term = logits.new_zeros(())
            for position in DECISION_POSITIONS:
                term = term - torch.log(logits[position][gold[position]] + 1e-12)
            losses.append(term)
        for release in response["releases"]:
            if int(release["after_turn"]) != number:
                continue
            state, count = train_release(
                graph, state, release, released, number, bool(graph.fast_enabled)
            )
            if truncate:
                state = detach_state(state)
            writes += count
            released.extend(
                {
                    "source_id": item["source_id"],
                    "feedback": item["feedback"],
                    "version": item["version"],
                    "after_turn": number,
                }
                for item in release["items"]
            )
    loss = torch.stack(losses).sum() if losses else None
    return {
        "loss": loss,
        "surfaces": surfaces,
        "decision_bit_accuracy": decision_bits / (surfaces * len(DECISION_POSITIONS)),
        "template_bit_accuracy": template_bits / (TOTAL_TURNS * len(template_positions)),
        "full_byte_accuracy": full_bytes / TOTAL_TURNS,
        "fast_writes": writes,
    }


def update_timed(graph, optimizer, worlds: list[dict], truncate: bool, keep_grads: bool) -> dict:
    PHASE["observe"] = 0.0
    PHASE["decode"] = 0.0
    started = time.process_time()
    optimizer.zero_grad(set_to_none=True)
    zero_cpu = time.process_time() - started
    started = time.process_time()
    rows = [run_world(graph, world, truncate) for world in worlds]
    forward_cpu = time.process_time() - started
    loss = torch.stack([row["loss"] for row in rows]).sum()
    started = time.process_time()
    loss.backward()
    backward_cpu = time.process_time() - started
    grads = None
    if keep_grads:
        grads = torch.cat(
            [
                (
                    parameter.grad.detach().reshape(-1).clone()
                    if parameter.grad is not None
                    else torch.zeros(parameter.numel(), dtype=parameter.dtype)
                )
                for parameter in graph.subject.parameters()
                if parameter.requires_grad
            ]
        )
    started = time.process_time()
    torch.nn.utils.clip_grad_norm_(
        [p for p in graph.subject.parameters() if p.requires_grad], GRAD_CLIP
    )
    optimizer.step()
    optimizer_cpu = time.process_time() - started
    total_cpu = zero_cpu + forward_cpu + backward_cpu + optimizer_cpu
    return {
        "loss": float(loss.detach()),
        "loss_tensor": loss.detach().clone(),
        "grads": grads,
        "rows": rows,
        "zero_cpu": zero_cpu,
        "observe_cpu": PHASE["observe"],
        "decode_cpu": PHASE["decode"],
        "forward_cpu": forward_cpu,
        "backward_cpu": backward_cpu,
        "optimizer_cpu": optimizer_cpu,
        "total_cpu": total_cpu,
        "surfaces": sum(row["surfaces"] for row in rows),
        "fast_writes": sum(row["fast_writes"] for row in rows),
    }


def build_state_upto(graph, world: dict, stop: int):
    response = world["response"]
    state = graph.initial_state("readout")
    released: list[dict] = []
    with torch.no_grad():
        for turn in response["turns"]:
            number = int(turn["turn"])
            if number >= stop:
                break
            _, _, state = train_turn(graph, state, turn, released, number)
            for release in response["releases"]:
                if int(release["after_turn"]) != number:
                    continue
                state, _ = train_release(
                    graph, state, release, released, number, bool(graph.fast_enabled)
                )
                released.extend(
                    {
                        "source_id": item["source_id"],
                        "feedback": item["feedback"],
                        "version": item["version"],
                        "after_turn": number,
                    }
                    for item in release["items"]
                )
    return state, released


def readout_signature(graph, state, turn: dict, released: list[dict], world: dict, inference: bool):
    """One reading-path turn under ``no_grad`` or ``inference_mode`` plus its signature."""
    context = torch.inference_mode() if inference else torch.no_grad()
    started = time.process_time()
    with context:
        logits, emitted, _ = train_turn(graph, state, turn, released, int(turn["turn"]))
        gold = gold_bytes(world, int(turn["turn"]))
        term = logits.new_zeros(())
        for position in DECISION_POSITIONS:
            term = term - torch.log(logits[position][gold[position]] + 1e-12)
        signature = (float(term), tuple(emitted), logits.detach().cpu().numpy().copy())
    return signature, time.process_time() - started


def top_ops(profiler, limit: int = 10) -> list[dict]:
    rows = []
    for event in profiler.key_averages():
        rows.append(
            {
                "op": event.key,
                "self_cpu_s": round(event.self_cpu_time_total / 1e6, 4),
                "calls": int(event.count),
            }
        )
    rows.sort(key=lambda item: item["self_cpu_s"], reverse=True)
    return rows[:limit]


def profile_micro(graph, world: dict) -> dict:
    """Profile one mid-world turn (forward) and its backward on the actual graph."""
    response = world["response"]
    state = graph.initial_state("profile")
    released: list[dict] = []
    with torch.no_grad():
        for turn in response["turns"][:5]:
            number = int(turn["turn"])
            _, _, state = train_turn(graph, state, turn, released, number)
            for release in response["releases"]:
                if int(release["after_turn"]) == number:
                    state, _ = train_release(
                        graph, state, release, released, number, bool(graph.fast_enabled)
                    )
                    released.extend(
                        {
                            "source_id": item["source_id"],
                            "feedback": item["feedback"],
                            "version": item["version"],
                            "after_turn": number,
                        }
                        for item in release["items"]
                    )
    turn = response["turns"][5]
    forward = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=False
    )
    backward = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=False
    )
    with forward:
        logits, _, _ = train_turn(graph, state, turn, released, int(turn["turn"]))
        gold = gold_bytes(world, int(turn["turn"]))
        term = logits.new_zeros(())
        for position in DECISION_POSITIONS:
            term = term - torch.log(logits[position][gold[position]] + 1e-12)
    with backward:
        term.backward()
    return {
        "turn": int(turn["turn"]),
        "released_lines": len(released),
        "forward_top": top_ops(forward),
        "backward_top": top_ops(backward),
    }


def phase_shares(row: dict) -> dict:
    accounting = {
        "data_prep": row["data_prep_cpu"],
        "observe": row["observe_cpu"],
        "decode": row["decode_cpu"],
        "harness_bookkeeping": max(
            row["forward_cpu"] - row["observe_cpu"] - row["decode_cpu"], 0.0
        ),
        "backward": row["backward_cpu"],
        "optimizer_step": row["optimizer_cpu"],
        "zero_grad": row.get("zero_cpu", 0.0),
    }
    total = row["total_cpu"] + row["data_prep_cpu"]
    shares = {key: round(value / total, 4) for key, value in accounting.items()}
    return {"seconds": {k: round(v, 3) for k, v in accounting.items()}, "shares": shares}


def main() -> int:
    existing = load_progress()
    base_reuse = [row for row in existing if row.get("stage") == "baseline_sample"]
    trunc_reuse = [row for row in existing if row.get("stage") == "truncated_sample"]
    trunc_done = next((row for row in existing if row.get("stage") == "truncated_done"), None)
    reuse = (
        len(base_reuse) >= BASE_SAMPLES
        and len(trunc_reuse) >= TRUNCATED_SAMPLES
        and trunc_done is not None
    )
    if reuse:
        append_progress(
            {
                "stage": "reuse_previous_measurement_pass",
                "baseline_samples": len(base_reuse),
                "truncated_samples": len(trunc_reuse),
                "note": "no new calibration update is executed in this pass",
            }
        )
    else:
        if PROGRESS.exists():
            PROGRESS.unlink()
        append_progress(
            {
                "stage": "protocol",
                "threads": THREADS,
                "dtype": "float64",
                "deterministic": True,
                "base_samples": BASE_SAMPLES,
                "truncated_samples": TRUNCATED_SAMPLES,
                "t3_samples": T3_SAMPLES,
                "cpu_cap": CPU_CAP,
            }
        )

    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    candidate = load_routing_bundle(BUNDLE)[0]
    gate = dict(bundle["growth_gates"][0]) if bundle.get("growth_gates") else None
    graph = build_graph(candidate, gate, "CF1")
    unfreeze_outer(graph.subject)
    initial = copy.deepcopy(graph.subject.state_dict())

    worlds = [fixture.build_world(SEEDS[0], index) for index in TRAIN_INDICES[:10]]
    reference_grads = None

    if reuse:
        base_samples = [dict(row) for row in base_reuse[:BASE_SAMPLES]]
        truncated_samples = [dict(row) for row in trunc_reuse[:TRUNCATED_SAMPLES]]
        loss_diff = float(trunc_done["loss_abs_diff"])
        grad_diff = float(trunc_done["grad_max_abs_diff"])
    else:
        base_samples = []
        for sample in range(BASE_SAMPLES):
            guard("baseline")
            graph.subject.load_state_dict(copy.deepcopy(initial))
            optimizer = torch.optim.Adam(
                [p for p in graph.subject.parameters() if p.requires_grad], lr=LEARNING_RATE
            )
            started = time.process_time()
            world = fixture.build_world(SEEDS[0], TRAIN_INDICES[sample])
            data_cpu = time.process_time() - started
            row = update_timed(graph, optimizer, [world], False, keep_grads=sample == 0)
            row["data_prep_cpu"] = data_cpu
            if sample == 0:
                reference_grads = row["grads"]
            base_samples.append(row)
            append_progress(
                {
                    "stage": "baseline_sample",
                    "sample": sample,
                    "data_prep_cpu": round(data_cpu, 3),
                    "observe_cpu": round(row["observe_cpu"], 3),
                    "decode_cpu": round(row["decode_cpu"], 3),
                    "forward_cpu": round(row["forward_cpu"], 3),
                    "backward_cpu": round(row["backward_cpu"], 3),
                    "optimizer_cpu": round(row["optimizer_cpu"], 3),
                    "zero_cpu": round(row["zero_cpu"], 3),
                    "total_cpu": round(row["total_cpu"], 3),
                }
            )

        truncated_samples = []
        for sample in range(TRUNCATED_SAMPLES):
            guard("truncated")
            graph.subject.load_state_dict(copy.deepcopy(initial))
            optimizer = torch.optim.Adam(
                [p for p in graph.subject.parameters() if p.requires_grad], lr=LEARNING_RATE
            )
            row = update_timed(graph, optimizer, [worlds[sample]], True, keep_grads=sample == 0)
            truncated_samples.append(row)
            append_progress(
                {
                    "stage": "truncated_sample",
                    "sample": sample,
                    "forward_cpu": round(row["forward_cpu"], 3),
                    "backward_cpu": round(row["backward_cpu"], 3),
                    "total_cpu": round(row["total_cpu"], 3),
                }
            )

        loss_diff = abs(truncated_samples[0]["loss"] - base_samples[0]["loss"])
        grad_diff = float((truncated_samples[0]["grads"] - reference_grads).abs().max())
        append_progress(
            {
                "stage": "truncated_done",
                "per_world_update_cpu": round(
                    sum(row["total_cpu"] for row in truncated_samples) / len(truncated_samples), 3
                ),
                "loss_abs_diff": round(loss_diff, 8),
                "grad_max_abs_diff": round(grad_diff, 8),
            }
        )

    base_mean = sum(row["total_cpu"] + row["data_prep_cpu"] for row in base_samples) / len(
        base_samples
    )
    trunc_mean = sum(row["total_cpu"] for row in truncated_samples) / len(truncated_samples)
    append_progress(
        {
            "stage": "aggregates",
            "baseline_per_world_update_cpu": round(base_mean, 3),
            "truncated_per_world_update_cpu": round(trunc_mean, 3),
        }
    )

    dev_world = fixture.build_world(SEEDS[0], DEV_INDICES[0])
    graph.subject.load_state_dict(copy.deepcopy(initial))
    readout_turn = dev_world["response"]["turns"][READOUT_TURN_INDEX]
    readout_state, readout_released = build_state_upto(graph, dev_world, int(readout_turn["turn"]))
    no_grad_rows = []
    inference_rows = []
    for _ in range(T3_SAMPLES):
        guard("t3_ab")
        signature, cpu_seconds = readout_signature(
            graph, readout_state, readout_turn, readout_released, dev_world, False
        )
        no_grad_rows.append((signature, cpu_seconds))
        signature, cpu_seconds = readout_signature(
            graph, readout_state, readout_turn, readout_released, dev_world, True
        )
        inference_rows.append((signature, cpu_seconds))
    t3_loss_abs_diff = max(
        abs(no_grad[0][0] - inference[0][0])
        for no_grad, inference in zip(no_grad_rows, inference_rows, strict=True)
    )
    t3_logits_abs_diff = max(
        float(np.abs(no_grad[0][2] - inference[0][2]).max())
        for no_grad, inference in zip(no_grad_rows, inference_rows, strict=True)
    )
    t3_emitted_identical = all(
        no_grad[0][1] == inference[0][1]
        for no_grad, inference in zip(no_grad_rows, inference_rows, strict=True)
    )
    t3_no_grad_cpu = sum(row[1] for row in no_grad_rows) / len(no_grad_rows)
    t3_inference_cpu = sum(row[1] for row in inference_rows) / len(inference_rows)
    eval_world_cpu = t3_inference_cpu * TOTAL_TURNS
    append_progress(
        {
            "stage": "t3_ab",
            "samples": T3_SAMPLES,
            "no_grad_cpu_per_turn": round(t3_no_grad_cpu, 4),
            "inference_cpu_per_turn": round(t3_inference_cpu, 4),
            "speedup": round(t3_no_grad_cpu / t3_inference_cpu, 3),
            "loss_abs_diff": round(t3_loss_abs_diff, 8),
            "logits_max_abs_diff": round(t3_logits_abs_diff, 8),
            "emitted_identical": t3_emitted_identical,
            "derived_eval_world_cpu": round(eval_world_cpu, 3),
        }
    )

    profile = profile_micro(graph, worlds[0])
    append_progress({"stage": "profile", "turn": profile["turn"]})

    per_world = base_mean
    equiv_unit_cpu = 2 * per_world
    truncated_unit_cpu = 2 * trunc_mean
    forward_only_unit_cpu = 2 * (
        sum(row["forward_cpu"] for row in base_samples) / len(base_samples)
    )
    runs = len(ARMS) * len(SEEDS)

    def project(unit: float, eval_world_cpu: float) -> dict:
        train = unit * UPDATES_PER_SEED * runs
        dev = eval_world_cpu * len(DEV_INDICES) * runs
        probe = eval_world_cpu * EARLY_STOP_PROBES_PER_RUN * runs
        total = train + dev + probe
        return {
            "unit_cpu": round(unit, 3),
            "train_cpu": round(train, 1),
            "dev_eval_cpu": round(dev, 1),
            "early_stop_probe_cpu": round(probe, 1),
            "total_cpu": round(total, 1),
            "ratio_to_frozen_budget": round(total / FROZEN_BUDGET_CPU, 2),
            "within_ceiling": total <= MAGNITUDE_CEILING_CPU,
        }

    equiv_projection = project(equiv_unit_cpu, eval_world_cpu)
    truncated_projection = project(truncated_unit_cpu, eval_world_cpu)
    forward_floor_projection = project(forward_only_unit_cpu, eval_world_cpu)

    verdict = (
        "pricing_admissible_after_optimization"
        if equiv_projection["within_ceiling"]
        else "pricing_still_over_budget"
    )
    shares = phase_shares(
        {
            "data_prep_cpu": sum(row["data_prep_cpu"] for row in base_samples) / len(base_samples),
            "observe_cpu": sum(row["observe_cpu"] for row in base_samples) / len(base_samples),
            "decode_cpu": sum(row["decode_cpu"] for row in base_samples) / len(base_samples),
            "forward_cpu": sum(row["forward_cpu"] for row in base_samples) / len(base_samples),
            "backward_cpu": sum(row["backward_cpu"] for row in base_samples) / len(base_samples),
            "optimizer_cpu": sum(row["optimizer_cpu"] for row in base_samples) / len(base_samples),
            "zero_cpu": sum(row.get("zero_cpu", 0.0) for row in base_samples) / len(base_samples),
            "total_cpu": per_world
            - sum(row["data_prep_cpu"] for row in base_samples) / len(base_samples),
        }
    )
    ranked = sorted(shares["shares"].items(), key=lambda item: item[1], reverse=True)
    top3 = [name for name, share in ranked[:3]]
    prior_pass_cpu = (
        sum(row["total_cpu"] + row.get("data_prep_cpu", 0.0) for row in base_samples)
        + sum(row["total_cpu"] for row in truncated_samples)
        if reuse
        else 0.0
    )

    payload = {
        "schema": "taiji-rule-learning-20-4a-unit-cost-v1",
        "evidence": "20_4a",
        "verdict": verdict,
        "detail": (
            f"best equivalent-tier unit is {equiv_unit_cpu:.1f} CPU per batch-2 update "
            f"(projected {equiv_projection['total_cpu']:.0f} CPU, "
            f"{equiv_projection['ratio_to_frozen_budget']:.2f}x the frozen budget); "
            "the forward pass alone already projects above the 2x ceiling"
            if verdict == "pricing_still_over_budget"
            else f"equivalent-tier unit is {equiv_unit_cpu:.1f} CPU and the projection fits"
        ),
        "setup": {
            "threads": THREADS,
            "dtype": "float64",
            "deterministic": True,
            "addressed_store_enabled": ADDRESSED_STORE,
            "batch_worlds": BATCH,
            "samples_are_single_world": True,
            "batch2_derivation": "two worlds run sequentially with no shared kernel; unit = 2x",
        },
        "breakdown": {
            "per_world_update_cpu": round(per_world, 3),
            "batch2_unit_cpu": round(equiv_unit_cpu, 3),
            "samples": len(base_samples),
            "sample_cpu": [round(row["total_cpu"], 3) for row in base_samples],
            "phases": shares,
            "top3": top3,
            "per_turn_cpu": round(per_world / TOTAL_TURNS, 4),
            "per_answer_step_cpu": round(per_world / TOTAL_TURNS / ANSWER_BYTES, 6),
        },
        "profiler": profile,
        "trials": [
            {
                "id": "T1_batched_worlds",
                "status": "not_measured",
                "reason": (
                    "merging two worlds into one batched forward requires a batched "
                    "LivingCore.step/observe; the frozen R021 path takes a single event "
                    "tensor, so this is a graph-level change that needs owner authorization"
                ),
            },
            {
                "id": "T2_reuse_buffers",
                "status": "not_measured",
                "reason": (
                    "source_view / goal_vector / align_positions are rebuilt per byte step "
                    "inside the graph; hoisting them is a graph edit, not a harness edit"
                ),
                "profiler_pointer": profile["forward_top"][:3],
            },
            {
                "id": "T3_readings_without_grad",
                "status": "measured",
                "samples": len(no_grad_rows),
                "no_grad_cpu_per_turn": round(t3_no_grad_cpu, 4),
                "inference_cpu_per_turn": round(t3_inference_cpu, 4),
                "speedup": round(t3_no_grad_cpu / t3_inference_cpu, 3),
                "loss_abs_diff": round(t3_loss_abs_diff, 8),
                "logits_max_abs_diff": round(t3_logits_abs_diff, 8),
                "emitted_identical": t3_emitted_identical,
                "equivalent": t3_loss_abs_diff == 0.0
                and t3_logits_abs_diff == 0.0
                and t3_emitted_identical,
                "derived_eval_world_cpu": round(eval_world_cpu, 3),
                "note": (
                    "single reading-path turn measured 5 times per context; the dev/probe "
                    "world cost in the projection is this per-turn cost x 51 turns"
                ),
            },
            {
                "id": "T4_backward_window",
                "status": "measured",
                "samples": len(truncated_samples),
                "before_cpu": round(per_world, 3),
                "after_cpu": round(trunc_mean, 3),
                "speedup": round(per_world / trunc_mean, 3),
                "equivalent": grad_diff <= EQUIVALENCE_TOLERANCE,
                "loss_abs_diff": round(loss_diff, 8),
                "grad_max_abs_diff": round(grad_diff, 8),
                "note": (
                    "detaching at every turn boundary keeps only the current turn's credit; "
                    "it is a relaxed tier and must not be mixed with the equivalent tier"
                ),
            },
        ],
        "equivalence": {
            "tolerance": EQUIVALENCE_TOLERANCE,
            "reference": "full 51-turn window, baseline sample 0",
            "truncated_loss_abs_diff": round(loss_diff, 8),
            "truncated_grad_max_abs_diff": round(grad_diff, 8),
            "t3_loss_abs_diff": round(t3_loss_abs_diff, 8),
            "t3_logits_max_abs_diff": round(t3_logits_abs_diff, 8),
            "verdict": "not_equivalent" if grad_diff > EQUIVALENCE_TOLERANCE else "equivalent",
            "equivalent_tier_trials": ["T3_readings_without_grad"],
            "relaxed_tier_trials": ["T4_backward_window"],
        },
        "projection": {
            "equivalent_tier": equiv_projection,
            "relaxed_tier_truncated_window": truncated_projection,
            "graph_level_floor_forward_only": forward_floor_projection,
            "frozen_budget_cpu": FROZEN_BUDGET_CPU,
            "magnitude_ceiling_cpu": MAGNITUDE_CEILING_CPU,
            "runs": runs,
            "updates_per_run": UPDATES_PER_SEED,
        },
        "options_for_owner": [
            {
                "id": "O1_truncated_backward",
                "ceiling_cpu": truncated_projection["total_cpu"],
                "what_it_gives_up": (
                    "credit that spans turn boundaries: a correction surface can only "
                    "move the parameters through its own turn, not through the 30 turns "
                    "that built the session state"
                ),
            },
            {
                "id": "O2_graph_level_batch_or_fuse",
                "ceiling_cpu": forward_floor_projection["total_cpu"],
                "what_it_gives_up": (
                    "nothing semantic, but it needs a batched/fused execution path inside "
                    "R021, which is out of scope without owner authorization; the floor "
                    "shown is forward-only at today's per-step cost"
                ),
            },
            {
                "id": "O3_replay_fewer_turns",
                "ceiling_cpu": round(forward_floor_projection["total_cpu"] * 22 / TOTAL_TURNS, 1),
                "what_it_gives_up": (
                    "the pre-feedback context that a later turn depends on; the 22 "
                    "correction surfaces keep their loss but lose the session history "
                    "needed to answer them"
                ),
            },
        ],
        "training_started": False,
        "formal_training_steps": 0,
        "cost": {
            "cpu_seconds": round(cost()[0], 3),
            "wall_seconds": round(cost()[1], 3),
            "earlier_measurement_pass_cpu_seconds": round(prior_pass_cpu, 3),
            "total_cpu_seconds": round(cost()[0] + prior_pass_cpu, 3),
            "cap_cpu_seconds": CPU_CAP,
            "note": (
                "the first measurement pass produced the 10 calibration samples but stopped "
                "before the reading-path A/B; this pass reuses those samples and only adds "
                "the T3 A/B and the micro profile"
                if reuse
                else "samples and A/B were measured in this pass"
            ),
        },
        "script": {
            "runner_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "submitted_runner_sha256": None,
        },
        "invariants": {
            "confirm_read": False,
            "graph_modified": False,
            "thresholds_changed": False,
            "freeze_written": False,
            "calibration_updates_executed_this_pass": (
                0 if reuse else BASE_SAMPLES + TRUNCATED_SAMPLES
            ),
            "calibration_updates_total": BASE_SAMPLES + TRUNCATED_SAMPLES,
            "threads": THREADS,
        },
    }
    write_json(REPORT, payload)
    write_json(
        MANIFEST,
        {
            "schema": "taiji-rule-learning-20-4a-manifest-v1",
            "evidence": "20_4a",
            "verdict": verdict,
            "report_path": str(REPORT.relative_to(ROOT)),
            "report_sha256": sha256_bytes(REPORT.read_bytes()),
            "runner_sha256": payload["script"]["runner_sha256"],
            "submitted_runner_sha256": None,
            "equivalent_unit_cpu": round(equiv_unit_cpu, 3),
            "projection_cpu": equiv_projection["total_cpu"],
            "training_started": False,
        },
    )
    print(json.dumps({"verdict": verdict, "detail": payload["detail"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BudgetStop as stop:
        cpu, wall = cost()
        print(
            json.dumps(
                {
                    "verdict": "budget_stop",
                    "stage": str(stop),
                    "cpu_seconds": round(cpu, 3),
                    "wall_seconds": round(wall, 3),
                },
                ensure_ascii=False,
            )
        )
        raise SystemExit(0) from None
