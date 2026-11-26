"""17.3 pricing for three-arm A/B/C and two-arm A/B, with position supervision.

Component pricing on the real 17.0 graph: forward-only, BPTT (forward+backward
minus forward), and one measured joint Adam step.  No task training: the only
parameter step is the engineering Adam used to price the optimizer overhead.

`--copy-span` re-measures only the copy stage with five repeats (0 Adam), to
separate the position-supervision increment from the run-to-run timing noise of
the first pass; it writes its own evidence file and does not touch the price
report.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji.binding_copy import initial_binding_state, prepare  # noqa: E402
from taiji.binding_copy_scoring import score_generation  # noqa: E402
from taiji.binding_select import load_bundle  # noqa: E402
from taiji.selection_routing import (  # noqa: E402
    OrderedSelectionHead,
    RoutingConfig,
    SpanPointerHead,
    build_context,
    route_labels,
    routing_generate,
    routing_loss,
    selection_ce,
    span_loss,
    teacher_span,
)

BUNDLE = Path("output/taiji_core_binding_select_16_1_compare/bundles/28031-A-locked-step300.pt")
COMPARE_SCRIPT = Path("scripts/training/verify_taiji_binding_select_16_1_compare.py")
OUT = Path("reports/taiji_selection_routing_17_3_price_20261103.json")
COPY_SPAN_OUT = Path("reports/taiji_selection_routing_17_3_copy_span_20261103.json")
INHERITED_BUNDLE_MIB = 27.2307

BATCH = 8
LR = 1e-3
SPAN_LAMBDA = 0.5
SEEDS = 2
DEV = 240
ENDPOINTS = 2
STAGE_UPDATES = (100, 100, 100)
OVERRUN = 0.10


def load_fixtures():
    spec = importlib.util.spec_from_file_location("cmp17p", ROOT / COMPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train, dev, _select, _data = module.load_data()
    return train[:BATCH], dev[:BATCH]


def timed(function):
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    value = function()
    return value, time.process_time() - cpu_start, time.perf_counter() - wall_start


def fresh():
    candidate = load_bundle(ROOT / BUNDLE, torch.float64)[0]
    head = OrderedSelectionHead(RoutingConfig(), seed=1731)
    span = SpanPointerHead(RoutingConfig(), seed=1732)
    return candidate, head, span


def set_trainable(candidate, head, span, stage: str) -> list[torch.nn.Parameter]:
    for parameter in candidate.binding.parameters():
        parameter.requires_grad = stage in ("copy", "joint")
    for parameter in candidate.head.parameters():
        parameter.requires_grad = False
    for parameter in head.parameters():
        parameter.requires_grad = stage in ("selection", "joint")
    for parameter in span.parameters():
        parameter.requires_grad = True
    params: list[torch.nn.Parameter] = []
    for module in (candidate.binding, head, span):
        params.extend(p for p in module.parameters() if p.requires_grad)
    return params


def zero_grads(candidate, head, span) -> None:
    for module in (candidate.binding, candidate.head, head, span):
        for parameter in module.parameters():
            parameter.grad = None


def fixture_loss(candidate, fixture, head, span, arm: str, stage: str, span_on: bool):
    binding = candidate.binding
    prepared = prepare(
        binding,
        initial_binding_state(binding, f"17-price-{fixture['index']}"),
        fixture["goal"].encode("utf-8"),
        fixture["records"],
    )
    context = build_context(binding, fixture["goal"].encode("utf-8"), tuple(fixture["record_keys"]))
    routes = route_labels(fixture)
    if stage == "selection":
        output = selection_ce(
            binding,
            prepared,
            tuple(fixture["target_prefix"]),
            tuple(fixture["target_bytes"]),
            routes[0],
            routes[1],
            context,
            head,
        )
        loss = output["loss"]
    else:
        output = routing_loss(
            binding,
            prepared,
            tuple(fixture["target_prefix"]),
            tuple(fixture["target_bytes"]),
            routes[0],
            routes[1],
            context,
            head,
            arm=arm,
        )
        loss = output["nll"]
        if stage == "joint":
            loss = loss + output["ce"]
    if span_on:
        object_index = int(fixture["selection_label"])
        object_record = fixture["raw_example"]["records"][object_index]["raw"].encode("utf-8")
        start, end = teacher_span(object_record, fixture["object_name"].encode("utf-8"))
        units = prepared.base.view.units[object_index]
        loss = loss + SPAN_LAMBDA * span_loss(span, units, start, end)["loss"]
    return loss


def forward_only(candidate, fixtures, head, span, stage: str, arm: str, span_on: bool) -> float:
    total = 0.0
    with torch.no_grad():
        for fixture in fixtures:
            total += float(
                fixture_loss(candidate, fixture, head, span, arm, stage, span_on).detach()
            )
    return total


def backward_only(candidate, fixtures, head, span, stage: str, arm: str, span_on: bool) -> float:
    total = 0.0
    for fixture in fixtures:
        loss = fixture_loss(candidate, fixture, head, span, arm, stage, span_on)
        loss.backward()
        total += float(loss.detach())
    return total


def measure_components(
    candidate, fixtures, head, span, stage: str, arm: str, span_on: bool
) -> dict:
    forward_rows = []
    for _ in range(2):
        forward_rows.append(
            timed(lambda: forward_only(candidate, fixtures, head, span, stage, arm, span_on))[1]
        )
    backward_rows = []
    for _ in range(2):
        zero_grads(candidate, head, span)
        backward_rows.append(
            timed(lambda: backward_only(candidate, fixtures, head, span, stage, arm, span_on))[1]
        )
    forward_cpu = statistics.median(forward_rows)
    forward_backward_cpu = statistics.median(backward_rows)
    return {
        "stage": stage,
        "arm": arm,
        "span": span_on,
        "forward_cpu": forward_cpu,
        "forward_backward_cpu": forward_backward_cpu,
        "bptt_cpu": forward_backward_cpu - forward_cpu,
        "forward_rows": forward_rows,
        "backward_rows": backward_rows,
    }


def measure_adam() -> dict:
    candidate, head, span = fresh()
    params = set_trainable(candidate, head, span, "joint")
    optimizer = torch.optim.Adam(params, lr=LR)
    train_batch, _dev = load_fixtures_from_module()
    for fixture in train_batch:
        loss = fixture_loss(candidate, fixture, head, span, "A_route_lock", "joint", True)
        loss.backward()
    _value, cpu, wall = timed(lambda: optimizer.step())
    return {
        "cpu": cpu,
        "wall": wall,
        "parameter_groups": len(params),
        "note": "one joint Adam step including first-step state allocation",
    }


_MODULE_CACHE = {}


def load_fixtures_from_module():
    if "data" not in _MODULE_CACHE:
        _MODULE_CACHE["data"] = load_fixtures()
    return _MODULE_CACHE["data"]


def measure_forward_selection(candidate, fixtures, head, span) -> dict:
    binding = candidate.binding

    def run():
        count = 0
        with torch.no_grad():
            for fixture in fixtures:
                prepared = prepare(
                    binding,
                    initial_binding_state(binding, f"17-fwd-{fixture['index']}"),
                    fixture["goal"].encode("utf-8"),
                    fixture["records"],
                )
                context = build_context(
                    binding, fixture["goal"].encode("utf-8"), tuple(fixture["record_keys"])
                )
                routes = route_labels(fixture)
                selection_ce(
                    binding,
                    prepared,
                    tuple(fixture["target_prefix"]),
                    tuple(fixture["target_bytes"]),
                    routes[0],
                    routes[1],
                    context,
                    head,
                )
                count += 1
        return count

    value, cpu, wall = timed(run)
    return {"examples": value, "cpu": cpu, "wall": wall, "cpu_per_example": cpu / max(value, 1)}


def measure_generation(candidate, fixtures, head, arm: str) -> dict:
    binding = candidate.binding
    rows = []
    for fixture in fixtures:
        prepared = prepare(
            binding,
            initial_binding_state(binding, f"17-gen-{fixture['index']}"),
            fixture["goal"].encode("utf-8"),
            fixture["records"],
        )
        context = build_context(
            binding, fixture["goal"].encode("utf-8"), tuple(fixture["record_keys"])
        )
        start = time.process_time()
        with torch.no_grad():
            routing_generate(binding, prepared, context, head, arm=arm)
        rows.append(time.process_time() - start)
    return {
        "examples": len(rows),
        "cpu": sum(rows),
        "cpu_per_example": statistics.median(rows),
        "rows": rows,
    }


def measure_evaluation(candidate, fixtures, head, arm: str) -> dict:
    binding = candidate.binding
    rows = []
    for fixture in fixtures:
        prepared = prepare(
            binding,
            initial_binding_state(binding, f"17-eval-{fixture['index']}"),
            fixture["goal"].encode("utf-8"),
            fixture["records"],
        )
        context = build_context(
            binding, fixture["goal"].encode("utf-8"), tuple(fixture["record_keys"])
        )
        with torch.no_grad():
            generated = routing_generate(binding, prepared, context, head, arm=arm)
        start = time.process_time()
        score_generation(
            generated["raw_bytes"],
            tuple(fixture["target_prefix"]),
            generated["terminated"],
        )
        rows.append(time.process_time() - start)
    return {"examples": len(rows), "cpu": sum(rows), "cpu_per_example": statistics.median(rows)}


def measure_restore() -> dict:
    _value, cpu, wall = timed(lambda: load_bundle(ROOT / BUNDLE, torch.float64))
    return {"cpu": cpu, "wall": wall}


def bundle_estimate() -> dict:
    new_head = 37120
    old_head = 25602
    span = 258
    delta_bytes = (new_head - old_head + span) * 8 * 3
    replaced_mib = INHERITED_BUNDLE_MIB + delta_bytes / (1024 * 1024)
    kept_mib = INHERITED_BUNDLE_MIB + (new_head + span) * 8 * 3 / (1024 * 1024)
    return {
        "inherited_mib": INHERITED_BUNDLE_MIB,
        "delta_bytes_replace_old_head": delta_bytes,
        "mib_replace_old_head": replaced_mib,
        "mib_keep_old_head": kept_mib,
        "note": "3x = parameter + two Adam moments, float64; inherited bytes from the frozen 16.1 bundle",
    }


def train_cpu(stage: str, arm: str, span_on: bool, components: dict, adam_cpu: float) -> float:
    key = (stage, arm, span_on)
    return components[key]["forward_cpu"] + components[key]["bptt_cpu"] + adam_cpu


def scope(
    arms: tuple[str, ...],
    components: dict,
    adam_cpu: float,
    gen: dict,
    evaluation: dict,
    select_forward: dict,
    restore: dict,
    bundle: dict,
) -> dict:
    n_arms = len(arms)
    runs = SEEDS * n_arms
    updates = runs * sum(STAGE_UPDATES)
    per_arm_train = {}
    for arm in arms:
        per_arm_train[arm] = (
            STAGE_UPDATES[0] * train_cpu("selection", "B_stop_lock", True, components, adam_cpu)
            + STAGE_UPDATES[1] * train_cpu("copy", "B_stop_lock", True, components, adam_cpu)
            + STAGE_UPDATES[2] * train_cpu("joint", arm, True, components, adam_cpu)
        )
    training = SEEDS * sum(per_arm_train.values())
    generations = runs * DEV * ENDPOINTS
    generation = SEEDS * sum(DEV * ENDPOINTS * gen[arm]["cpu_per_example"] for arm in arms)
    scoring = generations * evaluation["cpu_per_example"]
    probe = generations * select_forward["cpu_per_example"]
    restores = runs * ENDPOINTS
    restore_cpu = restores * restore["cpu"]
    subtotal = training + generation + scoring + probe + restore_cpu
    total = subtotal * (1 + OVERRUN) + 60.0
    bundles = runs * ENDPOINTS
    bundle_mib = bundles * bundle["mib_replace_old_head"]
    step150_mib = runs * bundle["mib_replace_old_head"]
    return {
        "arms": list(arms),
        "runs": runs,
        "updates": updates,
        "adam_formal": updates,
        "training_cpu": training,
        "training_cpu_per_run": training / runs,
        "per_arm_seed_training_cpu": per_arm_train,
        "generation_cpu": generation,
        "evaluation_cpu": scoring,
        "selection_probe_cpu": probe,
        "restore_cpu": restore_cpu,
        "subtotal_cpu": subtotal,
        "total_cpu": total,
        "total_wall_estimate": total * 1.15,
        "product_bundles": bundles,
        "bundle_mib_final": bundle_mib,
        "bundle_mib_step150_checkpoint": step150_mib,
        "peak_product_mib": bundle_mib + step150_mib,
    }


def step150_block(
    arms: tuple[str, ...],
    components: dict,
    adam_cpu: float,
    gen: dict,
    evaluation: dict,
    select_forward: dict,
    restore: dict,
    bundle: dict,
) -> dict:
    runs = SEEDS * len(arms)
    selection = train_cpu("selection", "B_stop_lock", True, components, adam_cpu)
    copy = train_cpu("copy", "B_stop_lock", True, components, adam_cpu)
    joint = {arm: train_cpu("joint", arm, True, components, adam_cpu) for arm in arms}
    per_endpoint = DEV * (
        statistics.mean(gen[arm]["cpu_per_example"] for arm in arms)
        + evaluation["cpu_per_example"]
        + select_forward["cpu_per_example"]
    )
    span_gate = 40 * select_forward["cpu_per_example"]
    restore_cpu = restore["cpu"]
    seg1_training = 100 * selection + 50 * copy
    seg2_training = 50 * copy + 100 * statistics.mean(joint.values())
    seg1_run = seg1_training + 2 * per_endpoint + span_gate + 2 * restore_cpu
    seg2_run = seg2_training + per_endpoint + span_gate + restore_cpu
    stop_100_run = 100 * selection + per_endpoint + restore_cpu
    stop_100_total = runs * stop_100_run
    seg1_total = runs * seg1_run
    seg2_total = runs * seg2_run
    full_total = (seg1_total + seg2_total) * (1 + OVERRUN) + 60.0
    return {
        "runs": runs,
        "segment1": {
            "updates": 150,
            "stage_split": {"selection": 100, "copy": 50, "joint": 0},
            "per_run_cpu": seg1_run,
            "total_cpu": seg1_total,
            "exit": "step150 readiness gate: selection teacher-prefix top-1 >= 0.90, R1 witnessed, gold-span object_exact >= 0.8 after position supervision; fail => early_gate_failed (no task claim)",
        },
        "segment2": {
            "updates": 150,
            "stage_split": {"selection": 0, "copy": 50, "joint": 100},
            "per_run_cpu": seg2_run,
            "total_cpu": seg2_total,
            "exit": "full step300 frozen task gates: object_exact >= 0.8 and A-C selection delta >= 0.10",
        },
        "stop_points": {
            "after_100_selection_only": {
                "per_run_cpu": stop_100_run,
                "total_cpu": stop_100_total,
                "share_of_full": stop_100_total / full_total,
            },
            "after_150_segment1": {
                "per_run_cpu": seg1_run,
                "total_cpu": seg1_total,
                "share_of_full": seg1_total / full_total,
            },
            "after_300_full": {
                "per_run_cpu": seg1_run + seg2_run,
                "total_cpu": full_total,
                "share_of_full": 1.0,
            },
        },
        "full_two_segment_cpu": full_total,
        "step150_checkpoint_mib": runs * bundle["mib_replace_old_head"],
        "dilution_note": "The final task gate stays at step300 and is the only task verdict. The step150 gate is a readiness/budget gate: a fail is labelled early_gate_failed, never task_failed, because the span head has only 50 updates by step150. Treating step150 as success or failure evidence would dilute the frozen judgment.",
    }


def remeasure_copy_span(repeats: int = 5) -> dict:
    """Copy-stage with/without span, five interleaved repeats, zero Adam."""
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    train_batch, _dev_batch = load_fixtures()
    blocks = {}
    for span_on in (True, False):
        candidate, head, span = fresh()
        forward_rows = []
        backward_rows = []
        forward_call = functools.partial(
            forward_only, candidate, train_batch, head, span, "copy", "B_stop_lock", span_on
        )
        backward_call = functools.partial(
            backward_only, candidate, train_batch, head, span, "copy", "B_stop_lock", span_on
        )
        for _ in range(repeats):
            forward_rows.append(timed(forward_call)[1])
            zero_grads(candidate, head, span)
            backward_rows.append(timed(backward_call)[1])
        total_rows = [
            forward + backward
            for forward, backward in zip(forward_rows, backward_rows, strict=True)
        ]
        blocks["with_span" if span_on else "without_span"] = {
            "forward_rows": forward_rows,
            "forward_backward_rows": backward_rows,
            "median_forward_cpu": statistics.median(forward_rows),
            "median_forward_backward_cpu": statistics.median(backward_rows),
            "median_total_cpu": statistics.median(total_rows),
        }
    increment = blocks["with_span"]["median_total_cpu"] - blocks["without_span"]["median_total_cpu"]
    all_totals = (
        blocks["with_span"]["forward_backward_rows"]
        + blocks["without_span"]["forward_backward_rows"]
    )
    noise_band = max(all_totals) - min(all_totals)
    report = {
        "schema": "taiji-selection-routing-17-3-copy-span-v1",
        "step": "N17.3.price.copy_span",
        "date": "2026-11-03",
        "dtype": "float64",
        "threads": 1,
        "batch": BATCH,
        "stage": "copy",
        "arm": "B_stop_lock",
        "repeats": repeats,
        "adam_steps": 0,
        "blocks": blocks,
        "copy_span_increment_cpu": increment,
        "row_noise_band_cpu": noise_band,
        "resolved": bool(abs(increment) >= noise_band),
        "note": (
            "The first price pass reported -0.2578 CPU for the copy span increment, which is "
            "smaller than the between-row spread; this five-repeat interleaved re-measurement "
            "is the value used in the 17.3 report."
        ),
        "timing": {
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.perf_counter() - start_wall,
        },
    }
    COPY_SPAN_OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"out": str(COPY_SPAN_OUT), "increment": increment, "noise_band": noise_band},
            ensure_ascii=False,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy-span", action="store_true")
    arguments = parser.parse_args()
    if arguments.copy_span:
        remeasure_copy_span()
        return 0
    train_batch, dev_batch = load_fixtures()
    start_cpu = time.process_time()
    start_wall = time.perf_counter()

    components: dict[tuple[str, str, bool], dict] = {}
    candidate, head, span = fresh()
    components[("selection", "B_stop_lock", True)] = measure_components(
        candidate, train_batch, head, span, "selection", "B_stop_lock", True
    )
    candidate, head, span = fresh()
    components[("selection", "B_stop_lock", False)] = measure_components(
        candidate, train_batch, head, span, "selection", "B_stop_lock", False
    )
    candidate, head, span = fresh()
    components[("copy", "B_stop_lock", True)] = measure_components(
        candidate, train_batch, head, span, "copy", "B_stop_lock", True
    )
    candidate, head, span = fresh()
    components[("copy", "B_stop_lock", False)] = measure_components(
        candidate, train_batch, head, span, "copy", "B_stop_lock", False
    )
    candidate, head, span = fresh()
    components[("joint", "A_route_lock", True)] = measure_components(
        candidate, train_batch, head, span, "joint", "A_route_lock", True
    )
    candidate, head, span = fresh()
    components[("joint", "A_route_lock", False)] = measure_components(
        candidate, train_batch, head, span, "joint", "A_route_lock", False
    )
    candidate, head, span = fresh()
    components[("joint", "B_stop_lock", True)] = measure_components(
        candidate, train_batch, head, span, "joint", "B_stop_lock", True
    )
    candidate, head, span = fresh()
    components[("joint", "C_unlocked", True)] = measure_components(
        candidate, train_batch, head, span, "joint", "C_unlocked", True
    )

    adam = measure_adam()
    adam_cpu = adam["cpu"]

    candidate, head, span = fresh()
    select_forward = measure_forward_selection(candidate, dev_batch, head, span)
    candidate, head, span = fresh()
    gen_a = measure_generation(candidate, dev_batch, head, "A_route_lock")
    candidate, head, span = fresh()
    gen_c = measure_generation(candidate, dev_batch, head, "C_unlocked")
    candidate, head, span = fresh()
    evaluation = measure_evaluation(candidate, dev_batch, head, "A_route_lock")
    restore = measure_restore()
    bundle = bundle_estimate()

    prices = {
        "selection_with": train_cpu("selection", "B_stop_lock", True, components, adam_cpu),
        "selection_without": train_cpu("selection", "B_stop_lock", False, components, adam_cpu),
        "copy_with": train_cpu("copy", "B_stop_lock", True, components, adam_cpu),
        "copy_without": train_cpu("copy", "B_stop_lock", False, components, adam_cpu),
        "joint_A_route_lock_with": train_cpu("joint", "A_route_lock", True, components, adam_cpu),
        "joint_A_route_lock_without": train_cpu(
            "joint", "A_route_lock", False, components, adam_cpu
        ),
        "joint_B_stop_lock_with": train_cpu("joint", "B_stop_lock", True, components, adam_cpu),
        "joint_C_unlocked_with": train_cpu("joint", "C_unlocked", True, components, adam_cpu),
        "selection_span_increment": train_cpu(
            "selection", "B_stop_lock", True, components, adam_cpu
        )
        - train_cpu("selection", "B_stop_lock", False, components, adam_cpu),
        "copy_span_increment": train_cpu("copy", "B_stop_lock", True, components, adam_cpu)
        - train_cpu("copy", "B_stop_lock", False, components, adam_cpu),
        "joint_span_increment_A": train_cpu("joint", "A_route_lock", True, components, adam_cpu)
        - train_cpu("joint", "A_route_lock", False, components, adam_cpu),
        "joint_B_span_increment_estimate": train_cpu(
            "joint", "A_route_lock", True, components, adam_cpu
        )
        - train_cpu("joint", "A_route_lock", False, components, adam_cpu),
        "joint_C_span_increment_estimate": train_cpu(
            "joint", "A_route_lock", True, components, adam_cpu
        )
        - train_cpu("joint", "A_route_lock", False, components, adam_cpu),
    }
    gen_by_arm = {"A_route_lock": gen_a, "B_stop_lock": gen_a, "C_unlocked": gen_c}
    three_arm = scope(
        ("A_route_lock", "B_stop_lock", "C_unlocked"),
        components,
        adam_cpu,
        gen_by_arm,
        evaluation,
        select_forward,
        restore,
        bundle,
    )
    two_arm = scope(
        ("A_route_lock", "B_stop_lock"),
        components,
        adam_cpu,
        gen_by_arm,
        evaluation,
        select_forward,
        restore,
        bundle,
    )
    step150 = step150_block(
        ("A_route_lock", "B_stop_lock", "C_unlocked"),
        components,
        adam_cpu,
        gen_by_arm,
        evaluation,
        select_forward,
        restore,
        bundle,
    )
    step150_two = step150_block(
        ("A_route_lock", "B_stop_lock"),
        components,
        adam_cpu,
        gen_by_arm,
        evaluation,
        select_forward,
        restore,
        bundle,
    )

    report = {
        "schema": "taiji-selection-routing-17-3-price-v1",
        "step": "N17.3.price",
        "date": "2026-11-03",
        "dtype": "float64",
        "threads": 1,
        "batch": BATCH,
        "lr": LR,
        "span_lambda": SPAN_LAMBDA,
        "engineering_adam": 1,
        "aborted_run_adam": 12,
        "formal_updates_three_arm": three_arm["updates"],
        "formal_updates_two_arm": two_arm["updates"],
        "components": {
            f"{key[0]}|{key[1]}|span={key[2]}": value for key, value in components.items()
        },
        "adam": adam,
        "prices": prices,
        "generation": {"A_route_lock": gen_a, "C_unlocked": gen_c},
        "evaluation": evaluation,
        "selection_forward": select_forward,
        "restore": restore,
        "bundle": bundle,
        "three_arm": three_arm,
        "two_arm": two_arm,
        "step150_three_arm": step150,
        "step150_two_arm": step150_two,
        "cost_ratios": {
            "three_over_two_updates": three_arm["updates"] / two_arm["updates"],
            "three_over_two_cpu": three_arm["total_cpu"] / two_arm["total_cpu"],
            "three_over_two_products": three_arm["peak_product_mib"] / two_arm["peak_product_mib"],
        },
        "position_supervision": {
            "enabled_from": "17.5 stage 1 (selection), because 0b T3 measured object_exact = 0/40",
            "reprice_needed": False,
            "reason": "the with-span rows are already measured; enabling at stage 1 uses those rows, changes no parameter count and no dosage",
            "parameter_count": 258,
            "bottleneck_parameter_count": 4194,
        },
        "timing": {
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.perf_counter() - start_wall,
        },
        "limits": {
            "task_training_updates": 0,
            "engineering_adam": 1,
            "aborted_run_adam": 12,
            "no_new_seeds": True,
            "no_dose_change": True,
            "no_criterion_change": True,
            "confirm_reads": 0,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(OUT),
                "prices": {key: round(value, 4) for key, value in prices.items()},
                "components": {
                    str(key): {
                        "forward": round(value["forward_cpu"], 4),
                        "bptt": round(value["bptt_cpu"], 4),
                    }
                    for key, value in components.items()
                    if value["span"]
                },
                "three_arm_cpu": round(three_arm["total_cpu"], 2),
                "two_arm_cpu": round(two_arm["total_cpu"], 2),
                "step150_three": round(step150["segment1"]["total_cpu"], 2),
                "cpu": round(report["timing"]["cpu_seconds"], 2),
                "wall": round(report["timing"]["wall_seconds"], 2),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
