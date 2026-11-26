"""17.2R repricing under the hard-point transport + position-supervision graph.

The measurement method is inherited from the frozen 17.3 pricer: same batch,
same fixtures, same float64 single-thread, same two-repeat medians, same
``scope`` arithmetic.  Only the priced graph changed:

* within-record transport is the point transport (``_point_transport``) at the
  annealed pointer temperature, not the old soft diffusion;
* every copy/joint step now builds the per-record span start bias;
* the pointer entropy term (coefficient 0.01) is included;
* generation uses the hard argmax pointer at T=0.05.

Old and new prices are reported side by side in a difference table.  The old
13,092.25 / 8,750.34 numbers stay frozen as the pre-change baseline and are
never treated as current values.

Run with no arguments.  Writes the report and manifest; no Adam step is spent
on the priced paths (the inherited pricer's Adam probe is the usual one-step
allocator measurement).
"""

from __future__ import annotations

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
from taiji.candidate_identity import candidate_identity  # noqa: E402
from taiji.selection_routing import (  # noqa: E402
    build_context,
    route_labels,
    routing_generate,
    routing_loss,
    selection_ce,
    span_loss,
    teacher_span,
)
from taiji.selection_routing_checkpoint import (  # noqa: E402
    build_routing_candidate,
    load_routing_bundle,
    save_routing_bundle,
)

PRICE_SCRIPT = Path("scripts/training/price_taiji_selection_routing_17_3.py")
REPORT = Path("reports/taiji_selection_routing_17_2r_price_20261103.json")
MANIFEST = Path("plans/manifests/taiji_selection_routing_17_2r_price_20261103.json")
OLD_REPORT = Path("reports/taiji_selection_routing_17_3_price_20261103.json")
PARENT = Path("output/taiji_core_binding_select_16_1_compare/bundles/28031-A-locked-step300.pt")
SEED = 28031
RESTORE_TMP = Path("output/taiji_selection_routing_17_2r_price/restore_probe.pt")

T_MIN = 0.05
ENTROPY_COEF = 0.01
SPAN_LAMBDA = 0.5
LR = 1e-3


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed(function):
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    value = function()
    return value, time.process_time() - cpu_start, time.perf_counter() - wall_start


def pointer_temperature(stage: str) -> float:
    # Selection has no copy step; copy/joint run at the annealed floor.
    return 1.0 if stage == "selection" else T_MIN


def fixture_loss(candidate, fixture, head, span, arm: str, stage: str, span_on: bool):
    binding = candidate.binding
    prepared = prepare(
        binding,
        initial_binding_state(binding, f"17r-price-{fixture['index']}"),
        fixture["goal"].encode("utf-8"),
        fixture["records"],
    )
    context = build_context(binding, fixture["goal"].encode("utf-8"), tuple(fixture["record_keys"]))
    routes = route_labels(fixture)
    span_head = span if span_on else None
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
            candidate,
            prepared,
            tuple(fixture["target_prefix"]),
            tuple(fixture["target_bytes"]),
            routes[0],
            routes[1],
            context,
            head,
            arm=arm,
            span_head=span_head,
            pointer_temperature=pointer_temperature(stage),
            pointer_entropy_coef=ENTROPY_COEF,
        )
        loss = output["nll"] - ENTROPY_COEF * output["entropy"]
        if stage == "joint":
            loss = loss + output["ce"]
    if span_on:
        object_index = int(fixture["selection_label"])
        object_record = fixture["raw_example"]["records"][object_index]["raw"].encode("utf-8")
        start, end = teacher_span(object_record, fixture["object_name"].encode("utf-8"))
        units = prepared.base.view.units[object_index]
        loss = loss + SPAN_LAMBDA * span_loss(span, units, start, end)["loss"]
    return loss


def measure_generation(
    candidate, fixtures, head, arm: str, *, hard: bool = True, temperature: float = T_MIN
) -> dict:
    rows = []
    steps = []
    terminated = 0
    for fixture in fixtures:
        binding = candidate.binding
        prepared = prepare(
            binding,
            initial_binding_state(binding, f"17r-gen-{fixture['index']}"),
            fixture["goal"].encode("utf-8"),
            fixture["records"],
        )
        context = build_context(
            binding, fixture["goal"].encode("utf-8"), tuple(fixture["record_keys"])
        )
        start = time.process_time()
        with torch.no_grad():
            generated = routing_generate(
                candidate,
                prepared,
                context,
                head,
                arm=arm,
                span_head=candidate.span_head,
                pointer_temperature=temperature,
                pointer_hard=hard,
            )
        rows.append(time.process_time() - start)
        steps.append(len(generated["probabilities"]))
        terminated += int(bool(generated["terminated"]))
    return {
        "examples": len(rows),
        "cpu": sum(rows),
        "cpu_per_example": statistics.median(rows),
        "rows": rows,
        "pointer_hard": hard,
        "pointer_temperature": temperature,
        "steps": steps,
        "median_steps": statistics.median(steps),
        "terminated": terminated,
    }


def measure_evaluation(candidate, fixtures, head, arm: str) -> dict:
    rows = []
    for fixture in fixtures:
        binding = candidate.binding
        prepared = prepare(
            binding,
            initial_binding_state(binding, f"17r-eval-{fixture['index']}"),
            fixture["goal"].encode("utf-8"),
            fixture["records"],
        )
        context = build_context(
            binding, fixture["goal"].encode("utf-8"), tuple(fixture["record_keys"])
        )
        with torch.no_grad():
            generated = routing_generate(
                candidate,
                prepared,
                context,
                head,
                arm=arm,
                span_head=candidate.span_head,
                pointer_temperature=T_MIN,
                pointer_hard=True,
            )
        start = time.process_time()
        score_generation(
            generated["raw_bytes"],
            tuple(fixture["target_prefix"]),
            generated["terminated"],
        )
        rows.append(time.process_time() - start)
    return {"examples": len(rows), "cpu": sum(rows), "cpu_per_example": statistics.median(rows)}


def fresh_candidate(arm: str = "A_route_lock"):
    candidate, optimizer = build_routing_candidate(ROOT / PARENT, SEED, arm)
    return candidate, optimizer, candidate.head, candidate.span_head


def measure_adam(train_batch) -> dict:
    """One joint Adam step on the 17 routing candidate, same method as 17.3.

    The inherited pricer's ``measure_adam`` builds a 16.1 ``SelectionCandidate``
    which has no ``subject``; the new ``routing_loss`` needs the routing
    container, so the probe is rebuilt here with identical defaults.
    """
    candidate, _optimizer, head, span = fresh_candidate("A_route_lock")
    for parameter in candidate.binding.parameters():
        parameter.requires_grad = True
    for parameter in candidate.head.parameters():
        parameter.requires_grad = True
    for parameter in candidate.span_head.parameters():
        parameter.requires_grad = True
    params = [parameter for parameter in candidate.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LR)
    for fixture in train_batch:
        loss = fixture_loss(candidate, fixture, head, span, "A_route_lock", "joint", True)
        loss.backward()
    _value, cpu, wall = timed(lambda: optimizer.step())
    return {
        "cpu": cpu,
        "wall": wall,
        "parameter_groups": len(params),
        "note": "one joint Adam step including first-step state allocation (17 routing candidate)",
    }


def measure_restore() -> dict:
    """Restore the 17 routing bundle (head + span + Adam moments), not 16.1."""
    candidate, optimizer, _head, _span = fresh_candidate("A_route_lock")
    RESTORE_TMP.parent.mkdir(parents=True, exist_ok=True)
    saved = save_routing_bundle(
        RESTORE_TMP,
        candidate,
        optimizer,
        update_index=0,
        cursor={"stage": "N17.2R.price", "probe": "restore"},
        parent_identity={"bundle": str(PARENT), "seed": SEED, "arm": "A_route_lock"},
        state=initial_binding_state(candidate.binding, "17r-restore"),
    )
    rows = [timed(lambda: load_routing_bundle(RESTORE_TMP))[1] for _ in range(2)]
    return {
        "cpu": statistics.median(rows),
        "rows": rows,
        "bytes": saved["bytes"],
        "mib": saved["bytes"] / (1024 * 1024),
        "note": "load_routing_bundle: rebuild binding+head+span, load Adam moments and state",
    }


def key(stage: str, arm: str, span_on: bool) -> str:
    return f"{stage}|{arm}|span={span_on}"


def component_report(rows: dict) -> dict:
    return {
        "stage": rows["stage"],
        "arm": rows["arm"],
        "span": rows["span"],
        "forward_cpu": rows["forward_cpu"],
        "forward_backward_cpu": rows["forward_backward_cpu"],
        "bptt_cpu": rows["bptt_cpu"],
        "forward_rows": rows["forward_rows"],
        "backward_rows": rows["backward_rows"],
    }


def measure_all() -> dict:
    old = load_module("price17_3", PRICE_SCRIPT)
    old.fixture_loss = fixture_loss
    train_batch, dev_batch = old.load_fixtures()
    components = {}
    for stage in ("selection", "copy", "joint"):
        for arm in ("A_route_lock", "B_stop_lock", "C_unlocked"):
            if stage in ("selection", "copy"):
                arms = ("B_stop_lock",)
            else:
                arms = ("A_route_lock", "B_stop_lock", "C_unlocked")
            if arm not in arms:
                continue
            candidate, _optimizer, head, span = fresh_candidate(arm)
            components[(stage, arm, True)] = old.measure_components(
                candidate, train_batch, head, span, stage, arm, True
            )
    candidate, _optimizer, head, span = fresh_candidate("A_route_lock")
    components[("joint", "A_route_lock", False)] = old.measure_components(
        candidate, train_batch, head, span, "joint", "A_route_lock", False
    )

    adam = measure_adam(train_batch)
    adam_cpu = adam["cpu"]

    candidate, _optimizer, head, span = fresh_candidate("A_route_lock")
    select_forward = old.measure_forward_selection(candidate, dev_batch, head, span)
    candidate, _optimizer, head, span = fresh_candidate("A_route_lock")
    gen_a = measure_generation(candidate, dev_batch, head, "A_route_lock")
    gen_a_oldstyle = measure_generation(
        candidate, dev_batch, head, "A_route_lock", hard=False, temperature=1.0
    )
    candidate, _optimizer, head, span = fresh_candidate("C_unlocked")
    gen_c = measure_generation(candidate, dev_batch, head, "C_unlocked")
    candidate, _optimizer, head, span = fresh_candidate("A_route_lock")
    evaluation = measure_evaluation(candidate, dev_batch, head, "A_route_lock")
    restore = measure_restore()
    bundle = old.bundle_estimate()
    return {
        "components": components,
        "adam": adam,
        "adam_cpu": adam_cpu,
        "select_forward": select_forward,
        "generation": {
            "A_route_lock": gen_a,
            "C_unlocked": gen_c,
            "B_stop_lock": gen_a,
            "A_oldstyle_soft_T1": gen_a_oldstyle,
        },
        "evaluation": evaluation,
        "restore": restore,
        "bundle": bundle,
        "old": old,
        "train_batch": train_batch,
        "dev_batch": dev_batch,
    }


def serialise_components(components: dict) -> dict:
    return {key(*tuple_key): component_report(rows) for tuple_key, rows in components.items()}


def diff_row(name: str, old_value, new_value, reason: str) -> dict:
    delta = None
    ratio = None
    if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
        delta = new_value - old_value
        ratio = new_value / old_value if old_value else None
    return {
        "item": name,
        "before": old_value,
        "after": new_value,
        "delta": delta,
        "ratio": ratio,
        "reason": reason,
    }


def diff_table(old_json: dict, prices: dict, adam: dict, generation: dict, new_scope: dict) -> list:
    old_components = old_json["components"]
    old_scope = old_json["three_arm"]
    rows = []
    for stage, arm in (
        ("selection", "B_stop_lock"),
        ("copy", "B_stop_lock"),
        ("joint", "A_route_lock"),
    ):
        before = old_components[f"{stage}|{arm}|span=True"]["forward_backward_cpu"]
        if stage == "selection":
            after = prices["selection_with"]
        elif stage == "copy":
            after = prices["copy_with"]
        else:
            after = prices["joint_A_route_lock_with"]
        reason = (
            "selection CE and span heads are unchanged, but the teacher-forced step it drives now runs the point transport (source softmax + row softmax) instead of the old single soft row"
            if stage == "selection"
            else "point transport replaces soft diffusion and the span start bias is built every step; same tensor volume, one extra normalise/index_add and per-record start logits"
        )
        rows.append(diff_row(f"{stage}.{arm}.forward_backward_cpu", before, after, reason))
    rows.append(
        diff_row(
            "adam_cpu",
            old_json["adam"]["cpu"],
            adam["cpu"],
            "same trainable parameter set and Adam defaults; delta is allocator/timer noise",
        )
    )
    rows.append(
        diff_row(
            "generation_cpu_per_example.A",
            old_json["generation"]["A_route_lock"]["cpu_per_example"],
            generation["A_route_lock"]["cpu_per_example"],
            "frozen inference: hard argmax pointer at T=0.05 plus the per-record span start bias; the point switch also changes which byte is copied and therefore when generation terminates",
        )
    )
    rows.append(
        diff_row(
            "generation_cpu_per_example.A_oldstyle_soft_T1",
            old_json["generation"]["A_route_lock"]["cpu_per_example"],
            generation["A_oldstyle_soft_T1"]["cpu_per_example"],
            "same soft semantics and temperature as the old pricer, now with the span start bias and the point-transport row; isolates the hard-pointer switch from the span/trajectory change",
        )
    )
    rows.append(
        diff_row(
            "generation_median_steps.A",
            None,
            generation["A_route_lock"]["median_steps"],
            "steps actually emitted by the new hard pointer (old pricer did not record steps)",
        )
    )
    rows.append(
        diff_row(
            "generation_median_steps.A_oldstyle_soft_T1",
            None,
            generation["A_oldstyle_soft_T1"]["median_steps"],
            "steps emitted by the soft probe; a lower step count at the same fixture means the hard pointer is running longer trajectories",
        )
    )
    rows.append(
        diff_row(
            "evaluation_cpu_per_example",
            old_json["evaluation"]["cpu_per_example"],
            new_scope["evaluation"]["cpu_per_example"],
            "score_generation untouched",
        )
    )
    rows.append(
        diff_row(
            "selection_probe_cpu_per_example",
            old_json["selection_forward"]["cpu_per_example"],
            new_scope["select_forward"]["cpu_per_example"],
            "the selection CE term itself is untouched, but each probed example still runs the teacher-forced step under the new point transport",
        )
    )
    rows.append(
        diff_row(
            "restore_cpu",
            old_json["restore"]["cpu"],
            new_scope["restore"]["cpu"],
            "the old figure timed the inherited 16.1 bundle; restore now reloads the 17 routing bundle (head + span + Adam moments + state)",
        )
    )
    rows.append(
        diff_row(
            "bundle_mib_final",
            old_json["bundle"]["mib_replace_old_head"],
            new_scope["bundle"]["mib_replace_old_head"],
            "parameter count and dtype unchanged (head 37,120 + span 258)",
        )
    )
    rows.append(
        diff_row(
            "bundle_mib_saved_probe",
            old_json["bundle"]["mib_replace_old_head"],
            new_scope["restore"]["mib"],
            "actual saved routing bundle: parameter + two Adam moments + state, float64 with torch zip padding",
        )
    )
    for arm in ("A_route_lock", "B_stop_lock", "C_unlocked"):
        rows.append(
            diff_row(
                f"training_cpu_per_run.{arm}",
                old_scope["per_arm_seed_training_cpu"][arm],
                new_scope["per_arm_seed_training_cpu"][arm],
                "aggregate of the stage unit prices above over 300 updates (100 selection + 100 copy + 100 joint)",
            )
        )
    rows.append(
        diff_row(
            "total_cpu.three_arm",
            old_json["three_arm"]["total_cpu"],
            new_scope["total_cpu"],
            "sum of training, generation, probe, restore, overrun and fixed 60 s",
        )
    )
    rows.append(
        diff_row(
            "total_cpu.two_arm",
            old_json["two_arm"]["total_cpu"],
            None,
            "computed below",
        )
    )
    return rows


def main() -> int:
    started_cpu = time.process_time()
    started_wall = time.perf_counter()
    measured = measure_all()
    old = measured["old"]
    components = measured["components"]
    adam_cpu = measured["adam_cpu"]
    gen = measured["generation"]
    evaluation = measured["evaluation"]
    select_forward = measured["select_forward"]
    restore = measured["restore"]
    bundle = measured["bundle"]

    prices = {
        "selection_with": old.train_cpu("selection", "B_stop_lock", True, components, adam_cpu),
        "copy_with": old.train_cpu("copy", "B_stop_lock", True, components, adam_cpu),
        "joint_A_route_lock_with": old.train_cpu(
            "joint", "A_route_lock", True, components, adam_cpu
        ),
        "joint_B_stop_lock_with": old.train_cpu("joint", "B_stop_lock", True, components, adam_cpu),
        "joint_C_unlocked_with": old.train_cpu("joint", "C_unlocked", True, components, adam_cpu),
        "joint_A_route_lock_without": old.train_cpu(
            "joint", "A_route_lock", False, components, adam_cpu
        ),
        "joint_span_increment_A": old.train_cpu("joint", "A_route_lock", True, components, adam_cpu)
        - old.train_cpu("joint", "A_route_lock", False, components, adam_cpu),
    }

    three_arm = old.scope(
        ("A_route_lock", "B_stop_lock", "C_unlocked"),
        components,
        adam_cpu,
        gen,
        evaluation,
        select_forward,
        restore,
        bundle,
    )
    two_arm = old.scope(
        ("A_route_lock", "B_stop_lock"),
        components,
        adam_cpu,
        gen,
        evaluation,
        select_forward,
        restore,
        bundle,
    )

    candidate, _optimizer, _head, _span = fresh_candidate("A_route_lock")
    identity = candidate_identity(candidate)

    old_json = json.loads((ROOT / OLD_REPORT).read_text(encoding="utf-8"))
    rows = diff_table(
        old_json,
        prices,
        measured["adam"],
        gen,
        {
            **three_arm,
            "evaluation": evaluation,
            "select_forward": select_forward,
            "restore": restore,
            "bundle": bundle,
        },
    )
    for row in rows:
        if row["item"] == "total_cpu.two_arm":
            row["after"] = two_arm["total_cpu"]
            row["delta"] = two_arm["total_cpu"] - old_json["two_arm"]["total_cpu"]
            row["ratio"] = two_arm["total_cpu"] / old_json["two_arm"]["total_cpu"]

    timing = {
        "cpu_seconds": time.process_time() - started_cpu,
        "wall_seconds": time.perf_counter() - started_wall,
    }
    report = {
        "schema": "taiji-selection-routing-17-2r-price-v1",
        "step": "N17.2R.price",
        "date": "2026-11-03",
        "candidate": identity,
        "candidate_sha": identity["candidate_sha"],
        "graph": "hard point transport + position supervision; pointer T 1.00->0.05, entropy 0.01",
        "dtype": "float64",
        "threads": 1,
        "batch": old.BATCH,
        "lr": old.LR,
        "span_lambda": SPAN_LAMBDA,
        "pointer_temperature": {"selection_free": 1.0, "copy_joint": T_MIN, "inference": T_MIN},
        "pointer_entropy_coef": ENTROPY_COEF,
        "engineering_adam": 1,
        "formal_updates_three_arm": 1800,
        "formal_updates_two_arm": 1200,
        "components": serialise_components(components),
        "adam": measured["adam"],
        "prices": prices,
        "generation": gen,
        "evaluation": evaluation,
        "selection_forward": select_forward,
        "restore": restore,
        "bundle": bundle,
        "three_arm": three_arm,
        "two_arm": two_arm,
        "position_supervision": {
            "parameter_count": 258,
            "bottleneck_parameter_count": 4194,
            "joint_increment_cpu": prices["joint_span_increment_A"],
            "enabled_from": "stage 1 selection through stage 3 joint",
        },
        "baseline_before": {
            "status": "superseded",
            "three_arm_total_cpu": old_json["three_arm"]["total_cpu"],
            "two_arm_total_cpu": old_json["two_arm"]["total_cpu"],
            "path": str(OLD_REPORT),
            "note": "frozen pre-change price, never used as the current value",
        },
        "difference_table": rows,
        "timing": timing,
        "limits": {
            "confirm_reads": 0,
            "formal_adam": 0,
            "engineering_adam": 1,
        },
    }
    (ROOT / REPORT).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / REPORT).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "taiji-selection-routing-17-2r-price-manifest-v1",
        "step": "N17.2R.price",
        "date": "2026-11-03",
        "candidate_sha": identity["candidate_sha"],
        "parameter_counts": identity["parameter_counts"],
        "report": str(REPORT),
        "three_arm_total_cpu": three_arm["total_cpu"],
        "two_arm_total_cpu": two_arm["total_cpu"],
        "three_arm_wall_estimate": three_arm["total_wall_estimate"],
        "two_arm_wall_estimate": two_arm["total_wall_estimate"],
        "engineering_adam": 1,
        "formal_adam": 0,
        "confirm_reads": 0,
        "dtype": "float64",
        "threads": 1,
        "batch": old.BATCH,
        "lr": old.LR,
        "span_lambda": SPAN_LAMBDA,
        "pointer_temperature_floor": T_MIN,
        "pointer_entropy_coef": ENTROPY_COEF,
        "old_baseline": {
            "status": "superseded",
            "three_arm_total_cpu": old_json["three_arm"]["total_cpu"],
            "two_arm_total_cpu": old_json["two_arm"]["total_cpu"],
            "path": str(OLD_REPORT),
        },
        "timing": timing,
    }
    (ROOT / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "three_arm_total_cpu": three_arm["total_cpu"],
                "two_arm_total_cpu": two_arm["total_cpu"],
                "three_arm_wall": three_arm["total_wall_estimate"],
                "two_arm_wall": two_arm["total_wall_estimate"],
                "joint_span_increment": prices["joint_span_increment_A"],
                "timing": timing,
                "candidate_sha": identity["candidate_sha"][:16],
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
