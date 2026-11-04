"""18.0 pricing on the new data premise (zero updates + four engineering Adam steps).

The pricing follows the 17.3 shape: measure the components on the real container,
then extrapolate 2 seeds x 3 arms x 300 updates with a step150 split.  Every
number is reported bare and multiplied by the 2.11 protection coefficient that
the 17.9 slowdown probe justified.  No formal training starts here; the four
optimiser steps are discarded and nothing is saved except a temporary bundle
that is removed immediately.

Run: ``python scripts/training/price_taiji_data_premise_18_0.py``
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.binding_copy import initial_binding_state  # noqa: E402
from taiji.binding_copy_scoring import score_generation  # noqa: E402
from taiji.selection_routing_checkpoint import (  # noqa: E402
    build_routing_candidate,
    load_routing_bundle,
    save_routing_bundle,
)

COMPARE = ROOT / "scripts/training/compare_taiji_selection_routing_17_5.py"
PARENT = ROOT / "output/taiji_core_binding_select_16_1_compare/bundles/28031-A-locked-step300.pt"
RESTORE_BUNDLE = ROOT / "output/taiji_selection_routing_17_5/28031-B_stop_lock-step150.pt"
TEMP_DIR = ROOT / "output/taiji_data_premise_18_0"
REPORT = ROOT / "reports/taiji_data_premise_18_0_pricing_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_data_premise_18_0_pricing_20261104.json"

SEED = 28031
ARM = "A_route_lock"
BATCH = 8
PROTECTION = 2.11
CELLS = 6
UPDATES_PER_STAGE = 100
CONTENT_SAMPLE_MODULO = 10
READINESS_EXAMPLES = 40
EVAL_SAMPLE = 24
MAX_ADAM = 4
MAX_CPU = 120.0
CLIP = 1.0
# 17.5 B_stop_lock triggered the teacher-forced self-prefix curriculum on 27 of
# its 300 updates; the two measured joint steps here start from a cold candidate
# with a 0.0 completion rate, so the self-prefix generation cost is not inside
# the measured joint figure and must be priced as a separate contingency.
SELF_PREFIX_UPDATES_B = 27

_MODULE: dict[str, object] = {}


def compare_module():
    if "module" not in _MODULE:
        spec = importlib.util.spec_from_file_location("cmp18p", COMPARE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE["module"] = module
    return _MODULE["module"]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cpu() -> float:
    return time.process_time()


def wall() -> float:
    return time.perf_counter()


def split_dev(dev: tuple[dict, ...], count: int) -> tuple[dict, ...]:
    if count >= len(dev):
        return dev
    step = max(1, len(dev) // count)
    return dev[::step][:count]


def measure_generation(candidate, dev: tuple[dict, ...], arm: str) -> float:
    compare = compare_module()
    with torch.no_grad():
        for fixture in dev:
            compare.generate(candidate, fixture, arm, tag="price-gen")
    start = cpu()
    with torch.no_grad():
        for fixture in dev:
            compare.generate(candidate, fixture, arm, tag="price-gen")
    return (cpu() - start) / len(dev)


def measure_eval_light(candidate, dev: tuple[dict, ...], arm: str) -> float:
    compare = compare_module()
    subset = split_dev(dev, EVAL_SAMPLE)
    start = cpu()
    compare.evaluate_dev(candidate, subset, arm, tag="price-eval", full=False)
    return (cpu() - start) / len(subset)


def measure_eval_full(candidate, dev: tuple[dict, ...], arm: str) -> float:
    compare = compare_module()
    subset = split_dev(dev, EVAL_SAMPLE)
    start = cpu()
    compare.evaluate_dev(candidate, subset, arm, tag="price-eval", full=True)
    return (cpu() - start) / len(subset)


def measure_reversed(candidate, dev: tuple[dict, ...], arm: str) -> float:
    compare = compare_module()
    subset = split_dev(dev, 8)
    start = cpu()
    with torch.no_grad():
        for fixture in subset:
            compare.generate(
                candidate,
                fixture,
                arm,
                records=tuple(reversed(fixture["records"])),
                keys=list(reversed(fixture["record_keys"])),
                tag="price-rev",
            )
    return (cpu() - start) / len(subset)


def measure_counterfactual(candidate, dev: tuple[dict, ...], arm: str) -> float:
    compare = compare_module()
    verify = importlib.util.spec_from_file_location(
        "v18price",
        ROOT / "scripts/training/verify_taiji_binding_select_16_1_compare.py",
    )
    module = importlib.util.module_from_spec(verify)
    verify.loader.exec_module(module)
    subset = split_dev(dev, 8)
    start = cpu()
    with torch.no_grad():
        for fixture in subset:
            swapped = module.swap_object_name(fixture)
            if swapped is None:
                continue
            compare.generate(candidate, fixture, arm, records=swapped["records"], tag="price-cf")
    return (cpu() - start) / len(subset)


def measure_rescue(candidate, dev: tuple[dict, ...], arm: str) -> float:
    compare = compare_module()
    subset = dev[:8]
    start = cpu()
    with torch.no_grad():
        for fixture in subset:
            prepared, context = compare.prepared_pair(candidate, fixture, "price-rescue", "rescue")
            state = prepared.state
            state, gate_index = compare.observe_goal(candidate.binding, prepared, gate_index=0)
            compare.routing_step(
                candidate.binding,
                prepared,
                state,
                compare.EOS,
                context,
                candidate.head,
                arm=arm,
                gate_index=gate_index,
                span_head=candidate.span_head,
                pointer_hard=True,
            )
            compare.routing_step(
                candidate.binding,
                prepared,
                state,
                compare.EOS,
                context,
                candidate.head,
                arm=arm,
                gate_index=gate_index,
                span_head=candidate.span_head,
                pointer_hard=True,
            )
    return cpu() - start


def measure_update(
    candidate,
    optimizer,
    batch: tuple[dict, ...],
    stage: str,
    update: int,
    curriculum: bool,
) -> dict:
    compare = compare_module()
    parts: dict[str, float] = {}
    start = cpu()
    use_self_prefix = False
    completions = 0.0
    if curriculum:
        completions = sum(
            1
            for fixture in batch
            if compare.teacher_forced_completion(candidate, fixture, ARM, update, "price-tf")
        ) / len(batch)
        use_self_prefix = completions >= compare.CURRICULUM_THRESHOLD
    parts["curriculum"] = cpu() - start
    compare.set_trainable(candidate, stage)
    optimizer.zero_grad(set_to_none=True)
    beginning = cpu()
    loss, diagnostics = compare.batch_loss(
        candidate, batch, ARM, stage, use_self_prefix, update, "price"
    )
    parts["forward"] = cpu() - beginning
    beginning = cpu()
    loss.backward()
    parts["bptt"] = cpu() - beginning
    beginning = cpu()
    parameters = [p for p in candidate.binding.parameters() if p.requires_grad]
    parameters += [p for p in candidate.head.parameters() if p.requires_grad]
    parameters += [p for p in candidate.span_head.parameters() if p.requires_grad]
    compare.group_norm(parameters)
    torch.nn.utils.clip_grad_norm_(parameters, CLIP)
    optimizer.step()
    parts["optimizer"] = cpu() - beginning
    return {
        "stage": stage,
        "update": update,
        "use_self_prefix": bool(use_self_prefix),
        "completion_rate": completions,
        "loss": float(loss.detach()),
        "diagnostics": {key: value for key, value in diagnostics.items() if key != "steps"},
        "parts": parts,
        "total": sum(parts.values()),
    }


def measure_save(candidate, optimizer) -> float:
    state = initial_binding_state(candidate.binding, "price-save")
    path = TEMP_DIR / "price-save-probe.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    start = cpu()
    save_routing_bundle(path, candidate, optimizer, 150, {}, {"note": "pricing"}, state)
    cost = cpu() - start
    if path.exists():
        path.unlink()
    return cost


def measure_restore() -> float:
    start = cpu()
    load_routing_bundle(RESTORE_BUNDLE)
    return cpu() - start


def main() -> None:
    started_cpu = cpu()
    started_wall = wall()
    premise_payload = premise.build_payload()
    train = premise.fixtures(premise_payload, "train")
    dev = premise.fixtures(premise_payload, "dev")
    compare = compare_module()

    candidate, optimizer = build_routing_candidate(PARENT, SEED, ARM)
    compare.set_trainable(candidate, "selection")
    schedule = compare.schedule(SEED, train, 4)

    zero_update_rows = []
    with torch.no_grad():
        for fixture in dev[:2]:
            generated = compare.generate(candidate, fixture, ARM, tag="price-zero")
            zero_update_rows.append(
                score_generation(
                    generated["raw_bytes"],
                    fixture["target_prefix"],
                    generated["terminated"],
                )["reason"]
            )

    generation = measure_generation(candidate, split_dev(dev, 12), ARM)
    eval_light = measure_eval_light(candidate, dev, ARM)
    eval_full = measure_eval_full(candidate, dev, ARM)
    reversed_cost = measure_reversed(candidate, dev, ARM)
    counterfactual = measure_counterfactual(candidate, dev, ARM)
    rescue = measure_rescue(candidate, dev, ARM)
    save_cost = measure_save(candidate, optimizer)
    restore_cost = measure_restore()

    updates = [
        measure_update(candidate, optimizer, schedule[0], "selection", 1, curriculum=False),
        measure_update(candidate, optimizer, schedule[1], "copy", 101, curriculum=True),
        measure_update(candidate, optimizer, schedule[2], "joint", 201, curriculum=True),
        measure_update(candidate, optimizer, schedule[3], "joint", 202, curriculum=True),
    ]

    per_update = {
        "selection": updates[0]["total"],
        "copy": updates[1]["total"],
        "joint": (updates[2]["total"] + updates[3]["total"]) / 2.0,
    }
    content_examples = len(
        [fixture for fixture in dev if fixture["index"] % CONTENT_SAMPLE_MODULO == 0]
    )
    eval_full_total = (eval_full - rescue / EVAL_SAMPLE) * len(dev) + rescue
    eval_cell = eval_light * len(dev) + eval_full_total + generation * READINESS_EXAMPLES
    eval_split_note = (
        "step0 light + step150 readiness + step300 full; reversed/counterfactual/rescue/content "
        "are inside the measured full-eval per-example figure"
    )
    save_cell = 2.0 * save_cost
    train_cell = (
        UPDATES_PER_STAGE * per_update["selection"]
        + UPDATES_PER_STAGE * per_update["copy"]
        + UPDATES_PER_STAGE * per_update["joint"]
    )
    cell = train_cell + eval_cell + save_cell + restore_cost
    six_cells = cell * CELLS
    self_prefix_contingency = 2.0 * SELF_PREFIX_UPDATES_B * BATCH * generation

    first_half = (
        UPDATES_PER_STAGE * per_update["selection"]
        + UPDATES_PER_STAGE / 2 * per_update["copy"]
        + eval_light * len(dev)
        + generation * READINESS_EXAMPLES
        + save_cost
        + restore_cost
    )
    second_half = (
        UPDATES_PER_STAGE / 2 * per_update["copy"]
        + UPDATES_PER_STAGE * per_update["joint"]
        + eval_full * len(dev)
        + save_cost
    )

    total_cpu = cpu() - started_cpu
    total_wall = wall() - started_wall
    if len(updates) > MAX_ADAM:
        raise RuntimeError("engineering Adam budget exceeded")
    if total_cpu > MAX_CPU:
        raise RuntimeError(f"pricing CPU budget exceeded: {total_cpu:.1f}")

    report = {
        "schema": "taiji-data-premise-18-0-pricing-v1",
        "step": "N18.0",
        "date": "2026-11-04",
        "dataset": {
            "file": "output/taiji_data_premise_18_0/data.json",
            "digest": premise.digest(premise_payload),
            "train_examples": len(train),
            "dev_examples": len(dev),
        },
        "zero_update_probe": {
            "examples": len(zero_update_rows),
            "optimizer_steps": 0,
            "reasons": zero_update_rows,
            "note": "the new records load and generate inside the frozen container",
        },
        "components_cpu": {
            "generation_per_example": generation,
            "eval_light_per_example": eval_light,
            "eval_full_per_example": eval_full,
            "reversed_per_example": reversed_cost,
            "counterfactual_per_example": counterfactual,
            "rescue_fixed": rescue,
            "save": save_cost,
            "restore": restore_cost,
        },
        "updates": [
            {key: value for key, value in update.items() if key != "diagnostics"}
            | {"diagnostics": update["diagnostics"]}
            for update in updates
        ],
        "per_update_cpu": per_update,
        "budget": {
            "cells": CELLS,
            "updates_per_stage": UPDATES_PER_STAGE,
            "eval_examples": len(dev),
            "readiness_examples": READINESS_EXAMPLES,
            "content_examples": content_examples,
            "train_cell_bare": train_cell,
            "eval_cell_bare": eval_cell,
            "save_cell_bare": save_cell,
            "restore_cell_bare": restore_cost,
            "cell_bare": cell,
            "six_cells_bare": six_cells,
            "self_prefix_updates_b": SELF_PREFIX_UPDATES_B,
            "self_prefix_contingency_b_bare": self_prefix_contingency,
            "self_prefix_contingency_b_protected": self_prefix_contingency * PROTECTION,
            "six_cells_with_contingency_b_bare": six_cells + self_prefix_contingency,
            "six_cells_with_contingency_b_protected": (six_cells + self_prefix_contingency)
            * PROTECTION,
            "protection_coefficient": PROTECTION,
            "six_cells_protected": six_cells * PROTECTION,
            "step150_first_half_cell_bare": first_half,
            "step150_second_half_cell_bare": second_half,
            "step150_first_half_six_cells_bare": first_half * CELLS,
            "step150_second_half_six_cells_bare": second_half * CELLS,
            "step150_first_half_six_cells_protected": first_half * CELLS * PROTECTION,
            "step150_second_half_six_cells_protected": second_half * CELLS * PROTECTION,
            "eval_split_note": eval_split_note,
        },
        "assumptions": [
            "same container and dose as 17.5: 2 seeds x 3 arms x 300 updates, batch 8",
            "selection/copy/joint each 100 updates",
            "step0 light eval, step150 readiness 40, step300 full eval",
            "one restore per cell and two saves per cell",
            "prices are CPU seconds on an idle machine; the 2.11 factor covers external "
            "contention only, not extra dose",
            "the two measured joint steps run on a cold candidate whose teacher-forced "
            "completion rate is 0.0, which matches the 17.5 A/C arms; the 17.5 B arm "
            "triggered the self-prefix curriculum 27 times, so its extra generation cost "
            "is reported separately as a contingency and added in the *_with_contingency "
            "fields rather than baked into the base cell price",
        ],
        "cost": {
            "optimizer_steps": len(updates),
            "training_updates": 0,
            "confirm_reads": 0,
            "cpu_seconds": total_cpu,
            "wall_seconds": total_wall,
        },
        "caps": {"max_adam": MAX_ADAM, "max_cpu": MAX_CPU},
    }
    write_json(REPORT, report)
    write_json(
        MANIFEST,
        {
            "schema": "taiji-data-premise-18-0-pricing-manifest-v1",
            "step": "N18.0",
            "date": "2026-11-04",
            "generated_by": str(Path(__file__).resolve().relative_to(ROOT)),
            "generator_sha256": __import__("hashlib")
            .sha256(Path(__file__).read_bytes())
            .hexdigest(),
            "report": {
                "path": str(REPORT.relative_to(ROOT)),
                "sha256": __import__("hashlib").sha256(REPORT.read_bytes()).hexdigest(),
                "bytes": REPORT.stat().st_size,
            },
            "budget": {
                key: report["budget"][key]
                for key in (
                    "six_cells_bare",
                    "six_cells_protected",
                    "self_prefix_contingency_b_bare",
                    "six_cells_with_contingency_b_bare",
                    "six_cells_with_contingency_b_protected",
                )
            },
            "cost": report["cost"],
        },
    )
    print(
        json.dumps(
            {
                "per_update_cpu": {key: round(value, 2) for key, value in per_update.items()},
                "cell_bare": round(cell, 1),
                "six_cells_bare": round(six_cells, 1),
                "six_cells_protected": round(six_cells * PROTECTION, 1),
                "self_prefix_contingency_b_bare": round(self_prefix_contingency, 1),
                "six_cells_with_contingency_b_bare": round(six_cells + self_prefix_contingency, 1),
                "six_cells_with_contingency_b_protected": round(
                    (six_cells + self_prefix_contingency) * PROTECTION, 1
                ),
                "step150_first_cell_bare": round(first_half, 1),
                "step150_second_cell_bare": round(second_half, 1),
                "cpu": round(total_cpu, 1),
                "wall": round(total_wall, 1),
                "adam": len(updates),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
