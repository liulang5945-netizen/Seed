"""18.1 six-cell comparison on the 18.0 data premise.

Same dose as 17.5 (2 seeds x 3 arms x 300 updates; batch 8; stages selection
1-100, copy 101-200, joint 201-300; LR 1e-3; clip 1.0), but the corpus is the
18.0 premise: 220/220 name pools, byte-disjoint train/dev/confirm name strings,
combination held-out.  The training and evaluation code is the frozen 17.5
implementation, loaded from ``compare_taiji_selection_routing_17_5.py`` so the
arm comparison stays comparable.

Discipline carried over from the 18.1 preflight:

* progress update == bundle exists == bundle step (triple check per cell);
* the bundle is written and checked *before* any evaluation, so an evaluation
  error only writes a failure receipt and never loses the weights;
* every cell can resume from its step150 bundle;
* failures are recorded per cell, never swallowed.

Mid-report at step150 (per cell): overall / new_name_copied / gamma and
1 - gamma / m magnitude / curriculum trigger count.   Confirmation examples stay
sealed: this script has no confirm path at all.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.binding_copy_scoring import score_generation  # noqa: E402
from taiji.candidate_identity import candidate_identity  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.selection_routing import (  # noqa: E402
    _advanced_state,
    observe_goal,
    routing_step,
)
from taiji.selection_routing_checkpoint import (  # noqa: E402
    build_routing_candidate,
    load_routing_bundle,
    save_routing_bundle,
)

COMPARE_MODULE = Path("scripts/training/compare_taiji_selection_routing_17_5.py")
PARENTS = {
    28031: Path("output/taiji_core_binding_select_16_1_compare/bundles/28031-A-locked-step300.pt"),
    28032: Path("output/taiji_core_binding_select_16_1_compare/bundles/28032-A-locked-step300.pt"),
}
SEEDS = (28031, 28032)
ARMS = ("A_route_lock", "B_stop_lock", "C_unlocked")

BATCH = 8
UPDATES = 300
LR = 1e-3
CLIP = 1.0
CHECKPOINT_UPDATE = 150
CURRICULUM_THRESHOLD = 0.5
STAGES = (("selection", 1, 100), ("copy", 101, 200), ("joint", 201, 300))

OUTPUT_DIR = Path("output/taiji_selection_routing_18_1")
REPORT = Path("reports/taiji_selection_routing_18_1_compare_20261104.json")
MANIFEST = Path("plans/manifests/taiji_selection_routing_18_1_compare_20261104.json")
PREREG = Path("plans/manifests/taiji_data_premise_18_0_prereg_20261104.json")
PRICING = Path("plans/manifests/taiji_data_premise_18_0_pricing_20261104.json")
DATA_FILE = Path("output/taiji_data_premise_18_0/data.json")

GATES = {
    "overall": 0.80,
    "unseen_output_name": 0.80,
    "new_description": 0.80,
    "reversed": 0.95,
    "selection_object": 0.90,
    "selection_place": 0.90,
    "content_drop": 0.10,
    "identity_rescue": 8,
    "a_minus_c": 0.10,
    "new_name_copied": 0.60,
}

WATCH_CPU = 19220.8471875
WATCH_WALL = 30000.0
M_PROBE_EXAMPLES = 8
M_PROBE_STEPS = 20

# Prior spend accepted into the ledger before this run.  17.5 is closed at
# 34,020.465625 CPU / 7,798.156692 wall.  18.0 pricing is metered (69.64 / 71.63);
# the 18.0 dataset build ran 0 optimizer steps and was not metered.  The 18.1
# preflight is booked at the cumulative 440 CPU / 49 Adam (script rework cost),
# not the 200.8 CPU of the successful pass alone.
PRIOR = {
    "cpu_seconds": 34020.465625 + 69.640625 + 440.0,
    "wall_seconds": 7798.156692 + 71.628229,
    "breakdown": {
        "n17_5_cumulative": {"cpu": 34020.465625, "wall": 7798.156692},
        "n18_0_pricing_probe": {"cpu": 69.640625, "wall": 71.628229},
        "n18_0_data_build": {"cpu": None, "wall": None, "note": "0 optimizer steps, unmetered"},
        "n18_1_preflight_cumulative": {
            "cpu": 440.0,
            "adam": 49,
            "note": "includes two aborted rework attempts; owner-requested booking",
        },
    },
}

_MODULE: dict[str, Any] = {}


def cmp():
    if "module" not in _MODULE:
        spec = importlib.util.spec_from_file_location("cmp18r", ROOT / COMPARE_MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE["module"] = module
    return _MODULE["module"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def order_seed(seed: int) -> int:
    return premise.ORDER_SEED + (seed - SEEDS[0])


def load_fixtures() -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    payload = premise.build_payload()
    return premise.fixtures(payload, "train"), premise.fixtures(payload, "dev")


def schedule(seed: int, train: tuple[dict, ...], updates: int) -> list[tuple[dict, ...]]:
    batches: list[tuple[dict, ...]] = []
    epoch = 0
    generator = torch.Generator().manual_seed(order_seed(seed))
    take = len(train) - len(train) % BATCH
    permutation = torch.randperm(take, generator=generator).tolist()
    cursor = 0
    while len(batches) < updates:
        if cursor + BATCH > len(permutation):
            epoch += 1
            generator = torch.Generator().manual_seed(order_seed(seed) + epoch)
            permutation = torch.randperm(take, generator=generator).tolist()
            cursor = 0
        batches.append(tuple(train[index] for index in permutation[cursor : cursor + BATCH]))
        cursor += BATCH
    return batches


def stage_of(update: int) -> str:
    for name, low, high in STAGES:
        if low <= update <= high:
            return name
    raise ValueError(f"update {update} outside the frozen stages")


def copy_probe(candidate, sample, arm: str, tag: str) -> dict:
    """Independent new_name_copied: object AND place name byte-exact in output."""
    rows = []
    for fixture in sample:
        generated = cmp().generate(candidate, fixture, arm, tag=tag)
        raw = generated["raw_bytes"]
        scored = score_generation(raw, fixture["target_prefix"], generated["terminated"])
        object_bytes = fixture["object_name"].encode("utf-8")
        place_bytes = fixture["place_name"].encode("utf-8")
        rows.append(
            {
                "world": fixture["world"],
                "index": fixture["index"],
                "object_in_output": object_bytes in raw,
                "place_in_output": place_bytes in raw,
                "both_in_output": object_bytes in raw and place_bytes in raw,
                "completed": bool(scored["completed"]),
                "flags": dict(fixture["flags"]),
            }
        )
    denominator = len(rows)
    copied = sum(1 for row in rows if row["both_in_output"]) / denominator if denominator else 0.0
    unseen_rows = [row for row in rows if row["flags"]["unseen_output_name"]]
    unseen_rate = (
        sum(1 for row in unseen_rows if row["completed"]) / len(unseen_rows) if unseen_rows else 0.0
    )
    return {
        "rows": rows,
        "new_name_copied": copied,
        "new_name_copied_denominator": denominator,
        "independent_unseen_output_name": {
            "rate": unseen_rate,
            "denominator": len(unseen_rows),
        },
    }


def m_probe(candidate, fixtures, arm: str, tag: str) -> dict:
    """Real mid-sequence copy read: peak |m|, coverage, cursor, keys.

    ``advance_copy`` rebuilds ``m`` from the accepted byte of the current step
    instead of accumulating it, and the routing graph never advances
    ``prefix_cursor``; this probe reports what the graph actually carries rather
    than a synthetic surface value.
    """
    binding = candidate.binding
    rows = []
    for fixture in fixtures:
        prepared, context = cmp().prepared_pair(candidate, fixture, tag, "m")
        state, gate_index = observe_goal(binding, prepared, gate_index=0)
        prefix = cmp().teacher_prefix(fixture)
        limit = min(M_PROBE_STEPS, len(prefix) - 1)
        previous = EOS
        peak = 0.0
        coverage_peak = 0.0
        with torch.no_grad():
            for index in range(limit):
                step = routing_step(
                    binding,
                    prepared,
                    state,
                    previous,
                    context,
                    candidate.head,
                    arm=arm,
                    gate_index=gate_index,
                    span_head=candidate.span_head,
                    pointer_temperature=1.0,
                )
                accepted = prefix[index]
                state, _extra, gate_index = _advanced_state(binding, prepared, step, accepted)
                previous = accepted
                peak = max(peak, float(state.m.abs().sum()))
                coverage_peak = max(coverage_peak, float(state.coverage.abs().sum()))
        rows.append(
            {
                "world": fixture["world"],
                "index": fixture["index"],
                "m_peak": peak,
                "m_last": float(state.m.abs().sum()),
                "coverage_peak": coverage_peak,
                "prefix_cursor": int(state.prefix_cursor),
                "keys": len(state.keys),
                "record_offsets": len(state.record_offsets),
                "advanced": limit,
            }
        )
    peaks = [row["m_peak"] for row in rows]
    return {
        "examples": len(rows),
        "m_peak_max": max(peaks) if peaks else 0.0,
        "m_peak_median": statistics.median(peaks) if peaks else 0.0,
        "prefix_cursor_real": rows[0]["prefix_cursor"] if rows else 0,
        "keys": rows[0]["keys"] if rows else 0,
        "record_offsets": rows[0]["record_offsets"] if rows else 0,
        "rows": rows,
    }


def triple_check(expected_update: int, bundle_path: Path, progress_path: Path) -> dict:
    progress_update = int(json.loads(progress_path.read_text(encoding="utf-8"))["update"])
    bundle_update = int(load_routing_bundle(bundle_path)[3])
    record = {
        "progress_update": progress_update,
        "bundle_exists": bundle_path.exists(),
        "bundle_update": bundle_update,
        "expected_update": expected_update,
    }
    record["consistent"] = (
        bundle_path.exists()
        and progress_update == expected_update
        and bundle_update == expected_update
    )
    return record


def readings(eval_out: dict, copy_out: dict) -> dict:
    gamma = eval_out["gamma"]["gamma_effective_mean"]
    return {
        "overall": eval_out["overall"]["completed"],
        "by_world": eval_out["overall"]["by_world"],
        "subsets": eval_out["overall"]["subsets"],
        "denominators": eval_out["overall"]["subset_denominators"],
        "new_name_copied": copy_out["new_name_copied"],
        "new_name_copied_denominator": copy_out["new_name_copied_denominator"],
        "independent_unseen_output_name": copy_out["independent_unseen_output_name"],
        "gamma_mean": gamma,
        "one_minus_gamma": 1.0 - gamma,
        "gamma_steps": eval_out["gamma"]["steps"],
        "selection_object": eval_out["selection"]["object_top1"],
        "selection_place": eval_out["selection"]["place_top1"],
        "utf8": eval_out["overall"]["utf8"],
        "eos": eval_out["overall"]["eos"],
        "examples": eval_out["examples"],
        "reversed": eval_out.get("reversed"),
        "content_interventions": eval_out.get("content_interventions"),
        "identity_rescue": eval_out.get("identity_rescue"),
        "nll": eval_out.get("nll"),
    }


def grade(entry: dict) -> dict:
    """Frozen prereg gate on a single step300 cell (+ independent copy gate)."""
    step = entry["step300"]
    checks = {
        "overall": step["overall"] >= GATES["overall"],
        "unseen_output_name": step["subsets"]["unseen_output_name"] >= GATES["unseen_output_name"],
        "new_description": step["subsets"]["new_description"] >= GATES["new_description"],
        # Prereg definition: completed rate when record order is reversed.
        "reversed": step["reversed"]["accuracy"] >= GATES["reversed"],
        "selection_object": step["selection_object"] >= GATES["selection_object"],
        "selection_place": step["selection_place"] >= GATES["selection_place"],
        "content_drop": all(
            value["drop"] >= GATES["content_drop"]
            for value in step["content_interventions"]["drops"].values()
            if value["denominator"]
        ),
        "identity_rescue": step["identity_rescue"]["equal"] >= GATES["identity_rescue"],
        "new_name_copied": step["new_name_copied"] >= GATES["new_name_copied"],
    }
    checks_without_copy = {key: value for key, value in checks.items() if key != "new_name_copied"}
    recitation = bool(step["overall"] >= GATES["overall"] and not checks["new_name_copied"])
    return {
        "checks": checks,
        "checks_without_copy": checks_without_copy,
        "passed": all(checks.values()),
        "passed_without_copy": all(checks_without_copy.values()),
        "recitation_remains": recitation,
        "reversed_paired_rate_report_only": step["reversed"]["paired_rate"],
        "readings": {key: step[key] for key in ("overall", "new_name_copied")},
    }


def bootstrap(values: list[float], resamples: int = 2000, seed: int = 58031) -> dict:
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(resamples):
        index = torch.randint(0, len(values), (len(values),), generator=generator).tolist()
        samples.append(statistics.fmean(values[value] for value in index))
    samples.sort()
    return {
        "mean": statistics.fmean(values),
        "lower": samples[int(0.025 * len(samples))],
        "upper": samples[int(0.975 * len(samples))],
        "resamples": resamples,
        "seed": seed,
    }


def run_one(
    seed: int,
    arm: str,
    resume: bool = False,
    smoke_examples: int = 0,
    update_limit: int = UPDATES,
) -> dict:
    tag = f"{seed}-{arm}"
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, dev = load_fixtures()
    if smoke_examples:
        dev = dev[:smoke_examples]
        train = train[:BATCH]
    parent = ROOT / PARENTS[seed]
    parent_sha = sha256_file(parent)
    candidate, optimizer = build_routing_candidate(parent, seed, arm)
    batches = schedule(seed, train, UPDATES)

    step0_path = OUTPUT_DIR / f"step0-{tag}.json"
    if step0_path.exists():
        step0 = json.loads(step0_path.read_text(encoding="utf-8"))
    else:
        step0 = {
            "update": 0,
            **readings(
                cmp().evaluate_dev(candidate, dev, arm, tag, full=False),
                copy_probe(candidate, dev, arm, tag),
            ),
        }
        write_json(step0_path, step0)

    checkpoint_path = OUTPUT_DIR / f"{tag}-step{CHECKPOINT_UPDATE}.pt"
    resumed_from = 0
    mid = None
    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume requested but {checkpoint_path} is missing")
        (
            candidate,
            optimizer,
            _state,
            resumed_from,
            _payload,
        ) = load_routing_bundle(checkpoint_path)
        candidate.configure_trainable()
        mid_path = OUTPUT_DIR / f"midreport-{tag}.json"
        if mid_path.exists():
            mid = json.loads(mid_path.read_text(encoding="utf-8"))

    progress_path = OUTPUT_DIR / f"progress-{tag}.json"
    rows: list[dict] = []
    grad_rows: list[dict] = []
    for update in range(resumed_from + 1, update_limit + 1):
        stage = stage_of(update)
        batch = batches[update - 1]
        if stage == "selection":
            use_self_prefix = False
            rate = 0.0
        else:
            completed = [
                cmp().teacher_forced_completion(candidate, fixture, arm, update, tag)
                for fixture in batch
            ]
            rate = sum(1 for flag in completed if flag) / len(completed)
            use_self_prefix = rate >= CURRICULUM_THRESHOLD
        parameters = cmp().set_trainable(candidate, stage)
        loss, stats = cmp().batch_loss(candidate, batch, arm, stage, use_self_prefix, update, tag)
        optimizer.zero_grad()
        loss.backward()
        norms = cmp().grad_norms(candidate)
        clip = float(torch.nn.utils.clip_grad_norm_(parameters, CLIP))
        optimizer.step()
        rows.append(
            {
                "update": update,
                "stage": stage,
                "loss": float(loss.detach()),
                "nll": stats["nll"],
                "ce": stats["ce"],
                "span": stats["span"],
                "entropy": stats["entropy"],
                "temperature": cmp().temperature(update),
                "entropy_coef": cmp().entropy_coefficient(update),
                "curriculum_rate": rate,
                "self_prefix": bool(use_self_prefix),
                "clip_scale": clip,
            }
        )
        grad_rows.append({"update": update, "stage": stage, **norms})
        if update % 10 == 0 or update == update_limit:
            write_json(
                progress_path,
                {
                    "seed": seed,
                    "arm": arm,
                    "update": update,
                    "updates_total": UPDATES,
                    "stage": stage,
                    "last_loss": rows[-1]["loss"],
                    "self_prefix_updates": sum(1 for row in rows if row["self_prefix"]),
                    "cost": {
                        "cpu_seconds": time.process_time() - start_cpu,
                        "wall_seconds": time.perf_counter() - start_wall,
                    },
                },
            )
        if update == CHECKPOINT_UPDATE and mid is None:
            # Bundle first, evaluation second: an evaluation error cannot lose weights.
            info = save_routing_bundle(
                checkpoint_path,
                candidate,
                optimizer,
                update,
                {"stage": "N18.1", "seed": seed, "arm": arm, "update": update},
                {"sha256": parent_sha, "path": str(PARENTS[seed])},
                _initial_state(candidate, tag),
            )
            triple = triple_check(update, checkpoint_path, progress_path)
            if not triple["consistent"]:
                raise RuntimeError(f"artifact inconsistency at step150 for {tag}: {triple}")
            mid_eval = cmp().evaluate_dev(candidate, dev, arm, tag, full=False)
            mid_copy = copy_probe(candidate, dev, arm, tag)
            mid_m = m_probe(candidate, batches[update - 1][:M_PROBE_EXAMPLES], arm, tag)
            mid = {
                "seed": seed,
                "arm": arm,
                "update": update,
                "bundle": info,
                "triple": triple,
                **readings(mid_eval, mid_copy),
                "field_magnitude": mid_m,
                "curriculum": {
                    "triggers_total": sum(1 for row in rows if row["self_prefix"]),
                    "by_stage": {
                        name: sum(1 for row in rows if row["stage"] == name and row["self_prefix"])
                        for name in ("selection", "copy", "joint")
                    },
                    "rate_mean": statistics.fmean(
                        row["curriculum_rate"] for row in rows if row["stage"] != "selection"
                    ),
                },
                "cost": {
                    "cpu_seconds": time.process_time() - start_cpu,
                    "wall_seconds": time.perf_counter() - start_wall,
                },
            }
            write_json(OUTPUT_DIR / f"midreport-{tag}.json", mid)
            print(
                json.dumps(
                    {
                        "mid": tag,
                        "overall": round(mid["overall"], 3),
                        "copy": round(mid["new_name_copied"], 3),
                        "gamma": round(mid["gamma_mean"], 3),
                        "m_peak": mid["field_magnitude"]["m_peak_max"],
                        "curriculum": mid["curriculum"]["triggers_total"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    final_path = OUTPUT_DIR / f"{tag}-step{update_limit}.pt"
    final_info = save_routing_bundle(
        final_path,
        candidate,
        optimizer,
        update_limit,
        {"stage": "N18.1", "seed": seed, "arm": arm, "update": update_limit},
        {"sha256": parent_sha, "path": str(PARENTS[seed])},
        _initial_state(candidate, tag),
    )
    consistency = triple_check(update_limit, final_path, progress_path)
    if not consistency["consistent"]:
        raise RuntimeError(f"artifact inconsistency at step300 for {tag}: {consistency}")

    evaluation_error = None
    try:
        final_eval = cmp().evaluate_dev(candidate, dev, arm, tag, full=True)
        final_copy = copy_probe(candidate, dev, arm, tag)
    except Exception:  # noqa: BLE001 - record the receipt, keep the bundle on disk
        evaluation_error = traceback.format_exc()
        write_json(
            OUTPUT_DIR / f"evaluation-{tag}.json",
            {
                "seed": seed,
                "arm": arm,
                "status": "evaluation_failed_weights_preserved",
                "bundle": final_info,
                "consistency": consistency,
                "traceback": evaluation_error,
            },
        )
        raise

    step300 = readings(final_eval, final_copy)
    final_m = m_probe(candidate, batches[update_limit - 1][:M_PROBE_EXAMPLES], arm, tag)

    def stage_rows(name: str) -> list[dict]:
        return [row for row in rows if row["stage"] == name]

    variance = {}
    for name in ("head", "span", "binding"):
        for stage in ("selection", "copy", "joint"):
            values = [row[name] for row in grad_rows if row["stage"] == stage]
            if values and any(values):
                variance[f"{name}|{stage}"] = {
                    "n": len(values),
                    "mean": statistics.fmean(values),
                    "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                }
    cpu = time.process_time() - start_cpu
    wall = time.perf_counter() - start_wall
    return {
        "schema": "taiji-selection-routing-18-1-run-v1",
        "step": "N18.1",
        "seed": seed,
        "arm": arm,
        "parent": {"path": str(PARENTS[seed]), "sha256": parent_sha},
        "candidate": {"candidate_sha": candidate_identity(candidate)["candidate_sha"]},
        "resumed_from": resumed_from,
        "step0": step0,
        "step150": mid,
        "step300": step300,
        "step300_magnitude": final_m,
        "train": {
            "updates": len(rows),
            "first_loss": rows[0]["loss"] if rows else None,
            "last_loss": rows[-1]["loss"] if rows else None,
            "loss_by_stage": {
                name: {
                    "updates": len(stage_rows(name)),
                    "first": stage_rows(name)[0]["loss"],
                    "last": stage_rows(name)[-1]["loss"],
                    "mean_nll": statistics.fmean(row["nll"] for row in stage_rows(name)),
                    "mean_ce": statistics.fmean(row["ce"] for row in stage_rows(name)),
                    "mean_span": statistics.fmean(row["span"] for row in stage_rows(name)),
                    "mean_entropy": statistics.fmean(row["entropy"] for row in stage_rows(name)),
                }
                for name in ("selection", "copy", "joint")
                if stage_rows(name)
            },
            "self_prefix_updates": sum(1 for row in rows if row["self_prefix"]),
            "curriculum_rate_max": max(row["curriculum_rate"] for row in rows) if rows else None,
            "curriculum_by_stage": {
                name: {
                    "updates": len(stage_rows(name)),
                    "mean_rate": statistics.fmean(
                        row["curriculum_rate"] for row in stage_rows(name)
                    ),
                    "self_prefix_updates": sum(1 for row in stage_rows(name) if row["self_prefix"]),
                }
                for name in ("selection", "copy", "joint")
                if stage_rows(name)
            },
            "grad_variance": variance,
        },
        "products": {
            "step150_bundle": {
                "path": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
            },
            "step300_bundle": {"path": str(final_info["path"]), "bytes": final_info["bytes"]},
        },
        "consistency": consistency,
        "evaluation": {"ok": True},
        "timing": {"cpu_seconds": cpu, "wall_seconds": wall},
    }


def _initial_state(candidate, tag: str):
    from taiji.binding_copy import initial_binding_state

    return initial_binding_state(candidate.binding, f"{tag}-bundle")


def aggregate(results: list[dict], driver_timing: dict, extra_cpu: float = 0.0) -> dict:
    runs = {f"{result['seed']}-{result['arm']}": result for result in results}
    total_cpu = sum(result["timing"]["cpu_seconds"] for result in results)
    total_wall = max(result["timing"]["wall_seconds"] for result in results)
    cell_grades = {tag: grade(run) for tag, run in runs.items()}

    seed_gate = {}
    pooled_deltas: list[float] = []
    for seed in SEEDS:
        a = runs[f"{seed}-A_route_lock"]
        c = runs[f"{seed}-C_unlocked"]
        gate = dict(cell_grades[f"{seed}-A_route_lock"])
        a_by = a["step300"]["by_world"]
        c_by = c["step300"]["by_world"]
        deltas = [
            a_by[world]["completed"] / a_by[world]["n"]
            - c_by[world]["completed"] / c_by[world]["n"]
            for world in sorted(a_by)
            if world in c_by
        ]
        pooled_deltas.extend(deltas)
        interval = bootstrap(deltas, seed=58031 + (seed - SEEDS[0]))
        ab_by = runs[f"{seed}-B_stop_lock"]["step300"]["by_world"]
        ab_deltas = [
            a_by[world]["completed"] / a_by[world]["n"]
            - ab_by[world]["completed"] / ab_by[world]["n"]
            for world in sorted(a_by)
            if world in ab_by
        ]
        gate["a_minus_c"] = {
            "delta": statistics.fmean(deltas),
            "bootstrap": interval,
            "gate_delta": statistics.fmean(deltas) >= GATES["a_minus_c"],
            "gate_lower": interval["lower"] > 0,
            "passed": statistics.fmean(deltas) >= GATES["a_minus_c"] and interval["lower"] > 0,
        }
        gate["a_minus_b_report_only"] = {
            "delta": statistics.fmean(ab_deltas),
            "bootstrap": bootstrap(ab_deltas, seed=59031 + (seed - SEEDS[0])),
            "lower_positive": bootstrap(ab_deltas, seed=59031 + (seed - SEEDS[0]))["lower"] > 0,
            "note": "owner-requested reading; not part of the frozen gate",
        }
        gate["passed_with_copy_and_a_minus_c"] = gate["passed"] and gate["a_minus_c"]["passed"]
        seed_gate[str(seed)] = gate

    pooled = bootstrap(pooled_deltas, seed=58051)
    pooled_gate = {
        "delta": statistics.fmean(pooled_deltas),
        "bootstrap": pooled,
        "gate_delta": statistics.fmean(pooled_deltas) >= GATES["a_minus_c"],
        "gate_lower": pooled["lower"] > 0,
        "passed": statistics.fmean(pooled_deltas) >= GATES["a_minus_c"] and pooled["lower"] > 0,
        "n_world_deltas": len(pooled_deltas),
        "note": "two-seed world bootstrap: 12 dev worlds per seed pooled",
    }
    dev_gate = (
        all(gate["passed_with_copy_and_a_minus_c"] for gate in seed_gate.values())
        and pooled_gate["passed"]
    )

    table = {}
    for tag, run in runs.items():
        rows = {}
        for label in ("step0", "step150", "step300"):
            entry = run[label]
            if entry is None:
                rows[label] = None
                continue
            rows[label] = {
                "overall": entry["overall"],
                "new_name_copied": entry["new_name_copied"],
                "unseen_output_name": entry["subsets"]["unseen_output_name"],
                "gamma_mean": entry["gamma_mean"],
                "one_minus_gamma": entry["one_minus_gamma"],
                "selection_object": entry["selection_object"],
                "selection_place": entry["selection_place"],
            }
        table[tag] = rows

    recitation = sorted(tag for tag, value in cell_grades.items() if value["recitation_remains"])
    total_watch = {
        "run_cpu_seconds": total_cpu,
        "run_wall_seconds": total_wall,
        "watch_cpu": WATCH_CPU,
        "watch_wall": WATCH_WALL,
        "below_watch": total_cpu + extra_cpu < WATCH_CPU and total_wall < WATCH_WALL,
        "headroom_cpu": WATCH_CPU - total_cpu - extra_cpu,
        "cumulative_cpu_seconds": PRIOR["cpu_seconds"] + total_cpu + extra_cpu,
        "cumulative_wall_seconds": PRIOR["wall_seconds"] + total_wall,
        "evaluation_only_cpu_seconds": extra_cpu,
        "evaluation_only_note": "by-world re-evaluation of saved step300 bundles; 0 optimizer steps",
    }
    return {
        "schema": "taiji-selection-routing-18-1-compare-v1",
        "step": "N18.1",
        "date": "2026-11-04",
        "dtype": "float64",
        "threads": 1,
        "data": {
            "file": str(DATA_FILE),
            "file_sha256": sha256_file(ROOT / DATA_FILE),
            "digest": premise.digest(premise.build_payload()),
            "train_examples": len(premise.fixtures(premise.build_payload(), "train")),
            "dev_examples": len(premise.fixtures(premise.build_payload(), "dev")),
            "confirm_examples_sealed": 192,
            "confirm_reads": 0,
        },
        "prereg": {"path": str(PREREG), "sha256": sha256_file(ROOT / PREREG), "gates": GATES},
        "pricing": {"path": str(PRICING), "sha256": sha256_file(ROOT / PRICING)},
        "config": {
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "updates": UPDATES,
            "batch": BATCH,
            "stages": {"selection": [1, 100], "copy": [101, 200], "joint": [201, 300]},
            "lr": LR,
            "clip": CLIP,
            "checkpoint_update": CHECKPOINT_UPDATE,
            "parallel": "ProcessPoolExecutor(max_workers=6), same as 17.5",
        },
        "runs": runs,
        "cells": table,
        "cell_grades": cell_grades,
        "seed_gates": seed_gate,
        "pooled_a_minus_c": pooled_gate,
        "development_gate": {
            "passed": dev_gate,
            "release_confirm_n18_2": dev_gate,
            "note": "18.2 only if true; otherwise no confirm read and attribution opens",
        },
        "recitation_remains_cells": recitation,
        "totals": {
            "formal_adam": len(results) * UPDATES,
            "engineering_adam": 0,
            "cpu_seconds": total_cpu,
            "wall_seconds_parallel": total_wall,
            "prior": PRIOR,
            "cumulative": total_watch,
        },
        "driver_timing": driver_timing,
        "limits": {
            "confirm_reads": 0,
            "no_seed_extension": True,
            "no_dose_change": True,
            "no_criterion_change": True,
        },
    }


def evaluate_step300_cell(seed: int, arm: str) -> dict:
    """Re-read a saved step300 bundle (0 optimizer steps) to recover ``by_world``.

    The first driver stored only the completed rate, not the per-world split, so
    the A-C bootstrap needs one deterministic forward pass over dev per cell.
    Weights are never touched; Adam state is never stepped.
    """
    tag = f"{seed}-{arm}"
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    _train, dev = load_fixtures()
    path = OUTPUT_DIR / f"{tag}-step300.pt"
    candidate, _optimizer, _state, update, _payload = load_routing_bundle(path)
    candidate.configure_trainable()
    out = cmp().evaluate_dev(candidate, dev, arm, tag, full=False)
    payload = {
        "seed": seed,
        "arm": arm,
        "update": update,
        "tag": tag,
        "bundle": {"path": str(path), "sha256": sha256_file(path)},
        "overall": out["overall"],
        "gamma": out["gamma"],
        "selection": out["selection"],
        "examples": out["examples"],
        "timing": {
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.perf_counter() - start_wall,
        },
        "note": "evaluation only; no optimizer step, by_world recovered for the A-C bootstrap",
    }
    write_json(OUTPUT_DIR / f"eval300-{tag}.json", payload)
    return payload


def load_saved_results() -> tuple[list[dict], float, float]:
    """Load the six finished run receipts plus the by-world sidecars."""
    results: list[dict] = []
    extra_cpu = 0.0
    extra_wall = 0.0
    for seed in SEEDS:
        for arm in ARMS:
            tag = f"{seed}-{arm}"
            run = json.loads((OUTPUT_DIR / f"run-{tag}.json").read_text(encoding="utf-8"))
            side_path = OUTPUT_DIR / f"eval300-{tag}.json"
            side = json.loads(side_path.read_text(encoding="utf-8"))
            if abs(float(side["overall"]["completed"]) - float(run["step300"]["overall"])) > 1e-12:
                raise RuntimeError(f"by-world re-evaluation disagrees with saved reading: {tag}")
            run["step300"]["by_world"] = side["overall"]["by_world"]
            run["re_evaluation"] = {
                "path": str(side_path),
                "sha256": sha256_file(side_path),
                "cpu_seconds": side["timing"]["cpu_seconds"],
                "adam_steps": 0,
            }
            extra_cpu += float(side["timing"]["cpu_seconds"])
            extra_wall += float(side["timing"]["wall_seconds"])
            results.append(run)
    return results, extra_cpu, extra_wall


def finalize(results: list[dict], driver_timing: dict, extra_cpu: float = 0.0) -> dict:
    report = aggregate(results, driver_timing, extra_cpu)
    report["source"] = {
        "compare_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "reused_training": str(COMPARE_MODULE),
        "routing": "taiji/selection_routing.py",
        "checkpoint": "taiji/selection_routing_checkpoint.py",
        "data_premise": "taiji/data_premise_18_0.py",
    }
    write_json(REPORT, report)
    write_json(
        MANIFEST,
        {
            "schema": "taiji-selection-routing-18-1-manifest-v1",
            "step": "N18.1",
            "date": "2026-11-04",
            "report": str(REPORT),
            "report_sha256": sha256_file(REPORT),
            "report_bytes": REPORT.stat().st_size,
            "prereg": report["prereg"],
            "pricing": report["pricing"],
            "data": report["data"],
            "inputs": {
                "parents": {
                    str(seed): {"path": str(path), "sha256": sha256_file(ROOT / path)}
                    for seed, path in PARENTS.items()
                },
            },
            "outputs": {
                "bundles": [
                    str(OUTPUT_DIR / f"{seed}-{arm}-step300.pt") for seed in SEEDS for arm in ARMS
                ],
                "step150": [
                    str(OUTPUT_DIR / f"{seed}-{arm}-step150.pt") for seed in SEEDS for arm in ARMS
                ],
                "midreports": [
                    str(OUTPUT_DIR / f"midreport-{seed}-{arm}.json")
                    for seed in SEEDS
                    for arm in ARMS
                ],
            },
            "formal_adam": len(results) * UPDATES,
            "confirm_reads": 0,
            "cells": {
                tag: {
                    "passed": value["passed"],
                    "passed_without_copy": value["passed_without_copy"],
                    "recitation_remains": value["recitation_remains"],
                    "checks": value["checks"],
                }
                for tag, value in report["cell_grades"].items()
            },
            "development_gate": report["development_gate"],
            "recitation_remains_cells": report["recitation_remains_cells"],
            "totals": report["totals"],
        },
    )
    print(
        json.dumps(
            {
                "report": str(REPORT),
                "dev_gate": report["development_gate"]["passed"],
                "cpu": round(report["totals"]["cpu_seconds"], 1),
                "cumulative_cpu": round(
                    report["totals"]["cumulative"]["cumulative_cpu_seconds"], 1
                ),
                "wall": round(driver_timing["wall_seconds"], 1),
                "recitation": report["recitation_remains_cells"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return report


def watchline_monitor(stop: threading.Event) -> None:
    """Report extrapolation and completed cells as soon as the line is crossed."""
    alerted = False
    while not stop.is_set():
        rows = []
        for seed in SEEDS:
            for arm in ARMS:
                path = OUTPUT_DIR / f"progress-{seed}-{arm}.json"
                if not path.exists():
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    rows.append(payload)
                except (json.JSONDecodeError, OSError):
                    continue
        observed = sum(float(row["cost"]["cpu_seconds"]) for row in rows)
        extrapolated = 0.0
        for row in rows:
            update = int(row["update"])
            extrapolated += float(row["cost"]["cpu_seconds"]) * UPDATES / max(update, 1)
        finished = [
            f"{row['seed']}-{row['arm']}"
            for seed in SEEDS
            for arm in ARMS
            if (OUTPUT_DIR / f"{seed}-{arm}-step300.pt").exists()
        ]
        if not alerted and (observed >= WATCH_CPU or extrapolated >= WATCH_CPU):
            alerted = True
            payload = {
                "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "observed_cpu_seconds": observed,
                "extrapolated_cpu_seconds": extrapolated,
                "watch_cpu": WATCH_CPU,
                "cells_with_progress": [
                    f"{row['seed']}-{row['arm']}:{row['update']}" for row in rows
                ],
                "cells_finished_step300": finished,
                "note": "observation line touched; run continues, report must flag it",
            }
            write_json(OUTPUT_DIR / "watchline.json", payload)
            print(json.dumps({"WATCHLINE_HIT": payload}, ensure_ascii=False), flush=True)
        stop.wait(60)


def main() -> int:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    arguments = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if arguments.smoke:
        OUTPUT_DIR = Path("output/taiji_selection_routing_18_1_smoke")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        result = run_one(
            SEEDS[0],
            ARMS[0],
            resume=False,
            smoke_examples=arguments.smoke,
            update_limit=2,
        )
        write_json(OUTPUT_DIR / f"run-{result['seed']}-{result['arm']}.json", result)
        print(
            json.dumps(
                {
                    "smoke": f"{result['seed']}-{result['arm']}",
                    "step0": result["step0"]["overall"],
                    "step300": result["step300"]["overall"],
                    "copy": result["step300"]["new_name_copied"],
                    "cpu": result["timing"]["cpu_seconds"],
                }
            )
        )
        return 0
    if arguments.eval_only:
        total_cpu = 0.0
        for seed in SEEDS:
            for arm in ARMS:
                payload = evaluate_step300_cell(seed, arm)
                total_cpu += payload["timing"]["cpu_seconds"]
                print(
                    json.dumps(
                        {
                            "eval": f"{seed}-{arm}",
                            "overall": round(payload["overall"]["completed"], 3),
                            "worlds": len(payload["overall"]["by_world"]),
                            "cpu": round(payload["timing"]["cpu_seconds"], 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        print(json.dumps({"eval_only_cpu_seconds": round(total_cpu, 1)}))
        return 0
    if arguments.aggregate_only:
        start_cpu = time.process_time()
        start_wall = time.perf_counter()
        results, extra_cpu, extra_wall = load_saved_results()
        driver_timing = {
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.perf_counter() - start_wall,
            "evaluation_only_cpu_seconds": extra_cpu,
            "evaluation_only_wall_seconds": extra_wall,
            "note": "offline aggregation of six finished cells; no training",
        }
        finalize(results, driver_timing, extra_cpu)
        return 0
    if arguments.worker:
        result = run_one(arguments.seed, arguments.arm, resume=arguments.resume)
        write_json(OUTPUT_DIR / f"run-{arguments.seed}-{arguments.arm}.json", result)
        print(
            json.dumps(
                {
                    "seed": arguments.seed,
                    "arm": arguments.arm,
                    "cpu": result["timing"]["cpu_seconds"],
                    "wall": result["timing"]["wall_seconds"],
                }
            )
        )
        return 0

    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    stop = threading.Event()
    monitor = threading.Thread(target=watchline_monitor, args=(stop,), daemon=True)
    monitor.start()

    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=len(SEEDS) * len(ARMS)) as pool:
        futures = {
            pool.submit(run_one, seed, arm, arguments.resume): (seed, arm)
            for seed in SEEDS
            for arm in ARMS
        }
        for future in as_completed(futures):
            seed, arm = futures[future]
            try:
                result = future.result()
                results.append(result)
                write_json(OUTPUT_DIR / f"run-{seed}-{arm}.json", result)
                print(
                    json.dumps(
                        {
                            "done": f"{seed}-{arm}",
                            "cpu": round(result["timing"]["cpu_seconds"], 1),
                            "wall": round(result["timing"]["wall_seconds"], 1),
                            "step300_overall": round(result["step300"]["overall"], 3),
                            "step300_copy": round(result["step300"]["new_name_copied"], 3),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception:  # noqa: BLE001 - record failures, do not hide them
                failures.append({"seed": seed, "arm": arm, "traceback": traceback.format_exc()})
                write_json(
                    OUTPUT_DIR / f"failure-{seed}-{arm}.json",
                    {
                        "seed": seed,
                        "arm": arm,
                        "status": "training_or_evaluation_failed",
                        "traceback": failures[-1]["traceback"],
                    },
                )
                print(json.dumps({"failed": f"{seed}-{arm}"}), flush=True)
    stop.set()

    if failures:
        write_json(OUTPUT_DIR / "failures.json", {"failures": failures})
        print(json.dumps({"failures": len(failures)}))
        return 1
    driver_timing = {
        "cpu_seconds": time.process_time() - start_cpu,
        "wall_seconds": time.perf_counter() - start_wall,
    }
    finalize(results, driver_timing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
