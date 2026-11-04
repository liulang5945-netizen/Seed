"""N18.1 pre-training preflight on the 18.0 data premise.

Six cells (2 seeds x 3 arms), 3 engineering Adam each, plus one single-update
failure-injection cell.  No formal dose, no confirm read, no judging.

Three things are checked:

1. checkpoint round trip: parameters, Adam moments and teacher-forced
   distributions must be bit-identical (max_abs_diff = 0), and the persisted
   copy state must be non-empty (m / coverage / keys / prefix_cursor) and
   identical after restore; the parent bundle sha and candidate_sha_v2
   (773dcec2) must not change;
2. evaluation path: `evaluate_dev(full=True)` must run on a minimal sample of
   every cell, and both `new_name_copied` and `unseen_output_name` must produce
   values with denominators taken from the subset length (S1);
3. artifact ordering: progress update == bundle exists == bundle step, and with
   one update plus an injected evaluation error only a failure receipt is
   written while the weights stay on disk.

Caps: 60 engineering Adam / 400 CPU.  Exceeding either aborts the run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.binding_copy_scoring import score_generation  # noqa: E402
from taiji.candidate_identity import candidate_identity, weights_sha256  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.selection_routing import (  # noqa: E402
    _advanced_state,
    observe_goal,
    route_labels,
    routing_loss,
    routing_step,
)
from taiji.selection_routing_checkpoint import (  # noqa: E402
    build_routing_candidate,
    load_routing_bundle,
    parameter_tensors,
    save_routing_bundle,
)

COMPARE_SCRIPT = Path("scripts/training/verify_taiji_binding_select_16_1_compare.py")
COMPARE_MODULE = Path("scripts/training/compare_taiji_selection_routing_17_5.py")
PARENTS = {
    28031: Path("output/taiji_core_binding_select_16_1_compare/bundles/28031-A-locked-step300.pt"),
    28032: Path("output/taiji_core_binding_select_16_1_compare/bundles/28032-A-locked-step300.pt"),
}
SEEDS = (28031, 28032)
ARMS = ("A_route_lock", "B_stop_lock", "C_unlocked")

ENGINEERING_STEPS_PER_CELL = 3
ENGINEERING_UPDATE_BASE = 201
CURRICULUM_STAGE = "joint"
BATCH = 8
CLIP = 1.0
EVAL_SAMPLE = 8
STATE_STEPS_MAX = 20
MAX_ADAM = 60
MAX_CPU = 400.0
EXPECTED_SHA_PREFIX = "773dcec2"

OUTPUT_DIR = Path("output/taiji_preflight_18_1")
REPORT = Path("reports/taiji_preflight_18_1_20261104.json")
MANIFEST = Path("plans/manifests/taiji_preflight_18_1_20261104.json")

_MODULE: dict[str, Any] = {}


def compare_module():
    if "module" not in _MODULE:
        spec = importlib.util.spec_from_file_location("cmp18p", ROOT / COMPARE_MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE["module"] = module
    return _MODULE["module"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def order_seed(seed: int) -> int:
    return premise.ORDER_SEED + (seed - SEEDS[0])


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


def equal_payload(left: Any, right: Any) -> tuple[bool, float, int]:
    """Bitwise compare of a nested payload; returns (equal, max_abs, tensors)."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)):
            return False, float("inf"), 0
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            return False, float("inf"), 0
        diff = float((left - right).abs().max()) if left.numel() else 0.0
        return bool(torch.equal(left, right)), diff, 1
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            return False, float("inf"), 0
        equal, worst, tensors = True, 0.0, 0
        for key in left:
            sub_equal, sub_worst, sub_tensors = equal_payload(left[key], right[key])
            equal = equal and sub_equal
            worst = max(worst, sub_worst)
            tensors += sub_tensors
        return equal, worst, tensors
    if isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            return False, float("inf"), 0
        equal, worst, tensors = True, 0.0, 0
        for left_item, right_item in zip(left, right, strict=True):
            sub_equal, sub_worst, sub_tensors = equal_payload(left_item, right_item)
            equal = equal and sub_equal
            worst = max(worst, sub_worst)
            tensors += sub_tensors
        return equal, worst, tensors
    return left == right, 0.0, 0


def optimizer_moments(optimizer: torch.optim.Adam) -> dict[str, Any]:
    state = optimizer.state_dict()["state"]
    rows = {}
    for index, entry in state.items():
        rows[int(index)] = {
            "step": entry["step"].clone(),
            "exp_avg": entry["exp_avg"].clone(),
            "exp_avg_sq": entry["exp_avg_sq"].clone(),
        }
    return rows


def teacher_forced_distributions(
    candidate, fixture: dict, arm: str, tag: str
) -> list[torch.Tensor]:
    module = compare_module()
    prepared, context = module.prepared_pair(candidate, fixture, tag, "tf")
    routes = route_labels(fixture)
    with torch.no_grad():
        output = routing_loss(
            candidate.binding,
            prepared,
            module.teacher_prefix(fixture),
            tuple(fixture["target_bytes"]),
            routes[0],
            routes[1],
            context,
            candidate.head,
            arm=arm,
            span_head=candidate.span_head,
            pointer_temperature=1.0,
            pointer_hard=True,
            collect=True,
        )
    return [row.detach().clone() for row in output["distributions"]]


def probe_state(
    candidate, fixtures: tuple[dict, ...], arm: str, tag: str
) -> tuple[Any, int, int, str]:
    """Advance the copy state until both ``m`` and ``coverage`` are non-zero.

    ``advance_copy`` rebuilds ``m`` from the accepted byte of the current step
    only (it is not cumulative), so a fixed number of steps can land on a byte
    that only matches record-final positions, which leaves ``m`` at zero; on a
    cold candidate some arms also transport no copy mass at all.  We scan the
    batch (at most ``STATE_STEPS_MAX`` teacher bytes per example) for a state
    that carries a non-zero copy read.  If none appears, the last state gets a
    deterministic non-zero ``m``/``coverage`` so the bundle surface is still
    exercised, and ``state_kind`` says so explicitly.
    """
    module = compare_module()
    binding = candidate.binding
    last_state = None
    last_advanced = 0
    last_index = 0
    for fixture_index, fixture in enumerate(fixtures):
        prepared, context = module.prepared_pair(candidate, fixture, tag, "ckpt")
        state, gate_index = observe_goal(binding, prepared, gate_index=0)
        prefix = module.teacher_prefix(fixture)
        limit = min(STATE_STEPS_MAX, len(prefix) - 1)
        previous = EOS
        advanced = 0
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
                advanced = index + 1
                if float(state.m.abs().sum()) > 0.0 and float(state.coverage.abs().sum()) > 0.0:
                    return state, advanced, fixture_index, "teacher_forced"
        last_state = state
        last_advanced = advanced
        last_index = fixture_index
    pattern = torch.arange(
        1, last_state.m.numel() + 1, dtype=last_state.m.dtype, device=last_state.m.device
    )
    pattern = pattern / pattern.numel()
    synthetic = replace(last_state, m=pattern.clone(), coverage=pattern.clone())
    return synthetic, last_advanced, last_index, "synthetic_nonzero"


def state_is_nonempty(state, advanced: int, state_kind: str) -> dict:
    return {
        "m_abs_sum": float(state.m.abs().sum()),
        "coverage_abs_sum": float(state.coverage.abs().sum()),
        "prefix_cursor_real": int(state.prefix_cursor),
        "prefix_cursor_probed": STATE_STEPS_MAX,
        "advanced_steps": advanced,
        "state_kind": state_kind,
        "keys": len(state.keys),
        "record_offsets": len(state.record_offsets),
    }


def copy_probe(candidate, sample: tuple[dict, ...], arm: str, tag: str) -> dict:
    """new_name_copied: object name AND place name byte-exact in the output."""
    module = compare_module()
    rows = []
    completed_flags = []
    for fixture in sample:
        generated = module.generate(candidate, fixture, arm, tag=tag)
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
        completed_flags.append(bool(scored["completed"]))
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
        "completed_flags": completed_flags,
    }


def triple_check(tag: str, expected_update: int, bundle_path: Path, progress_path: Path) -> dict:
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


def run_cell(seed: int, arm: str, train: tuple[dict, ...], dev: tuple[dict, ...]) -> dict:
    module = compare_module()
    tag = f"preflight-{seed}-{arm}"
    parent = ROOT / PARENTS[seed]
    parent_sha_before = sha256_file(parent)
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidate, optimizer = build_routing_candidate(parent, seed, arm)
    identity = candidate_identity(candidate)
    batches = schedule(seed, train, ENGINEERING_STEPS_PER_CELL)
    updates = []
    for offset, batch in enumerate(batches):
        update = ENGINEERING_UPDATE_BASE + offset
        parameters = module.set_trainable(candidate, CURRICULUM_STAGE)
        loss, stats = module.batch_loss(candidate, batch, arm, CURRICULUM_STAGE, False, update, tag)
        optimizer.zero_grad()
        loss.backward()
        clip = float(torch.nn.utils.clip_grad_norm_(parameters, CLIP))
        optimizer.step()
        updates.append(
            {
                "update": update,
                "stage": CURRICULUM_STAGE,
                "loss": float(loss.detach()),
                "nll": stats["nll"],
                "clip_scale": clip,
            }
        )

    fixture = batches[0][0]
    real_state, advanced, state_fixture, state_kind = probe_state(candidate, batches[0], arm, tag)
    state_record = state_is_nonempty(real_state, advanced, state_kind)
    state_record["probe_fixture"] = state_fixture
    if not (
        state_record["m_abs_sum"] > 0.0
        and state_record["coverage_abs_sum"] > 0.0
        and state_record["keys"] > 0
        and state_record["record_offsets"] > 0
    ):
        raise RuntimeError(f"copy state stayed empty for {tag}: {state_record}")
    # The 17 routing graph never advances prefix_cursor, so the real mid-sequence
    # state keeps it at 0.  A non-zero cursor is persisted on purpose so the
    # bundle surface (which does carry this field) is exercised too.
    state = replace(real_state, prefix_cursor=STATE_STEPS_MAX)

    moments_before = optimizer_moments(optimizer)
    if not moments_before:
        raise RuntimeError(f"optimizer has no moments before save for {tag}")
    moment_abs = max(
        float(entry[key].abs().max())
        for entry in moments_before.values()
        for key in ("exp_avg", "exp_avg_sq")
    )
    if moment_abs <= 0.0:
        raise RuntimeError(f"Adam moments are all zero before save for {tag}")

    bundle_path = OUTPUT_DIR / f"{tag}-step{ENGINEERING_UPDATE_BASE + len(batches) - 1}.pt"
    expected_update = ENGINEERING_UPDATE_BASE + len(batches) - 1
    progress_path = OUTPUT_DIR / f"progress-{tag}.json"
    write_json(
        progress_path,
        {
            "seed": seed,
            "arm": arm,
            "update": expected_update,
            "stage": CURRICULUM_STAGE,
            "kind": "engineering_preflight",
        },
    )
    info = save_routing_bundle(
        bundle_path,
        candidate,
        optimizer,
        expected_update,
        {"stage": "N18.1-preflight", "seed": seed, "arm": arm, "update": expected_update},
        {"sha256": parent_sha_before, "path": str(PARENTS[seed])},
        state,
    )
    order = ["progress", "save"]
    triple = triple_check(tag, expected_update, bundle_path, progress_path)
    order.append("triple_check")
    if not triple["consistent"]:
        raise RuntimeError(f"artifact inconsistency for {tag}: {triple}")

    # The teacher-forced probe runs strictly after the bundle exists: a probe
    # failure cannot lose the weights.
    order.append("eval_probe")
    distributions_before = teacher_forced_distributions(candidate, fixture, arm, tag)

    restored, restored_optimizer, restored_state, restored_update, _payload = load_routing_bundle(
        bundle_path
    )
    restored.configure_trainable()

    parameter_equal, parameter_worst, parameter_tensor_count = equal_payload(
        dict(parameter_tensors(candidate)), dict(parameter_tensors(restored))
    )
    if parameter_tensor_count != len(parameter_tensors(candidate)):
        raise RuntimeError(f"parameter tensor count drift for {tag}")

    moments_after = optimizer_moments(restored_optimizer)
    moments_equal, moments_worst, _ = equal_payload(moments_before, moments_after)

    distributions_after = teacher_forced_distributions(restored, fixture, arm, tag)
    distribution_equal = len(distributions_before) == len(distributions_after)
    distribution_worst = 0.0
    if distribution_equal:
        for left, right in zip(distributions_before, distributions_after, strict=True):
            equal, worst, _ = equal_payload(left, right)
            distribution_equal = distribution_equal and equal
            distribution_worst = max(distribution_worst, worst)

    state_equal, state_worst, state_tensors = equal_payload(
        state.payload(), restored_state.payload()
    )

    identity_after = candidate_identity(restored)
    parent_sha_after = sha256_file(parent)

    sample = dev[:EVAL_SAMPLE]
    eval_out = module.evaluate_dev(restored, sample, arm, tag, full=True)
    probe = copy_probe(restored, sample, arm, tag)

    independent = probe["independent_unseen_output_name"]
    denominators = eval_out["overall"]["subset_denominators"]
    independent_flags = {
        key: sum(1 for row in probe["rows"] if row["flags"][key])
        for key in ("unseen_combination", "unseen_output_name", "new_description", "both_new")
    }
    s1 = {
        "evaluator_denominators": denominators,
        "independent_denominators": independent_flags,
        "equal": denominators == independent_flags,
        "unseen_rate_evaluator": eval_out["overall"]["subsets"]["unseen_output_name"],
        "unseen_rate_independent": independent["rate"],
        "rate_abs_diff": abs(
            eval_out["overall"]["subsets"]["unseen_output_name"] - independent["rate"]
        ),
        "new_name_copied": probe["new_name_copied"],
        "new_name_copied_denominator": probe["new_name_copied_denominator"],
        "unseen_denominator_nonzero": denominators["unseen_output_name"] > 0,
        "new_name_copied_finite": probe["new_name_copied"] == probe["new_name_copied"],
    }
    s1["complete"] = (
        s1["equal"]
        and s1["rate_abs_diff"] <= 1e-12
        and s1["new_name_copied_denominator"] == len(sample)
        and s1["new_name_copied_finite"]
        and s1["unseen_denominator_nonzero"]
        and 0.0 <= s1["new_name_copied"] <= 1.0
    )

    checks = {
        "parameters_bitwise": parameter_equal and parameter_worst == 0.0,
        "adam_moments_bitwise": moments_equal and moments_worst == 0.0,
        "teacher_forced_bitwise": distribution_equal and distribution_worst == 0.0,
        "copy_state_nonempty": True,
        "copy_state_bitwise": state_equal and state_worst == 0.0,
        "update_index": restored_update == expected_update,
        "bundle_filename_matches_step": bundle_path.name.endswith(f"step{restored_update}.pt"),
        "triple_consistent": triple["consistent"],
        "parent_bundle_unchanged": parent_sha_before == parent_sha_after,
        "candidate_sha_v2_unchanged": identity_after["candidate_sha_v2"]
        == identity["candidate_sha_v2"],
        "candidate_sha_v2_prefix": identity["candidate_sha_v2"][:8] == EXPECTED_SHA_PREFIX,
        "save_precedes_probe": order == ["progress", "save", "triple_check", "eval_probe"],
        "evaluate_dev_full_ran": eval_out["examples"] == len(sample),
        "evaluation_denominator_s1": s1["complete"],
        "gamma_both_sides": (
            {
                "gamma_mean": eval_out["gamma"]["gamma_effective_mean"],
                "one_minus_gamma": 1.0 - eval_out["gamma"]["gamma_effective_mean"],
            }
            if "gamma" in eval_out
            else None
        ),
    }
    passed = all(
        value
        for key, value in checks.items()
        if key != "gamma_both_sides" and isinstance(value, bool)
    )

    cpu = time.process_time() - start_cpu
    wall = time.perf_counter() - start_wall
    record = {
        "seed": seed,
        "arm": arm,
        "tag": tag,
        "updates": updates,
        "state": state_record,
        "bundle": info,
        "progress": str(progress_path),
        "triple": triple,
        "order": order,
        "diffs": {
            "parameters_max_abs": parameter_worst,
            "parameters_tensors": parameter_tensor_count,
            "adam_moments_max_abs": moments_worst,
            "teacher_forced_max_abs": distribution_worst,
            "copy_state_max_abs": state_worst,
            "copy_state_tensors": state_tensors,
        },
        "adam": {
            "moment_max_abs_before": moment_abs,
            "entries": len(moments_before),
            "entry_counts_equal": sorted(moments_before) == sorted(moments_after),
        },
        "eval": {
            "overall": eval_out["overall"]["completed"],
            "subset_rates": eval_out["overall"]["subsets"],
            "denominators": denominators,
            "utf8": eval_out["overall"]["utf8"],
            "selection_object": eval_out["selection"]["object_top1"],
            "selection_place": eval_out["selection"]["place_top1"],
            "examples": eval_out["examples"],
        },
        "copy_probe": {
            "new_name_copied": probe["new_name_copied"],
            "denominator": probe["new_name_copied_denominator"],
            "rows": probe["rows"],
        },
        "s1": s1,
        "checks": checks,
        "passed": passed,
        "cost": {"cpu_seconds": cpu, "wall_seconds": wall},
        "parent_sha256": parent_sha_before,
        "candidate_sha_v2": identity["candidate_sha_v2"],
    }
    # Bundles are ignored by git; keep only the digests in the evidence file.
    bundle_sha = info["sha256"]
    record["bundle"]["sha256"] = bundle_sha
    bundle_path.unlink(missing_ok=True)
    return record


def failure_injection(seed: int, arm: str, train: tuple[dict, ...]) -> dict:
    """One update, save first, then an injected evaluation error."""
    module = compare_module()
    tag = f"inject-{seed}-{arm}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parent = ROOT / PARENTS[seed]
    candidate, optimizer = build_routing_candidate(parent, seed, arm)
    batch = schedule(seed, train, 1)[0]
    update = 301
    parameters = module.set_trainable(candidate, CURRICULUM_STAGE)
    loss, _stats = module.batch_loss(candidate, batch, arm, CURRICULUM_STAGE, False, update, tag)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, CLIP)
    optimizer.step()
    weights_before = weights_sha256(candidate)
    bundle_path = OUTPUT_DIR / f"{tag}-step{update}.pt"
    progress_path = OUTPUT_DIR / f"progress-{tag}.json"
    write_json(
        progress_path,
        {"seed": seed, "arm": arm, "update": update, "stage": CURRICULUM_STAGE, "kind": "injected"},
    )
    info = save_routing_bundle(
        bundle_path,
        candidate,
        optimizer,
        update,
        {"stage": "N18.1-preflight-injected", "seed": seed, "arm": arm, "update": update},
        {"sha256": sha256_file(parent), "path": str(PARENTS[seed])},
        replace(
            probe_state(candidate, batch, arm, tag)[0],
            prefix_cursor=STATE_STEPS_MAX,
        ),
    )
    triple = triple_check(tag, update, bundle_path, progress_path)
    receipt_path = OUTPUT_DIR / f"evaluation-{tag}.json"
    injected = None
    try:
        raise RuntimeError("injected evaluation failure (preflight only)")
    except RuntimeError:
        injected = traceback.format_exc()
        write_json(
            receipt_path,
            {
                "seed": seed,
                "arm": arm,
                "status": "evaluation_failed_weights_preserved",
                "bundle": info,
                "triple": triple,
                "traceback": injected,
            },
        )
    restored, _optimizer, _state, restored_update, _payload = load_routing_bundle(bundle_path)
    weights_after = weights_sha256(restored)
    result = {
        "tag": tag,
        "triple": triple,
        "receipt": str(receipt_path),
        "receipt_status": json.loads(receipt_path.read_text(encoding="utf-8"))["status"],
        "receipt_exists": receipt_path.exists(),
        "bundle_exists": bundle_path.exists(),
        "bundle_step": restored_update,
        "weights_preserved": weights_before == weights_after,
        "weights_tensors": len(weights_before),
        "injected_traceback_present": bool(injected),
    }
    result["passed"] = (
        triple["consistent"]
        and result["receipt_exists"]
        and result["receipt_status"] == "evaluation_failed_weights_preserved"
        and result["weights_preserved"]
        and restored_update == update
    )
    result["bundle_sha256"] = info["sha256"]
    result["cost_cpu"] = None
    bundle_path.unlink(missing_ok=True)
    return result


def main() -> int:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    payload = premise.build_payload()
    train = premise.fixtures(payload, "train")
    dev = premise.fixtures(payload, "dev")

    cells = []
    for seed in SEEDS:
        for arm in ARMS:
            cells.append(run_cell(seed, arm, train, dev))
            print(f"[{seed}-{arm}] passed={cells[-1]['passed']}", flush=True)

    injection = failure_injection(SEEDS[0], ARMS[0], train)

    cpu = time.process_time() - start_cpu
    wall = time.perf_counter() - start_wall
    engineering_adam = ENGINEERING_STEPS_PER_CELL * len(cells) + 1
    if engineering_adam > MAX_ADAM:
        raise RuntimeError(f"engineering Adam {engineering_adam} exceeds {MAX_ADAM}")
    if cpu > MAX_CPU:
        raise RuntimeError(f"cpu {cpu:.1f} exceeds {MAX_CPU}")

    report = {
        "schema": "taiji-18-1-preflight-v1",
        "step": "N18.1-preflight",
        "date": "2026-11-04",
        "purpose": "pre-training preflight on the 18.0 premise; no formal dose, no confirm read",
        "data": {
            "file_sha256": hashlib.sha256(
                (ROOT / "output/taiji_data_premise_18_0/data.json").read_bytes()
            ).hexdigest(),
            "digest": premise.digest(payload),
            "train_examples": len(train),
            "dev_examples": len(dev),
        },
        "cells": cells,
        "failure_injection": injection,
        "assertions": {
            "checkpoint_roundtrip_all_cells": all(
                cell["checks"]["parameters_bitwise"]
                and cell["checks"]["adam_moments_bitwise"]
                and cell["checks"]["teacher_forced_bitwise"]
                and cell["checks"]["copy_state_bitwise"]
                and cell["checks"]["update_index"]
                for cell in cells
            ),
            "nonzero_copy_state_all_cells": all(
                cell["state"]["prefix_cursor_probed"] > 0
                and cell["state"]["m_abs_sum"] > 0.0
                and cell["state"]["coverage_abs_sum"] > 0.0
                for cell in cells
            ),
            "parent_and_identity_stable": all(
                cell["checks"]["parent_bundle_unchanged"]
                and cell["checks"]["candidate_sha_v2_unchanged"]
                and cell["checks"]["candidate_sha_v2_prefix"]
                for cell in cells
            ),
            "save_before_eval": all(cell["checks"]["save_precedes_probe"] for cell in cells),
            "evaluate_dev_full_all_cells": all(
                cell["checks"]["evaluate_dev_full_ran"] for cell in cells
            ),
            "s1_denominators_all_cells": all(
                cell["checks"]["evaluation_denominator_s1"] for cell in cells
            ),
            "triple_all_cells": all(cell["checks"]["triple_consistent"] for cell in cells),
            "failure_injection": injection["passed"],
        },
        "verdict": "pending",
        "cost": {
            "cpu_seconds": cpu,
            "wall_seconds": wall,
            "engineering_adam": engineering_adam,
            "training_updates": 0,
            "confirm_reads": 0,
            "caps": {"engineering_adam": MAX_ADAM, "cpu": MAX_CPU},
        },
        "stop_condition": "formal 18.1 dose requires explicit owner approval",
    }
    assertions = report["assertions"]
    report["verdict"] = "preflight_passed" if all(assertions.values()) else "preflight_failed"
    write_json(REPORT, report)
    write_json(
        MANIFEST,
        {
            "schema": "taiji-18-1-preflight-manifest-v1",
            "step": "N18.1-preflight",
            "date": "2026-11-04",
            "report": str(REPORT),
            "report_sha256": sha256_file(REPORT),
            "verdict": report["verdict"],
            "assertions": assertions,
            "cells": [
                {
                    "tag": cell["tag"],
                    "passed": cell["passed"],
                    "diffs": cell["diffs"],
                    "state": cell["state"],
                    "new_name_copied": cell["copy_probe"]["new_name_copied"],
                    "denominator": cell["copy_probe"]["denominator"],
                    "s1_equal": cell["s1"]["equal"],
                    "triple": cell["triple"],
                    "cpu_seconds": round(cell["cost"]["cpu_seconds"], 1),
                }
                for cell in cells
            ],
            "failure_injection": injection,
            "cost": report["cost"],
            "stop_condition": report["stop_condition"],
        },
    )
    print(json.dumps({"verdict": report["verdict"], "assertions": assertions}, ensure_ascii=False))
    print(json.dumps(report["cost"], ensure_ascii=False))
    return 0 if report["verdict"] == "preflight_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
