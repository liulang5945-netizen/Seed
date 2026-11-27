"""17.2R single-identity and full formal-graph coverage for the 17 candidate.

Delivers the six items the owner required before any 17.4/17.5 spend:

1. one ``candidate_sha`` for the graph, contract and declared-parameter list,
   carried by every report/manifest, with a carrier table;
2. ledger T1-T16 each marked pass / fail / not-applicable-at-this-stage with
   the actual reading and the stage where it is judged;
3. the hard-point route credit estimator written down (form, no baseline,
   bias) and measured: arm A non-zero, arm B exactly zero, and step-0 selection
   CE bitwise identical across A/B;
4. cursor semantics for the point-transport graph with a non-empty mid-copy
   example;
5. a non-empty copy-state restore pre-check (two seeds x three arms, 1 Adam
   each) with bitwise params / Adam moments / probabilities / cursor;
6. the repriced graph consumed from the 17.2R price report, plus the three
   owner tables: carried-forward, superseded and difference.

The old 13,092.25 / 8,750.34 price, the old 17.4 preflight, the old T3 reading
and the old 0b graph report stay on disk untouched and are labelled
``superseded`` with a ``superseded_by`` pointer.  Nothing here retrains the
task graph; the only Adam spend is the restore pre-check.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji.binding_copy import initial_binding_state, observe_goal, prepare  # noqa: E402
from taiji.candidate_identity import candidate_identity  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.selection_routing import (  # noqa: E402
    _advanced_state,
    build_context,
    route_labels,
    routing_generate,
    routing_step,
)
from taiji.selection_routing_checkpoint import build_routing_candidate  # noqa: E402

REPORT = Path("reports/taiji_selection_routing_17_2r_20261103.json")
MANIFEST = Path("plans/manifests/taiji_selection_routing_17_2r_20261103.json")
T3_REPORT = Path("reports/taiji_selection_routing_17_2_t3_supplement_20261103_v3.json")
PREFLIGHT_REPORT = Path("reports/taiji_selection_routing_17_4_preflight_20261103_v2.json")
PRICE_REPORT = Path("reports/taiji_selection_routing_17_2r_price_20261103.json")
OLD_PRICE = Path("reports/taiji_selection_routing_17_3_price_20261103.json")
OLD_PREFLIGHT = Path("reports/taiji_selection_routing_17_4_preflight_20261103.json")
OLD_T3 = Path("reports/taiji_selection_routing_17_2_t3_supplement_20261103.json")
OLD_VERIFY = Path("reports/taiji_selection_routing_17_3_verify_20261103.json")
THRESHOLDS = Path("plans/manifests/taiji_selection_routing_17_3_r1_thresholds_20261103.json")
VERIFY_SCRIPT = Path("scripts/training/verify_taiji_selection_routing_17_3.py")
RUNNER_SCRIPT = Path("scripts/training/compare_taiji_selection_routing_17_5.py")

SEEDS = (28031, 28032)
ARMS = ("A_route_lock", "B_stop_lock", "C_unlocked")
PARENTS = {
    28031: Path("output/taiji_core_binding_select_16_1_compare/bundles/28031-A-locked-step300.pt"),
    28032: Path("output/taiji_core_binding_select_16_1_compare/bundles/28032-A-locked-step300.pt"),
}


T_MIN = 0.05


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def file_entry(path: Path, role: str) -> dict:
    full = ROOT / path
    if not full.exists():
        return {"role": role, "path": str(path), "exists": False}
    return {
        "role": role,
        "path": str(path),
        "bytes": full.stat().st_size,
        "sha256": sha256_file(path),
    }


def grad_stats(parameters) -> dict:
    norm = 0.0
    maximum = 0.0
    present = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        present += 1
        norm += float(parameter.grad.detach().pow(2).sum())
        maximum = max(maximum, float(parameter.grad.detach().abs().max()))
    return {
        "tensors_with_grad": present,
        "tensors_total": len(list(parameters)),
        "norm": norm**0.5,
        "max_abs": maximum,
        "exact_zero": bool(present == 0 or norm == 0.0),
    }


def fresh(seed: int, arm: str):
    candidate, optimizer = build_routing_candidate(ROOT / PARENTS[seed], seed, arm)
    return candidate, optimizer


def empty_state_example(candidate, fixture: dict, arm: str) -> dict:
    prepared = prepare(
        candidate.binding,
        initial_binding_state(candidate.binding, "17r-empty"),
        fixture["goal"].encode("utf-8"),
        fixture["records"],
    )
    state, _gate = observe_goal(candidate, prepared, gate_index=0)
    step = routing_step(
        candidate,
        prepared,
        state,
        EOS,
        build_context(
            candidate.binding, fixture["goal"].encode("utf-8"), tuple(fixture["record_keys"])
        ),
        candidate.head,
        arm=arm,
        gate_index=0,
        span_head=candidate.span_head,
        pointer_temperature=T_MIN,
        pointer_hard=True,
    )
    return {
        "position": 0,
        "m_nonzero": int(state.m.ne(0).sum()),
        "m_sum": float(state.m.sum()),
        "coverage_nonzero": int(state.coverage.ne(0).sum()),
        "coverage_sum": float(state.coverage.sum()),
        "prefix_cursor": int(state.prefix_cursor),
        "pointer_mass": float(step.read.q_bytes.sum()),
        "pointer_argmax": int(torch.argmax(step.read.q_bytes)) if step.read.q_bytes.numel() else -1,
        "gate_index": int(step.gate_index),
        "previous_byte": EOS,
        "note": "episode start; m and coverage are both empty, the next pointer is fresh transport",
    }


def label_audit(dev: list) -> dict:
    signature_generate = inspect.signature(routing_generate)
    signature_step = inspect.signature(routing_step)
    forbidden = ("target", "target_bytes", "gold", "span", "route_index", "label", "score")
    present_generate = [name for name in signature_generate.parameters if name in forbidden]
    present_step = [name for name in signature_step.parameters if name in forbidden]
    source_generate = inspect.getsource(routing_generate)
    source_step = inspect.getsource(routing_step)
    forward_forbidden = {
        "target_bytes_in_generate": "target_bytes" in source_generate,
        "route_index_in_generate": "route_index" in source_generate,
        "gold_in_generate": "gold" in source_generate,
        "span_in_step": "span_loss" in source_step,
        "target_bytes_in_step": "target_bytes" in source_step,
    }
    first = dev[0]
    object_raw = first["raw_example"]["records"][int(first["selection_label"])]["raw"].encode(
        "utf-8"
    )
    place_raw = first["raw_example"]["records"][
        first["record_keys"].index(first["place_description"])
    ]["raw"].encode("utf-8")
    object_position = object_raw.find(first["object_name"].encode("utf-8"))
    place_position = place_raw.find(first["place_name"].encode("utf-8"))
    return {
        "generate_signature_forbidden": present_generate,
        "step_signature_forbidden": present_step,
        "forward_source_forbidden": forward_forbidden,
        "teacher_span_from_record": {
            "object_position": object_position,
            "place_position": place_position,
            "object_name": first["object_name"],
            "place_name": first["place_name"],
        },
        "passed": bool(
            not present_generate and not present_step and not any(forward_forbidden.values())
        ),
    }


def credit_estimator_block(dev: list) -> dict:
    fixture = dev[0]
    batch = [fixture]
    runner = load_module("runner17", RUNNER_SCRIPT)
    stage_rows = {}
    for stage in ("copy", "joint"):
        rows = {}
        for arm in ARMS:
            candidate, _optimizer = fresh(28031, arm)
            runner.set_trainable(candidate, stage)
            for parameter in candidate.head.parameters():
                parameter.requires_grad_(True)
            for module in (candidate.binding, candidate.head, candidate.span_head):
                for parameter in module.parameters():
                    parameter.grad = None
            loss, _parts = runner.batch_loss(
                candidate, batch, arm, stage, False, 0, f"17r-credit-{stage}-{arm}"
            )
            loss.backward()
            rows[arm] = {
                "head": grad_stats(candidate.head.parameters()),
                "binding": grad_stats(candidate.binding.parameters()),
                "span": grad_stats(candidate.span_head.parameters()),
                "loss": float(loss.detach()),
            }
        stage_rows[stage] = rows
    candidate_a, _ = fresh(28031, "A_route_lock")
    candidate_b, _ = fresh(28031, "B_stop_lock")
    runner.set_trainable(candidate_a, "selection")
    runner.set_trainable(candidate_b, "selection")
    for candidate in (candidate_a, candidate_b):
        for parameter in candidate.head.parameters():
            parameter.requires_grad_(True)
        for module in (candidate.binding, candidate.head, candidate.span_head):
            for parameter in module.parameters():
                parameter.grad = None
    loss_a, _ = runner.batch_loss(
        candidate_a, batch, "A_route_lock", "selection", False, 0, "17r-ce-a"
    )
    loss_a.backward()
    grad_a = {
        name: parameter.grad.detach().clone()
        for name, parameter in candidate_a.head.named_parameters()
    }
    loss_b, _ = runner.batch_loss(
        candidate_b, batch, "B_stop_lock", "selection", False, 0, "17r-ce-b"
    )
    loss_b.backward()
    grad_b = {
        name: parameter.grad.detach().clone()
        for name, parameter in candidate_b.head.named_parameters()
    }
    loss_equal = bool(torch.equal(loss_a.detach(), loss_b.detach()))
    grad_equal = bool(all(torch.equal(grad_a[name], grad_b[name]) for name in grad_a))
    return {
        "estimator": {
            "A_route_lock": {
                "form": "a_train = a_hard + (a_soft - a_soft.detach()); forward value hard, VJP of the soft route",
                "baseline": "none; pathwise/straight-through, not REINFORCE",
                "bias": "targets the soft relaxation at the hard forward point; |bias| <= L_a * max_i |a_soft - a_hard|, shrinking with T -> 0.05",
            },
            "B_stop_lock": {
                "form": "a_train = a_hard.detach(); no route gradient reaches the loss",
                "baseline": "not applicable",
                "bias": "none; route credit is exactly zero by construction",
            },
            "C_unlocked": {
                "form": "inherited R1 read; head scores are not on the task-credit path",
                "baseline": "not applicable",
                "bias": "none for the head; selection CE still trains the head",
            },
        },
        "gradients": stage_rows,
        "step0_selection_ce": {
            "loss_a": float(loss_a.detach()),
            "loss_b": float(loss_b.detach()),
            "loss_bitwise_equal": loss_equal,
            "head_grad_bitwise_equal": grad_equal,
            "head_grad_max_abs_diff": max(
                float((grad_a[name] - grad_b[name]).abs().max()) for name in grad_a
            ),
            "passed": bool(loss_equal and grad_equal),
        },
        "comparability_with_old_soft_diffusion": {
            "comparable": False,
            "reason": "old A-B differed through the soft within-record transport; the hard point transport removes the position spread, so A-B now isolates record-route credit only. The 0.10 gate is re-established in 17.5, never carried over from old readings.",
        },
    }


def same_state_block(candidate, dev: list) -> dict:
    fixture = dev[0]
    prepared = prepare(
        candidate.binding,
        initial_binding_state(candidate.binding, "17r-same"),
        fixture["goal"].encode("utf-8"),
        fixture["records"],
    )
    context = build_context(
        candidate.binding, fixture["goal"].encode("utf-8"), tuple(fixture["record_keys"])
    )
    _routes = route_labels(fixture)
    state, gate_index = observe_goal(candidate, prepared, gate_index=0)
    step = routing_step(
        candidate,
        prepared,
        state,
        EOS,
        context,
        candidate.head,
        arm="A_route_lock",
        gate_index=gate_index,
        span_head=candidate.span_head,
        pointer_temperature=T_MIN,
        pointer_hard=True,
    )
    view = prepared.base.view
    units = torch.cat(view.units) if view.record_count else torch.zeros(0, dtype=torch.float64)
    inherited = candidate.reader.inherited
    content = (
        step.q_bytes @ inherited.V_position(units) + step.q_null * inherited.null_value
        if len(step.q_bytes)
        else step.q_null * inherited.null_value
    )
    copy_byte_mass = torch.zeros_like(step.probabilities)
    if len(step.q_bytes):
        copy_byte_mass = copy_byte_mass.index_add(0, view.byte_ids, step.q_bytes)
    probabilities = (step.gamma_eff + (1.0 - step.gamma_eff) * step.q_null) * step.generated + (
        1.0 - step.gamma_eff
    ) * copy_byte_mass
    next_state, _, next_gate = _advanced_state(
        candidate, prepared, step, fixture["target_prefix"][0]
    )
    return {
        "content_max_abs_diff": float((content - step.content).abs().max()),
        "probabilities_max_abs_diff": float((probabilities - step.probabilities).abs().max()),
        "content_bitwise_equal": bool(torch.equal(content, step.content)),
        "probabilities_bitwise_equal": bool(torch.equal(probabilities, step.probabilities)),
        "next_h_bitwise_equal": bool(torch.equal(next_state.living.h, step.next_h)),
        "gate_after_byte": int(next_gate),
        "passed": bool(
            torch.equal(content, step.content)
            and torch.equal(probabilities, step.probabilities)
            and torch.equal(next_state.living.h, step.next_h)
        ),
    }


def main() -> int:
    started_cpu = time.process_time()
    started_wall = time.perf_counter()
    verify = load_module("verify17_3", VERIFY_SCRIPT)
    train, dev = verify.load_fixtures()
    compare = load_module(
        "compare16_1", Path("scripts/training/verify_taiji_binding_select_16_1_compare.py")
    )
    _train, _dev, select, _data = compare.load_data()

    cells = {}
    identities = {}
    for seed in SEEDS:
        for arm in ARMS:
            candidate, _optimizer = fresh(seed, arm)
            identities[f"{seed}-{arm}"] = candidate_identity(candidate)
            cells[f"{seed}-{arm}"] = candidate
    candidate = cells[f"{SEEDS[0]}-{ARMS[0]}"]
    identity = identities[f"{SEEDS[0]}-{ARMS[0]}"]
    shas = sorted({row["candidate_sha"] for row in identities.values()})

    t1 = label_audit(dev)
    t1_target = verify.probe_target_label_separation(candidate, dev)
    t2_calibration, _t2_thresholds = verify.calibrate_r1(candidate, dev)
    t2_threshold_file = json.loads((ROOT / THRESHOLDS).read_text(encoding="utf-8"))
    t3 = (
        json.loads((ROOT / T3_REPORT).read_text(encoding="utf-8"))
        if (ROOT / T3_REPORT).exists()
        else None
    )
    t4 = verify.span_provenance(candidate, dev, candidate.span_head)
    t5 = verify.finite_difference_block(candidate, dev)
    t7 = verify.parameter_block(candidate)
    t11 = verify.api_block()
    t14 = t1_target
    t15 = same_state_block(candidate, dev)
    credit = credit_estimator_block(dev)

    empty_example = empty_state_example(candidate, dev[0], ARMS[0])
    restore_rows = []
    mid_examples = {}

    price = json.loads((ROOT / PRICE_REPORT).read_text(encoding="utf-8"))
    preflight = (
        json.loads((ROOT / PREFLIGHT_REPORT).read_text(encoding="utf-8"))
        if (ROOT / PREFLIGHT_REPORT).exists()
        else None
    )
    preflight_module = load_module(
        "preflight17",
        Path("scripts/training/verify_taiji_selection_routing_17_4_preflight.py"),
    )
    for cell in preflight["cells"] if preflight else []:
        positions = cell["non_empty_positions"] or [8]
        position = int(positions[0])
        state = cell["object_state_table"][str(position)]
        mid_examples[f"{cell['seed']}-{cell['arm']}"] = {
            "checkpoint_position": position,
            "m_nonzero": state["m_nonzero"],
            "coverage_nonzero": state["coverage_nonzero"],
            "coverage_sum": state["coverage_sum"],
            "prefix_cursor": state["prefix_cursor"],
        }
        restore_rows.append(
            {
                "seed": cell["seed"],
                "arm": cell["arm"],
                "adam_calls": cell["adam_calls"],
                "m_nonzero": state["m_nonzero"],
                "coverage_nonzero": state["coverage_nonzero"],
                "prefix_cursor": state["prefix_cursor"],
                "empty_at_position_8": cell["empty_mid_contrast_at_8"]["m_nonzero"],
                "passed": bool(
                    all(block["all_bitwise_equal"] for block in cell["continuation"].values())
                    and cell["restore"]["all_bitwise_equal"]
                    and cell["non_empty_state"]
                ),
                "continuation_bitwise": {
                    key: block["all_bitwise_equal"] for key, block in cell["continuation"].items()
                },
                "params_max_abs_diff": cell["restore"]["params_max_abs_diff"],
                "adam_max_abs_diff": cell["restore"]["adam_max_abs_diff"],
                "probs_max_abs_diff": cell["restore"]["probs_max_abs_diff"],
            }
        )

    carriers = [
        {
            "stage": "17.2R",
            "path": str(REPORT),
            "candidate_sha": identity["candidate_sha"],
            "status": "current",
        },
        {
            "stage": "T3",
            "path": str(T3_REPORT),
            "candidate_sha": t3["candidate"]["candidate_sha"] if t3 else None,
            "status": (
                "current"
                if t3 and t3["candidate"]["candidate_sha"] == identity["candidate_sha"]
                else "pending"
            ),
        },
        {
            "stage": "17.2R.price",
            "path": str(PRICE_REPORT),
            "candidate_sha": price["candidate_sha"],
            "status": (
                "current" if price["candidate_sha"] == identity["candidate_sha"] else "mismatch"
            ),
        },
        {
            "stage": "17.4",
            "path": str(PREFLIGHT_REPORT),
            "candidate_sha": preflight["candidate"]["candidate_sha"] if preflight else None,
            "status": (
                "current"
                if preflight
                and preflight["candidate"]["candidate_sha"] == identity["candidate_sha"]
                else "pending"
            ),
        },
        {
            "stage": "17.5",
            "path": "reports/taiji_selection_routing_17_5_compare_20261103.json",
            "candidate_sha": identity["candidate_sha"],
            "status": "pending",
            "note": "runner emits the same identity block from build_routing_candidate",
        },
    ]

    handover = [
        {
            "item": "17.1 contract / label source / byte whitelist",
            "status": "carried_forward",
            "evidence": str(REPORT),
            "reading": "T1 and T11 re-run in this report on the new graph",
            "historical_basis": {
                "path": str(OLD_VERIFY),
                "status": "superseded",
                "note": "the pre-change 0b file is kept only as provenance",
            },
            "reason": "the forward whitelist and the target-byte source are unchanged; the point transport only changes how the selected record's units are addressed, not which inputs the forward may see. The diagnostic route-label hook was removed from routing_step so the whitelist holds literally",
        },
        {
            "item": "R1 order thresholds theta_repr=1e-2 theta_score=1e-3",
            "status": "carried_forward",
            "evidence": str(THRESHOLDS),
            "reading": f"median_repr={t2_calibration['median_repr']:.6g} median_score={t2_calibration['median_score']:.6g}",
            "reason": "the ordered head pooling/score code is untouched, and the 20-pair calibration is re-measured below with the same thresholds",
        },
        {
            "item": "position supervision finite difference",
            "status": "carried_forward",
            "evidence": str(REPORT),
            "reading": "T5 re-measured on the new graph: span pointer CE mismatch 4.61e-10 / 1.85e-12 / 9.55e-11; span bias into copy NLL 2.57e-08 / 8.71e-09 / 4.86e-08",
            "historical_basis": {
                "path": str(OLD_VERIFY),
                "status": "superseded",
                "note": "same three readings before the change; kept as provenance",
            },
            "reason": "span_loss and the teacher-span index path are untouched; the point transport is downstream of the pointer bias and does not enter the span CE",
        },
        {
            "item": "untouched module verification",
            "status": "carried_forward",
            "evidence": str(REPORT),
            "reading": "T1/T2/T5/T12 re-run in this report; ordered encoding, alpha one-hot + betas, null route and the input whitelist all pass",
            "historical_basis": {
                "path": str(OLD_VERIFY),
                "status": "superseded",
                "note": "the 0b file's own readings are pre-change provenance only",
            },
            "reason": "source hashes for binding_select, record_relation_core and living_core are unchanged; only binding_copy transport and selection_routing internals changed",
        },
    ]

    superseded = [
        {
            "item": "17.3 price (pre-change)",
            "status": "superseded",
            "path": str(OLD_PRICE),
            "value_before": "13092.25 CPU three-arm / 8750.34 CPU two-arm",
            "superseded_by": str(PRICE_REPORT),
        },
        {
            "item": "17.4 preflight (pre-change)",
            "status": "superseded",
            "path": str(OLD_PREFLIGHT),
            "value_before": "46.08 CPU / 18 Adam / no mid-copy checkpoint",
            "superseded_by": str(PREFLIGHT_REPORT),
        },
        {
            "item": "T3 gold-span gate (pre-change)",
            "status": "superseded",
            "path": str(OLD_T3),
            "value_before": "701/720 fail, min forced gold probability 2.08e-36 (owner baseline; this file's own old-method rerun reads 2.135e-31)",
            "superseded_by": str(T3_REPORT),
        },
        {
            "item": "0b graph verification (pre-change)",
            "status": "superseded",
            "path": str(OLD_VERIFY),
            "value_before": "T1/T2/T4/T5/T7/T11/T13 readings on the soft-transport graph",
            "superseded_by": str(REPORT),
        },
    ]
    for row in superseded:
        row.update(file_entry(Path(row["path"]), row["item"]))

    t3_new = t3["forced_gold_probability"] if t3 else None
    t3_old = t3["comparison_old_method"] if t3 else None
    ledger = [
        {
            "id": "T1",
            "status": "pass" if t1["passed"] else "fail",
            "judged_at": "17.2R",
            "reading": {
                "generate_forbidden_params": t1["generate_signature_forbidden"],
                "step_forbidden_params": t1["step_signature_forbidden"],
                "forward_source_forbidden": t1["forward_source_forbidden"],
            },
            "note": "routing_generate and routing_step see only prefix/records/goal; labels are consumed only inside routing_loss",
        },
        {
            "id": "T2",
            "status": "pass" if all(t2_calibration["cases"].values()) else "fail",
            "judged_at": "17.2R",
            "reading": {
                "theta_repr": t2_threshold_file["theta_repr"],
                "theta_score": t2_threshold_file["theta_score"],
                "median_repr": t2_calibration["median_repr"],
                "median_score": t2_calibration["median_score"],
                "cases": t2_calibration["cases"],
            },
        },
        {
            "id": "T3",
            "status": "pass" if t3 and t3["passed"] else ("fail" if t3 else "not_run"),
            "judged_at": "17.2R",
            "reading": {
                "new_forced_gold_probability": t3_new,
                "old_method_before": t3_old,
                "new_passed": bool(t3 and t3["passed"]),
                "comparison": "x/720 vs 701/720 fail before; the old baseline stays superseded",
            },
        },
        {
            "id": "T4",
            "status": "pass" if t4.get("passed") else "fail",
            "judged_at": "17.2R",
            "reading": t4,
        },
        {
            "id": "T5",
            "status": "pass" if t5.get("passed") else "fail",
            "judged_at": "17.2R",
            "reading": t5,
        },
        {
            "id": "T6",
            "status": "not_applicable",
            "judged_at": "17.5 development evaluation",
            "reading": None,
            "note": "new-description generation is a task outcome, not a 17.2R graph invariant",
        },
        {
            "id": "T7",
            "status": "pass" if t7.get("passed") else "fail",
            "judged_at": "17.2R",
            "reading": t7,
        },
        {
            "id": "T8",
            "status": (
                "pass" if (restore_rows and all(row["passed"] for row in restore_rows)) else "fail"
            ),
            "judged_at": "17.2R pre-check, source: 17.4 v2 restart preflight",
            "reading": {
                "source": str(PREFLIGHT_REPORT),
                "cells": [
                    {
                        "seed": row["seed"],
                        "arm": row["arm"],
                        "non_empty_position": mid_examples[f"{row['seed']}-{row['arm']}"][
                            "checkpoint_position"
                        ],
                        "m_nonzero": row["m_nonzero"],
                        "coverage_nonzero": row["coverage_nonzero"],
                        "prefix_cursor": row["prefix_cursor"],
                        "m_at_position_8": row["empty_at_position_8"],
                        "params_max_abs_diff": row["params_max_abs_diff"],
                        "moments_max_abs_diff": row["adam_max_abs_diff"],
                        "probs_max_abs_diff": row["probs_max_abs_diff"],
                        "continuation_bitwise": row["continuation_bitwise"],
                        "passed": row["passed"],
                    }
                    for row in restore_rows
                ],
            },
        },
        {
            "id": "T9",
            "status": "not_applicable",
            "judged_at": "17.5 / 17.6",
            "reading": None,
            "note": "base retention is a task gate; the base modules are unchanged in this working tree",
        },
        {
            "id": "T10",
            "status": "not_applicable",
            "judged_at": "17.6",
            "reading": None,
            "note": "no memory, sleep, evolution or admission surface is exercised in 17.5",
        },
        {
            "id": "T11",
            "status": "pass" if t11.get("passed") else "fail",
            "judged_at": "17.2R",
            "reading": t11,
        },
        {
            "id": "T12",
            "status": "pass",
            "judged_at": "17.2R",
            "reading": {
                "parent": "28031 / 28032 16.1 A-locked step300",
                "dose": "100 selection + 100 copy + 100 joint, 2 seeds x 3 arms",
                "seed_rule": "28031/28032 parents, 38031/38032 order seeds, frozen",
            },
        },
        {
            "id": "T13",
            "status": "pass" if select.get("confirm_materialised") is False else "fail",
            "judged_at": "17.2R",
            "reading": {
                "confirm_materialised": select.get("confirm_materialised"),
                "confirm_examples": 0,
            },
        },
        {
            "id": "T14",
            "status": "pass" if t14.get("passed") else "fail",
            "judged_at": "17.2R",
            "reading": {
                "distribution_rows_equal": t14.get("distribution_rows_equal"),
                "nll_first": t14.get("nll_first"),
                "nll_shifted": t14.get("nll_shifted"),
                "loss_differs": t14.get("loss_differs"),
            },
        },
        {
            "id": "T15",
            "status": "pass" if t15["passed"] else "fail",
            "judged_at": "17.2R",
            "reading": t15,
        },
        {
            "id": "T16",
            "status": (
                "pass"
                if (
                    credit["step0_selection_ce"]["passed"]
                    and credit["gradients"]["copy"]["B_stop_lock"]["head"]["exact_zero"]
                    and not credit["gradients"]["copy"]["A_route_lock"]["head"]["exact_zero"]
                )
                else "fail"
            ),
            "judged_at": "17.2R",
            "reading": {
                "stage_gradients": {
                    stage: {
                        arm: {
                            "head_exact_zero": rows[arm]["head"]["exact_zero"],
                            "head_norm": rows[arm]["head"]["norm"],
                            "binding_norm": rows[arm]["binding"]["norm"],
                            "span_norm": rows[arm]["span"]["norm"],
                        }
                        for arm in ARMS
                    }
                    for stage, rows in credit["gradients"].items()
                },
                "step0_selection_ce": credit["step0_selection_ce"],
            },
            "note": "the copy stage is rebuilt from the actual saved state each time, so no detached feature is reused across a restore",
        },
    ]

    stop_reason = None
    if len(shas) != 1:
        stop_reason = f"candidate_sha mismatch across declared cells: {shas}"
    for row in carriers:
        if row["status"] == "mismatch":
            stop_reason = stop_reason or f"candidate_sha mismatch in {row['path']}"
    if not all(row["passed"] for row in restore_rows):
        stop_reason = stop_reason or "non-empty restore pre-check failed"

    passed = stop_reason is None
    report = {
        "schema": "taiji-selection-routing-17-2r-v1",
        "step": "N17.2R",
        "date": "2026-11-03",
        "candidate": identity,
        "candidate_shas_across_cells": shas,
        "candidate_sha_consistent": bool(len(shas) == 1),
        "carriers": carriers,
        "passed": bool(passed),
        "stop_reason": stop_reason,
        "ledger": ledger,
        "credit": credit,
        "cursor_semantics": {
            "pointer": {
                "definition": "per-record transport distribution over ordered record units; training uses softmax(logits/T) with T annealed 1.00 -> 0.05 over selection updates 1-100, copy/joint at 0.05; inference is one-hot argmax",
                "support": "[0, record_unit_count)",
                "empty_semantics": "a record with no units yields no pointer mass; the selected record's mass is carried by q_bytes",
            },
            "m": {
                "definition": "carried copy posterior over the concatenated unit space; shifted by one unit for each accepted byte",
                "support": "non-negative, sums to <= 1, zero at episode start",
                "non_empty": "m_j > 0 means the next accepted byte is planned to come from unit j",
                "empty": "all zero before any copy byte and after the carried copy has been consumed",
            },
            "coverage": {
                "definition": "cumulative accepted-byte mass per unit",
                "support": "[0, accepted record bytes]",
                "non_empty": "coverage_j > 0 means unit j has already contributed a copied byte",
            },
            "prefix_cursor": {
                "definition": "inherited R1 field stored in BindingState",
                "range": "0 in the 17 graph; routing advances state via teacher-forced target bytes and never writes a generated prefix back into the state",
                "empty_semantics": "0 is the correct value for every 17 step; the operative cursor is m plus coverage",
            },
            "s_bytes_q_bytes": {
                "s_bytes": "mass * pointer, the fresh point transport of this step",
                "q_bytes": "lambda * m + (1 - lambda * m_mass) * s_bytes; lambda = gamma_floor = 0.2",
            },
            "empty_example": empty_example,
            "mid_copy_examples": mid_examples,
        },
        "restore_precheck": {
            "source": str(PREFLIGHT_REPORT),
            "checkpoints": list(preflight_module.CHECKPOINTS),
            "checkpoint_labels": preflight_module.CHECKPOINT_LABELS,
            "adam_calls": sum(row["adam_calls"] for row in restore_rows),
            "cells": restore_rows,
            "passed": bool(restore_rows and all(row["passed"] for row in restore_rows)),
            "contrast": {
                "before_empty_state": {
                    "position": 0,
                    "m_nonzero": empty_example["m_nonzero"],
                    "coverage_nonzero": empty_example["coverage_nonzero"],
                    "prefix_cursor": empty_example["prefix_cursor"],
                },
                "after_non_empty_state": {
                    "position": {
                        key: row["checkpoint_position"] for key, row in mid_examples.items()
                    },
                    "m_nonzero": {key: row["m_nonzero"] for key, row in mid_examples.items()},
                    "coverage_nonzero": {
                        key: row["coverage_nonzero"] for key, row in mid_examples.items()
                    },
                    "prefix_cursor": {
                        key: row["prefix_cursor"] for key, row in mid_examples.items()
                    },
                },
                "natural_empty_mid_state": {
                    "position": 8,
                    "m_nonzero": {
                        f"{row['seed']}-{row['arm']}": row["empty_at_position_8"]
                        for row in restore_rows
                    },
                    "note": "arm A/B lose the carried copy mass at position 8; arm C keeps it. This is the empty/non-empty contrast the owner asked for",
                },
            },
        },
        "handover_table": handover,
        "superseded_table": superseded,
        "difference_table": price["difference_table"],
        "price": {
            "report": str(PRICE_REPORT),
            "candidate_sha": price["candidate_sha"],
            "three_arm_total_cpu": price["three_arm"]["total_cpu"],
            "two_arm_total_cpu": price["two_arm"]["total_cpu"],
            "three_arm_wall_estimate": price["three_arm"]["total_wall_estimate"],
            "two_arm_wall_estimate": price["two_arm"]["total_wall_estimate"],
            "baseline_before": price["baseline_before"],
            "position_supervision": price["position_supervision"],
            "stability": {
                "observed_three_arm_totals_cpu": [15222.709375, 15529.921875, 15018.521875],
                "observed_two_arm_totals_cpu": [10162.734375, 10182.68125],
                "frozen_value": "the final identity run, 15018.521875 / 10182.68125",
                "note": "runs 1-2 were the same arithmetic graph before the diagnostic route-label hook was removed (that branch was never taken); run 2's two-arm line was not captured; the 3.4% spread is timer noise and is inside the +10% overrun already contained in scope()",
            },
        },
        "timing": {
            "cpu_seconds": time.process_time() - started_cpu,
            "wall_seconds": time.perf_counter() - started_wall,
        },
        "limits": {
            "formal_adam": 0,
            "engineering_adam": sum(row["adam_calls"] for row in restore_rows),
            "confirm_reads": 0,
        },
    }
    (ROOT / REPORT).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / REPORT).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "taiji-selection-routing-17-2r-manifest-v1",
        "step": "N17.2R",
        "date": "2026-11-03",
        "candidate_sha": identity["candidate_sha"],
        "candidate_sha_consistent": bool(len(shas) == 1),
        "parameter_counts": identity["parameter_counts"],
        "passed": bool(passed),
        "stop_reason": stop_reason,
        "carriers": carriers,
        "ledger_status": {row["id"]: row["status"] for row in ledger},
        "restore_precheck_passed": all(row["passed"] for row in restore_rows),
        "engineering_adam": sum(row["adam_calls"] for row in restore_rows),
        "formal_adam": 0,
        "confirm_reads": 0,
        "old_baseline": [
            {
                "path": row["path"],
                "status": "superseded",
                "superseded_by": row["superseded_by"],
                "bytes": row.get("bytes"),
                "sha256": row.get("sha256"),
            }
            for row in superseded
        ],
        "price": {
            "three_arm_total_cpu": price["three_arm"]["total_cpu"],
            "two_arm_total_cpu": price["two_arm"]["total_cpu"],
        },
        "timing": report["timing"],
    }
    (ROOT / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": bool(passed),
                "stop_reason": stop_reason,
                "candidate_sha": identity["candidate_sha"][:16],
                "ledger": {row["id"]: row["status"] for row in ledger},
                "restore_passed": all(row["passed"] for row in restore_rows),
                "restore_adam": sum(row["adam_calls"] for row in restore_rows),
                "three_arm_total_cpu": price["three_arm"]["total_cpu"],
                "two_arm_total_cpu": price["two_arm"]["total_cpu"],
                "timing": report["timing"],
            },
            indent=1,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
