"""W3 write-timing retest: per-turn write policy, 0 parameters 0 structure.

Executes the W3 package pinned by
plans/manifests/taiji_w3_write_timing_feature_card_20261108.json under the
unchanged v3 criteria: the redesigned write path (content keys + tied reads +
content-conditioned values) at initialization WITH the W3 per-turn policy.
Binding instance = CF1 (the feedback-capable arm); B_EH is the arm-definition
control (its empty deployed write schedule is by design, not a timing defect).

The forced condition is expected to stay all-green (the policy flag does not
touch the forced path); the ACTUAL condition is re-run under the W3 policy.
If the actual key gates pass, the timing verdict no longer fires and
expressive_capacity_present is judged on the forced-side gates.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "training"))

import torch  # noqa: E402

torch.set_num_threads(1)

import probe_taiji_f_expressive_capacity_20261108 as fec1  # noqa: E402
import probe_taiji_fec1_w1a_retest_20261108 as w1a  # noqa: E402
import probe_taiji_fec1_wv2_retest_20261108 as wv2  # noqa: E402  (installs V2Capture)
import probe_taiji_package20_4c_r2_formal_comparison as r2  # noqa: E402

from taiji import rule_learning_20_1 as fixture  # noqa: E402
from taiji.rule_learning_20_1_net import (
    enable_content_path_v2,
    enable_write_timing_per_turn,
)  # noqa: E402

PREREG_V3 = ROOT / "plans/manifests/taiji_f_expressive_capacity_prereg_v3_20261108.json"
CRITERIA_V3_SHA256 = "875a55b9ee4e75ffc17a43aab4ff6498be8f0966fb696af3986902cb4d77ed3f"
CARD = ROOT / "plans/manifests/taiji_w3_write_timing_feature_card_20261108.json"
CARD_SHA256 = "89368c853609be2dfcd35c83772958882b0b1cb9b6d32dd3863f0f265a80b185"
REPORT = ROOT / "reports/taiji_w3_write_timing_20261108.json"
EXEC_MANIFEST = ROOT / "plans/manifests/taiji_w3_write_timing_execution_20261108.json"
PROGRESS = ROOT / "reports/taiji_w3_write_timing_20261108_progress.jsonl"
CONTENT_SEED = wv2.CONTENT_SEED

START_CPU = time.process_time()
START_WALL = time.time()
BUDGET = {"cpu_stop": 2000.0, "wall_cap": 3000.0}


class BudgetStop(RuntimeError):
    pass


class EngineeringStop(RuntimeError):
    pass


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_progress(record: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def cost() -> tuple[float, float]:
    return time.process_time() - START_CPU, time.time() - START_WALL


def check_budget() -> None:
    cpu, wall = cost()
    if cpu >= BUDGET["cpu_stop"] or wall >= BUDGET["wall_cap"]:
        raise BudgetStop(f"budget stop at cpu={cpu:.2f} wall={wall:.2f}")


def evaluate_instance(graph, world: dict, criteria: dict, timing_policy: bool) -> dict:
    """Forced replay (policy-independent) + actual replay under the graph's
    deployed policy; analysis identical to the v3 retest."""
    groups = graph.subject.config.groups
    heads = graph.subject.config.fast_heads
    off = torch.zeros((groups, heads), dtype=torch.bool)
    on = torch.ones((groups, heads), dtype=torch.bool)
    release_actual = torch.full((groups, heads), bool(graph.fast_enabled), dtype=torch.bool)

    actual = w1a.replay(graph, world, off if not timing_policy else release_actual, release_actual)
    actual_rows = [
        row for row in actual["rows"] if row["correction_surface"] and row["released_lines"] > 0
    ]
    if len(actual_rows) != 22:
        raise EngineeringStop(f"expected 22 actual faces, got {len(actual_rows)}")
    face_turns = {row["turn"] for row in actual_rows}
    timing_coverage = len(set(actual["face_written_turns"]) & face_turns) / len(actual_rows)

    forced = w1a.replay(graph, world, on, on)
    forced_rows = [
        row for row in forced["rows"] if row["correction_surface"] and row["released_lines"] > 0
    ]
    if len(forced_rows) != 22:
        raise EngineeringStop(f"expected 22 forced faces, got {len(forced_rows)}")
    gold = [row["gold"] for row in forced_rows]

    stacks = fec1.stack_fields(forced["face_events"], groups, heads)
    key_result = fec1.key_gates({name: f["keys"] for name, f in stacks.items()}, criteria)
    align = fec1.query_alignment(stacks)
    value_ranks = {name: fec1.effective_rank(f["values"])[0] for name, f in stacks.items()}
    query_ranks = {name: fec1.effective_rank(f["queries"])[0] for name, f in stacks.items()}
    cap = fec1.capacity_of(stacks)

    f_matrix = np.stack([row["fast"].double().reshape(-1).numpy() for row in forced_rows])
    dof = int(np.unique(f_matrix, axis=0).shape[0])
    m_by_stack = forced["final_fast"].double().numpy().reshape(groups, heads, 32, 32)

    face_readouts, mfinal_readouts = {}, {}
    for name, fields in stacks.items():
        keys, values = fields["keys"], fields["values"]
        m_face = values.T @ keys
        face_readouts[name] = fec1.closed_form_stack(m_face, keys, values, gold)
        mfinal_readouts[name] = fec1.closed_form_stack(
            m_by_stack[int(name[1]), int(name[3])], keys, values, gold
        )
    best_face = max(face_readouts, key=lambda n: face_readouts[n]["decision_bit_accuracy"])
    binding_accuracy = float(face_readouts[best_face]["decision_bit_accuracy"])
    best_bits = fec1.per_face_bits(face_readouts[best_face], gold)
    controls = wv2.w1r2_controls(forced_rows, f_matrix, criteria)
    margin = binding_accuracy - controls["strongest"]

    # ACTUAL-side keys captured under the deployed (W3) policy
    actual_stacks = fec1.release_stack_fields(actual["release_events"], groups, heads)
    if timing_policy:
        # under the W3 policy the face events themselves wrote: capture them
        # from the actual replay's face events, FILTERED to faces whose write
        # was actually COMMITTED (F changed) - trace write_keys exist even for
        # disallowed calls (computed-but-not-committed), which would otherwise
        # let the B_EH control arm pass on phantom keys
        written_faces = set(actual["face_written_turns"]) & face_turns
        committed_events = {t: e for t, e in actual["face_events"].items() if t in written_faces}
        actual_face_stacks = (
            fec1.stack_fields(committed_events, groups, heads) if committed_events else {}
        )
        actual_key_result = (
            fec1.key_gates({name: f["keys"] for name, f in actual_face_stacks.items()}, criteria)
            if actual_face_stacks
            else {"pass": False, "binding": {}, "per_stack": {}}
        )
        actual_key_protocol = f"committed_face_writes({len(committed_events)}/22)"
    else:
        actual_key_result = (
            fec1.key_gates({name: f["keys"] for name, f in actual_stacks.items()}, criteria)
            if actual_stacks
            else {"pass": False, "binding": {}, "per_stack": {}}
        )
        actual_key_protocol = "release_lines"

    extras = {
        "query_alignment": align["median_all"],
        "query_effective_rank": min(query_ranks.values()),
        "value_effective_rank": min(value_ranks.values()),
        "capacity": cap["capacity"],
        "closed_form_accuracy": binding_accuracy,
        "margin": margin,
        "effective_dof": dof,
    }
    boundaries = fec1.boundary_flags(key_result["binding"], extras, criteria["gates"])
    verdict = fec1.checkpoint_verdict(
        key_result["pass"], extras, actual_key_result["pass"], boundaries, dof, len(forced_rows)
    )
    result = {
        "verdict": verdict,
        "timing_policy": bool(timing_policy),
        "timing_coverage_actual": timing_coverage,
        "actual_written_lines": actual["written_lines"],
        "actual_turn_write_faces": len(actual["face_written_turns"]),
        "key_gates_forced": key_result,
        "key_gates_actual": {
            "pass": actual_key_result.get("pass", False),
            "binding": actual_key_result.get("binding", {}),
            "protocol": actual_key_protocol,
        },
        "query_alignment": align,
        "value_effective_ranks": value_ranks,
        "query_effective_ranks": query_ranks,
        "capacity": cap,
        "closed_form": {
            "binding_protocol": "face_isolated",
            "binding_stack": best_face,
            "binding_decision_bit_accuracy": binding_accuracy,
            "binding_event_id_accuracy": face_readouts[best_face]["event_id_accuracy"],
            "face_isolated_per_stack": {
                name: {
                    "decision_bit_accuracy": item["decision_bit_accuracy"],
                    "event_id_accuracy": item["event_id_accuracy"],
                }
                for name, item in face_readouts.items()
            },
            "m_final_diagnostic": {
                "best_decision_bit_accuracy": float(
                    max(item["decision_bit_accuracy"] for item in mfinal_readouts.values())
                ),
                "best_event_id_accuracy": float(
                    max(item["event_id_accuracy"] for item in mfinal_readouts.values())
                ),
            },
            "bootstrap": fec1.paired_bootstrap(best_bits, fec1.BOOTSTRAP_RESAMPLES),
        },
        "controls": controls,
        "margin": margin,
        "effective_dof": {"distinct_f_vectors": dof},
        "boundary_flags": boundaries,
    }
    # journal/bundle increment accounting (W3 exit requirement)
    result["increment_accounting"] = {
        "parameter_delta": 0,
        "structure_delta": 0,
        "journal_len_per_face": [row["journal_len"] for row in actual_rows[:6]],
        "state_tensors": "unchanged (policy flag only)",
    }
    return result


def policy_equivalence(graph, world: dict) -> dict:
    """graph.run_turn with write_timing_per_turn=True must reproduce the
    driver's W3-policy actual replay bit-identically (deployed equivalence).
    Per-turn F vectors are the STRONGER signal for the write path (emitted
    bytes do not depend on the write mask; F does)."""
    reference_fast = []
    state = graph.initial_state(world["response"]["world_id"])
    released: list[dict] = []
    for turn in world["response"]["turns"]:
        number = int(turn["turn"])
        result = graph.run_turn(state, turn, released, write_enabled=True, vocab=None, now=number)
        state = result["state"]
        for release in world["response"]["releases"]:
            if int(release["after_turn"]) != number:
                continue
            state, _info = graph.release(state, release, released, write_enabled=True, now=number)
            released.extend(
                {
                    "source_id": item["source_id"],
                    "feedback": item["feedback"],
                    "version": item["version"],
                    "after_turn": number,
                }
                for item in release["items"]
            )
        # capture AFTER trailing releases, matching the driver's row capture point
        reference_fast.append(state.living.fast.detach().cpu().clone())
    reference_final = state.living.fast.detach().cpu().clone()
    groups = graph.subject.config.groups
    heads = graph.subject.config.fast_heads
    gate = torch.full((groups, heads), bool(graph.fast_enabled), dtype=torch.bool)
    mine = w1a.replay(graph, world, gate, gate)
    mismatches = [
        index
        for index, (left, right) in enumerate(
            zip(reference_fast, (r["fast"] for r in mine["rows"]), strict=False)
        )
        if not torch.equal(left, right)
    ]
    final_delta = float((reference_final - mine["final_fast"]).abs().max())
    return {
        "ok": bool(not mismatches and final_delta == 0.0),
        "f_mismatch_turns": mismatches,
        "final_fast_max_abs_delta": final_delta,
    }


def main() -> int:
    runner_sha = fec1.sha256_file(Path(__file__).resolve())
    try:
        card = json.loads(CARD.read_text(encoding="utf-8"))
        if fec1.sha256_file(CARD) != CARD_SHA256:
            raise EngineeringStop("W3 feature card sha mismatch")
        if card["criteria_v3_sha256"] != CRITERIA_V3_SHA256:
            raise EngineeringStop("W3 card points at different criteria")
        prereg = json.loads(PREREG_V3.read_text(encoding="utf-8"))
        raw = json.dumps(
            prereg["criteria"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
        if hashlib.sha256(raw).hexdigest() != CRITERIA_V3_SHA256:
            raise EngineeringStop("v3 criteria sha mismatch")
        criteria = prereg["criteria"]
    except EngineeringStop as error:
        append_progress({"stage": "preflight", "ok": False, "reason": str(error)})
        write_json(
            REPORT,
            {
                "package": "W3",
                "status": "engineering_stop",
                "verdict": "engineering_stop",
                "reason": str(error),
                "runner_sha256": runner_sha,
                "readings_produced": 0,
            },
        )
        print(json.dumps({"verdict": "engineering_stop", "reason": str(error)}))
        return 0

    append_progress(
        {
            "stage": "preflight",
            "ok": True,
            "card_sha256": CARD_SHA256,
            "criteria_v3_sha256": CRITERIA_V3_SHA256,
            "runner_sha256": runner_sha,
            "cpu": round(cost()[0], 3),
        }
    )
    world = fixture.build_world(20261105, 0)
    centroid = wv2.probe_face_centroid(world)
    parent_payload = torch.load(r2.BUNDLE, map_location="cpu", weights_only=True)
    gates = parent_payload.get("growth_gates")
    gate = dict(gates[0]) if gates else None
    candidate = r2.load_routing_bundle(r2.BUNDLE)[0]

    results: list[dict] = []
    try:
        # parity gate: flag OFF must be bit-identical to graph.run_turn
        parity_graph = r2.build_graph(candidate, gate, "CF1")
        parity_graph.subject.eval()
        parity = fec1.driver_equivalence(parity_graph, world)
        append_progress({"stage": "parity", "ok": parity["ok"], "cpu": round(cost()[0], 3)})
        if not parity["ok"]:
            raise EngineeringStop(f"disabled-graph parity failed: {parity}")

        for arm in ("CF1", "B_EH"):
            graph = r2.build_graph(candidate, gate, arm)
            graph.subject.eval()
            enable_content_path_v2(graph, seed=CONTENT_SEED, centroid=centroid)
            enable_write_timing_per_turn(graph)
            graph.subject.eval()
            if arm == "CF1":
                policy_eq = policy_equivalence(graph, world)
                append_progress(
                    {
                        "stage": "policy_equivalence",
                        "ok": policy_eq["ok"],
                        "cpu": round(cost()[0], 3),
                    }
                )
                if not policy_eq["ok"]:
                    raise EngineeringStop(f"W3 policy deployment equivalence failed: {policy_eq}")
            result = evaluate_instance(graph, world, criteria, timing_policy=True)
            result["arm"] = arm
            result["binding"] = arm == "CF1"
            results.append(result)
            append_progress(
                {
                    "stage": "instance",
                    "arm": arm,
                    "binding": result["binding"],
                    "verdict": result["verdict"],
                    "actual_keys_pass": result["key_gates_actual"]["pass"],
                    "coverage": result["timing_coverage_actual"],
                    "cpu": round(cost()[0], 3),
                }
            )
    except BudgetStop as error:
        append_progress({"stage": "budget_stop", "reason": str(error)})
        write_json(
            REPORT,
            {
                "package": "W3",
                "status": "budget_stop",
                "verdict": "inconclusive",
                "reason": str(error),
                "results": results,
            },
        )
        print(json.dumps({"verdict": "inconclusive", "status": "budget_stop"}))
        return 0
    except EngineeringStop as error:
        append_progress({"stage": "engineering_stop", "reason": str(error)})
        write_json(
            REPORT,
            {
                "package": "W3",
                "status": "engineering_stop",
                "verdict": "engineering_stop",
                "reason": str(error),
                "runner_sha256": runner_sha,
                "readings_produced": len(results),
            },
        )
        print(json.dumps({"verdict": "engineering_stop", "reason": str(error)}))
        return 0

    binding = next(entry for entry in results if entry["binding"])
    control = next(entry for entry in results if not entry["binding"])
    cpu, wall = cost()
    write_json(
        REPORT,
        {
            "schema": "taiji-w3-write-timing-v1",
            "package": "W3",
            "date": "2026-11-08",
            "status": "completed",
            "verdict": binding["verdict"],
            "verdict_basis": "binding instance = CF1 (the feedback-capable arm); B_EH reported as the arm-definition control per the W3 card",
            "binding_instance": "CF1",
            "policy": card["policy_change"]["definition"],
            "discipline": {
                "training": 0,
                "optimizer_steps": 0,
                "confirm_reads": 0,
                "parameter_delta": 0,
                "structure_delta": 0,
                "criteria_v3_sha256": CRITERIA_V3_SHA256,
            },
            "provenance": {
                "card_sha256": CARD_SHA256,
                "prereg_v3_manifest": "plans/manifests/taiji_f_expressive_capacity_prereg_v3_20261108.json",
                "write_path_card": "plans/manifests/taiji_fec1_wv2_retest_feature_card_20261108.json",
                "runner_sha256": runner_sha,
                "parity": {"ok": True},
                "policy_deployment_equivalence": {"ok": True},
            },
            "binding_result": binding,
            "control_result": control,
            "cost": {"cpu_seconds": round(cpu, 3), "wall_seconds": round(wall, 3)},
            "scope": (
                "Engineering-package verdict under the W3 per-turn write policy; "
                "expressive_capacity_present, if reached, is an architecture-requirement "
                "input - NOT training admission, NOT a capability claim; the task-gain "
                "question is a separate window per the W1 branch plan."
            ),
        },
    )
    write_json(
        EXEC_MANIFEST,
        {
            "schema": "taiji-w3-write-timing-execution-manifest-v1",
            "package": "W3",
            "date": "2026-11-08",
            "card": "plans/manifests/taiji_w3_write_timing_feature_card_20261108.json",
            "card_sha256": CARD_SHA256,
            "criteria_v3_sha256": CRITERIA_V3_SHA256,
            "report": "reports/taiji_w3_write_timing_20261108.json",
            "report_sha256": fec1.sha256_file(REPORT),
            "runner": "scripts/training/probe_taiji_w3_write_timing_20261108.py",
            "runner_sha256": runner_sha,
            "submitted_runner_sha256": runner_sha,
            "verdict": binding["verdict"],
            "cost": {"cpu_seconds": round(cpu, 3), "wall_seconds": round(wall, 3)},
        },
    )
    print(
        json.dumps(
            {
                "verdict": binding["verdict"],
                "actual_keys_pass": binding["key_gates_actual"]["pass"],
                "coverage": binding["timing_coverage_actual"],
                "forced_all_green": binding["key_gates_forced"]["pass"]
                and binding["query_alignment"]["median_all"] >= 0.5
                and min(binding["value_effective_ranks"].values()) >= 3
                and binding["margin"] >= 0.2,
                "margin": round(binding["margin"], 4),
                "cpu": round(cpu, 2),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
