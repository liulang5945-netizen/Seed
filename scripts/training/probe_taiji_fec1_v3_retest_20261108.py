"""FEC-1 v3 retest: face-isolated (binding) + M-final (diagnostic) readouts.

Executes prereg v3 (plans/manifests/
taiji_f_expressive_capacity_prereg_v3_20261108.json) on the SAME graph,
population, capture and conventions as the committed W-v2 retest (card
9110e7a1...): the redesigned write path (content keys + tied reads +
content-conditioned values) at initialization, both arms, forced binding.

The ONLY analysis delta: the closed-form readout is computed in BOTH
protocols - face-isolated M_face (sum of the 22 face-event rank-one writes,
the tied-key existence condition, BINDING) and M_final (end-of-replay F,
persistence diagnostic, never binding).  Importing the committed W-v2 runner
installs its v2-aware capture instrument.
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
from taiji.rule_learning_20_1_net import enable_content_path_v2  # noqa: E402

PREREG_V3 = ROOT / "plans/manifests/taiji_f_expressive_capacity_prereg_v3_20261108.json"
CRITERIA_V3_SHA256 = "875a55b9ee4e75ffc17a43aab4ff6498be8f0966fb696af3986902cb4d77ed3f"
CARD = ROOT / "plans/manifests/taiji_fec1_v3_retest_feature_card_20261108.json"
CARD_SHA256 = "baf962977a66562aaaf68c135de8383235327d068ec1c378d9162e9fba9952d4"
REPORT = ROOT / "reports/taiji_fec1_v3_retest_20261108.json"
EXEC_MANIFEST = ROOT / "plans/manifests/taiji_fec1_v3_retest_execution_20261108.json"
PROGRESS = ROOT / "reports/taiji_fec1_v3_retest_20261108_progress.jsonl"
CONTENT_SEED = wv2.CONTENT_SEED
ARMS = wv2.ARMS

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


def evaluate_instance(graph, world: dict, criteria: dict) -> dict:
    groups = graph.subject.config.groups
    heads = graph.subject.config.fast_heads
    off = torch.zeros((groups, heads), dtype=torch.bool)
    on = torch.ones((groups, heads), dtype=torch.bool)
    release_actual = torch.full((groups, heads), bool(graph.fast_enabled), dtype=torch.bool)

    actual = w1a.replay(graph, world, off, release_actual)
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
        m_face = values.T @ keys  # sum of the 22 face rank-one writes
        face_readouts[name] = fec1.closed_form_stack(m_face, keys, values, gold)
        mfinal_readouts[name] = fec1.closed_form_stack(
            m_by_stack[int(name[1]), int(name[3])], keys, values, gold
        )
    best_face = max(face_readouts, key=lambda n: face_readouts[n]["decision_bit_accuracy"])
    binding_accuracy = float(face_readouts[best_face]["decision_bit_accuracy"])
    best_bits = fec1.per_face_bits(face_readouts[best_face], gold)
    controls = wv2.w1r2_controls(forced_rows, f_matrix, criteria)
    margin = binding_accuracy - controls["strongest"]

    actual_stacks = fec1.release_stack_fields(actual["release_events"], groups, heads)
    actual_key_result = (
        fec1.key_gates({name: f["keys"] for name, f in actual_stacks.items()}, criteria)
        if actual_stacks
        else {"pass": False, "binding": {}, "per_stack": {}}
    )

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
    # v3 value_space_collapsed extension: face-isolated < 0.3 with key gates
    # and alignment passing means the values are indistinguishable even
    # without interference (checked before the generic verdict)
    value_clause = bool(
        binding_accuracy < 0.3
        and key_result["pass"]
        and align["median_all"] >= criteria["gates"]["query_alignment_cosine_median_min"]
    )
    verdict = fec1.checkpoint_verdict(
        key_result["pass"], extras, actual_key_result["pass"], boundaries, dof, len(forced_rows)
    )
    if value_clause and verdict in ("key_space_collapsed_under_actual_timing", "inconclusive"):
        verdict = "value_space_collapsed"
    return {
        "verdict": verdict,
        "value_clause_fired": value_clause,
        "timing_coverage_actual": timing_coverage,
        "actual_written_lines": actual["written_lines"],
        "key_gates_forced": key_result,
        "key_gates_actual_pass": actual_key_result.get("pass", False),
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
                "binding_stack": max(
                    mfinal_readouts, key=lambda n: mfinal_readouts[n]["decision_bit_accuracy"]
                ),
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


def main() -> int:
    runner_sha = fec1.sha256_file(Path(__file__).resolve())
    try:
        card = json.loads(CARD.read_text(encoding="utf-8"))
        if fec1.sha256_file(CARD) != CARD_SHA256:
            raise EngineeringStop("v3 feature card sha mismatch")
        if card["criteria_v3_sha256"] != CRITERIA_V3_SHA256:
            raise EngineeringStop("v3 card points at different criteria")
        prereg = json.loads(PREREG_V3.read_text(encoding="utf-8"))
        raw = json.dumps(
            prereg["criteria"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
        if hashlib.sha256(raw).hexdigest() != CRITERIA_V3_SHA256:
            raise EngineeringStop("v3 criteria sha mismatch")
        gcard = json.loads(Path(card["graph_card"]).read_text(encoding="utf-8"))
        if fec1.sha256_file(Path(card["graph_card"])) != card["graph_card_sha256"]:
            raise EngineeringStop("referenced W-v2 graph card sha mismatch")
        if gcard["probe_face"]["seed"] != fixture.SEED or gcard["probe_face"]["world_index"] != 0:
            raise EngineeringStop("probe face drifted")
        criteria = prereg["criteria"]
    except EngineeringStop as error:
        append_progress({"stage": "preflight", "ok": False, "reason": str(error)})
        write_json(
            REPORT,
            {
                "package": "FEC-1/v3",
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
    world = fixture.build_world(gcard["probe_face"]["seed"], gcard["probe_face"]["world_index"])
    centroid = wv2.probe_face_centroid(world)
    parent_payload = torch.load(r2.BUNDLE, map_location="cpu", weights_only=True)
    gates = parent_payload.get("growth_gates")
    gate = dict(gates[0]) if gates else None
    candidate = r2.load_routing_bundle(r2.BUNDLE)[0]

    results: list[dict] = []
    identities: dict[str, str] = {}
    try:
        parity_graph = r2.build_graph(candidate, gate, "CF1")
        parity_graph.subject.eval()
        parity = fec1.driver_equivalence(parity_graph, world)
        append_progress({"stage": "parity", "ok": parity["ok"], "cpu": round(cost()[0], 3)})
        if not parity["ok"]:
            raise EngineeringStop(f"disabled-graph parity failed: {parity}")

        for arm in ARMS:
            graph = r2.build_graph(candidate, gate, arm)
            graph.subject.eval()
            enable_content_path_v2(graph, seed=CONTENT_SEED, centroid=centroid)
            graph.subject.eval()
            identities[arm] = r2.weights_digest(graph.subject)
            result = evaluate_instance(graph, world, criteria)
            result["arm"] = arm
            result["weights_digest"] = identities[arm]
            results.append(result)
            append_progress(
                {
                    "stage": "instance",
                    "arm": arm,
                    "verdict": result["verdict"],
                    "margin": result["margin"],
                    "face_isolated": result["closed_form"]["binding_decision_bit_accuracy"],
                    "face_event_id": result["closed_form"]["binding_event_id_accuracy"],
                    "cpu": round(cost()[0], 3),
                }
            )
    except BudgetStop as error:
        append_progress({"stage": "budget_stop", "reason": str(error)})
        write_json(
            REPORT,
            {
                "package": "FEC-1/v3",
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
                "package": "FEC-1/v3",
                "status": "engineering_stop",
                "verdict": "engineering_stop",
                "reason": str(error),
                "runner_sha256": runner_sha,
                "readings_produced": len(results),
            },
        )
        print(json.dumps({"verdict": "engineering_stop", "reason": str(error)}))
        return 0

    verdicts = [entry["verdict"] for entry in results]
    disagreement = len(set(verdicts)) > 1
    verdict = "inconclusive" if disagreement else verdicts[0]
    margins = [entry["margin"] for entry in results]
    cpu, wall = cost()
    write_json(
        REPORT,
        {
            "schema": "taiji-fec1-v3-retest-v1",
            "package": "FEC-1/v3",
            "date": "2026-11-08",
            "status": "completed",
            "verdict": verdict,
            "verdict_basis": "v3 decision rule (binding closed_form = face-isolated); 2/2 instance agreement",
            "per_instance_verdicts": verdicts,
            "disagreement": disagreement,
            "binding_margin": float(min(margins)),
            "discipline": {
                "training": 0,
                "optimizer_steps": 0,
                "confirm_reads": 0,
                "criteria_v3_sha256": CRITERIA_V3_SHA256,
            },
            "provenance": {
                "card_sha256": CARD_SHA256,
                "prereg_v3_manifest": "plans/manifests/taiji_f_expressive_capacity_prereg_v3_20261108.json",
                "graph_card": "plans/manifests/taiji_fec1_wv2_retest_feature_card_20261108.json",
                "runner_sha256": runner_sha,
                "weights_digests": identities,
                "parity": {"ok": True, "detail": "disabled graph bit-identical to graph.run_turn"},
            },
            "results": results,
            "cost": {"cpu_seconds": round(cpu, 3), "wall_seconds": round(wall, 3)},
            "scope": (
                "Function-class retest under the v3 binding protocol on the v2 write path at "
                "initialization; expressive_capacity_present is an architecture-requirement input, "
                "not training admission, not a capability claim."
            ),
        },
    )
    write_json(
        EXEC_MANIFEST,
        {
            "schema": "taiji-fec1-v3-retest-execution-manifest-v1",
            "package": "FEC-1/v3",
            "date": "2026-11-08",
            "card": "plans/manifests/taiji_fec1_v3_retest_feature_card_20261108.json",
            "card_sha256": CARD_SHA256,
            "criteria_v3_sha256": CRITERIA_V3_SHA256,
            "report": "reports/taiji_fec1_v3_retest_20261108.json",
            "report_sha256": fec1.sha256_file(REPORT),
            "runner": "scripts/training/probe_taiji_fec1_v3_retest_20261108.py",
            "runner_sha256": runner_sha,
            "submitted_runner_sha256": runner_sha,
            "verdict": verdict,
            "cost": {"cpu_seconds": round(cpu, 3), "wall_seconds": round(wall, 3)},
        },
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "per_instance": verdicts,
                "face_isolated": [
                    round(r["closed_form"]["binding_decision_bit_accuracy"], 4) for r in results
                ],
                "face_event_id": [
                    round(r["closed_form"]["binding_event_id_accuracy"], 4) for r in results
                ],
                "margins": [round(m, 4) for m in margins],
                "cpu": round(cpu, 2),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
