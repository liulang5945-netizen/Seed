"""FEC-1/W1 round-2 retest: centered content feature + tied read query.

Executes the round-2 retest pinned by
plans/manifests/taiji_fec1_w1r2_retest_feature_card_20261108.json under the
v2 pre-registration (plans/manifests/
taiji_f_expressive_capacity_prereg_v2_20261108.json, whose ONLY change from v1
is the capacity-capped fitted controls; every gate value is unchanged).

The replay instrument is imported verbatim from the committed round-1 runner;
its post-injection cross-check adapts automatically because the subject's
content_feature applies the centered mode from its own config.
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
import probe_taiji_package20_4c_r2_formal_comparison as r2  # noqa: E402

from taiji import rule_learning_20_1 as fixture  # noqa: E402
from taiji.rule_learning_20_1_net import enable_content_keys  # noqa: E402

PREREG_V2 = ROOT / "plans/manifests/taiji_f_expressive_capacity_prereg_v2_20261108.json"
CRITERIA_V2_SHA256 = "aa3a8aefbe932247c3fcb8eed711f6dfc062698b146ff3485939bec1e0a10877"
CARD = ROOT / "plans/manifests/taiji_fec1_w1r2_retest_feature_card_20261108.json"
CARD_SHA256 = "4855e43e793578aef27f3e3742c9f1f49da6bf5fc5449246d781c445b409b89e"
REPORT = ROOT / "reports/taiji_fec1_w1r2_retest_20261108.json"
EXEC_MANIFEST = ROOT / "plans/manifests/taiji_fec1_w1r2_retest_execution_20261108.json"
PROGRESS = ROOT / "reports/taiji_fec1_w1r2_retest_20261108_progress.jsonl"
CONTENT_SEED = 20261031
CONTROL_PROJ_SEED = 20261110
ARMS = ("CF1", "B_EH")

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


def capped_control_probes(rows: list[dict], f_matrix: np.ndarray, criteria: dict) -> dict:
    """v2 capacity-capped controls: every fitted probe on a 32-dim seeded
    random projection; window_identity keeps its native 2-d features."""
    gold = [row["gold"] for row in rows]
    queries = np.stack([fec1.query_key(row["query"]) for row in rows])
    x = np.concatenate([f_matrix, queries], axis=1)
    window = np.zeros((len(rows), 2), dtype=np.float64)
    for index, row in enumerate(rows):
        window[index, 0 if row["session"] == 1 else 1] = 1.0
    window_probe = fec1.p2s.probe(window, gold, fec1.p2s.decode_decision)

    def cap_projection(matrix: np.ndarray, seed: int) -> np.ndarray:
        generator = np.random.default_rng(seed)
        projection = generator.normal(size=(matrix.shape[1], 32)) / (matrix.shape[1] ** 0.5)
        reduced = matrix @ projection
        return reduced

    permutation = []
    for seed in criteria["label_permutation_seeds"]:
        order = np.random.default_rng(seed).permutation(len(gold))
        permuted = [gold[int(index)] for index in order]
        reduced = cap_projection(x, seed)
        permutation.append(
            {
                "seed": int(seed),
                "probe": fec1.p2s.probe(reduced, permuted, fec1.p2s.decode_decision)["metrics"][
                    "decision_bit_accuracy"
                ],
            }
        )
    perm_max = max(entry["probe"] for entry in permutation)
    random_reduced = cap_projection(x, CONTROL_PROJ_SEED)
    random_probe = fec1.p2s.probe(random_reduced, gold, fec1.p2s.decode_decision)
    other = np.stack([fec1.p2s.other_features(row) for row in rows])
    other_reduced = cap_projection(other, CONTROL_PROJ_SEED + 1)
    other_probe = fec1.p2s.probe(other_reduced, gold, fec1.p2s.decode_decision)
    values = {
        "window_identity": float(window_probe["metrics"]["decision_bit_accuracy"]),
        "label_permutation_5_seeds_max": float(perm_max),
        "random_projection_capped": float(random_probe["metrics"]["decision_bit_accuracy"]),
        "no_f_other_persistent_state_capped": float(
            other_probe["metrics"]["decision_bit_accuracy"]
        ),
    }
    return {
        "values": values,
        "per_seed": {str(entry["seed"]): entry["probe"] for entry in permutation},
        "strongest": float(max(values.values())),
        "capacity_cap": 32,
    }


def probe_face_centroid(world: dict) -> torch.Tensor:
    """Input-side centroid of the 22 probe-face query byte-count vectors."""
    queries = [
        turn["query"].encode("utf-8")
        for turn in world["response"]["turns"]
        if bool(turn["correction_surface"])
    ]
    counts = []
    for query in queries:
        vector = torch.zeros(128, dtype=torch.float64)
        for value in query:
            vector[int(value) % 128] += 1.0
        counts.append(vector)
    return torch.stack(counts).mean(0)


def evaluate_instance(graph, world: dict, criteria: dict, card: dict) -> dict:
    groups = graph.subject.config.groups
    heads = graph.subject.config.fast_heads
    off = torch.zeros((groups, heads), dtype=torch.bool)
    on = torch.ones((groups, heads), dtype=torch.bool)
    release_actual = torch.full((groups, heads), bool(graph.fast_enabled), dtype=torch.bool)

    actual = w1a.replay(graph, world, off, release_actual)
    actual_rows = [
        row for row in actual["rows"] if row["correction_surface"] and row["released_lines"] > 0
    ]
    if len(actual_rows) != card["probe_face"]["expected_surfaces"]:
        raise EngineeringStop(f"expected 22 actual faces, got {len(actual_rows)}")
    face_turns = {row["turn"] for row in actual_rows}
    timing_coverage = len(set(actual["face_written_turns"]) & face_turns) / len(actual_rows)

    forced = w1a.replay(graph, world, on, on)
    forced_rows = [
        row for row in forced["rows"] if row["correction_surface"] and row["released_lines"] > 0
    ]
    if len(forced_rows) != card["probe_face"]["expected_surfaces"]:
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
    stack_readouts = {
        name: fec1.closed_form_stack(
            m_by_stack[int(name[1]), int(name[3])], f["keys"], f["values"], gold
        )
        for name, f in stacks.items()
    }
    best_stack = max(stack_readouts, key=lambda n: stack_readouts[n]["decision_bit_accuracy"])
    closed_form_accuracy = float(stack_readouts[best_stack]["decision_bit_accuracy"])
    best_bits = fec1.per_face_bits(stack_readouts[best_stack], gold)
    controls = capped_control_probes(forced_rows, f_matrix, criteria)
    margin = closed_form_accuracy - controls["strongest"]

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
        "closed_form_accuracy": closed_form_accuracy,
        "margin": margin,
        "effective_dof": dof,
    }
    boundaries = fec1.boundary_flags(key_result["binding"], extras, criteria["gates"])
    verdict = fec1.checkpoint_verdict(
        key_result["pass"], extras, actual_key_result["pass"], boundaries, dof, len(forced_rows)
    )
    return {
        "verdict": verdict,
        "timing_coverage_actual": timing_coverage,
        "actual_written_lines": actual["written_lines"],
        "forced_turn_write_faces": len(forced["face_written_turns"]),
        "key_gates_forced": key_result,
        "key_gates_actual_pass": actual_key_result.get("pass", False),
        "query_alignment": align,
        "value_effective_ranks": value_ranks,
        "query_effective_ranks": query_ranks,
        "capacity": cap,
        "closed_form": {
            "binding_stack": best_stack,
            "binding_decision_bit_accuracy": closed_form_accuracy,
            "binding_event_id_accuracy": stack_readouts[best_stack]["event_id_accuracy"],
            "per_stack": {
                name: {
                    "decision_bit_accuracy": item["decision_bit_accuracy"],
                    "event_id_accuracy": item["event_id_accuracy"],
                }
                for name, item in stack_readouts.items()
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
            raise EngineeringStop("round-2 feature card sha mismatch")
        if card["parent_criteria_v2_sha256"] != CRITERIA_V2_SHA256:
            raise EngineeringStop("round-2 card points at different criteria")
        prereg = json.loads(PREREG_V2.read_text(encoding="utf-8"))
        raw = json.dumps(
            prereg["criteria"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
        if hashlib.sha256(raw).hexdigest() != CRITERIA_V2_SHA256:
            raise EngineeringStop("v2 criteria sha mismatch")
        if card["probe_face"]["seed"] != fixture.SEED or card["probe_face"]["world_index"] != 0:
            raise EngineeringStop("probe face drifted from round-2 card")
        criteria = prereg["criteria"]
    except EngineeringStop as error:
        append_progress({"stage": "preflight", "ok": False, "reason": str(error)})
        write_json(
            REPORT,
            {
                "package": "FEC-1/W1-round2",
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
            "criteria_v2_sha256": CRITERIA_V2_SHA256,
            "runner_sha256": runner_sha,
            "cpu": round(cost()[0], 3),
        }
    )
    world = fixture.build_world(card["probe_face"]["seed"], card["probe_face"]["world_index"])
    centroid = probe_face_centroid(world)
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
            enable_content_keys(graph, seed=CONTENT_SEED, centroid=centroid, centered=True)
            graph.subject.eval()
            identities[arm] = r2.weights_digest(graph.subject)
            result = evaluate_instance(graph, world, criteria, card)
            result["arm"] = arm
            result["weights_digest"] = identities[arm]
            results.append(result)
            append_progress(
                {
                    "stage": "instance",
                    "arm": arm,
                    "verdict": result["verdict"],
                    "margin": result["margin"],
                    "key_rank": result["key_gates_forced"]["binding"]["key_effective_rank"],
                    "pairwise": result["key_gates_forced"]["binding"][
                        "pairwise_separable_fraction"
                    ],
                    "cpu": round(cost()[0], 3),
                }
            )
    except BudgetStop as error:
        append_progress({"stage": "budget_stop", "reason": str(error)})
        write_json(
            REPORT,
            {
                "package": "FEC-1/W1-round2",
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
                "package": "FEC-1/W1-round2",
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
            "schema": "taiji-fec1-w1r2-retest-v1",
            "package": "FEC-1/W1-round2",
            "date": "2026-11-08",
            "status": "completed",
            "verdict": verdict,
            "verdict_basis": "v2 decision rule; 2/2 instance agreement required; round 2 is the LAST round of the W1 loop",
            "per_instance_verdicts": verdicts,
            "disagreement": disagreement,
            "binding_margin": float(min(margins)),
            "discipline": {
                "training": 0,
                "optimizer_steps": 0,
                "confirm_reads": 0,
                "checkpoint_writes": 0,
                "criteria_v1_unchanged": True,
                "criteria_v2_sha256": CRITERIA_V2_SHA256,
            },
            "provenance": {
                "round2_card_sha256": CARD_SHA256,
                "prereg_v2_manifest": "plans/manifests/taiji_f_expressive_capacity_prereg_v2_20261108.json",
                "runner_sha256": runner_sha,
                "content_seed": CONTENT_SEED,
                "control_projection_seed": CONTROL_PROJ_SEED,
                "weights_digests": identities,
                "parity": {"ok": True, "detail": "disabled graph bit-identical to graph.run_turn"},
                "centroid": "input-side mean byte-count of the 22 probe-face queries (no gold)",
            },
            "results": results,
            "cost": {"cpu_seconds": round(cpu, 3), "wall_seconds": round(wall, 3)},
            "scope": (
                "Function-class retest on the round-2 graph at initialization only; "
                "expressive_capacity_present under v2 is an architecture-requirement input, "
                "not training admission, not a capability claim."
            ),
        },
    )
    write_json(
        EXEC_MANIFEST,
        {
            "schema": "taiji-fec1-w1r2-retest-execution-manifest-v1",
            "package": "FEC-1/W1-round2",
            "date": "2026-11-08",
            "round2_card": "plans/manifests/taiji_fec1_w1r2_retest_feature_card_20261108.json",
            "round2_card_sha256": CARD_SHA256,
            "criteria_v2_sha256": CRITERIA_V2_SHA256,
            "report": "reports/taiji_fec1_w1r2_retest_20261108.json",
            "report_sha256": fec1.sha256_file(REPORT),
            "runner": "scripts/training/probe_taiji_fec1_w1r2_retest_20261108.py",
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
                "margins": [round(m, 4) for m in margins],
                "cpu": round(cpu, 2),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
