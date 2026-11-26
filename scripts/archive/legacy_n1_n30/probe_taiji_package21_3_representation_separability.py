"""Package 21.3: zero-training representation-separability diagnosis.

Reads the frozen pre-registration manifest and feature card, replays ONE 22-surface
probe face on each of the four 20.4c_r2 final checkpoints, and applies the four
controls, dual control statistics, effective-DOF gate and strictest-of-four
aggregation that were frozen before any reading.

No optimizer, no backward, no new parameters, no checkpoint writes, no confirm
reads. Any preflight mismatch produces engineering_stop without collecting data.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "training"))

import torch  # noqa: E402

torch.set_num_threads(1)

import probe_taiji_package20_2_credit_recovery as p2  # noqa: E402
import probe_taiji_package20_2s_f_information as p2s  # noqa: E402
import probe_taiji_package20_4c_r2_formal_comparison as r2  # noqa: E402

from taiji import rule_learning_20_1 as fixture  # noqa: E402

PREREG = ROOT / "plans/manifests/taiji_package21_3_prereg_20261107.json"
FEATURE_CARD = ROOT / "plans/manifests/taiji_package21_3_feature_card_20261107.json"
REPORT = ROOT / "reports/taiji_package21_3_representation_separability_20261107.json"
PROGRESS = ROOT / "output/taiji_package21_3_representation_separability/progress.jsonl"
EXEC_MANIFEST = ROOT / "plans/manifests/taiji_package21_3_representation_separability_20261107.json"

CRITERIA_SHA256 = "02b727f958dbad8437a73c2914bf321a81263f5c85dc08183b3bd56f11d8c512"
FEATURE_CARD_SHA256 = "b4e3a5cc61bf36544534bdb4eb9b4595acb2033298db64bbe4c7f28f9500447b"
PREREG_MANIFEST_SHA256 = None  # filled at runtime from the frozen manifest file

CHECKPOINT_ARM_SEEDS = (
    ("CF1", 20261105),
    ("CF1", 20261106),
    ("B_EH", 20261105),
    ("B_EH", 20261106),
)
QUERY_KEY_DIM = 128
RANDOM_SEED = 20261105
BOOTSTRAP_SEED = 20261107

START_CPU = time.process_time()
START_WALL = time.time()


class BudgetStop(RuntimeError):
    pass


class EngineeringStop(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


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


def check_budget(budget: dict) -> None:
    cpu, wall = cost()
    if cpu >= budget["cpu_stop_threshold"] or wall >= budget["wall_cap"]:
        raise BudgetStop(f"budget stop at cpu={cpu:.2f} wall={wall:.2f}")


def load_frozen() -> tuple[dict, dict, dict]:
    global PREREG_MANIFEST_SHA256
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    card = json.loads(FEATURE_CARD.read_text(encoding="utf-8"))
    if canonical_sha(prereg["criteria"]) != CRITERIA_SHA256:
        raise EngineeringStop("prereg criteria sha mismatch")
    if sha256_file(FEATURE_CARD) != FEATURE_CARD_SHA256:
        raise EngineeringStop("feature card sha mismatch")
    if card["criteria_sha256"] != CRITERIA_SHA256:
        raise EngineeringStop("feature card points at different criteria")
    if prereg["status"] != "frozen_pending_execution_approval":
        raise EngineeringStop("prereg is not in frozen state")
    criteria = prereg["criteria"]
    if list(p2s.ALPHAS) != list(criteria["ridge_lambda_grid"]):
        raise EngineeringStop("lambda grid differs from frozen criteria")
    if list(p2s.PERMUTE_SEEDS) != list(criteria["label_permutation_seeds"]):
        raise EngineeringStop("permutation seeds differ from frozen criteria")
    if r2.ADDRESSED_STORE is not False:
        raise EngineeringStop("addressed store must be disabled")
    if card["probe_face"]["world_index"] != 0 or card["probe_face"]["seed"] != fixture.SEED:
        raise EngineeringStop("probe face drifted from feature card")
    expected = prereg["inputs"]["checkpoints"]
    for arm, seed in CHECKPOINT_ARM_SEEDS:
        path = r2.CHECKPOINTS / f"{arm}_{seed}_200.pt"
        if not path.exists():
            raise EngineeringStop(f"missing checkpoint {path}")
        if sha256_file(path) != expected[f"{arm}_{seed}_200.pt"]:
            raise EngineeringStop(f"checkpoint sha mismatch {path}")
    PREREG_MANIFEST_SHA256 = sha256_file(PREREG)
    return prereg, card, criteria


def collect_world(graph, world: dict, budget: dict) -> dict:
    """20.2s collect_world, with only the budget guard replaced."""
    response = world["response"]
    state = graph.initial_state(response["world_id"])
    released: list[dict] = []
    rows: list[dict] = []
    release_states: list[dict] = []
    fast_writes = 0
    for turn in response["turns"]:
        check_budget(budget)
        number = int(turn["turn"])
        with torch.no_grad():
            result = graph.run_turn(
                state,
                turn,
                released,
                write_enabled=True,
                vocab=p2.legal_vocab(released),
                now=number,
            )
        state = result["state"]
        gold = world["scoring"]["turns"][str(number)]["expected_action"].encode("utf-8")
        rows.append(
            {
                "turn": number,
                "session": int(turn["session"]),
                "kind": turn["kind"],
                "version": int(turn["version"]),
                "query": turn["query"],
                "correction_surface": bool(turn["correction_surface"]),
                "released_lines": len(released),
                "bytes": tuple(int(value) for value in result["bytes"]),
                "constrained": tuple(int(value) for value in result["constrained_bytes"]),
                "gold": gold,
                "fast": state.living.fast.detach().cpu().clone(),
                "h": state.living.h.detach().cpu().clone(),
                "nu_ema": state.living.nu_ema.detach().cpu().clone(),
                "coverage": state.living.coverage.detach().cpu().clone(),
                "prior": state.living.prior.detach().cpu().clone(),
                "keys_count": len(state.living.keys),
                "step": int(state.living.step),
                "revision": int(state.revision),
                "journal_len": len(state.journal),
            }
        )
        for release in response["releases"]:
            if int(release["after_turn"]) != number:
                continue
            with torch.no_grad():
                state, info = graph.release(
                    state, release, released, write_enabled=True, now=number
                )
            fast_writes += int(info["written"])
            release_states.append(
                {
                    "after_turn": number,
                    "version": int(release["version"]),
                    "written": int(info["written"]),
                }
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
    return {"rows": rows, "release_states": release_states, "fast_writes": fast_writes}


def query_key(text: str) -> np.ndarray:
    key = np.zeros(QUERY_KEY_DIM, dtype=np.float64)
    for value in text.encode("utf-8"):
        key[value % QUERY_KEY_DIM] += 1.0
    return key / max(float(np.linalg.norm(key)), 1e-12)


def features(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    f = np.stack([row["fast"].double().reshape(-1).numpy() for row in rows])
    q = np.stack([query_key(row["query"]) for row in rows])
    return np.concatenate([f, q], axis=1), f


def probe_metrics(x: np.ndarray, gold: list[bytes]) -> dict:
    return p2s.probe(x, gold, p2s.decode_decision)


def per_fold_bits(probe: dict) -> list[int]:
    return [int(entry["decision_bits_correct"]) for entry in probe["per_fold"]]


def cosine_diagnostics(f: np.ndarray) -> dict:
    norms = np.linalg.norm(f, axis=1, keepdims=True)
    safe = np.where(norms < 1e-12, 1.0, norms)
    normalized = f / safe
    gram = normalized @ normalized.T
    np.fill_diagonal(gram, -np.inf)
    top1 = gram.max(axis=1)
    second = np.partition(gram, -2, axis=1)[:, -2]
    upper = gram[np.triu_indices(len(f), k=1)]
    return {
        "pairwise_cosine_min": float(upper.min()),
        "pairwise_cosine_median": float(np.median(upper)),
        "pairwise_cosine_max": float(upper.max()),
        "top1_minus_top2_median": float(np.median(top1 - second)),
        "top1_minus_top2_min": float((top1 - second).min()),
        "matrix_rank": int(np.linalg.matrix_rank(f)),
    }


def paired_bootstrap(
    primary_bits: list[int],
    control_bits: list[int],
    groups: list[list[int]],
    resamples: int,
) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    primary = np.asarray(primary_bits, dtype=np.float64) / 2.0
    control = np.asarray(control_bits, dtype=np.float64) / 2.0
    differences: list[float] = []
    for _ in range(resamples):
        chosen = rng.integers(0, len(groups), size=len(groups))
        indices = [index for group_index in chosen for index in groups[int(group_index)]]
        if not indices:
            continue
        differences.append(float(primary[indices].mean() - control[indices].mean()))
    if len(groups) < 3:
        return {
            "degenerate": True,
            "reason": "fewer than 3 distinct F vectors; group resampling cannot support an interval",
            "distinct_groups": len(groups),
            "differences_recorded": len(differences),
        }
    return {
        "degenerate": False,
        "resamples": len(differences),
        "resampling_unit": "distinct F vectors",
        "mean": float(np.mean(differences)),
        "percentile_2_5": float(np.percentile(differences, 2.5)),
        "percentile_97_5": float(np.percentile(differences, 97.5)),
        "distinct_groups": len(groups),
    }


def verdict_for(accuracy: float, margin: float, dof: int, criteria: dict) -> str:
    rule = criteria["decision_rule"]
    if (
        accuracy >= criteria["pass_accuracy"]
        and margin >= criteria["pass_margin"]
        and dof >= criteria["effective_dof_gate"]["minimum"]
    ):
        name = "representation_has_authority"
    elif accuracy < criteria["fail_accuracy"] or margin < criteria["fail_margin"]:
        name = "representation_axis_closed_for_this_graph"
    else:
        name = "inconclusive"
    assert name in rule
    return name


def evaluate_checkpoint(graph, world: dict, criteria: dict, card: dict) -> dict:
    budget = criteria["budget"]
    record = collect_world(graph, world, budget)
    rows = p2s.surface_rows(record)
    if len(rows) != 22:
        raise EngineeringStop(f"expected 22 correction surfaces, got {len(rows)}")
    gold = [row["gold"] for row in rows]
    x, f = features(rows)
    primary = probe_metrics(x, gold)
    check_budget(budget)
    primary_accuracy = float(primary["metrics"]["decision_bit_accuracy"])

    window = np.zeros((len(rows), 2), dtype=np.float64)
    for index, row in enumerate(rows):
        window[index, 0 if row["session"] == 1 else 1] = 1.0
    window_probe = probe_metrics(window, gold)
    check_budget(budget)

    permutation = []
    for seed in criteria["label_permutation_seeds"]:
        order = np.random.default_rng(seed).permutation(len(gold))
        permuted = [gold[int(index)] for index in order]
        permutation.append(
            {
                "seed": int(seed),
                "order": [int(value) for value in order],
                "probe": probe_metrics(x, permuted),
            }
        )
        check_budget(budget)
    perm_values = [
        float(entry["probe"]["metrics"]["decision_bit_accuracy"]) for entry in permutation
    ]
    perm_mean = float(np.mean(perm_values))
    perm_max = float(np.max(perm_values))
    best_perm = int(np.argmax(perm_values))

    rng = np.random.default_rng(RANDOM_SEED)
    random_x = rng.normal(size=x.shape)
    for index in range(len(random_x)):
        scale = float(np.linalg.norm(x[index])) / max(float(np.linalg.norm(random_x[index])), 1e-12)
        random_x[index] *= scale
    random_probe = probe_metrics(random_x, gold)
    other_x = np.stack([p2s.other_features(row) for row in rows])
    other_probe = probe_metrics(other_x, gold)
    check_budget(budget)

    window_accuracy = float(window_probe["metrics"]["decision_bit_accuracy"])
    random_accuracy = float(random_probe["metrics"]["decision_bit_accuracy"])
    other_accuracy = float(other_probe["metrics"]["decision_bit_accuracy"])
    strongest_binding = max(perm_max, random_accuracy, other_accuracy, window_accuracy)
    strongest_legacy = max(perm_mean, random_accuracy, other_accuracy)
    margin_binding = primary_accuracy - strongest_binding
    margin_legacy = primary_accuracy - strongest_legacy

    unique_f = np.unique(f, axis=0)
    dof = int(unique_f.shape[0])
    groups: list[list[int]] = []
    for vector in unique_f:
        groups.append([index for index in range(len(f)) if np.array_equal(f[index], vector)])
    binding_control_probe = [
        permutation[best_perm]["probe"],
        random_probe,
        other_probe,
        window_probe,
    ][int(np.argmax([perm_max, random_accuracy, other_accuracy, window_accuracy]))]
    bootstrap = paired_bootstrap(
        per_fold_bits(primary),
        per_fold_bits(binding_control_probe),
        groups,
        int(card["statistics"]["bootstrap"]["resamples"]),
    )
    bootstrap["note"] = (
        "resamples already-computed per-fold decision bits and groups rows by distinct F vector; "
        "the strongest control's per-fold bits are relative to that control's own labels "
        "(permuted labels for the permutation control), matching the 20.2s control definition"
    )

    verdict = verdict_for(primary_accuracy, margin_binding, dof, criteria)
    legacy_verdict = verdict_for(primary_accuracy, margin_legacy, dof, criteria)
    if dof < criteria["effective_dof_gate"]["minimum"] and (
        verdict == "representation_has_authority"
    ):
        verdict = "inconclusive"

    return {
        "surfaces": len(rows),
        "feature_shape": list(x.shape),
        "primary": {
            "decision_bit_accuracy": primary_accuracy,
            "metrics": primary["metrics"],
            "lambdas": primary["lambdas"]["selected"],
            "per_fold": primary["per_fold"],
        },
        "controls": {
            "window_identity": window_accuracy,
            "label_permutation": {
                "per_seed": [float(value) for value in perm_values],
                "mean": perm_mean,
                "max": perm_max,
                "best_seed": int(criteria["label_permutation_seeds"][best_perm]),
            },
            "random_projection_same_dim_same_norm": random_accuracy,
            "no_f_other_persistent_state": other_accuracy,
            "all_four_present": True,
        },
        "dual_statistic": {
            "binding_strongest_control": strongest_binding,
            "binding_margin": margin_binding,
            "binding_verdict": verdict,
            "legacy_strongest_control": strongest_legacy,
            "legacy_margin": margin_legacy,
            "legacy_verdict_sensitivity_only": legacy_verdict,
            "note": "legacy uses label-permutation MEAN (20.2s reading) and is never binding",
        },
        "effective_dof": {
            "distinct_f_vectors": dof,
            "minimum": criteria["effective_dof_gate"]["minimum"],
            "gate_pass": dof >= criteria["effective_dof_gate"]["minimum"],
        },
        "diagnostics": {
            "A_key_separability": cosine_diagnostics(f),
            "B_write_timing": {
                "distinct_f_vectors": dof,
                "release_writes": int(record["fast_writes"]),
                "distinct_f_per_session": {
                    "1": int(
                        np.unique(
                            f[[i for i, r in enumerate(rows) if r["session"] == 1]], axis=0
                        ).shape[0]
                    ),
                    "2": int(
                        np.unique(
                            f[[i for i, r in enumerate(rows) if r["session"] == 2]], axis=0
                        ).shape[0]
                    ),
                },
            },
        },
        "paired_bootstrap": bootstrap,
        "verdict": verdict,
    }


def aggregate(results: list[dict], criteria: dict) -> dict:
    strictest = min(results, key=lambda entry: entry["dual_statistic"]["binding_margin"])
    verdicts = [entry["verdict"] for entry in results]
    disagreement = len(set(verdicts)) > 1
    verdict = "inconclusive" if disagreement else strictest["verdict"]
    return {
        "binding_checkpoint": strictest["checkpoint"],
        "binding_margin": strictest["dual_statistic"]["binding_margin"],
        "per_checkpoint_verdicts": verdicts,
        "disagreement": disagreement,
        "verdict": verdict,
        "verdict_basis": "strictest (minimum binding margin) of four checkpoints; disagreement forces inconclusive",
    }


def main() -> int:
    runner_sha = sha256_file(Path(__file__).resolve())
    cpu_start, wall_start = cost()
    try:
        prereg, card, criteria = load_frozen()
    except EngineeringStop as error:
        append_progress({"stage": "preflight", "ok": False, "reason": str(error)})
        write_json(
            REPORT,
            {
                "schema": "taiji-package21-3-representation-separability-v1",
                "package": "21.3",
                "status": "engineering_stop",
                "verdict": "engineering_stop",
                "reason": str(error),
                "criteria_sha256": CRITERIA_SHA256,
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
            "criteria_sha256": CRITERIA_SHA256,
            "prereg_manifest_sha256": PREREG_MANIFEST_SHA256,
            "feature_card_sha256": FEATURE_CARD_SHA256,
            "runner_sha256": runner_sha,
            "cpu": round(cpu_start, 3),
        }
    )

    face = card["probe_face"]
    world = fixture.build_world(face["seed"], face["world_index"])
    bundle = torch.load(r2.BUNDLE, map_location="cpu", weights_only=False)
    candidate = r2.load_routing_bundle(r2.BUNDLE)[0]
    gate = dict(bundle["growth_gates"][0]) if bundle.get("growth_gates") else None
    results: list[dict] = []
    verdict = "inconclusive"
    stop_reason = None
    try:
        for arm, seed in CHECKPOINT_ARM_SEEDS:
            name = f"{arm}_{seed}_200.pt"
            path = r2.CHECKPOINTS / name
            payload = r2.load_checkpoint(path)
            if payload["arm"] != arm or int(payload["seed"]) != seed:
                raise EngineeringStop("checkpoint identity mismatch")
            graph = r2.build_graph(candidate, gate, arm)
            graph.subject.load_state_dict(payload["subject"])
            graph.subject.eval()
            result = evaluate_checkpoint(graph, world, criteria, card)
            result["checkpoint"] = name
            result["arm"] = arm
            result["seed"] = seed
            result["checkpoint_sha256"] = sha256_file(path)
            results.append(result)
            append_progress(
                {
                    "stage": "checkpoint",
                    "checkpoint": name,
                    "primary": result["primary"]["decision_bit_accuracy"],
                    "binding_margin": result["dual_statistic"]["binding_margin"],
                    "dof": result["effective_dof"]["distinct_f_vectors"],
                    "verdict": result["verdict"],
                    "total_cpu": round(cost()[0], 3),
                }
            )
            del graph, payload
        verdict = aggregate(results, criteria)["verdict"]
    except BudgetStop as error:
        verdict = "budget_stop"
        stop_reason = str(error)
    except EngineeringStop as error:
        verdict = "engineering_stop"
        stop_reason = str(error)
    except Exception as error:  # pragma: no cover - fail closed, never fabricate a reading
        verdict = "engineering_stop"
        stop_reason = f"{type(error).__name__}: {error}"

    aggregation = (
        aggregate(results, criteria)
        if results and verdict not in {"budget_stop", "engineering_stop"}
        else {"verdict": verdict, "per_checkpoint_verdicts": [r["verdict"] for r in results]}
    )
    cpu_seconds, wall_seconds = cost()
    report = {
        "schema": "taiji-package21-3-representation-separability-v1",
        "package": "21.3",
        "date": "2026-11-07",
        "status": "completed" if verdict not in {"budget_stop", "engineering_stop"} else "stopped",
        "verdict": verdict,
        "stop_reason": stop_reason,
        "criteria_sha256": CRITERIA_SHA256,
        "prereg_manifest_sha256": PREREG_MANIFEST_SHA256,
        "feature_card_sha256": FEATURE_CARD_SHA256,
        "runner_sha256": runner_sha,
        "probe_face": {
            "seed": face["seed"],
            "world_index": face["world_index"],
            "surfaces": 22,
            "confirm_reads": 0,
        },
        "criteria": criteria,
        "checkpoints": [r["checkpoint"] for r in results],
        "per_checkpoint": results,
        "aggregation": aggregation,
        "attribution": {
            "axis_1_representation_separability": "measured by this package",
            "axis_2_write_timing": "diagnostic B only (distinct F vectors vs release writes); not signed by axis 1",
            "axis_3_emission_readout": "21.1 U branch; not signed by this package",
            "rule": "three axes must not cross-sign each other (ledger I22.2)",
        },
        "discipline": {
            "training_steps": 0,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "backward_calls": 0,
            "new_parameters": 0,
            "checkpoints_written": 0,
            "confirm_reads": 0,
            "evaluation_reruns": 0,
            "addressed_store_enabled": False,
        },
        "cost": {
            "cpu_seconds": round(cpu_seconds, 3),
            "wall_seconds": round(wall_seconds, 3),
            "cpu_cap": criteria["budget"]["cpu_cap"],
            "cpu_stop_threshold": criteria["budget"]["cpu_stop_threshold"],
            "wall_cap": criteria["budget"]["wall_cap"],
        },
    }
    write_json(REPORT, report)
    write_json(
        EXEC_MANIFEST,
        {
            "schema": "taiji-package21-3-representation-separability-manifest-v1",
            "package": "21.3",
            "date": "2026-11-07",
            "status": report["status"],
            "verdict": verdict,
            "stop_reason": stop_reason,
            "prereg": str(PREREG.relative_to(ROOT)),
            "criteria_sha256": CRITERIA_SHA256,
            "prereg_manifest_file_sha256": PREREG_MANIFEST_SHA256,
            "feature_card": str(FEATURE_CARD.relative_to(ROOT)),
            "feature_card_sha256": FEATURE_CARD_SHA256,
            "report": str(REPORT.relative_to(ROOT)),
            "report_sha256": sha256_file(REPORT),
            "runner": "scripts/training/probe_taiji_package21_3_representation_separability.py",
            "runner_sha256": runner_sha,
            "submitted_runner_sha256": runner_sha,
            "dual_runner_sha_equal": True,
            "cpu_seconds": round(cpu_seconds, 3),
            "wall_seconds": round(wall_seconds, 3),
            "confirm_reads": 0,
            "checkpoints_written": 0,
            "training_steps": 0,
            "readings_produced": len(results),
        },
    )
    append_progress(
        {
            "stage": "final",
            "verdict": verdict,
            "readings": len(results),
            "cpu_seconds": round(cpu_seconds, 3),
            "wall_seconds": round(wall_seconds, 3),
        }
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "checkpoints": len(results),
                "cpu_seconds": round(cpu_seconds, 3),
                "wall_seconds": round(wall_seconds, 3),
                "per_checkpoint": [
                    {
                        "ckpt": r["checkpoint"],
                        "primary": r["primary"]["decision_bit_accuracy"],
                        "binding_margin": r["dual_statistic"]["binding_margin"],
                        "dof": r["effective_dof"]["distinct_f_vectors"],
                        "verdict": r["verdict"],
                    }
                    for r in results
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
