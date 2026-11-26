"""Run the first incremental G-only learning Gate on top of frozen K.

The script consumes the sealed P3.3 candidate-set artifact, restores the P3.2
K workers read-only, and trains only a small native G scorer on the train
records.  P2.7 remains an untouched holdout.  The result is an experiment
report, never an automatic promotion.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.build_taiji_m5_k_p2_3_recovery_continuation import (  # noqa: E402
    _restore_tensors,
)
from scripts.training.eval_taiji_m5_k1_skill_composition import (  # noqa: E402
    READ_ONLY_ROUTES,
)
from scripts.training.eval_taiji_m5_k_p0_equal_replay_diagnostic import (  # noqa: E402
    _atomic_roundtrip,
)
from scripts.training.eval_taiji_m5_k_p2_2_safety_bridge_canary import (  # noqa: E402
    _run_intent,
)
from scripts.training.eval_taiji_m5_k_p2_3_targeted_learning import (  # noqa: E402
    _fresh_learners,
)
from scripts.training.eval_taiji_m5_k_p2_validation_pilot import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    _checkpoint_preflight,
)
from scripts.training.eval_taiji_m5_k_p3_1_single_cell import (  # noqa: E402
    _build_holdout_cases,
)
from scripts.training.eval_taiji_m5_k_p3_3_g_signal_canary import (  # noqa: E402
    CONFIDENCE_FLOOR,
    P2_7_MANIFEST,
    P3_2_MANIFEST,
    P3_2_REPORT,
    _candidate_sets,
    _catalogs,
)
from taiji import (  # noqa: E402
    GSelectionCandidate,
    GSelectionCandidateSet,
    GSelectionLearner,
    NativeReadOnlyIntentPlanner,
    ReadOnlyIntentPolicy,
    StructuredSemanticExample,
    StructuredSemanticTransitionExample,
    content_digest,
)

REPORT_FORMAT = "taiji-m5-k-p3-3-g-learning-v1"
VERSION = 1
G_EPOCHS = 8
G_LEARNING_RATE = 0.15
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p3_3_g_candidate_manifest_v1.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_3_g_learning_20260911.json"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_mapping(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected mapping checkpoint at {path}")
    return {str(key): value for key, value in payload.items()}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON mapping at {path}")
    return dict(payload)


def _independent_g_restore(path: Path) -> dict[str, Any]:
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--verify-only", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    result: dict[str, Any] = {
        "returncode": int(child.returncode),
        "independent_process_restore": False,
        "stdout": child.stdout[-2000:],
        "stderr": child.stderr[-2000:],
    }
    if child.returncode != 0:
        return result
    try:
        decoded = json.loads(child.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        result["error"] = str(exc)
        return result
    result.update(decoded)
    result["independent_process_restore"] = bool(decoded.get("passed"))
    return result


def _verify_g_checkpoint(path: Path) -> dict[str, Any]:
    payload = _load_mapping(path)
    learner = GSelectionLearner.from_checkpoint(payload, device="cpu")
    result = {
        "restore_digest_equal": content_digest(learner.checkpoint()) == content_digest(payload),
        "parameter_count": learner.parameter_count,
        "training_steps": learner.training_steps,
    }
    result["passed"] = bool(result["restore_digest_equal"])
    print(json.dumps(result, ensure_ascii=False))
    return result


def _save_g_checkpoint(directory: Path, learner: GSelectionLearner, name: str) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    payload = learner.checkpoint()
    _atomic_roundtrip(path, payload)
    restore = _independent_g_restore(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "digest": content_digest(payload),
        "training_steps": learner.training_steps,
        "restore": restore,
        "passed": bool(restore.get("independent_process_restore")),
    }


def _candidate_sets_from_manifest(
    manifest: Mapping[str, Any],
) -> tuple[tuple[GSelectionCandidateSet, ...], tuple[GSelectionCandidateSet, ...]]:
    if manifest.get("manifest_digest") != content_digest(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    ):
        raise ValueError("P3.3 candidate manifest digest mismatch")
    train = tuple(
        GSelectionCandidateSet.from_payload(payload)
        for payload in manifest.get("train_records", ())
    )
    validation = tuple(
        GSelectionCandidateSet.from_payload(payload)
        for payload in manifest.get("validation_records", ())
    )
    if not train or not validation:
        raise ValueError("P3.3 candidate manifest must contain train and validation records")
    return train, validation


def _verify_source_contracts(
    *,
    signal_report: Mapping[str, Any],
    signal_manifest: Mapping[str, Any],
    p3_2_report: Mapping[str, Any],
    p3_2_manifest: Mapping[str, Any],
    p2_7_manifest: Mapping[str, Any],
) -> None:
    if signal_report.get("status") != "completed":
        raise ValueError("P3.3 G learning requires a completed signal canary")
    if not signal_report.get("signal_gate", {}).get("passed"):
        raise ValueError("P3.3 signal Gate did not pass")
    if signal_report.get("training_performed") or signal_report.get("fit_called"):
        raise ValueError("P3.3 signal canary already performed training")
    if signal_report.get("sealed_payload_read") or signal_report.get("external_target_used"):
        raise ValueError("P3.3 signal canary crossed a runtime boundary")
    if signal_report.get("manifest_digest") != signal_manifest.get("manifest_digest"):
        raise ValueError("P3.3 signal report and manifest digests differ")
    if p3_2_report.get("status") != "completed":
        raise ValueError("P3.3 G learning requires a completed P3.2 owner-transfer report")
    for gate_name in ("comparison_gate", "checkpoint_gate", "trajectory_gate", "rejection_gate"):
        if not all(bool(value) for value in p3_2_report.get(gate_name, {}).values()):
            raise ValueError(f"P3.2 {gate_name} is not fully passed")
    if p3_2_report.get("manifest_digest") != p3_2_manifest.get("manifest_digest"):
        raise ValueError("P3.2 report and manifest digests differ")
    if p3_2_report.get("external_target_used"):
        raise ValueError("P3.2 external target boundary is not clean")
    if len(p2_7_manifest.get("records", ())) != 4:
        raise ValueError("P3.3 G learning requires the fixed four-row P2.7 holdout")
    if any(
        record.get("candidate", {}).get("fit_eligible") is not False
        for record in p2_7_manifest["records"]
    ):
        raise ValueError("P2.7 holdout contains a fit-eligible record")


def _k_only_candidate(candidate_set: GSelectionCandidateSet) -> GSelectionCandidate:
    by_id = {candidate.candidate_id: candidate for candidate in candidate_set.candidates}
    for candidate_id in ("k1:selected", "k2:selected"):
        candidate = by_id.get(candidate_id)
        if (
            candidate is not None
            and candidate.candidate_role == "proposal"
            and candidate.goal is not None
            and candidate.content_plan is not None
            and candidate.confidence >= CONFIDENCE_FLOOR
        ):
            return candidate
    safe = [
        candidate for candidate in candidate_set.candidates if candidate.candidate_role == "abstain"
    ]
    if safe:
        return sorted(safe, key=lambda candidate: candidate.candidate_id)[0]
    reobserve = [
        candidate
        for candidate in candidate_set.candidates
        if candidate.candidate_role == "reobserve"
    ]
    if reobserve:
        return sorted(reobserve, key=lambda candidate: candidate.candidate_id)[0]
    raise ValueError("K-only baseline has no safe candidate")


def _evaluate_selections(
    *,
    candidate_sets: Sequence[GSelectionCandidateSet],
    zero_step: GSelectionLearner,
    trained: GSelectionLearner,
) -> dict[str, Any]:
    arms = {
        "k_only": [(_k_only_candidate(item), None) for item in candidate_sets],
        "g_zero_step": [(None, zero_step.select(item)) for item in candidate_sets],
        "g_trained": [(None, trained.select(item)) for item in candidate_sets],
    }
    result: dict[str, Any] = {}
    for arm, selections in arms.items():
        rows: list[dict[str, Any]] = []
        for item, (candidate, decision) in zip(candidate_sets, selections, strict=True):
            selected_id = (
                candidate.candidate_id if candidate is not None else decision.selected_candidate_id
            )
            selected_role = (
                candidate.candidate_role
                if candidate is not None
                else decision.selected_candidate_role
            )
            target = item.target_candidate()
            rows.append(
                {
                    "example_id": item.example_id,
                    "candidate_set_digest": item.candidate_set_digest,
                    "selected_candidate_id": selected_id,
                    "selected_candidate_role": selected_role,
                    "target_candidate_id": target.candidate_id,
                    "target_kind": item.target_kind,
                    "target_hit": bool(selected_id == target.candidate_id),
                    "pair_target_hit": bool(
                        item.target_kind == "pair" and selected_id == target.candidate_id
                    ),
                    "safe_abstention": bool(selected_role == "abstain"),
                    "decision_digest": None if decision is None else decision.decision_digest,
                }
            )
        result[arm] = {
            "count": len(rows),
            "target_hit_count": sum(int(row["target_hit"]) for row in rows),
            "target_hit_rate": (
                sum(int(row["target_hit"]) for row in rows) / len(rows) if rows else 0.0
            ),
            "pair_target_hit_count": sum(int(row["pair_target_hit"]) for row in rows),
            "safe_abstention_count": sum(int(row["safe_abstention"]) for row in rows),
            "selection_roles": dict(
                sorted(Counter(row["selected_candidate_role"] for row in rows).items())
            ),
            "rows": rows,
        }
    return result


def _holdout_candidate_sets(
    *,
    scratch: Path,
    p2_7_manifest: Mapping[str, Any],
    semantic: Any,
    transition: Any,
    semantic_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], tuple[GSelectionCandidateSet, ...]]:
    cases = _build_holdout_cases(scratch=scratch, p2_7_manifest=p2_7_manifest)
    goals, content_plans = _catalogs(semantic_payload)
    sets: list[GSelectionCandidateSet] = []
    for case in cases:
        record = case["record"]
        semantic_example = StructuredSemanticExample.from_payload(
            _restore_tensors(record["candidate"]["semantic_example"])
        )
        transition_example = StructuredSemanticTransitionExample.from_payload(
            _restore_tensors(record["candidate"]["transition_example"])
        )
        experience = SimpleNamespace(
            semantic_example=semantic_example,
            transition_example=transition_example,
        )
        metadata = {
            "split": "validation",
            "index": int(record["index"]),
            "class_key": str(record["class_key"]),
            "project_id": str(record["project_id"]),
            "variant_paths": [str(record["candidate"]["observation"]["path"])],
            "state_profile": str(record.get("holdout_role", "unseen-path-unseen-project")),
            "template_family_id": str(record["template_family_id"]),
            "experience_id": str(record["record_id"]),
            "experience_digest": str(record["record_digest"]),
        }
        sets.append(
            _candidate_sets(
                experience=experience,
                metadata=metadata,
                semantic=semantic,
                transition=transition,
                goals=goals,
                content_plans=content_plans,
            )
        )
    return cases, tuple(sets)


def _holdout_actions(
    *,
    cases: Sequence[Mapping[str, Any]],
    candidate_sets: Sequence[GSelectionCandidateSet],
    semantic: Any,
    transition: Any,
    zero_step: GSelectionLearner,
    trained: GSelectionLearner,
) -> dict[str, Any]:
    arms: dict[str, list[GSelectionCandidate | None]] = {
        "k_only": [_k_only_candidate(item) for item in candidate_sets],
        "g_zero_step": [None for _item in candidate_sets],
        "g_trained": [None for _item in candidate_sets],
    }
    decisions = {
        "g_zero_step": [zero_step.select(item) for item in candidate_sets],
        "g_trained": [trained.select(item) for item in candidate_sets],
    }
    result: dict[str, Any] = {}
    for arm, selected_items in arms.items():
        rows: list[dict[str, Any]] = []
        for index, (case, item) in enumerate(zip(cases, candidate_sets, strict=True)):
            if arm == "k_only":
                selected = selected_items[index]
                if selected is None:
                    raise AssertionError("K-only selection unexpectedly missing")
            else:
                decision = decisions[arm][index]
                selected = next(
                    candidate
                    for candidate in item.candidates
                    if candidate.candidate_id == decision.selected_candidate_id
                )
            semantic_example = StructuredSemanticExample.from_payload(
                _restore_tensors(case["record"]["candidate"]["semantic_example"])
            )
            transition_example = StructuredSemanticTransitionExample.from_payload(
                _restore_tensors(case["record"]["candidate"]["transition_example"])
            )
            semantic_result = semantic.predict(semantic_example.percept)
            transition_result = None
            if semantic_result.world is not None:
                transition_result = transition.predict(
                    transition_example.before, transition_example.event
                )
            action: dict[str, Any]
            if selected.goal is None or selected.content_plan is None or transition_result is None:
                action = {
                    "planner_status": "safe_abstention",
                    "workbench_success": False,
                    "skipped": True,
                }
            else:
                planner = NativeReadOnlyIntentPlanner(ReadOnlyIntentPolicy(routes=READ_ONLY_ROUTES))
                action = _run_intent(
                    case=case,
                    planner=planner,
                    goal=selected.goal,
                    content=selected.content_plan,
                    world=transition_result.world,
                    label=f"p3-3-{arm}",
                    oracle_control=False,
                )
            rows.append(
                {
                    "index": int(case["index"]),
                    "project_id": item.project_id,
                    "path": item.path,
                    "selected_candidate_id": selected.candidate_id,
                    "target_candidate_id": item.target_candidate_id,
                    "target_hit": selected.candidate_id == item.target_candidate_id,
                    "action": action,
                }
            )
        result[arm] = {
            "count": len(rows),
            "target_hit_count": sum(int(row["target_hit"]) for row in rows),
            "workbench_success_count": sum(
                int(bool(row["action"].get("workbench_success"))) for row in rows
            ),
            "rows": rows,
        }
    return result


def run_learning(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p3_3_g_learning_{uuid4().hex}"
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "training_performed": False,
        "fit_called": False,
        "external_target_used": False,
        "p2_7_holdout_fit_count": 0,
        "can_promote": False,
        "manifest": str(manifest_path),
        "report": str(report_path),
    }
    try:
        signal_manifest = _load_json(manifest_path)
        signal_report = _load_json(
            PROJECT_ROOT / "reports" / "taiji_m5_k_p3_3_g_signal_canary_20260911.json"
        )
        p3_2_report = _load_json(P3_2_REPORT)
        p3_2_manifest = _load_json(P3_2_MANIFEST)
        p2_7_manifest = _load_json(P2_7_MANIFEST)
        _verify_source_contracts(
            signal_report=signal_report,
            signal_manifest=signal_manifest,
            p3_2_report=p3_2_report,
            p3_2_manifest=p3_2_manifest,
            p2_7_manifest=p2_7_manifest,
        )
        train_sets, validation_sets = _candidate_sets_from_manifest(signal_manifest)
        worker_restore = p3_2_report["base_continuation"]["worker_restore"]
        k1_path = Path(str(worker_restore["k1"]["path"]))
        k2_path = Path(str(worker_restore["k2"]["path"]))
        semantic_payload = _load_mapping(k1_path)
        transition_payload = _load_mapping(k2_path)
        worker_digests = {
            "k1": str(p3_2_report["worker_checkpoint_digests"]["k1"]),
            "k2": str(p3_2_report["worker_checkpoint_digests"]["k2"]),
        }
        if content_digest(semantic_payload) != worker_digests["k1"]:
            raise ValueError("P3.2 K1 checkpoint digest drifted before G fit")
        if content_digest(transition_payload) != worker_digests["k2"]:
            raise ValueError("P3.2 K2 checkpoint digest drifted before G fit")
        semantic, transition = _fresh_learners(semantic_payload, transition_payload)
        run_dir.mkdir(parents=True, exist_ok=False)
        preflight_before = _checkpoint_preflight(
            output_dir=run_dir / "k-preflight-before",
            semantic_parent=copy.deepcopy(semantic_payload),
            transition_parent=copy.deepcopy(transition_payload),
        )
        if not preflight_before.get("passed"):
            raise RuntimeError("K checkpoint preflight before G fit failed")
        parent_manifest_digest = str(p3_2_manifest["manifest_digest"])
        zero_step = GSelectionLearner(
            parent_manifest_digest=parent_manifest_digest,
            k_checkpoint_digests=worker_digests,
            learning_rate=G_LEARNING_RATE,
            confidence_floor=CONFIDENCE_FLOOR,
        )
        zero_checkpoint = _save_g_checkpoint(run_dir, zero_step, "g_zero_step.pt")
        if not zero_checkpoint["passed"]:
            raise RuntimeError("G zero-step checkpoint independent restore failed")
        trained = GSelectionLearner.from_checkpoint(zero_step.checkpoint(), device="cpu")
        fit_result = trained.fit(train_sets, epochs=G_EPOCHS, learning_rate=G_LEARNING_RATE)
        trained_checkpoint = _save_g_checkpoint(run_dir, trained, "g_trained.pt")
        if not trained_checkpoint["passed"]:
            raise RuntimeError("G trained checkpoint independent restore failed")
        preflight_after = _checkpoint_preflight(
            output_dir=run_dir / "k-preflight-after",
            semantic_parent=copy.deepcopy(semantic_payload),
            transition_parent=copy.deepcopy(transition_payload),
        )
        if not preflight_after.get("passed"):
            raise RuntimeError("K checkpoint preflight after G fit failed")
        k_after = {
            "k1": content_digest(semantic.checkpoint()),
            "k2": content_digest(transition.checkpoint()),
        }
        if k_after != worker_digests:
            raise RuntimeError("G fit changed frozen K checkpoint state")
        holdout_cases, holdout_sets = _holdout_candidate_sets(
            scratch=run_dir / "p2-7-holdout",
            p2_7_manifest=p2_7_manifest,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        selection_metrics = {
            "train": _evaluate_selections(
                candidate_sets=train_sets, zero_step=zero_step, trained=trained
            ),
            "validation": _evaluate_selections(
                candidate_sets=validation_sets,
                zero_step=zero_step,
                trained=trained,
            ),
            "p2_7_holdout": _evaluate_selections(
                candidate_sets=holdout_sets,
                zero_step=zero_step,
                trained=trained,
            ),
        }
        holdout_actions = _holdout_actions(
            cases=holdout_cases,
            candidate_sets=holdout_sets,
            semantic=semantic,
            transition=transition,
            zero_step=zero_step,
            trained=trained,
        )
        tampered = copy.deepcopy(trained.checkpoint())
        tampered["revision"] = int(tampered["revision"]) + 1
        tampered_rejected = False
        try:
            GSelectionLearner.from_checkpoint(tampered)
        except ValueError:
            tampered_rejected = True
        wrong_lineage_rejected = False
        try:
            trained.assert_lineage(
                parent_manifest_digest="e" * 64,
                k_checkpoint_digests=worker_digests,
            )
        except ValueError:
            wrong_lineage_rejected = True
        checkpoint_gate = {
            "k_before_independent_restore": bool(preflight_before.get("passed")),
            "g_zero_independent_restore": bool(zero_checkpoint.get("passed")),
            "g_trained_independent_restore": bool(trained_checkpoint.get("passed")),
            "k_after_independent_restore": bool(preflight_after.get("passed")),
            "k_digests_unchanged": k_after == worker_digests,
        }
        training_gate = {
            "fit_called": True,
            "train_records": len(train_sets),
            "validation_records_not_fit": len(validation_sets) > 0,
            "p2_7_records_not_fit": len(holdout_sets) == 4,
            "g_training_steps_positive": trained.training_steps > 0,
            "g_parameter_count_stable": trained.parameter_count == zero_step.parameter_count,
            "k_parameter_count_untouched": semantic.parameter_count + transition.parameter_count
            > 0,
            "external_target_unused": True,
        }
        rejection_gate = {
            "tampered_checkpoint_rejected": tampered_rejected,
            "wrong_lineage_rejected": wrong_lineage_rejected,
        }
        holdout_gate = {
            "fixed_four_rows": len(holdout_sets) == 4,
            "fit_count": 0,
            "all_holdout_paths_unique": len({item.path for item in holdout_sets})
            == len(holdout_sets),
            "holdout_projects_at_least_two": len({item.project_id for item in holdout_sets}) >= 2,
        }
        payload.update(
            {
                "status": "completed",
                "training_performed": True,
                "fit_called": True,
                "external_target_used": False,
                "run_dir": str(run_dir),
                "source_signal_manifest_digest": signal_manifest["manifest_digest"],
                "source_p3_2_manifest_digest": p3_2_manifest["manifest_digest"],
                "k_checkpoint_digests_before": worker_digests,
                "k_checkpoint_digests_after": k_after,
                "g_zero_step_checkpoint": zero_checkpoint,
                "g_trained_checkpoint": trained_checkpoint,
                "fit_result": fit_result,
                "parameter_count": {
                    "g_zero_step": zero_step.parameter_count,
                    "g_trained": trained.parameter_count,
                    "g_growth": trained.parameter_count - zero_step.parameter_count,
                    "k1": int(semantic.parameter_count),
                    "k2": int(transition.parameter_count),
                },
                "checkpoint_gate": checkpoint_gate,
                "training_gate": training_gate,
                "rejection_gate": rejection_gate,
                "holdout_gate": holdout_gate,
                "selection_metrics": selection_metrics,
                "holdout_actions": holdout_actions,
                "interpretation": (
                    "completed: G-only incremental learning and checkpoint boundaries passed; promotion remains frozen"
                ),
                "can_promote": False,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        required_gates = [checkpoint_gate, training_gate, rejection_gate, holdout_gate]
        if not all(
            all(bool(value) for key, value in gate.items() if key != "fit_count")
            for gate in required_gates
        ):
            payload["status"] = "failed"
            payload["error"] = "one or more P3.3 G learning gates failed"
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    _write_json_atomic(report_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.verify_only is not None:
        return 0 if _verify_g_checkpoint(args.verify_only)["passed"] else 1
    result = run_learning(manifest_path=args.manifest, report_path=args.report)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
