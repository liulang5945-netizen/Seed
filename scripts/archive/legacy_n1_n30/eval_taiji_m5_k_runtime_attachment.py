"""M5.K default-runtime attachment acceptance runner.

Executes the preregistered attachment step: the P4.14 joint-course artifacts
are attached to a real ``SeedRuntime`` through the runtime-owned consumption
contract (``api/taiji_runtime_attachment.py``), and the frozen cohorts are
re-evaluated through the runtime's typed readout surface / runtime-owned
learner instances.  Acceptance requires field-for-field equality with the
P4.14 recorded values in all 4 cells, fail-closed refusal of tampered or
mixed manifests, detach/re-attach behavioral equivalence, zero
non-interference, and absolute resource budgets.

``fit_called`` stays False; nothing on disk is written except the report and
scratch data under the run directory.  Preregistration:
``plans/reference/M5_K_DEFAULT_RUNTIME_ATTACHMENT_PREREGISTRATION_20260912.md``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import taiji_runtime_attachment as attachment_module  # noqa: E402
from api.seed_runtime import SeedRuntime  # noqa: E402
from scripts.training.eval_taiji_m5_k_default_runtime_rollout_review import (  # noqa: E402
    _dict_diff,
    _rebuild_batch_cohorts,
)
from scripts.training.eval_taiji_m5_k_p2_6_novel_learning import (  # noqa: E402
    _score_arm as _score_phase_k_arm,
)
from scripts.training.eval_taiji_m5_k_p2_validation_pilot import (  # noqa: E402
    MODEL_SEED,
    P1_MANIFEST,
    WORKER_ROOT,
    _context,
)
from scripts.training.eval_taiji_m5_k_p3_5_g_learning import (  # noqa: E402
    _load_json,
    _load_mapping,
)
from scripts.training.eval_taiji_m5_k_p4_6_functional_parent_objective import (  # noqa: E402
    _metric_summary,
)
from scripts.training.eval_taiji_m5_k_p4_14_joint_course import (  # noqa: E402
    _evaluate_records,
    _strip_rows,
    _write_json_atomic,
)

REPORT_FORMAT = "taiji-m5-k-runtime-attachment-v1"
VERSION = 1
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_runtime_attachment_20260912.json"
ATTACHMENT_MANIFEST = attachment_module.ATTACHMENT_MANIFEST
V7_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_axis_scorecard_v7_20260912.json"
REVIEW_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_default_runtime_rollout_review_20260911.json"
P4_14_REPORT = attachment_module.P4_14_REPORT
P4_4_MANIFEST = (
    PROJECT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m5_k_p4_4_retention_identity_calibration_manifest_v1.json"
)
P4_13_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_13_promotion_course_manifest_v1.json"
)
P2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_validation_pilot_v2_20260910.json"
RUNTIME_DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "seed_corpus.pt"

SEEDS = (0, 1)
BATCHES = (0, 1)
CELLS = tuple((batch, seed) for batch in BATCHES for seed in SEEDS)
ROLLOUT_SPLITS = ("validation", "holdout", "retention-sibling", "retention-newtask")
RESOURCE_CAPS = {
    "cell_attach_seconds": 60.0,
    "acceptance_total_seconds": 600.0,
}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refusal_probe(runtime: SeedRuntime, manifest_path: Path, cell_index: int) -> dict[str, Any]:
    """Attach must be refused and the runtime must stay unattached."""

    refused = False
    try:
        runtime.attach_k_g_state(manifest_path, cell_index)
    except attachment_module.AttachmentRefused:
        refused = True
    status = runtime.k_g_attachment_status()
    return {
        "probe": manifest_path.name,
        "refused": refused,
        "runtime_still_unattached": status["attached"] is False,
        "refusal_recorded": bool(status.get("recent_refusals")),
        "passed": refused and status["attached"] is False and bool(status.get("recent_refusals")),
    }


def _runtime_phase_evaluations(
    state: Any,
    cohorts: dict[str, Any],
    recorded_cell: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Evaluate the frozen cohorts through the runtime-owned consumer."""

    phase_k_scores = _score_phase_k_arm(
        semantic=state.semantic,
        transition=state.transition,
        p1_cases=cohorts["p1_cases"],
        novel_cases=cohorts["validation_cases"],
        novel_records=cohorts["validation_records"],
    )
    phase_k = {
        "p1_validation_diffs": _dict_diff(
            _strip_rows(phase_k_scores["p1_validation"]),
            recorded_cell["phase_k"]["p1_validation_interleaved_summary"],
        ),
        "p2_6_novel_validation_diffs": _dict_diff(
            _strip_rows(phase_k_scores["p2_6_novel_validation"]),
            recorded_cell["phase_k"]["interleaved_novel_scores"],
        ),
        "parameter_count_total": int(phase_k_scores["parameter_count"]["total"]),
        "parameter_count_total_match": int(phase_k_scores["parameter_count"]["total"])
        == int(recorded_cell["parameter_count"]["k_total"]),
    }
    phase_k_match = (
        not phase_k["p1_validation_diffs"]
        and not phase_k["p2_6_novel_validation_diffs"]
        and phase_k["parameter_count_total_match"]
    )
    metrics = {
        split: _metric_summary(_evaluate_records(records, state.g_learner))
        for split, records in cohorts["split_records"].items()
    }
    phase_g_diffs = {
        split: _dict_diff(metrics[split], recorded_cell["phase_g"]["metrics"][split])
        for split in ROLLOUT_SPLITS
    }
    phase_g_match = not any(phase_g_diffs.values())
    phase_g: dict[str, Any] = {
        "metrics": metrics,
        "diffs": phase_g_diffs,
        "match": phase_g_match,
    }
    behavior_match: bool = bool(phase_k_match and phase_g_match)
    return phase_k, phase_g, behavior_match


def _typed_surface_smoke(state: Any, cohorts: dict[str, Any]) -> dict[str, bool]:
    """The typed readout surface must route to the attached consumer state."""

    candidate_set = cohorts["split_records"]["validation"][0]["candidate_set"]
    g_decision = state.g_select(candidate_set)
    g_direct = state.g_learner.select(candidate_set)
    case = cohorts["p1_cases"][0]
    percept = case["experience"].semantic_example.percept
    k1_via_surface = state.k1_predict(percept)
    k1_direct = state.semantic.predict(percept)
    return {
        "g_select_routed": g_decision.selected_candidate_id == g_direct.selected_candidate_id,
        "k1_predict_routed": k1_via_surface.status == k1_direct.status,
    }


def _k1_payload_for_cell(manifest: dict[str, Any], index: int) -> dict[str, Any]:
    path = Path(str(manifest["cells"][index]["artifacts"]["k1"]["path"]))
    return _load_mapping(path)


def _run(*, report_path: Path = DEFAULT_REPORT) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = PROJECT_ROOT / "output" / f"taiji_m5_k_runtime_attachment_{uuid4().hex}"
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "training_performed": False,
        "fit_called": False,
        "sealed_payload_read": False,
        "growth_admitted": False,
        "can_promote": False,
        "report": str(report_path),
    }
    try:
        v7 = _load_json(V7_REPORT)
        review = _load_json(REVIEW_REPORT)
        manifest = _load_json(ATTACHMENT_MANIFEST)

        # ---- Entry conditions (scorecard v7 frozen boundary).
        if v7.get("format") != "taiji-m5-k-axis-scorecard-v7":
            raise ValueError("the attachment step requires the scorecard v7 report")
        if v7["verdict"].get("promotion_review_recommended") is not True:
            raise ValueError("the attachment step requires the approved A8 review")
        gates_v7 = v7["promotion_gates"]
        if gates_v7.get("default_runtime_owner_attached") or gates_v7.get(
            "learning_mechanism_attached_default_runtime"
        ):
            raise ValueError("the attachment gates must still be false")
        if v7["verdict"]["promotion_gate"] or v7["verdict"]["can_promote"]:
            raise ValueError("the attachment step cannot start from an open gate")
        if review.get("outcome") != "default_runtime_rollout_review_supported":
            raise ValueError("the attachment step requires the passed rollout review")
        manifest_digest = str(manifest["manifest_digest"])
        if manifest_digest != str(review.get("attachment_manifest_digest")):
            raise ValueError("attachment manifest drifted from the review report")
        if manifest_digest != str(v7["rollout_review_evidence"]["attachment_manifest_digest"]):
            raise ValueError("attachment manifest drifted from scorecard v7")

        # ---- Baselines before anything runs.
        run_dir.mkdir(parents=True, exist_ok=False)
        runtime = SeedRuntime.load()
        baseline = {
            "default_checkpoint_sha256": _file_sha256(RUNTIME_DEFAULT_CHECKPOINT),
            "model_tick": int(runtime.model.tick),
            "model_parameter_count": int(runtime.model.parameter_count()),
            "p4_14_report_sha256": _file_sha256(P4_14_REPORT),
        }
        refusal_probes: list[dict[str, Any]] = []
        scratch_manifest_dir = run_dir / "refusal-probes"
        scratch_manifest_dir.mkdir(parents=True, exist_ok=True)

        # Probe 1: corrupted artifact digest.
        corrupt = copy.deepcopy(dict(manifest))
        corrupt["cells"] = copy.deepcopy(manifest["cells"])
        corrupt["cells"][0]["artifacts"]["g"]["digest"] = "0" * 64
        corrupt_path = scratch_manifest_dir / "corrupt_digest.json"
        corrupt_path.write_text(json.dumps(corrupt), encoding="utf-8")
        refusal_probes.append(_refusal_probe(runtime, corrupt_path, 0))

        # Probe 2: missing artifact file.
        missing = copy.deepcopy(dict(manifest))
        missing["cells"] = copy.deepcopy(manifest["cells"])
        missing["cells"][0]["artifacts"]["k1"]["path"] = str(scratch_manifest_dir / "missing_k1.pt")
        missing_path = scratch_manifest_dir / "missing_artifact.json"
        missing_path.write_text(json.dumps(missing), encoding="utf-8")
        refusal_probes.append(_refusal_probe(runtime, missing_path, 0))

        # Probe 3: wrong-cell mixing.
        mixed = copy.deepcopy(dict(manifest))
        mixed["cells"] = copy.deepcopy(manifest["cells"])
        mixed["cells"][0]["artifacts"]["g"]["digest"] = mixed["cells"][1]["artifacts"]["g"][
            "digest"
        ]
        mixed_path = scratch_manifest_dir / "wrong_cell_mixing.json"
        mixed_path.write_text(json.dumps(mixed), encoding="utf-8")
        refusal_probes.append(_refusal_probe(runtime, mixed_path, 0))
        refusal_gate = {
            "probes": refusal_probes,
            "all_refused_and_unattached": all(probe["passed"] for probe in refusal_probes),
        }

        # ---- Cohort re-materialization inputs (same parent as P4.14).
        p1_manifest = _load_json(P1_MANIFEST)
        p2_report = _load_json(P2_REPORT)
        p4_4_manifest = _load_json(P4_4_MANIFEST)
        p4_13_manifest = _load_json(P4_13_MANIFEST)
        _artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT,
            model_seed=MODEL_SEED,
        )
        if p2_report["contract"]["parent_checkpoint_digest"] != parent_digest:
            raise ValueError("P2 parent checkpoint digest drifted")
        p4_14_manifest = _load_json(attachment_module.P4_14_MANIFEST)
        p4_14_report = _load_json(P4_14_REPORT)

        # ---- Per-cell acceptance through the runtime attachment.
        cell_results: list[dict[str, Any]] = []
        batch_cohorts: dict[int, dict[str, Any]] = {}
        for index, (batch, seed) in enumerate(CELLS):
            cell_started = time.perf_counter()
            attach_started = time.perf_counter()
            runtime.attach_k_g_state(ATTACHMENT_MANIFEST, index)
            attach_wall = time.perf_counter() - attach_started
            state = runtime._k_g_attachment
            if state is None or int(state.cell_index) != index:
                raise RuntimeError(f"cell {index} attachment state missing")
            if batch not in batch_cohorts:
                # The first cell of each batch materializes the frozen
                # cohorts with the runtime-owned post-K learners.
                batch_cohorts[batch] = _rebuild_batch_cohorts(
                    batch=batch,
                    scratch=run_dir / f"batch-{batch}",
                    p4_14_manifest=p4_14_manifest,
                    p4_4_manifest=p4_4_manifest,
                    p4_13_manifest=p4_13_manifest,
                    p1_manifest=p1_manifest,
                    p2_report=p2_report,
                    parent_digest=parent_digest,
                    bundle=bundle,
                    projector=projector,
                    semantic=state.semantic,
                    transition=state.transition,
                    semantic_payload=_k1_payload_for_cell(manifest, index),
                )
            smoke = _typed_surface_smoke(state, batch_cohorts[batch])
            p4_14_cell = p4_14_report["cell_results"][index]
            recorded_cell = {
                "phase_k": p4_14_cell["phase_k"],
                "phase_g": p4_14_cell["phase_g"],
                "parameter_count": p4_14_cell["parameter_count"],
            }
            phase_k, phase_g, behavior_match = _runtime_phase_evaluations(
                state, batch_cohorts[batch], recorded_cell
            )
            # Rollback: detach -> re-attach must reproduce the behavior.
            runtime.detach_k_g_state()
            runtime.attach_k_g_state(ATTACHMENT_MANIFEST, index)
            state_again = runtime._k_g_attachment
            phase_k_again, phase_g_again, _ = _runtime_phase_evaluations(
                state_again, batch_cohorts[batch], recorded_cell
            )
            rollback_gate = {
                "phase_k_equal": phase_k_again == phase_k,
                "phase_g_equal": phase_g_again == phase_g,
                "passed": phase_k_again == phase_k and phase_g_again == phase_g,
            }
            cell_wall = time.perf_counter() - cell_started
            caps_ok = (
                attach_wall <= RESOURCE_CAPS["cell_attach_seconds"]
                and cell_wall <= RESOURCE_CAPS["cell_attach_seconds"] * 10
            )
            cell_results.append(
                {
                    "cell_index": index,
                    "batch": batch,
                    "seed": seed,
                    "attach_wall_seconds": round(attach_wall, 3),
                    "cell_wall_seconds": round(cell_wall, 3),
                    "typed_surface_smoke": smoke,
                    "phase_k": phase_k,
                    "phase_g": phase_g,
                    "behavior_match": behavior_match,
                    "rollback_gate": rollback_gate,
                    "caps_passed": caps_ok,
                    "passes_all": (
                        behavior_match
                        and all(smoke.values())
                        and rollback_gate["passed"]
                        and caps_ok
                    ),
                }
            )
        runtime.detach_k_g_state()

        # ---- Non-interference after everything.
        artifact_reverify = [
            attachment_module.verify_attachment(ATTACHMENT_MANIFEST, index)["passed"]
            for index in range(len(CELLS))
        ]
        non_interference = {
            "default_checkpoint_sha256_unchanged": baseline["default_checkpoint_sha256"]
            == _file_sha256(RUNTIME_DEFAULT_CHECKPOINT),
            "model_tick_unchanged": baseline["model_tick"] == int(runtime.model.tick),
            "model_parameter_count_unchanged": baseline["model_parameter_count"]
            == int(runtime.model.parameter_count()),
            "p4_14_report_unchanged": baseline["p4_14_report_sha256"] == _file_sha256(P4_14_REPORT),
            "research_artifacts_still_verified": all(artifact_reverify),
        }

        total_wall = time.perf_counter() - started
        resource_audit = {
            "caps": dict(RESOURCE_CAPS),
            "per_cell_attach_seconds": [cell["attach_wall_seconds"] for cell in cell_results],
            "cell_caps_passed": all(
                cell["attach_wall_seconds"] <= RESOURCE_CAPS["cell_attach_seconds"]
                for cell in cell_results
            ),
            "total_wall_seconds": round(total_wall, 3),
            "total_caps_passed": total_wall <= RESOURCE_CAPS["acceptance_total_seconds"],
        }
        mechanical_gate = {
            "refusal_gate": refusal_gate["all_refused_and_unattached"],
            "cohort_digests_match": all(
                cohorts["candidate_digests_match"] and all(cohorts["novel_digest_checks"].values())
                for cohorts in batch_cohorts.values()
            ),
            "non_interference": all(non_interference.values()),
            "resource_caps_passed": (
                resource_audit["cell_caps_passed"] and resource_audit["total_caps_passed"]
            ),
        }
        consumption_gate = {
            "behavior_all_match": all(cell["behavior_match"] for cell in cell_results),
            "rollback_all_passed": all(cell["rollback_gate"]["passed"] for cell in cell_results),
            "typed_surface_routed": all(
                all(cell["typed_surface_smoke"].values()) for cell in cell_results
            ),
        }
        if not all(mechanical_gate.values()):
            outcome = "mechanical_failure"
            status = "failed"
        elif not consumption_gate["rollback_all_passed"]:
            outcome = "rollback_unverified"
            status = "completed"
        elif not refusal_gate["all_refused_and_unattached"]:
            outcome = "runtime_attachment_refusal_gap"
            status = "completed"
        elif not (
            consumption_gate["behavior_all_match"] and consumption_gate["typed_surface_routed"]
        ):
            outcome = "runtime_attachment_consumption_gap"
            status = "completed"
        else:
            outcome = "runtime_attachment_supported"
            status = "completed"
        supported = outcome == "runtime_attachment_supported"
        payload.update(
            {
                "status": status,
                "training_performed": False,
                "fit_called": False,
                "run_dir": str(run_dir),
                "attachment_manifest_digest": manifest_digest,
                "baseline": baseline,
                "refusal_gate": refusal_gate,
                "cell_results": cell_results,
                "consumption_gate": consumption_gate,
                "non_interference": non_interference,
                "mechanical_gate": mechanical_gate,
                "resource_audit": resource_audit,
                "outcome": outcome,
                "runtime_attachment_supported": supported,
                "experiment_passed": bool(supported),
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    (
                        f"completed: outcome={outcome}; the P4.14 joint-course "
                        "state is attachable to the product runtime through the "
                        "runtime-owned fail-closed contract with field-for-field "
                        "behavioral reproduction; growth and the final promotion "
                        "decision remain fail-closed"
                    )
                    if status == "completed"
                    else "failed: mechanical gates did not pass; nothing is admitted"
                ),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    finally:
        for batch_dir in Path(run_dir).glob("batch-*"):
            shutil.rmtree(batch_dir / "phase-k-data", ignore_errors=True)
            shutil.rmtree(batch_dir / "p1", ignore_errors=True)
            shutil.rmtree(batch_dir / "p1-cases", ignore_errors=True)
            shutil.rmtree(batch_dir / "data", ignore_errors=True)
    _write_json_atomic(report_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = _run(report_path=args.report)
    print(
        json.dumps(
            {
                "report": str(args.report),
                "status": result.get("status"),
                "outcome": result.get("outcome"),
                "experiment_passed": result.get("experiment_passed"),
                "runtime_attachment_supported": result.get("runtime_attachment_supported"),
                "growth_admitted": result.get("growth_admitted"),
                "can_promote": result.get("can_promote"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
