"""M5.K default-runtime rollout review runner (consumption contract).

The P4.14 joint course proved the K-worker continuation and the G solver
mechanism inside the research harness (in-memory learners).  The default
runtime would consume the same artifacts through a different path: disk
artifacts -> manifest-driven fail-closed load -> independent-process
restore -> behavior rollout -> rollback.  This review freezes that contract
as a content-addressed attachment manifest and proves, cell by cell, that
the consumption path reproduces the P4.14 recorded gate values exactly.

No training happens here: ``fit_called`` stays False, no epoch, no
projection re-run; cohorts are re-materialized deterministically and
verified against the P4.14 manifest digests, while checkpoints are only
ever loaded and verified.  Preregistration:
``plans/reference/M5_K_DEFAULT_RUNTIME_ROLLOUT_REVIEW_PREREGISTRATION_20260911.md``.
Never admits growth or promotion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m5_k_p2_3_targeted_learning import (  # noqa: E402
    _fresh_learners as _fresh_k_learners,
)
from scripts.training.eval_taiji_m5_k_p2_3_targeted_learning import (
    _independent_restore,
)
from scripts.training.eval_taiji_m5_k_p2_6_novel_learning import (
    _score_arm as _score_phase_k_arm,
)
from scripts.training.eval_taiji_m5_k_p2_output_action_diagnostic import (  # noqa: E402
    _build_validation_cases,
)
from scripts.training.eval_taiji_m5_k_p2_validation_pilot import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    MODEL_SEED,
    P1_MANIFEST,
    PILOT_PER_CLASS,
    WORKER_ROOT,
    _context,
    _rebuild_and_verify_manifest,
    _select_balanced_wake,
)
from scripts.training.eval_taiji_m5_k_p3_5_g_learning import (  # noqa: E402
    _load_json,
    _load_mapping,
)
from scripts.training.eval_taiji_m5_k_p4_6_functional_parent_objective import (  # noqa: E402
    _metric_summary,
)
from scripts.training.eval_taiji_m5_k_p4_13_promotion_course import (  # noqa: E402
    _extended_tamper_rejected,
    _independent_extended_restore,
    _verify_extended_checkpoint,
)
from scripts.training.eval_taiji_m5_k_p4_14_joint_course import (  # noqa: E402
    P414_TRAIN_SPECS,
    P414_VALIDATION_SPECS,
    _digest_without,
    _evaluate_records,
    _k_tamper_rejected,
    _p414_build_record,
    _p414_pressure_split,
    _p414_structured_records,
    _strip_rows,
    _write_json_atomic,
)
from taiji import content_digest  # noqa: E402
from taiji.g_selection_extended import ExtendedGSelectionLearner  # noqa: E402

REPORT_FORMAT = "taiji-m5-k-default-runtime-rollout-review-v1"
ATTACHMENT_FORMAT = "taiji-default-runtime-rollout-attachment-v1"
VERSION = 1
DEFAULT_ATTACHMENT = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_default_runtime_rollout_attachment_v1.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_default_runtime_rollout_review_20260911.json"
)

P4_14_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_14_joint_course_20260911.json"
P4_14_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_14_joint_course_manifest_v1.json"
)
SCORECARD_V5_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_axis_scorecard_v5_20260911.json"
P4_1_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_1_context_contract_manifest_v1.json"
)
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
P3_2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_2_owner_transfer_20260910.json"
P3_5_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_5_g_learning_20260911.json"

RUNTIME_MODULE = PROJECT_ROOT / "api" / "seed_runtime.py"
RUNTIME_DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "seed_corpus.pt"

SEEDS = (0, 1)
BATCHES = (0, 1)
CELLS = tuple((batch, seed) for batch in BATCHES for seed in SEEDS)
ROLLOUT_SPLITS = ("validation", "holdout", "retention-sibling", "retention-newtask")
K_PARAMETER_TOTAL = 5648
G_PARAMETER_COUNT = 17
RESOURCE_CAPS = {
    "cell_rollout_seconds": 120.0,
    "review_total_seconds": 600.0,
}
# Frozen embedded invariants of the consumed checkpoints (P2.2/P3.1
# discipline: the runtime reads them from the checkpoints and verifies,
# never overrides).  The K2 transition learner uses a lower fact threshold
# than the K1 semantic learner; both are pinned from the P4.14 artifacts.
K_INVARIANTS = {
    "k1": {"confidence_floor": 0.55, "fact_threshold": 0.65, "ambiguity_ceiling": 0.12},
    "k2": {"confidence_floor": 0.55, "fact_threshold": 0.55, "ambiguity_ceiling": 0.12},
}
G_SELECTION_MARGIN = 0.05


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_inventory() -> dict[str, Any]:
    text = RUNTIME_MODULE.read_text(encoding="utf-8")
    references = [
        token
        for token in ("g_selection", "k_worker", "ExtendedGSelection", "KWorker")
        if token in text
    ]
    return {
        "runtime_module": str(RUNTIME_MODULE),
        "runtime_module_sha256": _file_sha256(RUNTIME_MODULE),
        "default_checkpoint": str(RUNTIME_DEFAULT_CHECKPOINT),
        "default_checkpoint_sha256": _file_sha256(RUNTIME_DEFAULT_CHECKPOINT),
        "runtime_k_g_references": references,
    }


def _dict_diff(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    if actual == expected:
        return []
    return sorted(
        key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)
    )


# ---------------------------------------------------------------------------
# Attachment contract (taiji-default-runtime-rollout-attachment-v1).
# ---------------------------------------------------------------------------


def _build_attachment(
    *,
    p4_14_report: Mapping[str, Any],
    p4_14_manifest: Mapping[str, Any],
    scorecard_v5: Mapping[str, Any],
    p4_1_manifest: Mapping[str, Any],
    p4_13_manifest: Mapping[str, Any],
    runtime_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for index, (batch, seed) in enumerate(CELLS):
        cell = p4_14_report["cell_results"][index]
        if (int(cell["batch"]), int(cell["seed"])) != (batch, seed):
            raise ValueError("P4.14 cell order drifted from the review matrix")
        k_checkpoint = cell["phase_k"]["checkpoint"]
        g_checkpoint = cell["phase_g"]["checkpoint"]
        cells.append(
            {
                "batch": batch,
                "seed": seed,
                "report_cell_index": index,
                "artifacts": {
                    "k1": {
                        "path": str(k_checkpoint["files"]["k1"]["path"]),
                        "digest": str(cell["phase_k"]["post_k_digests"]["k1"]),
                    },
                    "k2": {
                        "path": str(k_checkpoint["files"]["k2"]["path"]),
                        "digest": str(cell["phase_k"]["post_k_digests"]["k2"]),
                    },
                    "g": {
                        "path": str(g_checkpoint["path"]),
                        "digest": str(g_checkpoint["digest"]),
                    },
                },
            }
        )
    manifest: dict[str, Any] = {
        "format": ATTACHMENT_FORMAT,
        "version": VERSION,
        "source_p4_14_report_digest": content_digest(p4_14_report),
        "source_p4_14_manifest_digest": str(p4_14_manifest["manifest_digest"]),
        "source_scorecard_v5_report_digest": content_digest(scorecard_v5),
        "source_p4_1_manifest_digest": str(p4_1_manifest["manifest_digest"]),
        "source_p4_13_manifest_digest": str(p4_13_manifest["manifest_digest"]),
        "runtime_baseline": dict(runtime_inventory),
        "consumption_contract": {
            "load_order": [
                "attachment_manifest_digest_self_check",
                "p4_14_lineage_check",
                "artifact_digest_check",
                "embedded_invariant_check",
                "independent_process_restore",
                "tamper_and_cell_mixing_rejection",
            ],
            "forbidden": [
                "fit",
                "artifact_or_default_checkpoint_writes",
                "embedded_invariant_override",
                "checkpoint_rematerialization",
            ],
            "post_k_k_digests_shared": {
                "k1": str(cells[0]["artifacts"]["k1"]["digest"]),
                "k2": str(cells[0]["artifacts"]["k2"]["digest"]),
            },
            "resource_caps_absolute": dict(RESOURCE_CAPS),
        },
        "cells": cells,
    }
    manifest["manifest_digest"] = content_digest(manifest)
    return manifest


def _artifact_digest(role: str, payload: Mapping[str, Any]) -> str:
    """Digest of an artifact payload in the convention that produced the pin.

    K worker checkpoints carry no self-digest key, so the pin is the full
    content digest.  The extended G checkpoint embeds its own
    ``checkpoint_digest`` (the digest of the payload without that key), and
    the P4.14 report pinned exactly that value.
    """

    if role == "g":
        return content_digest(
            {name: value for name, value in payload.items() if name != "checkpoint_digest"}
        )
    return content_digest(payload)


def _consume_cell(attachment: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Fail-closed consumption of one cell's artifacts through the contract.

    Raises ``ValueError`` on any digest, invariant, lineage or parameter
    mismatch; the caller decides whether that is a mechanical failure or the
    recorded wrong-cell-mixing probe.
    """

    cell = attachment["cells"][index]
    shared = attachment["consumption_contract"]["post_k_k_digests_shared"]
    payloads: dict[str, Mapping[str, Any]] = {}
    verified: dict[str, Any] = {}
    for role in ("k1", "k2", "g"):
        artifact = cell["artifacts"][role]
        path = Path(str(artifact["path"]))
        payload = _load_mapping(path)
        digest = _artifact_digest(role, payload)
        if digest != str(artifact["digest"]):
            raise ValueError(
                f"cell {index} artifact {role} digest mismatch: "
                f"{digest} != {artifact['digest']}"
            )
        payloads[role] = payload
        verified[role] = {"path": str(path), "digest": digest}
    if verified["k1"]["digest"] != str(shared["k1"]) or verified["k2"]["digest"] != str(
        shared["k2"]
    ):
        raise ValueError("cell post-K workers drift from the shared anchors")
    # The K parameter total is a pinned invariant of the contract; the
    # learners' own from_checkpoint also revalidates their internal digests.
    k1_learner, k2_learner = _fresh_k_learners(payloads["k1"], payloads["k2"])
    invariant_checks: dict[str, bool] = {}
    for role in ("k1", "k2"):
        for name, expected in K_INVARIANTS[role].items():
            actual = payloads[role].get(name)
            invariant_checks[f"{role}_{name}"] = actual is not None and float(actual) == float(
                expected
            )
    invariant_checks["k_parameter_total"] = (
        int(k1_learner.parameter_count + k2_learner.parameter_count) == K_PARAMETER_TOTAL
    )
    g_payload = payloads["g"]
    invariant_checks["g_confidence_floor"] = float(g_payload["confidence_floor"]) == float(
        K_INVARIANTS["k1"]["confidence_floor"]
    )
    invariant_checks["g_selection_margin"] = (
        float(g_payload["selection_margin"]) == G_SELECTION_MARGIN
    )
    invariant_checks["g_parameter_count"] = int(g_payload["parameter_count"]) == G_PARAMETER_COUNT
    return {
        "payloads": payloads,
        "artifact_verification": verified,
        "invariant_checks": invariant_checks,
    }


def _cell_mixing_rejected(attachment: Mapping[str, Any]) -> bool:
    """A manifest that pins cell 1's G artifact into cell 0 must be rejected."""

    mixed = copy.deepcopy(dict(attachment))
    mixed["cells"][0]["artifacts"]["g"]["digest"] = mixed["cells"][1]["artifacts"]["g"]["digest"]
    try:
        _consume_cell(mixed, 0)
    except ValueError:
        return True
    return False


# ---------------------------------------------------------------------------
# Cohort re-materialization (deterministic; checkpoints are never rebuilt).
# ---------------------------------------------------------------------------


def _rebuild_batch_cohorts(
    *,
    batch: int,
    scratch: Path,
    p4_14_manifest: Mapping[str, Any],
    p4_4_manifest: Mapping[str, Any],
    p4_13_manifest: Mapping[str, Any],
    p1_manifest: Mapping[str, Any],
    p2_report: Mapping[str, Any],
    parent_digest: str,
    bundle: Any,
    projector: Any,
    semantic: Any,
    transition: Any,
    semantic_payload: Mapping[str, Any],
) -> dict[str, Any]:
    recorded = p4_14_manifest["batch_novel_records"][str(batch)]
    train_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    validation_cases: list[dict[str, Any]] = []
    for index, candidate_path in P414_TRAIN_SPECS[batch]:
        record, _case = _p414_build_record(
            root=scratch / "phase-k-data" / f"train-{index:04d}",
            batch=batch,
            index=index,
            candidate_path=candidate_path,
            project_id=f"p4-14-b{batch}-phase-k-train-project-{index}",
            task_seed=7100 + 100 * batch + index,
            split="train",
            source_manifest_digest=str(p1_manifest["manifest_digest"]),
            parent_digest=parent_digest,
            fit_eligible=True,
        )
        train_records.append(record)
    for index, candidate_path in P414_VALIDATION_SPECS[batch]:
        record, case = _p414_build_record(
            root=scratch / "phase-k-data" / f"validation-{index:04d}",
            batch=batch,
            index=index,
            candidate_path=candidate_path,
            project_id=f"p4-14-b{batch}-phase-k-validation-project-{index}",
            task_seed=7200 + 100 * batch + index,
            split="validation",
            source_manifest_digest=str(p1_manifest["manifest_digest"]),
            parent_digest=parent_digest,
            fit_eligible=False,
        )
        validation_records.append(record)
        validation_cases.append(case)
    novel_digest_checks = {
        "train": all(
            rebuilt["record_digest"] == str(stored["record_digest"])
            for rebuilt, stored in zip(train_records, recorded["train_records"], strict=True)
        ),
        "validation": all(
            rebuilt["record_digest"] == str(stored["record_digest"])
            for rebuilt, stored in zip(
                validation_records, recorded["validation_records"], strict=True
            )
        ),
    }
    if not all(novel_digest_checks.values()):
        raise ValueError(f"review batch {batch} phase-K record digests drifted from P4.14")

    (
        train_experiences,
        train_metadata,
        p1_validation,
        p1_validation_metadata,
        p1_verification,
    ) = _rebuild_and_verify_manifest(
        scratch=scratch / "p1",
        manifest=p1_manifest,
        parent_digest=parent_digest,
        bundle=bundle,
        projector=projector,
    )
    if not p1_verification["passed"]:
        raise RuntimeError("P1 manifest reconstruction failed before the review")
    p1_cases, p1_case_mismatches = _build_validation_cases(
        scratch=scratch / "p1-cases",
        validation=p1_validation,
        validation_metadata=p1_validation_metadata,
    )
    if p1_case_mismatches:
        raise RuntimeError(f"P1 validation case mismatch: {p1_case_mismatches[:3]}")
    wake, wake_metadata = _select_balanced_wake(train_experiences, train_metadata)
    expected_wake = p2_report["contract"]["wake_experience_digests"]
    if [str(item.experience_digest) for item in wake] != [str(item) for item in expected_wake]:
        raise RuntimeError("P2 rehearsal experience digest/order drifted")
    if any(
        sum(item["class_key"] == class_key for item in wake_metadata) != PILOT_PER_CLASS
        for class_key in ("A", "B", "C", "D", "R")
    ):
        raise RuntimeError("P2 rehearsal class balance drifted")

    base = 65000 + batch * 1000
    offsets = {
        "constraint": base,
        "sibling": base + 100,
        "train": base + 200,
        "validation": base + 300,
        "holdout": base + 400,
        "retention-newtask": base + 500,
    }
    materialize_common = dict(
        parent_digest=parent_digest,
        worker_bundle_digest=bundle.bundle_digest,
        source_manifest_digest=str(p4_13_manifest["manifest_digest"]),
        projector=projector,
        semantic=semantic,
        transition=transition,
        semantic_payload=semantic_payload,
    )
    train_g = _p414_pressure_split(
        split="train",
        offset=offsets["train"],
        batch=batch,
        scratch=scratch / "data" / "train",
        **materialize_common,
    )
    validation_g = _p414_pressure_split(
        split="validation",
        offset=offsets["validation"],
        batch=batch,
        scratch=scratch / "data" / "validation",
        **materialize_common,
    )
    holdout_g = _p414_pressure_split(
        split="holdout",
        offset=offsets["holdout"],
        batch=batch,
        scratch=scratch / "data" / "holdout",
        **materialize_common,
    )
    retention_newtask = _p414_pressure_split(
        split="retention-newtask",
        offset=offsets["retention-newtask"],
        batch=batch,
        scratch=scratch / "data" / "retention-newtask",
        **materialize_common,
    )
    constraint_records = _p414_structured_records(
        contract=p4_4_manifest["structure_contract"],
        namespace="constraint",
        seed_offset=offsets["constraint"],
        batch=batch,
        scratch=scratch / "data" / "constraint",
        **materialize_common,
    )
    retention_sibling_records = _p414_structured_records(
        contract=p4_4_manifest["structure_contract"],
        namespace="retention",
        seed_offset=offsets["sibling"],
        batch=batch,
        scratch=scratch / "data" / "retention",
        **materialize_common,
    )
    split_records = {
        "validation": validation_g,
        "holdout": holdout_g,
        "retention-sibling": retention_sibling_records,
        "retention-newtask": retention_newtask,
    }
    rebuilt_digests = sorted(
        str(record["candidate_set"].candidate_set_digest)
        for record in (
            *train_g,
            *validation_g,
            *holdout_g,
            *retention_newtask,
            *constraint_records,
            *retention_sibling_records,
        )
    )
    expected_digests = sorted(
        str(digest) for digest in p4_14_manifest["batch_candidate_set_digests"][str(batch)]
    )
    candidate_digests_match = rebuilt_digests == expected_digests
    if not candidate_digests_match:
        raise ValueError(f"review batch {batch} candidate-set digests drifted from P4.14")
    return {
        "split_records": split_records,
        "p1_cases": p1_cases,
        "validation_records": validation_records,
        "validation_cases": validation_cases,
        "novel_digest_checks": novel_digest_checks,
        "candidate_digests_match": candidate_digests_match,
    }


# ---------------------------------------------------------------------------
# Review runner.
# ---------------------------------------------------------------------------


def _run(
    *,
    attachment_path: Path = DEFAULT_ATTACHMENT,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_default_runtime_rollout_review_{uuid4().hex}"
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "training_performed": False,
        "fit_called": False,
        "sealed_payload_read": False,
        "growth_admitted": False,
        "can_promote": False,
        "attachment_manifest": str(attachment_path),
        "report": str(report_path),
    }
    try:
        p4_14_report = _load_json(P4_14_REPORT)
        p4_14_manifest = _load_json(P4_14_MANIFEST)
        scorecard_v5 = _load_json(SCORECARD_V5_REPORT)
        p4_1_manifest = _load_json(P4_1_MANIFEST)
        p4_4_manifest = _load_json(P4_4_MANIFEST)
        p4_13_manifest = _load_json(P4_13_MANIFEST)
        p1_manifest = _load_json(P1_MANIFEST)
        p2_report = _load_json(P2_REPORT)
        p3_2_report = _load_json(P3_2_REPORT)
        p3_5_report = _load_json(P3_5_REPORT)

        # ---- Entry conditions (scorecard v5 frozen boundary).
        if scorecard_v5.get("format") != "taiji-m5-k-axis-scorecard-v5":
            raise ValueError("the review requires the scorecard v5 report")
        gates_v5 = scorecard_v5["promotion_gates"]
        if not gates_v5.get("k_worker_joint_course_completed"):
            raise ValueError("the review requires k_worker_joint_course_completed")
        if gates_v5.get("default_runtime_rollout_review_completed"):
            raise ValueError("the review requires default_runtime_rollout_review_completed=false")
        if scorecard_v5["verdict"]["promotion_gate"] or scorecard_v5["verdict"]["can_promote"]:
            raise ValueError("the review cannot start from an open promotion gate")
        if (
            p4_14_report.get("status") != "completed"
            or p4_14_report.get("outcome") != "joint_course_supported"
            or not p4_14_report.get("experiment_passed")
        ):
            raise ValueError("the review requires the completed P4.14 joint_course_supported")
        if p4_14_report.get("can_promote") or p4_14_report.get("growth_admitted"):
            raise ValueError("the review cannot consume an admitted P4.14 artifact")
        if int(p4_14_report.get("passing_cells", 0)) != len(CELLS):
            raise ValueError("the review requires all P4.14 cells to have passed")
        if p4_14_report.get("manifest_digest") != p4_14_manifest.get("manifest_digest"):
            raise ValueError("P4.14 manifest/report digest mismatch")
        if _digest_without(p4_14_manifest, "manifest_digest") != p4_14_manifest.get(
            "manifest_digest"
        ):
            raise ValueError("P4.14 manifest content digest mismatch")
        if _digest_without(p4_13_manifest, "manifest_digest") != p4_13_manifest.get(
            "manifest_digest"
        ):
            raise ValueError("P4.13 manifest content digest mismatch")

        # ---- Runtime inventory (read-only; recorded before anything runs).
        runtime_inventory = _runtime_inventory()
        if runtime_inventory["runtime_k_g_references"]:
            raise ValueError("the runtime inventory unexpectedly references K/G machinery")

        # ---- Contract parent anchors (same parent as P4.14).
        _artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT,
            model_seed=MODEL_SEED,
        )
        if p2_report["contract"]["parent_checkpoint_digest"] != parent_digest:
            raise ValueError("P2 parent checkpoint digest drifted")
        worker_restore = p3_2_report["base_continuation"]["worker_restore"]
        parent_anchors: dict[str, Any] = {
            "k_parent_checkpoint_digests": {
                "k1": content_digest(_load_mapping(Path(str(worker_restore["k1"]["path"])))),
                "k2": content_digest(_load_mapping(Path(str(worker_restore["k2"]["path"])))),
            },
            "parent_g_checkpoint_digest": content_digest(
                _load_mapping(Path(str(p3_5_report["g_trained_checkpoint"]["path"])))
            ),
        }
        if (
            str(p3_5_report["g_trained_checkpoint"]["digest"])
            != parent_anchors["parent_g_checkpoint_digest"]
        ):
            raise ValueError("P3.5 parent G checkpoint digest drifted")

        # ---- Build and freeze the attachment contract.
        attachment = _build_attachment(
            p4_14_report=p4_14_report,
            p4_14_manifest=p4_14_manifest,
            scorecard_v5=scorecard_v5,
            p4_1_manifest=p4_1_manifest,
            p4_13_manifest=p4_13_manifest,
            runtime_inventory=runtime_inventory,
        )
        _write_json_atomic(attachment_path, attachment)
        stored_attachment = _load_json(attachment_path)
        if _digest_without(stored_attachment, "manifest_digest") != attachment["manifest_digest"]:
            raise ValueError("attachment manifest self-digest mismatch")

        # ---- Preflight: fail-closed consumption of every cell's artifacts.
        consumed: dict[int, dict[str, Any]] = {}
        preflight_cells: list[dict[str, Any]] = []
        for index in range(len(CELLS)):
            cell = _consume_cell(stored_attachment, index)
            consumed[index] = cell
            payloads = cell["payloads"]
            k_restore = _independent_restore(
                Path(str(cell["artifact_verification"]["k1"]["path"])).parent
            )
            g_restore = _independent_extended_restore(
                Path(str(cell["artifact_verification"]["g"]["path"]))
            )
            g_verify = _verify_extended_checkpoint(
                Path(str(cell["artifact_verification"]["g"]["path"]))
            )
            tamper = {
                "k1": _k_tamper_rejected(payloads["k1"]),
                "k2": _k_tamper_rejected(payloads["k2"]),
                "g": _extended_tamper_rejected(payloads["g"]),
            }
            preflight_cells.append(
                {
                    "cell_index": index,
                    "artifact_verification": cell["artifact_verification"],
                    "invariant_checks": cell["invariant_checks"],
                    "k_independent_restore": k_restore,
                    "g_independent_restore": g_restore,
                    "g_checkpoint_verify": g_verify,
                    "tamper_rejection": tamper,
                    "digest_and_invariants_passed": all(cell["invariant_checks"].values()),
                    "restore_passed": (
                        bool(k_restore.get("independent_process_restore"))
                        and bool(g_restore.get("independent_process_restore"))
                        and bool(g_verify.get("passed"))
                    ),
                    "tamper_rejected": all(tamper.values()),
                }
            )
        mixing_rejected = _cell_mixing_rejected(stored_attachment)
        shared_k = stored_attachment["consumption_contract"]["post_k_k_digests_shared"]
        recorded_post_k = p4_14_report["cell_results"][0]["phase_k"]["post_k_digests"]
        anchors_match = str(shared_k["k1"]) == str(recorded_post_k["k1"]) and str(
            shared_k["k2"]
        ) == str(recorded_post_k["k2"])

        g_payload = consumed[0]["payloads"]["g"]
        parent_manifest_expected = str(p4_1_manifest["source_p3_2_manifest_digest"])
        g_lineage_ok = str(g_payload["parent_manifest_digest"]) == (parent_manifest_expected)
        feature_source_digests = {
            str(consumed[index]["payloads"]["g"]["feature_source_state_digest"])
            for index in range(len(CELLS))
        }
        feature_source_shared = len(feature_source_digests) == 1

        mechanical_gate = {
            "preflight_digest_and_invariants": all(
                cell["digest_and_invariants_passed"] for cell in preflight_cells
            ),
            "preflight_tamper_rejected": all(cell["tamper_rejected"] for cell in preflight_cells),
            "cell_mixing_rejected": mixing_rejected,
            "post_k_anchors_match_report": anchors_match,
            "g_parent_lineage_matches_p4_1": g_lineage_ok,
            "feature_source_shared": feature_source_shared,
        }

        # ---- Per-batch cohort re-materialization (data only).
        run_dir.mkdir(parents=True, exist_ok=False)
        batch_cohorts: dict[int, dict[str, Any]] = {}
        post_k_semantic, post_k_transition = _fresh_k_learners(
            consumed[0]["payloads"]["k1"], consumed[0]["payloads"]["k2"]
        )
        for batch in BATCHES:
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
                semantic=post_k_semantic,
                transition=post_k_transition,
                semantic_payload=consumed[0]["payloads"]["k1"],
            )

        # ---- Per-cell rollout through the consumption path.
        cell_results: list[dict[str, Any]] = []
        resource_violations = 0
        for index, (batch, seed) in enumerate(CELLS):
            cell_started = time.perf_counter()
            cohorts = batch_cohorts[batch]
            cell = consumed[index]
            semantic, transition = _fresh_k_learners(cell["payloads"]["k1"], cell["payloads"]["k2"])
            learner = ExtendedGSelectionLearner.from_checkpoint(cell["payloads"]["g"], device="cpu")

            phase_k_scores = _score_phase_k_arm(
                semantic=semantic,
                transition=transition,
                p1_cases=cohorts["p1_cases"],
                novel_cases=cohorts["validation_cases"],
                novel_records=cohorts["validation_records"],
            )
            recorded_cell = p4_14_report["cell_results"][index]
            phase_k_equivalence = {
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
                not phase_k_equivalence["p1_validation_diffs"]
                and not phase_k_equivalence["p2_6_novel_validation_diffs"]
                and phase_k_equivalence["parameter_count_total_match"]
            )

            metrics = {
                split: _metric_summary(_evaluate_records(records, learner))
                for split, records in cohorts["split_records"].items()
            }
            phase_g_equivalence = {
                split: _dict_diff(metrics[split], recorded_cell["phase_g"]["metrics"][split])
                for split in ROLLOUT_SPLITS
            }
            phase_g_match = not any(phase_g_equivalence.values())

            # k_unchanged (consumption edition): the K consumer state is
            # re-serialized after the G-side evaluation and must still match
            # the shared post-K anchors.
            k_unchanged = {
                "k1": content_digest(semantic.checkpoint()) == str(shared_k["k1"]),
                "k2": content_digest(transition.checkpoint()) == str(shared_k["k2"]),
            }

            # Rollback: a fresh load of the same artifacts reproduces the
            # consumer's behavior exactly (P3.0/P4.13 mechanics).
            rollback_semantic, rollback_transition = _fresh_k_learners(
                _load_mapping(Path(str(cell["artifact_verification"]["k1"]["path"]))),
                _load_mapping(Path(str(cell["artifact_verification"]["k2"]["path"]))),
            )
            rollback_learner = ExtendedGSelectionLearner.from_checkpoint(
                _load_mapping(Path(str(cell["artifact_verification"]["g"]["path"]))),
                device="cpu",
            )
            rollback_g_mismatches = 0
            for split in ROLLOUT_SPLITS:
                for record in cohorts["split_records"][split]:
                    candidate_set = record["candidate_set"]
                    live = learner.select(candidate_set)
                    restored = rollback_learner.select(candidate_set)
                    if (
                        live.selected_candidate_id != restored.selected_candidate_id
                        or live.selection_status != restored.selection_status
                    ):
                        rollback_g_mismatches += 1
            rollback_k_scores = _score_phase_k_arm(
                semantic=rollback_semantic,
                transition=rollback_transition,
                p1_cases=cohorts["p1_cases"],
                novel_cases=cohorts["validation_cases"],
                novel_records=cohorts["validation_records"],
            )
            rollback_k_equivalence = {
                "p1_validation_diffs": _dict_diff(
                    _strip_rows(rollback_k_scores["p1_validation"]),
                    _strip_rows(phase_k_scores["p1_validation"]),
                ),
                "p2_6_novel_validation_diffs": _dict_diff(
                    _strip_rows(rollback_k_scores["p2_6_novel_validation"]),
                    _strip_rows(phase_k_scores["p2_6_novel_validation"]),
                ),
            }
            rollback_gate = {
                "g_selection_mismatches": rollback_g_mismatches,
                "k_p1_validation_equal": not rollback_k_equivalence["p1_validation_diffs"],
                "k_novel_validation_equal": not rollback_k_equivalence[
                    "p2_6_novel_validation_diffs"
                ],
                "passed": (
                    rollback_g_mismatches == 0
                    and not rollback_k_equivalence["p1_validation_diffs"]
                    and not rollback_k_equivalence["p2_6_novel_validation_diffs"]
                ),
            }

            cell_wall = time.perf_counter() - cell_started
            cell_resource: dict[str, Any] = {
                "cell_rollout_wall_seconds": round(cell_wall, 3),
                "caps": dict(RESOURCE_CAPS),
                "caps_passed": cell_wall <= RESOURCE_CAPS["cell_rollout_seconds"],
            }
            if not cell_resource["caps_passed"]:
                resource_violations += 1
            cell_results.append(
                {
                    "batch": batch,
                    "seed": seed,
                    "cell_index": index,
                    "phase_k": {
                        "equivalence": phase_k_equivalence,
                        "match": phase_k_match,
                    },
                    "phase_g": {
                        "metrics": metrics,
                        "equivalence": phase_g_equivalence,
                        "match": phase_g_match,
                    },
                    "k_unchanged_after_rollout": k_unchanged,
                    "rollback_gate": rollback_gate,
                    "resource_audit": cell_resource,
                    "passes_all": (
                        phase_k_match
                        and phase_g_match
                        and all(k_unchanged.values())
                        and rollback_gate["passed"]
                        and cell_resource["caps_passed"]
                    ),
                }
            )

        # ---- Non-interference: nothing on disk may have moved.
        post_runtime = _runtime_inventory()
        post_artifacts_ok = True
        for index in range(len(CELLS)):
            cell = _consume_cell(stored_attachment, index)
            post_artifacts_ok = post_artifacts_ok and all(cell["invariant_checks"].values())
        non_interference = {
            "runtime_module_sha256_unchanged": runtime_inventory["runtime_module_sha256"]
            == post_runtime["runtime_module_sha256"],
            "default_checkpoint_sha256_unchanged": runtime_inventory["default_checkpoint_sha256"]
            == post_runtime["default_checkpoint_sha256"],
            "research_artifacts_unchanged": post_artifacts_ok,
            "parent_anchors_unchanged": (
                parent_anchors["k_parent_checkpoint_digests"]["k1"]
                == content_digest(_load_mapping(Path(str(worker_restore["k1"]["path"]))))
                and parent_anchors["k_parent_checkpoint_digests"]["k2"]
                == content_digest(_load_mapping(Path(str(worker_restore["k2"]["path"]))))
                and parent_anchors["parent_g_checkpoint_digest"]
                == content_digest(
                    _load_mapping(Path(str(p3_5_report["g_trained_checkpoint"]["path"])))
                )
            ),
        }
        mechanical_gate["non_interference"] = all(non_interference.values())

        total_wall = time.perf_counter() - started
        resource_audit: dict[str, Any] = {
            "resource_violations": resource_violations,
            "caps": dict(RESOURCE_CAPS),
            "total_wall_seconds": round(total_wall, 3),
            "total_caps_passed": total_wall <= RESOURCE_CAPS["review_total_seconds"],
            "per_cell": [
                {
                    "batch": cell["batch"],
                    "seed": cell["seed"],
                    **cell["resource_audit"],
                }
                for cell in cell_results
            ],
        }
        mechanical_gate["resource_caps_passed"] = (
            resource_violations == 0 and resource_audit["total_caps_passed"]
        )
        restore_gate = {
            "k_independent_restore_all": all(
                cell["k_independent_restore"].get("independent_process_restore")
                for cell in preflight_cells
            ),
            "g_independent_restore_all": all(
                cell["g_independent_restore"].get("independent_process_restore")
                for cell in preflight_cells
            ),
            "g_checkpoint_verify_all": all(
                cell["g_checkpoint_verify"].get("passed") for cell in preflight_cells
            ),
            "behavioral_rollback_all": all(
                cell["rollback_gate"]["passed"] for cell in cell_results
            ),
        }
        consumption_gate = {
            "phase_k_all_match": all(cell["phase_k"]["match"] for cell in cell_results),
            "phase_g_all_match": all(cell["phase_g"]["match"] for cell in cell_results),
            "k_unchanged_after_rollout": all(
                all(cell["k_unchanged_after_rollout"].values()) for cell in cell_results
            ),
        }

        if not all(mechanical_gate.values()):
            outcome = "mechanical_failure"
            status = "failed"
        elif not all(restore_gate.values()):
            outcome = "rollback_unverified"
            status = "completed"
        elif not all(consumption_gate.values()):
            outcome = "rollout_consumption_gap"
            status = "completed"
        else:
            outcome = "default_runtime_rollout_review_supported"
            status = "completed"
        review_supported = outcome == "default_runtime_rollout_review_supported"

        payload.update(
            {
                "status": status,
                "training_performed": False,
                "fit_called": False,
                "run_dir": str(run_dir),
                "attachment_manifest_digest": str(attachment["manifest_digest"]),
                "runtime_inventory": runtime_inventory,
                "parent_anchors": parent_anchors,
                "preflight_cells": preflight_cells,
                "cell_mixing_rejected": mixing_rejected,
                "g_parent_manifest_digest_expected": parent_manifest_expected,
                "feature_source_state_digest": sorted(feature_source_digests)[0],
                "cohort_rematerialization": {
                    str(batch): {
                        "novel_digest_checks": batch_cohorts[batch]["novel_digest_checks"],
                        "candidate_digests_match": batch_cohorts[batch]["candidate_digests_match"],
                    }
                    for batch in BATCHES
                },
                "cell_results": cell_results,
                "restore_gate": restore_gate,
                "consumption_gate": consumption_gate,
                "non_interference": non_interference,
                "mechanical_gate": mechanical_gate,
                "resource_audit": resource_audit,
                "outcome": outcome,
                "default_runtime_rollout_review_supported": review_supported,
                "experiment_passed": bool(review_supported),
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    (
                        f"completed: outcome={outcome}; the P4.14 joint-course "
                        "artifacts were consumed through the frozen attachment "
                        "contract and re-evaluated on the re-materialized cohorts "
                        "without training; growth and promotion remain fail-closed"
                    )
                    if status == "completed"
                    else (
                        "failed: mechanical gates did not pass; the review stops "
                        "honestly and nothing is admitted"
                    )
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
    parser.add_argument("--attachment", type=Path, default=DEFAULT_ATTACHMENT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = _run(attachment_path=args.attachment, report_path=args.report)
    print(
        json.dumps(
            {
                "attachment": str(args.attachment),
                "report": str(args.report),
                "status": result.get("status"),
                "outcome": result.get("outcome"),
                "experiment_passed": result.get("experiment_passed"),
                "default_runtime_rollout_review_supported": result.get(
                    "default_runtime_rollout_review_supported"
                ),
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
