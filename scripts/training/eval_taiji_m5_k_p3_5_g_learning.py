"""Run the reobserve-aware G-only learning Gate.

P3.4 established an independent behavior utility but also exposed zero-margin
static ties.  P3.5 trains only on non-zero-margin candidate sets, preserves the
P2.7 holdout, and checks that a ``reobserve`` choice projects to a typed,
non-executable ``workspace.list`` abstention.
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
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.seed_runtime import SeedRuntime  # noqa: E402
from scripts.training.eval_taiji_m5_k_p2_validation_pilot import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    MODEL_SEED,
    P1_MANIFEST,
    WORKER_ROOT,
    _checkpoint_preflight,
    _context,
    _rebuild_and_verify_manifest,
)
from scripts.training.eval_taiji_m5_k_p3_3_g_learning import (  # noqa: E402
    CONFIDENCE_FLOOR,
    P2_7_MANIFEST,
    P3_2_MANIFEST,
    P3_2_REPORT,
    _fresh_learners,
    _holdout_actions,
    _holdout_candidate_sets,
)
from scripts.training.eval_taiji_m5_k_p3_3_g_signal_canary import (  # noqa: E402
    _candidate_identity,
)
from scripts.training.eval_taiji_m5_k_p3_4_behavior_signal_canary import (  # noqa: E402
    P2_6_MANIFEST,
    _case_for_metadata,
    _choose_train,
)
from seed import Seed  # noqa: E402
from seed.config import SeedConfig  # noqa: E402
from seed_platform.workbench import WorkbenchEnvironment  # noqa: E402
from taiji import (  # noqa: E402
    GSelectionBehaviorSet,
    GSelectionCandidate,
    GSelectionCandidateSet,
    GSelectionDecision,
    GSelectionLearner,
    NativeReadOnlyIntentPlanner,
    ReadOnlyAbstention,
    ReadOnlyIntentPolicy,
    TaijiConfig,
    WorkbenchObservationSchema,
    content_digest,
    project_g_decision,
)

REPORT_FORMAT = "taiji-m5-k-p3-5-g-learning-v1"
MANIFEST_FORMAT = "taiji-m5-k-p3-5-g-learning-manifest-v1"
VERSION = 1
G_EPOCHS = 8
G_LEARNING_RATE = 0.15
UTILITY_MARGIN_EPSILON = 1e-9
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p3_5_g_learning_manifest_v1.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_5_g_learning_20260911.json"
P3_4_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p3_4_behavior_manifest_v1.json"
)
P3_4_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_4_behavior_signal_20260911.json"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON mapping at {path}")
    return dict(payload)


def _load_mapping(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected mapping checkpoint at {path}")
    return {str(key): value for key, value in payload.items()}


def _atomic_roundtrip(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


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


def _load_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if content_digest(unsigned) != str(manifest.get("manifest_digest", "")):
        raise ValueError("P3.4 behavior manifest digest mismatch")
    records: list[dict[str, Any]] = []
    for split_name in ("train", "validation"):
        for raw in manifest.get(f"{split_name}_records", ()):
            if not isinstance(raw, Mapping):
                raise TypeError("P3.4 behavior record must be a mapping")
            candidate_set = GSelectionCandidateSet.from_payload(raw["candidate_set"])
            behavior_set = GSelectionBehaviorSet.from_payload(raw["behavior_set"])
            if candidate_set.split != split_name:
                raise ValueError("P3.4 behavior record split drifted")
            if behavior_set.candidate_set_digest != candidate_set.candidate_set_digest:
                raise ValueError("P3.4 behavior/candidate lineage mismatch")
            if behavior_set.behavior_target_candidate_id != candidate_set.target_candidate_id:
                raise ValueError("P3.4 candidate and behavior target drifted")
            margin = float(behavior_set.utility_margin)
            records.append(
                {
                    "candidate_set": candidate_set,
                    "behavior_set": behavior_set,
                    "diagnostic": dict(raw.get("diagnostic", {})),
                    "fit_eligible": margin > UTILITY_MARGIN_EPSILON,
                    "utility_margin": margin,
                }
            )
    if not records:
        raise ValueError("P3.4 behavior manifest has no records")
    return records


def _verify_sources(
    *,
    p3_4_report: Mapping[str, Any],
    p3_4_manifest: Mapping[str, Any],
    p3_2_report: Mapping[str, Any],
    p3_2_manifest: Mapping[str, Any],
    p2_7_manifest: Mapping[str, Any],
) -> None:
    if p3_4_report.get("status") != "completed":
        raise ValueError("P3.5 requires a completed P3.4 behavior canary")
    if p3_4_report.get("fit_called") or p3_4_report.get("training_performed"):
        raise ValueError("P3.4 source already performed training")
    if p3_4_report.get("external_target_used") or p3_4_report.get("sealed_payload_read"):
        raise ValueError("P3.4 source crossed a runtime boundary")
    if not p3_4_report.get("signal_gate", {}).get("passed"):
        raise ValueError("P3.4 behavior signal Gate did not pass")
    if p3_4_report.get("manifest_digest") != p3_4_manifest.get("manifest_digest"):
        raise ValueError("P3.4 report and manifest digests differ")
    if p3_4_report.get("can_promote"):
        raise ValueError("P3.5 cannot replace a promoted artifact")
    signal_gate = p3_4_report["signal_gate"]
    if signal_gate.get("train_count") != 40 or signal_gate.get("validation_count") != 10:
        raise ValueError("P3.4 split counts drifted")
    if signal_gate.get("nonzero_utility_margin_count") != 40:
        raise ValueError("P3.4 non-zero margin count drifted")
    if p3_2_report.get("status") != "completed":
        raise ValueError("P3.5 requires a completed P3.2 owner-transfer source")
    if p3_2_report.get("manifest_digest") != p3_2_manifest.get("manifest_digest"):
        raise ValueError("P3.2 report and manifest digests differ")
    if p3_2_report.get("external_target_used"):
        raise ValueError("P3.2 external target boundary is not clean")
    if len(p2_7_manifest.get("records", ())) != 4:
        raise ValueError("P3.5 requires the fixed four-row P2.7 holdout")
    if any(record.get("candidate", {}).get("fit_eligible") is not False for record in p2_7_manifest["records"]):
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
    safe = [candidate for candidate in candidate_set.candidates if candidate.candidate_role == "abstain"]
    if safe:
        return sorted(safe, key=lambda candidate: candidate.candidate_id)[0]
    reobserve = [candidate for candidate in candidate_set.candidates if candidate.candidate_role == "reobserve"]
    if reobserve:
        return sorted(reobserve, key=lambda candidate: candidate.candidate_id)[0]
    raise ValueError("P3.5 K-only baseline has no safe candidate")


def _select_row(
    record: Mapping[str, Any],
    *,
    arm: str,
    zero_step: GSelectionLearner,
    trained: GSelectionLearner,
) -> dict[str, Any]:
    candidate_set = record["candidate_set"]
    behavior_set = record["behavior_set"]
    if arm == "k_only":
        selected = _k_only_candidate(candidate_set)
        decision = None
    else:
        learner = zero_step if arm == "g_zero_step" else trained
        decision = learner.select(candidate_set)
        selected = next(
            candidate
            for candidate in candidate_set.candidates
            if candidate.candidate_id == decision.selected_candidate_id
        )
    outcome = next(item for item in behavior_set.outcomes if item.candidate_id == selected.candidate_id)
    return {
        "split": candidate_set.split,
        "example_id": candidate_set.example_id,
        "candidate_set_digest": candidate_set.candidate_set_digest,
        "fit_eligible": bool(record["fit_eligible"]),
        "utility_margin": float(record["utility_margin"]),
        "selected_candidate_id": selected.candidate_id,
        "selected_candidate_role": selected.candidate_role,
        "selected_utility": float(outcome.utility),
        "behavior_target_candidate_id": behavior_set.behavior_target_candidate_id,
        "target_kind": candidate_set.target_kind,
        "behavior_target_hit": selected.candidate_id == behavior_set.behavior_target_candidate_id,
        "safe_projection_required": selected.candidate_role in {"abstain", "reobserve"},
        "decision_digest": None if decision is None else decision.decision_digest,
        "decision": decision,
        "candidate_set": candidate_set,
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    utility_sum = sum(float(row["selected_utility"]) for row in rows)
    return {
        "count": len(rows),
        "fit_eligible_count": sum(int(bool(row["fit_eligible"])) for row in rows),
        "behavior_target_hit_count": sum(int(bool(row["behavior_target_hit"])) for row in rows),
        "behavior_target_hit_rate": (
            sum(int(bool(row["behavior_target_hit"])) for row in rows) / len(rows) if rows else 0.0
        ),
        "selected_utility_sum": utility_sum,
        "selected_utility_mean": utility_sum / len(rows) if rows else 0.0,
        "selected_roles": dict(sorted(Counter(str(row["selected_candidate_role"]) for row in rows).items())),
        "rows": [
            {
                key: value
                for key, value in row.items()
                if key not in {"decision", "candidate_set"}
            }
            for row in rows
        ],
    }


def _selection_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    zero_step: GSelectionLearner,
    trained: GSelectionLearner,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ("k_only", "g_zero_step", "g_trained"):
        rows = [_select_row(record, arm=arm, zero_step=zero_step, trained=trained) for record in records]
        result[arm] = _summarize(rows)
    return result


def _case_key(candidate_set: GSelectionCandidateSet) -> str:
    return candidate_set.example_id


def _reconstruct_cases(*, scratch: Path, p1_manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    _artifacts, parent_digest, bundle, projector = _context(
        worker_root=WORKER_ROOT,
        model_seed=MODEL_SEED,
    )
    p2_6_manifest = _load_json(P2_6_MANIFEST)
    if p2_6_manifest.get("parent_checkpoint_digest") != parent_digest:
        raise ValueError("P2.6 parent digest drifted during P3.5 reconstruction")
    train_experiences, train_metadata, validation_experiences, validation_metadata, reconstruction = (
        _rebuild_and_verify_manifest(
            scratch=scratch,
            manifest=p1_manifest,
            parent_digest=parent_digest,
            bundle=bundle,
            projector=projector,
        )
    )
    if not reconstruction.get("passed"):
        raise RuntimeError("P1 reconstruction failed before P3.5")
    _selected_train, selected_train_metadata = _choose_train(train_experiences, train_metadata)
    cases: dict[str, dict[str, Any]] = {}
    experience_metadata = [
        *zip(_selected_train, selected_train_metadata, strict=True),
        *zip(validation_experiences, validation_metadata, strict=True),
    ]
    for experience, metadata in experience_metadata:
        case = _case_for_metadata(scratch=scratch, metadata=metadata)
        key = _candidate_identity(experience=experience, metadata=metadata)[0]
        if key in cases:
            raise ValueError("P3.5 reconstruction case identity is not unique")
        cases[key] = case
    return cases


def _runtime_snapshot(case: Mapping[str, Any], *, label: str) -> Any:
    import seed_platform.workbench as workbench_module

    original_get_setting = workbench_module.get_setting
    workbench_module.get_setting = lambda key, default=None: (
        str(case["root"]) if key == "workspace_path" else default
    )
    try:
        runtime = SeedRuntime(
            Seed(
                SeedConfig(taiji=TaijiConfig(seed=MODEL_SEED)),
                episode_id=f"taiji-m5-k-p3-5-{label}",
            )
        )
        runtime._workbench_environment = WorkbenchEnvironment(
            root=case["root"],
            programming_language_registry=case["registry"],
        )
        return runtime.workbench_environment.capability_snapshot
    finally:
        workbench_module.get_setting = original_get_setting


def _projection_observation_schema() -> WorkbenchObservationSchema:
    return WorkbenchObservationSchema(
        language_ids=("cpp", "python", "rust", "typescript", "unknown"),
        selection_states=("ambiguous", "resolved", "unknown"),
        task_kinds=("inspect-language",),
        extensions=("<none>", ".cpp", ".h", ".py", ".rs", ".ts"),
    )


def _projection_preflight(
    records: Sequence[Mapping[str, Any]],
    *,
    zero_step: GSelectionLearner,
    trained: GSelectionLearner,
    cases: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    planner = NativeReadOnlyIntentPlanner(ReadOnlyIntentPolicy(routes=()))
    target_rows = [
        record
        for record in records
        if record["behavior_set"].behavior_target_candidate_id == "reobserve"
    ]
    selected_rows: list[dict[str, Any]] = []
    target_rows_result: list[dict[str, Any]] = []
    for index, record in enumerate(target_rows):
        candidate_set = record["candidate_set"]
        selected = next(
            candidate
            for candidate in candidate_set.candidates
            if candidate.candidate_id == candidate_set.target_candidate_id
        )
        decision = GSelectionDecision.create(
            candidate_set=candidate_set,
            selected=selected,
            selection_status="reobserve",
            candidate_scores=tuple(
                (candidate.candidate_id, 1.0 if candidate is selected else 0.0)
                for candidate in candidate_set.candidates
            ),
        )
        case = cases[_case_key(candidate_set)]
        snapshot = _runtime_snapshot(case, label=f"target-{index}")
        projected = project_g_decision(
            decision,
            candidate_set,
            planner=planner,
            observation=case["observation"],
            capability_snapshot=snapshot,
        )
        restored = ReadOnlyAbstention.from_payload(projected.to_payload())
        target_rows_result.append(
            {
                "split": candidate_set.split,
                "example_id": candidate_set.example_id,
                "candidate_set_digest": candidate_set.candidate_set_digest,
                "projected_type": type(projected).__name__,
                "next_step": projected.next_step,
                "action_intent_is_none": projected.to_payload().get("action_intent") is None,
                "roundtrip": restored == projected,
                "snapshot_match": projected.snapshot_id == case["observation"].capability_snapshot_id,
            }
        )
    for arm, learner in (("g_zero_step", zero_step), ("g_trained", trained)):
        for record in records:
            decision = learner.select(record["candidate_set"])
            if decision.selection_status != "reobserve":
                continue
            case = cases[_case_key(record["candidate_set"])]
            snapshot = _runtime_snapshot(case, label=f"selected-{arm}-{len(selected_rows)}")
            projected = project_g_decision(
                decision,
                record["candidate_set"],
                planner=planner,
                observation=case["observation"],
                capability_snapshot=snapshot,
            )
            selected_rows.append(
                {
                    "arm": arm,
                    "example_id": record["candidate_set"].example_id,
                    "projected_type": type(projected).__name__,
                    "next_step": projected.next_step,
                    "action_intent_is_none": projected.to_payload().get("action_intent") is None,
                }
            )
    target_passed = all(
        row["projected_type"] == "ReadOnlyAbstention"
        and row["next_step"] == "workspace.list"
        and row["action_intent_is_none"]
        and row["roundtrip"]
        and row["snapshot_match"]
        for row in target_rows_result
    )
    selected_passed = all(
        row["projected_type"] == "ReadOnlyAbstention"
        and row["next_step"] == "workspace.list"
        and row["action_intent_is_none"]
        for row in selected_rows
    )
    return {
        "target_reobserve_count": len(target_rows_result),
        "target_reobserve_projection_passed": target_passed,
        "selected_reobserve_count": len(selected_rows),
        "selected_reobserve_projection_passed": selected_passed,
        "target_rows": target_rows_result,
        "selected_rows": selected_rows,
    }


def _build_fit_manifest(
    *,
    records: Sequence[Mapping[str, Any]],
    p3_4_manifest: Mapping[str, Any],
    p3_4_report: Mapping[str, Any],
    worker_digests: Mapping[str, str],
) -> dict[str, Any]:
    train = [record for record in records if record["candidate_set"].split == "train"]
    validation = [record for record in records if record["candidate_set"].split == "validation"]
    fit_train = [record for record in train if record["fit_eligible"]]
    fit_validation = [record for record in validation if record["fit_eligible"]]
    payload: dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "version": VERSION,
        "source_p3_4_manifest_digest": str(p3_4_manifest["manifest_digest"]),
        "source_p3_4_report_digest": content_digest(p3_4_report),
        "k_checkpoint_digests": dict(worker_digests),
        "fit_policy": {
            "fit_called": True,
            "utility_margin_epsilon": UTILITY_MARGIN_EPSILON,
            "zero_margin_excluded": True,
            "runtime_behavior_label_used": False,
            "p2_7_holdout_fit_count": 0,
        },
        "train_candidate_set_digests": [
            record["candidate_set"].candidate_set_digest for record in fit_train
        ],
        "validation_candidate_set_digests": [
            record["candidate_set"].candidate_set_digest for record in fit_validation
        ],
        "excluded_zero_margin_train_digests": [
            record["candidate_set"].candidate_set_digest
            for record in train
            if not record["fit_eligible"]
        ],
        "excluded_zero_margin_validation_digests": [
            record["candidate_set"].candidate_set_digest
            for record in validation
            if not record["fit_eligible"]
        ],
    }
    payload["manifest_digest"] = content_digest(payload)
    return payload


def run_learning(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p3_5_g_learning_{uuid4().hex}"
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "training_performed": False,
        "fit_called": False,
        "external_target_used": False,
        "sealed_payload_read": False,
        "can_promote": False,
        "manifest": str(manifest_path),
        "report": str(report_path),
    }
    try:
        p3_4_report = _load_json(P3_4_REPORT)
        p3_4_manifest = _load_json(P3_4_MANIFEST)
        p3_2_report = _load_json(P3_2_REPORT)
        p3_2_manifest = _load_json(P3_2_MANIFEST)
        p2_7_manifest = _load_json(P2_7_MANIFEST)
        p1_manifest = _load_json(P1_MANIFEST)
        _verify_sources(
            p3_4_report=p3_4_report,
            p3_4_manifest=p3_4_manifest,
            p3_2_report=p3_2_report,
            p3_2_manifest=p3_2_manifest,
            p2_7_manifest=p2_7_manifest,
        )
        records = _load_records(p3_4_manifest)
        train_records = [record for record in records if record["candidate_set"].split == "train"]
        validation_records = [record for record in records if record["candidate_set"].split == "validation"]
        fit_train = [record for record in train_records if record["fit_eligible"]]
        fit_validation = [record for record in validation_records if record["fit_eligible"]]
        excluded_train = [record for record in train_records if not record["fit_eligible"]]
        excluded_validation = [record for record in validation_records if not record["fit_eligible"]]
        if (len(train_records), len(validation_records), len(fit_train), len(fit_validation)) != (40, 10, 32, 8):
            raise ValueError("P3.5 fit-eligible split drifted from P3.4")
        if any(record["candidate_set"].target_kind not in {"pair", "reobserve"} for record in fit_train):
            raise ValueError("P3.5 fit set contains an unsupported target kind")
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
            raise ValueError("P3.2 K1 checkpoint digest drifted before P3.5")
        if content_digest(transition_payload) != worker_digests["k2"]:
            raise ValueError("P3.2 K2 checkpoint digest drifted before P3.5")
        semantic, transition = _fresh_learners(semantic_payload, transition_payload)
        run_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_before = _checkpoint_preflight(
            output_dir=run_dir / "k-preflight-before",
            semantic_parent=copy.deepcopy(semantic_payload),
            transition_parent=copy.deepcopy(transition_payload),
        )
        if not checkpoint_before.get("passed"):
            raise RuntimeError("P3.5 K checkpoint preflight before fit failed")
        _write_json_atomic(manifest_path, _build_fit_manifest(
            records=records,
            p3_4_manifest=p3_4_manifest,
            p3_4_report=p3_4_report,
            worker_digests=worker_digests,
        ))
        fit_manifest = _load_json(manifest_path)
        cases = _reconstruct_cases(scratch=run_dir / "reconstruction", p1_manifest=p1_manifest)
        zero_step = GSelectionLearner(
            parent_manifest_digest=str(p3_2_manifest["manifest_digest"]),
            k_checkpoint_digests=worker_digests,
            learning_rate=G_LEARNING_RATE,
            confidence_floor=CONFIDENCE_FLOOR,
        )
        zero_checkpoint = _save_g_checkpoint(run_dir, zero_step, "g_zero_step.pt")
        if not zero_checkpoint["passed"]:
            raise RuntimeError("P3.5 G zero-step checkpoint restore failed")
        trained = GSelectionLearner.from_checkpoint(zero_step.checkpoint(), device="cpu")
        fit_result = trained.fit(
            (record["candidate_set"] for record in fit_train),
            epochs=G_EPOCHS,
            learning_rate=G_LEARNING_RATE,
        )
        trained_checkpoint = _save_g_checkpoint(run_dir, trained, "g_trained.pt")
        if not trained_checkpoint["passed"]:
            raise RuntimeError("P3.5 G trained checkpoint restore failed")
        checkpoint_after = _checkpoint_preflight(
            output_dir=run_dir / "k-preflight-after",
            semantic_parent=copy.deepcopy(semantic_payload),
            transition_parent=copy.deepcopy(transition_payload),
        )
        if not checkpoint_after.get("passed"):
            raise RuntimeError("P3.5 K checkpoint preflight after fit failed")
        k_after = {
            "k1": content_digest(semantic.checkpoint()),
            "k2": content_digest(transition.checkpoint()),
        }
        if k_after != worker_digests:
            raise RuntimeError("P3.5 G fit changed frozen K checkpoint state")
        selection_metrics = {
            "train": _selection_metrics(train_records, zero_step=zero_step, trained=trained),
            "validation": _selection_metrics(validation_records, zero_step=zero_step, trained=trained),
            "fit_eligible": _selection_metrics(
                [*fit_train, *fit_validation], zero_step=zero_step, trained=trained
            ),
            "excluded_zero_margin": _selection_metrics(
                [*excluded_train, *excluded_validation], zero_step=zero_step, trained=trained
            ),
        }
        contested = [
            record
            for record in [*fit_train, *fit_validation]
            if _k_only_candidate(record["candidate_set"]).candidate_id
            != record["behavior_set"].behavior_target_candidate_id
        ]
        contested_metrics = {
            arm: _summarize(
                [_select_row(record, arm=arm, zero_step=zero_step, trained=trained) for record in contested]
            )
            for arm in ("k_only", "g_zero_step", "g_trained")
        }
        projection = _projection_preflight(
            records=records,
            zero_step=zero_step,
            trained=trained,
            cases=cases,
        )
        holdout_scratch = run_dir / "p2-7-holdout"
        holdout_cases, holdout_sets = _holdout_candidate_sets(
            scratch=holdout_scratch,
            p2_7_manifest=p2_7_manifest,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
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
            "k_before_independent_restore": bool(checkpoint_before.get("passed")),
            "g_zero_independent_restore": bool(zero_checkpoint.get("passed")),
            "g_trained_independent_restore": bool(trained_checkpoint.get("passed")),
            "k_after_independent_restore": bool(checkpoint_after.get("passed")),
            "k_digests_unchanged": k_after == worker_digests,
        }
        training_gate = {
            "fit_called": True,
            "fit_train_records": len(fit_train),
            "fit_validation_records": len(fit_validation),
            "zero_margin_train_excluded": len(excluded_train) == 8,
            "zero_margin_validation_excluded": len(excluded_validation) == 2,
            "zero_margin_digest_excluded": not (
                {record["candidate_set"].candidate_set_digest for record in excluded_train}
                & set(fit_manifest["train_candidate_set_digests"])
            ),
            "p2_7_records_not_fit": True,
            "g_training_steps_positive": trained.training_steps > 0,
            "g_parameter_count_stable": trained.parameter_count == zero_step.parameter_count,
            "external_target_unused": True,
        }
        projection_gate = {
            "target_reobserve_count": projection["target_reobserve_count"] == 30,
            "target_reobserve_projection_passed": projection["target_reobserve_projection_passed"],
            "selected_reobserve_projection_passed": projection["selected_reobserve_projection_passed"],
        }
        rejection_gate = {
            "tampered_checkpoint_rejected": tampered_rejected,
            "wrong_lineage_rejected": wrong_lineage_rejected,
        }
        holdout_gate = {
            "fixed_four_rows": len(holdout_sets) == 4,
            "fit_count": 0,
            "all_holdout_paths_unique": len({item.path for item in holdout_sets}) == len(holdout_sets),
            "holdout_projects_at_least_two": len({item.project_id for item in holdout_sets}) >= 2,
            "g_trained_workbench_success": holdout_actions["g_trained"]["workbench_success_count"] >= 4,
        }
        zero_contested = contested_metrics["g_zero_step"]
        trained_contested = contested_metrics["g_trained"]
        behavior_gain_gate = {
            "contested_nonempty": bool(contested),
            "trained_utility_strictly_above_zero_step": (
                trained_contested["selected_utility_sum"]
                > zero_contested["selected_utility_sum"] + UTILITY_MARGIN_EPSILON
            ),
            "trained_target_hits_not_below_zero_step": (
                trained_contested["behavior_target_hit_count"]
                >= zero_contested["behavior_target_hit_count"]
            ),
            "trained_behavior_changed": any(
                row["g_zero_step"]["selected_candidate_id"] != row["g_trained"]["selected_candidate_id"]
                for row in (
                    {
                        "g_zero_step": _select_row(record, arm="g_zero_step", zero_step=zero_step, trained=trained),
                        "g_trained": _select_row(record, arm="g_trained", zero_step=zero_step, trained=trained),
                    }
                    for record in contested
                )
            ),
        }
        all_gates = [checkpoint_gate, training_gate, projection_gate, rejection_gate, holdout_gate, behavior_gain_gate]
        gate_passed = all(
            all(bool(value) for key, value in gate.items() if key != "fit_count")
            for gate in all_gates
        )
        payload.update(
            {
                "status": "completed",
                "training_performed": True,
                "fit_called": True,
                "run_dir": str(run_dir),
                "manifest_digest": fit_manifest["manifest_digest"],
                "source_p3_4_manifest_digest": p3_4_manifest["manifest_digest"],
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
                "projection_gate": projection_gate,
                "rejection_gate": rejection_gate,
                "holdout_gate": holdout_gate,
                "behavior_gain_gate": behavior_gain_gate,
                "selection_metrics": selection_metrics,
                "contested_metrics": contested_metrics,
                "reobserve_projection": projection,
                "holdout_actions": holdout_actions,
                "gate_passed": gate_passed,
                "interpretation": (
                    "passed: non-zero-margin reobserve-aware G learning changed contested behavior"
                    if gate_passed
                    else "completed: G checkpoint/action boundaries passed but behavior gain Gate failed; do not promote"
                ),
                "can_promote": False,
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
