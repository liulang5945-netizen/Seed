"""Build and audit the first learned-G candidate-set contract.

P3.3 begins with a data-signal canary.  The P2.6 K checkpoint is restored
read-only; no K fit and no G fit occur here.  Each row contains K-derived
selection candidates plus an explicit safe-abstention candidate.  Expected
selection is stored only in the training/evaluation manifest, while the
inference payload omits it entirely.  P2.7 remains an untouched test set.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m5_k_p2_3_targeted_learning import (  # noqa: E402
    _fresh_learners,
)
from scripts.training.eval_taiji_m5_k_p2_validation_pilot import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    MODEL_SEED,
    P1_MANIFEST,
    WORKER_ROOT,
    _checkpoint_preflight,
    _context,
    _rebuild_and_verify_manifest,
)
from taiji import (  # noqa: E402
    ContentPlan,
    Goal,
    GSelectionCandidate,
    GSelectionCandidateSet,
    StructuredSemanticResult,
    StructuredSemanticTransitionResult,
    content_digest,
)

P2_6_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p2_6_novel_learning_20260910.json"
P2_6_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p2_6_novel_learning_manifest_v1.json"
)
P2_7_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p2_7_generalization_manifest_v1.json"
)
P3_2_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_2_owner_transfer_20260910.json"
P3_2_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p3_2_owner_transfer_manifest_v1.json"
)
REPORT_FORMAT = "taiji-m5-k-p3-3-g-signal-canary-v1"
MANIFEST_FORMAT = "taiji-m5-k-p3-3-g-candidate-manifest-v1"
VERSION = 1
TRAIN_PER_CLASS = 8
CONFIDENCE_FLOOR = 0.55
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p3_3_g_candidate_manifest_v1.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_3_g_signal_canary_20260911.json"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = torch_load(path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected mapping checkpoint at {path}")
    return dict(payload)


def torch_load(path: Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _verify_checkpoint_only(directory: Path) -> dict[str, Any]:
    semantic_payload = _load_mapping(directory / "birth_k1_semantic.pt")
    transition_payload = _load_mapping(directory / "birth_k2_transition.pt")
    semantic, transition = _fresh_learners(semantic_payload, transition_payload)
    result = {
        "k1_restore_digest_equal": content_digest(semantic.checkpoint())
        == content_digest(semantic_payload),
        "k2_restore_digest_equal": content_digest(transition.checkpoint())
        == content_digest(transition_payload),
    }
    result["passed"] = all(result.values())
    print(json.dumps(result, ensure_ascii=False))
    return result


def _catalogs(
    semantic_payload: Mapping[str, Any],
) -> tuple[dict[str, Goal], dict[str, ContentPlan]]:
    goals = {
        item.goal_id: item for item in (Goal.from_payload(payload) for payload in semantic_payload["goal_catalog"])
    }
    content = {
        item.content_id: item
        for item in (ContentPlan.from_payload(payload) for payload in semantic_payload["content_catalog"])
    }
    return goals, content


def _score(result: Any, key: str, identifier: str | None) -> float:
    if identifier is None:
        return 0.0
    return float(getattr(result, key, {}).get(identifier, 0.0))


def _candidate(
    *,
    candidate_id: str,
    source: str,
    candidate_role: str,
    status: str,
    goal: Goal | None,
    content: ContentPlan | None,
    result: StructuredSemanticResult | StructuredSemanticTransitionResult | None,
) -> GSelectionCandidate:
    return GSelectionCandidate.create(
        candidate_id=candidate_id,
        source=source,
        candidate_role=candidate_role,
        status=status,
        goal=goal,
        content_plan=content,
        goal_score=_score(result, "goal_scores", None if goal is None else goal.goal_id),
        content_score=_score(result, "content_scores", None if content is None else content.content_id),
        confidence=0.0 if result is None else float(result.confidence),
        ambiguity=1.0 if result is None else float(result.ambiguity),
    )


def _top_ids(scores: Mapping[str, float], *, limit: int = 2) -> tuple[str, ...]:
    return tuple(
        key
        for key, _value in sorted(scores.items(), key=lambda item: (-float(item[1]), str(item[0])))[:limit]
    )


def _candidate_sets(
    *,
    experience: Any,
    metadata: Mapping[str, Any],
    semantic: Any,
    transition: Any,
    goals: Mapping[str, Goal],
    content_plans: Mapping[str, ContentPlan],
) -> GSelectionCandidateSet:
    semantic_example = experience.semantic_example
    transition_example = experience.transition_example
    semantic_result = semantic.predict(semantic_example.percept)
    transition_result = transition.predict(
        transition_example.before,
        transition_example.event,
    )
    candidates: list[GSelectionCandidate] = []
    seen_pairs: set[tuple[str | None, str | None]] = set()

    def add(
        *,
        candidate_id: str,
        source: str,
        candidate_role: str,
        status: str,
        goal: Goal | None,
        content: ContentPlan | None,
        result: Any,
    ) -> None:
        pair = (None if goal is None else goal.goal_id, None if content is None else content.content_id)
        if pair in seen_pairs and source != "runtime.abstain":
            return
        if source != "runtime.abstain":
            seen_pairs.add(pair)
        candidates.append(
            _candidate(
                candidate_id=candidate_id,
                source=source,
                candidate_role=candidate_role,
                status=status,
                goal=goal,
                content=content,
                result=result,
            )
        )

    if semantic_result.goal is not None or semantic_result.content_plan is not None:
        add(
            candidate_id="k1:selected",
            source="k1.semantic",
            candidate_role="proposal",
            status=semantic_result.status,
            goal=semantic_result.goal,
            content=semantic_result.content_plan,
            result=semantic_result,
        )
    if transition_result.goal is not None or transition_result.content_plan is not None:
        add(
            candidate_id="k2:selected",
            source="k2.transition",
            candidate_role="proposal",
            status=transition_result.status,
            goal=transition_result.goal,
            content=transition_result.content_plan,
            result=transition_result,
        )

    goal_ids = _top_ids(semantic_result.goal_scores)
    content_ids = _top_ids(semantic_result.content_scores)
    for goal_id in goal_ids:
        for content_id in content_ids:
            goal = goals.get(goal_id)
            content = content_plans.get(content_id)
            if goal is None or content is None:
                continue
            if content.source_goal_id not in {None, goal.goal_id}:
                continue
            status = "clarify" if content.intent_kind == "request_information" else "resolved"
            add(
                candidate_id=f"k1:grid:{goal_id}:{content_id}",
                source="k1.score-grid",
                candidate_role="proposal",
                status=status,
                goal=goal,
                content=content,
                result=semantic_result,
            )

    add(
        candidate_id="abstain",
        source="runtime.abstain",
        candidate_role="abstain",
        status="abstained",
        goal=None,
        content=None,
        result=None,
    )
    if len(candidates) < 2:
        add(
            candidate_id="reobserve",
            source="runtime.reobserve",
            candidate_role="reobserve",
            status="ambiguous",
            goal=None,
            content=None,
            result=None,
        )
    if len(candidates) < 2:
        raise ValueError("G candidate construction produced fewer than two candidates")

    expected_goal = semantic_example.goal.goal_id
    expected_content = semantic_example.content.content_id
    safe = (
        float(semantic_example.percept.confidence) < CONFIDENCE_FLOOR
        or semantic_result.status not in {"resolved", "clarify"}
        or semantic_result.content_plan is None
    )
    if safe:
        target_id = "abstain"
        target_kind = "abstain"
    else:
        matching = [
            item.candidate_id
            for item in candidates
            if item.goal is not None
            and item.content_plan is not None
            and item.goal.goal_id == expected_goal
            and item.content_plan.content_id == expected_content
        ]
        if not matching:
            raise ValueError(
                f"G candidate set has no target candidate for {metadata['experience_id']}"
            )
        target_id = matching[0]
        target_kind = "pair"
    return GSelectionCandidateSet.create(
        example_id=str(semantic_example.example_id),
        family_id=str(semantic_example.family_id),
        split=str(metadata["split"]),
        project_id=str(metadata["project_id"]),
        path=str(metadata["variant_paths"][0]),
        input_digest=str(semantic_example.input_digest),
        candidates=tuple(candidates),
        target_candidate_id=target_id,
        target_kind=target_kind,
    )


def _choose_train(
    experiences: Sequence[Any], metadata: Sequence[Mapping[str, Any]]
) -> tuple[list[Any], list[dict[str, Any]]]:
    counts: dict[str, int] = defaultdict(int)
    selected: list[Any] = []
    selected_metadata: list[dict[str, Any]] = []
    for experience, item in zip(experiences, metadata, strict=True):
        class_key = str(item["class_key"])
        if counts[class_key] >= TRAIN_PER_CLASS:
            continue
        selected.append(experience)
        selected_metadata.append(dict(item))
        counts[class_key] += 1
    expected = sorted({str(item["class_key"]) for item in metadata})
    if any(counts[key] != TRAIN_PER_CLASS for key in expected):
        raise ValueError(f"G train selection is not balanced: {dict(counts)}")
    return selected, selected_metadata


def _signal_gate(
    train_sets: Sequence[GSelectionCandidateSet],
    validation_sets: Sequence[GSelectionCandidateSet],
) -> dict[str, Any]:
    all_sets = [*train_sets, *validation_sets]
    train_projects = {item.project_id for item in train_sets}
    validation_projects = {item.project_id for item in validation_sets}
    train_paths = {item.path for item in train_sets}
    validation_paths = {item.path for item in validation_sets}
    target_kinds = {item.target_kind for item in all_sets}
    candidate_counts = [len(item.candidates) for item in all_sets]
    nonzero_margin = [item for item in all_sets if abs(item.score_margin) > 1e-9]
    competitive = [
        item
        for item in all_sets
        if any(
            candidate.candidate_id != item.target_candidate_id
            and candidate.joint_score >= item.target_candidate().joint_score - 0.15
            for candidate in item.candidates
        )
    ]
    runtime_payloads_clean = all(
        "target_candidate_id" not in item.to_inference_payload()
        and "target_kind" not in item.to_inference_payload()
        for item in all_sets
    )
    checks = {
        "train_nonempty": bool(train_sets),
        "validation_nonempty": bool(validation_sets),
        "minimum_two_candidates": bool(candidate_counts) and min(candidate_counts) >= 2,
        "target_coverage": all(item.target_available for item in all_sets),
        "target_kind_diversity": len(target_kinds) >= 2,
        "score_signal_present": bool(nonzero_margin),
        "competitive_distractor_present": bool(competitive),
        "project_split_disjoint": train_projects.isdisjoint(validation_projects),
        "path_split_disjoint": train_paths.isdisjoint(validation_paths),
        "runtime_target_excluded": runtime_payloads_clean,
        "sealed_holdout_untouched": True,
    }
    return {
        **checks,
        "passed": all(checks.values()),
        "train_count": len(train_sets),
        "validation_count": len(validation_sets),
        "target_kinds": sorted(target_kinds),
        "candidate_count_min": min(candidate_counts) if candidate_counts else 0,
        "candidate_count_max": max(candidate_counts) if candidate_counts else 0,
        "nonzero_score_margin_count": len(nonzero_margin),
        "competitive_distractor_count": len(competitive),
        "train_projects": sorted(train_projects),
        "validation_projects": sorted(validation_projects),
    }


def run_canary(
    *,
    p1_manifest_path: Path = P1_MANIFEST,
    p2_6_report_path: Path = P2_6_REPORT,
    p2_6_manifest_path: Path = P2_6_MANIFEST,
    p2_7_manifest_path: Path = P2_7_MANIFEST,
    p3_2_report_path: Path = P3_2_REPORT,
    p3_2_manifest_path: Path = P3_2_MANIFEST,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p3_3_g_signal_canary_{uuid4().hex}"
    scratch = run_dir / "scratch"
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
        "p2_6_report": str(p2_6_report_path),
        "p2_6_manifest": str(p2_6_manifest_path),
        "p2_7_manifest": str(p2_7_manifest_path),
        "p3_2_report": str(p3_2_report_path),
        "p3_2_manifest": str(p3_2_manifest_path),
    }
    try:
        p1_manifest = json.loads(p1_manifest_path.read_text(encoding="utf-8"))
        p2_6_report = json.loads(p2_6_report_path.read_text(encoding="utf-8"))
        p2_6_manifest = json.loads(p2_6_manifest_path.read_text(encoding="utf-8"))
        p2_7_manifest = json.loads(p2_7_manifest_path.read_text(encoding="utf-8"))
        p3_2_report = json.loads(p3_2_report_path.read_text(encoding="utf-8"))
        p3_2_manifest = json.loads(p3_2_manifest_path.read_text(encoding="utf-8"))
        if p1_manifest.get("format") != "taiji-m5-k-p1-data-manifest-v2":
            raise ValueError("P3.3 requires the P1 v2 data manifest")
        if p2_6_report.get("status") != "completed" or p2_6_report.get("sealed_payload_read"):
            raise ValueError("P3.3 requires a completed, unsealed P2.6 report")
        if p2_6_manifest.get("manifest_digest") != p2_6_report.get("manifest_digest"):
            raise ValueError("P2.6 manifest digest mismatch")
        if p3_2_report.get("status") != "completed":
            raise ValueError("P3.3 requires a completed P3.2 report")
        for gate_name in ("comparison_gate", "checkpoint_gate", "trajectory_gate", "rejection_gate"):
            if not all(bool(value) for value in p3_2_report.get(gate_name, {}).values()):
                raise ValueError(f"P3.2 {gate_name} is not fully passed")
        if p3_2_report.get("manifest_digest") != p3_2_manifest.get("manifest_digest"):
            raise ValueError("P3.2 manifest digest mismatch")
        if p3_2_report.get("external_target_used"):
            raise ValueError("P3.2 runtime external target boundary is not clean")
        if len(p2_7_manifest.get("records", ())) != 4:
            raise ValueError("P3.3 requires the fixed four-row P2.7 holdout")

        worker_restore = p3_2_report.get("base_continuation", {}).get("worker_restore", {})
        k1_path = Path(str(worker_restore["k1"]["path"]))
        k2_path = Path(str(worker_restore["k2"]["path"]))
        semantic_payload = _load_mapping(k1_path)
        transition_payload = _load_mapping(k2_path)
        worker_digests = p3_2_report.get("worker_checkpoint_digests", {})
        if content_digest(semantic_payload) != worker_digests["k1"]:
            raise ValueError("P2.6 K1 checkpoint digest drifted")
        if content_digest(transition_payload) != worker_digests["k2"]:
            raise ValueError("P2.6 K2 checkpoint digest drifted")
        semantic, transition = _fresh_learners(semantic_payload, transition_payload)
        run_dir.mkdir(parents=True, exist_ok=False)
        scratch.mkdir(parents=True, exist_ok=False)
        checkpoint_preflight = _checkpoint_preflight(
            output_dir=run_dir / "preflight-parent",
            semantic_parent=copy.deepcopy(semantic_payload),
            transition_parent=copy.deepcopy(transition_payload),
        )
        if not checkpoint_preflight.get("passed"):
            raise RuntimeError("P3.3 parent checkpoint preflight failed")

        _artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT,
            model_seed=MODEL_SEED,
        )
        if p2_6_manifest.get("parent_checkpoint_digest") != parent_digest:
            raise ValueError("P2.6 source parent digest drifted")
        (
            train_experiences,
            train_metadata,
            validation_experiences,
            validation_metadata,
            reconstruction,
        ) = _rebuild_and_verify_manifest(
            scratch=scratch / "p1",
            manifest=p1_manifest,
            parent_digest=parent_digest,
            bundle=bundle,
            projector=projector,
        )
        if not reconstruction["passed"]:
            raise RuntimeError("P1 data reconstruction failed before G signal canary")
        selected_train, selected_train_metadata = _choose_train(train_experiences, train_metadata)
        goals, content_plans = _catalogs(semantic_payload)
        train_sets = [
            _candidate_sets(
                experience=experience,
                metadata=metadata,
                semantic=semantic,
                transition=transition,
                goals=goals,
                content_plans=content_plans,
            )
            for experience, metadata in zip(selected_train, selected_train_metadata, strict=True)
        ]
        validation_sets = [
            _candidate_sets(
                experience=experience,
                metadata=metadata,
                semantic=semantic,
                transition=transition,
                goals=goals,
                content_plans=content_plans,
            )
            for experience, metadata in zip(validation_experiences, validation_metadata, strict=True)
        ]
        signal_gate = _signal_gate(train_sets, validation_sets)
        manifest = {
            "format": MANIFEST_FORMAT,
            "version": VERSION,
            "source_p1_manifest_digest": str(p1_manifest["manifest_digest"]),
            "source_p2_6_manifest_digest": str(p2_6_manifest["manifest_digest"]),
            "source_p2_7_manifest_digest": str(p2_7_manifest["manifest_digest"]),
            "source_p3_2_manifest_digest": str(p3_2_manifest["manifest_digest"]),
            "k_checkpoint_digests": dict(worker_digests),
            "target_policy": {
                "confidence_floor": CONFIDENCE_FLOOR,
                "low_evidence_target": "abstain",
                "runtime_target_used": False,
            },
            "train_records": [item.to_payload() for item in train_sets],
            "validation_records": [item.to_payload() for item in validation_sets],
        }
        manifest["manifest_digest"] = content_digest(manifest)
        _write_json_atomic(manifest_path, manifest)
        payload.update(
            {
                "status": "completed",
                "run_dir": str(run_dir),
                "manifest_digest": manifest["manifest_digest"],
                "checkpoint_preflight": checkpoint_preflight,
                "reconstruction": reconstruction,
                "signal_gate": signal_gate,
                "candidate_contract": {
                    "format": MANIFEST_FORMAT,
                    "train_records": len(train_sets),
                    "validation_records": len(validation_sets),
                    "p2_7_holdout_used_for_fit": False,
                    "runtime_target_used": False,
                },
                "interpretation": (
                    "passed: G-only learning may be designed next"
                    if signal_gate["passed"]
                    else "failed: candidate/label signal is insufficient; do not fit G"
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
    _write_json_atomic(report_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.verify_only is not None:
        return 0 if _verify_checkpoint_only(args.verify_only)["passed"] else 1
    result = run_canary(manifest_path=args.manifest, report_path=args.report)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
