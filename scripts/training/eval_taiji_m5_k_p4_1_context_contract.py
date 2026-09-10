"""Preflight a fair P4.1 capacity/representation comparison contract.

P4.0 found a non-monotonic fixed-G failure as candidate sets widened.  This
step does not fit another model on those validation artifacts.  It instead
binds candidate-level features to an explicit candidate-set context contract,
prepares content-addressed fixed-large and context-lesion reference arms, and
proves that the reference checkpoints can be restored independently.  The
result is deliberately attribution-inconclusive; an isolated train/holdout
experiment is required before any topology growth decision.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m5_k_p2_validation_pilot import DEFAULT_OUTPUT_ROOT  # noqa: E402
from scripts.training.eval_taiji_m5_k_p3_5_g_learning import (  # noqa: E402
    P3_2_MANIFEST,
    _independent_g_restore,
    _load_json,
    _load_mapping,
)
from taiji import (  # noqa: E402
    GSelectionBehaviorSet,
    GSelectionCandidateSet,
    GSelectionLearner,
    content_digest,
)
from taiji.g_selection import G_SELECTION_FEATURE_NAMES, GSelectionCandidate  # noqa: E402

REPORT_FORMAT = "taiji-m5-k-p4-1-context-contract-v1"
MANIFEST_FORMAT = "taiji-m5-k-p4-1-context-contract-manifest-v1"
REFERENCE_FORMAT = "taiji-m5-k-p4-1-reference-linear-v1"
VERSION = 1
P4_0_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_0_capacity_pressure_manifest_v1.json"
)
P4_0_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_0_capacity_pressure_20260911.json"
P3_5_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_5_g_learning_20260911.json"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_1_context_contract_manifest_v1.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_1_context_contract_20260911.json"

CONTEXT_FEATURE_NAMES = (
    "candidate_count_norm",
    "proposal_fraction",
    "safe_fraction",
    "reobserve_fraction",
    "joint_score_mean",
    "joint_score_std",
    "joint_score_max",
    "candidate_rank_norm",
    "candidate_joint_minus_mean",
)
REFERENCE_WEIGHTS = (0.0,) * len(CONTEXT_FEATURE_NAMES)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise ValueError(f"{name} must be finite")
    return result


def _context_payload(candidate_set: GSelectionCandidateSet) -> dict[str, Any]:
    candidates = tuple(candidate_set.candidates)
    if len(candidates) < 2:
        raise ValueError("P4.1 context requires at least two candidates")
    role_counts = {
        role: sum(item.candidate_role == role for item in candidates)
        for role in ("proposal", "abstain", "reobserve")
    }
    joint_scores = tuple(float(item.joint_score) for item in candidates)
    mean = sum(joint_scores) / len(joint_scores)
    variance = sum((value - mean) ** 2 for value in joint_scores) / len(joint_scores)
    maximum = max(joint_scores)
    role_priority = {"abstain": 2, "reobserve": 1, "proposal": 0}
    ranking = sorted(
        candidates,
        key=lambda item: (
            -item.joint_score,
            -role_priority[item.candidate_role],
            item.candidate_id,
        ),
    )
    rank_by_id = {item.candidate_id: index for index, item in enumerate(ranking)}
    set_features = (
        min(1.0, len(candidates) / 12.0),
        role_counts["proposal"] / len(candidates),
        (role_counts["abstain"] + role_counts["reobserve"]) / len(candidates),
        role_counts["reobserve"] / len(candidates),
        mean,
        variance**0.5,
        maximum,
    )
    candidate_features = {
        item.candidate_id: (
            *set_features,
            1.0 - rank_by_id[item.candidate_id] / max(1, len(candidates) - 1),
            item.joint_score - mean,
        )
        for item in candidates
    }
    payload = {
        "format": "taiji-g-selection-context-v1",
        "version": 1,
        "candidate_set_digest": candidate_set.candidate_set_digest,
        "feature_names": list(CONTEXT_FEATURE_NAMES),
        "set_feature_names": list(CONTEXT_FEATURE_NAMES[:7]),
        "candidate_feature_names": list(CONTEXT_FEATURE_NAMES[7:]),
        "set_features": [float(value) for value in set_features],
        "candidate_features": {
            candidate_id: [float(value) for value in values]
            for candidate_id, values in sorted(candidate_features.items())
        },
    }
    payload["context_digest"] = content_digest(payload)
    return payload


def _collision_rate(vectors: Mapping[str, Sequence[float]]) -> float:
    groups: dict[tuple[float, ...], int] = {}
    for values in vectors.values():
        key = tuple(round(float(value), 8) for value in values)
        groups[key] = groups.get(key, 0) + 1
    colliding = sum(count for count in groups.values() if count > 1)
    return colliding / len(vectors) if vectors else 0.0


def _reference_payload(
    *,
    arm: str,
    parent_g_checkpoint_digest: str,
    candidate_weights: Sequence[float],
    bias: float,
    context_enabled: bool,
) -> dict[str, Any]:
    context_weights = list(
        REFERENCE_WEIGHTS if context_enabled else (0.0,) * len(CONTEXT_FEATURE_NAMES)
    )
    payload: dict[str, Any] = {
        "format": REFERENCE_FORMAT,
        "version": VERSION,
        "arm": arm,
        "parent_g_checkpoint_digest": parent_g_checkpoint_digest,
        "candidate_feature_names": list(G_SELECTION_FEATURE_NAMES),
        "context_feature_names": list(CONTEXT_FEATURE_NAMES),
        "candidate_weights": [float(value) for value in candidate_weights],
        "context_weights": context_weights,
        "bias": float(bias),
        "input_dim": len(candidate_weights) + len(context_weights),
        "parameter_count": len(candidate_weights) + len(context_weights) + 1,
        "training_performed": False,
        "fit_called": False,
        "context_enabled": bool(context_enabled),
        "lesion": arm == "context-lesion",
        "reference_status": "prepared_no_fit",
    }
    payload["checkpoint_digest"] = content_digest(payload)
    return payload


def _validate_reference(payload: Mapping[str, Any]) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "checkpoint_digest"}
    if content_digest(unsigned) != str(payload["checkpoint_digest"]):
        raise ValueError("P4.1 reference checkpoint digest mismatch")
    if int(payload["input_dim"]) != len(payload["candidate_weights"]) + len(
        payload["context_weights"]
    ):
        raise ValueError("P4.1 reference input dimension drifted")
    if int(payload["parameter_count"]) != int(payload["input_dim"]) + 1:
        raise ValueError("P4.1 reference parameter count drifted")
    if payload.get("training_performed") or payload.get("fit_called"):
        raise ValueError("P4.1 reference unexpectedly trained")


def _independent_reference_restore(path: Path) -> dict[str, Any]:
    code = (
        "import json, sys; "
        "from taiji import content_digest; "
        "payload=json.load(open(sys.argv[1], encoding='utf-8')); "
        "digest=payload.pop('checkpoint_digest'); "
        "print(json.dumps({'restore_digest_equal': content_digest(payload)==digest, "
        "'parameter_count': payload['parameter_count'], 'passed': content_digest(payload)==digest}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout.strip()
    parsed: dict[str, Any] = {}
    if stdout:
        parsed = json.loads(stdout.splitlines()[-1])
    return {
        "returncode": completed.returncode,
        "independent_process_restore": completed.returncode == 0 and bool(parsed.get("passed")),
        "stdout": stdout,
        "stderr": completed.stderr,
        **parsed,
    }


def _reference_score(
    payload: Mapping[str, Any],
    candidate: GSelectionCandidate,
    context_features: Sequence[float],
) -> float:
    candidate_values = candidate.feature_vector
    candidate_weights = tuple(float(value) for value in payload["candidate_weights"])
    context_weights = tuple(float(value) for value in payload["context_weights"])
    if len(candidate_values) != len(candidate_weights):
        raise ValueError("P4.1 candidate feature dimension drifted")
    if len(context_features) != len(context_weights):
        raise ValueError("P4.1 context feature dimension drifted")
    return (
        sum(left * right for left, right in zip(candidate_values, candidate_weights, strict=True))
        + sum(left * right for left, right in zip(context_features, context_weights, strict=True))
        + float(payload["bias"])
    )


def _select_reference(
    candidate_set: GSelectionCandidateSet,
    scores: Mapping[str, float],
    *,
    confidence_floor: float,
    selection_margin: float,
) -> str:
    role_priority = {"abstain": 2, "reobserve": 1, "proposal": 0}
    ranked = sorted(
        candidate_set.candidates,
        key=lambda item: (
            -float(scores[item.candidate_id]),
            -role_priority[item.candidate_role],
            item.candidate_id,
        ),
    )
    selected = ranked[0]
    safe = max(
        (
            item
            for item in candidate_set.candidates
            if item.candidate_role in {"abstain", "reobserve"}
        ),
        key=lambda item: (
            1 if item.candidate_role == "abstain" else 0,
            item.confidence,
            item.candidate_id,
        ),
    )
    if selected.candidate_role == "proposal":
        unsafe = (
            selected.goal is None
            or selected.content_plan is None
            or selected.confidence < confidence_floor
            or float(scores[selected.candidate_id])
            <= float(scores[safe.candidate_id]) + selection_margin
        )
        if unsafe:
            selected = safe
    elif (
        float(scores[selected.candidate_id]) <= float(scores[safe.candidate_id]) + selection_margin
    ):
        selected = safe
    return selected.candidate_id


def run_preflight(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p4_1_context_contract_{uuid4().hex}"
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "training_performed": False,
        "fit_called": False,
        "external_target_used": False,
        "sealed_payload_read": False,
        "growth_admitted": False,
        "can_promote": False,
        "manifest": str(manifest_path),
        "report": str(report_path),
    }
    try:
        p4_manifest = _load_json(P4_0_MANIFEST)
        p4_report = _load_json(P4_0_REPORT)
        if p4_report.get("status") != "completed" or not p4_report.get("scan_passed"):
            raise ValueError("P4.1 requires the completed P4.0 scan")
        if p4_report.get("manifest_digest") != p4_manifest.get("manifest_digest"):
            raise ValueError("P4.0 manifest/report digest mismatch")
        if content_digest(
            {key: value for key, value in p4_manifest.items() if key != "manifest_digest"}
        ) != p4_manifest.get("manifest_digest"):
            raise ValueError("P4.0 manifest content digest mismatch")
        if p4_report.get("growth_admitted") or p4_manifest.get("fit_policy", {}).get(
            "growth_admitted"
        ):
            raise ValueError("P4.1 cannot consume an admitted growth artifact")
        p3_2_manifest = _load_json(P3_2_MANIFEST)
        p3_5_report = _load_json(P3_5_REPORT)
        trained_path = Path(str(p3_5_report["g_trained_checkpoint"]["path"]))
        trained_payload = _load_mapping(trained_path)
        trained = GSelectionLearner.from_checkpoint(trained_payload, device="cpu")
        trained.assert_lineage(
            parent_manifest_digest=str(p3_2_manifest["manifest_digest"]),
            k_checkpoint_digests=p4_manifest["k_checkpoint_digests"],
        )
        g_restore = _independent_g_restore(trained_path)
        if not g_restore.get("independent_process_restore"):
            raise RuntimeError("P4.1 trained-G independent restore failed")
        run_dir.mkdir(parents=True, exist_ok=False)
        reference_dir = run_dir / "references"
        reference_dir.mkdir(parents=True, exist_ok=False)
        base_weights = [
            float(value) for value in trained.model.weight.detach().cpu().flatten().tolist()
        ]
        bias = float(trained.model.bias.detach().cpu().reshape(()).item())
        fixed_large = _reference_payload(
            arm="fixed-large-reference",
            parent_g_checkpoint_digest=str(p4_manifest["g_trained_checkpoint_digest"]),
            candidate_weights=base_weights,
            bias=bias,
            context_enabled=True,
        )
        context_lesion = _reference_payload(
            arm="context-lesion",
            parent_g_checkpoint_digest=str(p4_manifest["g_trained_checkpoint_digest"]),
            candidate_weights=base_weights,
            bias=bias,
            context_enabled=False,
        )
        fixed_large_path = reference_dir / "fixed_large_reference.json"
        lesion_path = reference_dir / "context_lesion_reference.json"
        _write_json_atomic(fixed_large_path, fixed_large)
        _write_json_atomic(lesion_path, context_lesion)
        _validate_reference(fixed_large)
        _validate_reference(context_lesion)
        fixed_large_restore = _independent_reference_restore(fixed_large_path)
        lesion_restore = _independent_reference_restore(lesion_path)

        records: list[dict[str, Any]] = []
        context_digests: set[str] = set()
        for raw in p4_manifest["pressure_records"]:
            candidate_set = GSelectionCandidateSet.from_payload(raw["candidate_set"])
            behavior_set = GSelectionBehaviorSet.from_payload(raw["behavior_set"])
            if behavior_set.candidate_set_digest != candidate_set.candidate_set_digest:
                raise ValueError("P4.1 behavior/candidate binding drifted")
            context = _context_payload(candidate_set)
            context_digests.add(str(context["context_digest"]))
            if set(context["candidate_features"]) != {
                item.candidate_id for item in candidate_set.candidates
            }:
                raise ValueError("P4.1 context candidate coverage drifted")
            fixed_scores = {
                item.candidate_id: _reference_score(
                    fixed_large,
                    item,
                    context["candidate_features"][item.candidate_id],
                )
                for item in candidate_set.candidates
            }
            lesion_scores = {
                item.candidate_id: _reference_score(
                    context_lesion,
                    item,
                    context["candidate_features"][item.candidate_id],
                )
                for item in candidate_set.candidates
            }
            small_decision = trained.select(candidate_set)
            fixed_selected = _select_reference(
                candidate_set,
                fixed_scores,
                confidence_floor=trained.confidence_floor,
                selection_margin=trained.selection_margin,
            )
            lesion_selected = _select_reference(
                candidate_set,
                lesion_scores,
                confidence_floor=trained.confidence_floor,
                selection_margin=trained.selection_margin,
            )
            records.append(
                {
                    "candidate_set_digest": candidate_set.candidate_set_digest,
                    "behavior_digest": behavior_set.behavior_digest,
                    "candidate_width": len(candidate_set.candidates),
                    "context": context,
                    "small_selected_candidate_id": small_decision.selected_candidate_id,
                    "fixed_large_selected_candidate_id": fixed_selected,
                    "context_lesion_selected_candidate_id": lesion_selected,
                    "fixed_large_scores": fixed_scores,
                    "context_lesion_scores": lesion_scores,
                    "fixed_large_matches_small": fixed_selected
                    == small_decision.selected_candidate_id,
                    "context_lesion_matches_small": lesion_selected
                    == small_decision.selected_candidate_id,
                }
            )
        widths = {
            width: sum(record["candidate_width"] == width for record in records)
            for width in (2, 4, 8, 12)
        }
        context_within_set_collision_rate = sum(
            _collision_rate(record["context"]["candidate_features"]) for record in records
        ) / len(records)
        source_gate = {
            "p4_0_manifest_report_match": True,
            "p4_0_scan_passed": True,
            "p4_0_growth_not_admitted": True,
            "g_lineage_valid": True,
            "g_independent_restore": bool(g_restore.get("independent_process_restore")),
        }
        context_gate = {
            "record_count": len(records) == 20,
            "widths_complete": widths == {2: 5, 4: 5, 8: 5, 12: 5},
            "context_digests_unique": len(context_digests) == len(records),
            "context_feature_names_unique": len(set(CONTEXT_FEATURE_NAMES))
            == len(CONTEXT_FEATURE_NAMES),
            "context_within_set_collision_rate": context_within_set_collision_rate,
            "target_and_utility_excluded": all(
                set(record["context"])
                == {
                    "format",
                    "version",
                    "candidate_set_digest",
                    "feature_names",
                    "set_feature_names",
                    "candidate_feature_names",
                    "set_features",
                    "candidate_features",
                    "context_digest",
                }
                for record in records
            ),
        }
        reference_gate = {
            "fixed_large_parameter_count_gt_small": int(fixed_large["parameter_count"])
            > trained.parameter_count,
            "fixed_large_independent_restore": bool(
                fixed_large_restore.get("independent_process_restore")
            ),
            "context_lesion_independent_restore": bool(
                lesion_restore.get("independent_process_restore")
            ),
            "fixed_large_matches_small_without_fit": all(
                record["fixed_large_matches_small"] for record in records
            ),
            "context_lesion_matches_small_without_fit": all(
                record["context_lesion_matches_small"] for record in records
            ),
            "reference_fit_not_called": not fixed_large["fit_called"]
            and not context_lesion["fit_called"],
        }
        attribution = {
            "status": "inconclusive",
            "reason": "reference arms are zero-fit contract controls; P4.1 does not infer capacity from same-set replay",
            "requires_isolated_train_holdout": True,
        }
        manifest = {
            "format": MANIFEST_FORMAT,
            "version": VERSION,
            "source_p4_0_manifest_digest": p4_manifest["manifest_digest"],
            "source_p4_0_report_digest": content_digest(p4_report),
            "source_p3_2_manifest_digest": p3_2_manifest["manifest_digest"],
            "g_trained_checkpoint_digest": p4_manifest["g_trained_checkpoint_digest"],
            "context_feature_names": list(CONTEXT_FEATURE_NAMES),
            "fixed_large_reference_digest": fixed_large["checkpoint_digest"],
            "context_lesion_reference_digest": context_lesion["checkpoint_digest"],
            "fit_policy": {
                "fit_called": False,
                "training_performed": False,
                "validation_only": True,
                "growth_admitted": False,
            },
            "records": [
                {
                    "candidate_set_digest": record["candidate_set_digest"],
                    "behavior_digest": record["behavior_digest"],
                    "context": record["context"],
                }
                for record in records
            ],
        }
        manifest["manifest_digest"] = content_digest(manifest)
        _write_json_atomic(manifest_path, manifest)
        preflight_passed = (
            all(bool(value) for value in source_gate.values())
            and all(
                bool(value)
                for key, value in context_gate.items()
                if key != "context_within_set_collision_rate"
            )
            and all(bool(value) for value in reference_gate.values())
        )
        payload.update(
            {
                "status": "completed",
                "run_dir": str(run_dir),
                "manifest_digest": manifest["manifest_digest"],
                "source_gate": source_gate,
                "context_gate": context_gate,
                "reference_gate": reference_gate,
                "attribution": attribution,
                "fixed_large_reference": {
                    "path": str(fixed_large_path),
                    "digest": fixed_large["checkpoint_digest"],
                    "parameter_count": fixed_large["parameter_count"],
                    "restore": fixed_large_restore,
                },
                "context_lesion_reference": {
                    "path": str(lesion_path),
                    "digest": context_lesion["checkpoint_digest"],
                    "parameter_count": context_lesion["parameter_count"],
                    "restore": lesion_restore,
                },
                "parameter_count": {
                    "g_trained": trained.parameter_count,
                    "fixed_large_reference": fixed_large["parameter_count"],
                    "context_lesion_effective": trained.parameter_count,
                },
                "records": records,
                "preflight_passed": preflight_passed,
                "growth_admitted": False,
                "interpretation": (
                    "completed: context/fixed-large/lesion contracts are restorable; attribution remains inconclusive and no fit is admitted"
                    if preflight_passed
                    else "completed: P4.1 contract preflight failed; growth remains fail-closed"
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_preflight(manifest_path=args.manifest, report_path=args.report)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
