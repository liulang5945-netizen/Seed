"""Audit C-entry sealed-test and fixed-large input compatibility.

This is a read-only admission check.  It intentionally rejects an older
fixed-large artifact when its source course does not cover the exact
target-aware training paths declared by the C-entry manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m5_k2_multistep_composition import (  # noqa: E402
    _train_episode_paths,
)
from taiji import content_digest  # noqa: E402

REPORT_FORMAT = "taiji-m4v2-b3-k-c-entry-input-preflight-v1"
VERSION = 1
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m4v2_b3_k_c_entry_evaluation_v1.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_c_entry_input_preflight_20260910.json"
)
DEFAULT_FIXED_LARGE_ROOT = (
    PROJECT_ROOT / "checkpoints" / "taiji_k_fixed_large_c_entry"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError(f"fixed-large artifact is not a mapping: {path}")
    return dict(raw)


def _fixed_large_cell(
    *,
    model_seed: int,
    parent_digest: str,
    artifact_root: Path,
    required_training_paths: set[str],
    required_validation_paths: set[str],
    expected_course_contract_digest: str,
    course_seed: int,
) -> dict[str, Any]:
    path = (
        artifact_root
        / f"model_{model_seed}"
        / f"course_{course_seed}"
        / "taiji_c_entry_k_fixed_large_ensemble.pt"
    )
    cell: dict[str, Any] = {
        "model_seed": int(model_seed),
        "artifact_path": str(path),
        "artifact_exists": path.is_file(),
        "parent_matches": False,
        "course_seed_matches": False,
        "format_matches": False,
        "source_manifest_format_matches": False,
        "course_contract_digest_matches": False,
        "source_training_paths": [],
        "source_validation_paths": [],
        "missing_required_training_paths": [],
        "validation_paths_match": False,
        "optimizer_state_present": None,
        "fresh_restore_digest_present": False,
        "ready": False,
    }
    if not path.is_file():
        cell["blocking_reason"] = "fixed-large artifact is missing"
        return cell
    try:
        payload = _load_checkpoint(path)
        source = payload.get("source_manifest")
        if not isinstance(source, dict):
            raise TypeError("source_manifest is missing")
        path_sets = source.get("path_sets")
        if not isinstance(path_sets, dict):
            raise TypeError("source_manifest.path_sets is missing")
        training_paths = {
            str(item) for item in path_sets.get("training", [])
        }
        validation_paths = {
            str(item) for item in path_sets.get("formal_holdout", [])
        }
        missing = sorted(required_training_paths - training_paths)
        cell.update(
            {
                "format_matches": payload.get("format")
                == "taiji-k-fixed-large-c-entry-ensemble-v1",
                "source_manifest_format_matches": source.get("format")
                == "taiji-k-fixed-large-c-entry-source-v1",
                "course_contract_digest_matches": source.get(
                    "course_contract_digest"
                )
                == expected_course_contract_digest,
                "parent_matches": payload.get("parent_checkpoint_digest")
                == parent_digest,
                "course_seed_matches": payload.get("course_seed") == int(course_seed),
                "source_training_paths": sorted(training_paths),
                "source_validation_paths": sorted(validation_paths),
                "missing_required_training_paths": missing,
                "validation_paths_match": validation_paths
                == required_validation_paths,
                "optimizer_state_present": bool(
                    payload.get("optimizer_state_present", True)
                ),
                "fresh_restore_digest_present": bool(
                    payload.get("ensemble_checkpoint_digest")
                ),
            }
        )
        cell["ready"] = all(
            (
                cell["format_matches"],
                cell["source_manifest_format_matches"],
                cell["course_contract_digest_matches"],
                cell["parent_matches"],
                cell["course_seed_matches"],
                not missing,
                cell["validation_paths_match"],
                cell["optimizer_state_present"] is False,
                cell["fresh_restore_digest_present"],
            )
        )
        if not cell["ready"]:
            cell["blocking_reason"] = (
                "existing fixed-large source course does not match the C-entry "
                "training/validation contract"
            )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        cell["blocking_reason"] = f"fixed-large artifact unreadable: {exc}"
    return cell


def run_preflight(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
    fixed_large_root: Path = DEFAULT_FIXED_LARGE_ROOT,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sealed = manifest["sealed_test_split"]
    sealed_path = PROJECT_ROOT / str(sealed["artifact_path"])
    sealed_checks = {
        "path": str(sealed_path),
        "exists": sealed_path.is_file(),
        "status": sealed.get("status"),
        "artifact_digest_matches": False,
        "sha256_matches": False,
        "record_disjoint": False,
        "ready": False,
    }
    if sealed_path.is_file():
        artifact = json.loads(sealed_path.read_text(encoding="utf-8"))
        unsigned = {
            key: value
            for key, value in artifact.items()
            if key not in {"artifact_digest", "materializer_sha256"}
        }
        sealed_checks.update(
            {
                "artifact_digest_matches": content_digest(unsigned)
                == sealed.get("artifact_digest"),
                "sha256_matches": _sha256(sealed_path)
                == str(sealed.get("artifact_sha256", "")).lower(),
                "record_disjoint": artifact.get(
                    "record_disjoint_from_existing_fixture", False
                )
                and artifact.get("scores_or_targets_embedded") is False,
            }
        )
        sealed_checks["ready"] = all(
            (
                sealed_checks["exists"],
                sealed_checks["status"] == "materialized-unread",
                sealed_checks["artifact_digest_matches"],
                sealed_checks["sha256_matches"],
                sealed_checks["record_disjoint"],
            )
        )

    variants = manifest["course_contract"]["variants"]
    required_indexes = sorted(
        {
            int(index)
            for variant in variants
            for index in variant["episode_indexes"]
        }
    )
    all_train_variants = _train_episode_paths()
    required_training_paths_by_course = {
        int(variant["course_seed"]): sorted(
            {
                path
                for index in variant["episode_indexes"]
                for path in all_train_variants[int(index)]
            }
        )
        for variant in manifest["course_contract"]["variants"]
    }
    required_training_paths = {
        path
        for paths in required_training_paths_by_course.values()
        for path in paths
    }
    required_validation_paths = {
        path
        for episode in manifest["validation_split"]["episodes"]
        for path in episode
    }
    fixed_large_cells = [
        _fixed_large_cell(
            model_seed=int(parent["model_seed"]),
            course_seed=int(variant["course_seed"]),
            parent_digest=str(parent["parent_checkpoint_digest"]),
            artifact_root=fixed_large_root,
            required_training_paths=set(
                required_training_paths_by_course[int(variant["course_seed"])]
            ),
            required_validation_paths=required_validation_paths,
            expected_course_contract_digest=str(
                manifest["course_contract_digest"]
            ),
        )
        for parent in manifest["parent_models"]
        for variant in manifest["course_contract"]["variants"]
    ]
    fixed_large_ready = all(cell["ready"] for cell in fixed_large_cells)
    formal_input_ready = bool(sealed_checks["ready"] and fixed_large_ready)
    report = {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "status": "ready" if formal_input_ready else "blocked",
        "run_kind": "c-entry-input-preflight",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "sealed_test": sealed_checks,
        "required_course_episode_indexes": required_indexes,
        "required_training_paths": sorted(required_training_paths),
        "required_training_paths_by_course": required_training_paths_by_course,
        "required_validation_paths": sorted(required_validation_paths),
        "fixed_large_cells": fixed_large_cells,
        "fixed_large_ready": fixed_large_ready,
        "formal_input_ready": formal_input_ready,
        "can_start_formal": False,
        "can_promote": False,
        "blocking_reason": (
            None
            if formal_input_ready
            else "sealed-test or fixed-large input contract is not ready"
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--fixed-large-root", type=Path, default=DEFAULT_FIXED_LARGE_ROOT)
    args = parser.parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    report = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    fixed_large = (
        args.fixed_large_root
        if args.fixed_large_root.is_absolute()
        else PROJECT_ROOT / args.fixed_large_root
    )
    result = run_preflight(
        manifest_path=manifest,
        report_path=report,
        fixed_large_root=fixed_large,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
