"""Preflight the frozen C-entry capacity-parity artifact contract.

The preflight is intentionally non-training. It verifies the frozen contract
and the existing fixed-large control, then reports that a new candidate parity
artifact is still required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPORT_FORMAT = "taiji-m4v2-b3-k-c-capacity-parity-input-preflight-v1"
MANIFEST_FORMAT = "taiji-m4v2-b3-k-c-capacity-parity-manifest-v1"
FIXED_LARGE_REPORT_FORMAT = "taiji-m4v2-b3-k-c-fixed-large-build-v1"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    DEFAULT_ROOT
    / "plans"
    / "manifests"
    / "taiji_m4v2_b3_k_c_capacity_parity_v1.json"
)
DEFAULT_FIXED_LARGE_REPORT = (
    DEFAULT_ROOT
    / "reports"
    / "taiji_m4v2_b3_k_c_fixed_large_build_20260910.json"
)
DEFAULT_OUTPUT = (
    DEFAULT_ROOT
    / "reports"
    / "taiji_m4v2_b3_k_c_capacity_parity_input_preflight_20260910.json"
)


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _fixed_cell(
    report: dict[str, Any], model_seed: int, course_seed: int
) -> dict[str, Any]:
    for cell in report.get("cells", []):
        if (
            isinstance(cell, dict)
            and cell.get("model_seed") == model_seed
            and cell.get("course_seed") == course_seed
        ):
            return cell
    raise ValueError(f"missing fixed-large cell model={model_seed} course={course_seed}")


def _checkpoint_count(resource: dict[str, Any]) -> int:
    paths = resource.get("checkpoint_write_paths")
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise ValueError("fixed-large checkpoint paths must be a list of strings")
    if len(paths) != len(set(paths)):
        raise ValueError("fixed-large checkpoint paths contain duplicates")
    return len(paths)


def build_preflight(
    manifest: dict[str, Any],
    fixed_report: dict[str, Any],
    *,
    manifest_path: Path,
    fixed_report_path: Path,
) -> dict[str, Any]:
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("capacity-parity manifest format mismatch")
    if fixed_report.get("report_format") != FIXED_LARGE_REPORT_FORMAT:
        raise ValueError("fixed-large report format mismatch")

    model_seeds = manifest["scope"]["model_seeds"]
    course_seeds = manifest["scope"]["course_seeds"]
    target_parameter_bytes = _int(
        manifest["capacity_contract"]["target_candidate_parameter_bytes"],
        "target_candidate_parameter_bytes",
    )
    target_update_steps = _int(
        manifest["training_budget_contract"]["target_parameter_update_steps_per_cell"],
        "target_parameter_update_steps_per_cell",
    )
    target_checkpoint_count = _int(
        manifest["training_budget_contract"]["checkpoint_policy"][
            "logical_training_checkpoint_count_per_cell"
        ],
        "logical_training_checkpoint_count_per_cell",
    )

    cells = []
    fixed_ready = True
    for model_seed in model_seeds:
        for course_seed in course_seeds:
            fixed_cell = _fixed_cell(fixed_report, model_seed, course_seed)
            resource = fixed_cell["resource"]
            parameter_bytes = _int(resource["parameter_bytes"], "parameter_bytes")
            update_steps = _int(
                resource["training_update_steps"], "training_update_steps"
            )
            checkpoint_count = _checkpoint_count(resource)
            parent_key = f"model_{model_seed}"
            parent_matches = (
                fixed_cell.get("parent_checkpoint_digest")
                == manifest["parent_contract"][parent_key]
            )
            course = manifest["course_contract"]["variants"][str(course_seed)]
            episode_indexes_match = (
                fixed_cell.get("train_episode_indexes") == course["episode_indexes"]
            )
            cell_ready = (
                parameter_bytes == target_parameter_bytes
                and update_steps == target_update_steps
                and checkpoint_count == target_checkpoint_count
                and parent_matches
                and episode_indexes_match
                and fixed_cell.get("training_performed") is True
                and fixed_cell.get("ensemble_fresh_restore") is True
            )
            fixed_ready = fixed_ready and cell_ready
            cells.append(
                {
                    "model_seed": model_seed,
                    "course_seed": course_seed,
                    "fixed_large": {
                        "parameter_bytes": parameter_bytes,
                        "training_update_steps": update_steps,
                        "checkpoint_write_count": checkpoint_count,
                        "parent_matches": parent_matches,
                        "episode_indexes_match": episode_indexes_match,
                        "fresh_restore": fixed_cell.get("ensemble_fresh_restore"),
                        "training_performed": fixed_cell.get("training_performed"),
                    },
                    "ready": cell_ready,
                }
            )

    candidate_root = DEFAULT_ROOT / manifest["artifact_contract"]["candidate_root"]
    candidate_present = candidate_root.exists()
    candidate_design_ready = (
        manifest["artifact_contract"]["candidate_route_design"]["design_status"]
        == "frozen"
    )
    status = (
        "ready-for-candidate-design"
        if fixed_ready and not candidate_present and not candidate_design_ready
        else "ready-for-candidate-parity-build"
        if fixed_ready and not candidate_present and candidate_design_ready
        else "blocked"
    )
    blocking_reason = (
        "fixed-large input contract is ready; distinct candidate route design is not frozen"
        if fixed_ready and not candidate_present and not candidate_design_ready
        else "candidate parity artifact is not built; this preflight is non-training"
        if fixed_ready and not candidate_present
        else "fixed-large input contract failed or candidate artifact already exists"
    )
    return {
        "report_format": REPORT_FORMAT,
        "version": 1,
        "run_kind": "c-entry-capacity-parity-input-preflight",
        "status": status,
        "manifest_path": str(manifest_path.resolve()),
        "fixed_large_report_path": str(fixed_report_path.resolve()),
        "manifest_format": manifest.get("format"),
        "cell_count": len(cells),
        "fixed_large_ready": fixed_ready,
        "candidate_artifact_root": str(candidate_root),
        "candidate_artifact_present": candidate_present,
        "candidate_design_ready": candidate_design_ready,
        "can_start_candidate_design": fixed_ready and not candidate_present,
        "can_start_candidate_parity_build": (
            fixed_ready and candidate_design_ready and not candidate_present
        ),
        "can_start_formal": False,
        "can_promote": False,
        "blocking_reason": blocking_reason,
        "cells": cells,
        "contract": {
            "target_parameter_bytes": target_parameter_bytes,
            "target_update_steps_per_cell": target_update_steps,
            "target_checkpoint_count_per_cell": target_checkpoint_count,
            "parameter_ratio_tolerance": manifest["capacity_contract"][
                "max_parameter_ratio_difference"
            ],
            "candidate_upward_policy": manifest["capacity_contract"]["policy"],
            "do_not_shrink_fixed_large": manifest["capacity_contract"][
                "do_not_shrink_fixed_large"
            ],
            "naive_replica_duplication_forbidden": manifest["artifact_contract"][
                "candidate_route_design"
            ]["naive_replica_duplication_forbidden"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--fixed-large-report", type=Path, default=DEFAULT_FIXED_LARGE_REPORT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_preflight(
        _load(args.manifest),
        _load(args.fixed_large_report),
        manifest_path=args.manifest,
        fixed_report_path=args.fixed_large_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
