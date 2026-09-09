"""Audit per-episode K update signatures and compare them with batch updates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_b3_k_loss_diagnostic import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    run_diagnostic,
)

REPORT_FORMAT = "taiji-m4v2-b3-k-update-signature-audit-v1"
VERSION = 1
EPISODE_SEEDS = (0, 1, 2, 3, 4, 5)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_update_signature_audit_20260910.json"
)
DEFAULT_CANDIDATE_ROOT = (
    PROJECT_ROOT / "output" / "taiji_m4v2_b3_k_update_signature_audit_20260910"
)
BATCH_REPORTS = {
    "three_sliding": PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_three_batch_20260910.json",
    "three_non_sliding": PROJECT_ROOT
    / "reports"
    / "taiji_m4v2_b3_k_non_sliding_20260910.json",
}


def _collision_groups(values: list[str | None]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        if value is not None:
            groups[value].append(index)
    return {digest: indexes for digest, indexes in groups.items() if len(indexes) > 1}


def _batch_comparison(single_digests: set[str], report_paths: dict[str, Path]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for name, path in report_paths.items():
        batch = json.loads(path.read_text(encoding="utf-8"))
        digests = list(batch.get("candidate_update_digests", []))
        comparison[name] = {
            "report": str(path),
            "candidate_update_digests": digests,
            "candidate_digest_matches_single": [digest in single_digests for digest in digests],
            "candidate_updates_distinct": bool(batch.get("candidate_updates_distinct")),
            "train_episode_indexes": batch.get("train_episode_indexes", []),
        }
    return comparison


def run_audit(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
    model_seed: int = 17,
    episode_seeds: tuple[int, ...] = EPISODE_SEEDS,
) -> dict[str, Any]:
    cells = []
    for episode_seed in episode_seeds:
        report = run_diagnostic(
            artifact_dir=artifact_dir,
            candidate_dir=candidate_root / f"episode_{episode_seed}" / f"model_{model_seed}",
            model_seed=model_seed,
            course_seed=episode_seed,
            train_episode_count=1,
        )
        cells.append({"episode_seed": episode_seed, "report": report})

    reports = [item["report"] for item in cells]
    parent_values = [report.get("parent_checkpoint_digest") for report in reports]
    parent_digests = {value for value in parent_values if value is not None}
    semantic_inputs = [
        report.get("train_fit_input_digests", [{}])[0].get("semantic_input_digest")
        for report in reports
    ]
    transition_inputs = [
        report.get("train_fit_input_digests", [{}])[0].get("transition_input_digest")
        for report in reports
    ]
    candidate_digests = [report.get("candidate_worker_bundle_digest") for report in reports]
    worker_candidate_digests = {
        worker_id: [
            next(
                (
                    digest
                    for candidate_worker_id, digest in report.get("receipt", {}).get(
                        "candidate_worker_checkpoint_digests", []
                    )
                    if candidate_worker_id == worker_id
                ),
                None,
            )
            for report in reports
        ]
        for worker_id in ("k1.semantic", "k2.transition")
    }
    single_digest_set = {digest for digest in candidate_digests if digest is not None}
    technical_pass = all(report.get("status") == "passed" for report in reports)
    same_parent = len(parent_digests) == 1 and None not in parent_values
    inputs_distinct = (
        len(semantic_inputs) == len(set(semantic_inputs))
        and None not in semantic_inputs
        and len(transition_inputs) == len(set(transition_inputs))
        and None not in transition_inputs
    )
    report = {
        "report_format": REPORT_FORMAT,
        "version": VERSION,
        "status": "passed" if technical_pass and same_parent else "failed",
        "run_kind": "per-episode-update-signature-audit",
        "model_seed": model_seed,
        "episode_seeds": list(episode_seeds),
        "cell_count": len(cells),
        "same_parent": same_parent,
        "parent_checkpoint_digests": sorted(parent_digests),
        "semantic_input_digests": semantic_inputs,
        "transition_input_digests": transition_inputs,
        "input_signatures_distinct": inputs_distinct,
        "candidate_update_digests": candidate_digests,
        "candidate_update_collision_groups": _collision_groups(candidate_digests),
        "worker_candidate_update_digests": worker_candidate_digests,
        "worker_candidate_update_collision_groups": {
            worker_id: _collision_groups(digests)
            for worker_id, digests in worker_candidate_digests.items()
        },
        "candidate_updates_distinct": len(candidate_digests) == len(single_digest_set),
        "batch_comparison": _batch_comparison(single_digest_set, BATCH_REPORTS),
        "technical_gate_passed": technical_pass and same_parent and inputs_distinct,
        "can_start_r6_formal": False,
        "can_promote": False,
        "candidate_promoted": False,
        "cells": cells,
        "blocking_reason": (
            None
            if technical_pass and same_parent and inputs_distinct
            else "per-episode update signature audit technical Gate failed"
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--model-seed", type=int, default=17)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    artifact_dir = (
        args.artifact_dir
        if args.artifact_dir.is_absolute()
        else PROJECT_ROOT / args.artifact_dir
    )
    candidate_root = (
        args.candidate_root
        if args.candidate_root.is_absolute()
        else PROJECT_ROOT / args.candidate_root
    )
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report = run_audit(
        artifact_dir=artifact_dir,
        candidate_root=candidate_root,
        model_seed=args.model_seed,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
