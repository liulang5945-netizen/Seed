"""Audit K1/K2 fit tensors against the colliding per-worker updates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_b3_k_update_signature_audit import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    EPISODE_SEEDS,
    _collision_groups,
    run_audit,
)

REPORT_FORMAT = "taiji-m4v2-b3-k-feature-target-audit-v1"
VERSION = 1
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_feature_target_audit_20260910.json"
)
DEFAULT_CANDIDATE_ROOT = (
    PROJECT_ROOT / "output" / "taiji_m4v2_b3_k_feature_target_audit_20260910"
)
TENSOR_KEYS = (
    "k1.semantic.input_tensor_digest",
    "k1.semantic.target_tensor_digest",
    "k2.transition.input_tensor_digest",
    "k2.transition.target_tensor_digest",
)
WORKER_IDS = ("k1.semantic", "k2.transition")


def run_feature_target_audit(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
    model_seed: int = 17,
    episode_seeds: tuple[int, ...] = EPISODE_SEEDS,
) -> dict[str, Any]:
    report = run_audit(
        artifact_dir=artifact_dir,
        candidate_root=candidate_root,
        model_seed=model_seed,
        episode_seeds=episode_seeds,
    )
    tensor_digests = {
        key: [
            cell["report"].get("train_fit_tensor_digests", [{}])[0].get(key)
            for cell in report["cells"]
        ]
        for key in TENSOR_KEYS
    }
    worker_delta_digests = {
        worker_id: [
            cell["report"].get("parameter_delta_digest", {}).get(worker_id)
            for cell in report["cells"]
        ]
        for worker_id in WORKER_IDS
    }
    report["report_format"] = REPORT_FORMAT
    report["version"] = VERSION
    report["run_kind"] = "feature-target-collision-audit"
    report["fit_tensor_digests"] = tensor_digests
    report["fit_tensor_signatures_distinct"] = {
        key: len(values) == len(set(values)) and None not in values
        for key, values in tensor_digests.items()
    }
    report["fit_tensor_collision_groups"] = {
        key: _collision_groups(values) for key, values in tensor_digests.items()
    }
    report["worker_parameter_delta_digests"] = worker_delta_digests
    report["worker_parameter_delta_collision_groups"] = {
        worker_id: _collision_groups(values)
        for worker_id, values in worker_delta_digests.items()
    }
    report["all_fit_tensor_signatures_distinct"] = all(
        report["fit_tensor_signatures_distinct"].values()
    )
    report["can_start_r6_formal"] = False
    report["can_promote"] = False
    report["candidate_promoted"] = False
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
    report = run_feature_target_audit(
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
