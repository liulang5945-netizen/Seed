"""Run the bounded two-example K continuation diagnostic.

This is the next B3-K experiment after the one-example course-sensitivity
result.  It changes only the number of real train experiences per course;
the parent, worker ownership, learning rates, fixed holdout, and rollback
contract remain the same.  It is diagnostic evidence, never a promotion run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_b3_k_loss_diagnostic import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
)
from scripts.training.eval_taiji_m4v2_b3_k_loss_stability import (  # noqa: E402
    COURSE_SEEDS,
    run_stability,
)

REPORT_FORMAT = "taiji-m4v2-b3-k-bounded-batch-v1"
VERSION = 1
TRAIN_EPISODE_COUNT = 2
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_bounded_batch_20260910.json"
DEFAULT_CANDIDATE_ROOT = PROJECT_ROOT / "output" / "taiji_m4v2_b3_k_bounded_batch_20260910"


def run_bounded_batch(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
    model_seed: int = 17,
    course_seeds: tuple[int, ...] = COURSE_SEEDS,
    train_episode_count: int = TRAIN_EPISODE_COUNT,
) -> dict[str, Any]:
    report = run_stability(
        artifact_dir=artifact_dir,
        candidate_root=candidate_root,
        model_seed=model_seed,
        course_seeds=course_seeds,
        train_episode_count=train_episode_count,
    )
    report["report_format"] = REPORT_FORMAT
    report["version"] = VERSION
    report["run_kind"] = "bounded-multi-example-diagnostic"
    report["bounded_batch_count"] = int(train_episode_count)
    report["can_start_r6_formal"] = False
    report["can_promote"] = False
    report["candidate_promoted"] = False
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--model-seed", type=int, default=17)
    parser.add_argument("--train-episode-count", type=int, default=TRAIN_EPISODE_COUNT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    artifact_dir = (
        args.artifact_dir if args.artifact_dir.is_absolute() else PROJECT_ROOT / args.artifact_dir
    )
    candidate_root = (
        args.candidate_root
        if args.candidate_root.is_absolute()
        else PROJECT_ROOT / args.candidate_root
    )
    report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report = run_bounded_batch(
        artifact_dir=artifact_dir,
        candidate_root=candidate_root,
        model_seed=args.model_seed,
        train_episode_count=args.train_episode_count,
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
