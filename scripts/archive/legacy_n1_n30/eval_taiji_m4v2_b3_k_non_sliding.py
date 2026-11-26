"""Run the non-sliding three-example K course-combination diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m4v2_b3_k_loss_diagnostic import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
)
from scripts.training.eval_taiji_m4v2_b3_k_loss_stability import (  # noqa: E402
    run_stability,
)

REPORT_FORMAT = "taiji-m4v2-b3-k-non-sliding-v1"
VERSION = 1
TRAIN_EPISODE_COUNT = 3
TRAIN_VARIANT_STRATEGY = "non_sliding"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m4v2_b3_k_non_sliding_20260910.json"
DEFAULT_CANDIDATE_ROOT = PROJECT_ROOT / "output" / "taiji_m4v2_b3_k_non_sliding_20260910"


def run_non_sliding(
    *,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
    model_seed: int = 17,
) -> dict[str, object]:
    report = run_stability(
        artifact_dir=artifact_dir,
        candidate_root=candidate_root,
        model_seed=model_seed,
        train_episode_count=TRAIN_EPISODE_COUNT,
        train_variant_strategy=TRAIN_VARIANT_STRATEGY,
    )
    report["report_format"] = REPORT_FORMAT
    report["version"] = VERSION
    report["run_kind"] = "non-sliding-course-combination-diagnostic"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--model-seed", type=int, default=17)
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
    report = run_non_sliding(
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
