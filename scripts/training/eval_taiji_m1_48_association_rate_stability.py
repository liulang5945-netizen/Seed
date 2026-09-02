"""Repeat the M1-47 association-rate Gate on an independent seed cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m1_47_association_rate import run_diagnosis  # noqa: E402

FORMAT = "taiji-native-m1-48-association-rate-stability-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_48_association_rate_stability_20260902.json"
SEEDS = (73, 101, 131)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_diagnosis(seeds=SEEDS)
    result["format"] = FORMAT
    result["stability_review"] = {
        "independent_seed_cohort": list(SEEDS),
        "first_cohort_excluded": [11, 29, 47],
        "candidate_is_default": False,
    }
    result["report_path"] = str(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
