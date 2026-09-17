"""Read-only H3.7B learnability and residual credit consistency analysis.

Per the H3.7B contract's sole follow-up: check train learnability (zero-step
versus post-training checkpoints evaluated on the frozen train corpus without
learning) and residual credit consistency (the four-phase mixed-credit
finite-difference tests re-executed at analysis time).  No training, no
checkpoint writes, no final or dev evaluation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from taiji import LanguageAlignmentTrainer  # noqa: E402
from taiji.language_alignment import LanguageEpisodeCorpus  # noqa: E402

DATASET = Path("tests/fixtures/r2_h3_5a_response_plan_v3.jsonl")
DIGEST = "0bc5b5540d4b622f907942f42d7b809e825932e50c6e6c505a3cd091edabd691"
ARMS = ("legacy_target", "byte_aligned")
REPORT_FORMAT = "taiji-r2-h3-7b-learnability-analysis-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("reports/r2_h3_7b_validation"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    corpus = LanguageEpisodeCorpus.from_jsonl([DATASET])
    if corpus.digest != DIGEST:
        raise RuntimeError("frozen corpus digest mismatch")

    pytest_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/taiji_native/test_response_plan_target.py",
            "-k",
            "finite_difference or post_target_update_prior",
            "-q",
            "--no-header",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
    )
    credit_tests = {
        "returncode": pytest_result.returncode,
        "summary": (
            pytest_result.stdout.strip().splitlines()[-1] if pytest_result.stdout.strip() else ""
        ),
    }

    arms: dict[str, Any] = {}
    for arm in ARMS:
        arm_dir = args.validation_dir / arm
        arm_report: dict[str, Any] = {}
        for stage in ("zero", "trained"):
            payload_path = arm_dir / ("zero.pt" if stage == "zero" else "checkpoint.pt")
            payload = torch_load(payload_path)
            trainer = LanguageAlignmentTrainer.from_checkpoint(payload, corpus)
            evaluation = trainer.evaluate("train")
            arm_report[stage] = {
                "checkpoint_digest": payload["checkpoint_digest"],
                "teacher_forced_mean_surprise": evaluation["teacher_forced_mean_surprise"],
                "teacher_forced_accuracy": evaluation["teacher_forced_accuracy"],
                "exact_response_rate": evaluation["exact_response_rate"],
                "per_episode": [
                    {
                        "episode_id": record["episode_id"],
                        "teacher_forced_mean_surprise": round(
                            float(record["teacher_forced_mean_surprise"]), 6
                        ),
                        "teacher_forced_accuracy": round(
                            float(record["teacher_forced_accuracy"]), 6
                        ),
                    }
                    for record in evaluation["records"]
                ],
            }
        zero_surprise = arm_report["zero"]["teacher_forced_mean_surprise"]
        trained_surprise = arm_report["trained"]["teacher_forced_mean_surprise"]
        zero_accuracy = arm_report["zero"]["teacher_forced_accuracy"]
        trained_accuracy = arm_report["trained"]["teacher_forced_accuracy"]
        per_episode_deltas = [
            {
                "episode_id": zero_item["episode_id"],
                "surprise_delta": round(
                    trained_item["teacher_forced_mean_surprise"]
                    - zero_item["teacher_forced_mean_surprise"],
                    6,
                ),
                "accuracy_delta": round(
                    trained_item["teacher_forced_accuracy"] - zero_item["teacher_forced_accuracy"],
                    6,
                ),
            }
            for zero_item, trained_item in zip(
                arm_report["zero"]["per_episode"],
                arm_report["trained"]["per_episode"],
                strict=True,
            )
        ]
        improving_episodes = sum(1 for item in per_episode_deltas if item["surprise_delta"] < 0)
        arm_report["learnability"] = {
            "surprise_delta": round(trained_surprise - zero_surprise, 6),
            "accuracy_delta": round(trained_accuracy - zero_accuracy, 6),
            "episodes_improved": improving_episodes,
            "episodes_total": len(per_episode_deltas),
            "train_learnable": bool(trained_surprise < zero_surprise),
        }
        arms[arm] = arm_report

    payload = {
        "format": REPORT_FORMAT,
        "version": 1,
        "corpus_digest": corpus.digest,
        "credit_consistency_tests": credit_tests,
        "arms": arms,
        "read_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                arm: {
                    "surprise_zero": round(report["zero"]["teacher_forced_mean_surprise"], 6),
                    "surprise_trained": round(report["trained"]["teacher_forced_mean_surprise"], 6),
                    "train_learnable": report["learnability"]["train_learnable"],
                    "episodes_improved": report["learnability"]["episodes_improved"],
                }
                for arm, report in arms.items()
            },
            indent=2,
        )
    )
    credit_tests_ok = credit_tests["returncode"] == 0
    return 0 if credit_tests_ok else 1


def torch_load(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    raise SystemExit(main())
