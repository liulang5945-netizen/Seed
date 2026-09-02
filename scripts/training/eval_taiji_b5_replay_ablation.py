"""Compare memory replay write targets and action decoder locality on B5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_b5_memory import build_corpus  # noqa: E402
from taiji import TaijiConfig  # noqa: E402
from taiji.foundation_tasks import ContinualMemoryTask  # noqa: E402

FORMAT = "taiji-native-b5-replay-ablation-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_b10_replay_ablation.json"
DECODERS = ("shared", "local", "cue_selective")
LEARNING_TARGETS = ("all", "association", "readout")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("canary", "foundation"), default="canary")
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--holdout-count", type=int)
    parser.add_argument("--retention-count", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--replay-scale", type=float, default=0.5)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    defaults = {
        "canary": (16, 8, 8),
        "foundation": (1_000, 200, 200),
    }
    default_train, default_holdout, default_retention = defaults[args.profile]
    corpus = build_corpus(
        train_count=args.train_count or default_train,
        holdout_count=args.holdout_count or default_holdout,
        retention_count=args.retention_count or default_retention,
    )
    records: list[dict[str, object]] = []
    for decoder in DECODERS:
        for targets in LEARNING_TARGETS:
            config = TaijiConfig(
                memory_action_decoder=decoder,
                memory_confidence_decay=0.0,
                replay_memory_learning_scale=args.replay_scale,
            )
            measurement = ContinualMemoryTask(
                config,
                seeds=tuple(args.seeds),
                replay_learning_targets=targets,
            ).evaluate(corpus)
            records.append(
                {
                    "decoder": decoder,
                    "replay_learning_targets": targets,
                    "measurement": measurement.to_payload(),
                }
            )

    passing_candidates = [
        {
            "decoder": str(record["decoder"]),
            "replay_learning_targets": str(record["replay_learning_targets"]),
        }
        for record in records
        if record["measurement"]["status"] == "passed"  # type: ignore[index]
    ]
    result = {
        "format": FORMAT,
        "version": 1,
        "status": "blocked",
        "can_promote": False,
        "profile": args.profile,
        "replay_scale": args.replay_scale,
        "corpus_digest": corpus.digest,
        "corpus_sample_counts": corpus.sample_counts,
        "candidates_passing_all_b5_checks": passing_candidates,
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result["report_path"] = str(args.report)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
