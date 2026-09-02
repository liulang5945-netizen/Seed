"""Evaluate the dedicated Taiji B5 continual-memory replay Gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.train_taiji_memory import _memory_config  # noqa: E402
from taiji import (  # noqa: E402
    ContinualMemoryCorpus,
    ContinualMemoryTask,
    DelayedMemoryQuery,
    MemoryEpisode,
)
from taiji.foundation_tasks import (  # noqa: E402
    CONTINUAL_MEMORY_FORMAT,
    CONTINUAL_MEMORY_VERSION,
)

DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_b5_memory_canary_20260902.json"


def _queries(
    prefix: str,
    episodes: tuple[MemoryEpisode, ...],
    *,
    count: int,
    offset: int,
) -> tuple[DelayedMemoryQuery, ...]:
    return tuple(
        DelayedMemoryQuery(
            query_id=f"{prefix}-{index}",
            cue=episode.cue,
            expected_action=episode.action,
        )
        for index in range(int(count))
        for episode in (episodes[(index + offset) % len(episodes)],)
    )


def build_corpus(
    *,
    train_count: int,
    holdout_count: int,
    retention_count: int,
) -> ContinualMemoryCorpus:
    if min(int(train_count), int(holdout_count), int(retention_count)) < 4:
        raise ValueError("B5 memory corpus counts must be at least four")
    phase_a_train = tuple(
        MemoryEpisode(
            memory_id=f"m1-b5-phase-a-{index}",
            cue=65 + index % 80,
            action=48 + index % 2,
            outcome=43 if index % 2 == 0 else 45,
        )
        for index in range(int(train_count))
    )
    phase_b_train = tuple(
        MemoryEpisode(
            memory_id=f"m1-b5-phase-b-{index}",
            # Phase-B cues are disjoint from phase A.  Continual learning
            # should measure interference from shared capacity, not ask one
            # cue-only query to have two contradictory answers.
            cue=145 + index % 80,
            action=50 + index % 2,
            outcome=43 + index % 2 * 2,
        )
        for index in range(int(train_count))
    )
    return ContinualMemoryCorpus(
        phase_a_train=phase_a_train,
        phase_a_holdout=_queries(
            "m1-b5-phase-a-holdout",
            phase_a_train,
            count=holdout_count,
            offset=0,
        ),
        phase_a_retention=_queries(
            "m1-b5-phase-a-retention",
            phase_a_train,
            count=retention_count,
            offset=holdout_count,
        ),
        phase_b_train=phase_b_train,
        phase_b_holdout=_queries(
            "m1-b5-phase-b-holdout",
            phase_b_train,
            count=holdout_count,
            offset=int(train_count) // 2,
        ),
        phase_b_retention=_queries(
            "m1-b5-phase-b-retention",
            phase_b_train,
            count=retention_count,
            offset=int(train_count) // 2 + holdout_count,
        ),
        replay_train=phase_a_train,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("canary", "foundation"), default="canary")
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--holdout-count", type=int)
    parser.add_argument("--retention-count", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
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
    measurement = ContinualMemoryTask(
        _memory_config(int(args.seeds[0])),
        seeds=tuple(args.seeds),
    ).evaluate(corpus)
    result = {
        "format": CONTINUAL_MEMORY_FORMAT,
        "version": CONTINUAL_MEMORY_VERSION,
        "profile": args.profile,
        "corpus_digest": corpus.digest,
        "measurement": measurement.to_payload(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result["report_path"] = str(args.report)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if measurement.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
