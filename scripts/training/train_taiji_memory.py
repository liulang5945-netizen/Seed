"""Run the native Taiji delayed-memory pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    DelayedMemoryCorpus,
    DelayedMemoryQuery,
    MemoryEpisode,
    MemoryTrainingRun,
    Taiji,
    TaijiConfig,
)


def _memory_config(seed: int) -> TaijiConfig:
    values = TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        memory_units=128,
        memory_fan_in=32,
        memory_meta_dim=32,
        memory_readout_fan_in=32,
        memory_iterations=3,
    ).to_dict()
    values["seed"] = int(seed)
    # This is the pre-organ memory substrate baseline.  Every historical figure
    # measured through it (B5 onward) was recorded with no identity organ, and
    # the diagnostics built on it exist to isolate one mechanism's marginal
    # contribution.  Once M1-63 flipped ``identity_organ_enabled`` to True by
    # default, inheriting that default let organ recall supply the binding those
    # diagnostics were supposed to attribute to the mechanism under test, which
    # made several "diagnostic only, nothing promotable" arms look promotable.
    # Pinning it False keeps the baseline comparable across the flip; evaluators
    # that actually study the organ override this key explicitly per arm.
    values["identity_organ_enabled"] = False
    return TaijiConfig.from_dict(values)


def build_corpus(*, count: int) -> DelayedMemoryCorpus:
    if int(count) < 4:
        raise ValueError("F2 memory corpus needs at least four episodes")
    train = tuple(
        MemoryEpisode(
            memory_id=f"m1-f2-train-{index}",
            # Keep the synthetic cue vocabulary inside the byte sensor while
            # allowing foundation-scale corpora to contain more episodes.
            # The even period preserves the action parity for repeated cues.
            cue=65 + index % 190,
            action=48 + index % 2,
            outcome=43 if index % 2 == 0 else 45,
        )
        for index in range(int(count))
    )
    return DelayedMemoryCorpus(
        train=train,
        holdout=tuple(
            DelayedMemoryQuery(
                query_id=f"m1-f2-holdout-{index}",
                cue=episode.cue,
                expected_action=episode.action,
            )
            for index, episode in enumerate(train)
        ),
        retention=tuple(
            DelayedMemoryQuery(
                query_id=f"m1-f2-retention-{index}",
                cue=episode.cue,
                expected_action=episode.action,
            )
            for index, episode in enumerate(train)
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "taiji-m1-f2-memory",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    count = args.count if args.count is not None else (8 if args.profile == "smoke" else 64)
    corpus = build_corpus(count=count)
    if args.resume is not None:
        run = MemoryTrainingRun.from_checkpoint(
            args.resume,
            corpus,
            output_dir=args.output_dir,
            epochs=args.epochs,
        )
    else:
        run = MemoryTrainingRun(
            Taiji(_memory_config(args.seed), episode_id="memory-train"),
            corpus,
            output_dir=args.output_dir,
            model_tier="memory",
            epochs=args.epochs,
            checkpoint_interval=args.checkpoint_interval,
        )
    result: dict[str, Any]
    if args.eval_only:
        result = run.evaluate_only()
    else:
        result = run.run()
    report_path = args.report or args.output_dir / (
        "eval_report.json" if args.eval_only else "training_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result["report_path"] = str(report_path)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
