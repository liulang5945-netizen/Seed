"""Consolidation (i-a): nail down the 672-episode control reading with 6 seeds.

Read-only w.r.t. existing artifacts.  Trains the lambda=0 control arm at the 672
corpus scale (the scale sweep's peak) with six seeds, reports mean/std/each, and
adds a per-seed generation breakdown so the reading can be quoted safely.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from taiji.sequence_workspace import (  # noqa: E402
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

_spec = importlib.util.spec_from_file_location(
    "sweep", PROJECT_ROOT / "scripts" / "training" / "sweep_taiji_r2_course_scale.py"
)
_sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sweep)

SEEDS = (20260917, 20260918, 20260919, 20260920, 20260921, 20260922)
SCALE = 8
EPOCHS = 4
LEARNING_RATE = 0.01
DEFAULT_REPORT = Path("reports/r2_course_consolidation_20260917.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consolidate the 672-episode control reading")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    corpus = _sweep._build_corpus(SCALE)
    digest = _sweep._corpus_digest(corpus)
    episodes_per_epoch = len(corpus["train"])

    results: list[dict[str, Any]] = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        prototype = SequenceWorkspacePrototype(SequenceWorkspaceConfig(seed=seed))
        trainer = SequenceWorkspaceTrainer(prototype, learning_rate=LEARNING_RATE)
        trainer.set_episodes(corpus["train"])
        trainer.enable_epoch_shuffle(seed=20260917 + seed)
        for _ in range(EPOCHS):
            trainer.train_epoch(max_episodes=episodes_per_epoch)
        train_forced = _sweep._forced(prototype, corpus["train"])
        dev_forced = _sweep._forced(prototype, corpus["dev"])
        generation = _sweep._generation(prototype, corpus["dev"])
        results.append(
            {
                "seed": seed,
                "lambda": 0.0,
                "train_accuracy": train_forced["accuracy"],
                "dev_forced_accuracy": dev_forced["accuracy"],
                "dev_exact_rate": generation["exact_rate"],
                "dev_exact_count": generation["exact_count"],
                "dev_episodes": generation["episodes"],
                "boundary_stop_rate": generation.get("boundary_stop_rate"),
            }
        )
        print(
            f"seed {seed}: train {train_forced['accuracy']:.4f}  "
            f"dev forced {dev_forced['accuracy']:.4f}  "
            f"dev exact {generation['exact_rate']:.4f} ({generation['exact_count']}/{generation['episodes']})"
        )

    exact_rates = [item["dev_exact_rate"] for item in results]
    mean_rate = statistics.fmean(exact_rates)
    std_rate = statistics.pstdev(exact_rates) if len(exact_rates) > 1 else 0.0

    payload = {
        "format": "taiji-r2-course-consolidation-v1",
        "purpose": "(i-a) consolidate the 672-episode control reading with 6 seeds",
        "scale": SCALE,
        "epochs": EPOCHS,
        "episodes_per_epoch": episodes_per_epoch,
        "steps_per_seed": EPOCHS * episodes_per_epoch,
        "corpus_digest": digest,
        "seeds": list(SEEDS),
        "lambda": 0.0,
        "results": results,
        "dev_exact_rates": exact_rates,
        "dev_exact_mean": mean_rate,
        "dev_exact_pstdev": std_rate,
        "train_accuracy_mean": statistics.fmean(item["train_accuracy"] for item in results),
        "dev_forced_accuracy_mean": statistics.fmean(
            item["dev_forced_accuracy"] for item in results
        ),
        "final_evaluated": False,
        "growth_admitted": False,
        "can_promote": False,
        "scope": "isolated prototype only; must NOT be extrapolated to chat()",
    }
    target = PROJECT_ROOT / args.report
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print()
    print(
        f"672-episode control, 6 seeds: dev exact mean {mean_rate:.4f} "
        f"(pstdev {std_rate:.4f}), rates {exact_rates}"
    )
    print(f"report -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
