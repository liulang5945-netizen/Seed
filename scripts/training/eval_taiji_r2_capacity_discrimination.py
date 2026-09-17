"""Capacity discrimination (candidate 2): does a 2x wider prototype move dev exact?

Readout diagnosis and three loss-side rejections localized the gap to
"content extraction -> byte generation".  Before an architectural probe (candidate 1),
this cheap discriminator checks whether the current geometry is even large enough to
express the extraction.  Same corpus, same budget (30 x 25), same seeds; the only
change is geometry (prefix_width 48->96, renderer_width 64->128, slot_width 48->96).
Control readouts come from the existing H-FBW/H-OBJ/H-GEN runs (three-way reproduced).
"""

from __future__ import annotations

import argparse
import json
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

FIXTURE = Path("tests/fixtures/r2_h3_8_joint_sequence_v1.jsonl")
SEEDS = (20260917, 20260918, 20260919)
EPOCHS = 30
MAX_EPISODES = 25
LEARNING_RATE = 0.01
GENERATION_LIMIT = 64
DEFAULT_REPORT = Path("reports/r2_capacity_discrimination_20260917.json")
BOUNDARY = int(SequenceWorkspaceConfig().boundary_symbol)

# Frozen geometry (2x the defaults on every width axis; slots kept at 4).
GEOMETRY = {"prefix_width": 96, "renderer_width": 128, "slot_width": 96}


def _load_split(split: str) -> tuple[tuple[bytes, bytes], ...]:
    episodes: list[tuple[bytes, bytes]] = []
    for line in (PROJECT_ROOT / FIXTURE).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["split"] == split:
            episodes.append((record["prefix"].encode("utf-8"), record["response"].encode("utf-8")))
    if not episodes:
        raise RuntimeError(f"{split} split is empty")
    return tuple(sorted(episodes))


def _run_wide(seed: int, report_dir: Path) -> dict[str, Any]:
    torch.manual_seed(seed)
    config = SequenceWorkspaceConfig(seed=seed, **GEOMETRY)
    prototype = SequenceWorkspacePrototype(config)
    trainer = SequenceWorkspaceTrainer(prototype, learning_rate=LEARNING_RATE)
    train_split = _load_split("train")
    trainer.set_episodes(train_split)
    zero_digest = trainer.checkpoint()["checkpoint_digest"]

    import time

    started = time.monotonic()
    for _ in range(EPOCHS):
        trainer.train_epoch(max_episodes=MAX_EPISODES)
    elapsed = time.monotonic() - started

    def _forced(episodes: tuple[tuple[bytes, bytes], ...]) -> dict[str, Any]:
        positions = 0
        correct = 0
        losses: list[float] = []
        with torch.no_grad():
            for prefix, response in episodes:
                loss, metrics = prototype.sequence_loss(prefix, response)
                positions += int(metrics["positions"])
                correct += int(metrics["correct"])
                losses.append(float(loss))
        return {
            "accuracy": correct / max(1, positions),
            "mean_loss": sum(losses) / max(1, len(losses)),
        }

    dev_split = _load_split("dev")
    exact = 0
    generated_mode = 0
    samples: list[dict[str, Any]] = []
    with torch.no_grad():
        for prefix, response in dev_split:
            result = prototype.generate(prefix, max_bytes=GENERATION_LIMIT)
            produced = bytes(result.bytes_out)
            hit = produced == response
            exact += int(hit)
            generated_mode += int(produced.decode("utf-8", errors="replace") == "蓝")
            if len(samples) < 3:
                samples.append(
                    {
                        "reference": response.decode("utf-8", errors="replace"),
                        "generated": produced.decode("utf-8", errors="replace"),
                        "exact": hit,
                    }
                )

    checkpoint_path = report_dir / f"wide_{seed}.pt"
    report_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(checkpoint_path)
    restored = SequenceWorkspaceTrainer.from_checkpoint(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    )
    return {
        "seed": seed,
        "geometry": GEOMETRY,
        "parameter_count": (
            int(trainer.checkpoint()["parameter_count"])
            if "parameter_count" in trainer.checkpoint()
            else None
        ),
        "zero_step_checkpoint_digest": zero_digest,
        "checkpoint_digest": trainer.checkpoint()["checkpoint_digest"],
        "restore_matches": restored.checkpoint()["checkpoint_digest"]
        == trainer.checkpoint()["checkpoint_digest"],
        "elapsed_seconds": elapsed,
        "train": _forced(train_split),
        "dev": _forced(dev_split),
        "dev_exact_rate": exact / max(1, len(dev_split)),
        "dev_generated_mode_rate": generated_mode / max(1, len(dev_split)),
        "samples": samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capacity discrimination (2x widths)")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/r2_capacity_wide"))
    args = parser.parse_args(argv)

    report_dir = PROJECT_ROOT / args.output_dir
    results = [_run_wide(seed, report_dir) for seed in SEEDS]
    rates = [item["dev_exact_rate"] for item in results]
    mean_rate = sum(rates) / max(1, len(rates))
    train_accs = [item["train"]["accuracy"] for item in results]
    dev_accs = [item["dev"]["accuracy"] for item in results]

    # Reference: the standard-geometry control arms at the same budget reproduce
    # 0.25 / 0.0625 / 0.0 (mean 0.1042) three times independently.
    reference_mean = 0.10416666666666667
    if mean_rate > reference_mean:
        verdict = "capacity_supported: wide geometry beats the standard control mean"
    elif mean_rate == reference_mean:
        verdict = "capacity_no_change: identical to the standard control mean"
    else:
        verdict = "capacity_rejected: wide geometry is worse than the standard control mean"

    payload: dict[str, Any] = {
        "format": "taiji-r2-capacity-discrimination-v1",
        "purpose": "candidate 2 of the post-H-FBW decision (geometry capacity discriminator)",
        "reference_control_mean_dev_exact": reference_mean,
        "reference_control_source": "H-FBW/H-OBJ/H-GEN control arms (three-way reproduced)",
        "frozen": {
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "max_episodes": MAX_EPISODES,
            **GEOMETRY,
        },
        "wide": {
            "train_accuracy": train_accs,
            "dev_forced_accuracy": dev_accs,
            "dev_exact_per_seed": rates,
            "dev_exact_mean": mean_rate,
        },
        "verdict": verdict,
        "final_evaluated": False,
        "growth_admitted": False,
        "can_promote": False,
        "scope": "isolated sequence-workspace prototype only; must NOT be extrapolated to chat()",
    }

    target = PROJECT_ROOT / args.report
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wide  dev exact per-seed {rates}  mean {mean_rate:.4f}")
    print(f"reference (standard geometry) mean dev exact: {reference_mean:.4f}")
    print(f"verdict: {verdict}")
    print(f"report -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
