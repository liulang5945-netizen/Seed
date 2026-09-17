"""Readout diagnosis: where does the teacher-forced path diverge from generation?

Read-only.  Loads a saved prototype checkpoint and, for every dev episode, compares

* the **teacher-forced** per-position argmax (what the model predicts when fed the
  true response bytes), against
* the **generative** rollout (what it produces when fed its own outputs),

answering: at which position does the rollout first diverge, and is that position
the same one where teacher forcing first errs?  No training, no checkpoint writes.
"""

from __future__ import annotations

import argparse
import collections
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

CHECKPOINT = Path("reports/r2_hobj_matched_dev/control_checkpoint.pt")
FIXTURE = Path("tests/fixtures/r2_h3_8_joint_sequence_v1.jsonl")
DEFAULT_REPORT = Path("reports/r2_hgen_readout_diagnosis_20260917.json")
BOUNDARY = int(SequenceWorkspaceConfig().boundary_symbol)


def _load_split(split: str) -> tuple[tuple[bytes, bytes], ...]:
    episodes: list[tuple[bytes, bytes]] = []
    for line in (PROJECT_ROOT / FIXTURE).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["split"] != split:
            continue
        episodes.append((record["prefix"].encode("utf-8"), record["response"].encode("utf-8")))
    if not episodes:
        raise RuntimeError(f"{split} split is empty")
    return tuple(episodes)


def _first_divergence(a: bytes, b: bytes) -> int | None:
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def diagnose(
    prototype: SequenceWorkspacePrototype, episodes: tuple[tuple[bytes, bytes], ...]
) -> dict[str, Any]:
    tf_hit_total = 0
    tf_positions = 0
    gen_exact = 0
    tf_first_error: list[int | None] = []
    gen_first_divergence: list[int | None] = []
    per_episode: list[dict[str, Any]] = []
    with torch.no_grad():
        for prefix, response in episodes:
            logits = prototype.teacher_forced_logits(prefix, response)
            targets = [int(s) for s in response] + [BOUNDARY]
            argmaxes = [int(row.argmax(dim=0)) for row in logits]
            tf_hits = sum(int(a == t) for a, t in zip(argmaxes, targets))
            tf_positions += len(targets)
            tf_hit_total += tf_hits

            result = prototype.generate(prefix, max_bytes=64)
            produced = bytes(result.bytes_out)
            gen_div = _first_divergence(produced, response)

            tf_err = None
            for index, (a, t) in enumerate(zip(argmaxes, targets)):
                if a != t:
                    tf_err = index
                    break
            tf_first_error.append(tf_err)
            gen_first_divergence.append(gen_div)
            per_episode.append(
                {
                    "prefix_head": prefix[:18].decode("utf-8", errors="replace"),
                    "reference": response.decode("utf-8", errors="replace"),
                    "generated": produced.decode("utf-8", errors="replace"),
                    "tf_first_error": tf_err,
                    "gen_first_divergence": gen_div,
                    "tf_hit_rate": tf_hits / max(1, len(targets)),
                    "gen_exact": produced == response,
                }
            )
    gen_exact = sum(1 for item in per_episode if item["gen_exact"])

    def _dist(values: list[int | None]) -> dict[str, int]:
        counter: collections.Counter[str] = collections.Counter()
        for value in values:
            counter["never"] += int(value is None)
            if value is not None:
                counter[f"pos_{value}"] += 1
        return dict(counter)

    first_err_known = [v for v in tf_first_error if v is not None]
    first_div_known = [v for v in gen_first_divergence if v is not None]
    return {
        "episodes": len(episodes),
        "teacher_forced_hit_rate": tf_hit_total / max(1, tf_positions),
        "generation_exact_rate": gen_exact / max(1, len(episodes)),
        "tf_first_error_distribution": _dist(tf_first_error),
        "gen_first_divergence_distribution": _dist(gen_first_divergence),
        "tf_first_error_median": (
            sorted(first_err_known)[len(first_err_known) // 2] if first_err_known else None
        ),
        "gen_first_divergence_median": (
            sorted(first_div_known)[len(first_div_known) // 2] if first_div_known else None
        ),
        "per_episode": per_episode,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H-GEN readout diagnosis (read-only)")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    checkpoint_path = (
        args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trainer = SequenceWorkspaceTrainer.from_checkpoint(payload)
    prototype = trainer.prototype
    dev = _load_split("dev")

    summary = diagnose(prototype, dev)
    payload_out: dict[str, Any] = {
        "format": "taiji-r2-hgen-readout-diagnosis-v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_digest": trainer.checkpoint()["checkpoint_digest"],
        "global_step": int(trainer.global_step),
        "episodes_source": str(FIXTURE),
        **summary,
    }
    target = PROJECT_ROOT / args.report
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"checkpoint: {checkpoint_path.name}  (global_step={trainer.global_step})")
    print(f"teacher-forced hit rate : {summary['teacher_forced_hit_rate']:.4f}")
    print(f"generation exact rate   : {summary['generation_exact_rate']:.4f}")
    print(f"TF first-error   dist   : {summary['tf_first_error_distribution']}")
    print(f"gen first-divergence    : {summary['gen_first_divergence_distribution']}")
    print(f"report -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
