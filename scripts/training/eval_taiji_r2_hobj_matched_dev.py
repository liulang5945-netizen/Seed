"""H-OBJ matched dev: does an explicit context-contrastive term move dev ``exact``?

Candidate spec: plans/reference/M5_R2_NEXT_CANDIDATE_CONTEXT_CONTRASTIVE_20260917.md

Two arms, same corpus, same budget, same seed -- the **only** difference is the
frozen contrastive weight lambda (spec section 3.3):

* ``control``     : lambda = 0.0  (plain cross-entropy; reproduces the legacy path)
* ``treatment``   : lambda = 1.0  (adds the context-contrastive term, margin 1.0 nats)

Primary criterion (spec section 4, layer 3): **dev exact response**.  The final
split is never read and no promotion is implied by any outcome.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from taiji.internalization import content_digest  # noqa: E402
from taiji.sequence_workspace import (  # noqa: E402
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

FIXTURE = Path("tests/fixtures/r2_h3_8_joint_sequence_v1.jsonl")
DEFAULT_REPORT = Path("reports/r2_hobj_matched_dev_20260917.json")
REPORT_FORMAT = "taiji-r2-hobj-matched-dev-v1"

# ---- frozen budget (spec 4) and frozen objective (spec 3.3) ------------------
# Scaled to the H3.8 budget (30 x 25) after the budget-discrimination finding:
# at 2 x 12 no objective change could express itself at all (plans/reference/
# M5_R2_BUDGET_DISCRIMINATION_20260917.md), and 30 x 25 is where dev exact first
# becomes non-zero.  Three seeds so the verdict is not a single-seed accident.
SEEDS = (20260917, 20260918, 20260919)
EPOCHS = 30
MAX_EPISODES = 25
LAMBDA_TREATMENT = 1.0
LAMBDA_CONTROL = 0.0
CONTRASTIVE_MARGIN = 1.0
GENERATION_LIMIT = 64
WALL_CAP_SECONDS = 30 * 60.0

ARMS = ("control", "treatment")


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


def _phrase(split: str) -> tuple[tuple[bytes, bytes], ...]:
    return tuple(sorted(_load_split(split)))


def _teacher_forced(
    prototype: SequenceWorkspacePrototype, episodes: tuple[tuple[bytes, bytes], ...]
) -> dict[str, Any]:
    positions = 0
    correct = 0
    losses: list[float] = []
    with torch.no_grad():
        for prefix, response in episodes:
            loss, metrics = prototype.sequence_loss(prefix, response)
            losses.append(float(loss))
            positions += int(metrics["positions"])
            correct += int(metrics["correct"])
    return {
        "accuracy": correct / max(1, positions),
        "positions": positions,
        "mean_loss": sum(losses) / max(1, len(losses)),
    }


def _generation(
    prototype: SequenceWorkspacePrototype, episodes: tuple[tuple[bytes, bytes], ...]
) -> dict[str, Any]:
    """Primary criterion: how many dev responses are reproduced exactly."""

    exact = 0
    stopped = 0
    samples: list[dict[str, Any]] = []
    with torch.no_grad():
        for prefix, response in episodes:
            result = prototype.generate(prefix, max_bytes=GENERATION_LIMIT)
            hit = bytes(result.bytes_out) == bytes(response)
            exact += int(hit)
            stopped += int(bool(result.stopped_on_boundary))
            if len(samples) < 3:
                samples.append(
                    {
                        "reference": response.decode("utf-8", errors="replace"),
                        "generated": bytes(result.bytes_out).decode("utf-8", errors="replace"),
                        "exact": hit,
                        "boundary_stopped": bool(result.stopped_on_boundary),
                    }
                )
    total = len(episodes)
    return {
        "episodes": total,
        "exact": exact,
        "exact_rate": exact / max(1, total),
        "boundary_stop_rate": stopped / max(1, total),
        "samples": samples,
    }


def _run_arm(
    arm: str,
    *,
    seed: int,
    lam: float,
    report_dir: Path,
    epochs: int = EPOCHS,
    max_episodes: int = MAX_EPISODES,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    config = SequenceWorkspaceConfig(seed=seed)
    prototype = SequenceWorkspacePrototype(config)
    trainer = SequenceWorkspaceTrainer(prototype, learning_rate=0.01)
    train_split = _phrase("train")
    trainer.set_episodes(train_split)
    trainer.enable_context_contrastive(weight=lam, margin=CONTRASTIVE_MARGIN)
    zero_digest = trainer.checkpoint()["checkpoint_digest"]

    started = time.monotonic()
    epoch_records: list[dict[str, Any]] = []
    for epoch in range(epochs):
        record = trainer.train_epoch(max_episodes=max_episodes)
        epoch_records.append(
            {
                "epoch": epoch,
                "mean_loss": record["mean_loss"],
                "episodes": record["episodes"],
                "mean_contrastive_margin_gap": record["mean_contrastive_margin_gap"],
            }
        )
    elapsed = time.monotonic() - started

    train_metrics = _teacher_forced(prototype, train_split)
    dev_split = _phrase("dev")
    dev_metrics = _teacher_forced(prototype, dev_split)
    dev_generation = _generation(prototype, dev_split)

    checkpoint_path = report_dir / f"{arm}_checkpoint.pt"
    report_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(checkpoint_path)
    # Fresh-process restore check: the saved payload must round-trip.
    restored = SequenceWorkspaceTrainer.from_checkpoint(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    )
    restore_ok = (
        restored.checkpoint()["checkpoint_digest"] == trainer.checkpoint()["checkpoint_digest"]
    )

    return {
        "arm": arm,
        "seed": seed,
        "lambda": lam,
        "epochs": epochs,
        "max_episodes": max_episodes,
        "train_episodes_available": len(train_split),
        "zero_step_checkpoint_digest": zero_digest,
        "checkpoint_digest": trainer.checkpoint()["checkpoint_digest"],
        "restore_matches": restore_ok,
        "elapsed_seconds": elapsed,
        "epoch_records": epoch_records,
        "train": train_metrics,
        "dev": dev_metrics,
        "dev_generation": dev_generation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H-OBJ matched dev (two arms)")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=FIXTURE,
        help="joint-sequence corpus (default: the H3.8 v1 fixture)",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-episodes", type=int, default=MAX_EPISODES)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/r2_hobj_matched_dev"))
    args = parser.parse_args(argv)

    report_dir = PROJECT_ROOT / args.output_dir
    fixture_path = PROJECT_ROOT / args.fixture
    corpus_digest = __import__("hashlib").sha256(fixture_path.read_bytes()).hexdigest()

    arms = [
        _run_arm(
            arm,
            seed=seed,
            lam=lam,
            report_dir=report_dir,
            epochs=args.epochs,
            max_episodes=args.max_episodes,
        )
        for seed in SEEDS
        for arm, lam in (("control", LAMBDA_CONTROL), ("treatment", LAMBDA_TREATMENT))
    ]

    control_rates = [a["dev_generation"]["exact_rate"] for a in arms if a["arm"] == "control"]
    treatment_rates = [a["dev_generation"]["exact_rate"] for a in arms if a["arm"] == "treatment"]
    control_exact = sum(control_rates) / max(1, len(control_rates))
    treatment_exact = sum(treatment_rates) / max(1, len(treatment_rates))

    # Frozen stop lines (spec section 4).
    both_zero = control_exact == 0.0 and treatment_exact == 0.0
    if both_zero:
        verdict = "h_obj_rejected: both arms have dev exact = 0"
    elif treatment_exact <= control_exact:
        verdict = "h_obj_not_supported: treatment dev exact is not above control"
    else:
        verdict = "h_obj_supported: treatment dev exact is above control"

    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "candidate_spec": "plans/reference/M5_R2_NEXT_CANDIDATE_GENERATION_STATE_TRAINING_20260917.md",
        "implementability": "plans/reference/M5_R2_CANDIDATE_HGEN_IMPLEMENTABILITY_20260917.md",
        "code_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
        ).stdout.strip(),
        "fixture": str(FIXTURE),
        "corpus_sha256": corpus_digest,
        "frozen": {
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "max_episodes": MAX_EPISODES,
            "lambda_control": LAMBDA_CONTROL,
            "lambda_treatment": LAMBDA_TREATMENT,
            "contrastive_margin": CONTRASTIVE_MARGIN,
            "wall_cap_seconds": WALL_CAP_SECONDS,
        },
        "arms": arms,
        "dev_exact_per_seed": {"control": control_rates, "treatment": treatment_rates},
        "dev_exact_mean": {"control": control_exact, "treatment": treatment_exact},
        "verdict": verdict,
        "final_evaluated": False,
        "growth_admitted": False,
        "can_promote": False,
        "scope": (
            "isolated sequence-workspace prototype only; the result must NOT be extrapolated "
            "to the chat() path (H-OBJ spec section 3, residual risk 1)"
        ),
    }
    payload["report_digest"] = content_digest(
        {key: value for key, value in payload.items() if key != "report_digest"}
    )

    target = PROJECT_ROOT / args.report
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"control   dev exact per-seed {control_rates}  mean {control_exact:.4f}")
    print(f"treatment dev exact per-seed {treatment_rates}  mean {treatment_exact:.4f}")
    print(f"verdict: {verdict}")
    print(f"report -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
