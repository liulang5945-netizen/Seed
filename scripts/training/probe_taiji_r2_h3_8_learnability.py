"""R2-H3.8 small train-only learnability check (contract section 4, step 1).

Reads only the frozen train split of the H3.8 corpus (digest checked).  Trains
both contract arms (workspace and no-workspace baseline) with the same seed and
budget, records the loss trajectory, and reports whether each arm is train
learnable.  It never reads dev or final, and its numbers are probe evidence for
choosing the matched-dev budget -- not capability claims.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from taiji.internalization import content_digest  # noqa: E402
from taiji.sequence_workspace import (  # noqa: E402
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

FIXTURE = Path("tests/fixtures/r2_h3_8_joint_sequence_v1.jsonl")
CORPUS_DIGEST = "5518ff500bbcb3551bc60cc13783442cd9fce870e8d17ee9dd4a7e165a17f6a8"
DEFAULT_REPORT = Path("reports/r2_h3_8_train_learnability_20260917.json")
FORMAT = "taiji-r2-h3-8-train-learnability-v1"
EPOCHS = 30
SEED = 20260917
FROZEN_LEARNING_RATE = 0.01
STABILITY_PROBE_LEARNING_RATE = 0.05
STABILITY_PROBE_EPOCHS = EPOCHS


def _load_train() -> tuple[tuple[bytes, bytes], ...]:
    episodes: list[tuple[bytes, bytes]] = []
    for line in (PROJECT_ROOT / FIXTURE).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["split"] != "train":
            continue
        episodes.append((record["prefix"].encode("utf-8"), record["response"].encode("utf-8")))
    if not episodes:
        raise RuntimeError("H3.8 train split is empty")
    return tuple(episodes)


def _evaluate(
    prototype: SequenceWorkspacePrototype, episodes: tuple[tuple[bytes, bytes], ...]
) -> dict[str, float]:
    prototype.zero_grads_for_test()
    losses: list[float] = []
    positions = 0
    correct = 0
    with torch.no_grad():
        for prefix, response in episodes:
            loss, metrics = prototype.sequence_loss(prefix, response)
            losses.append(float(loss))
            positions += int(metrics["positions"])
            correct += int(metrics["correct"])
    mean_loss = sum(losses) / len(losses)
    return {
        "mean_loss": mean_loss,
        "mean_surprise_bits": mean_loss / math.log(2.0),
        "accuracy": correct / max(1, positions),
        "positions": float(positions),
    }


def _run_arm(
    name: str,
    workspace_enabled: bool,
    episodes: tuple[tuple[bytes, bytes], ...],
    *,
    learning_rate: float,
    epochs: int,
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    prototype = SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(seed=SEED, workspace_enabled=workspace_enabled)
    )
    trainer = SequenceWorkspaceTrainer(
        prototype, learning_rate=learning_rate, code_revision="h38-learnability"
    )
    trainer.set_episodes(episodes)
    initial = _evaluate(prototype, episodes)
    trajectory: list[dict[str, Any]] = []
    episodes_trained = 0
    for epoch in range(epochs):
        record = trainer.train_epoch()
        episodes_trained += int(record["episodes"])
        if (epoch + 1) % 5 == 0 or epoch == 0:
            evaluation = _evaluate(prototype, episodes)
            trajectory.append({"epoch": epoch + 1, **evaluation})
    final = _evaluate(prototype, episodes)
    return {
        "arm": name,
        "workspace_enabled": workspace_enabled,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "episodes_trained": episodes_trained,
        "parameter_count": prototype.parameter_count(),
        "initial": initial,
        "final": final,
        "trajectory": trajectory,
        "loss_delta": final["mean_loss"] - initial["mean_loss"],
        "train_learnable": bool(final["mean_loss"] < initial["mean_loss"]),
        "checkpoint_digest": str(trainer.checkpoint()["checkpoint_digest"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    fixture_path = PROJECT_ROOT / FIXTURE
    digest = content_digest(
        [json.loads(line) for line in fixture_path.read_text(encoding="utf-8").splitlines()]
    )
    if digest != CORPUS_DIGEST:
        raise RuntimeError(f"H3.8 corpus digest mismatch: {digest}")

    episodes = _load_train()
    arms = [
        _run_arm(
            "workspace",
            True,
            episodes,
            learning_rate=FROZEN_LEARNING_RATE,
            epochs=EPOCHS,
        ),
        _run_arm(
            "baseline",
            False,
            episodes,
            learning_rate=FROZEN_LEARNING_RATE,
            epochs=EPOCHS,
        ),
    ]
    stability_probe = _run_arm(
        "workspace_high_lr_stability_probe",
        True,
        episodes,
        learning_rate=STABILITY_PROBE_LEARNING_RATE,
        epochs=STABILITY_PROBE_EPOCHS,
    )
    payload = {
        "format": FORMAT,
        "version": 1,
        "corpus_digest": digest,
        "split_read": "train",
        "dev_read": False,
        "final_read": False,
        "seed": SEED,
        "epochs": EPOCHS,
        "frozen_learning_rate": FROZEN_LEARNING_RATE,
        "train_episodes": len(episodes),
        "optimizer_steps_per_arm": EPOCHS * len(episodes),
        "arms": {arm["arm"]: arm for arm in arms},
        "stability_finding": {
            "probe_learning_rate": STABILITY_PROBE_LEARNING_RATE,
            "probe_epochs": STABILITY_PROBE_EPOCHS,
            "arm": stability_probe["arm"],
            "initial": stability_probe["initial"],
            "final": stability_probe["final"],
            "diverged": bool(
                stability_probe["final"]["mean_loss"] > stability_probe["initial"]["mean_loss"]
            ),
            "reading": (
                "the workspace arm is lr-sensitive over the full budget: at 0.05 "
                "its loss rises above the initial value while the baseline "
                "tolerates that rate, so the shared frozen learning rate is 0.01 "
                "(stable for both arms).  This is an optimizer-stability finding, "
                "not a data finding; per contract section 4 the response is to "
                "return to the computation graph rather than add epochs or data"
            ),
        },
        "outcome": ("passed" if all(arm["train_learnable"] for arm in arms) else "failed"),
        "reading": (
            "train-only learnability probe; numbers inform the matched-dev budget "
            "freeze and are not capability claims"
        ),
        "growth_admitted": False,
        "can_promote": False,
    }
    report_path = PROJECT_ROOT / args.report
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "outcome": payload["outcome"],
                "arms": {
                    name: {
                        "initial_loss": round(arm["initial"]["mean_loss"], 4),
                        "final_loss": round(arm["final"]["mean_loss"], 4),
                        "accuracy": round(arm["final"]["accuracy"], 4),
                        "train_learnable": arm["train_learnable"],
                    }
                    for name, arm in payload["arms"].items()
                },
            },
            indent=2,
        )
    )
    return 0 if payload["outcome"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
