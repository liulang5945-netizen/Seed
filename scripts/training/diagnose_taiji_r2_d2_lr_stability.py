"""R2-D2 implementation-debugging diagnostic: is the v3 graph trainable at the
H3.8-frozen stable learning rate 0.01?

Contract section 6 route ("probe fails -> inspect graph/optimization; do not
add epochs, data, or change lr in the gate itself").  The frozen probe used
the trainer-default 0.05 and showed the same instability the H3.8 learnability
report documented for the workspace graph at that rate (loss to 2.09 then
oscillation).  This script is explicitly **diagnostic, not a gate run**: it
records initial gradient norms and a 30-epoch trajectory at 0.01.  Its output
never overwrites the frozen probe report.
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
    EVIDENCE_BROADCAST_FINAL,
    EVIDENCE_FINAL_STATE_SLOTS,
    EVIDENCE_PER_POSITION,
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

FIXTURE = Path("tests/fixtures/r2_d1_measurement_v1.jsonl")
SEED = 20260917
EPOCHS = 30
ARMS = (
    ("A_per_position", EVIDENCE_PER_POSITION),
    ("C1_broadcast_final", EVIDENCE_BROADCAST_FINAL),
    ("C0_final_state_slots", EVIDENCE_FINAL_STATE_SLOTS),
)


def _load_train() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (PROJECT_ROOT / FIXTURE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row for row in rows if row["split"] == "train"]


def _run_arm(
    arm_name: str,
    evidence_source: str,
    episodes: tuple[tuple[bytes, bytes], ...],
    rows: list[dict[str, Any]],
    learning_rate: float,
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    prototype = SequenceWorkspacePrototype(
        SequenceWorkspaceConfig(seed=SEED, evidence_source=evidence_source)
    )
    prototype.zero_grads_for_test()
    first_loss, _ = prototype.sequence_loss(*episodes[0])
    first_loss.backward()
    grad_norms = {
        name: float(parameter.grad.norm())
        for name, parameter in prototype.named_parameters()
        if parameter.grad is not None
    }
    trainer = SequenceWorkspaceTrainer(
        prototype, learning_rate=learning_rate, code_revision="r2d2-diagnostic"
    )
    trainer.set_episodes(episodes)
    trajectory: list[dict[str, Any]] = []
    for epoch in range(EPOCHS):
        trainer.train_epoch()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            losses: list[float] = []
            byte_correct = 0
            byte_positions = 0
            shape_hits: dict[str, int] = {}
            shape_total: dict[str, int] = {}
            samples: list[dict[str, str]] = []
            with torch.no_grad():
                for index, row in enumerate(rows):
                    prefix = row["prefix"].encode("utf-8")
                    result = prototype.generate(prefix)
                    generated = result.bytes_out.decode("utf-8", errors="replace")
                    episode_loss, metrics = prototype.sequence_loss(
                        prefix, row["response"].encode("utf-8")
                    )
                    losses.append(float(episode_loss))
                    byte_correct += int(metrics["correct"])
                    byte_positions += int(metrics["positions"])
                    shape = row["shape"]
                    shape_total[shape] = shape_total.get(shape, 0) + 1
                    if generated == row["response"]:
                        shape_hits[shape] = shape_hits.get(shape, 0) + 1
                    if index < 6:
                        samples.append({"gold": row["response"], "generated": generated})
            trajectory.append(
                {
                    "epoch": epoch + 1,
                    "mean_loss": sum(losses) / len(losses),
                    "byte_accuracy": byte_correct / max(1, byte_positions),
                    "shape_exact": {
                        shape: shape_hits.get(shape, 0) / shape_total[shape]
                        for shape in sorted(shape_total)
                    },
                }
            )
    return {
        "arm": arm_name,
        "parameter_count": prototype.parameter_count(),
        "initial_gradient_norms": grad_norms,
        "initial_loss": float(first_loss.detach()),
        "trajectory": trajectory,
        "samples_after_30": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/r2_d2_lr_stability_diagnostic_20260918.json"),
    )
    parser.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.parse_args()

    rows = _load_train()
    episodes = tuple(
        (row["prefix"].encode("utf-8"), row["response"].encode("utf-8")) for row in rows
    )
    arm_reports = [
        _run_arm(name, source, episodes, rows, args.learning_rate) for name, source in ARMS
    ]
    payload = {
        "format": "taiji-r2-d2-lr-stability-diagnostic-v1",
        "diagnostic_not_gate": True,
        "learning_rate": args.learning_rate,
        "seed": SEED,
        "epochs": EPOCHS,
        "train_episodes": len(rows),
        "arms": {report["arm"]: report for report in arm_reports},
        "growth_admitted": False,
        "can_promote": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        report["arm"]: {
            "final_loss": round(report["trajectory"][-1]["mean_loss"], 4),
            "final_byte_acc": round(report["trajectory"][-1]["byte_accuracy"], 4),
            "shape_exact": {
                shape: round(value, 2)
                for shape, value in report["trajectory"][-1]["shape_exact"].items()
            },
        }
        for report in arm_reports
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
