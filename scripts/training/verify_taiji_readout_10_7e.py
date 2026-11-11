"""10.7e: train a 1,024-parameter readout over the subject's own goal and action features.

The subject is frozen. Two arms, same budget: one trained on the real labels, one on
the swapped labels (the adversarial constant control). Only the readout moves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import scripts.training.verify_taiji_subject_selection_10_4 as prior  # noqa: E402
from taiji.goal_action_readout import BilinearReadout  # noqa: E402
from taiji.goal_conditioned_training import sample_loss, samples  # noqa: E402

TRAIN_STATES = 3
STEPS = 300
LR = 0.05
SEED = 0
BUDGET_CPU_SECONDS = 120
BUDGET_WALL_SECONDS = 240
PASS_HITS = 5
ABSENT_HITS = 3
CONTROL_MAX_HITS = 3
OUTPUT_DIR = "output/taiji_core_10_7e"


def sentence(move) -> str:
    return f"把{move[0]}放到{move[1]}"


def features(encoder, sample) -> dict:
    with torch.no_grad():
        goal = encoder.goal_vector(sample["goal"].encode()).detach()
        correct = encoder.context(sentence(sample["correct"]).encode()).mean(dim=0).detach()
        other = encoder.context(sentence(sample["other"]).encode()).mean(dim=0).detach()
    return {
        "goal_text": sample["goal"],
        "goal": goal,
        "correct": correct,
        "other": other,
        "correct_move": list(sample["correct"]),
        "other_move": list(sample["other"]),
    }


def fit(
    rows,
    *,
    flip: bool,
    steps: int = STEPS,
    lr: float = LR,
    seed: int = SEED,
    width: int = 128,
    rank: int = 4,
):
    head = BilinearReadout(width=width, rank=rank, seed=seed)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loss_value = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for row in rows:
            correct = head.score(row["goal"], row["correct"])
            other = head.score(row["goal"], row["other"])
            margin = (other - correct) if flip else (correct - other)
            losses.append(torch.nn.functional.softplus(-margin))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
    return head, loss_value


def evaluate(head, rows) -> dict:
    hits = ties = 0
    detail = []
    with torch.no_grad():
        for row in rows:
            correct = float(head.score(row["goal"], row["correct"]))
            other = float(head.score(row["goal"], row["other"]))
            hit = correct > other
            hits += int(hit)
            ties += int(correct == other)
            detail.append(
                {
                    "goal": row["goal_text"],
                    "correct_move": row["correct_move"],
                    "correct_score": correct,
                    "other_score": other,
                    "hit": hit,
                }
            )
    return {"hits": hits, "total": len(rows), "ties": ties, "rows": detail}


def flat(parameters) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def status_from(*, budget_ok: bool, frozen: bool, control_hits: int, arm_hits: int) -> str:
    if not budget_ok:
        return "budget_stop"
    if not frozen:
        return "backbone_mutated"
    if control_hits > CONTROL_MAX_HITS:
        return "control_not_null"
    if arm_hits >= PASS_HITS:
        return "readout_binding_supported"
    if arm_hits <= ABSENT_HITS:
        return "readout_binding_absent"
    return "readout_inconclusive"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.7e report is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    parent, subject, provider = prior.build_subject()
    digest = prior.parent_digest(parent)
    encoder = subject.encoder

    rows = samples()
    per_state = len(rows) // 6
    train_rows, held_rows = rows[: TRAIN_STATES * per_state], rows[TRAIN_STATES * per_state :]
    train_features = [features(encoder, sample) for sample in train_rows]
    held_features = [features(encoder, sample) for sample in held_rows]

    before = flat(list(subject.parameters()))
    with torch.no_grad():
        losses_before = [
            float(sample_loss(subject, sample, provider).detach()) for sample in held_rows
        ]

    arm, arm_loss = fit(train_features, flip=False)
    control, control_loss = fit(train_features, flip=True)
    empty = BilinearReadout(seed=SEED)

    after = flat(list(subject.parameters()))
    with torch.no_grad():
        losses_after = [
            float(sample_loss(subject, sample, provider).detach()) for sample in held_rows
        ]
    backbone_frozen = bool(torch.equal(before, after)) and losses_before == losses_after

    arm_result = evaluate(arm, held_features)
    control_result = evaluate(control, held_features)
    empty_result = evaluate(empty, held_features)

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    status = status_from(
        budget_ok=cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
        frozen=backbone_frozen,
        control_hits=control_result["hits"],
        arm_hits=arm_result["hits"],
    )

    output_dir = ROOT / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "readout_rank4.pt"
    torch.save(
        {
            "left": arm.left.detach(),
            "right": arm.right.detach(),
            "width": arm.width,
            "rank": arm.rank,
        },
        weights_path,
    )

    report = {
        "schema": "taiji-readout-10-7e-v1",
        "step": "10.7e",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "samples": len(rows),
            "train_samples": len(train_rows),
            "held_out_samples": len(held_rows),
            "steps": STEPS,
            "lr": LR,
            "seed": SEED,
            "parameters": arm.parameter_count(),
            "pass_hits": PASS_HITS,
            "absent_hits": ABSENT_HITS,
            "control_max_hits": CONTROL_MAX_HITS,
            "backbone_frozen": "parameters and held-out sample losses bitwise identical",
        },
        "parent_model_digest": digest,
        "arm": {**arm_result, "final_loss": arm_loss},
        "control": {**control_result, "final_loss": control_loss},
        "untrained": empty_result,
        "backbone_frozen": backbone_frozen,
        "held_out_losses_before": losses_before,
        "held_out_losses_after": losses_after,
        "weights_path": str(weights_path.relative_to(ROOT)),
        "weights_published": False,
        "default_model_replaced": False,
        "model_involved": True,
        "capability_claimed": status == "readout_binding_supported",
        "training_admitted": False,
        "limits": (
            "One bounded run on a six-sample synthetic fixture. The readout sees the subject's "
            "goal and action encodings only -- not the recurrent state, not the world -- so a "
            "pass means word-level binding is learnable from these features here; it is not "
            "natural-language task capability, not P2, and authorises no further training."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/goal_action_readout.py",
                "taiji/goal_conditioned_training.py",
                "scripts/training/verify_taiji_readout_10_7e.py",
            )
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": status,
                "arm_hits": arm_result["hits"],
                "control_hits": control_result["hits"],
                "untrained_hits": empty_result["hits"],
                "untrained_ties": empty_result["ties"],
                "backbone_frozen": backbone_frozen,
                "parameters": arm.parameter_count(),
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
