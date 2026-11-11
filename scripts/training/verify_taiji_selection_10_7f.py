"""10.7f: train a 1,536-parameter readout to pick one of six actions. Subject frozen.

10.7e's head only ever compared the correct move with a single distractor. Here the
training target is the real one: cross-entropy over all six (object, place) pairs, with
the goal **and** the state feeding the readout. A mismatched-goal arm is the control.
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
from torch.nn import functional as F  # noqa: E402

import scripts.training.verify_taiji_subject_selection_10_4 as prior  # noqa: E402
from taiji.action_verdict_scorer import state_text  # noqa: E402
from taiji.goal_conditioned_training import sample_loss, samples  # noqa: E402
from taiji.goal_state_readout import StateGoalActionReadout  # noqa: E402
from taiji.native_goal_world import OBJECTS, PLACES  # noqa: E402

TRAIN_STATES = 3
STEPS = 300
LR = 0.05
SEED = 0
BUDGET_CPU_SECONDS = 300
BUDGET_WALL_SECONDS = 600
PASS_TOP1 = 5
ABSENT_TOP1 = 2
CONTROL_MAX_TOP1 = 2
OUTPUT_DIR = "output/taiji_core_10_7f"


def sentence(move) -> str:
    return f"把{move[0]}放到{move[1]}"


def all_moves() -> list[tuple[str, str]]:
    return [(obj, place) for obj in OBJECTS for place in PLACES]


def features(encoder, sample) -> dict:
    moves = all_moves()
    with torch.no_grad():
        goal = encoder.goal_vector(sample["goal"].encode()).detach()
        state = encoder.context(state_text(sample["state"]).encode()).mean(dim=0).detach()
        actions = [encoder.context(sentence(move).encode()).mean(dim=0).detach() for move in moves]
    return {
        "goal_text": sample["goal"],
        "goal": goal,
        "state": state,
        "actions": actions,
        "moves": [list(move) for move in moves],
        "label": moves.index(tuple(sample["correct"])),
        "correct_move": list(sample["correct"]),
    }


def fit(
    rows,
    *,
    mismatch: bool,
    steps: int = STEPS,
    lr: float = LR,
    seed: int = SEED,
    width: int = 128,
    rank: int = 4,
):
    head = StateGoalActionReadout(width=width, rank=rank, seed=seed)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    goals = [row["goal"] for row in rows]
    loss_value = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index, row in enumerate(rows):
            goal = goals[(index + 1) % len(rows)] if mismatch else row["goal"]
            logits = torch.stack(
                [head.score(goal, row["state"], action) for action in row["actions"]]
            )
            label = torch.tensor([row["label"]])
            losses.append(F.cross_entropy(logits.unsqueeze(0), label))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
    return head, loss_value


def evaluate(head, rows) -> dict:
    top1 = 0
    detail = []
    with torch.no_grad():
        for row in rows:
            logits = torch.stack(
                [head.score(row["goal"], row["state"], action) for action in row["actions"]]
            )
            pick = int(torch.argmax(logits))
            hit = pick == row["label"]
            top1 += int(hit)
            detail.append(
                {
                    "goal": row["goal_text"],
                    "correct_move": row["correct_move"],
                    "picked": row["moves"][pick],
                    "hit": hit,
                    "margin": float(logits[row["label"]] - logits.max()),
                }
            )
    return {"top1": top1, "total": len(rows), "rows": detail}


def flat(parameters) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def status_from(*, budget_ok: bool, frozen: bool, control_top1: int, top1: int) -> str:
    if not budget_ok:
        return "budget_stop"
    if not frozen:
        return "backbone_mutated"
    if control_top1 > CONTROL_MAX_TOP1:
        return "control_not_null"
    if top1 >= PASS_TOP1:
        return "readout_selection_supported"
    if top1 <= ABSENT_TOP1:
        return "readout_selection_absent"
    return "readout_selection_inconclusive"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.7f report is write-once")

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

    arm, arm_loss = fit(train_features, mismatch=False)
    control, control_loss = fit(train_features, mismatch=True)
    empty = StateGoalActionReadout(seed=SEED)

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
        control_top1=control_result["top1"],
        top1=arm_result["top1"],
    )

    output_dir = ROOT / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "readout_state_goal_rank4.pt"
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
        "schema": "taiji-selection-10-7f-v1",
        "step": "10.7f",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "samples": len(rows),
            "train_samples": len(train_rows),
            "held_out_samples": len(held_rows),
            "actions_per_sample": len(all_moves()),
            "steps": STEPS,
            "lr": LR,
            "seed": SEED,
            "parameters": arm.parameter_count(),
            "pass_top1": PASS_TOP1,
            "absent_top1": ABSENT_TOP1,
            "control_max_top1": CONTROL_MAX_TOP1,
            "chance_top1": 1 / len(all_moves()),
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
        "capability_claimed": status == "readout_selection_supported",
        "training_admitted": False,
        "limits": (
            "One bounded run on a six-sample synthetic fixture with six actions. The readout "
            "reads the subject's goal/state/action encodings only, not its recurrent state; a "
            "pass means six-way goal-conditioned selection is learnable from these features "
            "here, not that the subject does it, not natural-language capability, not P2."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/goal_state_readout.py",
                "taiji/action_verdict_scorer.py",
                "scripts/training/verify_taiji_selection_10_7f.py",
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
                "arm_top1": arm_result["top1"],
                "control_top1": control_result["top1"],
                "untrained_top1": empty_result["top1"],
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
