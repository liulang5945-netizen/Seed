"""10.13: six-way selection from the subject's own recurrent state, glossary in records.

The final authorised run. Features come from ``state.h`` after the goal prompt (the only
place the glossary can reach), actions from the encoder; the subject stays frozen and
only a small bilinear head trains. Three arms separate "learned the mapping" from
"learned anything":

* A: train real -> test real   (the reading)
* B: train real -> test permuted (should collapse to chance if the mapping is used)
* C: train permuted -> test real (should stay at chance if the mapping is what was learned)
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
from taiji.goal_conditioned_training import sample_loss, samples  # noqa: E402
from taiji.goal_paraphrase import action_sentence  # noqa: E402
from taiji.living_core import RawSource  # noqa: E402
from taiji.native_goal_attributes import (  # noqa: E402
    OBJECT_DESCRIPTIONS,
    PLACE_DESCRIPTIONS,
    describe_goal,
)
from taiji.native_goal_glossary import glossary_records, journal_with_records  # noqa: E402
from taiji.native_goal_world import OBJECTS, PLACES  # noqa: E402
from taiji.state_action_readout import StateActionReadout  # noqa: E402
from taiji.state_migration import run_career  # noqa: E402

TRAIN_STATES = 3
STEPS = 300
LR = 0.05
SEED = 0
BUDGET_CPU_SECONDS = 300
BUDGET_WALL_SECONDS = 600
PASS_TOP1 = 5
ABSENT_TOP1 = 2
CONTROL_MAX_TOP1 = 2
OUTPUT_DIR = "output/taiji_core_10_13"


def permuted_records():
    object_items = list(OBJECT_DESCRIPTIONS.items())
    place_items = list(PLACE_DESCRIPTIONS.items())
    lines = [
        f"{text}是{object_items[(index + 1) % len(object_items)][0]}。"
        for index, (_, text) in enumerate(object_items)
    ]
    lines += [
        f"{text}是{place_items[(index + 1) % len(place_items)][0]}。"
        for index, (_, text) in enumerate(place_items)
    ]
    return tuple(
        RawSource(source_id=f"permuted-{index}", raw=line.encode())
        for index, line in enumerate(lines)
    )


def state_feature(subject, provider, goal: str, records, tag: str):
    journal = journal_with_records(goal, "x", tag, records)
    _, steps = run_career(subject, journal, growth_gate_provider=provider, retain_outputs=True)
    last_patch = max(position for position, (_, byte) in enumerate(steps) if byte is None)
    return steps[last_patch][0].state.h.detach().reshape(-1)


def fit(readout, rows, *, steps: int = STEPS, lr: float = LR):
    optimizer = torch.optim.Adam(readout.parameters(), lr=lr)
    loss_value = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for row in rows:
            logits = torch.stack([readout.score(row["state"], action) for action in row["actions"]])
            losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([row["label"]])))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
    return loss_value


def evaluate(readout, rows) -> dict:
    top1 = 0
    detail = []
    with torch.no_grad():
        for row in rows:
            logits = torch.stack([readout.score(row["state"], action) for action in row["actions"]])
            pick = int(torch.argmax(logits))
            hit = pick == row["label"]
            top1 += int(hit)
            detail.append(
                {
                    "goal": row["goal"],
                    "correct": row["moves"][row["label"]],
                    "picked": row["moves"][pick],
                    "hit": hit,
                }
            )
    return {"top1": top1, "total": len(rows), "rows": detail}


def flat(parameters) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def status_from(*, budget_ok, frozen, control_top1, top1) -> str:
    if not budget_ok:
        return "budget_stop"
    if not frozen:
        return "backbone_mutated"
    if control_top1 > CONTROL_MAX_TOP1:
        return "control_not_null"
    if top1 >= PASS_TOP1:
        return "state_selection_supported"
    if top1 <= ABSENT_TOP1:
        return "state_selection_absent"
    return "state_selection_inconclusive"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.13 report is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    parent, subject, provider = prior.build_subject()
    digest = prior.parent_digest(parent)
    encoder = subject.encoder
    real = glossary_records()
    wrong = permuted_records()
    moves = [(obj, place) for obj in OBJECTS for place in PLACES]

    with torch.no_grad():
        action_vectors = [
            encoder.context(action_sentence(*move).encode()).mean(dim=0).detach() for move in moves
        ]

    rows = samples()
    per_state = len(rows) // 6
    features = []
    for index, sample in enumerate(rows):
        obj, free = sample["correct"]
        goal = describe_goal(obj, free)
        features.append(
            {
                "goal": goal,
                "label": moves.index((obj, free)),
                "moves": [list(move) for move in moves],
                "actions": action_vectors,
                "real": state_feature(subject, provider, goal, real, f"r{index}"),
                "permuted": state_feature(subject, provider, goal, wrong, f"p{index}"),
                "group": 0 if index < TRAIN_STATES * per_state else 1,
            }
        )
    train = [row for row in features if row["group"] == 0]
    held = [row for row in features if row["group"] == 1]

    before = flat(list(subject.parameters()))
    with torch.no_grad():
        losses_before = [
            float(sample_loss(subject, sample, provider).detach())
            for sample in rows[TRAIN_STATES * per_state :]
        ]

    width = int(held[0]["real"].numel())
    report_parameters = StateActionReadout(width, seed=SEED).parameter_count()
    arm_rows = {}
    arms = {
        "A_real_to_real": ("real", "real"),
        "B_real_to_permuted": ("real", "permuted"),
        "C_permuted_to_real": ("permuted", "real"),
    }
    parameter_counts = set()
    for name, (train_key, test_key) in arms.items():
        readout = StateActionReadout(width, seed=SEED)
        parameter_counts.add(readout.parameter_count())
        train_rows = [dict(row, state=row[train_key]) for row in train]
        loss_value = fit(readout, train_rows)
        test_rows = [dict(row, state=row[test_key]) for row in held]
        arm_rows[name] = {**evaluate(readout, test_rows), "train_loss": loss_value}

    after = flat(list(subject.parameters()))
    with torch.no_grad():
        losses_after = [
            float(sample_loss(subject, sample, provider).detach())
            for sample in rows[TRAIN_STATES * per_state :]
        ]
    backbone_frozen = bool(torch.equal(before, after)) and losses_before == losses_after

    if parameter_counts != {report_parameters}:
        raise RuntimeError(f"declared parameters {report_parameters} != trained {parameter_counts}")

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    status = status_from(
        budget_ok=cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
        frozen=backbone_frozen,
        control_top1=arm_rows["B_real_to_permuted"]["top1"],
        top1=arm_rows["A_real_to_real"]["top1"],
    )

    output_dir = ROOT / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "state_action_rank4.pt"
    torch.save({"state_width": width, "rank": 4}, weights_path)

    report = {
        "schema": "taiji-state-selection-10-13-v1",
        "step": "10.13",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "samples": len(rows),
            "train_samples": len(train),
            "held_out_samples": len(held),
            "actions": len(moves),
            "state_width": width,
            "parameters": report_parameters,
            "steps": STEPS,
            "lr": LR,
            "pass_top1": PASS_TOP1,
            "absent_top1": ABSENT_TOP1,
            "control_max_top1": CONTROL_MAX_TOP1,
            "chance_top1": 1 / len(moves),
        },
        "parent_model_digest": digest,
        "arms": arm_rows,
        "backbone_frozen": backbone_frozen,
        "held_out_losses_before": losses_before,
        "held_out_losses_after": losses_after,
        "weights_path": str(weights_path.relative_to(ROOT)),
        "weights_published": False,
        "default_model_replaced": False,
        "model_involved": True,
        "capability_claimed": status == "state_selection_supported",
        "training_admitted": False,
        "limits": (
            "One bounded run on six training samples of a synthetic fixture. Features are "
            "the frozen subject's recurrent state after the goal prompt; the glossary only "
            "reaches the model through records. A pass means six-way selection is learnable "
            "from this state at this data size, not that the subject selects, not natural "
            "language, not P2."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/state_action_readout.py",
                "taiji/native_goal_glossary.py",
                "scripts/training/verify_taiji_state_selection_10_13.py",
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
                "A": arm_rows["A_real_to_real"]["top1"],
                "B": arm_rows["B_real_to_permuted"]["top1"],
                "C": arm_rows["C_permuted_to_real"]["top1"],
                "total": len(held),
                "backbone_frozen": backbone_frozen,
                "parameters": report["declared"]["parameters"],
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
