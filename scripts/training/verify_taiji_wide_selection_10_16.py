"""10.16: 20-way selection on the wide fixture, state features, glossary in records.

480 samples over 120 states. The state feature depends only on the goal prompt, and
there are just 20 distinct (object, free place) goals, so feature extraction is 40
``run_career`` calls, not 480.

Arms: A train real -> test real (the reading), B train real -> test permuted, C train
permuted -> test real. The subject is frozen; only the 2,560-parameter head trains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

import scripts.training.verify_taiji_subject_selection_10_4 as prior  # noqa: E402
from taiji.living_core import RawSource  # noqa: E402
from taiji.native_goal_glossary import journal_with_records  # noqa: E402
from taiji.native_goal_world_wide import (  # noqa: E402
    GLOSSARY,
    OBJECTS,
    PLACES,
    action_sentence,
    describe_goal,
    samples,
    states,
)
from taiji.state_action_readout import StateActionReadout  # noqa: E402
from taiji.state_migration import run_career  # noqa: E402

TRAIN_STATES = 80
STEPS = 300
LR = 0.05
SEED = 0
BUDGET_CPU_SECONDS = 600
BUDGET_WALL_SECONDS = 1200
PASS_SHARE = 0.40
ABSENT_SHARE = 0.10
CONTROL_MAX_SHARE = 0.10


def glossary_records(permuted: bool = False) -> tuple[RawSource, ...]:
    items = list(GLOSSARY.items())
    if permuted:
        entities = [entity for _, entity in items]
        items = [
            (text, entities[(index + 1) % len(entities)]) for index, (text, _) in enumerate(items)
        ]
    return tuple(
        RawSource(source_id=f"glossary-{index}", raw=f"{text}是{entity}。".encode())
        for index, (text, entity) in enumerate(items)
    )


def state_feature(subject, provider, goal: str, records, tag: str):
    journal = journal_with_records(goal, "x", tag, records)
    _, steps = run_career(subject, journal, growth_gate_provider=provider, retain_outputs=True)
    last_patch = max(position for position, (_, byte) in enumerate(steps) if byte is None)
    return steps[last_patch][0].state.h.detach().reshape(-1)


def scores_for(head, features, action_vectors):
    return (features @ head.left) @ (action_vectors @ head.right).transpose(0, 1)


def fit(head, features, labels, action_vectors, *, steps: int = STEPS, lr: float = LR):
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loss_value = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        scores = scores_for(head, features, action_vectors)
        loss = F.cross_entropy(scores, labels)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
    return loss_value


def evaluate(head, features, labels, action_vectors) -> dict:
    with torch.no_grad():
        scores = scores_for(head, features, action_vectors)
        picks = scores.argmax(dim=1)
    hits = int((picks == labels).sum())
    return {
        "hits": hits,
        "total": int(labels.numel()),
        "share": hits / int(labels.numel()),
        "picks": Counter(int(value) for value in picks).most_common(5),
    }


def flat(parameters) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def status_from(*, budget_ok, frozen, control_share, share) -> str:
    if not budget_ok:
        return "budget_stop"
    if not frozen:
        return "backbone_mutated"
    if control_share > CONTROL_MAX_SHARE:
        return "control_not_null"
    if share >= PASS_SHARE:
        return "wide_selection_supported"
    if share <= ABSENT_SHARE:
        return "wide_selection_absent"
    return "wide_selection_below_floor"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.16 report is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    parent, subject, provider = prior.build_subject()
    digest = prior.parent_digest(parent)
    encoder = subject.encoder

    moves = [(obj, place) for obj in OBJECTS for place in PLACES]
    with torch.no_grad():
        action_vectors = torch.stack(
            [
                encoder.context(action_sentence(*move).encode()).mean(dim=0).detach()
                for move in moves
            ]
        )

    real = glossary_records(permuted=False)
    permuted = glossary_records(permuted=True)
    state_features = {}
    for obj in OBJECTS:
        for place in PLACES:
            goal = describe_goal(obj, place)
            state_features[(obj, place, "real")] = state_feature(
                subject, provider, goal, real, f"r-{obj}-{place}"
            )
            state_features[(obj, place, "permuted")] = state_feature(
                subject, provider, goal, permuted, f"p-{obj}-{place}"
            )

    rows = samples()
    per_state = len(rows) // len(states())
    train_rows = rows[: TRAIN_STATES * per_state]
    held_rows = rows[TRAIN_STATES * per_state :]

    def build(subset, variant):
        features = torch.stack(
            [state_features[(row["correct"][0], row["correct"][1], variant)] for row in subset]
        )
        labels = torch.tensor([moves.index(tuple(row["correct"])) for row in subset])
        return features, labels

    before = flat(list(subject.parameters()))

    width = int(state_features[(OBJECTS[0], PLACES[0], "real")].numel())
    arms = {}
    counts = set()
    for name, train_variant, test_variant in (
        ("A_real_to_real", "real", "real"),
        ("B_real_to_permuted", "real", "permuted"),
        ("C_permuted_to_real", "permuted", "real"),
    ):
        head = StateActionReadout(width, seed=SEED)
        counts.add(head.parameter_count())
        train_features, train_labels = build(train_rows, train_variant)
        held_features, held_labels = build(held_rows, test_variant)
        loss_value = fit(head, train_features, train_labels, action_vectors)
        arms[name] = {
            **evaluate(head, held_features, held_labels, action_vectors),
            "train_loss": loss_value,
        }

    after = flat(list(subject.parameters()))
    backbone_frozen = bool(torch.equal(before, after))
    constant = Counter(tuple(row["correct"]) for row in held_rows).most_common(1)[0]
    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    status = status_from(
        budget_ok=cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
        frozen=backbone_frozen,
        control_share=arms["B_real_to_permuted"]["share"],
        share=arms["A_real_to_real"]["share"],
    )

    report = {
        "schema": "taiji-wide-selection-10-16-v1",
        "step": "10.16",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "states": len(states()),
            "samples": len(rows),
            "train_samples": len(train_rows),
            "held_out_samples": len(held_rows),
            "actions": len(moves),
            "state_width": width,
            "parameters": sorted(counts)[0] if len(counts) == 1 else list(counts),
            "steps": STEPS,
            "lr": LR,
            "pass_share": PASS_SHARE,
            "absent_share": ABSENT_SHARE,
            "control_max_share": CONTROL_MAX_SHARE,
            "chance_share": 1 / len(moves),
            "feature_calls": 2 * len(OBJECTS) * len(PLACES),
        },
        "parent_model_digest": digest,
        "arms": arms,
        "best_constant": {"move": list(constant[0]), "hits": constant[1], "total": len(held_rows)},
        "backbone_frozen": backbone_frozen,
        "weights_published": False,
        "default_model_replaced": False,
        "model_involved": True,
        "capability_claimed": status == "wide_selection_supported",
        "training_admitted": False,
        "limits": (
            "One bounded run on a synthetic wide fixture. The readout sees the frozen "
            "subject's state after the goal prompt; the glossary enters only through "
            "records. A pass would mean 20-way selection is learnable from this state at "
            "this data size -- not P2, not natural language, not the subject acting."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/native_goal_world_wide.py",
                "taiji/state_action_readout.py",
                "scripts/training/verify_taiji_wide_selection_10_16.py",
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
                "A": arms["A_real_to_real"]["share"],
                "B": arms["B_real_to_permuted"]["share"],
                "C": arms["C_permuted_to_real"]["share"],
                "held": len(held_rows),
                "best_constant": constant[1],
                "parameters": report["declared"]["parameters"],
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
