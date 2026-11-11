"""10.17: the wide fixture split by *object*, not by state.

10.16 split by state and every arm scored 100%: the 480 samples only use 20 distinct
goals, and a state split holds none of them out, so the head memorised all 20. This run
holds out one whole object (its descriptor and its five actions never appear as a
correct label in training) and re-uses 10.16's helpers.

Arms: A train real -> test real, B train real -> test permuted, C train permuted -> test
real. The subject stays frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import scripts.training.verify_taiji_subject_selection_10_4 as prior  # noqa: E402
from taiji.native_goal_world_wide import (  # noqa: E402
    OBJECTS,
    PLACES,
    action_sentence,
    describe_goal,
    samples,
)
from taiji.state_action_readout import StateActionReadout  # noqa: E402

CORE_PATH = ROOT / "scripts" / "training" / "verify_taiji_wide_selection_10_16.py"

BUDGET_CPU_SECONDS = 600
BUDGET_WALL_SECONDS = 1200


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


WIDE = _load("verify_taiji_wide_selection_10_16", CORE_PATH)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.17 report is write-once")

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

    held_object = OBJECTS[-1]
    real = WIDE.glossary_records(permuted=False)
    permuted = WIDE.glossary_records(permuted=True)
    features = {}
    for obj in OBJECTS:
        for place in PLACES:
            goal = describe_goal(obj, place)
            features[(obj, place, "real")] = WIDE.state_feature(
                subject, provider, goal, real, f"r-{obj}-{place}"
            )
            features[(obj, place, "permuted")] = WIDE.state_feature(
                subject, provider, goal, permuted, f"p-{obj}-{place}"
            )

    rows = samples()
    train_rows = [row for row in rows if row["correct"][0] != held_object]
    held_rows = [row for row in rows if row["correct"][0] == held_object]

    def build(subset, variant):
        stacked = torch.stack(
            [features[(row["correct"][0], row["correct"][1], variant)] for row in subset]
        )
        labels = torch.tensor([moves.index(tuple(row["correct"])) for row in subset])
        return stacked, labels

    before = WIDE.flat(list(subject.parameters()))
    width = int(features[(OBJECTS[0], PLACES[0], "real")].numel())
    arms = {}
    counts = set()
    for name, train_variant, test_variant in (
        ("A_real_to_real", "real", "real"),
        ("B_real_to_permuted", "real", "permuted"),
        ("C_permuted_to_real", "permuted", "real"),
    ):
        head = StateActionReadout(width, seed=WIDE.SEED)
        counts.add(head.parameter_count())
        train_features, train_labels = build(train_rows, train_variant)
        held_features, held_labels = build(held_rows, test_variant)
        loss_value = WIDE.fit(head, train_features, train_labels, action_vectors)
        arms[name] = {
            **WIDE.evaluate(head, held_features, held_labels, action_vectors),
            "train_loss": loss_value,
        }
    after = WIDE.flat(list(subject.parameters()))
    backbone_frozen = bool(torch.equal(before, after))

    constant = Counter(tuple(row["correct"]) for row in held_rows).most_common(1)[0]
    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    status = WIDE.status_from(
        budget_ok=cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
        frozen=backbone_frozen,
        control_share=arms["B_real_to_permuted"]["share"],
        share=arms["A_real_to_real"]["share"],
    )

    report = {
        "schema": "taiji-wide-object-split-10-17-v1",
        "step": "10.17",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "held_out_object": held_object,
            "train_samples": len(train_rows),
            "held_out_samples": len(held_rows),
            "unseen_goals_in_training": True,
            "actions": len(moves),
            "parameters": sorted(counts)[0] if len(counts) == 1 else list(counts),
            "pass_share": WIDE.PASS_SHARE,
            "absent_share": WIDE.ABSENT_SHARE,
            "control_max_share": WIDE.CONTROL_MAX_SHARE,
            "chance_share": 1 / len(moves),
            "note": "10.16 split by state and scored 100% in all arms; that split holds no goal out",
        },
        "parent_model_digest": digest,
        "arms": arms,
        "best_constant": {
            "move": list(constant[0]),
            "hits": constant[1],
            "total": len(held_rows),
        },
        "backbone_frozen": backbone_frozen,
        "weights_published": False,
        "default_model_replaced": False,
        "model_involved": True,
        "capability_claimed": status == "wide_selection_supported",
        "training_admitted": False,
        "limits": (
            "One bounded run with one whole object held out of training. A pass would mean "
            "the readout generalises to an unseen object's descriptor through the glossary; "
            "it is still a synthetic fixture, not P2 and not the subject acting."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/native_goal_world_wide.py",
                "taiji/state_action_readout.py",
                "scripts/training/verify_taiji_wide_object_split_10_17.py",
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
                "held_object": held_object,
                "held": len(held_rows),
                "best_constant": constant[1],
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
