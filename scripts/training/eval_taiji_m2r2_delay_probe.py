"""M2.R2.R0.2 controlled delayed-dependency probe.

R2.R0 measured training-byte budgets, which cannot identify a temporal credit
span.  This probe therefore uses record-disjoint synthetic sequences with a
known cue-to-target distance.  Each record contains a random two-byte prefix,
a cue byte, random distractors, and the cue again at the target position.
The prefix makes even distance-one records distinct across train/dev/test;
the target rule is unchanged across splits.

The probe is deliberately small and CPU-runnable.  It compares the current
private predictive residual with frozen, protected-readout-only, and shuffled
sequence controls.  It is diagnostic only and never promotes a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m2r1_phase_c_canary import (  # noqa: E402
    _process_working_set_bytes,
    load_joint_child,
)
from scripts.training.eval_taiji_m2r2_learning_curve import (  # noqa: E402
    DEFAULT_CORPUS,
    _owner_digests,
    _preflight_checkpoint,
)
from taiji import Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-r2-delay-probe-v1"
DEFAULT_DISTANCES = (1, 8, 32, 128, 512)
DEFAULT_ARMS = (
    "frozen",
    "joint_predictive",
    "shuffled_joint",
    "readout_only",
    "gated_joint",
)
ALL_ARMS = frozenset(DEFAULT_ARMS)
DEFAULT_TRAIN_RECORDS = 32
DEFAULT_DEV_RECORDS = 16
DEFAULT_TEST_RECORDS = 16
PREFIX_SIZE = 2
DEFAULT_EPOCHS = 3
CUE_VALUES = (16, 32, 64, 96)


def _record(
    *,
    distance: int,
    split_seed: int,
    index: int,
) -> bytes:
    rng = random.Random(int(split_seed) * 1_000_003 + int(distance) * 97 + int(index))
    prefix = bytes(rng.randrange(240, 256) for _ in range(PREFIX_SIZE))
    cue = CUE_VALUES[rng.randrange(len(CUE_VALUES))]
    distractors = bytes(rng.randrange(128, 240) for _ in range(int(distance) - 1))
    return prefix + bytes((cue,)) + distractors + bytes((cue,))


def build_delay_records(
    distance: int,
    *,
    seed: int,
    train_count: int = DEFAULT_TRAIN_RECORDS,
    dev_count: int = DEFAULT_DEV_RECORDS,
    test_count: int = DEFAULT_TEST_RECORDS,
) -> dict[str, list[bytes]]:
    """Build deterministic, exact-record-disjoint delayed-copy splits."""

    if int(distance) <= 0:
        raise ValueError("distance must be positive")
    counts = {
        "train": int(train_count),
        "dev": int(dev_count),
        "test": int(test_count),
    }
    if any(value <= 0 for value in counts.values()):
        raise ValueError("record counts must be positive")
    records: dict[str, list[bytes]] = {}
    seen: set[bytes] = set()
    for offset, (split, count) in enumerate(counts.items()):
        split_records: list[bytes] = []
        cursor = 0
        while len(split_records) < count:
            value = _record(
                distance=int(distance),
                split_seed=int(seed) + 101 * (offset + 1),
                index=cursor,
            )
            cursor += 1
            if value in seen:
                continue
            seen.add(value)
            split_records.append(value)
        records[split] = split_records
    return records


def _delay_target_index(record: bytes, distance: int) -> int:
    expected = PREFIX_SIZE + int(distance)
    if len(record) != expected + 1:
        raise ValueError("delay record length does not match its distance")
    if record[PREFIX_SIZE] != record[-1]:
        raise ValueError("delay record target is not equal to its cue")
    return int(expected)


def _arm_write_set(arm: str) -> set[str]:
    return {
        "frozen": set(),
        "joint_predictive": {"predictive_context", "protected_predictive_readout"},
        "shuffled_joint": {"predictive_context", "protected_predictive_readout"},
        "readout_only": {"protected_predictive_readout"},
        "gated_joint": {"gated_temporal_candidate", "protected_predictive_readout"},
    }[arm]


def _owner_contract(
    arm: str,
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> dict[str, Any]:
    expected = _arm_write_set(arm)
    changed = {
        owner: str(before.get(owner, "")) != str(after.get(owner, ""))
        for owner in sorted(set(before) | set(after))
    }
    unexpected = sorted(owner for owner, value in changed.items() if value and owner not in expected)
    expected_changed = sorted(owner for owner in expected if changed.get(owner, False))
    return {
        "changed": changed,
        "expected_writable_owners": sorted(expected),
        "changed_expected_owners": expected_changed,
        "unexpected_changed_owners": unexpected,
        "contract_passed": not unexpected,
        "required_owner_changed": bool(expected) and bool(expected_changed),
    }


def _train_arm(
    model: Taiji,
    records: Sequence[bytes],
    *,
    distance: int,
    arm: str,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    observations = 0
    correct = 0
    shuffled_records = []
    for index, record in enumerate(records):
        if arm == "shuffled_joint":
            values = list(record)
            random.Random(seed + distance * 1009 + index).shuffle(values)
            shuffled_records.append(bytes(values))
        else:
            shuffled_records.append(record)

    for epoch in range(int(epochs)):
        for index, record in enumerate(shuffled_records):
            model.reset_dynamics(episode_id=f"delay-{arm}-{distance}-{epoch}-{index}")
            learn = arm != "frozen"
            learn_context = arm in {"joint_predictive", "shuffled_joint", "gated_joint"}
            learn_readout = arm in {
                "joint_predictive",
                "shuffled_joint",
                "readout_only",
                "gated_joint",
            }
            for symbol in record:
                step = model.observe(
                    int(symbol),
                    learn=learn,
                    learn_fabric=False,
                    learn_predictive_context=learn_context,
                    learn_predictive_readout=learn_readout,
                    readout="predictive",
                    use_memory=False,
                    use_identity=False,
                )
                if step.prior_prediction is not None:
                    observations += 1
                    correct += int(step.prior_prediction == int(symbol))
    elapsed = time.perf_counter() - started
    return {
        "epochs": int(epochs),
        "records": len(records),
        "observations": int(observations),
        "online_accuracy": correct / max(1, observations),
        "seconds": float(elapsed),
        "working_set_bytes": _process_working_set_bytes(),
    }


def _evaluate_records(
    model: Taiji,
    records: Sequence[bytes],
    *,
    distance: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate only the delayed target, restoring all state afterwards."""

    checkpoint = model.checkpoint()
    before = content_digest(checkpoint)
    correct = 0
    losses: list[float] = []
    try:
        for index, record in enumerate(records):
            target_index = _delay_target_index(record, distance)
            model.reset_dynamics(episode_id=f"delay-eval-{distance}-{seed}-{index}")
            target_step = None
            for position, symbol in enumerate(record):
                step = model.observe(
                    int(symbol),
                    learn=False,
                    readout="predictive",
                    use_memory=False,
                    use_identity=False,
                )
                if position == target_index:
                    target_step = step
            if target_step is None or target_step.prior_prediction is None:
                raise RuntimeError("delay target did not produce a prior prediction")
            correct += int(target_step.prior_prediction == int(record[target_index]))
            losses.append(-math.log2(max(float(target_step.prior_probability or 0.0), 1e-12)))
        return {
            "records": len(records),
            "target_accuracy": correct / max(1, len(records)),
            "target_bpb": sum(losses) / max(1, len(losses)),
            "chance_accuracy": 1.0 / float(model.config.alphabet_size),
            "read_only": True,
        }
    finally:
        model.restore(checkpoint)
        after = content_digest(model.checkpoint())
        if before != after:
            raise RuntimeError("delay evaluation mutated the model checkpoint")


def _target_unigram_bpb(
    train_records: Sequence[bytes],
    test_records: Sequence[bytes],
    *,
    distance: int,
    alphabet_size: int,
) -> float:
    counts = [1.0] * int(alphabet_size)
    for record in train_records:
        counts[int(record[_delay_target_index(record, distance)])] += 1.0
    total = sum(counts)
    losses = [
        -math.log2(
            counts[int(record[_delay_target_index(record, distance)])] / total
        )
        for record in test_records
    ]
    return sum(losses) / max(1, len(losses))


def _lesion_score(
    model: Taiji,
    records: Sequence[bytes],
    *,
    distance: int,
    seed: int,
    candidate: bool,
) -> dict[str, Any]:
    lesioned = Taiji.from_checkpoint(model.checkpoint())
    if candidate:
        if not lesioned.gated_temporal_candidate_enabled:
            raise RuntimeError("candidate lesion requires an enabled candidate")
        owner_before = content_digest(
            lesioned.checkpoint()["gated_temporal_candidate"]
        )
        lesioned.zero_gated_temporal_candidate()
        owner_after = content_digest(
            lesioned.checkpoint()["gated_temporal_candidate"]
        )
        owner = "gated_temporal_candidate"
    else:
        owner_before = content_digest(lesioned.predictive_context.recurrent.to_payload())
        with torch.no_grad():
            lesioned.predictive_context.recurrent.edge_weight.zero_()
        owner_after = content_digest(lesioned.predictive_context.recurrent.to_payload())
        owner = "predictive_context"
    score = _evaluate_records(
        lesioned,
        records,
        distance=distance,
        seed=seed,
    )
    return {
        "score": score,
        "lesion_owner": owner,
        "lesion_applied": owner_before != owner_after,
        "owner_digest_before": owner_before,
        "owner_digest_after": owner_after,
    }


def _run_point(
    *,
    source_payload: Mapping[str, Any],
    records: Mapping[str, Sequence[bytes]],
    distance: int,
    arm: str,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    model = Taiji.from_checkpoint(source_payload)
    if arm == "gated_joint":
        model.enable_gated_temporal_candidate()
    before = _owner_digests(model)
    if model.gated_temporal_candidate_enabled:
        before["gated_temporal_candidate"] = content_digest(
            model.checkpoint()["gated_temporal_candidate"]
        )
    train_metrics = _train_arm(
        model,
        records["train"],
        distance=distance,
        arm=arm,
        epochs=epochs,
        seed=seed,
    )
    after = _owner_digests(model)
    if model.gated_temporal_candidate_enabled:
        after["gated_temporal_candidate"] = content_digest(
            model.checkpoint()["gated_temporal_candidate"]
        )
    attribution = _owner_contract(arm, before, after)
    train_score = _evaluate_records(
        model,
        records["train"],
        distance=distance,
        seed=seed,
    )
    dev_score = _evaluate_records(
        model,
        records["dev"],
        distance=distance,
        seed=seed + 1,
    )
    test_score = _evaluate_records(
        model,
        records["test"],
        distance=distance,
        seed=seed + 2,
    )
    lesion = None
    if arm in {"joint_predictive", "shuffled_joint", "gated_joint"}:
        lesion = _lesion_score(
            model,
            records["test"],
            distance=distance,
            seed=seed + 3,
            candidate=arm == "gated_joint",
        )
    payload = model.checkpoint()
    checkpoint_digest = content_digest(payload)
    restored = Taiji.from_checkpoint(payload)
    restored_digest = content_digest(restored.checkpoint())
    temporary = PROJECT_ROOT / "output" / "taiji-m2r2-delay-point.pt"
    torch.save(payload, temporary)
    checkpoint_bytes = temporary.stat().st_size
    temporary.unlink(missing_ok=True)
    checks = {
        "owner_contract": bool(attribution["contract_passed"]),
        "evaluation_read_only": bool(
            train_score["read_only"]
            and dev_score["read_only"]
            and test_score["read_only"]
        ),
        "checkpoint_round_trip": checkpoint_digest == restored_digest,
        "lesion_effective": lesion is None or bool(lesion["lesion_applied"]),
        "lesion_evaluation_read_only": lesion is None or bool(lesion["score"]["read_only"]),
    }
    return {
        "distance": int(distance),
        "arm": arm,
        "dataset_digests": {
            split: content_digest({"distance": int(distance), "records": list(values)})
            for split, values in records.items()
        },
        "train": train_metrics,
        "train_target": train_score,
        "dev": dev_score,
        "test": test_score,
        "temporal_lesion": lesion,
        "target_unigram_bpb": _target_unigram_bpb(
            records["train"],
            records["test"],
            distance=distance,
            alphabet_size=model.config.alphabet_size,
        ),
        "owners": {
            "before": before,
            "after": after,
            "attribution": attribution,
        },
        "checkpoint": {
            "digest": checkpoint_digest,
            "restored_digest": restored_digest,
            "bytes": int(checkpoint_bytes),
        },
        "checks": checks,
    }


def run_probe(
    checkpoint: Path,
    *,
    corpus_paths: Sequence[Path],
    seed: int,
    distances: Sequence[int] = DEFAULT_DISTANCES,
    arms: Sequence[str] = DEFAULT_ARMS,
    epochs: int = DEFAULT_EPOCHS,
    train_records: int = DEFAULT_TRAIN_RECORDS,
    dev_records: int = DEFAULT_DEV_RECORDS,
    test_records: int = DEFAULT_TEST_RECORDS,
) -> dict[str, Any]:
    normalized_distances = tuple(sorted(dict.fromkeys(int(value) for value in distances)))
    normalized_arms = tuple(dict.fromkeys(str(value) for value in arms))
    if not normalized_distances or any(value <= 0 for value in normalized_distances):
        raise ValueError("distances must contain positive values")
    if not normalized_arms or any(value not in ALL_ARMS for value in normalized_arms):
        raise ValueError(f"arms must be drawn from {sorted(ALL_ARMS)}")
    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")

    payload, source_model = load_joint_child(checkpoint, expected_seed=seed)
    # The native delayed probe is independent of the corpus bytes, but the
    # source lineage is still validated so a wrong child cannot masquerade as
    # a continuation of the measured R2 parent.
    from scripts.training.eval_taiji_m2r1_phase_c_canary import build_disjoint_phase_chain

    chain = build_disjoint_phase_chain(corpus_paths, cohort_seeds=(11, 29, 47))
    phase_a = chain.phase_a_by_seed[int(seed)]
    phase_b = chain.phase_b_by_seed[int(seed)]
    if str(payload.get("protected_dataset_digest")) != phase_a.digest:
        raise ValueError("source protected_dataset_digest does not match phase-A lineage")
    if str(payload.get("dataset_digest")) != phase_b.digest:
        raise ValueError("source dataset_digest does not match phase-B lineage")

    source_payload = source_model.checkpoint()
    source_digest = content_digest(source_payload)
    preflight = _preflight_checkpoint(
        source_model,
        PROJECT_ROOT / "output" / "taiji-m2r2-delay-preflight.pt",
    )
    if not preflight["matching_digest"]:
        raise RuntimeError("source checkpoint save/restore preflight failed")

    points: list[dict[str, Any]] = []
    dataset_metadata: dict[str, Any] = {}
    for distance in normalized_distances:
        records = build_delay_records(
            distance,
            seed=seed,
            train_count=train_records,
            dev_count=dev_records,
            test_count=test_records,
        )
        train_digests = {
            split: content_digest({"distance": distance, "records": values})
            for split, values in records.items()
        }
        dataset_metadata[str(distance)] = {
            "distance": int(distance),
            "record_counts": {key: len(value) for key, value in records.items()},
            "split_digests": train_digests,
            "record_overlap": {
                "train_dev": len(set(records["train"]) & set(records["dev"])),
                "train_test": len(set(records["train"]) & set(records["test"])),
                "dev_test": len(set(records["dev"]) & set(records["test"])),
            },
        }
        for arm in normalized_arms:
            points.append(
                _run_point(
                    source_payload=source_payload,
                    records=records,
                    distance=distance,
                    arm=arm,
                    epochs=epochs,
                    seed=seed,
                )
            )

    all_passed = all(
        bool(value) for point in points for value in point["checks"].values()
    )
    return {
        "format": FORMAT,
        "version": 1,
        "status": "passed" if all_passed else "failed",
        "can_promote": False,
        "source_checkpoint": str(checkpoint),
        "source_digest": source_digest,
        "seed": int(seed),
        "distances": list(normalized_distances),
        "arms": list(normalized_arms),
        "epochs": int(epochs),
        "dataset": {
            "prefix_size": PREFIX_SIZE,
            "cue_values": list(CUE_VALUES),
            "distractor_range": [128, 239],
            "prefix_range": [240, 255],
            "by_distance": dataset_metadata,
        },
        "lineage": {
            "phase_a_digest": phase_a.digest,
            "phase_b_digest": phase_b.digest,
            "corpus_paths": [str(path) for path in corpus_paths],
        },
        "preflight": preflight,
        "points": points,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "output" / "taiji-m2s-seed11-identity-generation-20260905" / "last.pt",
    )
    parser.add_argument("--corpus", type=Path, nargs="+", default=[DEFAULT_CORPUS])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--distances", type=int, nargs="+", default=list(DEFAULT_DISTANCES))
    parser.add_argument("--arms", nargs="+", choices=sorted(ALL_ARMS), default=list(DEFAULT_ARMS))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--train-records", type=int, default=DEFAULT_TRAIN_RECORDS)
    parser.add_argument("--dev-records", type=int, default=DEFAULT_DEV_RECORDS)
    parser.add_argument("--test-records", type=int, default=DEFAULT_TEST_RECORDS)
    parser.add_argument("--report", type=Path, required=True, help="JSON report path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_probe(
        args.checkpoint,
        corpus_paths=args.corpus,
        seed=args.seed,
        distances=args.distances,
        arms=args.arms,
        epochs=args.epochs,
        train_records=args.train_records,
        dev_records=args.dev_records,
        test_records=args.test_records,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "status": report["status"],
                "points": len(report["points"]),
                "technical_checks_passed": sum(
                    bool(value)
                    for point in report["points"]
                    for value in point["checks"].values()
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
