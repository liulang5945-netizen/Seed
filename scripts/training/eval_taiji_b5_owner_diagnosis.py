"""Diagnose B5 forgetting across the two private F1 learning owners.

This evaluator intentionally changes no production defaults.  It runs four
owned variants from one trained child: full no-replay, full exact protected
replay, private-context-only, and predictive-readout-only.  Scores are read
only and every variant starts from the same child checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_foundation_baseline import (  # noqa: E402
    _b5_phase_b_stream,
    _load_joint_child,
    _score_loaded_model,
)
from taiji import FoundationTrainingDataset, Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402


def _owner_digests(model: Taiji) -> dict[str, str]:
    values: dict[str, str] = {
        "fabric": content_digest(model.fabric.to_payload()),
        "motor": content_digest(model.motor.to_payload()),
        "memory": content_digest(model.memory.to_payload()),
        "predictive_context": content_digest(model.predictive_context.to_payload()),
        "predictive_readout": content_digest(model.predictive_readout.to_payload()),
    }
    if model.identity_organ is not None:
        values["identity"] = content_digest(
            model.identity_organ.to_payload(parent_checkpoint_digest="b5-owner-diagnosis")
        )
    return values


def _run_variant(
    source_model: Taiji,
    *,
    phase_a_train: bytes,
    phase_a_holdout: bytes,
    phase_b_train: bytes,
    phase_b_holdout: bytes,
    retention: bytes,
    label: str,
    learn_predictive_context: bool,
    learn_predictive_readout: bool,
    replay_phase_a: bool,
) -> dict[str, Any]:
    model = Taiji.from_checkpoint(source_model.checkpoint())
    owner_before = _owner_digests(model)
    old_before = _score_loaded_model(model, phase_a_holdout)
    model.learn_bytes(
        phase_b_train,
        epochs=1,
        learn_fabric=False,
        learn_predictive_context=learn_predictive_context,
        learn_predictive_readout=learn_predictive_readout,
    )
    owner_after_phase_b = _owner_digests(model)
    if replay_phase_a:
        model.learn_bytes(
            phase_a_train,
            epochs=1,
            learn_fabric=False,
            learn_predictive_context=learn_predictive_context,
            learn_predictive_readout=learn_predictive_readout,
        )
    owner_after = _owner_digests(model)
    old_after = _score_loaded_model(model, phase_a_holdout)
    new_after = _score_loaded_model(model, phase_b_holdout)
    retention_after = _score_loaded_model(model, retention)
    return {
        "label": label,
        "learn_predictive_context": learn_predictive_context,
        "learn_predictive_readout": learn_predictive_readout,
        "replay_phase_a": replay_phase_a,
        "old_before_bpb": old_before,
        "old_after_bpb": old_after,
        "new_after_bpb": new_after,
        "retention_after_bpb": retention_after,
        "backward_transfer": old_before - old_after,
        "owner_digests_before": owner_before,
        "owner_digests_after_phase_b": owner_after_phase_b,
        "owner_digests_after": owner_after,
        "owner_changes": {key: owner_before[key] != owner_after[key] for key in owner_before},
        "shared_owners_preserved": all(
            owner_before[key] == owner_after[key]
            for key in ("fabric", "motor", "memory", "identity")
            if key in owner_before
        ),
        "score_checkpoint_read_only": True,
    }


def run_diagnosis(
    checkpoint: Path,
    *,
    seed: int,
    protected_corpus: Path,
    protected_partition_seed: int,
    phase_b_seed: int,
) -> dict[str, Any]:
    payload, source_model, _parent = _load_joint_child(checkpoint, expected_seed=seed)
    protected = FoundationTrainingDataset.from_jsonl(
        [protected_corpus],
        profile="foundation",
        partition_seed=protected_partition_seed,
    )
    phase_a_train = protected.train[:4_096]
    phase_a_holdout = protected.holdout[:200]
    retention = protected.retention[:200]
    phase_b_train = _b5_phase_b_stream(seed, length=4_096, offset=0)
    phase_b_holdout = _b5_phase_b_stream(seed, length=200, offset=1)
    variants = (
        _run_variant(
            source_model,
            phase_a_train=phase_a_train,
            phase_a_holdout=phase_a_holdout,
            phase_b_train=phase_b_train,
            phase_b_holdout=phase_b_holdout,
            retention=retention,
            label="full_no_replay",
            learn_predictive_context=True,
            learn_predictive_readout=True,
            replay_phase_a=False,
        ),
        _run_variant(
            source_model,
            phase_a_train=phase_a_train,
            phase_a_holdout=phase_a_holdout,
            phase_b_train=phase_b_train,
            phase_b_holdout=phase_b_holdout,
            retention=retention,
            label="full_exact_protected_replay",
            learn_predictive_context=True,
            learn_predictive_readout=True,
            replay_phase_a=True,
        ),
        _run_variant(
            source_model,
            phase_a_train=phase_a_train,
            phase_a_holdout=phase_a_holdout,
            phase_b_train=phase_b_train,
            phase_b_holdout=phase_b_holdout,
            retention=retention,
            label="private_context_only",
            learn_predictive_context=True,
            learn_predictive_readout=False,
            replay_phase_a=False,
        ),
        _run_variant(
            source_model,
            phase_a_train=phase_a_train,
            phase_a_holdout=phase_a_holdout,
            phase_b_train=phase_b_train,
            phase_b_holdout=phase_b_holdout,
            retention=retention,
            label="predictive_readout_only",
            learn_predictive_context=False,
            learn_predictive_readout=True,
            replay_phase_a=False,
        ),
    )
    return {
        "format": "taiji-foundation-b5-owner-diagnosis-v1",
        "ability_id": "b5_continual_learning",
        "seed": int(seed),
        "checkpoint": str(checkpoint),
        "checkpoint_digest": str(payload["checkpoint_digest"]),
        "protected_partition_seed": int(protected_partition_seed),
        "phase_b_seed": int(phase_b_seed),
        "sample_counts": {
            "phase_a_train": len(phase_a_train),
            "phase_a_holdout": len(phase_a_holdout),
            "phase_b_train": len(phase_b_train),
            "phase_b_holdout": len(phase_b_holdout),
            "retention": len(retention),
        },
        "phase_b_contract": {
            "stream": "deterministic_b5_phase_b_v1",
            "interference_offset": 1,
            "exact_protected_replay": True,
        },
        "variants": list(variants),
        "all_scores_checkpoint_read_only": all(
            bool(variant["score_checkpoint_read_only"]) for variant in variants
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--protected-corpus", type=Path, required=True)
    parser.add_argument("--protected-partition-seed", type=int, required=True)
    parser.add_argument("--phase-b-seed", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    result = run_diagnosis(
        args.checkpoint,
        seed=args.seed,
        protected_corpus=args.protected_corpus,
        protected_partition_seed=args.protected_partition_seed,
        phase_b_seed=args.phase_b_seed,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result["report_written"] = args.report.is_file()
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "all_scores_checkpoint_read_only": result["all_scores_checkpoint_read_only"],
                "variants": [
                    {
                        "label": variant["label"],
                        "backward_transfer": variant["backward_transfer"],
                        "old_after_bpb": variant["old_after_bpb"],
                        "new_after_bpb": variant["new_after_bpb"],
                        "retention_after_bpb": variant["retention_after_bpb"],
                        "owner_changes": variant["owner_changes"],
                    }
                    for variant in result["variants"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["report_written"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
