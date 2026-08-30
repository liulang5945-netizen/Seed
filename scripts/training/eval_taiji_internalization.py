"""Run the deterministic R5A-S1 native internalization canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    GroundedOutcomeEvidence,
    InternalizationCausalGate,
    InternalizationConverter,
    InternalizationLedger,
    InternalizedFeatureLearner,
    Outcome,
    WorldAffordance,
    content_digest,
)

REPORT_FORMAT = "taiji-w7-r5a-s1-internalization-v1"


def _source(name: str, features: tuple[float, ...], reward: float) -> GroundedOutcomeEvidence:
    affordance = WorldAffordance(
        affordance_id=f"affordance:{name}",
        action_kind="grounded-action",
        actor_id="workbench",
        target_id=f"target:{name}",
        features=torch.tensor(features, dtype=torch.float32),
        feature_provenance="world-state-grounding",
        grounding_lineage=(f"world-state:{name}",),
    )
    return GroundedOutcomeEvidence(
        evidence_id=f"evidence:{name}",
        outcome_id=f"outcome:{name}",
        outcome=Outcome(
            intent_id=f"intent:{name}",
            reward=reward,
            success=reward >= 0.0,
            tick=1,
        ),
        affordance=affordance,
        capability_snapshot_digest="capability-sha256:s1",
        parent_checkpoint_id="checkpoint:s1-parent",
        owner_id="taiji:workbench-outcome",
        reward_terms={"outcome": reward},
        world_digest=f"world-sha256:{name}",
    )


def evaluate_internalization() -> dict[str, object]:
    converter = InternalizationConverter(seed=17, replay_budget=8)
    ledger = InternalizationLedger(converter=converter)
    train_sources = (
        _source("left", (1.0, 0.0, 0.0), 0.8),
        _source("right", (0.0, 1.0, 0.0), 0.4),
    )
    train_results = tuple(ledger.ingest(source) for source in train_sources)
    train = tuple(result.example for result in train_results if result.example is not None)
    if len(train) != len(train_sources):
        raise RuntimeError("S1 train conversion unexpectedly rejected grounded evidence")
    holdout_result = converter.convert(_source("blend", (0.5, 0.5, 0.0), 0.6))
    holdout = holdout_result.example
    if holdout is None:
        raise RuntimeError("S1 holdout conversion unexpectedly rejected grounded evidence")

    learner = InternalizedFeatureLearner(feature_dim=3, learning_rate=0.5)
    report = learner.consolidate(
        train,
        holdout_examples=(holdout,),
        retention_examples=train,
        replay_digest=ledger.replay_digest,
        passes=8,
    )
    checkpoint = learner.checkpoint()
    restored = InternalizedFeatureLearner.from_checkpoint(checkpoint)
    checkpoint_roundtrip = content_digest(restored.checkpoint()) == content_digest(checkpoint)

    example_id = train[0].example_id
    ledger.advance_status(example_id, "shadow")
    gate = InternalizationCausalGate(
        external_sufficiency=report.holdout_loss_after < report.holdout_loss_before,
        internalization_necessity=report.holdout_internalized_lesion_loss
        > report.holdout_loss_after,
        grounding_necessity=report.holdout_grounding_lesion_loss > report.holdout_loss_after,
        checkpoint_recoverable=checkpoint_roundtrip,
        old_task_retention=report.retention_loss_after <= report.retention_loss_before + 0.05,
    )
    lifecycle = ledger.advance_status(example_id, "internalized", causal_gate=gate)
    continuation = InternalizedFeatureLearner.from_checkpoint(checkpoint)
    continuation.online_update(train[0])

    return {
        "format": REPORT_FORMAT,
        "train_examples": len(train),
        "holdout_examples": 1,
        "replay_digest": ledger.replay_digest,
        "metrics": report.to_payload(),
        "checkpoint": {
            "roundtrip": checkpoint_roundtrip,
            "fit_updates": learner.fit_updates,
            "online_updates_after_continuation": continuation.online_updates,
            "lineage_depth": len(learner.lineage),
        },
        "lifecycle": {
            "example_id": example_id,
            "status": lifecycle.status,
            "events": list(lifecycle.events),
            "causal_gate": gate.to_payload(),
        },
        "gate": {
            "passed": bool(report.passed and gate.passed and checkpoint_roundtrip),
            "criterion": "native consolidation improves unseen grounded holdout while feature, grounding, retention, and checkpoint controls remain causal",
        },
        "boundary": "S1 synthetic native canary only; no external description deletion, provider execution, structural growth, or AGI claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_r5a_s1_internalization_20260830.json",
    )
    args = parser.parse_args()
    report = evaluate_internalization()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
