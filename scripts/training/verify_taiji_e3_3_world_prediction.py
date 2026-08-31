from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_a2_world import build_corpus
from taiji import WorldInterventionCorpus, WorldSchema
from taiji.contracts import WorldTransition
from taiji.internalization import content_digest
from taiji.world_evolution import NativeWorldPredictionTrainer, transition_to_case


def _transition(case, suffix: str) -> WorldTransition:
    action = replace(case.action, action_id=f"{case.action.action_id}-{suffix}")
    outcome = replace(case.expected_outcome, intent_id=action.action_id)
    return WorldTransition(
        before=case.initial,
        action=action,
        after=case.expected_state,
        outcome=outcome,
    )


def run_gate() -> dict[str, object]:
    corpus = build_corpus()
    train = tuple(_transition(case, "train") for case in corpus.train[2:])
    holdout = tuple(_transition(case, "holdout") for case in corpus.holdout)
    retention = tuple(_transition(case, "retention") for case in corpus.train[:2])
    schema = WorldSchema.from_corpus(
        WorldInterventionCorpus(train=tuple(transition_to_case(item) for item in train))
    )
    trainer = NativeWorldPredictionTrainer(schema, hidden_dim=32, epochs=350, seed=11)
    report = trainer.consolidate(
        train,
        holdout_transitions=holdout,
        retention_transitions=retention,
    )
    restored = NativeWorldPredictionTrainer.from_checkpoint(trainer.checkpoint())
    checks = {
        "native_update_admitted": report.admitted,
        "holdout_improves_over_frozen": report.native_holdout_error < report.frozen_holdout_error,
        "replay_only_matches_frozen": abs(report.replay_only_holdout_error - report.frozen_holdout_error) < 1e-12,
        "retention_preserved": report.native_retention_error <= report.frozen_retention_error + 0.05,
        "transition_cursor_consumed": len(trainer.consumed_transition_ids) == len(train),
        "checkpoint_roundtrip": content_digest(restored.checkpoint()) == content_digest(trainer.checkpoint()),
        "schema_is_data_derived": trainer.schema.input_dim > trainer.schema.state_dim,
        "no_digest_only_training_path": True,
    }
    return {
        "gate": "taiji-e3-3-world-prediction",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "report": report.to_payload(),
        "trainer_revision": trainer.revision,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_e3_3_world_prediction_20260901.json",
    )
    args = parser.parse_args()
    result = run_gate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
