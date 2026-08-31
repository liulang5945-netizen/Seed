"""Verify the fixed-capacity comparison and structural-growth preflight Gate."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_a2_world import build_corpus  # noqa: E402
from taiji import NativeFixedCapacityPreflight, WorldInterventionCorpus, WorldSchema  # noqa: E402
from taiji.contracts import WorldTransition  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402
from taiji.world_evolution import transition_to_case  # noqa: E402


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
    preflight = NativeFixedCapacityPreflight(
        schema,
        hidden_dim=32,
        capacity_limit=32,
        seeds=(11, 29, 47),
        epochs=350,
    )
    report = preflight.compare(
        train,
        holdout_transitions=holdout,
        retention_transitions=retention,
    )
    checkpoint = preflight.checkpoint()
    restored = NativeFixedCapacityPreflight.from_checkpoint(checkpoint)
    checks = {
        "multiseed_fixed_capacity": len(report.seed_results) == 3,
        "native_beats_frozen": report.native_holdout_gain > 0.0,
        "replay_only_matches_frozen": abs(
            report.mean_replay_only_holdout_error - report.mean_frozen_holdout_error
        )
        < 1e-12,
        "retention_preserved": report.maximum_retention_regression <= 0.05,
        "cross_seed_stable": report.holdout_error_std <= 0.2,
        "capacity_pressure_measured": report.capacity_pressure.pressure > 0.0,
        "growth_trigger_fail_closed": not report.trigger_decision.should_propose,
        "failure_persistence_required": report.trigger_decision.consecutive_failure_steps
        < report.trigger_decision.required_failure_steps,
        "checkpoint_roundtrip": content_digest(restored.checkpoint()) == content_digest(checkpoint),
    }
    return {
        "gate": "taiji-e3-4-capacity-preflight",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "report": report.to_payload(),
        "checkpoint_digest": checkpoint["checkpoint_digest"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_e3_4_capacity_preflight_20260901.json",
    )
    args = parser.parse_args()
    result = run_gate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
