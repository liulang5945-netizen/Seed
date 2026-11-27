"""Evaluate capacity, interference, and selective branch removal for traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_concept_branch import (  # noqa: E402
    ALT_SEQUENCE,
    GOOD_SEQUENCE,
    _rollout,
    build_branch_records,
)
from taiji import (  # noqa: E402
    ConceptFormationOrgan,
    Goal,
    GoalPlanner,
    GoalState,
    PlanningConfig,
)

MANIFEST_FORMAT = "taiji-concept-trace-capacity-manifest-v1"
REPORT_FORMAT = "taiji-concept-trace-capacity-v1"


def _select_branch(
    organ: ConceptFormationOrgan, concept, current_state
) -> tuple[float, float, str]:
    good_affinity = organ.suffix_sequence_affinity(
        concept, GOOD_SEQUENCE[1:], current_state=current_state
    )
    alt_affinity = organ.suffix_sequence_affinity(
        concept, ALT_SEQUENCE[1:], current_state=current_state
    )
    decision = GoalPlanner(PlanningConfig(concept_sequence_weight=2.0)).plan_rollouts(
        GoalState(tick=0, goals=(Goal("branch-goal", "choose the better branch", 1.0),)),
        (
            _rollout("branch-good", GOOD_SEQUENCE[1:], tick=1, prior=good_affinity),
            _rollout("branch-alt", ALT_SEQUENCE[1:], tick=1, prior=alt_affinity),
        ),
        tick=1,
    )
    return good_affinity, alt_affinity, decision.selected.rollout_id


def evaluate() -> dict[str, object]:
    cue = torch.tensor([1.0, 0.0, 0.0, 0.0])
    records = build_branch_records(cue)
    common_after = records[0].world_transition.after
    capacities: dict[str, dict[str, object]] = {}
    for capacity in (1, 2, 4):
        organ = ConceptFormationOrgan(
            capacity=4,
            trace_capacity=capacity,
            prune_threshold=0.0,
        )
        concept = organ.consolidate(records, tick=3)[0]
        good_affinity, alt_affinity, selected = _select_branch(organ, concept, common_after)
        capacities[str(capacity)] = {
            "trace_count": len(concept.sequence_traces),
            "good_affinity": good_affinity,
            "alternative_affinity": alt_affinity,
            "selected_branch": selected,
        }

    good_records = tuple(
        record for record in records if record.episode_id.startswith("branch-episode-0-")
    )
    alternative_records = tuple(
        record for record in records if record.episode_id.startswith("branch-episode-1-")
    )
    incremental = ConceptFormationOrgan(
        capacity=4,
        trace_capacity=2,
        prune_threshold=0.0,
    )
    incremental.consolidate(good_records, tick=3)
    before_add = incremental.concepts[0]
    incremental.consolidate(alternative_records, tick=3)
    after_add = incremental.concepts[0]
    good_trace = next(
        trace for trace in after_add.sequence_traces if trace.action_kinds == GOOD_SEQUENCE
    )
    alternative_trace = next(
        trace for trace in after_add.sequence_traces if trace.action_kinds == ALT_SEQUENCE
    )
    removed = incremental.lesion_sequence_trace(after_add.concept_id, (alternative_trace.trace_id,))
    after_remove = incremental.concepts[0]
    good_after_remove = next(
        trace for trace in after_remove.sequence_traces if trace.action_kinds == GOOD_SEQUENCE
    )
    restored = ConceptFormationOrgan.from_checkpoint(incremental.checkpoint())
    restored_concept = restored.concepts[0]
    restored_trace_ids = tuple(trace.trace_id for trace in restored_concept.sequence_traces)
    gate_passed = bool(
        capacities["1"]["trace_count"] == 1
        and capacities["2"]["trace_count"] == 2
        and capacities["4"]["trace_count"] == 2
        and all(item["selected_branch"] == "branch-good" for item in capacities.values())
        and capacities["1"]["alternative_affinity"] == 0.0
        and capacities["2"]["good_affinity"] > capacities["2"]["alternative_affinity"]
        and len(before_add.sequence_traces) == 1
        and len(after_add.sequence_traces) == 2
        and removed == (alternative_trace.trace_id,)
        and len(after_remove.sequence_traces) == 1
        and good_after_remove.trace_id == good_trace.trace_id
        and restored_trace_ids == (good_trace.trace_id,)
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "capacity_curve": capacities,
            "branch_count_before_add": len(before_add.sequence_traces),
            "branch_count_after_add": len(after_add.sequence_traces),
            "removed_trace_id": removed[0] if removed else None,
            "remaining_trace_ids": [trace.trace_id for trace in after_remove.sequence_traces],
            "checkpoint_remaining_trace_ids": list(restored_trace_ids),
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "trace capacity must preserve the strongest branch under interference, permit a second learned branch when capacity allows, selectively remove one branch without damaging the other, and recover the remaining branch through checkpoint",
        },
        "boundary": "This is a closed-world branch-capacity gate; it does not claim open-domain planning or general intelligence.",
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "same-concept multi-branch trace capacity and selective lesion",
        "trace_capacities": [1, 2, 4],
        "branch_add": "consolidating a second episode family adds its trace when capacity allows",
        "branch_remove": "lesion one trace_id while retaining the other trace",
        "checkpoint": "remaining trace identity survives ConceptFormationOrgan checkpoint",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_concept_trace_capacity_manifest_20260826.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_concept_trace_capacity_20260826.json",
    )
    args = parser.parse_args()
    report = evaluate()
    args.manifest.write_text(json.dumps(build_manifest(), indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
