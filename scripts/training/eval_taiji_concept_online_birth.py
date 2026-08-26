"""Evaluate online birth and plastic continuation of a novel sequence branch."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_concept_branch import (  # noqa: E402
    _state,
    build_branch_records,
)
from scripts.training.eval_taiji_concept_suffix import _runtime_config  # noqa: E402
from taiji import (  # noqa: E402
    ConceptFormationOrgan,
    Observation,
    Outcome,
    TSKV8Adapter,
    WorldAction,
    WorldTransition,
)

MANIFEST_FORMAT = "taiji-concept-online-birth-manifest-v1"
REPORT_FORMAT = "taiji-concept-online-birth-v1"


def _novel_transitions() -> tuple[WorldTransition, ...]:
    object_id = "online-birth-object"
    common_after = _state(1, torch.tensor([0.0, 1.0, 0.0, 0.0]), object_id)
    rescue_after = _state(2, torch.tensor([0.0, 0.0, -1.0, 0.0]), object_id)
    handoff_after = _state(3, torch.tensor([0.0, -1.0, 0.0, 0.0]), object_id)
    rescue = WorldAction(
        "online-rescue",
        "rescue",
        common_after.tick,
        target_id=object_id,
    )
    handoff = WorldAction(
        "online-handoff",
        "handoff",
        rescue_after.tick,
        target_id=object_id,
    )
    return (
        WorldTransition(
            common_after,
            rescue,
            rescue_after,
            Outcome("online-rescue", 1.5, success=True, tick=rescue_after.tick),
        ),
        WorldTransition(
            rescue_after,
            handoff,
            handoff_after,
            Outcome("online-handoff", 1.0, success=True, tick=handoff_after.tick),
        ),
    )


def evaluate() -> dict[str, object]:
    cue = torch.tensor([1.0, 0.0, 0.0, 0.0])
    records = build_branch_records(cue)
    good_records = tuple(
        record for record in records if record.episode_id.startswith("branch-episode-0-")
    )
    organ = ConceptFormationOrgan(capacity=4, trace_capacity=3, prune_threshold=0.0)
    concept = organ.consolidate(good_records, tick=3)[0]
    transitions = _novel_transitions()
    novel_trace_id = organ.grow_sequence_trace(
        concept.concept_id,
        tuple(
            (transition, error) for transition, error in zip(transitions, (0.05, 0.10), strict=True)
        ),
    )
    grown = organ.concepts[0]
    duplicate = organ.grow_sequence_trace(
        grown.concept_id,
        tuple(
            (transition, error) for transition, error in zip(transitions, (0.05, 0.10), strict=True)
        ),
    )
    grown = organ.concepts[0]
    novel_trace = next(trace for trace in grown.sequence_traces if trace.trace_id == novel_trace_id)
    affinity_before_feedback = organ.suffix_sequence_affinity(
        grown,
        ("rescue", "handoff"),
        current_state=transitions[0].before,
    )
    old_credit = novel_trace.step_credit[0]
    updates = organ.update_sequence_trace(
        "rescue",
        before_state=transitions[0].before,
        after_state=transitions[0].after,
        outcome=Outcome(
            "online-rescue-feedback",
            -1.0,
            success=False,
            tick=transitions[0].after.tick,
        ),
        prediction_error=0.90,
    )
    after_feedback = organ.concepts[0]
    updated_trace = next(
        trace for trace in after_feedback.sequence_traces if trace.trace_id == novel_trace_id
    )
    restored = ConceptFormationOrgan.from_checkpoint(organ.checkpoint())
    restored_trace = next(
        trace for trace in restored.concepts[0].sequence_traces if trace.trace_id == novel_trace_id
    )
    runtime = TSKV8Adapter(_runtime_config(), episode_id="online-birth-runtime")
    runtime.concept_formation.consolidate(good_records, tick=3)
    runtime._cognitive_state = replace(
        runtime._cognitive_state,
        concepts=runtime.concept_formation.concepts,
    )
    runtime_trace_id = runtime.grow_online_concept_branch(
        runtime.concept_formation.concepts[0].concept_id,
        tuple(
            (transition, error) for transition, error in zip(transitions, (0.05, 0.10), strict=True)
        ),
    )
    runtime_snapshot = runtime.cognitive_snapshot()
    runtime_restored = TSKV8Adapter.from_native_checkpoint(runtime.native_checkpoint())
    runtime_checkpoint_ids = tuple(
        trace.trace_id for trace in runtime_restored.concept_formation.concepts[0].sequence_traces
    )
    automatic = TSKV8Adapter(_runtime_config(), episode_id="online-birth-buffer")
    automatic.concept_formation.consolidate(good_records, tick=3)
    automatic.observe_event(
        Observation("text-byte", 97, timestamp=0, source="online-birth-evaluation"),
        learn=False,
        world_state=transitions[0].before,
    )
    automatic._cognitive_state = replace(
        automatic._cognitive_state,
        memory=replace(
            automatic._cognitive_state.memory,
            concept_ids=(automatic.concept_formation.concepts[0].concept_id,),
            concept_confidence=1.0,
        ),
    )
    automatic.act((10,), procedural_action_kinds=("rescue",))
    automatic.settle_action(
        1.0,
        world_state=transitions[0].after,
        success=True,
        terminal=False,
        learn=False,
    )
    mid_checkpoint = automatic.native_checkpoint()
    recovered_mid = TSKV8Adapter.from_native_checkpoint(mid_checkpoint)
    mid_buffer_steps = len(
        recovered_mid._online_concept_branches[
            recovered_mid.concept_formation.concepts[0].concept_id
        ]
    )
    recovered_mid.observe_event(
        Observation("text-byte", 98, timestamp=1, source="online-birth-evaluation"),
        learn=False,
        world_state=transitions[0].after,
    )
    recovered_mid._cognitive_state = replace(
        recovered_mid._cognitive_state,
        memory=replace(
            recovered_mid._cognitive_state.memory,
            concept_ids=(recovered_mid.concept_formation.concepts[0].concept_id,),
            concept_confidence=1.0,
        ),
    )
    recovered_mid.act((11,), procedural_action_kinds=("handoff",))
    recovered_mid.settle_action(
        1.0,
        world_state=transitions[1].after,
        success=True,
        terminal=True,
        learn=False,
    )
    automatic_concept = recovered_mid.concept_formation.concepts[0]
    automatic_trace_ids = tuple(trace.trace_id for trace in automatic_concept.sequence_traces)
    automatic_checkpoint = TSKV8Adapter.from_native_checkpoint(recovered_mid.native_checkpoint())
    automatic_checkpoint_ids = tuple(
        trace.trace_id
        for trace in automatic_checkpoint.concept_formation.concepts[0].sequence_traces
    )
    gate_passed = bool(
        novel_trace_id is not None
        and duplicate is None
        and len(concept.sequence_traces) == 1
        and len(grown.sequence_traces) == 2
        and affinity_before_feedback > 0.0
        and updates == 1
        and updated_trace.visits == novel_trace.visits + 1
        and updated_trace.step_credit[0] < old_credit
        and restored_trace.trace_id == novel_trace_id
        and restored_trace.visits == updated_trace.visits
        and restored_trace.step_credit == updated_trace.step_credit
        and runtime_trace_id is not None
        and runtime_snapshot.concepts[0].sequence_traces[-1].trace_id == runtime_trace_id
        and runtime_trace_id in runtime_checkpoint_ids
        and mid_buffer_steps == 1
        and len(automatic_trace_ids) == 2
        and novel_trace_id in automatic_trace_ids
        and automatic_checkpoint_ids == automatic_trace_ids
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "trace_count_before_birth": len(concept.sequence_traces),
            "trace_count_after_birth": len(grown.sequence_traces),
            "novel_trace_id": novel_trace_id,
            "duplicate_birth": duplicate,
            "novel_affinity_before_feedback": affinity_before_feedback,
            "feedback_updates": updates,
            "novel_visits_before_feedback": novel_trace.visits,
            "novel_visits_after_feedback": updated_trace.visits,
            "credit_before_feedback": old_credit,
            "credit_after_feedback": updated_trace.step_credit[0],
            "checkpoint_trace_id": restored_trace.trace_id,
            "checkpoint_recovery": restored_trace.visits == updated_trace.visits,
            "runtime_trace_id": runtime_trace_id,
            "runtime_checkpoint_trace_ids": list(runtime_checkpoint_ids),
            "mid_buffer_steps_after_checkpoint": mid_buffer_steps,
            "automatic_runtime_trace_ids": list(automatic_trace_ids),
            "automatic_runtime_checkpoint_trace_ids": list(automatic_checkpoint_ids),
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "a contiguous real transition chain that misses existing traces must birth one stable trace, reject duplicate birth, become retrievable immediately, accept later outcome/error plasticity, and recover through checkpoint",
        },
        "boundary": "This is a closed-world online branch-birth gate; it does not claim open-domain planning or general intelligence.",
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "online birth of a novel same-concept sequence trace",
        "novel_actions": ["rescue", "handoff"],
        "duplicate_policy": "do not create a second trace when action and initial state already match",
        "feedback": "negative rescue outcome and high prediction error lower branch credit without deleting the trace",
        "checkpoint": "trace identity, visits, and credit survive organ checkpoint",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_concept_online_birth_manifest_20260826.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_concept_online_birth_20260826.json",
    )
    args = parser.parse_args()
    report = evaluate()
    args.manifest.write_text(json.dumps(build_manifest(), indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
