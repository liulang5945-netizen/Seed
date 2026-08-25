from __future__ import annotations

import torch

from taiji import (
    ActionIntent,
    ContentPlan,
    ExecutiveCandidate,
    ExecutiveContext,
    ExecutiveController,
    ExecutiveTrainingExample,
    Outcome,
    TaijiConfig,
    TSKV8Adapter,
)


def _config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(24, 16),
        synapse_fan_in=6,
        motor_fan_in=8,
        seed=53,
    )


def _candidate(candidate_id: str, features: tuple[float, ...]) -> ExecutiveCandidate:
    intent = ActionIntent(
        intent_id=f"intent:{candidate_id}",
        kind=f"kind:{candidate_id}",
        parameters={"candidate": candidate_id},
        confidence=0.7,
        tick=0,
    )
    content = ContentPlan(
        content_id=f"content:{candidate_id}",
        intent_id=intent.intent_id,
        intent_kind=intent.kind,
        semantic_slots={"candidate": candidate_id},
        confidence=0.7,
        tick=0,
    )
    return ExecutiveCandidate(
        candidate_id=candidate_id,
        action_intent=intent,
        content_plan=content,
        features=features,
    )


def test_executive_learns_candidate_utility_without_semantic_mapping() -> None:
    controller = ExecutiveController()
    context = ExecutiveContext(features=torch.zeros(25), tick=4)
    preferred = _candidate("preferred", (1.0, 0.0, 0.0, 0.8, 0.1, 0.1))
    rejected = _candidate("rejected", (0.0, 1.0, 0.0, 0.2, 0.8, 0.9))

    controller.fit(
        (
            ExecutiveTrainingExample(preferred, context, 1.0),
            ExecutiveTrainingExample(rejected, context, -1.0),
        ),
        epochs=80,
        learning_rate=0.1,
    )
    decision = controller.select((rejected, preferred), context)

    assert decision.selected.candidate_id == "preferred"
    assert decision.action_intent.intent_id == preferred.action_intent.intent_id
    assert decision.content_plan.content_id == preferred.content_plan.content_id
    assert controller.training_steps == 160


def test_adapter_executive_owns_selection_feedback_and_native_checkpoint() -> None:
    adapter = TSKV8Adapter(_config())
    adapter.observe(65, learn=False)
    adapter.attach_executive(ExecutiveController())
    candidate = _candidate("runtime", (0.8, 0.6, 0.5, 0.7, 0.1, 0.2))

    decision = adapter.select_executive((candidate,), novelty=0.4)
    assert decision.content_plan.intent_id == decision.action_intent.intent_id
    assert adapter.cognitive_snapshot().action_intent == decision.action_intent

    error = adapter.record_executive_outcome(
        Outcome(
            intent_id=decision.action_intent.intent_id,
            reward=1.0,
            success=True,
            tick=adapter.tick,
        )
    )
    assert error < 0.0
    assert adapter.last_executive_prediction_error == error

    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    assert restored.last_executive_decision is not None
    assert restored.last_executive_decision.selected == decision.selected
    assert restored.last_executive_decision.scores == decision.scores
    assert restored.last_executive_decision.context.tick == decision.context.tick
    assert torch.equal(
        restored.last_executive_decision.context.features,
        decision.context.features,
    )
    assert restored.cognitive_snapshot().action_intent == decision.action_intent
