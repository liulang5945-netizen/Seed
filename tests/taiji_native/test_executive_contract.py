from __future__ import annotations

import pytest
import torch

from taiji import (
    ActionIntent,
    AffordanceFeatureTrainingExample,
    ContentPlan,
    EnvironmentOutcome,
    ExecutiveCandidate,
    ExecutiveContext,
    ExecutiveController,
    ExecutiveTrainingExample,
    Goal,
    LearnedAffordanceFeatures,
    Observation,
    Outcome,
    TaijiConfig,
    TSKV8Adapter,
    WorldAffordance,
    WorldState,
)


def _config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(24, 16),
        synapse_fan_in=6,
        motor_fan_in=8,
        seed=53,
    )


def _candidate(
    candidate_id: str,
    features: tuple[float, ...],
    *,
    action_symbol: int | None = None,
) -> ExecutiveCandidate:
    parameters: dict[str, object] = {"candidate": candidate_id}
    if action_symbol is not None:
        parameters.update({"action_symbol": action_symbol, "available_actions": (10, 11)})
    intent = ActionIntent(
        intent_id=f"intent:{candidate_id}",
        kind=f"kind:{candidate_id}",
        parameters=parameters,
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


class ExecutiveEnvironment:
    def __init__(self) -> None:
        self.actions: list[int] = []

    def reset(self) -> tuple[int, tuple[int, ...]]:
        return 65, (10, 11)

    def step(self, action_symbol: int) -> EnvironmentOutcome:
        self.actions.append(action_symbol)
        success = action_symbol == 11
        return EnvironmentOutcome(
            sensation=43 if success else 45,
            reward=1.0 if success else -1.0,
            terminal=False,
            success=success,
        )


def test_executive_environment_loop_updates_and_replans() -> None:
    adapter = TSKV8Adapter(_config())
    adapter.observe(65, learn=False)
    adapter.attach_executive(ExecutiveController())
    failed = _candidate("failed", (0.9, 0.1, 0.1, 0.8, 0.2, 0.2), action_symbol=10)
    recovery = _candidate("recovery", (0.1, 0.9, 0.8, 0.8, 0.1, 0.1), action_symbol=11)
    environment = ExecutiveEnvironment()

    adapter.select_executive((failed, recovery))
    first = adapter.execute_executive_action(environment, learn=False)
    assert first.success is False
    assert adapter.replan_required is True
    assert adapter.last_executive_prediction_error is not None
    assert adapter.last_executive_world_action is not None
    assert adapter.last_executive_world_action.action_id == failed.action_intent.intent_id

    adapter.replan_executive_after_failure((failed, recovery))
    second = adapter.execute_executive_action(environment, learn=False)

    assert second.success is True
    assert environment.actions == [10, 11]
    assert adapter.replan_required is False
    assert adapter.last_executive_world_action is not None
    assert adapter.last_executive_world_action.action_id == recovery.action_intent.intent_id
    assert adapter._executive is not None
    assert adapter._executive.training_steps == 2

    adapter.attach_executive(None)
    with pytest.raises(RuntimeError, match="executive controller is not attached"):
        adapter.execute_executive_action(environment, learn=False)


def test_adapter_synthesizes_candidates_from_world_affordances() -> None:
    adapter = TSKV8Adapter(_config())
    adapter.observe(65, learn=False)
    adapter.set_goals((Goal("goal-1", "complete the task", priority=1.0),))
    context = torch.zeros(adapter.perception.feature_dim)
    source = LearnedAffordanceFeatures(
        input_dim=3,
        feature_dim=6,
        context_dim=adapter.perception.feature_dim,
        seed=7,
    )
    source.fit(
        (
            AffordanceFeatureTrainingExample(
                "train-open",
                "open-v1",
                "seen_open",
                torch.tensor([1.0, 0.0, 0.0]),
                1.0,
                percept_features=context,
                world_latent=context,
            ),
            AffordanceFeatureTrainingExample(
                "train-close",
                "close-v1",
                "seen_close",
                torch.tensor([0.0, 1.0, 0.0]),
                -1.0,
                percept_features=context,
                world_latent=context,
            ),
        ),
        epochs=80,
    )
    adapter.attach_affordance_features(source)
    adapter.attach_executive(ExecutiveController())
    world = WorldState(
        tick=adapter.tick + 1,
        affordances=(
            WorldAffordance(
                affordance_id="unseen.affordance.v2",
                action_kind="unseen_action",
                parameters={"action_symbol": 10},
                confidence=0.75,
                features=torch.tensor([0.8, 0.2, 0.0]),
                feature_provenance="world-organ-test",
            ),
        ),
    )
    adapter.observe_event(
        Observation(
            modality="text-byte",
            value=66,
            timestamp=world.tick,
            source="test.world",
        ),
        learn=False,
        world_state=world,
    )

    candidates = adapter.synthesize_executive_candidates()
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.provenance == "affordance-derived/learned"
    assert candidate.source_affordance_id == "unseen.affordance.v2"
    assert candidate.source_percept_id is not None
    assert candidate.action_intent.kind == "unseen_action"
    assert candidate.action_intent.source_goal_id == "goal-1"
    assert candidate.action_intent.parameters["action_symbol"] == 10

    decision = adapter.select_executive()
    assert decision.selected.candidate_id == candidate.candidate_id
    assert decision.content_plan.provenance == "affordance-derived"
    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    assert restored._affordance_features is not None
    restored_state = restored.cognitive_snapshot()
    assert restored_state.percept is not None
    restored_world_latent = restored_state.world.latent
    if restored_world_latent.numel() == 0:
        restored_world_latent = restored_state.percept.features
    assert torch.equal(
        restored._affordance_features.features_for(
            world.affordances[0],
            percept_features=restored_state.percept.features,
            world_latent=restored_world_latent,
            world_uncertainty=restored_state.world.uncertainty,
        ),
        candidate.feature_tensor(),
    )


def test_affordance_features_transfer_to_unseen_affordance_and_action() -> None:
    source = LearnedAffordanceFeatures(input_dim=3, feature_dim=6, seed=13)
    source.fit(
        (
            AffordanceFeatureTrainingExample(
                "train-positive",
                "known-positive",
                "known_open",
                torch.tensor([1.0, 0.0, 0.0]),
                1.0,
            ),
            AffordanceFeatureTrainingExample(
                "train-negative",
                "known-negative",
                "known_close",
                torch.tensor([0.0, 1.0, 0.0]),
                -1.0,
            ),
            AffordanceFeatureTrainingExample(
                "train-neutral",
                "known-neutral",
                "known_hold",
                torch.tensor([0.0, 0.0, 1.0]),
                0.0,
            ),
        ),
        epochs=240,
    )

    unseen_positive = WorldAffordance(
        affordance_id="never-seen-affordance",
        action_kind="never-seen-action",
        features=torch.tensor([0.8, 0.2, 0.0]),
    )
    unseen_negative = WorldAffordance(
        affordance_id="never-seen-affordance-2",
        action_kind="never-seen-action-2",
        features=torch.tensor([0.2, 0.8, 0.0]),
    )

    assert source.predict_affordance_reward(unseen_positive) > source.predict_affordance_reward(
        unseen_negative
    )
    assert not torch.equal(
        source.features_for(unseen_positive),
        source.features_for(unseen_negative),
    )


def test_affordance_features_learn_from_environment_outcome_and_lesion() -> None:
    adapter = TSKV8Adapter(_config())
    adapter.observe(65, learn=False)
    adapter.set_goals((Goal("goal-1", "complete the task", priority=1.0),))
    source = LearnedAffordanceFeatures(
        input_dim=3,
        feature_dim=6,
        context_dim=adapter.perception.feature_dim,
        seed=19,
    )
    adapter.attach_affordance_features(source)
    adapter.attach_executive(ExecutiveController())
    world = WorldState(
        tick=adapter.tick + 1,
        affordances=(
            WorldAffordance(
                affordance_id="runtime-affordance",
                action_kind="runtime_action",
                parameters={"action_symbol": 10, "available_actions": (10, 11)},
                features=torch.tensor([0.7, 0.3, 0.0]),
            ),
        ),
    )
    adapter.observe_event(
        Observation(
            modality="text-byte",
            value=66,
            timestamp=world.tick,
            source="test.runtime-world",
        ),
        learn=False,
        world_state=world,
    )

    adapter.select_executive()
    adapter.execute_executive_action(ExecutiveEnvironment(), learn=True)

    assert source.online_updates == 1
    assert adapter.last_affordance_prediction_error is not None
    restored = TSKV8Adapter.from_native_checkpoint(adapter.native_checkpoint())
    assert restored._affordance_features is not None
    assert restored._affordance_features.online_updates == 1
    assert restored.last_affordance_prediction_error == adapter.last_affordance_prediction_error

    adapter.attach_affordance_features(None)
    with pytest.raises(RuntimeError, match="learned affordance feature source"):
        adapter.synthesize_executive_candidates()


def test_affordance_grounding_producer_reads_percept_and_world_context() -> None:
    source = LearnedAffordanceFeatures(input_dim=2, feature_dim=4, context_dim=3, seed=23)
    zero = torch.zeros(3)
    source.fit(
        (
            AffordanceFeatureTrainingExample(
                "context-fit",
                "context-affordance",
                "context-action",
                torch.tensor([0.5, 0.5]),
                0.25,
                percept_features=zero,
                world_latent=zero,
            ),
        ),
        epochs=8,
    )
    affordance = WorldAffordance(
        affordance_id="context-affordance",
        action_kind="unseen-context-action",
        features=torch.tensor([0.5, 0.5]),
    )

    base = source.features_for(
        affordance,
        percept_features=zero,
        world_latent=zero,
        world_uncertainty=0.0,
    )
    shifted = source.features_for(
        affordance,
        percept_features=torch.ones(3),
        world_latent=torch.full((3,), 0.5),
        world_uncertainty=0.8,
    )

    assert not torch.equal(base, shifted)


def test_contextual_affordance_features_transfer_compositionally_to_holdout() -> None:
    source = LearnedAffordanceFeatures(input_dim=2, feature_dim=4, context_dim=2, seed=31)
    zero = torch.zeros(2)
    source.fit(
        (
            AffordanceFeatureTrainingExample(
                "holdout-fit-positive",
                "known-positive",
                "known-open",
                torch.tensor([1.0, 0.0]),
                1.0,
                percept_features=zero,
                world_latent=zero,
            ),
            AffordanceFeatureTrainingExample(
                "holdout-fit-negative",
                "known-negative",
                "known-close",
                torch.tensor([0.0, 1.0]),
                -1.0,
                percept_features=zero,
                world_latent=zero,
            ),
            AffordanceFeatureTrainingExample(
                "holdout-fit-neutral",
                "known-neutral",
                "known-hold",
                torch.tensor([0.0, 0.0]),
                0.0,
                percept_features=zero,
                world_latent=zero,
            ),
        ),
        epochs=240,
    )
    unseen_positive = WorldAffordance(
        affordance_id="holdout-unseen-positive",
        action_kind="unseen-open-variant",
        features=torch.tensor([0.8, 0.2]),
    )
    unseen_negative = WorldAffordance(
        affordance_id="holdout-unseen-negative",
        action_kind="unseen-close-variant",
        features=torch.tensor([0.2, 0.8]),
    )

    positive = source.predict_affordance_reward(
        unseen_positive,
        percept_features=zero,
        world_latent=zero,
    )
    negative = source.predict_affordance_reward(
        unseen_negative,
        percept_features=zero,
        world_latent=zero,
    )
    assert positive > negative
