"""Compatibility adapter exposing TSK-v8 through Taiji v1 contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from .affordance import LearnedAffordanceFeatures, WorldAffordanceGroundingProducer
from .content_selection import (
    ContentCandidate,
    ContentSelectionContext,
    ContentSelectionDecision,
    ContentSelector,
)
from .contracts import (
    CONTRACT_FORMAT,
    ActionIntent,
    CognitiveState,
    DevelopmentState,
    EpisodicMemoryRecord,
    Goal,
    GoalState,
    HomeostaticState,
    LearningState,
    NativeCheckpoint,
    Observation,
    Outcome,
    PlanCandidate,
    PlanState,
    SelfState,
    WorkingMemoryItem,
    WorkspaceCandidate,
    WorkspaceSelection,
    WorkspaceState,
    WorldAction,
    WorldAffordance,
    WorldCalibrationTrace,
    WorldPredictionRecord,
    WorldState,
    WorldTransition,
)
from .contracts import MemoryState as NativeMemoryState
from .environment import EnvironmentOutcome, TaijiEnvironment, TaijiToolEnvironment
from .episodic_memory import EpisodicMemoryStore
from .executive import (
    ExecutiveCandidate,
    ExecutiveContext,
    ExecutiveController,
    ExecutiveDecision,
)
from .generation import (
    ContentPlan,
    ExpressionPlan,
    GenerationController,
    GenerationTrace,
    ToolCall,
)
from .homeostasis import HomeostaticController, HomeostaticDrive
from .input_boundary import InputFrame, InputTrace
from .language_organ import (
    LanguageBackendRegistry,
    LanguageEmission,
    LanguageOrgan,
    LanguageProviderArtifact,
    LanguageValidation,
    StructuredTextLanguageOrgan,
)
from .model import Taiji
from .perception import LearnedPerception
from .planning import (
    GoalPlanner,
    ImaginedRollout,
    PlanningCandidate,
    PlanningDecision,
    RolloutDecision,
)
from .procedural_memory import ProceduralMemoryLearner
from .semantic_memory import SemanticMemoryLearner
from .state import TaijiDecision, TaijiOutcome, TaijiStep
from .workspace import WorkspaceRouter
from .world_learning import WorldDynamicsLearner, WorldSchema


@dataclass(frozen=True)
class _PendingExecutiveCredit:
    """Keep an executive decision's causal context across a replan."""

    decision: ExecutiveDecision
    affordance: WorldAffordance | None
    percept_features: torch.Tensor | None
    world_latent: torch.Tensor | None
    world_uncertainty: float
    learn: bool


class TSKV8Adapter(Taiji):
    """Keep the TSK-v8 API while making v1 ownership explicit.

    This subclass is intentional: old callers still see a ``Taiji`` and old
    ``taiji-native-v8`` checkpoints remain readable.  New callers can use
    ``native_checkpoint`` and ``cognitive_snapshot`` without treating the
    kernel's byte prediction state as the complete v1 cognitive state.
    """

    ADAPTER_NAME = "tsk-v8"
    NATIVE_CHECKPOINT_FORMAT = CONTRACT_FORMAT
    SUPPORTED_INPUT_MODALITIES = frozenset({"text", "text-utf8", "text-byte"})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.perception = LearnedPerception(self.config, device=self.device)
        self._cognitive_state = self._empty_cognitive_state(self._state.episode_id)
        self._world_dynamics: WorldDynamicsLearner | None = None
        self._workspace_router: WorkspaceRouter | None = None
        self._episodic_memory: EpisodicMemoryStore | None = None
        self._semantic_memory: SemanticMemoryLearner | None = None
        self._procedural_memory: ProceduralMemoryLearner | None = None
        self._homeostatic_controller: HomeostaticController | None = None
        self._goal_planner: GoalPlanner | None = None
        self._affordance_features: LearnedAffordanceFeatures | None = None
        self._affordance_grounding: WorldAffordanceGroundingProducer | None = None
        self._executive: ExecutiveController | None = None
        self._last_executive_decision: ExecutiveDecision | None = None
        self._last_executive_prediction_error: float | None = None
        self._last_delayed_executive_prediction_error: float | None = None
        self._last_affordance_prediction_error: float | None = None
        self._last_executive_world_action: WorldAction | None = None
        self._pending_executive_credit: _PendingExecutiveCredit | None = None
        self._planned_rollout: ImaginedRollout | None = None
        self._replan_required = False
        self._last_rollout_prediction_error: float | None = None
        self._last_rollout_calibrated_confidence: float | None = None
        self._generation_controller: GenerationController | None = None
        self._last_generation_trace: GenerationTrace | None = None
        self._content_selector: ContentSelector | None = None
        self._last_content_selection: ContentSelectionDecision | None = None
        self._last_content_prediction_error: float | None = None
        self._content_feedback_applied = False
        self._language_backend_registry = LanguageBackendRegistry.default()
        self._language_provider_artifact: LanguageProviderArtifact | None = None
        self._language_organ: LanguageOrgan | None = None
        self._last_language_emission: LanguageEmission | None = None
        self._language_fallback_count = 0
        self._language_fallback_requires_replan = False

    def _empty_cognitive_state(self, episode_id: str) -> CognitiveState:
        empty = torch.empty(0, device=self.device)
        return CognitiveState(
            episode_id=episode_id,
            tick=0,
            observation=None,
            percept=None,
            workspace=WorkspaceState(tick=0, broadcast=empty),
            world=WorldState(tick=0, latent=empty, uncertainty=1.0),
            memory=NativeMemoryState(
                tick=0,
                semantic_context=empty,
                procedural_context=empty,
            ),
            goals=GoalState(tick=0),
            plan=PlanState(tick=0),
            self_state=SelfState(tick=0),
            homeostasis=HomeostaticState(tick=0),
            development=DevelopmentState(tick=0),
            learning=LearningState(tick=0),
        )

    @property
    def architecture_name(self) -> str:
        return "Taiji Native Architecture v1 via TSK-v8"

    def cognitive_snapshot(self) -> CognitiveState:
        """Return a detached contract snapshot owned by Taiji."""

        return CognitiveState.from_payload(self._cognitive_state.to_payload(), device=self.device)

    def attach_world_dynamics(self, learner: WorldDynamicsLearner | None) -> None:
        """Attach a Taiji-owned predictor used for runtime intervention scoring."""

        if learner is not None and not isinstance(learner, WorldDynamicsLearner):
            raise TypeError("learner must be a WorldDynamicsLearner or None")
        self._world_dynamics = learner

    def attach_workspace_router(self, router: WorkspaceRouter | None) -> None:
        """Attach the capacity-limited candidate router used by runtime cognition."""

        if router is not None and not isinstance(router, WorkspaceRouter):
            raise TypeError("router must be a WorkspaceRouter or None")
        self._workspace_router = router

    def attach_episodic_memory(self, store: EpisodicMemoryStore | None) -> None:
        """Attach Taiji-owned working/episodic memory for the v1 runtime."""

        if store is not None and not isinstance(store, EpisodicMemoryStore):
            raise TypeError("store must be an EpisodicMemoryStore or None")
        self._episodic_memory = store

    def attach_semantic_memory(self, learner: SemanticMemoryLearner | None) -> None:
        """Attach the slow semantic learner fed by Taiji episodic outcomes."""

        if learner is not None and not isinstance(learner, SemanticMemoryLearner):
            raise TypeError("learner must be a SemanticMemoryLearner or None")
        if learner is not None and learner.cue_dim != self.perception.feature_dim:
            raise ValueError("semantic learner cue_dim must match the perception feature dimension")
        self._semantic_memory = learner

    def consolidate_semantic_memory(
        self, *, epochs: int = 300, learning_rate: float = 0.1
    ) -> float:
        """Replay Taiji-owned episodic outcomes into the attached semantic learner."""

        if self._semantic_memory is None:
            raise RuntimeError("semantic memory learner is not attached")
        if self._episodic_memory is None or self._episodic_memory.count == 0:
            raise RuntimeError("semantic consolidation requires episodic records")
        return self._semantic_memory.consolidate(
            self._episodic_memory,
            epochs=epochs,
            learning_rate=learning_rate,
        )

    def attach_procedural_memory(self, learner: ProceduralMemoryLearner | None) -> None:
        """Attach the slow procedural learner used by explicit action routing."""

        if learner is not None and not isinstance(learner, ProceduralMemoryLearner):
            raise TypeError("learner must be a ProceduralMemoryLearner or None")
        if learner is not None and learner.cue_dim != self.perception.feature_dim:
            raise ValueError("procedural learner cue_dim must match the perception feature dimension")
        self._procedural_memory = learner

    def consolidate_procedural_memory(
        self, *, epochs: int = 300, learning_rate: float = 0.1
    ) -> float:
        """Replay Taiji-owned action experiences into the procedural learner."""

        if self._procedural_memory is None:
            raise RuntimeError("procedural memory learner is not attached")
        if self._episodic_memory is None or self._episodic_memory.count == 0:
            raise RuntimeError("procedural consolidation requires episodic records")
        return self._procedural_memory.consolidate(
            self._episodic_memory,
            epochs=epochs,
            learning_rate=learning_rate,
        )

    def attach_homeostatic_controller(
        self, controller: HomeostaticController | None
    ) -> None:
        """Attach the event-driven controller for Taiji internal drives."""

        if controller is not None and not isinstance(controller, HomeostaticController):
            raise TypeError("controller must be a HomeostaticController or None")
        self._homeostatic_controller = controller

    def homeostatic_drive(self) -> HomeostaticDrive:
        if self._homeostatic_controller is None:
            raise RuntimeError("homeostatic controller is not attached")
        return self._homeostatic_controller.drive(self._cognitive_state.homeostasis)

    def homeostatic_transition(
        self,
        *,
        mode: str = "auto",
        prediction_error: float = 0.0,
        novelty: float = 0.0,
        reward: float = 0.0,
        resource_cost: float = 0.0,
    ) -> HomeostaticState:
        if self._homeostatic_controller is None:
            raise RuntimeError("homeostatic controller is not attached")
        if mode == "auto":
            mode = self._homeostatic_controller.select_mode(self._cognitive_state.homeostasis)
        state = self._homeostatic_controller.update(
            self._cognitive_state.homeostasis,
            prediction_error=prediction_error,
            novelty=novelty,
            reward=reward,
            resource_cost=resource_cost,
            mode=mode,
        )
        state = replace(state, tick=self.tick)
        self._cognitive_state = replace(self._cognitive_state, homeostasis=state)
        return state

    def attach_executive(self, controller: ExecutiveController | None) -> None:
        """Attach the learned Taiji executive over structured candidates."""

        if controller is not None and not isinstance(controller, ExecutiveController):
            raise TypeError("controller must be an ExecutiveController or None")
        self._executive = None if controller is None else controller.to(self.device)

    def attach_affordance_features(
        self, source: LearnedAffordanceFeatures | None
    ) -> None:
        """Attach the learned numeric feature source for world affordances."""

        if source is not None and not isinstance(source, LearnedAffordanceFeatures):
            raise TypeError("source must be a LearnedAffordanceFeatures or None")
        if source is not None and source.context_dim != self.perception.feature_dim:
            raise ValueError(
                "affordance feature source context_dim must match Taiji perception feature_dim"
            )
        self._affordance_features = None if source is None else source.to(self.device)
        self._affordance_grounding = (
            None
            if source is None
            else WorldAffordanceGroundingProducer(source.input_dim)
        )
        self._cognitive_state = replace(
            self._cognitive_state,
            world=self._ground_world_state(self._cognitive_state.world),
        )

    def _ground_world_state(self, world: WorldState) -> WorldState:
        if self._affordance_grounding is None or not world.affordances:
            return world
        return replace(
            world,
            affordances=tuple(
                self._affordance_grounding.ground(world, affordance)
                for affordance in world.affordances
            ),
        )

    @property
    def last_executive_decision(self) -> ExecutiveDecision | None:
        return self._last_executive_decision

    @property
    def last_executive_prediction_error(self) -> float | None:
        return self._last_executive_prediction_error

    @property
    def last_delayed_executive_prediction_error(self) -> float | None:
        return self._last_delayed_executive_prediction_error

    @property
    def last_affordance_prediction_error(self) -> float | None:
        return self._last_affordance_prediction_error

    @property
    def last_executive_world_action(self) -> WorldAction | None:
        return self._last_executive_world_action

    def select_executive(
        self,
        candidates: Sequence[ExecutiveCandidate] | None = None,
        *,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
    ) -> ExecutiveDecision:
        """Select an intent/content pair from current Taiji cognitive state."""

        if self._executive is None:
            raise RuntimeError("executive controller is not attached")
        candidates = (
            self.synthesize_executive_candidates()
            if candidates is None
            else tuple(candidates)
        )
        context = ExecutiveContext.from_state(
            self._cognitive_state,
            novelty=novelty,
            resource_budget=resource_budget,
        )
        decision = self._executive.select(candidates, context)
        plan_candidates = tuple(
            PlanCandidate(
                plan_id=candidate.candidate_id,
                action_kind=candidate.action_intent.kind,
                expected_value=float(decision.scores[candidate.candidate_id]),
                risk=1.0 - candidate.action_intent.confidence,
            )
            for candidate in candidates
        )
        self._last_executive_decision = decision
        self._last_executive_prediction_error = None
        self._last_delayed_executive_prediction_error = None
        self._last_affordance_prediction_error = None
        self._last_executive_world_action = None
        self._pending_executive_credit = None
        self._cognitive_state = replace(
            self._cognitive_state,
            plan=PlanState(
                tick=self.tick,
                candidates=plan_candidates,
                selected_plan_id=decision.selected.candidate_id,
            ),
            action_intent=decision.action_intent,
        )
        return decision

    def synthesize_executive_candidates(self) -> tuple[ExecutiveCandidate, ...]:
        """Derive structured candidates from Taiji-owned current affordances."""

        if self._cognitive_state.percept is None:
            raise RuntimeError("executive candidate synthesis requires a current perception")
        if self._affordance_features is None:
            raise RuntimeError(
                "executive candidate synthesis requires an attached learned affordance feature source"
            )
        percept_features, world_latent, world_uncertainty = self._affordance_context()
        feature_map = {
            affordance.affordance_id: self._affordance_features.features_for(
                affordance,
                percept_features=percept_features,
                world_latent=world_latent,
                world_uncertainty=world_uncertainty,
            )
            for affordance in self._cognitive_state.world.affordances
        }
        return ExecutiveCandidate.synthesize_from_state(
            self._cognitive_state,
            features_by_affordance=feature_map,
        )

    def _affordance_context(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        if self._cognitive_state.percept is None:
            raise RuntimeError("affordance context requires a current perception")
        percept_features = self._cognitive_state.percept.features
        world_latent = self._cognitive_state.world.latent
        if world_latent.numel() == 0:
            world_latent = percept_features
        return percept_features, world_latent, self._cognitive_state.world.uncertainty

    def record_executive_outcome(
        self,
        outcome: Outcome,
        *,
        learn: bool = True,
        source_affordance: WorldAffordance | None = None,
        affordance_context: tuple[torch.Tensor, torch.Tensor, float] | None = None,
    ) -> float:
        """Train executive selection from an outcome produced by an environment."""

        if self._executive is None:
            raise RuntimeError("executive controller is not attached")
        if self._last_executive_decision is None:
            raise RuntimeError("executive outcome requires a prior selection")
        if not isinstance(outcome, Outcome):
            raise TypeError("outcome must be a Taiji Outcome")
        if outcome.intent_id != self._last_executive_decision.action_intent.intent_id:
            raise ValueError("executive outcome must reference the selected ActionIntent")
        error = self._executive.update(self._last_executive_decision, outcome.reward)
        self._last_executive_prediction_error = error
        self._last_affordance_prediction_error = None
        if learn and self._affordance_features is not None:
            affordance_id = self._last_executive_decision.selected.source_affordance_id
            affordance = source_affordance
            if affordance is not None and affordance.affordance_id != affordance_id:
                raise ValueError("source_affordance must match the selected executive candidate")
            if affordance_id is not None and affordance is None:
                affordance = next(
                    (
                        item
                        for item in self._cognitive_state.world.affordances
                        if item.affordance_id == affordance_id
                    ),
                    None,
                )
            if affordance is not None:
                if affordance_context is None:
                    affordance_context = self._affordance_context()
                percept_features, world_latent, world_uncertainty = affordance_context
                self._last_affordance_prediction_error = (
                    self._affordance_features.online_update(
                        affordance,
                        outcome.reward,
                        percept_features=percept_features,
                        world_latent=world_latent,
                        world_uncertainty=world_uncertainty,
                    )
                )
        self._cognitive_state = replace(self._cognitive_state, outcome=outcome)
        return error

    def record_delayed_executive_credit(
        self,
        reward: float,
        *,
        learn: bool | None = None,
    ) -> float:
        """Credit the action that led to a later reward, even after replanning."""

        if self._executive is None:
            raise RuntimeError("executive controller is not attached")
        pending = self._pending_executive_credit
        if pending is None:
            raise RuntimeError("delayed executive credit requires an executed action")
        apply_learning = pending.learn if learn is None else bool(learn)
        error = self._executive.update(pending.decision, float(reward))
        self._last_delayed_executive_prediction_error = error
        self._last_executive_prediction_error = error
        self._last_affordance_prediction_error = None
        if apply_learning and self._affordance_features is not None and pending.affordance is not None:
            if pending.percept_features is None or pending.world_latent is None:
                raise RuntimeError("delayed affordance credit is missing its causal context")
            self._last_affordance_prediction_error = self._affordance_features.online_update(
                pending.affordance,
                float(reward),
                percept_features=pending.percept_features,
                world_latent=pending.world_latent,
                world_uncertainty=pending.world_uncertainty,
            )
        self._pending_executive_credit = None
        return error

    def execute_executive_action(
        self,
        environment: TaijiEnvironment,
        *,
        decision: ExecutiveDecision | None = None,
        action_symbol: int | None = None,
        learn: bool = True,
        learn_world: bool | None = None,
    ) -> Outcome:
        """Execute a selected executive intent through a motor environment.

        ``TaijiEnvironment`` currently exposes an integer motor channel.  The
        selected structured intent remains the owner of the action metadata;
        only its explicit ``action_symbol`` parameter crosses this terminal
        organ boundary.  The environment's returned sensation is fed back as
        the next Taiji observation.
        """

        if self._executive is None:
            raise RuntimeError("executive controller is not attached")
        if not isinstance(environment, TaijiEnvironment):
            raise TypeError("environment must implement TaijiEnvironment")
        selected = decision or self._last_executive_decision
        if selected is None:
            raise RuntimeError("executive action requires a prior selection")
        if decision is not None:
            self._last_executive_decision = decision
            self._cognitive_state = replace(
                self._cognitive_state,
                action_intent=decision.action_intent,
            )
        parameters = selected.action_intent.parameters
        selected_symbol = parameters.get("action_symbol") if action_symbol is None else action_symbol
        if isinstance(selected_symbol, bool) or not isinstance(selected_symbol, int):
            raise ValueError("executive ActionIntent requires an integer action_symbol")
        available = parameters.get("available_actions")
        if available is not None and int(selected_symbol) not in tuple(int(item) for item in available):
            raise ValueError("executive action_symbol is not in the ActionIntent available_actions")
        world_action = selected.to_world_action(
            tick=self._cognitive_state.world.tick,
            provenance="planned",
        )
        kernel_decision = super().act((int(selected_symbol),), sample=False)
        if kernel_decision.action_symbol != int(selected_symbol):
            raise RuntimeError("Taiji motor bridge did not preserve executive action_symbol")
        self._last_executive_world_action = world_action
        if self._world_dynamics is not None:
            prediction = self._world_dynamics.predict(
                self._cognitive_state.world,
                world_action,
            )
            self._cognitive_state = replace(
                self._cognitive_state,
                world_prediction=WorldPredictionRecord(
                    action=world_action,
                    predicted_state=prediction.state,
                    predicted_reward=prediction.reward,
                    predicted_success_probability=prediction.success_probability,
                    online_update_count=self._world_dynamics.online_updates,
                ),
            )
        selected_affordance = next(
            (
                item
                for item in self._cognitive_state.world.affordances
                if item.affordance_id == selected.selected.source_affordance_id
            ),
            None,
        )
        affordance_context = (
            self._affordance_context()
            if selected_affordance is not None and self._affordance_features is not None
            else None
        )
        result = environment.step(int(selected_symbol))
        if not isinstance(result, EnvironmentOutcome):
            raise TypeError("environment must return an EnvironmentOutcome")
        self.settle_action(
            result.reward,
            learn=learn,
            success=result.success,
            terminal=result.terminal,
            world_state=result.world_state,
            world_action=world_action if result.world_state is not None else None,
            learn_world=learn_world,
            provenance="experienced",
        )
        experienced = self._cognitive_state.outcome
        if experienced is None:
            raise RuntimeError("executive environment outcome was not recorded")
        transition = self._cognitive_state.world_transition
        prediction_record = self._cognitive_state.world_prediction
        self.record_executive_outcome(
            experienced,
            learn=learn,
            source_affordance=selected_affordance,
            affordance_context=affordance_context,
        )
        if not result.terminal:
            self._pending_executive_credit = _PendingExecutiveCredit(
                decision=selected,
                affordance=selected_affordance,
                percept_features=(
                    None
                    if affordance_context is None
                    else affordance_context[0].detach().clone()
                ),
                world_latent=(
                    None
                    if affordance_context is None
                    else affordance_context[1].detach().clone()
                ),
                world_uncertainty=(
                    0.0 if affordance_context is None else affordance_context[2]
                ),
                learn=learn,
            )
        self._replan_required = bool(
            not result.terminal and (result.success is False or result.reward < 0.0)
        )
        self.observe(result.sensation, learn=learn)
        if transition is not None:
            self._cognitive_state = replace(
                self._cognitive_state,
                action_intent=selected.action_intent,
                outcome=experienced,
                world_transition=transition,
                world_prediction=prediction_record,
            )
        return experienced

    def replan_executive_after_failure(
        self,
        candidates: Sequence[ExecutiveCandidate],
        *,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
    ) -> ExecutiveDecision:
        """Select an alternative executive candidate after a failed action."""

        if not self._replan_required:
            raise RuntimeError("executive replanning has not been requested")
        if self._last_executive_decision is None:
            raise RuntimeError("executive replanning requires a prior selection")
        alternatives = tuple(
            candidate
            for candidate in candidates
            if candidate.candidate_id != self._last_executive_decision.selected.candidate_id
        )
        if not alternatives:
            raise RuntimeError("executive replanning requires an alternative candidate")
        pending_credit = self._pending_executive_credit
        decision = self.select_executive(
            alternatives,
            novelty=novelty,
            resource_budget=resource_budget,
        )
        if pending_credit is not None:
            self._pending_executive_credit = pending_credit
        self._replan_required = False
        return decision

    def attach_generation_controller(
        self, controller: GenerationController | None
    ) -> None:
        """Attach the organ bridge for content and structured tool generation."""

        if controller is not None and not isinstance(controller, GenerationController):
            raise TypeError("controller must be a GenerationController or None")
        self._generation_controller = controller

    @property
    def generation_trace(self) -> GenerationTrace | None:
        return self._last_generation_trace

    def generate_tool_call(
        self,
        *,
        tool_name: str | None = None,
        channel: str | None = None,
        provenance: str = "planned",
    ) -> ToolCall:
        """Render the current Taiji action intent through the tool organ."""

        if self._generation_controller is None:
            raise RuntimeError("generation controller is not attached")
        intent = self._cognitive_state.action_intent
        if intent is None:
            raise RuntimeError("tool generation requires a pending ActionIntent")
        source_goal_id = intent.source_goal_id
        if source_goal_id is None and self._planned_rollout is not None:
            source_goal_id = self._planned_rollout.goal_id
        trace = self._generation_controller.generate_tool_call(
            intent,
            tool_name=tool_name,
            source_goal_id=source_goal_id,
            channel=channel,
            provenance=provenance,
        )
        self._last_generation_trace = trace
        return trace.tool_call

    def attach_content_selector(self, selector: ContentSelector | None) -> None:
        """Attach Taiji-owned learned selection of semantic content candidates."""

        if selector is not None and not isinstance(selector, ContentSelector):
            raise TypeError("selector must be a ContentSelector or None")
        self._content_selector = selector

    @property
    def last_content_selection(self) -> ContentSelectionDecision | None:
        return self._last_content_selection

    @property
    def last_content_prediction_error(self) -> float | None:
        return self._last_content_prediction_error

    def select_content(
        self,
        candidates: Sequence[ContentCandidate],
        *,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
    ) -> ContentSelectionDecision:
        """Select semantic content from the current Taiji goal/world state."""

        if self._content_selector is None:
            raise RuntimeError("content selector is not attached")
        context = ContentSelectionContext.from_state(
            self._cognitive_state.goals,
            self._cognitive_state.world,
            novelty=novelty,
            resource_budget=resource_budget,
        )
        decision = self._content_selector.select(tuple(candidates), context)
        self._last_content_selection = decision
        self._last_content_prediction_error = None
        self._content_feedback_applied = False
        return decision

    def replan_content_after_language_fallback(
        self,
        candidates: Sequence[ContentCandidate],
        *,
        novelty: float = 0.0,
        resource_budget: float = 1.0,
    ) -> ContentSelectionDecision:
        """Select an alternative semantic plan after unsafe text realization.

        The failed candidate is excluded for this replan attempt.  The
        language organ remains an effector: this method only chooses a new
        Taiji-owned content plan; ``express_selected_content`` performs the
        subsequent organ-specific realization.
        """

        if not self._language_fallback_requires_replan:
            raise RuntimeError("language fallback has not requested content replanning")
        if self._last_content_selection is None:
            raise RuntimeError("language fallback replanning requires a prior content selection")
        previous_id = self._last_content_selection.selected.candidate_id
        alternatives = tuple(
            candidate for candidate in candidates if candidate.candidate_id != previous_id
        )
        if not alternatives:
            raise RuntimeError("language fallback replanning requires an alternative content candidate")
        decision = self.select_content(
            alternatives,
            novelty=novelty,
            resource_budget=resource_budget,
        )
        self._replan_required = False
        self._language_fallback_requires_replan = False
        return decision

    def selected_content_plan(self) -> ContentPlan:
        if self._last_content_selection is None:
            raise RuntimeError("content selection has not been performed")
        return self._last_content_selection.selected.to_content_plan()

    def express_selected_content(
        self,
        *,
        modality: str = "tool",
        channel: str | None = None,
    ) -> ExpressionPlan:
        if self._generation_controller is None:
            raise RuntimeError("generation controller is not attached")
        return self._generation_controller.plan_expression(
            self.selected_content_plan(),
            modality=modality,
            channel=channel,
        )

    def attach_language_organ(self, organ: LanguageOrgan | None) -> None:
        """Attach a replaceable terminal text organ owned by Taiji's boundary."""

        if organ is not None:
            if (
                self._language_provider_artifact is not None
                and organ.backend_id != self._language_provider_artifact.backend_id
            ):
                raise ValueError("language organ backend does not match provider artifact")
            self._language_backend_registry.validate(organ)
        self._language_organ = organ

    def attach_language_provider_artifact(
        self, artifact: LanguageProviderArtifact | None
    ) -> None:
        """Record an externally loaded provider without importing its runtime."""

        if artifact is not None:
            if not isinstance(artifact, LanguageProviderArtifact):
                raise TypeError("artifact must be a LanguageProviderArtifact or None")
            self._language_backend_registry.get(artifact.backend_id)
            if (
                self._language_organ is not None
                and self._language_organ.backend_id != artifact.backend_id
            ):
                raise ValueError("provider artifact backend does not match language organ")
        self._language_provider_artifact = artifact

    @property
    def language_provider_artifact(self) -> LanguageProviderArtifact | None:
        return self._language_provider_artifact

    def attach_language_backend_registry(
        self, registry: LanguageBackendRegistry | None
    ) -> None:
        """Attach descriptors for allowed terminal language-organ backends."""

        selected_registry = registry or LanguageBackendRegistry.default()
        if not isinstance(selected_registry, LanguageBackendRegistry):
            raise TypeError("registry must be a LanguageBackendRegistry or None")
        if self._language_organ is not None:
            selected_registry.validate(self._language_organ)
        self._language_backend_registry = selected_registry

    @property
    def last_language_emission(self) -> LanguageEmission | None:
        return self._last_language_emission

    @property
    def last_language_validation(self) -> LanguageValidation | None:
        return None if self._last_language_emission is None else self._last_language_emission.validation

    @property
    def language_fallback_count(self) -> int:
        return self._language_fallback_count

    def _apply_content_feedback(self, reward: float) -> None:
        if (
            self._content_selector is not None
            and self._last_content_selection is not None
            and not self._content_feedback_applied
        ):
            self._last_content_prediction_error = self._content_selector.update(
                self._last_content_selection.selected,
                self._last_content_selection.context,
                reward,
            )
            self._content_feedback_applied = True

    def emit_language(
        self,
        expression: ExpressionPlan | None = None,
        *,
        channel: str = "message",
    ) -> LanguageEmission:
        """Emit text through the terminal organ without creating cognition.

        When no expression is supplied, the expression is derived from the
        already selected semantic content.  The organ cannot create content,
        goals, plans, or actions on its own.
        """

        if self._language_organ is None:
            raise RuntimeError("language organ is not attached")
        selected_expression = expression
        if selected_expression is None:
            selected_expression = self.express_selected_content(
                modality="text",
                channel=channel,
            )
        if not isinstance(selected_expression, ExpressionPlan):
            raise TypeError("language emission requires an ExpressionPlan")
        if selected_expression.modality != "text":
            raise ValueError("language emission requires a text ExpressionPlan")
        emission = self._language_organ.emit(selected_expression)
        if not isinstance(emission, LanguageEmission):
            raise TypeError("language organ must return a LanguageEmission")
        self._last_language_emission = emission
        if emission.fallback_used:
            self._language_fallback_count += 1
            self._language_fallback_requires_replan = True
            self._replan_required = True
            self._apply_content_feedback(-1.0)
        return emission

    def execute_tool_call(
        self,
        environment: TaijiToolEnvironment,
        *,
        call: ToolCall | None = None,
        learn: bool = True,
    ) -> Outcome:
        """Execute a generated call and feed its outcome back into Taiji."""

        if not isinstance(environment, TaijiToolEnvironment):
            raise TypeError("environment must implement TaijiToolEnvironment")
        selected_call = call
        if selected_call is None:
            if self._last_generation_trace is None:
                raise RuntimeError("tool execution requires a generated ToolCall")
            selected_call = self._last_generation_trace.tool_call
        intent = self._cognitive_state.action_intent
        if intent is None or intent.intent_id != selected_call.intent_id:
            raise ValueError("tool call must reference the pending ActionIntent")
        result = environment.execute_tool(
            selected_call.tool_name,
            dict(selected_call.parameters),
        )
        if not isinstance(result, EnvironmentOutcome):
            raise TypeError("tool environment must return an EnvironmentOutcome")
        taiji_outcome = self.settle_action(
            result.reward,
            learn=learn,
            success=result.success,
            terminal=result.terminal,
            provenance="experienced",
        )
        experienced = self._cognitive_state.outcome
        self.observe(result.sensation, learn=learn)
        return (
            experienced
            if experienced is not None
            else Outcome(
                intent_id=intent.intent_id,
                reward=taiji_outcome.reward,
                success=result.success,
                terminal=result.terminal,
                tick=taiji_outcome.tick,
            )
        )

    def attach_goal_planner(self, planner: GoalPlanner | None) -> None:
        """Attach the Taiji-owned planner for executable goal candidates."""

        if planner is not None and not isinstance(planner, GoalPlanner):
            raise TypeError("planner must be a GoalPlanner or None")
        self._goal_planner = planner

    def set_goals(self, goals: Sequence[Goal]) -> None:
        """Register the current goal hierarchy in the Taiji cognitive state."""

        goals = tuple(goals)
        if any(not isinstance(goal, Goal) for goal in goals):
            raise TypeError("goals must contain Goal contracts")
        self._cognitive_state = replace(
            self._cognitive_state,
            goals=GoalState(tick=self.tick, goals=goals),
        )

    def plan_actions(
        self,
        candidates: Sequence[PlanningCandidate],
        *,
        goal_id: str | None = None,
    ) -> PlanningDecision:
        """Compare executable world candidates and persist the selected plan."""

        if self._goal_planner is None:
            raise RuntimeError("goal planner is not attached")
        decision = self._goal_planner.plan(
            self._cognitive_state.goals,
            tuple(candidates),
            tick=self.tick,
            goal_id=goal_id,
        )
        self._cognitive_state = replace(
            self._cognitive_state,
            plan=decision.plan,
            goals=replace(self._cognitive_state.goals, tick=self.tick),
        )
        return decision

    def plan_rollouts(
        self,
        rollouts: Sequence[ImaginedRollout],
        *,
        goal_id: str | None = None,
    ) -> RolloutDecision:
        """Persist the selected multi-step imagined rollout for execution."""

        if self._goal_planner is None:
            raise RuntimeError("goal planner is not attached")
        decision = self._goal_planner.plan_rollouts(
            self._cognitive_state.goals,
            tuple(rollouts),
            tick=self.tick,
            goal_id=goal_id,
        )
        self._planned_rollout = decision.selected
        self._replan_required = False
        self._language_fallback_requires_replan = False
        self._last_rollout_prediction_error = None
        self._last_rollout_calibrated_confidence = None
        self._cognitive_state = replace(
            self._cognitive_state,
            plan=decision.plan,
            goals=replace(self._cognitive_state.goals, tick=self.tick),
        )
        return decision

    @property
    def replan_required(self) -> bool:
        return self._replan_required

    @property
    def last_rollout_prediction_error(self) -> float | None:
        return self._last_rollout_prediction_error

    @property
    def last_rollout_calibrated_confidence(self) -> float | None:
        return self._last_rollout_calibrated_confidence

    def reset_dynamics(self, *, episode_id: str | None = None) -> None:
        super().reset_dynamics(episode_id=episode_id)
        self.perception.reset_dynamics()
        self._cognitive_state = self._empty_cognitive_state(self._state.episode_id)
        self._planned_rollout = None
        self._replan_required = False
        self._last_rollout_prediction_error = None
        self._last_rollout_calibrated_confidence = None
        self._last_executive_decision = None
        self._last_executive_prediction_error = None
        self._last_affordance_prediction_error = None
        self._last_executive_world_action = None
        self._last_generation_trace = None
        self._last_language_emission = None
        self._language_fallback_count = 0
        self._language_fallback_requires_replan = False
        self._last_content_selection = None
        self._last_content_prediction_error = None
        self._content_feedback_applied = False

    def observe(self, symbol: int, *args: Any, **kwargs: Any) -> TaijiStep:
        workspace_candidates: Sequence[WorkspaceCandidate] | None = kwargs.pop(
            "workspace_candidates", None
        )
        workspace_mode = str(kwargs.pop("workspace_mode", "learned"))
        step = super().observe(symbol, *args, **kwargs)
        observation = Observation(
            modality="text-byte",
            value=int(symbol),
            timestamp=self.tick,
            source="byte-sensor",
            provenance="external",
        )
        percept = self.perception.observe(
            int(symbol),
            tick=self.tick,
            stream_id=self._state.episode_id,
            learn=bool(kwargs.get("learn", True)),
        )
        features = percept.features.detach().clone()
        recall = step.memory_recall
        previous = self._cognitive_state
        homeostasis = previous.homeostasis
        if self._homeostatic_controller is not None:
            homeostasis = self._homeostatic_controller.update(
                previous.homeostasis,
                prediction_error=percept.prediction_error,
                novelty=max(percept.prediction_error, 1.0 - recall.confidence),
                resource_cost=0.05,
                mode="wake",
            )
        selection: WorkspaceSelection | None = None
        candidates: tuple[WorkspaceCandidate, ...] = ()
        if self._workspace_router is not None and (
            workspace_candidates is not None or workspace_mode != "learned"
        ):
            candidates = tuple(workspace_candidates or ())
            selection = self._workspace_router.route(
                candidates,
                tick=self.tick,
                mode=workspace_mode,
            )
            workspace = WorkspaceState(
                tick=self.tick,
                focus=selection.selected_ids,
                broadcast=selection.broadcast,
                capacity=selection.capacity,
                candidates=candidates,
                selection=selection,
            )
        else:
            workspace = WorkspaceState(
                tick=self.tick,
                focus=("predictive-context",),
                broadcast=features,
                capacity=1,
            )
        world = WorldState(
            tick=self.tick,
            latent=features,
            entities=previous.world.entities,
            relations=previous.world.relations,
            objects=previous.world.objects,
            events=previous.world.events,
            affordances=previous.world.affordances,
            uncertainty=max(0.0, min(1.0, 1.0 - recall.confidence)),
        )
        memory = NativeMemoryState(
            tick=self.tick,
            episodic_confidence=max(0.0, min(1.0, recall.confidence)),
            semantic_context=recall.cortical_feedback.detach().clone(),
            procedural_context=recall.action_evidence.detach().clone(),
            working_ids=(f"{self._state.episode_id}:working:{self.tick}",),
            working_items=(
                WorkingMemoryItem(
                    item_id=f"{self._state.episode_id}:working:{self.tick}",
                    value=features,
                    salience=max(0.0, min(1.0, recall.confidence)),
                ),
            ),
            episodic_ids=(
                ()
                if self._episodic_memory is None
                else tuple(
                    hit.record.memory_id
                    for hit in self._episodic_memory.retrieve(features, limit=3)
                )
            ),
        )
        self._cognitive_state = replace(
            previous,
            tick=self.tick,
            episode_id=self._state.episode_id,
            observation=observation,
            percept=percept,
            workspace=workspace,
            world=world,
            memory=memory,
            goals=replace(previous.goals, tick=self.tick),
            plan=replace(previous.plan, tick=self.tick),
            self_state=replace(
                previous.self_state,
                tick=self.tick,
                confidence=max(0.0, min(1.0, recall.confidence)),
            ),
            homeostasis=replace(homeostasis, tick=self.tick),
            development=replace(previous.development, tick=self.tick),
            learning=replace(
                previous.learning,
                tick=self.tick,
                local_updates=previous.learning.local_updates + int(kwargs.get("learn", True)),
            ),
            action_intent=None,
            outcome=None,
            world_transition=None,
            world_prediction=None,
        )
        return step

    def observe_event(
        self,
        observation: Observation,
        *,
        learn: bool = True,
        learn_motor: bool | None = None,
        use_memory: bool = True,
        world_state: WorldState | None = None,
        workspace_candidates: Sequence[WorkspaceCandidate] | None = None,
        workspace_mode: str = "learned",
    ) -> TaijiStep:
        """Ingest a v1 ``Observation`` while retaining the kernel API."""

        if observation.modality != "text-byte" or not isinstance(observation.value, int):
            raise ValueError("TSK-v8 adapter accepts only integer text-byte observations")
        step = self.observe(
            observation.value,
            learn=learn,
            learn_motor=learn_motor,
            use_memory=use_memory,
            workspace_candidates=workspace_candidates,
            workspace_mode=workspace_mode,
        )
        self._cognitive_state = replace(self._cognitive_state, observation=observation)
        if world_state is not None:
            if not isinstance(world_state, WorldState):
                raise TypeError("world_state must be a Taiji WorldState")
            if world_state.tick != self.tick:
                raise ValueError("observed world_state must match the adapter tick")
            self._cognitive_state = replace(
                self._cognitive_state,
                world=self._ground_world_state(world_state),
            )
        return step

    def ingest_input(
        self,
        frame: InputFrame,
        *,
        learn: bool = True,
        learn_motor: bool | None = None,
        use_memory: bool = True,
        workspace_candidates: Sequence[WorkspaceCandidate] | None = None,
        workspace_mode: str = "learned",
    ) -> InputTrace:
        """Route one client frame through Taiji-owned perception.

        The adapter currently exposes byte-level text perception.  The frame
        keeps the product transport metadata intact, while each byte becomes
        a versioned ``Observation`` and learned ``PerceptEvent``.  No action
        intent is inferred at this boundary; an executive must earn that
        decision from the resulting cognitive state.
        """

        if not isinstance(frame, InputFrame):
            raise TypeError("frame must be a Taiji InputFrame")
        if frame.modality not in self.SUPPORTED_INPUT_MODALITIES:
            supported = ", ".join(sorted(self.SUPPORTED_INPUT_MODALITIES))
            raise ValueError(f"unsupported input modality {frame.modality!r}; supported: {supported}")

        observations: list[Observation] = []
        percepts: list[Any] = []
        for symbol in frame.payload:
            observation = Observation(
                modality="text-byte",
                value=int(symbol),
                timestamp=frame.timestamp,
                source=frame.source,
                provenance=frame.provenance,
                confidence=frame.confidence,
            )
            self.observe_event(
                observation,
                learn=learn,
                learn_motor=learn_motor,
                use_memory=use_memory,
                workspace_candidates=workspace_candidates,
                workspace_mode=workspace_mode,
            )
            current = self.cognitive_snapshot()
            if current.observation is None or current.percept is None:
                raise RuntimeError("Taiji perception did not emit a complete input trace")
            observations.append(current.observation)
            percepts.append(current.percept)

        current = self.cognitive_snapshot()
        return InputTrace(
            input_id=frame.input_id,
            modality=frame.modality,
            observations=tuple(observations),
            percepts=tuple(percepts),
            action_intent=current.action_intent,
        )

    @torch.no_grad()
    def generate_input(
        self,
        frame: InputFrame,
        length: int,
        *,
        stop_at_boundary: bool = False,
        sample: bool = False,
        reset: bool = True,
    ) -> bytes:
        """Generate from a validated Taiji input frame.

        Generation remains a byte-level effector path for compatibility.  It
        does not manufacture an ``ActionIntent`` or semantic ``ContentPlan``
        from the client text.
        """

        if not isinstance(frame, InputFrame):
            raise TypeError("frame must be a Taiji InputFrame")
        if frame.modality not in self.SUPPORTED_INPUT_MODALITIES:
            supported = ", ".join(sorted(self.SUPPORTED_INPUT_MODALITIES))
            raise ValueError(f"unsupported input modality {frame.modality!r}; supported: {supported}")
        return self.generate(
            frame.payload,
            length,
            stop_at_boundary=stop_at_boundary,
            sample=sample,
            reset=reset,
        )

    def act(self, available_actions: Any, *args: Any, **kwargs: Any) -> TaijiDecision:
        supplied_world_action = kwargs.pop("world_action", None)
        procedural_action_kinds = kwargs.pop("procedural_action_kinds", None)
        use_procedural = bool(kwargs.pop("use_procedural", False))
        use_plan = bool(kwargs.pop("use_plan", False))
        planned_kind = None
        if use_plan:
            if use_procedural:
                raise ValueError("use_plan and use_procedural cannot be enabled together")
            selected_plan_id = self._cognitive_state.plan.selected_plan_id
            if selected_plan_id is None:
                raise RuntimeError("planned action routing requires a selected plan")
            selected_plan = next(
                (
                    candidate
                    for candidate in self._cognitive_state.plan.candidates
                    if candidate.plan_id == selected_plan_id
                ),
                None,
            )
            if selected_plan is None:
                raise RuntimeError("selected plan is missing from the current plan state")
            planned_kind = selected_plan.action_kind
        action_kinds = (
            None
            if procedural_action_kinds is None
            else tuple(str(kind) for kind in procedural_action_kinds)
        )
        if action_kinds is not None:
            action_symbols = tuple(int(action) for action in available_actions)
            if len(action_kinds) != len(action_symbols):
                raise ValueError("procedural_action_kinds must align with available_actions")
            if len(set(action_kinds)) != len(action_kinds):
                raise ValueError("procedural_action_kinds must be unique")
        if use_procedural:
            if self._procedural_memory is None:
                raise RuntimeError("procedural action routing requires an attached learner")
            if action_kinds is None:
                raise ValueError("procedural action routing requires procedural_action_kinds")
            if self._cognitive_state.percept is None:
                raise RuntimeError("procedural action routing requires a current perception")
            predicted_kind = self._procedural_memory.predict(
                self._cognitive_state.percept.features
            )
            if predicted_kind not in action_kinds:
                raise ValueError("procedural learner predicted an unavailable action kind")
        else:
            predicted_kind = None
        if planned_kind is not None:
            if action_kinds is None:
                raise ValueError("planned action routing requires procedural_action_kinds")
            if planned_kind not in action_kinds:
                raise ValueError("selected plan is not available in the current affordances")
        decision = super().act(available_actions, *args, **kwargs)
        route_kind = predicted_kind if use_procedural else planned_kind
        if route_kind is not None:
            assert action_kinds is not None
            selected_action = tuple(action_kinds).index(route_kind)
            selected_symbol = tuple(int(action) for action in available_actions)[selected_action]
            pending = self._state.pending_action
            if pending is None:
                raise RuntimeError("kernel did not preserve a pending action")
            self._state.pending_action = replace(pending, action_symbol=selected_symbol)
            decision = replace(decision, action_symbol=selected_symbol)
        intent_parameters = {
            **(
                {}
                if supplied_world_action is None
                else dict(supplied_world_action.parameters)
            ),
            "action_symbol": decision.action_symbol,
            "available_actions": decision.available_actions,
        }
        intent = ActionIntent(
            intent_id=f"{self._state.episode_id}:intent:{decision.tick}",
            kind=(
                "byte-motor"
                if action_kinds is None
                else action_kinds[decision.available_actions.index(decision.action_symbol)]
            ),
            parameters=intent_parameters,
            expected_outcome="environment-feedback",
            confidence=float(decision.policy_probabilities[decision.action_symbol]),
            tick=decision.tick,
        )
        candidates = tuple(
            PlanCandidate(
                plan_id=f"{intent.intent_id}:candidate:{index}",
                action_kind=("byte-motor" if action_kinds is None else action_kinds[index]),
                expected_value=float(decision.policy_probabilities[action]),
            )
            for index, action in enumerate(decision.available_actions)
        )
        selected_index = decision.available_actions.index(decision.action_symbol)
        plan_state = (
            replace(self._cognitive_state.plan, tick=self.tick)
            if use_plan
            else PlanState(
                tick=self.tick,
                candidates=candidates,
                selected_plan_id=(candidates[selected_index].plan_id if candidates else None),
            )
        )
        world_action = None
        if supplied_world_action is not None:
            if not isinstance(supplied_world_action, WorldAction):
                raise TypeError("world_action must be a Taiji WorldAction")
            if supplied_world_action.tick != self._cognitive_state.world.tick:
                raise ValueError("world_action must act at the current world tick")
            world_action = WorldAction(
                action_id=intent.intent_id,
                kind=supplied_world_action.kind,
                tick=supplied_world_action.tick,
                actor_id=supplied_world_action.actor_id,
                target_id=supplied_world_action.target_id,
                parameters=supplied_world_action.parameters,
                provenance=supplied_world_action.provenance,
            )
        elif self._world_dynamics is not None:
            world_action = WorldAction(
                action_id=intent.intent_id,
                kind=intent.kind,
                tick=self._cognitive_state.world.tick,
                parameters=intent.parameters,
            )
        world_prediction = None
        if self._world_dynamics is not None and world_action is not None:
            prediction = self._world_dynamics.predict(self._cognitive_state.world, world_action)
            world_prediction = WorldPredictionRecord(
                action=world_action,
                predicted_state=prediction.state,
                predicted_reward=prediction.reward,
                predicted_success_probability=prediction.success_probability,
                online_update_count=self._world_dynamics.online_updates,
            )
        self._cognitive_state = replace(
            self._cognitive_state,
            plan=plan_state,
            action_intent=intent,
            world_prediction=world_prediction,
        )
        return decision

    def settle_action(self, reward: float, *args: Any, **kwargs: Any) -> TaijiOutcome:
        intent = self._cognitive_state.action_intent
        world_state = kwargs.pop("world_state", None)
        world_action = kwargs.pop("world_action", None)
        success = kwargs.pop("success", None)
        terminal = bool(kwargs.pop("terminal", False))
        learn_world = kwargs.pop("learn_world", None)
        world_learning_rate = float(kwargs.pop("world_learning_rate", 0.005))
        world_learning_repeats = int(kwargs.pop("world_learning_repeats", 1))
        intent_id = intent.intent_id if intent is not None else f"kernel-action:{self.tick}"
        before = self._cognitive_state.world
        if world_state is not None:
            if not isinstance(world_state, WorldState):
                raise TypeError("world_state must be a Taiji WorldState")
            if world_state.tick != before.tick + 1:
                raise ValueError("world_state must advance the cognitive world tick by one")
            world_state = self._ground_world_state(world_state)
            if world_action is not None:
                if not isinstance(world_action, WorldAction):
                    raise TypeError("world_action must be a Taiji WorldAction")
                if world_action.action_id != intent_id:
                    raise ValueError("world_action must reference the pending ActionIntent")
        result = super().settle_action(reward, *args, **kwargs)
        outcome = Outcome(
            intent_id=intent_id,
            reward=float(result.reward),
            success=(float(result.reward) > 0.0 if success is None else bool(success)),
            terminal=terminal,
            provenance=str(kwargs.get("provenance", "experienced")),
            tick=(self.tick if world_state is None else int(world_state.tick)),
        )
        self._apply_content_feedback(outcome.reward)
        transition = None
        prediction_record = self._cognitive_state.world_prediction
        calibration_trace = self._cognitive_state.world_calibration_trace
        if world_state is not None:
            if world_action is None:
                if intent is None:
                    raise RuntimeError("world_state requires a pending ActionIntent")
                world_action = (
                    prediction_record.action
                    if prediction_record is not None
                    else WorldAction(
                    action_id=intent_id,
                    kind=intent.kind,
                    tick=before.tick,
                    parameters=intent.parameters,
                    provenance=str(kwargs.get("provenance", "experienced")),
                    )
                    )
            if prediction_record is None and self._world_dynamics is not None:
                prediction = self._world_dynamics.predict(before, world_action)
                prediction_record = WorldPredictionRecord(
                    action=world_action,
                    predicted_state=prediction.state,
                    predicted_reward=prediction.reward,
                    predicted_success_probability=prediction.success_probability,
                    online_update_count=self._world_dynamics.online_updates,
                )
            transition = WorldTransition(
                before=before,
                action=world_action,
                after=world_state,
                outcome=outcome,
            )
            if prediction_record is not None and self._world_dynamics is not None:
                online_update_count_before = self._world_dynamics.online_updates
                predicted = self._world_dynamics.schema.state_values(
                    prediction_record.predicted_state
                )
                actual = self._world_dynamics.schema.state_values(world_state)
                prediction_record = replace(
                    prediction_record,
                    state_error=float(torch.mean((predicted - actual) ** 2)),
                    reward_error=(prediction_record.predicted_reward - outcome.reward) ** 2,
                )
                if learn_world is None:
                    learn_world = bool(kwargs.get("learn", True))
                if learn_world:
                    self._world_dynamics.online_update(
                        transition,
                        learning_rate=world_learning_rate,
                        repeats=world_learning_repeats,
                    )
                    prediction_record = replace(
                        prediction_record,
                        online_update_count=self._world_dynamics.online_updates,
                    )
                calibration_trace = (
                    *calibration_trace,
                    WorldCalibrationTrace(
                        transition=transition,
                        prediction=prediction_record,
                        calibration_applied=bool(learn_world),
                        online_update_count_before=online_update_count_before,
                        online_update_count_after=self._world_dynamics.online_updates,
                    ),
                )[-self.config.world_calibration_history_limit :]
        memory = self._cognitive_state.memory
        if self._planned_rollout is not None and self._goal_planner is not None:
            rollout = self._planned_rollout
            self._last_rollout_prediction_error = self._goal_planner.rollout_prediction_error(
                rollout, outcome
            )
            self._replan_required = self._language_fallback_requires_replan or (
                self._last_rollout_prediction_error
                > self._goal_planner.config.replan_error_threshold
            )
            self._last_rollout_calibrated_confidence = self._goal_planner.record_rollout_outcome(
                rollout, outcome
            )
            self._planned_rollout = None
        goals = self._cognitive_state.goals
        if self._goal_planner is not None:
            goals = self._goal_planner.apply_outcome(goals, outcome)
        homeostasis = self._cognitive_state.homeostasis
        if self._homeostatic_controller is not None:
            homeostasis = self._homeostatic_controller.update(
                homeostasis,
                prediction_error=(
                    0.0
                    if prediction_record is None or prediction_record.state_error is None
                    else prediction_record.state_error
                ),
                reward=outcome.reward,
                resource_cost=0.10,
                mode="wake",
            )
        if self._episodic_memory is not None:
            cue = (
                self._cognitive_state.percept.features
                if self._cognitive_state.percept is not None
                else self._cognitive_state.world.latent
            )
            record = EpisodicMemoryRecord(
                memory_id=f"{self._state.episode_id}:memory:{self.tick}:{intent_id}",
                episode_id=self._state.episode_id,
                tick=outcome.tick,
                cue=cue.detach().clone(),
                action_intent=intent,
                outcome=outcome,
                world_transition=transition,
                provenance=outcome.provenance,
            )
            self._episodic_memory.write(record)
            memory = replace(
                memory,
                tick=self.tick,
                episodic_confidence=1.0,
                episodic_ids=(record.memory_id,),
            )
        self._cognitive_state = replace(
            self._cognitive_state,
            tick=self.tick,
            world=(world_state if world_state is not None else self._cognitive_state.world),
            memory=memory,
            goals=goals,
            homeostasis=homeostasis,
            outcome=outcome,
            world_transition=transition,
            world_prediction=prediction_record,
            world_calibration_trace=calibration_trace,
            learning=replace(
                self._cognitive_state.learning,
                tick=self.tick,
                lifetime_updates=self._cognitive_state.learning.lifetime_updates
                + int(kwargs.get("learn", True)),
            ),
        )
        return result

    def parameter_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            *super().parameter_tensors(),
            *self.perception.parameter_tensors(),
            *(() if self._executive is None else self._executive.parameter_tensors()),
        )

    def parameter_count(self, *, active_only: bool = True) -> int:
        del active_only
        return int(
            super().parameter_count()
            + sum(parameter.numel() for parameter in self.perception.parameters())
            + (
                0
                if self._executive is None
                else sum(parameter.numel() for parameter in self._executive.parameter_tensors())
            )
        )

    def dense_equivalent_parameter_count(self) -> int:
        return int(
            super().dense_equivalent_parameter_count()
            + sum(parameter.numel() for parameter in self.perception.parameters())
            + (
                0
                if self._workspace_router is None
                else sum(parameter.numel() for parameter in self._workspace_router.parameters())
            )
        )

    def checkpoint(self) -> dict[str, Any]:
        payload = super().checkpoint()
        payload["adapter"] = self.ADAPTER_NAME
        payload["perception"] = self.perception.checkpoint()
        if self._world_dynamics is not None:
            payload["world_dynamics"] = self._world_dynamics_checkpoint()
        if self._workspace_router is not None:
            payload["workspace_router"] = self._workspace_router.checkpoint()
        if self._episodic_memory is not None:
            payload["episodic_memory"] = self._episodic_memory.checkpoint()
        if self._semantic_memory is not None:
            payload["semantic_memory"] = self._semantic_memory.checkpoint()
        if self._procedural_memory is not None:
            payload["procedural_memory"] = self._procedural_memory.checkpoint()
        if self._homeostatic_controller is not None:
            payload["homeostasis"] = self._homeostatic_controller.checkpoint()
        if self._goal_planner is not None:
            payload["planning"] = self._goal_planner.checkpoint()
        if self._affordance_features is not None:
            payload["affordance_features"] = self._affordance_features.checkpoint()
        if self._executive is not None:
            payload["executive"] = self._executive.checkpoint()
        if self._generation_controller is not None:
            payload["generation"] = self._generation_controller.checkpoint()
        if self._content_selector is not None:
            payload["content_selection"] = self._content_selector.checkpoint()
        if self._language_organ is not None:
            payload["language_organ"] = self._language_organ.checkpoint()
        payload["language_backend_registry"] = self._language_backend_registry.checkpoint()
        payload["language_provider_artifact"] = (
            None
            if self._language_provider_artifact is None
            else self._language_provider_artifact.to_payload()
        )
        payload["planned_rollout"] = (
            None if self._planned_rollout is None else self._planned_rollout.to_payload()
        )
        payload["replan_required"] = self._replan_required
        payload["last_rollout_prediction_error"] = self._last_rollout_prediction_error
        payload["last_rollout_calibrated_confidence"] = self._last_rollout_calibrated_confidence
        payload["last_content_selection"] = (
            None
            if self._last_content_selection is None
            else self._last_content_selection.to_payload()
        )
        payload["last_content_prediction_error"] = self._last_content_prediction_error
        payload["content_feedback_applied"] = self._content_feedback_applied
        payload["last_executive_decision"] = (
            None
            if self._last_executive_decision is None
            else self._last_executive_decision.to_payload()
        )
        payload["last_executive_prediction_error"] = self._last_executive_prediction_error
        payload["last_delayed_executive_prediction_error"] = (
            self._last_delayed_executive_prediction_error
        )
        payload["last_affordance_prediction_error"] = self._last_affordance_prediction_error
        payload["last_executive_world_action"] = (
            None
            if self._last_executive_world_action is None
            else self._last_executive_world_action.to_payload()
        )
        payload["pending_executive_credit"] = self._pending_executive_credit_checkpoint()
        payload["language_fallback_count"] = self._language_fallback_count
        payload["language_fallback_requires_replan"] = self._language_fallback_requires_replan
        payload["last_language_emission"] = (
            None
            if self._last_language_emission is None
            else self._last_language_emission.to_payload()
        )
        return payload

    def restore(self, checkpoint: dict[str, Any]) -> None:
        super().restore(checkpoint)
        if "perception" in checkpoint:
            self.perception.restore(checkpoint["perception"])
        self._restore_world_dynamics(checkpoint.get("world_dynamics"))
        self._restore_workspace_router(checkpoint.get("workspace_router"))
        self._restore_episodic_memory(checkpoint.get("episodic_memory"))
        self._restore_semantic_memory(checkpoint.get("semantic_memory"))
        self._restore_procedural_memory(checkpoint.get("procedural_memory"))
        self._restore_homeostatic_controller(checkpoint.get("homeostasis"))
        self._restore_goal_planner(checkpoint.get("planning"))
        self._restore_affordance_features(checkpoint.get("affordance_features"))
        self._restore_executive(checkpoint.get("executive"))
        self._restore_generation_controller(checkpoint.get("generation"))
        self._restore_content_selector(checkpoint.get("content_selection"))
        self._restore_language_backend_registry(checkpoint.get("language_backend_registry"))
        self._restore_language_provider_artifact(checkpoint)
        self._restore_language_organ(checkpoint.get("language_organ"))
        self._restore_rollout_state(checkpoint)
        self._restore_generation_trace(checkpoint)
        self._restore_content_selection(checkpoint)
        self._restore_executive_state(checkpoint)
        self._restore_language_emission(checkpoint)
        self._restore_language_fallback_state(checkpoint)

    def _world_dynamics_checkpoint(self) -> dict[str, Any]:
        if self._world_dynamics is None:
            raise RuntimeError("world dynamics is not attached")
        return {
            "schema": self._world_dynamics.schema.payload(),
            "hidden_dim": self._world_dynamics.hidden_dim,
            "online_updates": self._world_dynamics.online_updates,
            "state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self._world_dynamics.state_dict().items()
            },
        }

    def _restore_world_dynamics(self, payload: Any) -> None:
        if payload is None:
            self._world_dynamics = None
            return
        schema = WorldSchema.from_payload(dict(payload["schema"]))
        learner = WorldDynamicsLearner(
            schema,
            hidden_dim=int(payload["hidden_dim"]),
            seed=0,
        )
        learner.load_state_dict(payload["state_dict"])
        learner.online_updates = int(payload.get("online_updates", 0))
        self._world_dynamics = learner

    def _restore_workspace_router(self, payload: Any) -> None:
        self._workspace_router = (
            None if payload is None else WorkspaceRouter.from_checkpoint(dict(payload), device=self.device)
        )

    def _restore_episodic_memory(self, payload: Any) -> None:
        self._episodic_memory = (
            None
            if payload is None
            else EpisodicMemoryStore.from_checkpoint(dict(payload), device=self.device)
        )

    def _restore_semantic_memory(self, payload: Any) -> None:
        self._semantic_memory = (
            None
            if payload is None
            else SemanticMemoryLearner.from_checkpoint(dict(payload), device=self.device)
        )

    def _restore_procedural_memory(self, payload: Any) -> None:
        self._procedural_memory = (
            None
            if payload is None
            else ProceduralMemoryLearner.from_checkpoint(dict(payload), device=self.device)
        )

    def _restore_homeostatic_controller(self, payload: Any) -> None:
        self._homeostatic_controller = (
            None
            if payload is None
            else HomeostaticController.from_checkpoint(dict(payload))
        )

    def _restore_goal_planner(self, payload: Any) -> None:
        self._goal_planner = (
            None if payload is None else GoalPlanner.from_checkpoint(dict(payload))
        )

    def _restore_affordance_features(self, payload: Any) -> None:
        self._affordance_features = (
            None
            if payload is None
            else LearnedAffordanceFeatures.from_checkpoint(dict(payload)).to(self.device)
        )
        self._affordance_grounding = (
            None
            if self._affordance_features is None
            else WorldAffordanceGroundingProducer(self._affordance_features.input_dim)
        )

    def _restore_executive(self, payload: Any) -> None:
        self._executive = (
            None
            if payload is None
            else ExecutiveController.from_checkpoint(dict(payload)).to(self.device)
        )

    def _restore_executive_state(self, payload: Any) -> None:
        decision = payload.get("last_executive_decision") if isinstance(payload, dict) else None
        self._last_executive_decision = (
            None
            if decision is None
            else ExecutiveDecision.from_payload(dict(decision), device=self.device)
        )
        error = payload.get("last_executive_prediction_error") if isinstance(payload, dict) else None
        self._last_executive_prediction_error = None if error is None else float(error)
        delayed_error = (
            payload.get("last_delayed_executive_prediction_error")
            if isinstance(payload, dict)
            else None
        )
        self._last_delayed_executive_prediction_error = (
            None if delayed_error is None else float(delayed_error)
        )
        affordance_error = (
            payload.get("last_affordance_prediction_error")
            if isinstance(payload, dict)
            else None
        )
        self._last_affordance_prediction_error = (
            None if affordance_error is None else float(affordance_error)
        )
        action = payload.get("last_executive_world_action") if isinstance(payload, dict) else None
        self._last_executive_world_action = (
            None
            if action is None
            else WorldAction.from_payload(dict(action), device=self.device)
        )
        self._restore_pending_executive_credit(
            payload.get("pending_executive_credit") if isinstance(payload, dict) else None
        )

    def _pending_executive_credit_checkpoint(self) -> dict[str, Any] | None:
        pending = self._pending_executive_credit
        if pending is None:
            return None
        return {
            "decision": pending.decision.to_payload(),
            "affordance": (
                None if pending.affordance is None else pending.affordance.to_payload()
            ),
            "percept_features": (
                None
                if pending.percept_features is None
                else pending.percept_features.detach().cpu().clone()
            ),
            "world_latent": (
                None
                if pending.world_latent is None
                else pending.world_latent.detach().cpu().clone()
            ),
            "world_uncertainty": pending.world_uncertainty,
            "learn": pending.learn,
        }

    def _restore_pending_executive_credit(self, payload: Any) -> None:
        if payload is None:
            self._pending_executive_credit = None
            return
        if not isinstance(payload, dict):
            raise ValueError("pending executive credit checkpoint must be a mapping")
        affordance_payload = payload.get("affordance")
        self._pending_executive_credit = _PendingExecutiveCredit(
            decision=ExecutiveDecision.from_payload(
                dict(payload["decision"]), device=self.device
            ),
            affordance=(
                None
                if affordance_payload is None
                else WorldAffordance.from_payload(
                    dict(affordance_payload), device=self.device
                )
            ),
            percept_features=(
                None
                if payload.get("percept_features") is None
                else payload["percept_features"].detach().to(self.device).clone()
            ),
            world_latent=(
                None
                if payload.get("world_latent") is None
                else payload["world_latent"].detach().to(self.device).clone()
            ),
            world_uncertainty=float(payload.get("world_uncertainty", 1.0)),
            learn=bool(payload.get("learn", True)),
        )

    def _restore_generation_controller(self, payload: Any) -> None:
        self._generation_controller = (
            None
            if payload is None
            else GenerationController.from_checkpoint(dict(payload))
        )

    def _restore_generation_trace(self, payload: Any) -> None:
        trace = payload.get("last_generation_trace") if isinstance(payload, dict) else None
        self._last_generation_trace = (
            None if trace is None else GenerationTrace.from_payload(dict(trace))
        )

    def _restore_content_selector(self, payload: Any) -> None:
        self._content_selector = (
            None if payload is None else ContentSelector.from_checkpoint(dict(payload))
        )

    def _restore_content_selection(self, payload: Any) -> None:
        selection = payload.get("last_content_selection") if isinstance(payload, dict) else None
        self._last_content_selection = (
            None
            if selection is None
            else ContentSelectionDecision.from_payload(dict(selection))
        )
        error = payload.get("last_content_prediction_error") if isinstance(payload, dict) else None
        self._last_content_prediction_error = None if error is None else float(error)
        self._content_feedback_applied = bool(
            payload.get("content_feedback_applied", False) if isinstance(payload, dict) else False
        )

    def _restore_language_fallback_state(self, payload: Any) -> None:
        self._language_fallback_count = int(
            payload.get("language_fallback_count", 0) if isinstance(payload, dict) else 0
        )
        self._language_fallback_requires_replan = bool(
            payload.get("language_fallback_requires_replan", False)
            if isinstance(payload, dict)
            else False
        )

    def _restore_language_organ(self, payload: Any) -> None:
        if payload is None:
            self._language_organ = None
            return
        if not isinstance(payload, dict):
            raise ValueError("language organ checkpoint must be a mapping")
        if payload.get("backend") != StructuredTextLanguageOrgan.BACKEND_ID:
            raise ValueError(
                "only the structured language-organ stub can be restored without a backend registry"
            )
        restored = StructuredTextLanguageOrgan.from_checkpoint(payload)
        self._language_backend_registry.validate(restored)
        if (
            self._language_provider_artifact is not None
            and restored.backend_id != self._language_provider_artifact.backend_id
        ):
            raise ValueError("restored language organ backend does not match provider artifact")
        self._language_organ = restored

    def _restore_language_backend_registry(self, payload: Any) -> None:
        self._language_backend_registry = (
            LanguageBackendRegistry.default()
            if payload is None
            else LanguageBackendRegistry.from_checkpoint(dict(payload))
        )

    def _restore_language_provider_artifact(self, payload: Any) -> None:
        artifact = payload.get("language_provider_artifact") if isinstance(payload, dict) else None
        self._language_provider_artifact = (
            None if artifact is None else LanguageProviderArtifact.from_payload(dict(artifact))
        )

    def _restore_language_emission(self, payload: Any) -> None:
        emission = payload.get("last_language_emission") if isinstance(payload, dict) else None
        self._last_language_emission = (
            None if emission is None else LanguageEmission.from_payload(dict(emission))
        )

    def _restore_rollout_state(self, payload: Any) -> None:
        rollout = payload.get("planned_rollout") if isinstance(payload, dict) else None
        self._planned_rollout = (
            None if rollout is None else ImaginedRollout.from_payload(dict(rollout))
        )
        self._replan_required = bool(
            payload.get("replan_required", False) if isinstance(payload, dict) else False
        )
        error = payload.get("last_rollout_prediction_error") if isinstance(payload, dict) else None
        self._last_rollout_prediction_error = None if error is None else float(error)
        confidence = (
            payload.get("last_rollout_calibrated_confidence")
            if isinstance(payload, dict)
            else None
        )
        self._last_rollout_calibrated_confidence = (
            None if confidence is None else float(confidence)
        )

    def native_checkpoint(self) -> dict[str, Any]:
        """Serialize the v1 cognitive state and its TSK-v8 compatibility kernel."""

        components: dict[str, Any] = {"perception": self.perception.checkpoint()}
        if self._world_dynamics is not None:
            components["world_dynamics"] = self._world_dynamics_checkpoint()
        if self._workspace_router is not None:
            components["workspace_router"] = self._workspace_router.checkpoint()
        if self._episodic_memory is not None:
            components["episodic_memory"] = self._episodic_memory.checkpoint()
        if self._semantic_memory is not None:
            components["semantic_memory"] = self._semantic_memory.checkpoint()
        if self._procedural_memory is not None:
            components["procedural_memory"] = self._procedural_memory.checkpoint()
        if self._homeostatic_controller is not None:
            components["homeostasis"] = self._homeostatic_controller.checkpoint()
        if self._goal_planner is not None:
            components["planning"] = self._goal_planner.checkpoint()
        if self._affordance_features is not None:
            components["affordance_features"] = self._affordance_features.checkpoint()
        if self._executive is not None:
            components["executive"] = self._executive.checkpoint()
        if self._generation_controller is not None:
            components["generation"] = self._generation_controller.checkpoint()
        if self._content_selector is not None:
            components["content_selection"] = self._content_selector.checkpoint()
        if self._language_organ is not None:
            components["language_organ"] = self._language_organ.checkpoint()
        components["language_backend_registry"] = self._language_backend_registry.checkpoint()
        components["language_provider_artifact"] = (
            None
            if self._language_provider_artifact is None
            else self._language_provider_artifact.to_payload()
        )
        components["planned_rollout"] = (
            None if self._planned_rollout is None else self._planned_rollout.to_payload()
        )
        components["replan_required"] = self._replan_required
        components["last_rollout_prediction_error"] = self._last_rollout_prediction_error
        components["last_rollout_calibrated_confidence"] = (
            self._last_rollout_calibrated_confidence
        )
        components["last_generation_trace"] = (
            None if self._last_generation_trace is None else self._last_generation_trace.to_payload()
        )
        components["last_content_selection"] = (
            None
            if self._last_content_selection is None
            else self._last_content_selection.to_payload()
        )
        components["last_content_prediction_error"] = self._last_content_prediction_error
        components["content_feedback_applied"] = self._content_feedback_applied
        components["last_executive_decision"] = (
            None
            if self._last_executive_decision is None
            else self._last_executive_decision.to_payload()
        )
        components["last_executive_prediction_error"] = self._last_executive_prediction_error
        components["last_delayed_executive_prediction_error"] = (
            self._last_delayed_executive_prediction_error
        )
        components["last_affordance_prediction_error"] = self._last_affordance_prediction_error
        components["last_executive_world_action"] = (
            None
            if self._last_executive_world_action is None
            else self._last_executive_world_action.to_payload()
        )
        components["pending_executive_credit"] = self._pending_executive_credit_checkpoint()
        components["language_fallback_count"] = self._language_fallback_count
        components["language_fallback_requires_replan"] = self._language_fallback_requires_replan
        components["last_language_emission"] = (
            None
            if self._last_language_emission is None
            else self._last_language_emission.to_payload()
        )
        return NativeCheckpoint(
            kernel=super().checkpoint(),
            cognitive_state=self.cognitive_snapshot(),
            adapter=self.ADAPTER_NAME,
            components=components,
        ).to_payload()

    def restore_native(self, checkpoint: dict[str, Any]) -> None:
        envelope = NativeCheckpoint.from_payload(checkpoint, device=self.device)
        if envelope.adapter != self.ADAPTER_NAME:
            raise ValueError(f"unsupported Taiji adapter: {envelope.adapter}")
        super().restore(envelope.kernel)
        if "perception" in envelope.components:
            self.perception.restore(envelope.components["perception"])
        self._restore_world_dynamics(envelope.components.get("world_dynamics"))
        self._restore_workspace_router(envelope.components.get("workspace_router"))
        self._restore_episodic_memory(envelope.components.get("episodic_memory"))
        self._restore_semantic_memory(envelope.components.get("semantic_memory"))
        self._restore_procedural_memory(envelope.components.get("procedural_memory"))
        self._restore_homeostatic_controller(envelope.components.get("homeostasis"))
        self._restore_goal_planner(envelope.components.get("planning"))
        self._restore_affordance_features(envelope.components.get("affordance_features"))
        self._restore_executive(envelope.components.get("executive"))
        self._restore_generation_controller(envelope.components.get("generation"))
        self._restore_content_selector(envelope.components.get("content_selection"))
        self._restore_language_backend_registry(
            envelope.components.get("language_backend_registry")
        )
        self._restore_language_provider_artifact(envelope.components)
        self._restore_language_organ(envelope.components.get("language_organ"))
        self._restore_rollout_state(envelope.components)
        self._restore_generation_trace(envelope.components)
        self._restore_content_selection(envelope.components)
        self._restore_executive_state(envelope.components)
        self._restore_language_emission(envelope.components)
        self._restore_language_fallback_state(envelope.components)
        state = envelope.cognitive_state
        if state.tick != self.tick or state.episode_id != self._state.episode_id:
            raise ValueError("native cognitive state is out of sync with kernel state")
        self._cognitive_state = state

    @classmethod
    def from_native_checkpoint(
        cls,
        checkpoint: dict[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> TSKV8Adapter:
        """Construct an adapter from the atomic v1 envelope."""

        envelope = NativeCheckpoint.from_payload(checkpoint, device=device)
        config = cls._config_from_kernel_checkpoint(envelope.kernel)
        model = cls(config, device=device, episode_id=envelope.cognitive_state.episode_id)
        model.restore_native(checkpoint)
        return model

    @staticmethod
    def _config_from_kernel_checkpoint(checkpoint: dict[str, Any]) -> Any:
        # Import locally so the adapter's public import surface stays small.
        from .config import TaijiConfig

        return TaijiConfig.from_dict(dict(checkpoint["config"]))
