"""Compatibility adapter exposing TSK-v8 through Taiji v1 contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch

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
    WorldPredictionRecord,
    WorldState,
    WorldTransition,
)
from .contracts import MemoryState as NativeMemoryState
from .environment import EnvironmentOutcome, TaijiToolEnvironment
from .episodic_memory import EpisodicMemoryStore
from .generation import GenerationController, GenerationTrace, ToolCall
from .homeostasis import HomeostaticController, HomeostaticDrive
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


class TSKV8Adapter(Taiji):
    """Keep the TSK-v8 API while making v1 ownership explicit.

    This subclass is intentional: old callers still see a ``Taiji`` and old
    ``taiji-native-v8`` checkpoints remain readable.  New callers can use
    ``native_checkpoint`` and ``cognitive_snapshot`` without treating the
    kernel's byte prediction state as the complete v1 cognitive state.
    """

    ADAPTER_NAME = "tsk-v8"
    NATIVE_CHECKPOINT_FORMAT = CONTRACT_FORMAT

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
        self._planned_rollout: ImaginedRollout | None = None
        self._replan_required = False
        self._last_rollout_prediction_error: float | None = None
        self._last_rollout_calibrated_confidence: float | None = None
        self._generation_controller: GenerationController | None = None
        self._last_generation_trace: GenerationTrace | None = None

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
        self._last_generation_trace = None

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
            self._cognitive_state = replace(self._cognitive_state, world=world_state)
        return step

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
        intent = ActionIntent(
            intent_id=f"{self._state.episode_id}:intent:{decision.tick}",
            kind=(
                "byte-motor"
                if action_kinds is None
                else action_kinds[decision.available_actions.index(decision.action_symbol)]
            ),
            parameters={
                "action_symbol": decision.action_symbol,
                "available_actions": decision.available_actions,
            },
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
        transition = None
        prediction_record = self._cognitive_state.world_prediction
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
            transition = WorldTransition(
                before=before,
                action=world_action,
                after=world_state,
                outcome=outcome,
            )
            if prediction_record is not None and self._world_dynamics is not None:
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
        memory = self._cognitive_state.memory
        if self._planned_rollout is not None and self._goal_planner is not None:
            rollout = self._planned_rollout
            self._last_rollout_prediction_error = self._goal_planner.rollout_prediction_error(
                rollout, outcome
            )
            self._replan_required = (
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
            learning=replace(
                self._cognitive_state.learning,
                tick=self.tick,
                lifetime_updates=self._cognitive_state.learning.lifetime_updates
                + int(kwargs.get("learn", True)),
            ),
        )
        return result

    def parameter_tensors(self) -> tuple[torch.Tensor, ...]:
        return (*super().parameter_tensors(), *self.perception.parameter_tensors())

    def parameter_count(self, *, active_only: bool = True) -> int:
        del active_only
        return int(
            super().parameter_count()
            + sum(parameter.numel() for parameter in self.perception.parameters())
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
        if self._generation_controller is not None:
            payload["generation"] = self._generation_controller.checkpoint()
        payload["planned_rollout"] = (
            None if self._planned_rollout is None else self._planned_rollout.to_payload()
        )
        payload["replan_required"] = self._replan_required
        payload["last_rollout_prediction_error"] = self._last_rollout_prediction_error
        payload["last_rollout_calibrated_confidence"] = self._last_rollout_calibrated_confidence
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
        self._restore_generation_controller(checkpoint.get("generation"))
        self._restore_rollout_state(checkpoint)
        self._restore_generation_trace(checkpoint)

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
        if self._generation_controller is not None:
            components["generation"] = self._generation_controller.checkpoint()
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
        self._restore_generation_controller(envelope.components.get("generation"))
        self._restore_rollout_state(envelope.components)
        self._restore_generation_trace(envelope.components)
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
