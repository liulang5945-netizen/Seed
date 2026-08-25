"""Compatibility adapter exposing TSK-v8 through Taiji v1 contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from .contracts import (
    CONTRACT_FORMAT,
    ActionIntent,
    CognitiveState,
    DevelopmentState,
    GoalState,
    HomeostaticState,
    LearningState,
    NativeCheckpoint,
    Observation,
    Outcome,
    PlanCandidate,
    PlanState,
    SelfState,
    WorkspaceState,
    WorldAction,
    WorldPredictionRecord,
    WorldState,
    WorldTransition,
)
from .contracts import MemoryState as NativeMemoryState
from .model import Taiji
from .perception import LearnedPerception
from .state import TaijiDecision, TaijiOutcome, TaijiStep
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

    def reset_dynamics(self, *, episode_id: str | None = None) -> None:
        super().reset_dynamics(episode_id=episode_id)
        self.perception.reset_dynamics()
        self._cognitive_state = self._empty_cognitive_state(self._state.episode_id)

    def observe(self, symbol: int, *args: Any, **kwargs: Any) -> TaijiStep:
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
            homeostasis=replace(previous.homeostasis, tick=self.tick),
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
    ) -> TaijiStep:
        """Ingest a v1 ``Observation`` while retaining the kernel API."""

        if observation.modality != "text-byte" or not isinstance(observation.value, int):
            raise ValueError("TSK-v8 adapter accepts only integer text-byte observations")
        step = self.observe(
            observation.value,
            learn=learn,
            learn_motor=learn_motor,
            use_memory=use_memory,
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
        decision = super().act(available_actions, *args, **kwargs)
        intent = ActionIntent(
            intent_id=f"{self._state.episode_id}:intent:{decision.tick}",
            kind="byte-motor",
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
                action_kind="byte-motor",
                expected_value=float(decision.policy_probabilities[action]),
            )
            for index, action in enumerate(decision.available_actions)
        )
        selected_index = decision.available_actions.index(decision.action_symbol)
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
            plan=PlanState(
                tick=self.tick,
                candidates=candidates,
                selected_plan_id=(candidates[selected_index].plan_id if candidates else None),
            ),
            action_intent=intent,
            world_prediction=world_prediction,
        )
        return decision

    def settle_action(self, reward: float, *args: Any, **kwargs: Any) -> TaijiOutcome:
        intent = self._cognitive_state.action_intent
        world_state = kwargs.pop("world_state", None)
        world_action = kwargs.pop("world_action", None)
        success = kwargs.pop("success", None)
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
        self._cognitive_state = replace(
            self._cognitive_state,
            tick=self.tick,
            world=(world_state if world_state is not None else self._cognitive_state.world),
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
        )

    def checkpoint(self) -> dict[str, Any]:
        payload = super().checkpoint()
        payload["adapter"] = self.ADAPTER_NAME
        payload["perception"] = self.perception.checkpoint()
        if self._world_dynamics is not None:
            payload["world_dynamics"] = self._world_dynamics_checkpoint()
        return payload

    def restore(self, checkpoint: dict[str, Any]) -> None:
        super().restore(checkpoint)
        if "perception" in checkpoint:
            self.perception.restore(checkpoint["perception"])
        self._restore_world_dynamics(checkpoint.get("world_dynamics"))

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

    def native_checkpoint(self) -> dict[str, Any]:
        """Serialize the v1 cognitive state and its TSK-v8 compatibility kernel."""

        components: dict[str, Any] = {"perception": self.perception.checkpoint()}
        if self._world_dynamics is not None:
            components["world_dynamics"] = self._world_dynamics_checkpoint()
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
