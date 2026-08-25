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
    PerceptEvent,
    PlanCandidate,
    PlanState,
    SelfState,
    WorkspaceState,
    WorldState,
)
from .contracts import MemoryState as NativeMemoryState
from .model import Taiji
from .state import TaijiDecision, TaijiOutcome, TaijiStep


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
        self._cognitive_state = self._empty_cognitive_state(self._state.episode_id)

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

    def reset_dynamics(self, *, episode_id: str | None = None) -> None:
        super().reset_dynamics(episode_id=episode_id)
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
        features = self._state.motor_context.detach().clone()
        percept = PerceptEvent(
            event_id=f"{self._state.episode_id}:percept:{self.tick}",
            observation_tick=self.tick,
            modality=observation.modality,
            features=features,
            boundary=int(symbol) == self.config.boundary_symbol,
            confidence=1.0,
        )
        recall = step.memory_recall
        workspace = WorkspaceState(
            tick=self.tick,
            focus=("predictive-context",),
            broadcast=features,
            capacity=1,
        )
        world = WorldState(
            tick=self.tick,
            latent=features,
            uncertainty=max(0.0, min(1.0, 1.0 - recall.confidence)),
        )
        memory = NativeMemoryState(
            tick=self.tick,
            episodic_confidence=max(0.0, min(1.0, recall.confidence)),
            semantic_context=recall.cortical_feedback.detach().clone(),
            procedural_context=recall.action_evidence.detach().clone(),
        )
        previous = self._cognitive_state
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
        )
        return step

    def observe_event(
        self,
        observation: Observation,
        *,
        learn: bool = True,
        learn_motor: bool | None = None,
        use_memory: bool = True,
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
        return step

    def act(self, available_actions: Any, *args: Any, **kwargs: Any) -> TaijiDecision:
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
        self._cognitive_state = replace(
            self._cognitive_state,
            plan=PlanState(
                tick=self.tick,
                candidates=candidates,
                selected_plan_id=(candidates[selected_index].plan_id if candidates else None),
            ),
            action_intent=intent,
        )
        return decision

    def settle_action(self, reward: float, *args: Any, **kwargs: Any) -> TaijiOutcome:
        intent = self._cognitive_state.action_intent
        result = super().settle_action(reward, *args, **kwargs)
        intent_id = intent.intent_id if intent is not None else f"kernel-action:{result.tick}"
        outcome = Outcome(
            intent_id=intent_id,
            reward=float(result.reward),
            success=float(result.reward) > 0.0,
            provenance=str(kwargs.get("provenance", "experienced")),
            tick=self.tick,
        )
        self._cognitive_state = replace(
            self._cognitive_state,
            tick=self.tick,
            outcome=outcome,
            learning=replace(
                self._cognitive_state.learning,
                tick=self.tick,
                lifetime_updates=self._cognitive_state.learning.lifetime_updates
                + int(kwargs.get("learn", True)),
            ),
        )
        return result

    def native_checkpoint(self) -> dict[str, Any]:
        """Serialize the v1 cognitive state and its TSK-v8 compatibility kernel."""

        return NativeCheckpoint(
            kernel=super().checkpoint(),
            cognitive_state=self.cognitive_snapshot(),
            adapter=self.ADAPTER_NAME,
        ).to_payload()

    def restore_native(self, checkpoint: dict[str, Any]) -> None:
        envelope = NativeCheckpoint.from_payload(checkpoint, device=self.device)
        if envelope.adapter != self.ADAPTER_NAME:
            raise ValueError(f"unsupported Taiji adapter: {envelope.adapter}")
        super().restore(envelope.kernel)
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
