"""Persistent native state for the Taiji predictive fabric."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .identity_organ import IdentityRecall


@dataclass
class RegionState:
    membrane: torch.Tensor
    activity: torch.Tensor
    trace: torch.Tensor
    prediction: torch.Tensor
    error: torch.Tensor
    threshold: torch.Tensor
    inhibition: torch.Tensor

    def clone(self) -> RegionState:
        return RegionState(
            membrane=self.membrane.detach().clone(),
            activity=self.activity.detach().clone(),
            trace=self.trace.detach().clone(),
            prediction=self.prediction.detach().clone(),
            error=self.error.detach().clone(),
            threshold=self.threshold.detach().clone(),
            inhibition=self.inhibition.detach().clone(),
        )

    def to_payload(self) -> dict[str, Any]:
        cloned = self.clone()
        return {
            "membrane": cloned.membrane.cpu(),
            "activity": cloned.activity.cpu(),
            "trace": cloned.trace.cpu(),
            "prediction": cloned.prediction.cpu(),
            "error": cloned.error.cpu(),
            "threshold": cloned.threshold.cpu(),
            "inhibition": cloned.inhibition.cpu(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, device: torch.device | str) -> RegionState:
        return cls(
            membrane=payload["membrane"].detach().to(device).clone(),
            activity=payload["activity"].detach().to(device).clone(),
            trace=payload["trace"].detach().to(device).clone(),
            prediction=payload["prediction"].detach().to(device).clone(),
            error=payload["error"].detach().to(device).clone(),
            threshold=payload["threshold"].detach().to(device).clone(),
            inhibition=payload["inhibition"].detach().to(device).clone(),
        )


@dataclass
class MemoryState:
    """Fast activity of the episodic field; learned engrams live in synapses."""

    activity: torch.Tensor
    trace: torch.Tensor
    cortical_feedback: torch.Tensor
    threshold: torch.Tensor
    inhibition: float
    last_confidence: float

    def clone(self) -> MemoryState:
        return MemoryState(
            activity=self.activity.detach().clone(),
            trace=self.trace.detach().clone(),
            cortical_feedback=self.cortical_feedback.detach().clone(),
            threshold=self.threshold.detach().clone(),
            inhibition=float(self.inhibition),
            last_confidence=float(self.last_confidence),
        )

    def to_payload(self) -> dict[str, Any]:
        cloned = self.clone()
        return {
            "activity": cloned.activity.cpu(),
            "trace": cloned.trace.cpu(),
            "cortical_feedback": cloned.cortical_feedback.cpu(),
            "threshold": cloned.threshold.cpu(),
            "inhibition": cloned.inhibition,
            "last_confidence": cloned.last_confidence,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, device: torch.device | str) -> MemoryState:
        return cls(
            activity=payload["activity"].detach().to(device).clone(),
            trace=payload["trace"].detach().to(device).clone(),
            cortical_feedback=(payload["cortical_feedback"].detach().to(device).clone()),
            threshold=payload["threshold"].detach().to(device).clone(),
            inhibition=float(payload["inhibition"]),
            last_confidence=float(payload["last_confidence"]),
        )


@dataclass
class PendingAction:
    """Unsettled motor eligibility waiting for an environment outcome."""

    tick: int
    action_symbol: int
    available_actions: tuple[int, ...]
    context: torch.Tensor
    policy_probabilities: torch.Tensor

    def clone(self) -> PendingAction:
        return PendingAction(
            tick=int(self.tick),
            action_symbol=int(self.action_symbol),
            available_actions=tuple(int(value) for value in self.available_actions),
            context=self.context.detach().clone(),
            policy_probabilities=self.policy_probabilities.detach().clone(),
        )

    def to_payload(self) -> dict[str, Any]:
        cloned = self.clone()
        return {
            "tick": cloned.tick,
            "action_symbol": cloned.action_symbol,
            "available_actions": list(cloned.available_actions),
            "context": cloned.context.cpu(),
            "policy_probabilities": cloned.policy_probabilities.cpu(),
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str
    ) -> PendingAction:
        return cls(
            tick=int(payload["tick"]),
            action_symbol=int(payload["action_symbol"]),
            available_actions=tuple(int(value) for value in payload["available_actions"]),
            context=payload["context"].detach().to(device).clone(),
            policy_probabilities=(payload["policy_probabilities"].detach().to(device).clone()),
        )


@dataclass
class PendingExperience:
    """Settled action waiting to bind its next environmental sensation."""

    tick: int
    action_symbol: int
    reward: float
    cortical_context: torch.Tensor
    episode_id: str
    provenance: str
    learn_memory: bool
    memory_learning_scale: float = 1.0
    memory_learning_targets: str = "all"

    def clone(self) -> PendingExperience:
        return PendingExperience(
            tick=int(self.tick),
            action_symbol=int(self.action_symbol),
            reward=float(self.reward),
            cortical_context=self.cortical_context.detach().clone(),
            episode_id=str(self.episode_id),
            provenance=str(self.provenance),
            learn_memory=bool(self.learn_memory),
            memory_learning_scale=float(self.memory_learning_scale),
            memory_learning_targets=str(self.memory_learning_targets),
        )

    def to_payload(self) -> dict[str, Any]:
        cloned = self.clone()
        return {
            "tick": cloned.tick,
            "action_symbol": cloned.action_symbol,
            "reward": cloned.reward,
            "cortical_context": cloned.cortical_context.cpu(),
            "episode_id": cloned.episode_id,
            "provenance": cloned.provenance,
            "learn_memory": cloned.learn_memory,
            "memory_learning_scale": cloned.memory_learning_scale,
            "memory_learning_targets": cloned.memory_learning_targets,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, device: torch.device | str
    ) -> PendingExperience:
        return cls(
            tick=int(payload["tick"]),
            action_symbol=int(payload["action_symbol"]),
            reward=float(payload["reward"]),
            cortical_context=(payload["cortical_context"].detach().to(device).clone()),
            episode_id=str(payload["episode_id"]),
            provenance=str(payload["provenance"]),
            learn_memory=bool(payload["learn_memory"]),
            memory_learning_scale=float(payload.get("memory_learning_scale", 1.0)),
            memory_learning_targets=str(payload.get("memory_learning_targets", "all")),
        )


@dataclass
class TaijiState:
    version: int
    tick: int
    episode_id: str
    regions: tuple[RegionState, ...]
    memory: MemoryState
    motor_context: torch.Tensor
    motor_probabilities: torch.Tensor
    last_symbol: int | None
    pending_action: PendingAction | None
    pending_experience: PendingExperience | None

    def clone(self) -> TaijiState:
        return TaijiState(
            version=int(self.version),
            tick=int(self.tick),
            episode_id=str(self.episode_id),
            regions=tuple(region.clone() for region in self.regions),
            memory=self.memory.clone(),
            motor_context=self.motor_context.detach().clone(),
            motor_probabilities=self.motor_probabilities.detach().clone(),
            last_symbol=None if self.last_symbol is None else int(self.last_symbol),
            pending_action=(None if self.pending_action is None else self.pending_action.clone()),
            pending_experience=(
                None if self.pending_experience is None else self.pending_experience.clone()
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "tick": int(self.tick),
            "episode_id": str(self.episode_id),
            "regions": [region.to_payload() for region in self.regions],
            "memory": self.memory.to_payload(),
            "motor_context": self.motor_context.detach().cpu().clone(),
            "motor_probabilities": self.motor_probabilities.detach().cpu().clone(),
            "last_symbol": self.last_symbol,
            "pending_action": (
                None if self.pending_action is None else self.pending_action.to_payload()
            ),
            "pending_experience": (
                None if self.pending_experience is None else self.pending_experience.to_payload()
            ),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, device: torch.device | str) -> TaijiState:
        return cls(
            version=int(payload["version"]),
            tick=int(payload["tick"]),
            episode_id=str(payload["episode_id"]),
            regions=tuple(
                RegionState.from_payload(region, device=device) for region in payload["regions"]
            ),
            memory=MemoryState.from_payload(payload["memory"], device=device),
            motor_context=payload["motor_context"].detach().to(device).clone(),
            motor_probabilities=(payload["motor_probabilities"].detach().to(device).clone()),
            last_symbol=(
                None if payload.get("last_symbol") is None else int(payload["last_symbol"])
            ),
            pending_action=(
                None
                if payload.get("pending_action") is None
                else PendingAction.from_payload(payload["pending_action"], device=device)
            ),
            pending_experience=(
                None
                if payload.get("pending_experience") is None
                else PendingExperience.from_payload(payload["pending_experience"], device=device)
            ),
        )


@dataclass(frozen=True)
class TaijiStep:
    tick: int
    observed_symbol: int
    predicted_symbol: int
    probabilities: torch.Tensor
    prior_prediction: int | None
    prior_probability: float | None
    surprise: float | None
    activity_rates: tuple[float, ...]
    local_error_norms: tuple[float, ...]
    memory_recall: MemoryRecall
    memory_write_strength: float
    identity_recall: IdentityRecall | None = None


@dataclass(frozen=True, eq=False)
class MemoryRecall:
    action_evidence: torch.Tensor
    action_probabilities: torch.Tensor
    outcome_probabilities: torch.Tensor
    cortical_feedback: torch.Tensor
    time_code: torch.Tensor
    episode_code: torch.Tensor
    provenance_probabilities: torch.Tensor
    expected_reward: float
    confidence: float
    used_long_term: bool


@dataclass(frozen=True, eq=False)
class TaijiDecision:
    tick: int
    action_symbol: int
    available_actions: tuple[int, ...]
    policy_probabilities: torch.Tensor


@dataclass(frozen=True)
class TaijiOutcome:
    tick: int
    action_symbol: int
    reward: float
    reward_prediction_error: float
    learning_error_norm: float


@dataclass(frozen=True)
class TaijiConsolidation:
    """Result of one endogenous replay/consolidation bout.

    ``cycles`` counts attempted spontaneous reactivations, ``accepted`` counts
    the ones whose own priority passed the field's internal gate and therefore
    drove cortical learning.  No external replay list participates.

    ``structural_events`` counts the cortical contacts this bout rewired.  It
    reports work done during the bout rather than any property of the network,
    so it is a result field and never enters the checkpoint.
    """

    cycles: int
    accepted: int
    mean_priority: float
    mean_novelty: float
    mean_value: float
    mean_confidence: float
    mean_error_norm: float
    replayed_probability: float
    structural_events: int
