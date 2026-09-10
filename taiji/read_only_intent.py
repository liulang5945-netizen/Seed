"""Native, policy-bounded rendering from semantic state to read-only intents.

This is an output boundary, not a second planner.  The learned semantic and
transition owners choose the Goal/ContentPlan; this module only binds a
verified Workbench observation to a capability that an explicit, versioned
read-only policy allows.  It never creates write, terminal, MCP, or language
selection parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import ActionIntent, Goal, WorldState
from .generation import ContentPlan
from .internalization import content_digest
from .semantic_training import semantic_fact_key
from .workbench_observation import WorkbenchObservation

READ_ONLY_INTENT_POLICY_FORMAT = "taiji-read-only-intent-policy-v1"
READ_ONLY_INTENT_POLICY_VERSION = 1
READ_ONLY_INTENT_PLANNER_FORMAT = "taiji-native-read-only-intent-planner-v1"
READ_ONLY_INTENT_PLANNER_VERSION = 1
READ_ONLY_ABSTENTION_FORMAT = "taiji-read-only-abstention-v1"
READ_ONLY_ABSTENTION_VERSION = 1
READ_ONLY_ABSTENTION_NEXT_STEPS = frozenset(
    {"none", "request_clarification", "workspace.list"}
)
READ_ONLY_INTENT_CAPABILITIES = frozenset(
    {
        "workspace.list",
        "workspace.read",
        "workspace.programming_language.resolve",
    }
)


def _text(value: Any, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _unit(value: Any, name: str) -> float:
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return normalized


@dataclass(frozen=True)
class ReadOnlyIntentPolicy:
    """Explicit content-to-capability policy supplied by the host boundary."""

    routes: tuple[tuple[str, str], ...]
    route_parameters: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = ()
    format: str = READ_ONLY_INTENT_POLICY_FORMAT
    version: int = READ_ONLY_INTENT_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.format != READ_ONLY_INTENT_POLICY_FORMAT:
            raise ValueError("unsupported read-only intent policy format")
        if int(self.version) != READ_ONLY_INTENT_POLICY_VERSION:
            raise ValueError("unsupported read-only intent policy version")
        normalized = tuple(
            sorted(
                (
                    _text(key, "content route"),
                    _text(value, "capability route"),
                )
                for key, value in self.routes
            )
        )
        if len({key for key, _ in normalized}) != len(normalized):
            raise ValueError("read-only intent content routes must be unique")
        if any(capability not in READ_ONLY_INTENT_CAPABILITIES for _, capability in normalized):
            raise ValueError("read-only intent policy contains a non-read-only capability")
        object.__setattr__(self, "routes", normalized)
        route_keys = {key for key, _ in normalized}
        normalized_parameters: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        for content_id, raw_parameters in self.route_parameters:
            content_key = _text(content_id, "route parameter content id")
            if content_key not in route_keys:
                raise ValueError("route parameters must reference a declared content route")
            if isinstance(raw_parameters, Mapping):
                parameter_items = raw_parameters.items()
            else:
                parameter_items = raw_parameters
            parameters = tuple(
                sorted(
                    (
                        _text(name, "route parameter name"),
                        _text(value, "route parameter value"),
                    )
                    for name, value in parameter_items
                )
            )
            if len({name for name, _ in parameters}) != len(parameters):
                raise ValueError("route parameter names must be unique")
            normalized_parameters.append((content_key, parameters))
        if len({key for key, _ in normalized_parameters}) != len(normalized_parameters):
            raise ValueError("route parameter content ids must be unique")
        object.__setattr__(self, "route_parameters", tuple(sorted(normalized_parameters)))

    def capability_for(self, content_id: str) -> str | None:
        return dict(self.routes).get(str(content_id))

    def parameters_for(self, content_id: str) -> dict[str, str]:
        return dict(dict(self.route_parameters).get(str(content_id), ()))

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "format": self.format,
            "version": self.version,
            "routes": {content_id: capability for content_id, capability in self.routes},
        }
        if self.route_parameters:
            payload["route_parameters"] = {
                content_id: dict(parameters)
                for content_id, parameters in self.route_parameters
            }
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ReadOnlyIntentPolicy:
        raw_routes = payload.get("routes", {})
        if not isinstance(raw_routes, Mapping):
            raise ValueError("read-only intent policy routes must be a mapping")
        raw_parameters = payload.get("route_parameters", {})
        if not isinstance(raw_parameters, Mapping):
            raise ValueError("read-only intent policy route_parameters must be a mapping")
        route_parameters: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        for content_id, parameters in raw_parameters.items():
            if not isinstance(parameters, Mapping):
                raise ValueError(
                    "read-only intent policy route parameters must be mappings"
                )
            route_parameters.append(
                (
                    str(content_id),
                    tuple((str(name), str(value)) for name, value in parameters.items()),
                )
            )
        return cls(
            routes=tuple((str(content_id), str(capability)) for content_id, capability in raw_routes.items()),
            route_parameters=tuple(route_parameters),
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
        )


@dataclass(frozen=True)
class ReadOnlyAbstention:
    """Typed, non-executable result for low-evidence read-only decisions."""

    abstention_id: str
    reason_code: str
    next_step: str
    snapshot_id: str
    capability_revision: int
    observation_digest: str
    confidence: float
    format: str = READ_ONLY_ABSTENTION_FORMAT
    version: int = READ_ONLY_ABSTENTION_VERSION

    def __post_init__(self) -> None:
        if self.format != READ_ONLY_ABSTENTION_FORMAT:
            raise ValueError("unsupported read-only abstention format")
        if int(self.version) != READ_ONLY_ABSTENTION_VERSION:
            raise ValueError("unsupported read-only abstention version")
        _text(self.abstention_id, "abstention_id")
        _text(self.reason_code, "abstention reason_code")
        if self.next_step not in READ_ONLY_ABSTENTION_NEXT_STEPS:
            raise ValueError("unsupported read-only abstention next_step")
        _text(self.snapshot_id, "abstention snapshot_id")
        _text(self.observation_digest, "abstention observation_digest")
        if int(self.capability_revision) < 1:
            raise ValueError("abstention capability_revision must be positive")
        _unit(self.confidence, "abstention confidence")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "abstention_id": self.abstention_id,
            "reason_code": self.reason_code,
            "next_step": self.next_step,
            "snapshot_id": self.snapshot_id,
            "capability_revision": self.capability_revision,
            "observation_digest": self.observation_digest,
            "confidence": self.confidence,
            "action_intent": None,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ReadOnlyAbstention:
        if payload.get("action_intent") is not None:
            raise ValueError("read-only abstention cannot carry an ActionIntent")
        return cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            abstention_id=str(payload["abstention_id"]),
            reason_code=str(payload["reason_code"]),
            next_step=str(payload["next_step"]),
            snapshot_id=str(payload["snapshot_id"]),
            capability_revision=int(payload["capability_revision"]),
            observation_digest=str(payload["observation_digest"]),
            confidence=float(payload.get("confidence", 0.0)),
        )


@dataclass(frozen=True)
class ReadOnlyIntentDecision:
    """Auditable result of native intent rendering before Workbench policy."""

    accepted: bool
    candidate_id: str
    reason_code: str
    snapshot_id: str
    capability_revision: int
    observation_digest: str
    action_intent: ActionIntent | None = None
    format: str = READ_ONLY_INTENT_PLANNER_FORMAT
    version: int = READ_ONLY_INTENT_PLANNER_VERSION

    def __post_init__(self) -> None:
        if self.format != READ_ONLY_INTENT_PLANNER_FORMAT:
            raise ValueError("unsupported read-only intent decision format")
        if int(self.version) != READ_ONLY_INTENT_PLANNER_VERSION:
            raise ValueError("unsupported read-only intent decision version")
        _text(self.candidate_id, "intent candidate_id")
        _text(self.reason_code, "intent reason_code")
        _text(self.snapshot_id, "intent snapshot_id")
        _text(self.observation_digest, "intent observation_digest")
        if int(self.capability_revision) < 1:
            raise ValueError("intent capability_revision must be positive")
        if self.accepted and self.action_intent is None:
            raise ValueError("accepted intent decision requires an ActionIntent")
        if not self.accepted and self.action_intent is not None:
            raise ValueError("rejected intent decision cannot contain an ActionIntent")

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "accepted": self.accepted,
            "candidate_id": self.candidate_id,
            "reason_code": self.reason_code,
            "snapshot_id": self.snapshot_id,
            "capability_revision": self.capability_revision,
            "observation_digest": self.observation_digest,
            "action_intent": (
                None if self.action_intent is None else self.action_intent.to_payload()
            ),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ReadOnlyIntentDecision:
        raw_intent = payload.get("action_intent")
        return cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            accepted=bool(payload.get("accepted", False)),
            candidate_id=str(payload["candidate_id"]),
            reason_code=str(payload["reason_code"]),
            snapshot_id=str(payload["snapshot_id"]),
            capability_revision=int(payload["capability_revision"]),
            observation_digest=str(payload["observation_digest"]),
            action_intent=(
                None if raw_intent is None else ActionIntent.from_payload(raw_intent)
            ),
        )


class NativeReadOnlyIntentPlanner:
    """Render only the capabilities declared by :class:`ReadOnlyIntentPolicy`."""

    CHECKPOINT_FORMAT = READ_ONLY_INTENT_PLANNER_FORMAT
    CHECKPOINT_VERSION = READ_ONLY_INTENT_PLANNER_VERSION

    def __init__(self, policy: ReadOnlyIntentPolicy) -> None:
        if not isinstance(policy, ReadOnlyIntentPolicy):
            raise TypeError("native read-only intent planner requires a policy")
        self.policy = policy

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": self.CHECKPOINT_FORMAT,
            "version": self.CHECKPOINT_VERSION,
            "policy": self.policy.to_payload(),
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> NativeReadOnlyIntentPlanner:
        if payload.get("format") != cls.CHECKPOINT_FORMAT:
            raise ValueError("unsupported native read-only intent planner format")
        if int(payload.get("version", -1)) != cls.CHECKPOINT_VERSION:
            raise ValueError("unsupported native read-only intent planner version")
        return cls(ReadOnlyIntentPolicy.from_payload(payload["policy"]))

    def abstain(
        self,
        *,
        observation: WorkbenchObservation,
        capability_snapshot: Any,
        reason_code: str,
        next_step: str,
        confidence: float,
    ) -> ReadOnlyAbstention:
        if not isinstance(observation, WorkbenchObservation):
            raise TypeError("read-only abstention observation must be WorkbenchObservation")
        snapshot_id = _text(getattr(capability_snapshot, "snapshot_id", ""), "snapshot_id")
        capability_revision = int(getattr(capability_snapshot, "revision", 0))
        if capability_revision < 1:
            raise ValueError("capability snapshot revision must be positive")
        abstention_id = content_digest(
            {
                "observation_digest": observation.observation_digest,
                "reason_code": str(reason_code),
                "next_step": str(next_step),
                "snapshot_id": snapshot_id,
                "capability_revision": capability_revision,
            }
        )[:32]
        return ReadOnlyAbstention(
            abstention_id=abstention_id,
            reason_code=_text(reason_code, "abstention reason_code"),
            next_step=_text(next_step, "abstention next_step"),
            snapshot_id=snapshot_id,
            capability_revision=capability_revision,
            observation_digest=observation.observation_digest,
            confidence=float(confidence),
        )

    @staticmethod
    def _world_matches_observation(
        world: WorldState,
        observation: WorkbenchObservation,
    ) -> bool:
        expected_event = observation.to_percept_event(tick=world.tick)
        if (
            world.percept_event_id != expected_event.event_id
            or world.percept_assembly_id != expected_event.assembly_id
        ):
            return False
        actual = {semantic_fact_key(*relation) for relation in world.relations}
        expected = {
            semantic_fact_key(
                "workbench",
                "read",
                "success" if observation.read_success else "failure",
            ),
            semantic_fact_key(
                "workbench",
                "target",
                "file" if observation.file_is_file else "missing",
            ),
            semantic_fact_key("workbench", "language", observation.language_id),
            semantic_fact_key("workbench", "language_state", observation.selection_state),
            semantic_fact_key(
                "workbench",
                "toolchain",
                "available" if observation.toolchain_available else "missing",
            ),
            semantic_fact_key(
                "workbench",
                "diagnostics",
                "connected" if observation.diagnostics_connected else "disconnected",
            ),
        }
        return actual == expected

    def propose(
        self,
        *,
        observation: WorkbenchObservation,
        world: WorldState,
        goal: Goal,
        content: ContentPlan,
        capability_snapshot: Any,
        tick: int,
    ) -> ReadOnlyIntentDecision:
        if not isinstance(observation, WorkbenchObservation):
            raise TypeError("read-only intent observation must be WorkbenchObservation")
        if not isinstance(world, WorldState):
            raise TypeError("read-only intent world must be WorldState")
        if not isinstance(goal, Goal) or not isinstance(content, ContentPlan):
            raise TypeError("read-only intent requires Goal and ContentPlan")
        snapshot_id = _text(getattr(capability_snapshot, "snapshot_id", ""), "snapshot_id")
        capability_revision = int(getattr(capability_snapshot, "revision", 0))
        if capability_revision < 1:
            raise ValueError("capability snapshot revision must be positive")
        candidate_id = content_digest(
            {
                "observation_digest": observation.observation_digest,
                "goal_id": goal.goal_id,
                "content_id": content.content_id,
                "snapshot_id": snapshot_id,
                "tick": int(tick),
            }
        )[:32]

        def reject(reason_code: str) -> ReadOnlyIntentDecision:
            return ReadOnlyIntentDecision(
                accepted=False,
                candidate_id=candidate_id,
                reason_code=reason_code,
                snapshot_id=snapshot_id,
                capability_revision=capability_revision,
                observation_digest=observation.observation_digest,
            )

        if observation.capability_snapshot_id != snapshot_id:
            return reject("stale_capability_snapshot")
        if observation.capability_revision != capability_revision:
            return reject("stale_capability_revision")
        if content.source_goal_id not in {None, goal.goal_id}:
            return reject("content_goal_mismatch")
        if not self._world_matches_observation(world, observation):
            return reject("stale_world_observation")
        capability_id = self.policy.capability_for(content.content_id)
        if capability_id is None:
            return reject("clarification_required")
        descriptor = capability_snapshot.get(capability_id)
        if descriptor is None or not descriptor.enabled:
            return reject("capability_not_connected")
        if descriptor.risk != "read_only" or capability_id not in READ_ONLY_INTENT_CAPABILITIES:
            return reject("read_only_policy_rejected")
        if capability_id != "workspace.list" and not observation.read_success:
            return reject("workspace_target_unavailable")
        if capability_id == "workspace.programming_language.resolve" and (
            observation.selection_state != "resolved"
        ):
            return reject("language_evidence_ambiguous")
        if capability_id == "workspace.read" and not observation.file_is_file:
            return reject("workspace_target_not_file")
        parameters = {"path": observation.path}
        parameters.update(self.policy.parameters_for(content.content_id))
        if capability_id == "workspace.list" and parameters.get("path") != ".":
            return reject("recovery_scope_not_root")
        if set(parameters) - descriptor.parameter_names:
            return reject("capability_parameter_drift")
        confidence = min(
            float(content.confidence),
            (
                float(observation.language_confidence)
                if capability_id == "workspace.programming_language.resolve"
                else 1.0
            ),
        )
        confidence = _unit(confidence, "intent confidence")
        intent = ActionIntent(
            intent_id=f"native-read-only:{candidate_id}",
            kind=capability_id,
            parameters=parameters,
            source_goal_id=goal.goal_id,
            expected_outcome=content.expected_outcome,
            confidence=confidence,
            tick=int(tick),
        )
        return ReadOnlyIntentDecision(
            accepted=True,
            candidate_id=candidate_id,
            reason_code="native_read_only_intent_ready",
            snapshot_id=snapshot_id,
            capability_revision=capability_revision,
            observation_digest=observation.observation_digest,
            action_intent=intent,
        )


__all__ = [
    "READ_ONLY_INTENT_CAPABILITIES",
    "READ_ONLY_INTENT_POLICY_FORMAT",
    "READ_ONLY_INTENT_POLICY_VERSION",
    "READ_ONLY_INTENT_PLANNER_FORMAT",
    "READ_ONLY_INTENT_PLANNER_VERSION",
    "ReadOnlyIntentPolicy",
    "ReadOnlyIntentDecision",
    "NativeReadOnlyIntentPlanner",
]
