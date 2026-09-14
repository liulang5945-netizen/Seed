"""Content-addressed owner transfer from K1 candidates to a native G state.

P3.2 does not add a learner.  The existing K1 semantic learner remains the
source of candidate evidence, while ``GSelectionState`` becomes the explicit
owner of the selected goal/content pair used by the action boundary.  This
keeps the useful inherited representation and makes the capability boundary
auditable before any G-specific fit is considered.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import Goal
from .generation import ContentPlan
from .internalization import content_digest
from .semantic_training import StructuredSemanticResult
from .single_cell import (
    _budget_pairs,
    _digest,
    _digest_pairs,
    _state_pairs,
    _tags,
    _text,
    _text_pairs,
)

TAIJI_OWNER_TRANSFER_FORMAT = "taiji-skg-owner-transfer-v1"
TAIJI_OWNER_TRANSFER_VERSION = 1
OWNER_TRANSFER_OWNER_IDS = ("S", "G", "K")
OWNER_TRANSFER_EVENT_TYPES = ("observation", "k_candidate", "g_selection", "action")
OWNER_TRANSFER_CURSOR_STAGES = (*OWNER_TRANSFER_EVENT_TYPES, "complete")
OWNER_TRANSFER_EVENT_OWNERS = {
    "observation": ((), ("S",)),
    "k_candidate": (("S",), ("K",)),
    "g_selection": (("K",), ("G",)),
    "action": (("G", "K"), ()),
}
OWNER_TRANSFER_ROLES = tuple(
    sorted(
        (
            ("S", "runtime_evidence"),
            ("G", "selection_owner"),
            ("K", "learned_candidate_provider"),
        )
    )
)


def _float_pairs(value: Any, name: str) -> tuple[tuple[str, float], ...]:
    if isinstance(value, Mapping):
        items = tuple(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = tuple(value)
    else:
        raise TypeError(f"{name} must be a mapping or pair sequence")
    normalized = tuple((_text(item[0], f"{name} key"), float(item[1])) for item in items)
    if len({key for key, _ in normalized}) != len(normalized):
        raise ValueError(f"{name} must contain unique keys")
    return tuple(sorted(normalized))


def _optional_goal(value: Any, name: str) -> Goal | None:
    if value is None:
        return None
    if isinstance(value, Goal):
        return value
    if isinstance(value, Mapping):
        return Goal.from_payload(value)
    raise TypeError(f"{name} must be a Goal or None")


def _optional_content(value: Any, name: str) -> ContentPlan | None:
    if value is None:
        return None
    if isinstance(value, ContentPlan):
        return value
    if isinstance(value, Mapping):
        return ContentPlan.from_payload(value)
    raise TypeError(f"{name} must be a ContentPlan or None")


@dataclass(frozen=True)
class GSelectionState:
    """Native G-owned selection derived only from a K candidate payload."""

    candidate_digest: str
    goal_scores: tuple[tuple[str, float], ...]
    content_scores: tuple[tuple[str, float], ...]
    selected_goal: Goal | None
    selected_content: ContentPlan | None
    selection_status: str
    confidence: float
    ambiguity: float
    external_target_used: bool
    selection_digest: str
    format: str = TAIJI_OWNER_TRANSFER_FORMAT
    version: int = TAIJI_OWNER_TRANSFER_VERSION

    def __post_init__(self) -> None:
        if self.format != TAIJI_OWNER_TRANSFER_FORMAT:
            raise ValueError("unsupported owner-transfer selection format")
        if int(self.version) != TAIJI_OWNER_TRANSFER_VERSION:
            raise ValueError("unsupported owner-transfer selection version")
        candidate_digest = _digest(self.candidate_digest, "candidate_digest")
        goal_scores = _float_pairs(self.goal_scores, "goal_scores")
        content_scores = _float_pairs(self.content_scores, "content_scores")
        selected_goal = _optional_goal(self.selected_goal, "selected_goal")
        selected_content = _optional_content(self.selected_content, "selected_content")
        status = _text(self.selection_status, "selection_status")
        if status not in {"selected", "abstained"}:
            raise ValueError("owner-transfer selection status must be selected or abstained")
        if status == "selected" and (selected_goal is None or selected_content is None):
            raise ValueError("selected owner-transfer state requires goal and content")
        if status == "abstained" and (selected_goal is not None or selected_content is not None):
            raise ValueError("abstained owner-transfer state cannot own a selection")
        if bool(self.external_target_used):
            raise ValueError("owner-transfer G state cannot use an external target")
        confidence = float(self.confidence)
        ambiguity = float(self.ambiguity)
        if not 0.0 <= confidence <= 1.0 or not 0.0 <= ambiguity <= 1.0:
            raise ValueError("owner-transfer confidence and ambiguity must be in [0, 1]")
        unsigned = {
            "format": self.format,
            "version": int(self.version),
            "candidate_digest": candidate_digest,
            "goal_scores": dict(goal_scores),
            "content_scores": dict(content_scores),
            "selected_goal": None if selected_goal is None else selected_goal.to_payload(),
            "selected_content": (
                None if selected_content is None else selected_content.to_payload()
            ),
            "selection_status": status,
            "confidence": confidence,
            "ambiguity": ambiguity,
            "external_target_used": False,
        }
        if _digest(self.selection_digest, "selection_digest") != content_digest(unsigned):
            raise ValueError("owner-transfer selection digest mismatch")
        object.__setattr__(self, "candidate_digest", candidate_digest)
        object.__setattr__(self, "goal_scores", goal_scores)
        object.__setattr__(self, "content_scores", content_scores)
        object.__setattr__(self, "selected_goal", selected_goal)
        object.__setattr__(self, "selected_content", selected_content)
        object.__setattr__(self, "selection_status", status)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "ambiguity", ambiguity)
        object.__setattr__(self, "external_target_used", False)
        object.__setattr__(self, "selection_digest", _digest(self.selection_digest, "selection_digest"))

    @classmethod
    def from_k1_result(cls, result: StructuredSemanticResult) -> GSelectionState:
        if not isinstance(result, StructuredSemanticResult):
            raise TypeError("owner-transfer G state requires a StructuredSemanticResult")
        candidate = result.to_payload()
        selected = (
            result.status in {"resolved", "clarify"}
            and result.goal is not None
            and result.content_plan is not None
        )
        candidate_digest = content_digest(candidate)
        goal_scores = _float_pairs(result.goal_scores, "goal_scores")
        content_scores = _float_pairs(result.content_scores, "content_scores")
        selected_goal = result.goal.to_payload() if selected and result.goal is not None else None
        selected_content = (
            result.content_plan.to_payload()
            if selected and result.content_plan is not None
            else None
        )
        selection_status = "selected" if selected else "abstained"
        confidence = float(result.confidence)
        ambiguity = float(result.ambiguity)
        unsigned = {
            "format": TAIJI_OWNER_TRANSFER_FORMAT,
            "version": TAIJI_OWNER_TRANSFER_VERSION,
            "candidate_digest": candidate_digest,
            "goal_scores": dict(goal_scores),
            "content_scores": dict(content_scores),
            "selected_goal": selected_goal,
            "selected_content": selected_content,
            "selection_status": selection_status,
            "confidence": confidence,
            "ambiguity": ambiguity,
            "external_target_used": False,
        }
        return cls(
            candidate_digest=candidate_digest,
            goal_scores=goal_scores,
            content_scores=content_scores,
            selected_goal=_optional_goal(selected_goal, "selected_goal"),
            selected_content=_optional_content(selected_content, "selected_content"),
            selection_status=selection_status,
            confidence=confidence,
            ambiguity=ambiguity,
            external_target_used=False,
            selection_digest=content_digest(unsigned),
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": int(self.version),
            "candidate_digest": self.candidate_digest,
            "goal_scores": dict(self.goal_scores),
            "content_scores": dict(self.content_scores),
            "selected_goal": None if self.selected_goal is None else self.selected_goal.to_payload(),
            "selected_content": (
                None if self.selected_content is None else self.selected_content.to_payload()
            ),
            "selection_status": self.selection_status,
            "confidence": self.confidence,
            "ambiguity": self.ambiguity,
            "external_target_used": False,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "selection_digest": self.selection_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GSelectionState:
        item = cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            candidate_digest=str(payload["candidate_digest"]),
            goal_scores=tuple(payload.get("goal_scores", {}).items()),
            content_scores=tuple(payload.get("content_scores", {}).items()),
            selected_goal=_optional_goal(payload.get("selected_goal"), "selected_goal"),
            selected_content=_optional_content(payload.get("selected_content"), "selected_content"),
            selection_status=str(payload["selection_status"]),
            confidence=float(payload["confidence"]),
            ambiguity=float(payload["ambiguity"]),
            external_target_used=bool(payload.get("external_target_used", False)),
            selection_digest=str(payload["selection_digest"]),
        )
        if content_digest(item._payload_without_digest()) != item.selection_digest:
            raise ValueError("normalized owner-transfer selection digest mismatch")
        return item


@dataclass(frozen=True)
class TaijiOwnerTransferManifest:
    """Manifest binding the P3.1 cell to the K→G owner transfer."""

    base_single_cell_manifest_digest: str
    base_continuation_checkpoint_digest: str
    worker_checkpoint_digests: tuple[tuple[str, str], ...]
    owner_roles: tuple[tuple[str, str], ...]
    event_types: tuple[str, ...]
    external_target_used: bool
    manifest_digest: str
    format: str = TAIJI_OWNER_TRANSFER_FORMAT
    version: int = TAIJI_OWNER_TRANSFER_VERSION

    def __post_init__(self) -> None:
        if self.format != TAIJI_OWNER_TRANSFER_FORMAT:
            raise ValueError("unsupported owner-transfer manifest format")
        if int(self.version) != TAIJI_OWNER_TRANSFER_VERSION:
            raise ValueError("unsupported owner-transfer manifest version")
        base_cell = _digest(self.base_single_cell_manifest_digest, "base_single_cell_manifest_digest")
        base = _digest(self.base_continuation_checkpoint_digest, "base_continuation_checkpoint_digest")
        workers = _digest_pairs(self.worker_checkpoint_digests, "worker_checkpoint_digests")
        if tuple(key for key, _ in workers) != ("k1", "k2"):
            raise ValueError("owner-transfer manifest requires k1 and k2 workers")
        roles = _text_pairs(self.owner_roles, "owner_roles")
        if roles != OWNER_TRANSFER_ROLES:
            raise ValueError("owner-transfer roles do not match S/G/K ownership")
        event_types = tuple(str(value) for value in self.event_types)
        if event_types != OWNER_TRANSFER_EVENT_TYPES:
            raise ValueError("owner-transfer event types do not match the contract")
        if bool(self.external_target_used):
            raise ValueError("owner-transfer manifest cannot allow external targets")
        unsigned = {
            "format": self.format,
            "version": int(self.version),
            "base_single_cell_manifest_digest": base_cell,
            "base_continuation_checkpoint_digest": base,
            "worker_checkpoint_digests": dict(workers),
            "owner_roles": dict(roles),
            "event_types": list(event_types),
            "external_target_used": False,
        }
        if _digest(self.manifest_digest, "manifest_digest") != content_digest(unsigned):
            raise ValueError("owner-transfer manifest digest mismatch")
        object.__setattr__(self, "base_single_cell_manifest_digest", base_cell)
        object.__setattr__(self, "base_continuation_checkpoint_digest", base)
        object.__setattr__(self, "worker_checkpoint_digests", workers)
        object.__setattr__(self, "owner_roles", roles)
        object.__setattr__(self, "event_types", event_types)
        object.__setattr__(self, "external_target_used", False)
        object.__setattr__(self, "manifest_digest", _digest(self.manifest_digest, "manifest_digest"))

    @classmethod
    def create(
        cls,
        *,
        base_single_cell_manifest_digest: str,
        base_continuation_checkpoint_digest: str,
        worker_checkpoint_digests: Mapping[str, str],
    ) -> TaijiOwnerTransferManifest:
        workers = _digest_pairs(worker_checkpoint_digests, "worker_checkpoint_digests")
        unsigned = {
            "format": TAIJI_OWNER_TRANSFER_FORMAT,
            "version": TAIJI_OWNER_TRANSFER_VERSION,
            "base_single_cell_manifest_digest": str(base_single_cell_manifest_digest),
            "base_continuation_checkpoint_digest": str(base_continuation_checkpoint_digest),
            "worker_checkpoint_digests": dict(workers),
            "owner_roles": dict(OWNER_TRANSFER_ROLES),
            "event_types": list(OWNER_TRANSFER_EVENT_TYPES),
            "external_target_used": False,
        }
        return cls(
            base_single_cell_manifest_digest=str(base_single_cell_manifest_digest),
            base_continuation_checkpoint_digest=str(base_continuation_checkpoint_digest),
            worker_checkpoint_digests=workers,
            owner_roles=OWNER_TRANSFER_ROLES,
            event_types=OWNER_TRANSFER_EVENT_TYPES,
            external_target_used=False,
            manifest_digest=content_digest(unsigned),
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": int(self.version),
            "base_single_cell_manifest_digest": self.base_single_cell_manifest_digest,
            "base_continuation_checkpoint_digest": self.base_continuation_checkpoint_digest,
            "worker_checkpoint_digests": dict(self.worker_checkpoint_digests),
            "owner_roles": dict(self.owner_roles),
            "event_types": list(self.event_types),
            "external_target_used": False,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "manifest_digest": self.manifest_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TaijiOwnerTransferManifest:
        return cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            base_single_cell_manifest_digest=str(payload["base_single_cell_manifest_digest"]),
            base_continuation_checkpoint_digest=str(payload["base_continuation_checkpoint_digest"]),
            worker_checkpoint_digests=tuple(payload["worker_checkpoint_digests"].items()),
            owner_roles=tuple(payload["owner_roles"].items()),
            event_types=tuple(str(value) for value in payload["event_types"]),
            external_target_used=bool(payload.get("external_target_used", False)),
            manifest_digest=str(payload["manifest_digest"]),
        )


@dataclass(frozen=True)
class OwnerTransferCursor:
    event_index: int
    event_total: int
    case_index: int
    stage: str

    def __post_init__(self) -> None:
        event_index = int(self.event_index)
        event_total = int(self.event_total)
        case_index = int(self.case_index)
        stage = _text(self.stage, "owner-transfer cursor stage")
        if stage not in OWNER_TRANSFER_CURSOR_STAGES:
            raise ValueError(f"unsupported owner-transfer cursor stage: {stage}")
        if event_total < 0 or event_index < 0 or event_index > event_total:
            raise ValueError("owner-transfer cursor is outside the event stream")
        if case_index < 0:
            raise ValueError("owner-transfer cursor case index cannot be negative")
        if stage == "complete" and event_index != event_total:
            raise ValueError("complete owner-transfer cursor must consume the event stream")
        object.__setattr__(self, "event_index", event_index)
        object.__setattr__(self, "event_total", event_total)
        object.__setattr__(self, "case_index", case_index)
        object.__setattr__(self, "stage", stage)

    def to_payload(self) -> dict[str, Any]:
        return {
            "event_index": self.event_index,
            "event_total": self.event_total,
            "case_index": self.case_index,
            "stage": self.stage,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> OwnerTransferCursor:
        return cls(
            event_index=int(payload["event_index"]),
            event_total=int(payload["event_total"]),
            case_index=int(payload["case_index"]),
            stage=str(payload["stage"]),
        )


@dataclass(frozen=True)
class TaijiOwnerTransferEvent:
    """One owner-transfer event with an explicit S/G/K mask."""

    event_index: int
    event_type: str
    input_digest: str
    read_owners: tuple[str, ...]
    write_owners: tuple[str, ...]
    state_before: tuple[tuple[str, str | None], ...]
    state_after: tuple[tuple[str, str | None], ...]
    output_digest: str
    confidence: float
    status: str
    attributes: Mapping[str, Any]
    event_digest: str
    format: str = TAIJI_OWNER_TRANSFER_FORMAT
    version: int = TAIJI_OWNER_TRANSFER_VERSION

    def __post_init__(self) -> None:
        if self.format != TAIJI_OWNER_TRANSFER_FORMAT:
            raise ValueError("unsupported owner-transfer event format")
        if int(self.version) != TAIJI_OWNER_TRANSFER_VERSION:
            raise ValueError("unsupported owner-transfer event version")
        event_index = int(self.event_index)
        event_type = _text(self.event_type, "owner-transfer event_type")
        if event_index < 0 or event_type not in OWNER_TRANSFER_EVENT_TYPES:
            raise ValueError("invalid owner-transfer event identity")
        expected_reads, expected_writes = OWNER_TRANSFER_EVENT_OWNERS[event_type]
        reads = _tags(self.read_owners, "owner-transfer read_owners")
        writes = _tags(self.write_owners, "owner-transfer write_owners")
        if reads != expected_reads or writes != expected_writes:
            raise ValueError(f"owner-transfer {event_type} owner mask does not match its contract")
        before = _state_pairs(self.state_before, "owner-transfer state_before")
        after = _state_pairs(self.state_after, "owner-transfer state_after")
        input_digest = _digest(self.input_digest, "owner-transfer input_digest")
        output_digest = _digest(self.output_digest, "owner-transfer output_digest")
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("owner-transfer event confidence must be in [0, 1]")
        status = _text(self.status, "owner-transfer event status")
        attributes = dict(self.attributes)
        unsigned = {
            "format": self.format,
            "version": int(self.version),
            "event_index": event_index,
            "event_type": event_type,
            "input_digest": input_digest,
            "read_owners": list(reads),
            "write_owners": list(writes),
            "state_before": dict(before),
            "state_after": dict(after),
            "output_digest": output_digest,
            "confidence": confidence,
            "status": status,
            "attributes": attributes,
        }
        if _digest(self.event_digest, "owner-transfer event_digest") != content_digest(unsigned):
            raise ValueError("owner-transfer event digest mismatch")
        object.__setattr__(self, "event_index", event_index)
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "input_digest", input_digest)
        object.__setattr__(self, "read_owners", reads)
        object.__setattr__(self, "write_owners", writes)
        object.__setattr__(self, "state_before", before)
        object.__setattr__(self, "state_after", after)
        object.__setattr__(self, "output_digest", output_digest)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "attributes", attributes)
        object.__setattr__(self, "event_digest", _digest(self.event_digest, "owner-transfer event_digest"))

    @classmethod
    def create(
        cls,
        *,
        event_index: int,
        event_type: str,
        input_payload: Mapping[str, Any],
        state_before: Mapping[str, str | None],
        state_after: Mapping[str, str | None],
        output_payload: Mapping[str, Any],
        confidence: float,
        status: str,
        attributes: Mapping[str, Any],
    ) -> TaijiOwnerTransferEvent:
        reads, writes = OWNER_TRANSFER_EVENT_OWNERS[str(event_type)]
        unsigned = {
            "format": TAIJI_OWNER_TRANSFER_FORMAT,
            "version": TAIJI_OWNER_TRANSFER_VERSION,
            "event_index": int(event_index),
            "event_type": str(event_type),
            "input_digest": content_digest(dict(input_payload)),
            "read_owners": list(reads),
            "write_owners": list(writes),
            "state_before": dict(state_before),
            "state_after": dict(state_after),
            "output_digest": content_digest(dict(output_payload)),
            "confidence": float(confidence),
            "status": str(status),
            "attributes": dict(attributes),
        }
        return cls.from_payload({**unsigned, "event_digest": content_digest(unsigned)})

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": int(self.version),
            "event_index": self.event_index,
            "event_type": self.event_type,
            "input_digest": self.input_digest,
            "read_owners": list(self.read_owners),
            "write_owners": list(self.write_owners),
            "state_before": dict(self.state_before),
            "state_after": dict(self.state_after),
            "output_digest": self.output_digest,
            "confidence": self.confidence,
            "status": self.status,
            "attributes": dict(self.attributes),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "event_digest": self.event_digest}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TaijiOwnerTransferEvent:
        item = cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            event_index=int(payload["event_index"]),
            event_type=str(payload["event_type"]),
            input_digest=str(payload["input_digest"]),
            read_owners=tuple(str(value) for value in payload["read_owners"]),
            write_owners=tuple(str(value) for value in payload["write_owners"]),
            state_before=tuple(payload["state_before"].items()),
            state_after=tuple(payload["state_after"].items()),
            output_digest=str(payload["output_digest"]),
            confidence=float(payload["confidence"]),
            status=str(payload["status"]),
            attributes=dict(payload.get("attributes", {})),
            event_digest=str(payload["event_digest"]),
        )
        if content_digest(item._payload_without_digest()) != item.event_digest:
            raise ValueError("normalized owner-transfer event digest mismatch")
        return item


@dataclass(frozen=True)
class TaijiOwnerTransferCheckpoint:
    """Restorable K→G owner-transfer boundary."""

    base_continuation_checkpoint_digest: str
    base_continuation_ref: str
    base_single_cell_manifest_digest: str
    base_single_cell_manifest_ref: str
    manifest_digest: str
    worker_checkpoint_digests: tuple[tuple[str, str], ...]
    worker_checkpoint_refs: tuple[tuple[str, str], ...]
    owner_state_digests: tuple[tuple[str, str], ...]
    owner_state_refs: tuple[tuple[str, str], ...]
    event_digests: tuple[str, ...]
    event_refs: tuple[str, ...]
    cursor: OwnerTransferCursor
    rng_state: Any
    rng_state_digest: str
    budget_counts: tuple[tuple[str, int], ...]
    lineage_chain: tuple[str, ...]
    checkpoint_digest: str
    format: str = TAIJI_OWNER_TRANSFER_FORMAT
    version: int = TAIJI_OWNER_TRANSFER_VERSION

    def __post_init__(self) -> None:
        if self.format != TAIJI_OWNER_TRANSFER_FORMAT:
            raise ValueError("unsupported owner-transfer checkpoint format")
        if int(self.version) != TAIJI_OWNER_TRANSFER_VERSION:
            raise ValueError("unsupported owner-transfer checkpoint version")
        base = _digest(self.base_continuation_checkpoint_digest, "base_continuation_checkpoint_digest")
        base_cell = _digest(self.base_single_cell_manifest_digest, "base_single_cell_manifest_digest")
        manifest = _digest(self.manifest_digest, "manifest_digest")
        workers = _digest_pairs(self.worker_checkpoint_digests, "worker_checkpoint_digests")
        worker_refs = _text_pairs(self.worker_checkpoint_refs, "worker_checkpoint_refs")
        if tuple(key for key, _ in workers) != ("k1", "k2"):
            raise ValueError("owner-transfer checkpoint requires k1 and k2 workers")
        if tuple(key for key, _ in workers) != tuple(key for key, _ in worker_refs):
            raise ValueError("owner-transfer worker digest/ref owners do not match")
        owner_digests = _digest_pairs(self.owner_state_digests, "owner_state_digests")
        owner_refs = _text_pairs(self.owner_state_refs, "owner_state_refs")
        if set(key for key, _ in owner_digests) != set(OWNER_TRANSFER_OWNER_IDS):
            raise ValueError("owner-transfer owner state digests must contain S/G/K")
        if set(key for key, _ in owner_refs) != set(OWNER_TRANSFER_OWNER_IDS):
            raise ValueError("owner-transfer owner state refs must contain S/G/K")
        digest_map = dict(owner_digests)
        ref_map = dict(owner_refs)
        owner_digests = tuple((owner, digest_map[owner]) for owner in OWNER_TRANSFER_OWNER_IDS)
        owner_refs = tuple((owner, ref_map[owner]) for owner in OWNER_TRANSFER_OWNER_IDS)
        event_digests = tuple(_digest(item, "owner-transfer event digest") for item in self.event_digests)
        event_refs = tuple(_text(item, "owner-transfer event ref") for item in self.event_refs)
        if len(event_digests) != len(event_refs):
            raise ValueError("owner-transfer event digest/ref lengths differ")
        base_ref = _text(self.base_continuation_ref, "base_continuation_ref")
        base_cell_ref = _text(self.base_single_cell_manifest_ref, "base_single_cell_manifest_ref")
        rng_digest = _digest(self.rng_state_digest, "rng_state_digest")
        if content_digest(self.rng_state) != rng_digest:
            raise ValueError("owner-transfer RNG state digest mismatch")
        budgets = _budget_pairs(self.budget_counts, "budget_counts")
        lineage = tuple(_digest(item, "lineage item") for item in self.lineage_chain)
        if lineage != (base, base_cell, manifest):
            raise ValueError("owner-transfer lineage must bind continuation, cell, and manifest")
        checkpoint = _digest(self.checkpoint_digest, "checkpoint_digest")
        object.__setattr__(self, "base_continuation_checkpoint_digest", base)
        object.__setattr__(self, "base_continuation_ref", base_ref)
        object.__setattr__(self, "base_single_cell_manifest_digest", base_cell)
        object.__setattr__(self, "base_single_cell_manifest_ref", base_cell_ref)
        object.__setattr__(self, "manifest_digest", manifest)
        object.__setattr__(self, "worker_checkpoint_digests", workers)
        object.__setattr__(self, "worker_checkpoint_refs", worker_refs)
        object.__setattr__(self, "owner_state_digests", owner_digests)
        object.__setattr__(self, "owner_state_refs", owner_refs)
        object.__setattr__(self, "event_digests", event_digests)
        object.__setattr__(self, "event_refs", event_refs)
        object.__setattr__(self, "rng_state_digest", rng_digest)
        object.__setattr__(self, "budget_counts", budgets)
        object.__setattr__(self, "lineage_chain", lineage)
        object.__setattr__(self, "checkpoint_digest", checkpoint)

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": int(self.version),
            "base_continuation_checkpoint_digest": self.base_continuation_checkpoint_digest,
            "base_continuation_ref": self.base_continuation_ref,
            "base_single_cell_manifest_digest": self.base_single_cell_manifest_digest,
            "base_single_cell_manifest_ref": self.base_single_cell_manifest_ref,
            "manifest_digest": self.manifest_digest,
            "worker_checkpoint_digests": dict(self.worker_checkpoint_digests),
            "worker_checkpoint_refs": dict(self.worker_checkpoint_refs),
            "owner_state_digests": dict(self.owner_state_digests),
            "owner_state_refs": dict(self.owner_state_refs),
            "event_digests": list(self.event_digests),
            "event_refs": list(self.event_refs),
            "cursor": self.cursor.to_payload(),
            "rng_state": self.rng_state,
            "rng_state_digest": self.rng_state_digest,
            "budget_counts": dict(self.budget_counts),
            "lineage_chain": list(self.lineage_chain),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "checkpoint_digest": self.checkpoint_digest}

    @property
    def logical_digest(self) -> str:
        payload = self._payload_without_digest()
        for key in (
            "base_continuation_ref",
            "base_single_cell_manifest_ref",
            "worker_checkpoint_refs",
            "owner_state_refs",
            "event_refs",
        ):
            payload.pop(key)
        return content_digest(payload)

    def assert_base(self, expected_digest: str) -> None:
        if _digest(expected_digest, "expected base digest") != self.base_continuation_checkpoint_digest:
            raise ValueError("owner-transfer checkpoint crosses the P3.0 continuation boundary")

    def assert_single_cell(self, expected_digest: str) -> None:
        if _digest(expected_digest, "expected single-cell digest") != self.base_single_cell_manifest_digest:
            raise ValueError("owner-transfer checkpoint crosses the P3.1 single-cell boundary")

    def assert_manifest(self, expected_digest: str) -> None:
        if _digest(expected_digest, "expected manifest digest") != self.manifest_digest:
            raise ValueError("owner-transfer checkpoint crosses its manifest")

    @classmethod
    def create(
        cls,
        *,
        base_continuation_checkpoint_digest: str,
        base_continuation_ref: str,
        base_single_cell_manifest_digest: str,
        base_single_cell_manifest_ref: str,
        manifest_digest: str,
        worker_checkpoint_digests: Mapping[str, str],
        worker_checkpoint_refs: Mapping[str, str],
        owner_state_digests: Mapping[str, str],
        owner_state_refs: Mapping[str, str],
        event_digests: Sequence[str],
        event_refs: Sequence[str],
        cursor: OwnerTransferCursor,
        rng_state: Any,
        budget_counts: Mapping[str, int],
        lineage_chain: Sequence[str],
    ) -> TaijiOwnerTransferCheckpoint:
        provisional = {
            "format": TAIJI_OWNER_TRANSFER_FORMAT,
            "version": TAIJI_OWNER_TRANSFER_VERSION,
            "base_continuation_checkpoint_digest": _digest(
                base_continuation_checkpoint_digest, "base_continuation_checkpoint_digest"
            ),
            "base_continuation_ref": _text(base_continuation_ref, "base_continuation_ref"),
            "base_single_cell_manifest_digest": _digest(
                base_single_cell_manifest_digest, "base_single_cell_manifest_digest"
            ),
            "base_single_cell_manifest_ref": _text(
                base_single_cell_manifest_ref, "base_single_cell_manifest_ref"
            ),
            "manifest_digest": _digest(manifest_digest, "manifest_digest"),
            "worker_checkpoint_digests": dict(
                _digest_pairs(worker_checkpoint_digests, "worker_checkpoint_digests")
            ),
            "worker_checkpoint_refs": dict(_text_pairs(worker_checkpoint_refs, "worker_checkpoint_refs")),
            "owner_state_digests": dict(_digest_pairs(owner_state_digests, "owner_state_digests")),
            "owner_state_refs": dict(_text_pairs(owner_state_refs, "owner_state_refs")),
            "event_digests": list(_digest(item, "owner-transfer event digest") for item in event_digests),
            "event_refs": list(_text(item, "owner-transfer event ref") for item in event_refs),
            "cursor": cursor.to_payload(),
            "rng_state": rng_state,
            "rng_state_digest": content_digest(rng_state),
            "budget_counts": dict(_budget_pairs(budget_counts, "budget_counts")),
            "lineage_chain": list(_digest(item, "lineage item") for item in lineage_chain),
        }
        checkpoint_digest = content_digest(provisional)
        provisional.pop("cursor")
        return cls(
            **provisional,
            cursor=cursor,
            checkpoint_digest=checkpoint_digest,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TaijiOwnerTransferCheckpoint:
        unsigned = {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        expected = content_digest(unsigned)
        if str(payload.get("checkpoint_digest", "")) != expected:
            raise ValueError("owner-transfer checkpoint digest mismatch")
        item = cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            base_continuation_checkpoint_digest=str(payload["base_continuation_checkpoint_digest"]),
            base_continuation_ref=str(payload["base_continuation_ref"]),
            base_single_cell_manifest_digest=str(payload["base_single_cell_manifest_digest"]),
            base_single_cell_manifest_ref=str(payload["base_single_cell_manifest_ref"]),
            manifest_digest=str(payload["manifest_digest"]),
            worker_checkpoint_digests=tuple(payload["worker_checkpoint_digests"].items()),
            worker_checkpoint_refs=tuple(payload["worker_checkpoint_refs"].items()),
            owner_state_digests=tuple(payload["owner_state_digests"].items()),
            owner_state_refs=tuple(payload["owner_state_refs"].items()),
            event_digests=tuple(payload.get("event_digests", ())),
            event_refs=tuple(payload.get("event_refs", ())),
            cursor=OwnerTransferCursor.from_payload(payload["cursor"]),
            rng_state=payload["rng_state"],
            rng_state_digest=str(payload["rng_state_digest"]),
            budget_counts=tuple(payload["budget_counts"].items()),
            lineage_chain=tuple(payload["lineage_chain"]),
            checkpoint_digest=str(payload["checkpoint_digest"]),
        )
        if content_digest(item._payload_without_digest()) != item.checkpoint_digest:
            raise ValueError("normalized owner-transfer checkpoint digest mismatch")
        return item


__all__ = [
    "GSelectionState",
    "OWNER_TRANSFER_CURSOR_STAGES",
    "OWNER_TRANSFER_EVENT_TYPES",
    "OWNER_TRANSFER_OWNER_IDS",
    "TAIJI_OWNER_TRANSFER_FORMAT",
    "TAIJI_OWNER_TRANSFER_VERSION",
    "TaijiOwnerTransferCheckpoint",
    "TaijiOwnerTransferEvent",
    "TaijiOwnerTransferManifest",
    "OwnerTransferCursor",
]
