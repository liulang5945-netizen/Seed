"""Content-addressed S/G/K single-cell state and replay contracts.

P3.1 is intentionally a state-integration preflight, not a new learner.  The
contract makes the current ownership boundary explicit:

* ``S`` owns observation/evidence snapshots;
* ``G`` owns the externally supplied goal-selection context used by the
  validation replay (it is control-only until a learned goal owner exists);
* ``K`` owns the existing learned K1/K2 worker outputs and their checkpoint
  references.

The contract refuses implicit ownership.  Every event declares its readers,
writers, before/after owner digests, and output digest.  Checkpoints reference
the P3.0 continuation boundary and are independently verifiable without
re-running inference or silently inventing an S/G learner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .internalization import content_digest

TAIJI_SINGLE_CELL_FORMAT = "taiji-sgk-single-cell-v1"
TAIJI_SINGLE_CELL_VERSION = 1
SINGLE_CELL_OWNER_IDS = ("S", "G", "K")
SINGLE_CELL_EVENT_TYPES = (
    "observation",
    "s_update",
    "g_select",
    "k_readout",
    "action",
)
SINGLE_CELL_CURSOR_STAGES = (*SINGLE_CELL_EVENT_TYPES, "complete")
SINGLE_CELL_EVENT_OWNERS = {
    "observation": ((), ("S",)),
    "s_update": (("S",), ("S",)),
    "g_select": (("S",), ("G",)),
    "k_readout": (("S",), ("K",)),
    "action": (("G", "K"), ()),
}


def _text(value: Any, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _digest(value: Any, name: str) -> str:
    normalized = _text(value, name)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _tags(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    normalized = tuple(_text(item, f"{name} item") for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def _scope_tags(value: Any, name: str) -> tuple[str, ...]:
    return tuple(sorted(_tags(value, name)))


def _digest_pairs(value: Any, name: str) -> tuple[tuple[str, str], ...]:
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = value
    else:
        raise TypeError(f"{name} must be a mapping or pair sequence")
    normalized = tuple(
        (_text(item[0], f"{name} key"), _digest(item[1], f"{name} value"))
        for item in items
    )
    if len({key for key, _ in normalized}) != len(normalized):
        raise ValueError(f"{name} must contain unique keys")
    return tuple(sorted(normalized))


def _text_pairs(value: Any, name: str) -> tuple[tuple[str, str], ...]:
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = value
    else:
        raise TypeError(f"{name} must be a mapping or pair sequence")
    normalized = tuple(
        (_text(item[0], f"{name} key"), _text(item[1], f"{name} value"))
        for item in items
    )
    if len({key for key, _ in normalized}) != len(normalized):
        raise ValueError(f"{name} must contain unique keys")
    return tuple(sorted(normalized))


def _budget_pairs(value: Any, name: str) -> tuple[tuple[str, int], ...]:
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = value
    else:
        raise TypeError(f"{name} must be a mapping or pair sequence")
    normalized = tuple(
        (_text(item[0], f"{name} key"), int(item[1]))
        for item in items
    )
    if len({key for key, _ in normalized}) != len(normalized):
        raise ValueError(f"{name} must contain unique keys")
    if any(value < 0 for _, value in normalized):
        raise ValueError(f"{name} cannot contain negative counts")
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class SingleCellOwnerContract:
    """One explicit S/G/K ownership and mask contract."""

    owner_id: str
    owner_kind: str
    availability: str
    schema: str
    read_scopes: tuple[str, ...]
    write_scopes: tuple[str, ...]
    learning_owner: str
    contract_digest: str
    format: str = TAIJI_SINGLE_CELL_FORMAT
    version: int = TAIJI_SINGLE_CELL_VERSION

    def __post_init__(self) -> None:
        if self.format != TAIJI_SINGLE_CELL_FORMAT:
            raise ValueError("unsupported single-cell owner format")
        if int(self.version) != TAIJI_SINGLE_CELL_VERSION:
            raise ValueError("unsupported single-cell owner version")
        owner_id = _text(self.owner_id, "owner_id")
        if owner_id not in SINGLE_CELL_OWNER_IDS:
            raise ValueError(f"unsupported single-cell owner: {owner_id}")
        if self.availability not in {"active", "absent"}:
            raise ValueError("single-cell owner availability must be active or absent")
        owner_kind = _text(self.owner_kind, "owner_kind")
        schema = _text(self.schema, "owner schema")
        read_scopes = _scope_tags(self.read_scopes, "owner read_scopes")
        write_scopes = _scope_tags(self.write_scopes, "owner write_scopes")
        learning_owner = _text(self.learning_owner, "learning_owner")
        if self.availability == "active" and not write_scopes:
            raise ValueError("active single-cell owner must declare write scopes")
        if self.availability == "absent" and (read_scopes or write_scopes):
            raise ValueError("absent single-cell owner cannot declare active scopes")
        unsigned = {
            "format": self.format,
            "version": int(self.version),
            "owner_id": owner_id,
            "owner_kind": owner_kind,
            "availability": self.availability,
            "schema": schema,
            "read_scopes": list(read_scopes),
            "write_scopes": list(write_scopes),
            "learning_owner": learning_owner,
        }
        if _digest(self.contract_digest, "contract_digest") != content_digest(unsigned):
            raise ValueError("single-cell owner contract digest mismatch")
        object.__setattr__(self, "owner_id", owner_id)
        object.__setattr__(self, "owner_kind", owner_kind)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "read_scopes", read_scopes)
        object.__setattr__(self, "write_scopes", write_scopes)
        object.__setattr__(self, "learning_owner", learning_owner)
        object.__setattr__(self, "contract_digest", _digest(self.contract_digest, "contract_digest"))

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": int(self.version),
            "owner_id": self.owner_id,
            "owner_kind": self.owner_kind,
            "availability": self.availability,
            "schema": self.schema,
            "read_scopes": list(self.read_scopes),
            "write_scopes": list(self.write_scopes),
            "learning_owner": self.learning_owner,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "contract_digest": self.contract_digest}

    @classmethod
    def create(
        cls,
        *,
        owner_id: str,
        owner_kind: str,
        availability: str,
        schema: str,
        read_scopes: Sequence[str],
        write_scopes: Sequence[str],
        learning_owner: str,
    ) -> SingleCellOwnerContract:
        unsigned = {
            "format": TAIJI_SINGLE_CELL_FORMAT,
            "version": TAIJI_SINGLE_CELL_VERSION,
            "owner_id": str(owner_id),
            "owner_kind": str(owner_kind),
            "availability": str(availability),
            "schema": str(schema),
            "read_scopes": list(_scope_tags(read_scopes, "read_scopes")),
            "write_scopes": list(_scope_tags(write_scopes, "write_scopes")),
            "learning_owner": str(learning_owner),
        }
        return cls(**unsigned, contract_digest=content_digest(unsigned))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SingleCellOwnerContract:
        return cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            owner_id=str(payload["owner_id"]),
            owner_kind=str(payload["owner_kind"]),
            availability=str(payload["availability"]),
            schema=str(payload["schema"]),
            read_scopes=tuple(str(value) for value in payload.get("read_scopes", ())),
            write_scopes=tuple(str(value) for value in payload.get("write_scopes", ())),
            learning_owner=str(payload["learning_owner"]),
            contract_digest=str(payload["contract_digest"]),
        )


@dataclass(frozen=True)
class TaijiSingleCellManifest:
    """Content address for one explicit S/G/K state graph."""

    base_continuation_checkpoint_digest: str
    source_cohort_digest: str
    owner_contracts: tuple[SingleCellOwnerContract, ...]
    event_types: tuple[str, ...]
    rollback_parent_digest: str
    manifest_digest: str
    format: str = TAIJI_SINGLE_CELL_FORMAT
    version: int = TAIJI_SINGLE_CELL_VERSION

    def __post_init__(self) -> None:
        if self.format != TAIJI_SINGLE_CELL_FORMAT:
            raise ValueError("unsupported single-cell manifest format")
        if int(self.version) != TAIJI_SINGLE_CELL_VERSION:
            raise ValueError("unsupported single-cell manifest version")
        base = _digest(
            self.base_continuation_checkpoint_digest,
            "base_continuation_checkpoint_digest",
        )
        cohort = _digest(self.source_cohort_digest, "source_cohort_digest")
        rollback = _digest(self.rollback_parent_digest, "rollback_parent_digest")
        owners = tuple(self.owner_contracts)
        if tuple(item.owner_id for item in owners) != SINGLE_CELL_OWNER_IDS:
            raise ValueError("single-cell owners must be exactly S/G/K in order")
        if any(not isinstance(item, SingleCellOwnerContract) for item in owners):
            raise TypeError("single-cell manifest contains an invalid owner contract")
        event_types = tuple(str(value) for value in self.event_types)
        if event_types != SINGLE_CELL_EVENT_TYPES:
            raise ValueError("single-cell event types do not match the ownership contract")
        unsigned = {
            "format": self.format,
            "version": int(self.version),
            "base_continuation_checkpoint_digest": base,
            "source_cohort_digest": cohort,
            "owner_contracts": [item.to_payload() for item in owners],
            "event_types": list(event_types),
            "rollback_parent_digest": rollback,
        }
        if _digest(self.manifest_digest, "manifest_digest") != content_digest(unsigned):
            raise ValueError("single-cell manifest digest mismatch")
        object.__setattr__(self, "base_continuation_checkpoint_digest", base)
        object.__setattr__(self, "source_cohort_digest", cohort)
        object.__setattr__(self, "rollback_parent_digest", rollback)
        object.__setattr__(self, "owner_contracts", owners)
        object.__setattr__(self, "event_types", event_types)
        object.__setattr__(self, "manifest_digest", _digest(self.manifest_digest, "manifest_digest"))

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": int(self.version),
            "base_continuation_checkpoint_digest": self.base_continuation_checkpoint_digest,
            "source_cohort_digest": self.source_cohort_digest,
            "owner_contracts": [item.to_payload() for item in self.owner_contracts],
            "event_types": list(self.event_types),
            "rollback_parent_digest": self.rollback_parent_digest,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "manifest_digest": self.manifest_digest}

    @classmethod
    def create(
        cls,
        *,
        base_continuation_checkpoint_digest: str,
        source_cohort_digest: str,
        owner_contracts: Sequence[SingleCellOwnerContract],
        rollback_parent_digest: str,
    ) -> TaijiSingleCellManifest:
        owners = tuple(owner_contracts)
        unsigned = {
            "format": TAIJI_SINGLE_CELL_FORMAT,
            "version": TAIJI_SINGLE_CELL_VERSION,
            "base_continuation_checkpoint_digest": str(base_continuation_checkpoint_digest),
            "source_cohort_digest": str(source_cohort_digest),
            "owner_contracts": [item.to_payload() for item in owners],
            "event_types": list(SINGLE_CELL_EVENT_TYPES),
            "rollback_parent_digest": str(rollback_parent_digest),
        }
        return cls(
            base_continuation_checkpoint_digest=str(base_continuation_checkpoint_digest),
            source_cohort_digest=str(source_cohort_digest),
            owner_contracts=owners,
            event_types=SINGLE_CELL_EVENT_TYPES,
            rollback_parent_digest=str(rollback_parent_digest),
            manifest_digest=content_digest(unsigned),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TaijiSingleCellManifest:
        return cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            base_continuation_checkpoint_digest=str(
                payload["base_continuation_checkpoint_digest"]
            ),
            source_cohort_digest=str(payload["source_cohort_digest"]),
            owner_contracts=tuple(
                SingleCellOwnerContract.from_payload(item)
                for item in payload["owner_contracts"]
            ),
            event_types=tuple(str(value) for value in payload["event_types"]),
            rollback_parent_digest=str(payload["rollback_parent_digest"]),
            manifest_digest=str(payload["manifest_digest"]),
        )


def _state_pairs(value: Any, name: str) -> tuple[tuple[str, str | None], ...]:
    if isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = value
    else:
        raise TypeError(f"{name} must be a mapping or pair sequence")
    normalized = []
    for item in items:
        key = _text(item[0], f"{name} key")
        raw = item[1]
        normalized.append((key, None if raw is None else _digest(raw, f"{name}[{key}]")))
    if tuple(key for key, _ in normalized) != SINGLE_CELL_OWNER_IDS:
        raise ValueError(f"{name} must contain S/G/K in order")
    return tuple(normalized)


@dataclass(frozen=True)
class SingleCellEventCursor:
    """Position in the replayed single-cell event stream."""

    event_index: int
    event_total: int
    case_index: int
    stage: str

    def __post_init__(self) -> None:
        event_index = int(self.event_index)
        event_total = int(self.event_total)
        case_index = int(self.case_index)
        stage = _text(self.stage, "single-cell cursor stage")
        if stage not in SINGLE_CELL_CURSOR_STAGES:
            raise ValueError(f"unsupported single-cell cursor stage: {stage}")
        if event_total < 0 or event_index < 0 or event_index > event_total:
            raise ValueError("single-cell cursor is outside the event stream")
        if case_index < 0:
            raise ValueError("single-cell cursor case index cannot be negative")
        if stage == "complete" and event_index != event_total:
            raise ValueError("complete single-cell cursor must consume the event stream")
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
    def from_payload(cls, payload: Mapping[str, Any]) -> SingleCellEventCursor:
        return cls(
            event_index=int(payload["event_index"]),
            event_total=int(payload["event_total"]),
            case_index=int(payload["case_index"]),
            stage=str(payload["stage"]),
        )


@dataclass(frozen=True)
class TaijiSingleCellEvent:
    """One auditable state transition inside the S/G/K cell."""

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
    format: str = TAIJI_SINGLE_CELL_FORMAT
    version: int = TAIJI_SINGLE_CELL_VERSION

    def __post_init__(self) -> None:
        if self.format != TAIJI_SINGLE_CELL_FORMAT:
            raise ValueError("unsupported single-cell event format")
        if int(self.version) != TAIJI_SINGLE_CELL_VERSION:
            raise ValueError("unsupported single-cell event version")
        event_index = int(self.event_index)
        event_type = _text(self.event_type, "event_type")
        if event_index < 0 or event_type not in SINGLE_CELL_EVENT_TYPES:
            raise ValueError("invalid single-cell event identity")
        expected_reads, expected_writes = SINGLE_CELL_EVENT_OWNERS[event_type]
        reads = _tags(self.read_owners, "read_owners")
        writes = _tags(self.write_owners, "write_owners")
        if reads != expected_reads or writes != expected_writes:
            raise ValueError(f"single-cell {event_type} owner mask does not match its contract")
        before = _state_pairs(self.state_before, "state_before")
        after = _state_pairs(self.state_after, "state_after")
        input_digest = _digest(self.input_digest, "input_digest")
        output_digest = _digest(self.output_digest, "output_digest")
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("single-cell event confidence must be in [0, 1]")
        status = _text(self.status, "event status")
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
        if _digest(self.event_digest, "event_digest") != content_digest(unsigned):
            raise ValueError("single-cell event digest mismatch")
        object.__setattr__(self, "event_index", event_index)
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "read_owners", reads)
        object.__setattr__(self, "write_owners", writes)
        object.__setattr__(self, "state_before", before)
        object.__setattr__(self, "state_after", after)
        object.__setattr__(self, "input_digest", input_digest)
        object.__setattr__(self, "output_digest", output_digest)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "attributes", attributes)
        object.__setattr__(self, "event_digest", _digest(self.event_digest, "event_digest"))

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
    def create(
        cls,
        *,
        event_index: int,
        event_type: str,
        input_digest: str,
        state_before: Mapping[str, str | None],
        state_after: Mapping[str, str | None],
        output_digest: str,
        confidence: float,
        status: str,
        attributes: Mapping[str, Any],
    ) -> TaijiSingleCellEvent:
        reads, writes = SINGLE_CELL_EVENT_OWNERS[str(event_type)]
        unsigned = {
            "format": TAIJI_SINGLE_CELL_FORMAT,
            "version": TAIJI_SINGLE_CELL_VERSION,
            "event_index": int(event_index),
            "event_type": str(event_type),
            "input_digest": str(input_digest),
            "read_owners": list(reads),
            "write_owners": list(writes),
            "state_before": dict(state_before),
            "state_after": dict(state_after),
            "output_digest": str(output_digest),
            "confidence": float(confidence),
            "status": str(status),
            "attributes": dict(attributes),
        }
        return cls(
            **unsigned,
            event_digest=content_digest(unsigned),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TaijiSingleCellEvent:
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
            raise ValueError("single-cell normalized event digest mismatch")
        return item


@dataclass(frozen=True)
class TaijiSingleCellCheckpoint:
    """Restorable S/G/K boundary anchored to one P3.0 continuation state."""

    base_continuation_checkpoint_digest: str
    base_continuation_ref: str
    manifest_digest: str
    worker_checkpoint_digests: tuple[tuple[str, str], ...]
    worker_checkpoint_refs: tuple[tuple[str, str], ...]
    owner_state_digests: tuple[tuple[str, str], ...]
    owner_state_refs: tuple[tuple[str, str], ...]
    event_digests: tuple[str, ...]
    event_refs: tuple[str, ...]
    cursor: SingleCellEventCursor
    rng_state: Any
    rng_state_digest: str
    budget_counts: tuple[tuple[str, int], ...]
    lineage_chain: tuple[str, ...]
    checkpoint_digest: str
    format: str = TAIJI_SINGLE_CELL_FORMAT
    version: int = TAIJI_SINGLE_CELL_VERSION

    def __post_init__(self) -> None:
        if self.format != TAIJI_SINGLE_CELL_FORMAT:
            raise ValueError("unsupported single-cell checkpoint format")
        if int(self.version) != TAIJI_SINGLE_CELL_VERSION:
            raise ValueError("unsupported single-cell checkpoint version")
        base = _digest(
            self.base_continuation_checkpoint_digest,
            "base_continuation_checkpoint_digest",
        )
        manifest = _digest(self.manifest_digest, "manifest_digest")
        worker_digests = _digest_pairs(self.worker_checkpoint_digests, "worker_checkpoint_digests")
        worker_refs = _text_pairs(self.worker_checkpoint_refs, "worker_checkpoint_refs")
        if tuple(key for key, _ in worker_digests) != tuple(key for key, _ in worker_refs):
            raise ValueError("single-cell worker digest/ref owners do not match")
        if len(worker_digests) < 2:
            raise ValueError("single-cell checkpoint requires at least K1/K2 workers")
        owner_digests = _digest_pairs(self.owner_state_digests, "owner_state_digests")
        owner_refs = _text_pairs(self.owner_state_refs, "owner_state_refs")
        if set(key for key, _ in owner_digests) != set(SINGLE_CELL_OWNER_IDS):
            raise ValueError("single-cell owner state digests must contain S/G/K")
        if set(key for key, _ in owner_refs) != set(SINGLE_CELL_OWNER_IDS):
            raise ValueError("single-cell owner state refs must contain S/G/K")
        owner_digest_map = dict(owner_digests)
        owner_ref_map = dict(owner_refs)
        owner_digests = tuple((owner, owner_digest_map[owner]) for owner in SINGLE_CELL_OWNER_IDS)
        owner_refs = tuple((owner, owner_ref_map[owner]) for owner in SINGLE_CELL_OWNER_IDS)
        event_digests = tuple(_digest(item, "event digest") for item in self.event_digests)
        event_refs = tuple(_text(item, "event ref") for item in self.event_refs)
        if len(event_digests) != len(event_refs):
            raise ValueError("single-cell event digest/ref lengths differ")
        base_ref = _text(self.base_continuation_ref, "base_continuation_ref")
        rng_digest = _digest(self.rng_state_digest, "rng_state_digest")
        if content_digest(self.rng_state) != rng_digest:
            raise ValueError("single-cell RNG state digest mismatch")
        normalized_budgets = _budget_pairs(self.budget_counts, "budget_counts")
        lineage = tuple(_digest(item, "lineage item") for item in self.lineage_chain)
        if not lineage or lineage[0] != base or lineage[-1] != manifest:
            raise ValueError("single-cell lineage must bind base continuation to manifest")
        checkpoint = _digest(self.checkpoint_digest, "checkpoint_digest")
        object.__setattr__(self, "base_continuation_checkpoint_digest", base)
        object.__setattr__(self, "base_continuation_ref", base_ref)
        object.__setattr__(self, "manifest_digest", manifest)
        object.__setattr__(self, "worker_checkpoint_digests", worker_digests)
        object.__setattr__(self, "worker_checkpoint_refs", worker_refs)
        object.__setattr__(self, "owner_state_digests", owner_digests)
        object.__setattr__(self, "owner_state_refs", owner_refs)
        object.__setattr__(self, "event_digests", event_digests)
        object.__setattr__(self, "event_refs", event_refs)
        object.__setattr__(self, "rng_state_digest", rng_digest)
        object.__setattr__(self, "budget_counts", tuple(normalized_budgets))
        object.__setattr__(self, "lineage_chain", lineage)
        object.__setattr__(self, "checkpoint_digest", checkpoint)

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": int(self.version),
            "base_continuation_checkpoint_digest": self.base_continuation_checkpoint_digest,
            "base_continuation_ref": self.base_continuation_ref,
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
        """Digest of state, excluding machine-local file paths."""

        payload = self._payload_without_digest()
        payload.pop("base_continuation_ref")
        payload.pop("worker_checkpoint_refs")
        payload.pop("owner_state_refs")
        payload.pop("event_refs")
        return content_digest(payload)

    def assert_base(self, expected_digest: str) -> None:
        if _digest(expected_digest, "expected base digest") != self.base_continuation_checkpoint_digest:
            raise ValueError("single-cell checkpoint crosses the P3.0 continuation boundary")

    def assert_manifest(self, expected_digest: str) -> None:
        if _digest(expected_digest, "expected manifest digest") != self.manifest_digest:
            raise ValueError("single-cell checkpoint crosses its manifest")

    @classmethod
    def create(
        cls,
        *,
        base_continuation_checkpoint_digest: str,
        base_continuation_ref: str,
        manifest_digest: str,
        worker_checkpoint_digests: Mapping[str, str] | Sequence[tuple[str, str]],
        worker_checkpoint_refs: Mapping[str, str] | Sequence[tuple[str, str]],
        owner_state_digests: Mapping[str, str] | Sequence[tuple[str, str]],
        owner_state_refs: Mapping[str, str] | Sequence[tuple[str, str]],
        event_digests: Sequence[str],
        event_refs: Sequence[str],
        cursor: SingleCellEventCursor,
        rng_state: Any,
        budget_counts: Mapping[str, int] | Sequence[tuple[str, int]],
        lineage_chain: Sequence[str],
    ) -> TaijiSingleCellCheckpoint:
        owner_digest_items = _digest_pairs(owner_state_digests, "owner_state_digests")
        owner_ref_items = _text_pairs(owner_state_refs, "owner_state_refs")
        owner_digest_map = dict(owner_digest_items)
        owner_ref_map = dict(owner_ref_items)
        if set(owner_digest_map) != set(SINGLE_CELL_OWNER_IDS) or set(owner_ref_map) != set(
            SINGLE_CELL_OWNER_IDS
        ):
            raise ValueError("single-cell owner state inputs must contain S/G/K")
        provisional = {
            "format": TAIJI_SINGLE_CELL_FORMAT,
            "version": TAIJI_SINGLE_CELL_VERSION,
            "base_continuation_checkpoint_digest": _digest(
                base_continuation_checkpoint_digest,
                "base_continuation_checkpoint_digest",
            ),
            "base_continuation_ref": _text(base_continuation_ref, "base_continuation_ref"),
            "manifest_digest": _digest(manifest_digest, "manifest_digest"),
            "worker_checkpoint_digests": dict(
                _digest_pairs(worker_checkpoint_digests, "worker_checkpoint_digests")
            ),
            "worker_checkpoint_refs": dict(
                _text_pairs(worker_checkpoint_refs, "worker_checkpoint_refs")
            ),
            "owner_state_digests": {
                owner: owner_digest_map[owner] for owner in SINGLE_CELL_OWNER_IDS
            },
            "owner_state_refs": {owner: owner_ref_map[owner] for owner in SINGLE_CELL_OWNER_IDS},
            "event_digests": list(_digest(item, "event digest") for item in event_digests),
            "event_refs": list(_text(item, "event ref") for item in event_refs),
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
    def from_payload(cls, payload: Mapping[str, Any]) -> TaijiSingleCellCheckpoint:
        if not isinstance(payload, Mapping):
            raise TypeError("single-cell checkpoint payload must be a mapping")
        unsigned = {key: value for key, value in payload.items() if key != "checkpoint_digest"}
        expected = content_digest(unsigned)
        if str(payload.get("checkpoint_digest", "")) != expected:
            raise ValueError("single-cell checkpoint digest mismatch")
        item = cls(
            format=str(payload.get("format", "")),
            version=int(payload.get("version", -1)),
            base_continuation_checkpoint_digest=str(payload["base_continuation_checkpoint_digest"]),
            base_continuation_ref=str(payload["base_continuation_ref"]),
            manifest_digest=str(payload["manifest_digest"]),
            worker_checkpoint_digests=tuple(payload["worker_checkpoint_digests"].items()),
            worker_checkpoint_refs=tuple(payload["worker_checkpoint_refs"].items()),
            owner_state_digests=tuple(payload["owner_state_digests"].items()),
            owner_state_refs=tuple(payload["owner_state_refs"].items()),
            event_digests=tuple(payload.get("event_digests", ())),
            event_refs=tuple(payload.get("event_refs", ())),
            cursor=SingleCellEventCursor.from_payload(payload["cursor"]),
            rng_state=payload["rng_state"],
            rng_state_digest=str(payload["rng_state_digest"]),
            budget_counts=tuple(payload["budget_counts"].items()),
            lineage_chain=tuple(payload["lineage_chain"]),
            checkpoint_digest=str(payload["checkpoint_digest"]),
        )
        if content_digest(item._payload_without_digest()) != item.checkpoint_digest:
            raise ValueError("single-cell normalized checkpoint digest mismatch")
        return item
