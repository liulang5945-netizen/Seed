"""Full learning-state capsule for the P3.8.2b Native credit transfer.

SPEC-TAIJI_U2_NATIVE_CREDIT_TRANSFER_20261113 section 2: the typed runtime
envelope alone cannot represent the whole learning state.  The capsule owns
five parts and is the only persistence unit allowed to resume learning.
A body-only export covers inference recovery, not learning continuation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .u2_history import atomic_load, atomic_save, value_digest

SCHEMA = "taiji-u2-native-learner-capsule-v1"
PARTS = ("native_runtime", "learning_state", "sampling", "proposal", "identity")
BODY_TENSORS = 214
HEAD_TENSORS = 2
REGISTRY_TENSORS = BODY_TENSORS + HEAD_TENSORS
HEAD_ONLY_KEYS = (
    "content_head",
    "content_head_moment",
    "label_source",
    "class_weights",
    "label_index",
    "training_labels",
)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _validate_parts(parts: dict[str, Any]) -> None:
    _require(isinstance(parts, dict) and set(parts) == set(PARTS), "capsule_parts_mismatch")

    runtime = parts["native_runtime"]
    _require(isinstance(runtime, dict) and bool(runtime), "capsule_runtime_type")

    identity = parts["identity"]
    _require(isinstance(identity, dict), "capsule_identity_type")
    _require(bool(identity.get("factory_schema")), "capsule_factory_schema_missing")
    _require(bool(identity.get("source_pins")), "capsule_source_pins_missing")
    registry = identity.get("parameter_registry")
    _require(isinstance(registry, dict), "capsule_registry_missing")
    ids = registry.get("parameter_ids")
    _require(isinstance(ids, (list, tuple)), "capsule_registry_ids_type")
    _require(len(ids) == REGISTRY_TENSORS, "capsule_registry_count")
    _require(len(set(ids)) == REGISTRY_TENSORS, "capsule_registry_duplicate")
    owners = registry.get("owners")
    _require(
        isinstance(owners, dict) and set(owners) == {"body", "content_head"},
        "capsule_registry_owners",
    )
    _require(
        owners["body"] == BODY_TENSORS and owners["content_head"] == HEAD_TENSORS,
        "capsule_registry_owner_count",
    )

    state = parts["learning_state"]
    _require(isinstance(state, dict), "capsule_learning_state_type")
    u, v = state.get("proposal_clock"), state.get("parameter_version")
    _require(type(u) is int and type(v) is int and 0 <= v <= u, "capsule_clock_order")
    for key in (
        "body_moment",
        "content_head",
        "content_head_moment",
        "label_source",
        "class_weights",
        "parameter_ancestor",
    ):
        _require(key in state, f"capsule_learning_state_missing:{key}")

    sampling = parts["sampling"]
    _require(isinstance(sampling, dict), "capsule_sampling_type")
    for key in (
        "episode_cursor",
        "first_answer_prefix",
        "prefix_source",
        "data_rights",
        "sampling_rng",
    ):
        _require(key in sampling, f"capsule_sampling_missing:{key}")

    proposal = parts["proposal"]
    _require(proposal is None or isinstance(proposal, dict), "capsule_proposal_type")
    if proposal:
        for key in (
            "proposal_id",
            "alpha",
            "ancestor_digest",
            "objective_value",
            "search_state",
        ):
            _require(key in proposal, f"capsule_proposal_missing:{key}")


def build_capsule(
    *,
    native_runtime: dict[str, Any],
    learning_state: dict[str, Any],
    sampling: dict[str, Any],
    proposal: dict[str, Any] | None,
    identity: dict[str, Any],
) -> dict[str, Any]:
    parts = {
        "native_runtime": native_runtime,
        "learning_state": learning_state,
        "sampling": sampling,
        "proposal": proposal,
        "identity": identity,
    }
    _validate_parts(parts)
    return {**parts, "schema": SCHEMA, "capsule_digest": value_digest(parts)}


def validate_capsule(capsule: Any) -> dict[str, Any]:
    _require(isinstance(capsule, dict), "capsule_type")
    _require(capsule.get("schema") == SCHEMA, "capsule_schema_mismatch")
    parts = {key: capsule.get(key) for key in PARTS}
    _validate_parts(parts)
    _require(value_digest(parts) == capsule.get("capsule_digest"), "capsule_digest_mismatch")
    return parts


def save_capsule(destination: str | Path, capsule: dict[str, Any]) -> Path:
    validate_capsule(capsule)
    return atomic_save(capsule, destination)


def load_capsule(source: str | Path) -> dict[str, Any]:
    capsule = atomic_load(source)
    validate_capsule(capsule)
    return capsule


def headless_runtime(capsule: dict[str, Any]) -> dict[str, Any]:
    """Inference-only view: training labels and the content head never leak here."""
    parts = validate_capsule(capsule)
    runtime = parts["native_runtime"]
    leaked = [key for key in HEAD_ONLY_KEYS if key in runtime]
    _require(not leaked, "capsule_headless_leak:" + ",".join(leaked))
    return dict(runtime)


def capsule_summary(capsule: dict[str, Any]) -> dict[str, Any]:
    parts = validate_capsule(capsule)
    state = parts["learning_state"]
    return {
        "schema": capsule["schema"],
        "capsule_digest": capsule["capsule_digest"],
        "proposal_clock": state["proposal_clock"],
        "parameter_version": state["parameter_version"],
        "proposal_present": parts["proposal"] is not None,
        "registry_tensors": REGISTRY_TENSORS,
        "episode_cursor": parts["sampling"]["episode_cursor"],
    }
