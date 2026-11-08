"""Capacity growth wired onto the real store; default off, zero parameters.

Three entry points sit on the real graph and nothing else:

* ``session_boundary`` -- called once at a session boundary.  Disabled, it
  returns the *same* ``LivingState`` object and writes nothing.  Enabled, it
  checks the writable headroom for the session's planned writes and, when the
  reserve is short, pads the slot tensors with zeros / ``-1`` ages.  Existing
  slots are never rewritten, so every previously written byte survives.
* ``session_write`` -- one store write.  Disabled, it calls the frozen
  ``addressed_write`` verbatim.  Enabled, it refuses instead of evicting when
  the store is full, and refuses a write whose direction is a copy of the
  occupied span (the ``growth_gate`` direction rule: a copy adds no capacity).
* ``session_read`` -- the read path.  Growth does not change it; the entry is
  here only so a caller cannot half-install the wire.

No parameter, no optimiser and no training lives in this module; a grown store
is state, not weights.  The frozen files listed in
``plans/reference/FROZEN-READSET_20261108.md`` are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from taiji.growth_gate import (
    GATE_DIRECTION_MAX_OVERLAP,
    GATE_DIRECTION_MIN_OVERLAP,
    GATE_DIRECTION_MIN_SEPARATION,
)
from taiji.growth_runtime import DEFAULT_RESERVE_RATIO, plan_growth, write_headroom
from taiji.living_core import LivingState

SESSION_SCHEMA = "taiji-growth-wire-v1"


@dataclass(frozen=True)
class GrowthWireConfig:
    """``enabled=False`` is the production default and the frozen behaviour."""

    enabled: bool = False
    reserve_ratio: float = DEFAULT_RESERVE_RATIO
    growth_factor: int = 2
    slot_cap: int | None = None


def store_capacity(state: LivingState) -> int:
    if state.sc_sfk is not None:
        return int(state.sc_sfk.capacity)
    return 0 if state.store_keys is None else int(state.store_keys.shape[0])


def store_occupied(state: LivingState) -> int:
    if state.sc_sfk is not None:
        return len(state.sc_sfk.addresses)
    if state.store_age is None:
        return 0
    return int((state.store_age >= 0).sum())


def store_bytes(state: LivingState) -> int:
    if state.sc_sfk is not None:
        return int(state.sc_sfk.nbytes())
    total = 0
    for tensor in (state.store_keys, state.store_values, state.store_age):
        if tensor is not None:
            total += int(tensor.numel()) * int(tensor.element_size())
    return total


def state_bytes(state: LivingState) -> int:
    total = store_bytes(state)
    for name in ("h", "fast", "nu_ema", "coverage", "prior"):
        tensor = getattr(state, name)
        total += int(tensor.numel()) * int(tensor.element_size())
    return total


def copy_band(keys: torch.Tensor, key: torch.Tensor) -> dict[str, Any]:
    """The ``growth_gate`` direction band, computed on the store's own keys."""
    unit = key.double() / max(float(key.double().norm()), 1e-12)
    if keys.numel() == 0:
        return {"overlap": 0.0, "separation": 1.0, "verdict": "growth_new_direction"}
    normalized = keys.double() / torch.clamp(keys.double().norm(dim=-1, keepdim=True), min=1e-12)
    basis = torch.linalg.svd(normalized, full_matrices=False)[2]
    rank = int(torch.linalg.matrix_rank(normalized))
    basis = basis[: max(rank, 1)]
    overlap = min(float(torch.linalg.vector_norm(basis @ unit)), 1.0)
    separation = 1.0 - float((normalized @ unit).max())
    if overlap > GATE_DIRECTION_MAX_OVERLAP or separation < GATE_DIRECTION_MIN_SEPARATION:
        verdict = "growth_is_copy"
    elif overlap < GATE_DIRECTION_MIN_OVERLAP:
        verdict = "growth_isolated"
    else:
        verdict = "growth_new_direction"
    return {
        "overlap": round(overlap, 9),
        "separation": round(separation, 9),
        "verdict": verdict,
    }


def grow_store(state: LivingState, slots: int) -> tuple[LivingState, dict[str, Any]]:
    """Pad the addressed store to ``slots``.  The occupied prefix is untouched."""
    current = store_capacity(state)
    if state.store_keys is None or int(slots) <= current:
        return state, {"action": "unchanged", "slots": current, "added": 0, "bytes_added": 0}
    extra = int(slots) - current
    keys = torch.cat(
        [state.store_keys, state.store_keys.new_zeros((extra, state.store_keys.shape[1]))]
    )
    values = torch.cat(
        [
            state.store_values,
            state.store_values.new_zeros((extra, *state.store_values.shape[1:])),
        ]
    )
    age = torch.cat([state.store_age, state.store_age.new_full((extra,), -1)])
    grown = replace(state, store_keys=keys, store_values=values, store_age=age)
    return grown, {
        "action": "grown",
        "slots": int(slots),
        "added": extra,
        "bytes_added": int(store_bytes(grown) - store_bytes(state)),
    }


def session_boundary(
    state: LivingState,
    *,
    pending: int,
    config: GrowthWireConfig,
    session: str,
    generation: int = 0,
) -> tuple[LivingState, dict[str, Any]]:
    """Session-boundary hook.  Disabled: identity passthrough, no allocation."""
    record: dict[str, Any] = {
        "schema": SESSION_SCHEMA,
        "session": session,
        "enabled": bool(config.enabled),
        "generation": int(generation),
        "slots": store_capacity(state),
        "occupied": store_occupied(state),
        "pending": int(pending),
    }
    if state.sc_sfk is not None:
        if config.enabled:
            raise ValueError("two capacity authorities: state.sc_sfk and GrowthWireConfig.enabled")
        record.update(
            {
                "action": "table_owns_capacity",
                "reason": "sc_sfk_capacity",
                "generation": int(state.sc_sfk.generation),
                "state_same_object": True,
                "bytes_added": 0,
            }
        )
        return state, record
    if not config.enabled:
        record.update({"action": "legacy_path", "state_same_object": True, "bytes_added": 0})
        return state, record
    headroom = write_headroom(
        store_occupied(state),
        store_capacity(state),
        pending,
        reserve_ratio=config.reserve_ratio,
    )
    plan = plan_growth(headroom, slot_cap=config.slot_cap, growth_factor=config.growth_factor)
    before = store_bytes(state)
    if plan["action"] == "write_now":
        grown = state
        action = "write_now"
        reason = str(plan["reason"])
        generation_out = int(generation)
    elif plan["action"] == "reject_write":
        grown = state
        action = "reject_write"
        reason = str(plan["reason"])
        generation_out = int(generation)
    else:
        grown, info = grow_store(state, int(plan["slots"]))
        action = "grow_then_write"
        reason = str(plan["reason"])
        generation_out = int(generation) + 1
        record["added"] = int(info["added"])
    record.update(
        {
            "action": action,
            "reason": reason,
            "h_slots": round(float(headroom["h_slots"]), 9),
            "h_min_slots": float(headroom["h_min_slots"]),
            "slots": store_capacity(grown),
            "bytes_added": int(store_bytes(grown) - before),
            "generation": generation_out,
            "state_same_object": grown is state,
        }
    )
    return grown, record


def session_write(
    subject,
    state: LivingState,
    raw: bytes,
    *,
    config: GrowthWireConfig,
    generation: int = 0,
    key: torch.Tensor | None = None,
) -> tuple[LivingState, dict[str, Any]]:
    """One store write.  Disabled: the frozen ``addressed_write`` verbatim."""
    if getattr(subject.config, "sc_sfk_enabled", False) or state.sc_sfk is not None:
        if config.enabled:
            raise ValueError("two store authorities: sc_sfk_enabled and GrowthWireConfig.enabled")
        out, info = subject.addressed_write(state, raw)
        table = out.sc_sfk
        info.update(
            {
                "dropped": False,
                "capacity": table.capacity,
                "occupied": len(table.addresses),
                "generation": table.generation,
            }
        )
        return out, info
    if not config.enabled:
        return subject.addressed_write(state, raw)
    if state.store_keys is None:
        state = subject.attach_store(state)
    capacity = store_capacity(state)
    occupied = store_occupied(state)
    key = subject.store_query_key(state, raw) if key is None else key
    if occupied >= capacity:
        return state, {
            "action": "dropped",
            "dropped": True,
            "evicted": False,
            "slot": None,
            "capacity": capacity,
            "occupied": occupied,
            "generation": int(generation),
        }
    valid = torch.nonzero(state.store_age >= 0, as_tuple=False).flatten()
    band = copy_band(state.store_keys[valid], key)
    if band["verdict"] == "growth_is_copy" and valid.numel():
        return state, {
            "action": "rejected_copy",
            "dropped": False,
            "evicted": False,
            "slot": None,
            "band": band,
            "capacity": capacity,
            "occupied": occupied,
            "generation": int(generation),
        }
    out, info = subject.addressed_write(state, raw, key=key)
    info.update({"action": "appended", "dropped": False, "band": band, "capacity": capacity})
    info["generation"] = int(generation)
    return out, info


def session_read(
    subject,
    state: LivingState,
    raw: bytes,
    *,
    config: GrowthWireConfig,
    key: torch.Tensor | None = None,
):
    """The read path is growth-free; this entry exists so the wire is whole."""
    if (getattr(subject.config, "sc_sfk_enabled", False) or state.sc_sfk is not None) and (
        config.enabled
    ):
        raise ValueError("two store authorities: sc_sfk_enabled and GrowthWireConfig.enabled")
    return subject.addressed_read(state, raw, key=key)


def session_step(subject, state: LivingState, *, config: GrowthWireConfig, **kwargs):
    """The forward is growth-free; capacity changes only at a session boundary."""
    return subject.step(state, **kwargs)
