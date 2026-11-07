"""Read-key encoder and the capacity-first write protocol (numpy only).

Implements plans/reference/SPEC-TAIJI-GROWTH_DESIGN_20261108.md §3A and §5A:

* ``EncoderConfig`` / ``encode_read_key`` / ``store_centroid`` -- a key encoder
  that keeps per-sentence differences instead of averaging the whole context
  into one point (``mode="localized_centered"``), plus the legacy ``mean``
  path which is bit-identical to today's ``context.mean(0)``.
* ``write_headroom`` / ``plan_growth`` / ``write_batch`` -- capacity first:
  check the writable headroom *before* the write, grow first when it is short,
  and roll the whole batch back on failure so no partial write survives.

This module is deliberately numpy-only, has no RNG in its decision path and is
NOT imported by any training runner (see plans/reference/FROZEN-READSET_20261108.md).
Nothing here trains, loads weights or touches a checkpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_RESERVE_RATIO = 0.125
DEFAULT_HEADROOM_MIN = 0.0
ENCODER_SCHEMA_MEAN = "mean-v1"
ENCODER_SCHEMA_LOCALIZED = "localized_centered-v1"


@dataclass(frozen=True)
class EncoderConfig:
    """Encoder contract.  ``mode="mean"`` is the frozen 20.3a behaviour."""

    mode: str = "mean"
    tau: float = 0.5
    top_k: int = 4
    center: bool = False

    @property
    def schema(self) -> str:
        return ENCODER_SCHEMA_MEAN if self.mode == "mean" else ENCODER_SCHEMA_LOCALIZED


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


def store_centroid(keys: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Mean of the occupied read keys; zeros when the store is empty."""
    keys = np.asarray(keys, dtype=np.float64)
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        keys = keys[mask]
    if keys.size == 0 or keys.shape[0] == 0:
        return np.zeros(keys.shape[1] if keys.ndim == 2 else 0, dtype=np.float64)
    return keys.mean(axis=0)


def encode_read_key(
    context: np.ndarray,
    cfg: EncoderConfig,
    centroid: np.ndarray | None = None,
) -> np.ndarray:
    """Encode a ``(T, d)`` context into a ``(d,)`` read key.

    ``mean``: ``context.mean(0)`` -- the current 20.3a key, unchanged.
    ``localized_centered``: per-dimension variance-weighted pooling plus a
    top-k position residual, then optional centroid subtraction, then L2.
    """
    if cfg.mode == "mean":
        # Legacy path: literally the same operator, in the same framework, dtype
        # and reduction order as the frozen construction.  Do not coerce to numpy
        # first: numpy float32 mean and torch float32 mean differ in the last bits
        # for T >= 18 (max 1.2e-7), which would perturb the key bytes.
        if context.ndim == 1:
            context = context[None, :]
        key = context.mean(axis=0)
        if cfg.center and centroid is not None:
            if hasattr(key, "new_tensor"):
                key = key - key.new_tensor(centroid)
            else:
                key = key - np.asarray(centroid, dtype=key.dtype)
        return key
    if cfg.mode == "localized_centered":
        context = np.asarray(context, dtype=np.float64)
        if context.ndim == 1:
            context = context[None, :]
        mu = context.mean(axis=0)
        deviation = np.abs(context - mu)
        logits = deviation / max(cfg.tau, 1e-6)
        weights = np.exp(logits - logits.max(axis=0, keepdims=True))
        weights /= weights.sum(axis=0, keepdims=True)
        pooled = (weights * context).sum(axis=0)
        residual = context - mu
        if cfg.top_k > 0 and residual.shape[0] > 0:
            norms = np.linalg.norm(residual, axis=1)
            keep = np.argsort(-norms)[: min(cfg.top_k, residual.shape[0])]
            local = residual[keep].sum(axis=0) / math.sqrt(len(keep))
        else:
            local = np.zeros_like(mu)
        key = pooled + local
    else:
        raise ValueError(f"unknown encoder mode {cfg.mode!r}")
    if cfg.center and centroid is not None:
        key = key - centroid
    return _normalize(key)


def write_headroom(
    occupied: int,
    slots: int,
    pending: int,
    *,
    reserve_ratio: float = DEFAULT_RESERVE_RATIO,
) -> dict[str, Any]:
    """Writable headroom after the pending batch, as a fraction of capacity."""
    free = max(int(slots) - int(occupied), 0)
    remaining = free - int(pending)
    fraction = remaining / max(int(slots), 1)
    return {
        "occupied": int(occupied),
        "slots": int(slots),
        "pending": int(pending),
        "free": free,
        "remaining": remaining,
        "h_slots": fraction,
        "h_min_slots": float(reserve_ratio),
        "sufficient": bool(fraction >= reserve_ratio),
        "spills": bool(remaining < 0),
    }


def plan_growth(
    headroom: dict[str, Any],
    *,
    slot_cap: int | None = None,
    growth_factor: int = 2,
) -> dict[str, Any]:
    """Smallest doubling of slots that restores the reserve; ``None`` if capped."""
    if headroom["sufficient"]:
        return {"action": "write_now", "slots": headroom["slots"], "reason": "headroom_ok"}
    slots = int(headroom["slots"])
    needed = int(headroom["occupied"]) + int(headroom["pending"])
    target = slots
    while True:
        target *= growth_factor
        if (target - needed) / target >= headroom["h_min_slots"]:
            break
        if slot_cap is not None and target >= slot_cap:
            target = slot_cap
            break
    if slot_cap is not None and target > slot_cap:
        target = slot_cap
    ok = (target - needed) / max(target, 1) >= headroom["h_min_slots"]
    if not ok:
        return {
            "action": "reject_write",
            "slots": target,
            "reason": "growth_capped",
            "needed": needed,
            "required_reserve": headroom["h_min_slots"],
        }
    return {
        "action": "grow_then_write",
        "slots": target,
        "added": target - slots,
        "reason": "headroom_short",
        "needed": needed,
    }


def write_batch(
    store: dict[int, bytes],
    batch: dict[int, bytes],
    *,
    headroom: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """All-or-nothing write.  On any failure the store is left untouched."""
    if plan["action"] == "reject_write":
        return {"committed": [], "store": store, "reason": plan["reason"], "rolled_back": False}
    if plan["action"] == "grow_then_write" and plan["slots"] < headroom["occupied"] + len(batch):
        return {
            "committed": [],
            "store": store,
            "reason": "grown_capacity_still_short",
            "rolled_back": True,
        }
    snapshot = dict(store)
    try:
        candidate = dict(store)
        for slot, key in batch.items():
            candidate[int(slot)] = key
        if len(candidate) > int(plan["slots"]):
            raise ValueError("batch exceeds grown capacity")
    except Exception as error:  # noqa: BLE001 - rollback path must swallow everything
        return {
            "committed": [],
            "store": snapshot,
            "reason": f"rolled_back:{type(error).__name__}",
            "rolled_back": True,
        }
    return {
        "committed": sorted(int(slot) for slot in batch),
        "store": candidate,
        "reason": "committed",
        "rolled_back": False,
    }
