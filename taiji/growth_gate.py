"""Consensus gate: the only place where a local write may be refused.

The growth gate is a *check*, not a knob.  Before this module the nearest thing
to a gate was ``growth_gates={"consensus": 1.0}``: a constant multiplier on the
contribution of a newborn unit, never compared against anything, and short
circuited entirely while the subject has no grown units.  This module replaces
the constant with a submit-time protocol::

    snap    = snapshot(state)          # bit-exact bytes + sha256
    shadow  = shadow_apply(state, w)   # copy-on-write, never evicts
    cert    = evaluate(state, shadow, battery, w)
    receipt = commit(...)              # publish if passed, else restore + quarantine

Four conditions decide, all computable offline on numpy arrays (no torch, no
forward, no gradient):

===========  ==================================================  ==================
G1           retention battery correct count must not drop        integer, no tol
G2           read margin and effective rank must not drop        0.002 / 0
G3           state must not grow beyond one slot, headroom >= h 33280 B / 0.125
G4           the write must actually fix its own case            residual <= 0.5
===========  ==================================================  ==================

Every threshold below is a module-level constant and the single source of
truth; ``evaluate`` reads them and nothing else hard-codes a number.

Deliberate scope: this module does not import ``torch``, is not imported by the
frozen graph, and is not imported by the training runner.  ``shadow_apply``
mirrors the frozen ``addressed_write`` semantics (append to a free slot; a full
store is refused, never evicted) so that a rejection here describes the code
that actually runs today.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# --- thresholds: single source of truth -------------------------------------
GATE_G1_MAX_RETENTION_DROP = 0  # absolute count; must stay zero
GATE_G2_MAX_MARGIN_DROP = 0.002  # read margin tolerance
GATE_G2_MAX_RANK_DROP = 0  # effective rank tolerance
GATE_MAX_STATE_BYTES = 33_280  # one slot, accounted at the deployment dtype
GATE_MIN_SLOT_HEADROOM = 0.125  # capacity-first reserve, from the growth spec
GATE_MAX_VALUE_RESIDUAL = 0.5  # value fidelity of the write it claims to fix
GATE_FLOAT_BYTES = 4  # deployment dtype is torch float32; the fixture is float64
GATE_TEMPERATURE = 0.1  # ADDRESSED_TEMPERATURE, unchanged

# --- exit codes: one per failing condition ----------------------------------
EXIT_ACCEPTED = "commit_accepted"
EXIT_RETENTION = "commit_rejected_retention"
EXIT_SEPARABILITY = "commit_rejected_separability"
EXIT_INFLATION = "commit_rejected_inflation"
EXIT_NO_CAPACITY = "commit_rejected_no_capacity"
EXIT_NO_GAIN = "commit_rejected_no_gain"
EXIT_CODES = (
    EXIT_ACCEPTED,
    EXIT_RETENTION,
    EXIT_SEPARABILITY,
    EXIT_INFLATION,
    EXIT_NO_CAPACITY,
    EXIT_NO_GAIN,
)
EXIT_ZH = {
    EXIT_ACCEPTED: "准入",
    EXIT_RETENTION: "旧能力下降",
    EXIT_SEPARABILITY: "可分性下降",
    EXIT_INFLATION: "状态膨胀超限",
    EXIT_NO_CAPACITY: "可写余量不足",
    EXIT_NO_GAIN: "定向收益不成立（写入没有真正修好它自己那一问）",
}

# --- data -------------------------------------------------------------------


@dataclass(frozen=True)
class GateState:
    """A content-addressed store, offline.  Rows ``>= occupied`` are free."""

    keys: np.ndarray  # (slots, d)
    values: np.ndarray  # (slots, value_dim)
    age: np.ndarray  # (slots,) int64, -1 = free
    hits: np.ndarray  # (slots,) int64
    occupied: int
    metric_version: int

    @property
    def slots(self) -> int:
        return int(self.keys.shape[0])


@dataclass(frozen=True)
class WriteCell:
    key: np.ndarray
    value: np.ndarray
    family: str
    policy: str = "append"  # "append" mirrors the frozen addressed_write


@dataclass(frozen=True)
class BatteryQuery:
    name: str
    key: np.ndarray
    expect_slot: int
    family: str


@dataclass(frozen=True)
class StateSnapshot:
    blobs: dict[str, bytes]
    meta: dict[str, Any]
    sha256: str
    n_floats: int


@dataclass(frozen=True)
class GateCertificate:
    passed: bool
    exit_code: str
    failed: tuple[str, ...]
    g1_before: int
    g1_after: int
    g2_margin_before: float
    g2_margin_after: float
    g2_rank_before: float
    g2_rank_after: float
    g3_floats_before: int
    g3_floats_after: int
    g3_bytes_delta: int
    g3_headroom_after: float
    g4_top1_slot: int | None
    g4_value_residual: float | None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "exit_code": self.exit_code,
            "exit_zh": EXIT_ZH[self.exit_code],
            "failed": list(self.failed),
            "g1_retention_before": self.g1_before,
            "g1_retention_after": self.g1_after,
            "g2_margin_before": self.g2_margin_before,
            "g2_margin_after": self.g2_margin_after,
            "g2_rank_before": self.g2_rank_before,
            "g2_rank_after": self.g2_rank_after,
            "g3_floats_before": self.g3_floats_before,
            "g3_floats_after": self.g3_floats_after,
            "g3_bytes_delta": self.g3_bytes_delta,
            "g3_headroom_after": self.g3_headroom_after,
            "g4_top1_slot": self.g4_top1_slot,
            "g4_value_residual": self.g4_value_residual,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class GateRun:
    """Everything one submission needs, so ``commit`` cannot re-evaluate."""

    before_snapshot: StateSnapshot
    before: GateState
    shadow: GateState
    certificate: GateCertificate
    write: WriteCell
    apply_receipt: dict[str, Any]


@dataclass(frozen=True)
class CommitReceipt:
    published: bool
    exit_code: str
    state: GateState
    sha_before: str
    sha_after: str
    bit_identical: bool
    evictions: int
    quarantine_id: str | None


# --- helpers ----------------------------------------------------------------


def _active(state: GateState) -> tuple[np.ndarray, np.ndarray]:
    rows = int(state.occupied)
    return state.keys[:rows], state.values[:rows]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.clip(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-12, None)
    return matrix / norms


def _read(state: GateState, key: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Cosine softmax read over occupied rows, exactly as ``addressed_read``."""
    keys, values = _active(state)
    query = key / max(float(np.linalg.norm(key)), 1e-12)
    cosine = _normalize(keys) @ query
    logits = cosine / GATE_TEMPERATURE
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    top = int(np.argmax(cosine))
    return top, weights, (weights[:, None] * values).sum(axis=0)


def _margin(state: GateState, key: np.ndarray) -> float:
    keys, _ = _active(state)
    query = key / max(float(np.linalg.norm(key)), 1e-12)
    cosine = np.sort(_normalize(keys) @ query)[::-1]
    if cosine.size < 2:
        return float("nan")
    return float(cosine[0] - cosine[1])


def _effective_rank(state: GateState) -> float:
    keys, _ = _active(state)
    if keys.shape[0] < 2:
        return 0.0
    gram = _normalize(keys) @ _normalize(keys).T
    eigenvalues = np.clip(np.linalg.eigvalsh(gram), 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0
    return float(total * total / float((eigenvalues**2).sum()))


def _state_floats(state: GateState) -> int:
    """Occupied floats only: allocation is fixed, usage is what may grow."""
    rows = int(state.occupied)
    return int(rows * state.keys.shape[1] + rows * state.values[0].size)


def _residual(read_value: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(read_value - target) / max(float(np.linalg.norm(target)), 1e-12))


# --- the six functions -------------------------------------------------------


def snapshot(state: GateState) -> StateSnapshot:
    """Bit-exact bytes of the four arrays plus metadata, sealed with sha256."""
    blobs = {
        "keys": state.keys.tobytes(),
        "values": state.values.tobytes(),
        "age": state.age.tobytes(),
        "hits": state.hits.tobytes(),
    }
    meta = {
        "keys_shape": list(state.keys.shape),
        "values_shape": list(state.values.shape),
        "keys_dtype": str(state.keys.dtype),
        "values_dtype": str(state.values.dtype),
        "age_dtype": str(state.age.dtype),
        "hits_dtype": str(state.hits.dtype),
        "occupied": int(state.occupied),
        "metric_version": int(state.metric_version),
    }
    digest = hashlib.sha256()
    for name in ("keys", "values", "age", "hits"):
        digest.update(name.encode("ascii"))
        digest.update(blobs[name])
    digest.update(json.dumps(meta, sort_keys=True).encode("utf-8"))
    return StateSnapshot(
        blobs=blobs,
        meta=meta,
        sha256=digest.hexdigest(),
        n_floats=_state_floats(state),
    )


def shadow_apply(state: GateState, write: WriteCell) -> tuple[GateState, dict[str, Any]]:
    """Copy-on-write apply.  Never evicts; a full store is refused, not trimmed.

    ``policy="append"`` mirrors ``living_core.addressed_write`` as it exists
    today: a free slot is taken, or the store is full and nothing happens.
    ``policy="replace"`` is the supersede rule the F1 reading says is missing;
    it is available but is not what the running code does.
    """
    keys = state.keys.copy()
    values = state.values.copy()
    age = state.age.copy()
    hits = state.hits.copy()
    rows = int(state.occupied)
    receipt: dict[str, Any] = {
        "policy": write.policy,
        "applied": False,
        "slot": None,
        "evictions": 0,
        "dropped": 0,
        "reason": None,
    }

    if write.policy == "append":
        if rows >= state.slots:
            receipt["dropped"] = 1
            receipt["reason"] = "store_full_refused_not_evicted"
            return state, receipt
        slot = rows
        receipt.update(applied=True, slot=slot, reason="free_slot")
    elif write.policy == "replace":
        if rows == 0:
            receipt.update(dropped=1, reason="empty_store")
            return state, receipt
        keys_before, _ = _active(state)
        cosine = _normalize(keys_before) @ (
            write.key / max(float(np.linalg.norm(write.key)), 1e-12)
        )
        slot = int(np.argmax(cosine))
        receipt.update(applied=True, slot=slot, reason="same_key_superseded")
        receipt["replaced_slot_cosine"] = float(cosine[slot])
    else:
        raise ValueError(f"unknown write policy: {write.policy}")

    keys[slot] = write.key
    values[slot] = write.value
    age[slot] = int(age.max()) + 1 if rows else 0
    hits[slot] = 0
    occupied = max(rows, slot + 1)
    shadow = GateState(
        keys=keys,
        values=values,
        age=age,
        hits=hits,
        occupied=occupied,
        metric_version=state.metric_version,
    )
    return shadow, receipt


def evaluate(
    before: GateState,
    after: GateState,
    battery: list[BatteryQuery],
    write: WriteCell,
) -> GateCertificate:
    """Run G1-G4 on the shadow state.  Pure: no randomness, no time, no globals."""
    if not battery:
        raise ValueError("battery is empty: a gate without a retention set is not a gate")
    contaminated = [row.name for row in battery if row.family == write.family]
    if contaminated:
        raise ValueError(
            "retention battery contains the write's own family "
            f"({contaminated}): the queries a write is meant to change cannot also be "
            "the evidence that it changed nothing"
        )

    correct_before = sum(1 for row in battery if _read(before, row.key)[0] == row.expect_slot)
    correct_after = sum(1 for row in battery if _read(after, row.key)[0] == row.expect_slot)

    margin_before = float(np.median([_margin(before, row.key) for row in battery]))
    margin_after = float(np.median([_margin(after, row.key) for row in battery]))
    rank_before = _effective_rank(before)
    rank_after = _effective_rank(after)

    floats_before = _state_floats(before)
    floats_after = _state_floats(after)
    bytes_delta = (floats_after - floats_before) * GATE_FLOAT_BYTES
    headroom_after = (int(after.slots) - int(after.occupied)) / max(int(after.slots), 1)

    top1, _, read_value = _read(after, write.key)
    written_slot = top1
    residual = _residual(read_value, write.value)

    failed: list[str] = []
    if correct_before - correct_after > GATE_G1_MAX_RETENTION_DROP:
        failed.append(EXIT_RETENTION)
    if margin_after < margin_before - GATE_G2_MAX_MARGIN_DROP:
        failed.append(EXIT_SEPARABILITY)
    if rank_after < rank_before - GATE_G2_MAX_RANK_DROP:
        failed.append(EXIT_SEPARABILITY)
    if bytes_delta > GATE_MAX_STATE_BYTES:
        failed.append(EXIT_INFLATION)
    if headroom_after < GATE_MIN_SLOT_HEADROOM:
        failed.append(EXIT_NO_CAPACITY)
    if top1 != written_slot or residual > GATE_MAX_VALUE_RESIDUAL:
        failed.append(EXIT_NO_GAIN)

    # de-duplicate while keeping the G1 -> G4 order
    ordered: list[str] = []
    for code in failed:
        if code not in ordered:
            ordered.append(code)
    passed = not ordered
    exit_code = EXIT_ACCEPTED if passed else ordered[0]

    return GateCertificate(
        passed=passed,
        exit_code=exit_code,
        failed=tuple(ordered),
        g1_before=correct_before,
        g1_after=correct_after,
        g2_margin_before=margin_before,
        g2_margin_after=margin_after,
        g2_rank_before=rank_before,
        g2_rank_after=rank_after,
        g3_floats_before=floats_before,
        g3_floats_after=floats_after,
        g3_bytes_delta=bytes_delta,
        g3_headroom_after=headroom_after,
        g4_top1_slot=top1,
        g4_value_residual=residual,
        detail={
            "write_slot": written_slot,
            "write_family": write.family,
            "battery_size": len(battery),
            "battery_families": sorted({row.family for row in battery}),
            "thresholds": {
                "g1_max_retention_drop": GATE_G1_MAX_RETENTION_DROP,
                "g2_max_margin_drop": GATE_G2_MAX_MARGIN_DROP,
                "g2_max_rank_drop": GATE_G2_MAX_RANK_DROP,
                "g3_max_state_bytes": GATE_MAX_STATE_BYTES,
                "g3_min_slot_headroom": GATE_MIN_SLOT_HEADROOM,
                "g4_max_value_residual": GATE_MAX_VALUE_RESIDUAL,
                "float_bytes": GATE_FLOAT_BYTES,
                "temperature": GATE_TEMPERATURE,
            },
        },
    )


def commit(
    run: GateRun,
    *,
    quarantine_path: Path | None = None,
    log_path: Path | None = None,
) -> CommitReceipt:
    """Publish the shadow state only if every condition passed.

    On rejection the pre-write snapshot is restored **bit for bit**: arrays are
    rebuilt from the recorded bytes and the sha256 is recomputed, so "rolled
    back" is verified rather than asserted.  Nothing is evicted and the refused
    write is not discarded -- it goes to the quarantine journal.
    """
    if run.certificate.passed:
        after = run.shadow
        bit_identical = False
        quarantine_id = None
        state = after
    else:
        state = restore(run.before_snapshot)
        bit_identical = snapshot(state).sha256 == run.before_snapshot.sha256
        quarantine_id = None
        if quarantine_path is not None:
            row = quarantine(
                {
                    "family": run.write.family,
                    "policy": run.write.policy,
                    "key_sha256": hashlib.sha256(
                        np.asarray(run.write.key, dtype=np.float64).tobytes()
                    ).hexdigest(),
                    "value_sha256": hashlib.sha256(
                        np.asarray(run.write.value, dtype=np.float64).tobytes()
                    ).hexdigest(),
                    "shadow_slot": run.apply_receipt.get("slot"),
                    "certificate": run.certificate.as_dict(),
                },
                quarantine_path,
            )
            quarantine_id = row["id"]

    sha_after = snapshot(state).sha256
    receipt = CommitReceipt(
        published=bool(run.certificate.passed),
        exit_code=run.certificate.exit_code,
        state=state,
        sha_before=run.before_snapshot.sha256,
        sha_after=sha_after,
        bit_identical=bit_identical or sha_after == run.before_snapshot.sha256,
        evictions=int(run.apply_receipt.get("evictions", 0)),
        quarantine_id=quarantine_id,
    )
    if log_path is not None:
        append_jsonl(
            {
                "event": "gate_decision",
                "published": receipt.published,
                "exit_code": receipt.exit_code,
                "sha_before": receipt.sha_before,
                "sha_after": receipt.sha_after,
                "bit_identical": receipt.bit_identical,
                "evictions": receipt.evictions,
                "quarantine_id": receipt.quarantine_id,
                "certificate": run.certificate.as_dict(),
            },
            log_path,
        )
    return receipt


def restore(snap: StateSnapshot) -> GateState:
    """Rebuild the state from bytes and refuse anything that is not bit-exact."""
    meta = snap.meta
    keys = np.frombuffer(snap.blobs["keys"], dtype=np.dtype(meta["keys_dtype"]))
    keys = keys.reshape(tuple(meta["keys_shape"])).copy()
    values = np.frombuffer(snap.blobs["values"], dtype=np.dtype(meta["values_dtype"]))
    values = values.reshape(tuple(meta["values_shape"])).copy()
    age = np.frombuffer(snap.blobs["age"], dtype=np.dtype(meta["age_dtype"]))
    age = age.reshape(keys.shape[0]).copy()
    hits = np.frombuffer(snap.blobs["hits"], dtype=np.dtype(meta["hits_dtype"]))
    hits = hits.reshape(age.shape).copy()
    state = GateState(
        keys=keys,
        values=values,
        age=age,
        hits=hits,
        occupied=int(meta["occupied"]),
        metric_version=int(meta["metric_version"]),
    )
    rebuilt = snapshot(state).sha256
    if rebuilt != snap.sha256:
        raise RuntimeError(f"rollback_corrupt: {rebuilt} != {snap.sha256}")
    return state


def quarantine(record: dict[str, Any], path: Path) -> dict[str, Any]:
    """Append a refused write to the journal.  Nothing is deleted on rejection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = 0
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = sum(1 for line in handle if line.strip())
    row = dict(record)
    row["id"] = f"q-{existing + 1:04d}"
    row["seq"] = existing + 1
    append_jsonl(row, path)
    return row


def append_jsonl(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def run_gate(
    state: GateState,
    write: WriteCell,
    battery: list[BatteryQuery],
    *,
    quarantine_path: Path | None = None,
    log_path: Path | None = None,
) -> tuple[CommitReceipt, GateRun]:
    """snapshot -> shadow -> evaluate -> commit, in that order, on one write."""
    before_snapshot = snapshot(state)
    shadow, apply_receipt = shadow_apply(state, write)
    certificate = evaluate(state, shadow, battery, write)
    run = GateRun(
        before_snapshot=before_snapshot,
        before=state,
        shadow=shadow,
        certificate=certificate,
        write=write,
        apply_receipt=apply_receipt,
    )
    return commit(run, quarantine_path=quarantine_path, log_path=log_path), run
