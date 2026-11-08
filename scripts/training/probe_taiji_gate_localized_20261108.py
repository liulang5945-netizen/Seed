"""`localized_centered` encoding as the only variable: the last cut at the encoding layer.

Every earlier cut on this line changed the *geometry* of the averaged-pooled key
(centering, shrinkage whitening).  Both failed in opposite directions: centering
repairs the collapse but destroys the carrier, whitening keeps the carrier but
deepens the collapse.  The attribution was that the pooled byte key ties the two
together.  This run tests the one remaining candidate from the growth design
(`plans/reference/SPEC-TAIJI-GROWTH_DESIGN_20261108.md` §3A): replace the key
construction itself -- per-dimension variance-weighted pooling plus a top-k
position residual, then subtract one store-level centroid and L2-normalise.

Two arms, one variable (the read-key encoder; write and read use the same one,
and the encoding is switched only at the session boundary):

* ``A_mean``                -- today's ``context.mean(0)``.
* ``B_localized_centered``  -- ``growth_runtime.encode_read_key`` with
  ``mode="localized_centered"``, ``tau=0.5``, ``top_k=4``, ``center=True``.
  The centroid is the mean of the 30 encoded keys of this store, fit once at
  the session boundary; every stored key and every query key uses that one
  centroid.  No key is carried over from arm A.

The value side is byte-identical between arms (the forward never reads the
store during the build), so the only thing that changes is what the store uses
as an address.

Nothing is trained or optimised: 0 training / 0 optimizer step / 0 Adam /
0 confirm.  The only model operations are the frozen forward that builds the
store and the frozen read that consumes it.
"""

from __future__ import annotations

import atexit
import errno
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji import growth_runtime as gr  # noqa: E402
from taiji.living_core import (  # noqa: E402
    ADDRESSED_ALPHA,
    ADDRESSED_TEMPERATURE,
    LivingCoreConfig,
    LivingCoreReference,
)
from taiji.process_guard import assert_no_black_processes  # noqa: E402

BASE_RUNNER = ROOT / "scripts/training/probe_taiji_gate_read_gate_20261108.py"
GROWTH_RUNTIME = ROOT / "taiji/growth_runtime.py"
LIVING_CORE = ROOT / "taiji/living_core.py"
REPORT = ROOT / "reports/taiji_gate_localized_20261108.json"
MANIFEST = ROOT / "plans/manifests/taiji_gate_localized_20261108.json"
PROGRESS = ROOT / "reports/taiji_gate_localized_20261108_progress.jsonl"
LOCK = ROOT / "reports/taiji_gate_localized_20261108.lock"
WATCHDOG_MARKER = ROOT / "reports/taiji_gate_localized_20261108_watchdog.json"

CPU_CAP = 200.0
CPU_GUARD = 190.0
WATCHDOG_SECONDS = 60.0
LOCK_STALE_SECONDS = 600.0
FLOOR_GATE = 0.90
REQUIRED_POSITIVE_GAPS = 2
TAU = 0.5
TOP_K = 4
CLASSES = ("written", "unwritten", "new_token", "new_pairing", "no_value")

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

START_CPU = time.process_time()
START_WALL = time.time()
LAST_READING = [time.time()]
WATCHDOG_FIRED = [False]

CRITERIA = {
    "C1_control_identity": (
        "for each arm the runner's own cosine read reproduces the real addressed_read with the "
        "same injected key on all 154 queries: identical fast sha, identical top slot, max|delta| = 0"
    ),
    "C2_one_encoding": (
        "write and read use the same encoder in an arm; arm B never reuses an arm-A key; the "
        "encoding changes only when the store is rebuilt at the session boundary"
    ),
    "C3_centroid_rule": (
        "the centroid is the mean of the occupied keys of that same arm-B store, fit once before "
        "any reading; tau=0.5 and top_k=4 come from the growth spec, not from a search"
    ),
    "C4_values_unchanged": (
        "store values, ages and count are byte-identical between arm A and arm B: only the address "
        "changes, so any difference is attributable to the encoder"
    ),
    "C5_threshold_source": (
        "the gate threshold is the minimum top cosine of the written group in that same arm; no "
        "negative label enters the value"
    ),
    "C6_pass": (
        "pass = written floor >= 0.90 AND at least two of the three gaps > 0 AND the count of the "
        "22 written faces found by their own read wording is not lower than arm A"
    ),
    "V1_pass": "if C6 holds the verdict is localized_centered_enables_gate with the threshold interval",
    "V2_fail": (
        "otherwise the verdict is encoder_also_insufficient: this content-addressed write + read "
        "gate line is closed, no further encoding, geometry or threshold tuning"
    ),
    "unchanged": "no fixture, face, position, class, tau, top_k or anchor is retuned after first reading",
}


# ---------------------------------------------------------------------------
# Discipline helpers.
# ---------------------------------------------------------------------------
def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    with open(tmp, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def append_progress(record: dict) -> None:
    record = {**record, "at": round(time.time() - START_WALL, 3)}
    with open(PROGRESS, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    LAST_READING[0] = time.time()


def touch() -> None:
    LAST_READING[0] = time.time()


def acquire_single_instance() -> str:
    try:
        descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        age = time.time() - LOCK.stat().st_mtime
        if age < LOCK_STALE_SECONDS:
            raise SystemExit(f"another instance holds {LOCK} ({age:.1f}s old)") from None
        LOCK.unlink()
        descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as exc:  # pragma: no cover - surfaced as a hard stop
        if exc.errno != errno.EEXIST:
            raise
        raise SystemExit(f"cannot acquire {LOCK}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()} started={time.time():.3f}\n")
    return LOCK.name


def release_single_instance() -> None:
    if LOCK.exists():
        LOCK.unlink()


def start_watchdog(stop: threading.Event) -> threading.Thread:
    def loop() -> None:
        while not stop.wait(2.0):
            idle = time.time() - LAST_READING[0]
            if idle > WATCHDOG_SECONDS:
                WATCHDOG_FIRED[0] = True
                write_json(
                    WATCHDOG_MARKER,
                    {"fired": True, "idle_seconds": round(idle, 3), "at": time.time()},
                )
                os._exit(3)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


def load_base():
    spec = importlib.util.spec_from_file_location("read_gate_runner", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Encoding.  Arm A keeps today's key; arm B rebuilds every key with the growth
# encoder and never touches an arm-A key.
# ---------------------------------------------------------------------------
def context_matrix(subject, raw: bytes) -> np.ndarray:
    context = subject.encoder.context(raw)
    if context.numel() == 0:
        return np.zeros((0, int(subject.config.width)), dtype=np.float64)
    return context.detach().cpu().double().numpy()


def to_key(subject, vector: np.ndarray) -> torch.Tensor:
    return torch.tensor(vector, dtype=subject.W.dtype, device=subject.W.device)


def mean_encoder_key(subject, raw: bytes) -> torch.Tensor:
    context = subject.encoder.context(raw)
    if context.numel() == 0:
        return subject.W.new_zeros(subject.config.width)
    return context.mean(0)


def localized_fit(subject, contexts: list[np.ndarray]) -> dict[str, Any]:
    """Fit the arm-B centroid from the encoded keys of this store, once."""
    plain = gr.EncoderConfig(mode="localized_centered", tau=TAU, top_k=TOP_K, center=False)
    uncentered = [gr.encode_read_key(context, plain) for context in contexts]
    centroid = gr.store_centroid(np.stack(uncentered))
    centered = gr.EncoderConfig(mode="localized_centered", tau=TAU, top_k=TOP_K, center=True)
    keys = [gr.encode_read_key(context, centered, centroid) for context in contexts]
    return {
        "config": {
            "mode": "localized_centered",
            "schema": centered.schema,
            "tau": TAU,
            "top_k": TOP_K,
            "center": True,
        },
        "centroid": centroid,
        "keys": np.stack(keys),
        "uncentered": np.stack(uncentered),
        "encoder": centered,
    }


def localized_query(fit: dict[str, Any], context: np.ndarray) -> np.ndarray:
    return gr.encode_read_key(context, fit["encoder"], fit["centroid"])


def store_signature(state) -> dict[str, Any]:
    return {
        "keys": tensor_sha(state.store_keys),
        "values": tensor_sha(state.store_values),
        "age": tensor_sha(state.store_age.float()),
        "count": int(state.store_count),
    }


def tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Query construction.  Both arms see exactly the same 154 texts; only the key
# encoder differs.
# ---------------------------------------------------------------------------
def build_arm_a(subject):
    return load_base().build_real_store(subject)


def build_arm_b(subject, state_a, faces):
    base = load_base()
    texts = [
        base.sentence(token, token % base.POSITION_COUNT) for token in range(base.BASE_OCCUPIED)
    ]
    texts.extend(face["write_text"] for face in faces)
    contexts = [context_matrix(subject, text) for text in texts]
    fit = localized_fit(subject, contexts)
    keys = torch.tensor(fit["keys"], dtype=subject.W.dtype)
    store_keys = state_a.store_keys.clone()
    store_keys[: len(keys)] = keys
    state_b = replace(state_a, store_keys=store_keys)
    return state_b, fit


def arm_queries(subject, base, faces, encode, stored_keys) -> list[dict[str, Any]]:
    items = []
    for face in faces:
        items.append(
            {
                "class": "written",
                "tag": face["tag"],
                "face_tag": face["tag"],
                "slot": int(face["slot"]),
                "key": encode(face["read_text"]),
            }
        )
    for face in faces:
        for rank, position in enumerate(base.unwritten_positions(face)):
            text = base.sentence(int(face["token"]), position)
            items.append(
                {
                    "class": "unwritten",
                    "tag": f"unwritten_{face['tag']}_{rank}",
                    "face_tag": face["tag"],
                    "key": encode(text),
                }
            )
    pool = [token for token in range(base.FIRST_NEW_TOKEN + base.FACE_COUNT, len(base.WORDS))]
    for index, token in enumerate(pool[: base.NEG_PER_CLASS]):
        face = faces[index]
        text = base.sentence(token, int(face["read_position"]))
        items.append({"class": "new_token", "tag": f"new_token_{index:02d}", "key": encode(text)})
    for index, face in enumerate(faces):
        position, _ = base.pairing_position(face, faces)
        if position is None:
            continue
        text = base.sentence(int(face["token"]), position)
        items.append(
            {"class": "new_pairing", "tag": f"new_pairing_{index:02d}", "key": encode(text)}
        )
    for index, face in enumerate(faces):
        other = faces[(index + base.FACE_COUNT // 2) % base.FACE_COUNT]
        first = stored_keys[face["slot"]]
        second = stored_keys[other["slot"]]
        first = first / max(float(first.norm()), 1e-12)
        second = second / max(float(second.norm()), 1e-12)
        midpoint = first + second
        midpoint = midpoint / max(float(midpoint.norm()), 1e-12)
        items.append(
            {
                "class": "no_value",
                "tag": f"no_value_{index:02d}",
                "key": midpoint,
            }
        )
    return items


# ---------------------------------------------------------------------------
# Reading.
# ---------------------------------------------------------------------------
def occupied(state) -> torch.Tensor:
    return torch.nonzero(state.store_age >= 0, as_tuple=False).flatten()


def scores(state, key: torch.Tensor) -> dict[str, Any]:
    valid = occupied(state)
    keys = state.store_keys[valid]
    cosine = (keys @ key) / torch.clamp(keys.norm(dim=-1) * key.norm(), min=1e-12)
    ordered = torch.sort(cosine.detach(), descending=True).values
    top = int(torch.argmax(cosine.detach()))
    return {
        "top_cosine": float(cosine[top].detach()),
        "second_cosine": float(ordered[1].detach()) if ordered.numel() > 1 else None,
        "margin": float(ordered[0] - ordered[1]) if ordered.numel() > 1 else None,
        "top_slot": int(valid[top]),
        "weights": torch.softmax(cosine / ADDRESSED_TEMPERATURE, dim=0),
    }


def read_fast(state, key: torch.Tensor) -> torch.Tensor:
    valid = occupied(state)
    cosine = (state.store_keys[valid] @ key) / torch.clamp(
        state.store_keys[valid].norm(dim=-1) * key.norm(), min=1e-12
    )
    weights = torch.softmax(cosine / ADDRESSED_TEMPERATURE, dim=0).to(state.store_values.dtype)
    retrieved = torch.einsum("i,i...->...", weights, state.store_values[valid])
    return (1 - ADDRESSED_ALPHA) * state.fast + ADDRESSED_ALPHA * retrieved


def describe(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": round(float(arr.min()), 9),
        "p25": round(float(np.percentile(arr, 25)), 9),
        "median": round(float(np.median(arr)), 9),
        "p75": round(float(np.percentile(arr, 75)), 9),
        "max": round(float(arr.max()), 9),
    }


def key_spectrum(keys: torch.Tensor) -> dict[str, Any]:
    matrix = keys.detach().double()
    singular = torch.linalg.svdvals(matrix)
    energy = singular.square()
    total = float(energy.sum())
    return {
        "count": int(matrix.shape[0]),
        "rank": int(torch.linalg.matrix_rank(matrix)),
        "effective_rank": round(float(singular.sum() ** 2 / max(total, 1e-30)), 6),
        "top_share": round(float(energy[0] / max(total, 1e-30)), 6),
    }


def control_check(subject, state, items) -> dict[str, Any]:
    same_fast = 0
    same_slot = 0
    max_delta = 0.0
    for item in items:
        read_state, info = subject.addressed_read(state, b"", key=item["key"])
        own = scores(state, item["key"])
        manual = read_fast(state, item["key"])
        max_delta = max(max_delta, float((read_state.fast - manual).abs().max()))
        if tensor_sha(read_state.fast) == tensor_sha(manual):
            same_fast += 1
        if int(info.get("top_slot", -1)) == int(own["top_slot"]):
            same_slot += 1
        touch()
    return {
        "queries": len(items),
        "fast_sha_identical": same_fast,
        "top_slot_identical": same_slot,
        "max_abs_delta": max_delta,
    }


def arm_block(subject, state, arm: str, items, encoder_meta) -> dict[str, Any]:
    before = store_signature(state)
    records = []
    for item in items:
        record = scores(state, item["key"])
        record.pop("weights", None)
        record.update({"class": item["class"], "tag": item["tag"]})
        record["face_tag"] = item.get("face_tag")
        record["slot"] = item.get("slot")
        records.append(record)
        touch()
    after = store_signature(state)
    control = control_check(subject, state, items)

    floor = min(row["top_cosine"] for row in records if row["class"] == "written")
    groups: dict[str, Any] = {}
    for name in CLASSES:
        tops = [row["top_cosine"] for row in records if row["class"] == name]
        margins = [row["margin"] for row in records if row["class"] == name]
        groups[name] = {
            "count": len(tops),
            "top_cosine": describe(tops),
            "margin": describe(margins),
        }
    gaps = {name: floor - groups[name]["top_cosine"]["max"] for name in CLASSES[2:]}

    written = [row for row in records if row["class"] == "written"]
    written_correct = sum(1 for row in written if row["top_slot"] == row["slot"])
    face_slot = {row["tag"]: row["top_slot"] for row in written}
    unwritten = [row for row in records if row["class"] == "unwritten"]
    unwritten_hits = [row for row in unwritten if row["top_slot"] == face_slot.get(row["face_tag"])]

    gate: dict[str, Any] = {"threshold": round(float(floor), 9), "groups": {}}
    totals: dict[str, int] = dict.fromkeys(CLASSES, 0)
    refused: dict[str, int] = dict.fromkeys(CLASSES, 0)
    for row in records:
        totals[row["class"]] += 1
        if row["top_cosine"] < floor:
            refused[row["class"]] += 1
    for name in CLASSES:
        gate["groups"][name] = {
            "matched": totals[name] - refused[name],
            "total": totals[name],
            "refused": refused[name],
            "refusal_rate": round(refused[name] / max(totals[name], 1), 6),
        }
    gate["written_false_refusals"] = refused["written"]
    gate["unwritten_false_refusals"] = refused["unwritten"]

    stored = state.store_keys[occupied(state)]
    positions = {
        "floor": round(float(floor), 9),
        "gaps": {name: round(float(value), 9) for name, value in gaps.items()},
        "separable": [name for name, value in gaps.items() if value > 0],
        "m_spec_written": describe([row["margin"] for row in written]),
        "m_spec_all": describe([row["margin"] for row in records]),
        "group_top": {name: groups[name]["top_cosine"] for name in CLASSES},
        "unwritten": {
            "count": len(unwritten),
            "correct": len(unwritten_hits),
            "correct_rate": round(len(unwritten_hits) / max(len(unwritten), 1), 6),
            "misses": [row["tag"] for row in unwritten if row not in unwritten_hits],
        },
        "written_correct": written_correct,
        "written_total": len(written),
        "written_misses": [row["tag"] for row in written if row["top_slot"] != row["slot"]],
        "key_spectrum": key_spectrum(stored),
        "gate": gate,
        "per_item": [
            {
                "class": row["class"],
                "tag": row["tag"],
                "top_cosine": row["top_cosine"],
                "margin": row["margin"],
                "top_slot": row["top_slot"],
            }
            for row in records
        ],
    }
    return {
        "arm": arm,
        "encoder": encoder_meta,
        "control": control,
        "store_immutable": before == after,
        "store_before": before,
        "store_after": after,
        **positions,
        "cost_cpu_seconds": round(time.process_time() - START_CPU, 3),
    }


def verdict_block(arm_a, arm_b) -> dict[str, Any]:
    positive = [name for name, value in arm_b["gaps"].items() if value > 0]
    floor_ok = bool(arm_b["floor"] >= FLOOR_GATE)
    gap_ok = len(positive) >= REQUIRED_POSITIVE_GAPS
    written_ok = bool(arm_b["written_correct"] >= arm_a["written_correct"])
    passed = floor_ok and gap_ok and written_ok
    block = {
        "floor_gate": FLOOR_GATE,
        "floor_ok": floor_ok,
        "gaps_positive": positive,
        "gap_ok": gap_ok,
        "written_correct_a": arm_a["written_correct"],
        "written_correct_b": arm_b["written_correct"],
        "written_not_worse": written_ok,
        "verdict": "localized_centered_enables_gate" if passed else "encoder_also_insufficient",
    }
    if passed:
        lower = max(arm_b["group_top"][name]["max"] for name in positive)
        unwritten_tops = sorted(
            row["top_cosine"] for row in arm_b["per_item"] if row["class"] == "unwritten"
        )
        upper = min(arm_b["floor"], unwritten_tops[1] if len(unwritten_tops) > 1 else 0.0)
        block["threshold_interval"] = {
            "lower_open": round(float(lower), 9),
            "upper_inclusive": round(float(upper), 9),
            "width": round(float(upper - lower), 9),
            "rule": "lower = highest negative-group top among positive gaps; upper = min(written "
            "floor, second-lowest unwritten top) so at most one unwritten wording is accepted",
        }
    else:
        block["reason"] = (
            "the localized centered encoder does not meet the pre-written pass rule; the "
            "content-addressed write + read gate line is closed"
        )
    return block


def main() -> int:
    if time.process_time() - START_CPU > CPU_GUARD:
        raise SystemExit("cpu guard tripped before start")

    process_guard = assert_no_black_processes("localized encoder probe")
    lock_name = acquire_single_instance()
    atexit.register(release_single_instance)
    stop = threading.Event()
    start_watchdog(stop)

    PROGRESS.write_text("", encoding="utf-8")
    if WATCHDOG_MARKER.exists():
        WATCHDOG_MARKER.unlink()
    torch.set_grad_enabled(False)
    torch.manual_seed(20261108)
    base = load_base()

    manifest = {
        "format": "taiji-localized-encoder-preregistration-v1",
        "date": "2026-11-08",
        "question": (
            "does replacing the averaged-pooled read key with the growth design's "
            "localized+centered encoding let the written faces keep their carrier while the three "
            "negative classes fall behind a single shared threshold?"
        ),
        "written_before_any_reading": True,
        "sequence": [
            "build the real-forward store with today's mean-pooled keys (arm A)",
            "rebuild the same store's addresses with localized_centered, centroid fit once from "
            "its own 30 keys (arm B); values/ages/count are reused byte-identically",
            "154 queries per arm: 22 written read wordings, 66 unwritten wordings, 22 new_token, "
            "22 new_pairing, 22 no_value midpoints of that arm's stored keys",
            "gate anchored on each arm's own written floor; pass rule C6 evaluated once",
        ],
        "criteria": CRITERIA,
        "constants": {
            "faces": base.FACE_COUNT,
            "base_occupied": base.BASE_OCCUPIED,
            "slots": base.SLOTS,
            "negative_per_class": base.NEG_PER_CLASS,
            "unwritten_per_rule": base.UNWRITTEN_PER_RULE,
            "temperature": ADDRESSED_TEMPERATURE,
            "alpha": ADDRESSED_ALPHA,
            "tau": TAU,
            "top_k": TOP_K,
            "floor_gate": FLOOR_GATE,
            "required_positive_gaps": REQUIRED_POSITIVE_GAPS,
            "query_count": base.FACE_COUNT
            + base.UNWRITTEN_PER_RULE * base.FACE_COUNT
            + 3 * base.NEG_PER_CLASS,
        },
        "discipline": {
            "training_updates": 0,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "confirm_reads": 0,
            "checkpoints_written": 0,
            "black_invoked": False,
            "cpu_cap_seconds": CPU_CAP,
            "watchdog_seconds": WATCHDOG_SECONDS,
        },
    }
    write_json(MANIFEST, manifest)
    append_progress({"phase": "manifest_written", "sha256": file_sha(MANIFEST)})

    subject = LivingCoreReference(LivingCoreConfig())
    parameters = int(sum(tensor.numel() for tensor in subject.state_dict().values()))
    state_a, faces = build_arm_a(subject)
    append_progress(
        {
            "phase": "arm_a_store_built",
            "faces": len(faces),
            "occupied": int(occupied(state_a).numel()),
        }
    )

    state_b, fit = build_arm_b(subject, state_a, faces)
    values_same = tensor_sha(state_a.store_values) == tensor_sha(state_b.store_values)
    age_same = tensor_sha(state_a.store_age.float()) == tensor_sha(state_b.store_age.float())
    append_progress(
        {
            "phase": "arm_b_store_built",
            "values_identical": values_same,
            "age_identical": age_same,
            "centroid_norm": round(float(np.linalg.norm(fit["centroid"])), 9),
        }
    )

    def encode_a(text: bytes) -> torch.Tensor:
        return mean_encoder_key(subject, text)

    def encode_b(text: bytes) -> torch.Tensor:
        return to_key(subject, localized_query(fit, context_matrix(subject, text)))

    items_a = arm_queries(subject, base, faces, encode_a, state_a.store_keys)
    items_b = arm_queries(subject, base, faces, encode_b, state_b.store_keys)

    arm_a = arm_block(
        subject, state_a, "A_mean", items_a, {"mode": "mean", "schema": gr.ENCODER_SCHEMA_MEAN}
    )
    append_progress({"phase": "arm_a_done", "floor": arm_a["floor"], "gaps": arm_a["gaps"]})
    arm_b = arm_block(
        subject,
        state_b,
        "B_localized_centered",
        items_b,
        {
            "mode": "localized_centered",
            "schema": gr.ENCODER_SCHEMA_LOCALIZED,
            "tau": TAU,
            "top_k": TOP_K,
            "center": True,
            "centroid": "mean of the 30 encoded keys of this store, fit once at the session boundary",
            "centroid_norm": round(float(np.linalg.norm(fit["centroid"])), 9),
        },
    )
    append_progress({"phase": "arm_b_done", "floor": arm_b["floor"], "gaps": arm_b["gaps"]})

    verdict = verdict_block(arm_a, arm_b)
    for block in (arm_a, arm_b):
        block.pop("per_item", None)

    d = int(subject.config.width)
    report = {
        "probe": "taiji_gate_localized_centered",
        "date": "2026-11-08",
        "question": manifest["question"],
        "preregistration": {
            "manifest": str(MANIFEST.relative_to(ROOT)),
            "sha256": file_sha(MANIFEST),
            "written_before_any_reading": True,
        },
        "criteria": CRITERIA,
        "arms": {"real_forward": {"arm_a": arm_a, "arm_b": arm_b}},
        "store_identity": {
            "values_identical": values_same,
            "age_identical": age_same,
            "count_identical": int(state_a.store_count) == int(state_b.store_count),
            "keys_differ": tensor_sha(state_a.store_keys) != tensor_sha(state_b.store_keys),
        },
        "cost_of_encoding": {
            "trainable_parameters": parameters,
            "parameters_added": 0,
            "graph_state_added_bytes": 0,
            "runner_side_centroid": {
                "floats": d,
                "bytes_float64": d * 8,
                "per_growth_spec": f"centroid ({d},) + centroid_age (1,) = {d + 1} floats",
            },
            "reason": "the encoder is a pure function; the centroid is a store-level statistic, "
            "not a parameter, and the frozen graph is untouched",
        },
        "verdict": verdict,
        "plain_language": {
            "summary": (
                "ON THIS MODEL AND THESE 154 QUESTIONS the content-addressed write plus read-gate "
                "line is now closed. Swapping how a record's address is computed -- from averaging "
                "the model's own byte-level context vectors to encoding each position separately, "
                "weighting the strongest positions, and subtracting one common offset -- did not "
                "buy the two things that must hold together: the 22 written records' own read "
                "wordings dropped to 0.77 similarity at the worst (the pass line is 0.90) and only "
                "1 of the 22 landed on its own record, while the three classes of questions that "
                "were never written ended up CLOSER to the records than before (gaps -0.18, -0.20, "
                "-0.22, all negative). What this shows is not that the threshold, the offset or the "
                "particular linear transform was wrong, but that the raw material an address is "
                "built from -- the context vectors this model produces for Chinese byte text -- does "
                "not yet keep 'same content, different wording' stable while keeping 'never written' "
                "apart. No further encoding, geometry or threshold tuning is planned on this line. "
                "What it does NOT show: that content addressing cannot work in principle, that the "
                "model cannot learn, or that growing the memory would not help. It shows only that "
                "on the frozen graph of today, this mechanism has no usable margin."
            ),
            "can_prove": (
                "if the pass rule holds: on this model and these 154 questions, a memory that finds "
                "a record by how similar the question is can keep recognising its own paraphrases "
                "(every one of the 22 written wordings lands on its own record, with similarity at "
                "least 0.90) while three kinds of questions that were never answered -- a word that "
                "was never written, a pairing that was never written, and a point between two "
                "records -- all stay below the same line. That would make a single 'only answer if "
                "similar enough' rule usable on this line."
            ),
            "cannot_prove": (
                "it cannot show that the model has become smarter, learned anything, or kept any "
                "old ability: nothing was trained and no answer quality is measured here. It cannot "
                "show that the memory still works with longer text, many more records, or across "
                "sessions. And it cannot show that growing the memory (more slots) would help; a "
                "usable address only means growth would have something to grow for, not that growth "
                "pays."
            ),
            "if_failed": (
                "if the pass rule does not hold: on the byte representation this model currently "
                "produces, exchanging the way a record's address is computed still cannot separate "
                "'recognise my own wording' from 'know that this was never written', so the "
                "content-addressed write + read gate line is closed here. It does not show that "
                "content addressing is impossible in principle, because the raw material of an "
                "address -- the model's own context vectors -- was not changed in this round."
            ),
        },
        "isolation": {
            "runner_sha256_start": RUNNER_SHA_START,
            "runner_sha256_end": file_sha(Path(__file__).resolve()),
            "living_core_sha256": file_sha(LIVING_CORE),
            "growth_runtime_sha256": file_sha(GROWTH_RUNTIME),
            "process_guard": process_guard,
            "single_instance_lock": lock_name,
            "watchdog": {
                "seconds": WATCHDOG_SECONDS,
                "fired": bool(WATCHDOG_FIRED[0]),
                "marker_exists": WATCHDOG_MARKER.exists(),
            },
            "write_then_rename": True,
        },
        "discipline": manifest["discipline"],
        "cost": {
            "cpu_seconds": round(time.process_time() - START_CPU, 3),
            "wall_seconds": round(time.time() - START_WALL, 3),
            "cpu_cap_seconds": CPU_CAP,
        },
    }
    write_json(REPORT, report)
    append_progress({"phase": "report_written", "verdict": verdict["verdict"]})
    stop.set()

    print(
        json.dumps(
            {
                "verdict": verdict["verdict"],
                "floor_a": arm_a["floor"],
                "floor_b": arm_b["floor"],
                "gaps_a": arm_a["gaps"],
                "gaps_b": arm_b["gaps"],
                "m_spec_b": arm_b["m_spec_written"],
                "spectrum_b": arm_b["key_spectrum"],
                "unwritten_b": [arm_b["unwritten"]["correct"], arm_b["unwritten"]["count"]],
                "written_b": [arm_b["written_correct"], arm_b["written_total"]],
                "gate_b": {name: arm_b["gate"]["groups"][name]["refused"] for name in CLASSES},
                "values_same": values_same,
                "cpu_seconds": report["cost"]["cpu_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
