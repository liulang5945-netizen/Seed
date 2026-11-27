"""Simplified gate retest: does a gate that keeps only capacity and copy hold?

The previous run (``probe_taiji_gate_task_value_20261108``) judged the four-clause
gate against no gate on package 20's task shape and returned ``gate_hurts``: it
refused two writes that broke nothing and cost two correction faces.  This run
takes the owner's decision and measures the gate that survives it:

    S  simplified gate + growth   only two refusals exist:
                                  ``no_direction`` (the copy band) and
                                  ``no_capacity`` (arithmetic headroom).  Before
                                  refusing on capacity, the store grows by
                                  doubling (``growth_runtime.plan_growth``) and
                                  the reserve ``h >= 0.125`` is re-checked on the
                                  grown store; only a failed growth refuses, and
                                  a refusal rolls back bit-identically.
    U  no gate at all             append every write, never refuse, never evict;
                                  grows only when the store is physically full.
                                  This is the ceiling the simplified gate is
                                  measured against.

G1 (retention), G2 (read margin), G4 (value residual) stay in the certificate as
readings only.  They are computed by ``growth_gate.evaluate`` and cannot refuse
here: the refusal set is intersected with the two retained codes, and the
dropped codes are listed in the report so the exclusion is auditable.

Same fixture as the previous run, same 22 correction faces, same write order,
same 7/7 exact-key battery, no eviction, no training, no optimizer step, no
Adam, no confirm read.

Registered before the first arm reading (also written to the manifest):

    S corrections == 22, retention == 7, evictions == 0, dropped == 0
        -> simplified_holds_at_parity
    S corrections < 22
        -> simplified_below_parity, with every missing face classified as
           refused / written-but-read-wrong / slot-value-corrupted
    evictions > 0 or dropped > 0
        -> simplified_failed
    margin below the 0.02 collapse line after a growth
        -> recorded only, never judged (the owner decides the capacity/precision
           trade-off)

Three readings are always reported, none optional: the no-gate ceiling, what the
simplified gate loses against it and why, and the number of grown slots and
refused writes.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji import growth_gate as gg  # noqa: E402
from taiji import growth_runtime as gr  # noqa: E402
from taiji.process_guard import assert_no_black_processes  # noqa: E402

REPORT = ROOT / "reports/taiji_gate_simplified_20261108.json"
MANIFEST = ROOT / "plans/manifests/taiji_gate_simplified_20261108.json"
PROGRESS = ROOT / "reports/taiji_gate_simplified_20261108_progress.jsonl"
JOURNAL_DIR = ROOT / "reports/taiji_gate_simplified_20261108_journal"
LOCK = ROOT / "reports/taiji_gate_simplified_20261108.lock"
WATCHDOG_MARKER = ROOT / "reports/taiji_gate_simplified_20261108_watchdog.json"
F1_RUNNER = ROOT / "scripts/training/probe_taiji_growth_f1_correction_20261108.py"
GATE_FILE = ROOT / "taiji/growth_gate.py"
RUNTIME_FILE = ROOT / "taiji/growth_runtime.py"

CPU_CAP = 200.0
CPU_GUARD = 190.0
WATCHDOG_SECONDS = 60.0
BASE_OCCUPIED = 8
SLOTS = 32
FACE_COUNT = 22
CORRECTION_POSITION = 5
POSITION_COUNT = 8
SYNONYM_POSITIONS = (1, 2, 3, 4)
FIRST_NEW_TOKEN = 20
COLLAPSE_LINE = 0.02  # growth design S-B line; recorded, never judged here

# the only two clauses allowed to refuse in the simplified gate
RETAINED_CLAUSES = (gg.EXIT_NO_CAPACITY, gg.EXIT_NO_DIRECTION)
# clauses removed from the decision path; still measured by evaluate()
DROPPED_CLAUSES = (
    gg.EXIT_RETENTION,
    gg.EXIT_SEPARABILITY,
    gg.EXIT_INFLATION,
    gg.EXIT_NO_GAIN,
)

CRITERIA = {
    "S1_corrections": (
        "the simplified gate must reach 22/22: face i writes its correction at position 5 "
        "and is read with the same token at another position; correct iff the final store's "
        "top-1 slot for that read is the slot the face was written to"
    ),
    "S2_retention": (
        "the 7 held-out exact keys of the 8-row baseline must still read back to their own "
        "slots; integer count out of 7"
    ),
    "S3_no_loss": "no eviction and no dropped write may occur; either one fails the run",
    "S4_failure_attribution": (
        "if a face is not correct, it is classified as refused (owns no slot), "
        "written_but_read_wrong (its slot still holds its value but the synonym reads another "
        "slot) or slot_value_corrupted (its own slot no longer holds its value)"
    ),
    "S5_ceiling": (
        "the same 22 faces are appended with no gate and no refusal; that reading is the "
        "ceiling, and the simplified gate's loss is reported face by face with its cause"
    ),
    "S6_growth": (
        "before any capacity refusal the store doubles per growth_runtime.plan_growth and the "
        "reserve h >= 0.125 is re-checked on the grown store; a failed growth is the only "
        "capacity refusal and it rolls back bit-identically without evicting"
    ),
    "S7_collapse_line": (
        "if the post-growth read margin falls below 0.02 it is recorded and marked as not used "
        "by the verdict; the capacity/precision trade-off is the owner's call"
    ),
    "judge_rule": (
        "22/22 + 7/7 + 0 evictions + 0 dropped -> simplified_holds_at_parity; corrections < 22 "
        "-> simplified_below_parity with per-face causes; any eviction or dropped write -> "
        "simplified_failed"
    ),
    "no_soft_language": (
        "a face that the gate loses is reported as lost, never as 'close'; a run with no "
        "refusals may not be described as evidence that refusals are safe"
    ),
    "unchanged": "no threshold, fixture, face or battery is retuned after the first reading",
}

START_CPU = time.process_time()
START_WALL = time.time()
LAST_READING = [time.time()]
WATCHDOG_FIRED = [False]

f1 = None
base0 = None
tokens0 = None
contexts0 = None
payloads = None
encoder = None
centroid = None


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with tmp.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def append_progress(record: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    LAST_READING[0] = time.time()


def reset_run_logs() -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("", encoding="utf-8")
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    for stale in JOURNAL_DIR.glob("*.jsonl"):
        stale.unlink()
    leftover = REPORT.with_suffix(REPORT.suffix + ".tmp")
    if leftover.exists():
        leftover.unlink()
    WATCHDOG_MARKER.unlink(missing_ok=True)


def acquire_single_instance() -> str:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 5.0
    while True:
        try:
            handle = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            age = time.time() - LOCK.stat().st_mtime
            if age > 600.0:
                LOCK.unlink(missing_ok=True)
                continue
            if time.time() >= deadline:
                raise SystemExit(
                    f"refusing to start: another run holds {LOCK.name} (age {age:.0f}s)"
                ) from None
            time.sleep(0.2)
            continue
        os.write(handle, f"{os.getpid()}\n".encode())
        os.close(handle)
        return LOCK.name


def release_single_instance() -> None:
    LOCK.unlink(missing_ok=True)


def start_watchdog(stop: threading.Event) -> threading.Thread:
    """Abort the run if no new reading lands for WATCHDOG_SECONDS."""

    def loop() -> None:
        while not stop.wait(1.0):
            idle = time.time() - LAST_READING[0]
            if idle <= WATCHDOG_SECONDS:
                continue
            WATCHDOG_FIRED[0] = True
            write_json(
                WATCHDOG_MARKER,
                {
                    "fired": True,
                    "idle_seconds": round(idle, 3),
                    "limit_seconds": WATCHDOG_SECONDS,
                    "pid": os.getpid(),
                    "cpu_seconds": round(time.process_time() - START_CPU, 3),
                },
            )
            os._exit(3)

    thread = threading.Thread(target=loop, name="no-read-watchdog", daemon=True)
    thread.start()
    return thread


def journal_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_f1():
    spec = importlib.util.spec_from_file_location("f1_probe", F1_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {F1_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def active(state: gg.GateState) -> tuple[np.ndarray, np.ndarray]:
    rows = int(state.occupied)
    return state.keys[:rows], state.values[:rows]


def top_slot(state: gg.GateState, key: np.ndarray) -> int:
    keys, _ = active(state)
    unit = key / max(float(np.linalg.norm(key)), 1e-12)
    cosine = (keys / np.clip(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12, None)) @ unit
    return int(np.argmax(cosine))


def read_value(state: gg.GateState, key: np.ndarray) -> np.ndarray:
    keys, values = active(state)
    unit = key / max(float(np.linalg.norm(key)), 1e-12)
    cosine = (keys / np.clip(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12, None)) @ unit
    logits = cosine / gg.GATE_TEMPERATURE
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    return (weights[:, None] * values).sum(axis=0)


def relative_residual(read: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(read - target) / max(float(np.linalg.norm(target)), 1e-12))


def spec_margin(state: gg.GateState) -> float:
    keys, _ = active(state)
    count = int(keys.shape[0])
    if count < 3:
        return 0.0
    normalized = keys / np.clip(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12, None)
    gram = normalized @ normalized.T
    np.fill_diagonal(gram, -np.inf)
    ordered = np.sort(gram, axis=1)[:, ::-1]
    return float(np.median(ordered[:, 0] - ordered[:, 1]))


def build_state(
    occupied: int,
    slots: int,
    contexts: list[np.ndarray],
    values: np.ndarray,
) -> gg.GateState:
    keys = np.zeros((slots, f1.DIM), dtype=np.float64)
    store_values = np.zeros((slots, int(np.prod(f1.VALUE_SHAPE))), dtype=np.float64)
    for slot in range(occupied):
        keys[slot] = gr.encode_read_key(contexts[slot], encoder, centroid)
        store_values[slot] = values[slot]
    return gg.GateState(
        keys=keys,
        values=store_values,
        age=np.array(list(range(occupied)) + [-1] * (slots - occupied)),
        hits=np.zeros(slots, dtype=np.int64),
        occupied=occupied,
        metric_version=0,
    )


def grow_state(state: gg.GateState, target_slots: int) -> gg.GateState:
    """Allocate a wider store, keeping every byte of the occupied rows."""
    target_slots = int(target_slots)
    keys = np.zeros((target_slots, state.keys.shape[1]), dtype=state.keys.dtype)
    values = np.zeros((target_slots, state.values.shape[1]), dtype=state.values.dtype)
    age = np.full(target_slots, -1, dtype=state.age.dtype)
    hits = np.zeros(target_slots, dtype=state.hits.dtype)
    old = int(state.slots)
    keys[:old] = state.keys
    values[:old] = state.values
    age[:old] = state.age
    hits[:old] = state.hits
    return gg.GateState(
        keys=keys,
        values=values,
        age=age,
        hits=hits,
        occupied=int(state.occupied),
        metric_version=state.metric_version,
    )


def battery(state: gg.GateState, occupied: int) -> list[gg.BatteryQuery]:
    return [
        gg.BatteryQuery(
            name=f"held_out_rule_{index}",
            key=state.keys[index].copy(),
            expect_slot=index,
            family=f"rule_{index}",
        )
        for index in range(1, occupied)
    ]


def retention_count(state: gg.GateState, queries: list[gg.BatteryQuery]) -> int:
    return sum(1 for row in queries if top_slot(state, row.key) == row.expect_slot)


def new_tokens(count: int) -> list[int]:
    pool = [
        token
        for token in range(FIRST_NEW_TOKEN, len(tokens0))
        if token % POSITION_COUNT != CORRECTION_POSITION
    ]
    return pool[:count]


def synonym_position(token: int, index: int) -> int:
    """A deterministic position for the read phrasing, never the write position."""
    position = SYNONYM_POSITIONS[index % len(SYNONYM_POSITIONS)]
    if position == token % POSITION_COUNT:
        position = SYNONYM_POSITIONS[(index + 1) % len(SYNONYM_POSITIONS)]
    return position


def items() -> list[dict[str, Any]]:
    tokens = new_tokens(FACE_COUNT)
    rows = []
    for index, token in enumerate(tokens):
        write_context = f1.paraphrase(base0, tokens0[token], CORRECTION_POSITION)
        position = synonym_position(token, index)
        read_context = f1.paraphrase(base0, tokens0[token], position)
        rows.append(
            {
                "index": index,
                "tag": f"face_{index:02d}",
                "token": int(token),
                "read_position": int(position),
                "key": gr.encode_read_key(write_context, encoder, centroid),
                "read_key": gr.encode_read_key(read_context, encoder, centroid),
                "value": payloads[BASE_OCCUPIED + index],
                "family": "new_direction",
                "policy": "append",
            }
        )
    return rows


def old_paraphrase_queries() -> list[dict[str, Any]]:
    """Labelled diagnostic: the 7 baseline entries asked with a new word order."""
    rows = []
    for index in range(1, BASE_OCCUPIED):
        position = SYNONYM_POSITIONS[(index - 1) % len(SYNONYM_POSITIONS)]
        if position == index % POSITION_COUNT:
            position = SYNONYM_POSITIONS[index % len(SYNONYM_POSITIONS)]
        context = f1.paraphrase(base0, tokens0[index], position)
        rows.append(
            {
                "index": index,
                "read_position": int(position),
                "key": gr.encode_read_key(context, encoder, centroid),
            }
        )
    return rows


def face_outcome(state: gg.GateState, face: dict[str, Any], slot: int | None) -> dict[str, Any]:
    """Correct, or one of the three registered failure kinds."""
    if slot is None:
        return {
            "tag": face["tag"],
            "token": face["token"],
            "read_position": face["read_position"],
            "written_slot": None,
            "read_top_slot": None,
            "correct": False,
            "outcome": "refused",
            "value_residual": None,
        }
    observed = top_slot(state, face["read_key"])
    intact = bool(np.array_equal(state.values[int(slot)], face["value"]))
    if observed == int(slot):
        outcome = "correct"
    elif intact:
        outcome = "written_but_read_wrong"
    else:
        outcome = "slot_value_corrupted"
    return {
        "tag": face["tag"],
        "token": face["token"],
        "read_position": face["read_position"],
        "written_slot": int(slot),
        "read_top_slot": int(observed),
        "correct": bool(outcome == "correct"),
        "outcome": outcome,
        "value_residual": round(
            relative_residual(read_value(state, face["read_key"]), face["value"]), 6
        ),
    }


def run_simplified_arm(
    faces: list[dict[str, Any]],
    initial: gg.GateState,
    queries: list[gg.BatteryQuery],
) -> dict[str, Any]:
    """Capacity + copy only, with growth before a capacity refusal."""
    state = initial
    slots: dict[str, int] = {}
    rows = []
    growth_events = []
    journal_path = JOURNAL_DIR / "simplified_refusals.jsonl"
    journal_path.write_text("", encoding="utf-8")
    for face in faces:
        write = gg.WriteCell(
            key=face["key"],
            value=face["value"],
            family=face["family"],
            policy=face["policy"],
        )
        occupancy_before = int(state.occupied)
        slots_before = int(state.slots)
        sha_before = gg.snapshot(state).sha256
        headroom_before = gr.write_headroom(occupancy_before, slots_before, 1)
        grew = 0
        if not headroom_before["sufficient"]:
            plan = gr.plan_growth(headroom_before)
            if plan["action"] != "grow_then_write":
                record = {
                    "arm": "S_capacity_copy",
                    "tag": face["tag"],
                    "occupancy_before": occupancy_before,
                    "slots_before": slots_before,
                    "slots_after": slots_before,
                    "grown_slots": 0,
                    "accepted": False,
                    "exit_code": gg.EXIT_NO_CAPACITY,
                    "failed": [gg.EXIT_NO_CAPACITY],
                    "growth_plan": plan,
                    "written_slot": None,
                    "bit_identical": gg.snapshot(state).sha256 == sha_before,
                }
                with journal_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                append_progress(record)
                rows.append(record)
                continue
            target = int(plan["slots"])
            state = grow_state(state, target)
            grew = target - slots_before
            recheck = gr.write_headroom(int(state.occupied), int(state.slots), 1)
            growth_events.append(
                {
                    "tag": face["tag"],
                    "slots_from": slots_before,
                    "slots_to": int(state.slots),
                    "grown_slots": grew,
                    "occupancy": int(state.occupied),
                    "headroom_before_growth": round(headroom_before["h_slots"], 6),
                    "headroom_after_growth": round(recheck["h_slots"], 6),
                    "reserve_rechecked": bool(recheck["sufficient"]),
                    "margin_before_growth": round(spec_margin(state), 6),
                }
            )
        shadow, applied = gg.shadow_apply(state, write)
        cert = gg.evaluate(state, shadow, queries, write, applied["slot"])
        blocking = [code for code in cert.failed if code in RETAINED_CLAUSES]
        accepted = bool(applied["applied"]) and not blocking
        blocking_code = blocking[0] if blocking else cert.exit_code
        record = {
            "arm": "S_capacity_copy",
            "tag": face["tag"],
            "occupancy_before": occupancy_before,
            "slots_before": slots_before,
            "slots_after": int(state.slots),
            "grown_slots": grew,
            "accepted": accepted,
            "exit_code": blocking_code if not accepted else cert.exit_code,
            "failed": list(cert.failed),
            "blocking": blocking,
            "dropped_clause_failures": [c for c in cert.failed if c in DROPPED_CLAUSES],
            "growth_plan": {"action": "write_now", "slots": int(state.slots)},
            "written_slot": int(applied["slot"]) if accepted else None,
            "headroom_after_write": (
                round((int(state.slots) - occupancy_before - 1) / int(state.slots), 6)
            ),
            "margin_after": round(cert.g2_margin_after, 6),
            "value_residual": (
                None if cert.g4_value_residual is None else round(float(cert.g4_value_residual), 6)
            ),
            "direction_verdict": cert.direction_verdict,
            "evictions": 0,
            "dropped": 0,
            "bit_identical": None if accepted else gg.snapshot(state).sha256 == sha_before,
        }
        if accepted:
            state = shadow
            slots[face["tag"]] = int(applied["slot"])
        else:
            with journal_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        append_progress(record)
        rows.append(record)
    return {
        "state": state,
        "slots": slots,
        "rows": rows,
        "growth_events": growth_events,
        "quarantine_rows": journal_rows(journal_path),
    }


def run_ceiling_arm(
    faces: list[dict[str, Any]],
    initial: gg.GateState,
    queries: list[gg.BatteryQuery],
) -> dict[str, Any]:
    """No gate: append every write, never refuse, never evict.

    The store only widens when it is physically full; that is storage, not a
    judgement, and at 8 + 22 of 32 slots it does not happen.
    """
    state = initial
    slots: dict[str, int] = {}
    rows = []
    grown = 0
    for face in faces:
        write = gg.WriteCell(
            key=face["key"],
            value=face["value"],
            family=face["family"],
            policy=face["policy"],
        )
        shadow, applied = gg.shadow_apply(state, write)
        if not applied["applied"]:
            slots_before = int(state.slots)
            target = slots_before
            while target <= int(state.occupied):
                target *= 2
            state = grow_state(state, target)
            grown += target - slots_before
            shadow, applied = gg.shadow_apply(state, write)
        record = {
            "arm": "U_no_gate",
            "tag": face["tag"],
            "occupancy_before": int(state.occupied),
            "slots_after": int(state.slots),
            "accepted": bool(applied["applied"]),
            "exit_code": "not_evaluated_no_gate",
            "failed": [],
            "written_slot": int(applied["slot"]) if applied["applied"] else None,
            "evictions": int(applied["evictions"]),
            "dropped": int(applied["dropped"]),
        }
        if applied["applied"]:
            state = shadow
            slots[face["tag"]] = int(applied["slot"])
        append_progress(record)
        rows.append(record)
    return {"state": state, "slots": slots, "rows": rows, "grown_slots": grown}


def arm_summary(
    name: str,
    run: dict[str, Any],
    faces: list[dict[str, Any]],
    queries: list[gg.BatteryQuery],
    paraphrases: list[dict[str, Any]],
) -> dict[str, Any]:
    state = run["state"]
    outcomes = [face_outcome(state, face, run["slots"].get(face["tag"])) for face in faces]
    counts: dict[str, int] = {}
    for row in outcomes:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    return {
        "arm": name,
        "occupied_final": int(state.occupied),
        "slots_final": int(state.slots),
        "headroom_final": round(
            (int(state.slots) - int(state.occupied)) / max(int(state.slots), 1), 6
        ),
        "spec_margin_final": round(spec_margin(state), 6),
        "retention_correct": retention_count(state, queries),
        "retention_total": len(queries),
        "corrections_correct": sum(1 for row in outcomes if row["correct"]),
        "corrections_total": len(faces),
        "rejected_writes": sum(1 for row in run["rows"] if not row["accepted"]),
        "evictions_total": int(sum(row.get("evictions", 0) for row in run["rows"])),
        "dropped_writes_total": int(sum(row.get("dropped", 0) for row in run["rows"])),
        "outcome_counts": counts,
        "paraphrase_retention_correct": sum(
            1 for row in paraphrases if top_slot(state, row["key"]) == row["index"]
        ),
        "paraphrase_retention_total": len(paraphrases),
        "faces": outcomes,
        "writes": run["rows"],
    }


def main() -> int:
    if time.process_time() - START_CPU > CPU_GUARD:
        raise SystemExit("cpu guard tripped before start")

    process_guard = assert_no_black_processes("simplified gate retest")
    lock_name = acquire_single_instance()
    atexit.register(release_single_instance)
    stop = threading.Event()
    start_watchdog(stop)

    global f1, base0, tokens0, contexts0, payloads, encoder, centroid

    reset_run_logs()
    f1 = load_f1()
    base0, tokens0, contexts0 = f1.fixture()
    payloads = f1.value_payloads(BASE_OCCUPIED + FACE_COUNT)
    legacy = gr.EncoderConfig(mode="mean")
    parents = np.stack([gr.encode_read_key(contexts0[i], legacy) for i in range(4)])
    centroid = gr.store_centroid(parents)
    encoder = gr.EncoderConfig(mode="localized_centered", center=True)
    if int(f1.DIM) != 128:
        raise SystemExit(f"fixture width {f1.DIM} is not the 128-d store width")

    initial = build_state(
        BASE_OCCUPIED, SLOTS, [contexts0[i] for i in range(BASE_OCCUPIED)], payloads
    )
    queries = battery(initial, BASE_OCCUPIED)
    faces = items()
    paraphrases = old_paraphrase_queries()

    baseline = {
        "occupied": int(initial.occupied),
        "slots": int(initial.slots),
        "headroom": round((SLOTS - BASE_OCCUPIED) / SLOTS, 6),
        "spec_margin": round(spec_margin(initial), 6),
        "retention_correct": retention_count(initial, queries),
        "retention_total": len(queries),
    }
    append_progress({"phase": "baseline", **baseline})

    manifest = {
        "format": "taiji-gate-simplified-preregistration-v1",
        "date": "2026-11-08",
        "question": (
            "does a gate that keeps only capacity and copy reach parity with no gate on the "
            "22-face task, after growing the store instead of refusing?"
        ),
        "written_before_any_arm_reading": True,
        "written_before_any_case_verdict": True,
        "sequence": [
            "phase 0 baseline geometry (read only, no verdict)",
            "this manifest is written and hashed",
            "arm U: no gate at all (ceiling)",
            "arm S: simplified gate, capacity + copy only, growth before refusal",
            "verdict from the registered rule",
        ],
        "mapping": {
            "task_shape": (
                "package 20: 22 correction surfaces per world on a content-addressed store "
                "that starts at 32 slots (living_core.ADDRESSED_SLOTS) and may grow"
            ),
            "store": (
                "existing F1/F3 fixture unchanged: same encoder (localized_centered, legacy "
                "centroid), 128-d keys, 8192-d values, ADDRESSED_TEMPERATURE 0.1"
            ),
            "why_not_the_model": (
                "the gate is a numpy store check; inserting it into the torch graph would change "
                "the frozen read path. This probe measures it where it acts, on store writes"
            ),
            "initial_occupancy": BASE_OCCUPIED,
            "initial_slots": SLOTS,
            "face_count": FACE_COUNT,
            "face_definition": (
                "face i writes a correction whose value token is the i-th new token at "
                "position 5, and is read with the same token at a different position; a face "
                "is correct iff the read's top-1 slot is the slot it was written to"
            ),
            "no_eviction": "append only; a full store is refused, never evicted",
        },
        "retained_clauses": list(RETAINED_CLAUSES),
        "dropped_clauses": list(DROPPED_CLAUSES),
        "dropped_clauses_are_readings_only": True,
        "growth_protocol": {
            "trigger": "growth_runtime.write_headroom(...).sufficient is false for the next write",
            "step": "growth_runtime.plan_growth doubles until (slots - needed)/slots >= 0.125",
            "recheck": "the reserve is re-checked on the grown store before the write",
            "on_failure": "refuse with commit_rejected_no_capacity, roll back bit-identically",
            "slot_cap": None,
        },
        "constants": {
            "base_occupied": BASE_OCCUPIED,
            "initial_slots": SLOTS,
            "face_count": FACE_COUNT,
            "correction_position": CORRECTION_POSITION,
            "synonym_positions": list(SYNONYM_POSITIONS),
            "collapse_line": COLLAPSE_LINE,
            "gate_thresholds": {
                "g1_max_retention_drop": gg.GATE_G1_MAX_RETENTION_DROP,
                "g2_max_margin_drop": gg.GATE_G2_MAX_MARGIN_DROP,
                "g3_max_state_bytes": gg.GATE_MAX_STATE_BYTES,
                "g3_min_slot_headroom": gg.GATE_MIN_SLOT_HEADROOM,
                "g4_max_value_residual": gg.GATE_MAX_VALUE_RESIDUAL,
                "direction": [
                    gg.GATE_DIRECTION_MIN_OVERLAP,
                    gg.GATE_DIRECTION_MAX_OVERLAP,
                    gg.GATE_DIRECTION_MIN_SEPARATION,
                ],
            },
        },
        "baseline_readings": baseline,
        "criteria": CRITERIA,
        "retention_caveat": (
            "the 7/7 battery is exact stored keys, so an append can never lower it (a key's "
            "cosine to its own slot is 1.0). It is reported as the registered convention; the "
            "paraphrase retention is a labelled diagnostic and is not used by the verdict"
        ),
        "discipline": {
            "training_updates": 0,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "confirm_reads": 0,
            "black_invoked": False,
            "cpu_cap_seconds": CPU_CAP,
            "watchdog_seconds": WATCHDOG_SECONDS,
            "evictions": 0,
            "dropped_writes": 0,
        },
    }
    write_json(MANIFEST, manifest)
    manifest_sha = file_sha(MANIFEST)
    append_progress({"phase": "manifest_written", "manifest_sha256": manifest_sha})

    ceiling = run_ceiling_arm(faces, initial, queries)
    ceiling_summary = arm_summary("U_no_gate", ceiling, faces, queries, paraphrases)
    append_progress(
        {
            "phase": "ceiling_done",
            "retention_correct": ceiling_summary["retention_correct"],
            "corrections_correct": ceiling_summary["corrections_correct"],
            "rejected_writes": ceiling_summary["rejected_writes"],
            "occupied_final": ceiling_summary["occupied_final"],
        }
    )

    simplified = run_simplified_arm(faces, initial, queries)
    simplified_summary = arm_summary("S_capacity_copy", simplified, faces, queries, paraphrases)
    append_progress(
        {
            "phase": "simplified_done",
            "retention_correct": simplified_summary["retention_correct"],
            "corrections_correct": simplified_summary["corrections_correct"],
            "rejected_writes": simplified_summary["rejected_writes"],
            "occupied_final": simplified_summary["occupied_final"],
            "slots_final": simplified_summary["slots_final"],
            "grown_slots_total": int(sum(e["grown_slots"] for e in simplified["growth_events"])),
        }
    )

    ceiling_faces = {row["tag"]: row for row in ceiling_summary["faces"]}
    losses = []
    for row in simplified_summary["faces"]:
        ceiling_row = ceiling_faces[row["tag"]]
        if ceiling_row["correct"] and not row["correct"]:
            losses.append(
                {
                    "tag": row["tag"],
                    "simplified_outcome": row["outcome"],
                    "read_top_slot": row["read_top_slot"],
                    "written_slot": row["written_slot"],
                }
            )
        elif not ceiling_row["correct"] and row["correct"]:
            losses.append(
                {
                    "tag": row["tag"],
                    "simplified_outcome": "gained_over_ceiling",
                    "read_top_slot": row["read_top_slot"],
                    "written_slot": row["written_slot"],
                }
            )

    corrections_s = int(simplified_summary["corrections_correct"])
    corrections_u = int(ceiling_summary["corrections_correct"])
    retention_s = int(simplified_summary["retention_correct"])
    retention_u = int(ceiling_summary["retention_correct"])
    evictions = int(simplified_summary["evictions_total"] + ceiling_summary["evictions_total"])
    dropped = int(
        simplified_summary["dropped_writes_total"] + ceiling_summary["dropped_writes_total"]
    )
    grown_total = int(sum(event["grown_slots"] for event in simplified["growth_events"]))
    refused_total = int(simplified_summary["rejected_writes"])

    if evictions > 0 or dropped > 0:
        verdict = "simplified_failed"
    elif corrections_s == FACE_COUNT and retention_s == len(queries):
        verdict = "simplified_holds_at_parity"
    else:
        verdict = "simplified_below_parity"

    margins_after_growth = [
        round(spec_margin(simplified["state"]), 6),
        *[event["margin_before_growth"] for event in simplified["growth_events"]],
    ]
    min_margin_after_growth = min(margins_after_growth)

    cost_cpu = time.process_time() - START_CPU
    cost_wall = time.time() - START_WALL
    if cost_cpu > CPU_CAP:
        raise SystemExit(f"cpu cap exceeded: {cost_cpu:.3f} > {CPU_CAP}")

    stop.set()

    report = {
        "probe": "taiji_gate_simplified",
        "date": "2026-11-08",
        "question": manifest["question"],
        "verdict": verdict,
        "verdict_detail": {
            "rule": CRITERIA["judge_rule"],
            "simplified": {
                "corrections_correct": corrections_s,
                "corrections_total": FACE_COUNT,
                "retention_correct": retention_s,
                "retention_total": len(queries),
                "rejected_writes": refused_total,
                "grown_slots": grown_total,
            },
            "ceiling_no_gate": {
                "corrections_correct": corrections_u,
                "corrections_total": FACE_COUNT,
                "retention_correct": retention_u,
                "retention_total": len(queries),
            },
            "delta_corrections_vs_ceiling": corrections_s - corrections_u,
            "delta_retention_vs_ceiling": retention_s - retention_u,
            "loss_faces": losses,
            "evictions": evictions,
            "dropped_writes": dropped,
            "collapse_line": {
                "value": COLLAPSE_LINE,
                "min_margin_after_growth": min_margin_after_growth,
                "below_collapse_line": bool(min_margin_after_growth < COLLAPSE_LINE),
                "used_by_verdict": False,
            },
        },
        "three_required_readings": {
            "1_ceiling_no_gate": {
                "corrections_correct": corrections_u,
                "corrections_total": FACE_COUNT,
                "retention_correct": retention_u,
                "retention_total": len(queries),
            },
            "2_simplified_loss_vs_ceiling": {
                "count": len(losses),
                "faces": losses,
            },
            "3_growth_and_refusals": {
                "grown_slots_total": grown_total,
                "growth_events": simplified["growth_events"],
                "refused_total": refused_total,
                "refused_tags": [row["tag"] for row in simplified["rows"] if not row["accepted"]],
            },
        },
        "preregistration": {
            "manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "manifest_sha256": manifest_sha,
            "written_before_any_arm_reading": True,
            "criteria": CRITERIA,
            "mapping": manifest["mapping"],
            "growth_protocol": manifest["growth_protocol"],
            "retained_clauses": list(RETAINED_CLAUSES),
            "dropped_clauses": list(DROPPED_CLAUSES),
            "retention_caveat": manifest["retention_caveat"],
        },
        "fixture": {
            "source": str(F1_RUNNER.relative_to(ROOT)).replace("\\", "/"),
            "encoder": "localized_centered (legacy 4-parent mean centroid)",
            "dim": 128,
            "value_shape": list(f1.VALUE_SHAPE),
            "temperature": gg.GATE_TEMPERATURE,
            "initial_occupancy": BASE_OCCUPIED,
            "initial_slots": SLOTS,
            "face_count": FACE_COUNT,
            "baseline": baseline,
        },
        "arms": {"U_no_gate": ceiling_summary, "S_capacity_copy": simplified_summary},
        "refusals": {
            "count": refused_total,
            "by_exit_code": {
                code: sum(
                    1
                    for row in simplified["rows"]
                    if not row["accepted"] and row["exit_code"] == code
                )
                for code in gg.EXIT_CODES
            },
            "quarantine_rows": int(simplified["quarantine_rows"]),
            "all_bit_identical": all(
                row["bit_identical"] for row in simplified["rows"] if not row["accepted"]
            ),
        },
        "retention_diagnostic_paraphrase": {
            "simplified": {
                "correct": simplified_summary["paraphrase_retention_correct"],
                "total": simplified_summary["paraphrase_retention_total"],
            },
            "ceiling": {
                "correct": ceiling_summary["paraphrase_retention_correct"],
                "total": ceiling_summary["paraphrase_retention_total"],
            },
            "used_by_verdict": False,
        },
        "discipline": {
            "training_updates": 0,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "confirm_reads": 0,
            "checkpoints_written": 0,
            "black_invoked": False,
            "black_gate": "in-process only",
            "evictions": evictions,
            "dropped_writes": dropped,
            "store_full_refusals": refused_total,
        },
        "isolation": {
            "runner_sha256_start": RUNNER_SHA_START,
            "runner_sha256_end": file_sha(Path(__file__).resolve()),
            "runner_sha_stable": file_sha(Path(__file__).resolve()) == RUNNER_SHA_START,
            "gate_sha256": file_sha(GATE_FILE),
            "runtime_sha256": file_sha(RUNTIME_FILE),
            "f1_runner_sha256": file_sha(F1_RUNNER),
            "manifest_sha256": manifest_sha,
            "process_guard": process_guard,
            "single_instance_lock": lock_name,
            "watchdog": {
                "seconds": WATCHDOG_SECONDS,
                "fired": bool(WATCHDOG_FIRED[0]),
                "marker_exists": WATCHDOG_MARKER.exists(),
            },
            "write_then_rename": True,
        },
        "cost": {
            "cpu_seconds": round(cost_cpu, 3),
            "wall_seconds": round(cost_wall, 3),
            "cpu_cap_seconds": CPU_CAP,
        },
        "prior_run": {
            "report": "reports/taiji_gate_task_value_20261108.json",
            "verdict": "gate_hurts",
            "recalled": (
                "the four-clause gate refused face_07 and face_08 on separability, kept the "
                "retention set at 7/7 and lost two correction faces; this run takes only the "
                "two clauses that survived that verdict"
            ),
        },
    }
    write_json(REPORT, report)
    release_single_instance()

    print(
        json.dumps(
            {
                "verdict": verdict,
                "S": {
                    "retention": retention_s,
                    "corrections": corrections_s,
                    "refused": refused_total,
                    "grown_slots": grown_total,
                },
                "U": {"retention": retention_u, "corrections": corrections_u},
                "loss_faces": [row["tag"] for row in losses],
                "collapse_line": min_margin_after_growth,
                "cpu_seconds": round(cost_cpu, 3),
                "wall_seconds": round(cost_wall, 3),
            },
            ensure_ascii=False,
        )
    )
    return 0


RUNNER_SHA_START = file_sha(Path(__file__).resolve())

if __name__ == "__main__":
    raise SystemExit(main())
