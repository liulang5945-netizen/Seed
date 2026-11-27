"""Generalization probe: did the correction transfer, or was it just a lookup?

The simplified gate run (``probe_taiji_gate_simplified_20261108``) reached 22/22
correction faces with zero refusals, but every one of those faces was read with a
phrasing the probe had itself chosen, so the reading cannot separate "the store
holds the correction" from "the store returns the nearest key it already holds".
This run asks the separation question directly, on the same fixture, the same
encoder and the same simplified gate (capacity + copy only, growth before a
capacity refusal):

    written group     the 22 faces, each still read with its registered synonym.
                      These are the faces whose rule was written into a slot.
    unwritten group   for every face, three phrasings of the same content that
                      were never written and never used as the registered read:
                      the same value token at positions outside
                      {write position 5, registered read position, the fixture's
                      own canonical position token % 8}.  Each is constructed
                      only on the scoring side, from ``paraphrase(base, token,
                      position)``; no rule truth and no solver enters the answer
                      chain, and the expected slot is the store's own write
                      bookkeeping, not a fixture truth.

Second task, in the same run: rewrite an already occupied face with the same
value at a phrasing that was never written, and check that the simplified gate
still refuses it with ``commit_rejected_no_direction``, rolls back bit for bit
and leaves all 22 registered face readings unchanged.

Registered before the first reading (also in the manifest):

    any rule with < 2 constructed phrasings          -> not_assessable
    unwritten rate >= 0.8                            -> rule_generalized
    unwritten rate <= 0.2 and written rate >= 0.9    -> memorized_lookup_only
    unwritten rate <= 0.2 and written rate <  0.9    -> inconclusive_both_low
    otherwise                                        -> partially_generalized

The bands are written with their tie-break: exactly 0.8 belongs to the upper
branch, exactly 0.2 to the lower.  A high unwritten rate is reported next to the
nearest-neighbour similarity distribution, because a phrasing that lands on the
same key is a lookup however the count is labelled.

Zero training, zero optimizer steps, zero Adam, zero confirm reads, zero model
forward.  numpy only.
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

REPORT = ROOT / "reports/taiji_gate_generalization_20261108.json"
MANIFEST = ROOT / "plans/manifests/taiji_gate_generalization_20261108.json"
PROGRESS = ROOT / "reports/taiji_gate_generalization_20261108_progress.jsonl"
JOURNAL_DIR = ROOT / "reports/taiji_gate_generalization_20261108_journal"
LOCK = ROOT / "reports/taiji_gate_generalization_20261108.lock"
WATCHDOG_MARKER = ROOT / "reports/taiji_gate_generalization_20261108_watchdog.json"
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
UNWRITTEN_PER_RULE = 3
COPY_FACE_TAG = "face_00"
COLLAPSE_LINE = 0.02

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
    "P1_written_group": (
        "the 22 faces are written by the simplified gate and read with their registered "
        "synonym; a face is correct iff the final store's top-1 slot for that read is the "
        "slot the face was written to; report the integer count out of 22"
    ),
    "P2_unwritten_group": (
        "for every face, three phrasings of the same content at positions that are neither "
        "the write position (5), the registered read position, nor the fixture's canonical "
        "position (token % 8); none of these keys is written or read anywhere else. A "
        "phrasing is correct iff the final store's top-1 slot is the slot of its own face"
    ),
    "P3_construction": (
        "the unwritten phrasings are built only from fixture arrays (base, the face's value "
        "token, a position) on the scoring side; the expected slot is the store write "
        "bookkeeping of that face, never a rule truth or solver output. If any face yields "
        "fewer than two phrasings the run is not assessable, not a pass"
    ),
    "P4_similarity": (
        "for every unwritten key, report its cosine to the nearest occupied key and to its "
        "own face key; report min / p25 / median / p75 / max of the nearest-key cosine and "
        "the count whose nearest key is the own face key"
    ),
    "P5_copy_refusal": (
        "after the 22 writes, rewrite one already occupied face with the same value at a "
        "never-written phrasing; the simplified gate must refuse it with "
        "commit_rejected_no_direction, the attempt must evict and drop nothing"
    ),
    "P6_copy_isolation": (
        "the refused copy write must leave the state bit-identical (sha256 before == after) "
        "and every one of the 22 registered face readings unchanged face by face"
    ),
    "verdict_rule": (
        "any rule with < 2 constructed phrasings -> not_assessable; unwritten >= 0.8 -> "
        "rule_generalized (0.8 belongs here); unwritten <= 0.2 with written >= 0.9 -> "
        "memorized_lookup_only (0.2 belongs here); unwritten <= 0.2 with written < 0.9 -> "
        "inconclusive_both_low; otherwise -> partially_generalized"
    ),
    "copy_verdict_rule": (
        "refused with no_direction + bit-identical + 0 evictions + 0 dropped + all face "
        "readings unchanged -> copy_refused_no_direction_intact; refused by another retained "
        "clause -> copy_refused_other_clause; admitted -> copy_admitted"
    ),
    "no_soft_language": (
        "an unwritten rate that is high because every phrasing lands on the face's own key "
        "may not be described as learning a rule; the similarity distribution is reported "
        "next to the count for exactly that reason"
    ),
    "unchanged": "no threshold, fixture, face, position or battery is retuned after the first reading",
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


def unwritten_positions(face: dict[str, Any]) -> list[int]:
    """Positions that are neither the write, the registered read, nor the fixture's own."""
    used = {
        CORRECTION_POSITION,
        int(face["read_position"]),
        int(face["token"]) % POSITION_COUNT,
    }
    candidates = [position for position in range(POSITION_COUNT) if position not in used]
    return candidates[:UNWRITTEN_PER_RULE]


def unwritten_queries(faces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Same content, new word order; constructed on the scoring side only."""
    rows = []
    for face in faces:
        for rank, position in enumerate(unwritten_positions(face)):
            context = f1.paraphrase(base0, tokens0[int(face["token"])], position)
            rows.append(
                {
                    "tag": face["tag"],
                    "token": int(face["token"]),
                    "position": int(position),
                    "rank": rank,
                    "key": gr.encode_read_key(context, encoder, centroid),
                }
            )
    return rows


def construction_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["tag"]] = counts.get(row["tag"], 0) + 1
    short = {tag: count for tag, count in counts.items() if count < 2}
    return {
        "per_rule": counts,
        "rules_with_fewer_than_two": short,
        "sufficient": not short,
        "positions_per_rule": UNWRITTEN_PER_RULE,
    }


def similarity_rows(
    state: gg.GateState,
    rows: list[dict[str, Any]],
    slots: dict[str, int],
) -> list[dict[str, Any]]:
    keys, _ = active(state)
    normalized = keys / np.clip(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12, None)
    out = []
    for row in rows:
        unit = row["key"] / max(float(np.linalg.norm(row["key"])), 1e-12)
        cosine = normalized @ unit
        order = np.argsort(-cosine)
        own = slots.get(row["tag"])
        out.append(
            {
                "tag": row["tag"],
                "position": row["position"],
                "nearest_slot": int(order[0]),
                "nearest_cos": round(float(cosine[order[0]]), 6),
                "second_cos": round(float(cosine[order[1]]), 6),
                "own_slot": None if own is None else int(own),
                "own_cos": None if own is None else round(float(cosine[int(own)]), 6),
                "nearest_is_own": bool(own is not None and int(order[0]) == int(own)),
            }
        )
    return out


def describe(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": round(float(arr.min()), 6),
        "p25": round(float(np.percentile(arr, 25)), 6),
        "median": round(float(np.median(arr)), 6),
        "p75": round(float(np.percentile(arr, 75)), 6),
        "max": round(float(arr.max()), 6),
    }


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


def unwritten_outcome(state: gg.GateState, row: dict[str, Any], slot: int | None) -> dict[str, Any]:
    """Score one never-written phrasing against the store's own write bookkeeping."""
    if slot is None:
        return {
            "tag": row["tag"],
            "token": row["token"],
            "position": row["position"],
            "rank": row["rank"],
            "expected_slot": None,
            "read_top_slot": None,
            "correct": False,
        }
    observed = top_slot(state, row["key"])
    return {
        "tag": row["tag"],
        "token": row["token"],
        "position": row["position"],
        "rank": row["rank"],
        "expected_slot": int(slot),
        "read_top_slot": int(observed),
        "correct": bool(observed == int(slot)),
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
                    "evictions": 0,
                    "dropped": 0,
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
            "dropped_clause_failures": [code for code in cert.failed if code in DROPPED_CLAUSES],
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
            "evictions": int(applied["evictions"]),
            "dropped": int(applied["dropped"]),
            "bit_identical": (None if accepted else gg.snapshot(state).sha256 == sha_before),
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


def copy_attempt(
    state: gg.GateState,
    face: dict[str, Any],
    position: int,
    queries: list[gg.BatteryQuery],
) -> dict[str, Any]:
    """Rewrite an occupied face with the same value at a never-written phrasing."""
    context = f1.paraphrase(base0, tokens0[int(face["token"])], position)
    write = gg.WriteCell(
        key=gr.encode_read_key(context, encoder, centroid),
        value=face["value"],
        family="copy_check",
        policy="append",
    )
    occupancy_before = int(state.occupied)
    slots_before = int(state.slots)
    sha_before = gg.snapshot(state).sha256
    headroom_before = gr.write_headroom(occupancy_before, slots_before, 1)
    grew = 0
    headroom_after_growth = headroom_before
    if not headroom_before["sufficient"]:
        plan = gr.plan_growth(headroom_before)
        if plan["action"] == "grow_then_write":
            state = grow_state(state, int(plan["slots"]))
            grew = int(state.slots) - slots_before
            headroom_after_growth = gr.write_headroom(int(state.occupied), int(state.slots), 1)
    shadow, applied = gg.shadow_apply(state, write)
    cert = gg.evaluate(state, shadow, queries, write, applied["slot"])
    blocking = [code for code in cert.failed if code in RETAINED_CLAUSES]
    admitted = bool(applied["applied"]) and not blocking
    after = shadow if admitted else gg.restore(gg.snapshot(state))
    sha_after = gg.snapshot(after).sha256
    return {
        "state": after,
        "face": face["tag"],
        "token": int(face["token"]),
        "copy_position": int(position),
        "same_value": True,
        "write_policy": "append",
        "occupancy_before": occupancy_before,
        "slots_before": slots_before,
        "slots_after_attempt": int(state.slots),
        "grown_slots": int(grew),
        "headroom_before": round(float(headroom_before["h_slots"]), 6),
        "headroom_after_growth": round(float(headroom_after_growth["h_slots"]), 6),
        "shadow_slot": None if applied["slot"] is None else int(applied["slot"]),
        "applied_to_shadow": bool(applied["applied"]),
        "admitted": bool(admitted),
        "exit_code": cert.exit_code,
        "failed": list(cert.failed),
        "blocking": blocking,
        "direction_overlap": round(float(cert.direction_overlap), 6),
        "direction_separation": round(float(cert.direction_separation), 6),
        "direction_verdict": cert.direction_verdict,
        "value_residual_on_shadow": round(float(cert.g4_value_residual), 6),
        "evictions": int(applied["evictions"]),
        "dropped": int(applied["dropped"]),
        "sha_before": sha_before,
        "sha_after": sha_after,
        "bit_identical": bool(sha_after == sha_before),
    }


def compare_readings(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> dict[str, Any]:
    changed = []
    for old, new in zip(before, after, strict=True):
        if old["read_top_slot"] != new["read_top_slot"]:
            changed.append(
                {
                    "tag": old["tag"],
                    "field": "read_top_slot",
                    "before": old["read_top_slot"],
                    "after": new["read_top_slot"],
                }
            )
        elif old["value_residual"] != new["value_residual"]:
            changed.append(
                {
                    "tag": old["tag"],
                    "field": "value_residual",
                    "before": old["value_residual"],
                    "after": new["value_residual"],
                }
            )
    return {"changed": changed, "unchanged_all": not changed, "compared": len(before)}


def main() -> int:
    if time.process_time() - START_CPU > CPU_GUARD:
        raise SystemExit("cpu guard tripped before start")

    process_guard = assert_no_black_processes("gate generalization probe")
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
    by_tag = {face["tag"]: face for face in faces}
    unwritten = unwritten_queries(faces)
    construction = construction_report(unwritten)
    copy_face = by_tag[COPY_FACE_TAG]
    copy_position = unwritten_positions(copy_face)[0]

    baseline = {
        "occupied": int(initial.occupied),
        "slots": int(initial.slots),
        "headroom": round((SLOTS - BASE_OCCUPIED) / SLOTS, 6),
        "spec_margin": round(spec_margin(initial), 6),
        "retention_correct": retention_count(initial, queries),
        "retention_total": len(queries),
        "unwritten_queries": len(unwritten),
        "unwritten_rules": len(construction["per_rule"]),
        "copy_face": COPY_FACE_TAG,
        "copy_position": int(copy_position),
    }
    append_progress({"phase": "baseline", **baseline})

    manifest = {
        "format": "taiji-gate-generalization-preregistration-v1",
        "date": "2026-11-08",
        "question": (
            "on the same fixture and simplified gate, does the 22/22 correction reading "
            "survive on phrasings that were never written, or is it memorised lookup?"
        ),
        "written_before_any_arm_reading": True,
        "written_before_any_case_verdict": True,
        "sequence": [
            "phase 0 baseline geometry (read only, no verdict)",
            "this manifest is written and hashed",
            "simplified gate writes the 22 faces (capacity + copy, growth before refusal)",
            "written group: the 22 registered synonym reads",
            "unwritten group: 3 never-written phrasings per face, built on the scoring side",
            "copy check: same value at a never-written phrasing of an occupied face",
            "verdict from the registered rules",
        ],
        "mapping": {
            "task_shape": (
                "package 20: 22 correction surfaces per world on a content-addressed store "
                "that starts at 32 slots and may grow"
            ),
            "store": (
                "existing F1/F3 fixture unchanged: same encoder (localized_centered, legacy "
                "centroid), 128-d keys, 8192-d values, ADDRESSED_TEMPERATURE 0.1"
            ),
            "why_not_the_model": (
                "the question is about what the store can answer, not about a torch forward; "
                "the probe measures the store where it acts and runs no model step"
            ),
            "initial_occupancy": BASE_OCCUPIED,
            "initial_slots": SLOTS,
            "face_count": FACE_COUNT,
            "face_definition": (
                "face i writes a correction whose value token is the i-th new token at "
                "position 5, and is read with the same token at a registered other position"
            ),
            "unwritten_definition": (
                "for face i, the same value token at positions outside {5, registered read "
                "position, token % 8}; the first three such positions, sorted, are used"
            ),
            "expected_answer": (
                "the slot the face was written to, taken from the store write bookkeeping; "
                "no rule truth, solver or fixture label enters the answer chain"
            ),
            "no_eviction": "append only; a full store is refused, never evicted",
        },
        "counts": {
            "written_queries": FACE_COUNT,
            "unwritten_queries": len(unwritten),
            "unwritten_per_rule": UNWRITTEN_PER_RULE,
            "copy_attempts": 1,
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
        "copy_check": {
            "face": COPY_FACE_TAG,
            "position": int(copy_position),
            "value": "the same value the face was written with",
            "expected": "commit_rejected_no_direction, bit-identical rollback, 0 evictions",
        },
        "constants": {
            "base_occupied": BASE_OCCUPIED,
            "initial_slots": SLOTS,
            "face_count": FACE_COUNT,
            "correction_position": CORRECTION_POSITION,
            "synonym_positions": list(SYNONYM_POSITIONS),
            "unwritten_per_rule": UNWRITTEN_PER_RULE,
            "copy_face_tag": COPY_FACE_TAG,
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

    arm = run_simplified_arm(faces, initial, queries)
    state = arm["state"]
    slots = arm["slots"]
    append_progress(
        {
            "phase": "writes_done",
            "accepted": len(slots),
            "refused": sum(1 for row in arm["rows"] if not row["accepted"]),
            "occupied_final": int(state.occupied),
            "slots_final": int(state.slots),
        }
    )

    written_readings = [face_outcome(state, face, slots.get(face["tag"])) for face in faces]
    written_correct = sum(1 for row in written_readings if row["correct"])
    written_rate = written_correct / FACE_COUNT
    append_progress(
        {
            "phase": "written_group",
            "correct": written_correct,
            "total": FACE_COUNT,
            "rate": round(written_rate, 6),
            "outcomes": {
                kind: sum(1 for row in written_readings if row["outcome"] == kind)
                for kind in {row["outcome"] for row in written_readings}
            },
        }
    )

    unwritten_readings = [unwritten_outcome(state, row, slots.get(row["tag"])) for row in unwritten]
    unwritten_correct = sum(1 for row in unwritten_readings if row["correct"])
    unwritten_rate = unwritten_correct / max(len(unwritten_readings), 1)
    per_rule = {
        tag: {
            "correct": sum(1 for row in unwritten_readings if row["tag"] == tag and row["correct"]),
            "total": sum(1 for row in unwritten_readings if row["tag"] == tag),
        }
        for tag in construction["per_rule"]
    }
    append_progress(
        {
            "phase": "unwritten_group",
            "correct": unwritten_correct,
            "total": len(unwritten_readings),
            "rate": round(unwritten_rate, 6),
            "rules": len(per_rule),
        }
    )

    similarity = similarity_rows(state, unwritten, slots)
    nearest_cosines = [row["nearest_cos"] for row in similarity]
    own_cosines = [row["own_cos"] for row in similarity if row["own_cos"] is not None]
    similarity_summary = {
        "nearest_key_cosine": describe(nearest_cosines),
        "own_face_key_cosine": describe(own_cosines),
        "nearest_is_own_count": sum(1 for row in similarity if row["nearest_is_own"]),
        "nearest_is_own_total": len(similarity),
        "used_by_verdict": False,
    }
    append_progress(
        {
            "phase": "similarity",
            "nearest_median": similarity_summary["nearest_key_cosine"].get("median"),
            "near_as_own": similarity_summary["nearest_is_own_count"],
        }
    )

    before_copy = [face_outcome(state, face, slots.get(face["tag"])) for face in faces]
    copy_result = copy_attempt(state, copy_face, copy_position, queries)
    after_copy = [
        face_outcome(copy_result["state"], face, slots.get(face["tag"])) for face in faces
    ]
    isolation = compare_readings(before_copy, after_copy)
    no_direction_hit = gg.EXIT_NO_DIRECTION in copy_result["blocking"]
    if copy_result["admitted"]:
        copy_verdict = "copy_admitted"
    elif no_direction_hit:
        copy_verdict = "copy_refused_no_direction_intact"
    else:
        copy_verdict = "copy_refused_other_clause"
    if copy_verdict == "copy_refused_no_direction_intact" and not (
        copy_result["bit_identical"] and isolation["unchanged_all"]
    ):
        copy_verdict = "copy_refused_but_isolation_failed"
    append_progress(
        {
            "phase": "copy_check",
            "verdict": copy_verdict,
            "direction_verdict": copy_result["direction_verdict"],
            "bit_identical": copy_result["bit_identical"],
            "unchanged_all": isolation["unchanged_all"],
            "evictions": copy_result["evictions"],
            "dropped": copy_result["dropped"],
        }
    )

    if not construction["sufficient"]:
        verdict = "not_assessable_construction_short"
    elif unwritten_rate >= 0.8:
        verdict = "rule_generalized"
    elif unwritten_rate <= 0.2:
        verdict = "memorized_lookup_only" if written_rate >= 0.9 else "inconclusive_both_low"
    else:
        verdict = "partially_generalized"

    evictions = int(sum(row["evictions"] for row in arm["rows"])) + int(copy_result["evictions"])
    dropped = int(sum(row["dropped"] for row in arm["rows"])) + int(copy_result["dropped"])
    cost_cpu = time.process_time() - START_CPU
    cost_wall = time.time() - START_WALL
    if cost_cpu > CPU_CAP:
        raise SystemExit(f"cpu cap exceeded: {cost_cpu:.3f} > {CPU_CAP}")

    stop.set()

    report = {
        "probe": "taiji_gate_generalization",
        "date": "2026-11-08",
        "question": manifest["question"],
        "verdict": verdict,
        "verdict_detail": {
            "rule": CRITERIA["verdict_rule"],
            "written_group": {
                "correct": written_correct,
                "total": FACE_COUNT,
                "rate": round(written_rate, 6),
                "definition": CRITERIA["P1_written_group"],
            },
            "unwritten_group": {
                "correct": unwritten_correct,
                "total": len(unwritten_readings),
                "rate": round(unwritten_rate, 6),
                "definition": CRITERIA["P2_unwritten_group"],
                "per_rule": per_rule,
            },
            "construction": construction,
            "similarity": similarity_summary,
            "copy_check": {
                "verdict": copy_verdict,
                "rule": CRITERIA["copy_verdict_rule"],
                "no_direction_hit": bool(no_direction_hit),
                "isolation": isolation,
            },
            "evictions": evictions,
            "dropped_writes": dropped,
        },
        "written_group": {
            "faces": written_readings,
            "self_read_diagnostic": {
                "correct": sum(
                    1 for face in faces if top_slot(state, face["key"]) == slots.get(face["tag"])
                ),
                "total": FACE_COUNT,
            },
        },
        "unwritten_group": {
            "rows": unwritten_readings,
            "per_rule": per_rule,
        },
        "similarity": {
            "summary": similarity_summary,
            "rows": similarity,
        },
        "copy_check": {key: value for key, value in copy_result.items() if key != "state"},
        "preregistration": {
            "manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "manifest_sha256": manifest_sha,
            "written_before_any_arm_reading": True,
            "criteria": CRITERIA,
            "mapping": manifest["mapping"],
            "counts": manifest["counts"],
            "growth_protocol": manifest["growth_protocol"],
            "retained_clauses": list(RETAINED_CLAUSES),
            "dropped_clauses": list(DROPPED_CLAUSES),
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
        "arm": {
            "name": "S_capacity_copy",
            "accepted": len(slots),
            "refused": sum(1 for row in arm["rows"] if not row["accepted"]),
            "occupied_final": int(state.occupied),
            "slots_final": int(state.slots),
            "headroom_final": round(
                (int(state.slots) - int(state.occupied)) / max(int(state.slots), 1), 6
            ),
            "spec_margin_final": round(spec_margin(state), 6),
            "retention_correct": retention_count(state, queries),
            "retention_total": len(queries),
            "grown_slots": int(sum(event["grown_slots"] for event in arm["growth_events"])),
            "growth_events": arm["growth_events"],
            "writes": arm["rows"],
        },
        "collapse_line": {
            "value": COLLAPSE_LINE,
            "min_margin_after_growth": round(spec_margin(state), 6),
            "below_collapse_line": bool(spec_margin(state) < COLLAPSE_LINE),
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
            "store_full_refusals": int(sum(1 for row in arm["rows"] if not row["accepted"])),
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
            "report": "reports/taiji_gate_simplified_20261108.json",
            "verdict": "simplified_holds_at_parity",
            "recalled": (
                "22/22 with 0 refusals, but every read used a phrasing the probe chose; this "
                "run asks whether that count survives never-written phrasings"
            ),
        },
        "implementation_corrections": [
            {
                "what": (
                    "the first execution raised TypeError while serialising the report: the copy "
                    "attempt dict carried its GateState"
                ),
                "fix": (
                    "the state object is excluded from the published dict; every reading had "
                    "already been taken and written to progress before serialisation"
                ),
                "effect_on_readings": "none",
            }
        ],
        "what_this_does_not_mean": [
            "no training, no optimizer step and no model forward ran; nothing here says the "
            "correction survives training",
            "the unwritten phrasings are the same value token at new positions; a high count "
            "means the correction transfers to new word orders, which is neighbour retrieval "
            "of one content token, not abstraction of a symbolic rule",
            "the fixture is synthetic and the expected slot is store bookkeeping; no "
            "language-level correctness is claimed",
            "one session, one seed, no persistence across checkpoints",
        ],
    }
    write_json(REPORT, report)
    release_single_instance()

    print(
        json.dumps(
            {
                "verdict": verdict,
                "written": {"correct": written_correct, "total": FACE_COUNT},
                "unwritten": {
                    "correct": unwritten_correct,
                    "total": len(unwritten_readings),
                },
                "nearest_cos_median": similarity_summary["nearest_key_cosine"].get("median"),
                "copy": {
                    "verdict": copy_verdict,
                    "direction": copy_result["direction_verdict"],
                    "bit_identical": copy_result["bit_identical"],
                },
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
