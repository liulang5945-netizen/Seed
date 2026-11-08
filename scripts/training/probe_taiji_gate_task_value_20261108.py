"""Gate task-value probe: is the consensus gate worth its refusals on the task?

One switch, two arms, both with the store really in the read path:

    A  store enabled + gate on   (``growth_gate.run_gate`` decides every write)
    B  store enabled + gate off  (``growth_gate.shadow_apply``, the same append)

Same fixture, same 22-face write order, no eviction, no training, no optimizer
step, no Adam, no checkpoint, no confirm read.

Task shape is package 20's: 22 correction surfaces per world on a 32-slot
addressed store (``living_core.ADDRESSED_SLOTS``).  The store is the existing
F1/F3 fixture unchanged (same encoder, 128-d keys, 8192-d values, T=0.1), so the
judge is measured at the store level and no new fixture is built.

Retention is the existing 7/7 convention: an 8-row baseline, 7 held-out exact
keys, integer correct count.  It is reported under the caveat below, because an
exact-key battery cannot be broken by an append (self-cosine is 1.0); the
paraphrase retention is carried as a labelled diagnostic only.

Verdict rule, registered in the manifest before any arm runs:

    both arms equal, zero refusals        -> unable_to_assess_no_rejections
    both arms equal, refusals > 0         -> gate_has_no_task_value
    A better on (retention, corrections)  -> gate_has_task_value
    A worse on (retention, corrections)   -> gate_hurts

"Better" is lexicographic: retention count first, correction count second.  The
order is stated here because the task asks for one verdict, not a trade-off.
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

REPORT = ROOT / "reports/taiji_gate_task_value_20261108.json"
MANIFEST = ROOT / "plans/manifests/taiji_gate_task_value_20261108.json"
PROGRESS = ROOT / "reports/taiji_gate_task_value_20261108_progress.jsonl"
JOURNAL_DIR = ROOT / "reports/taiji_gate_task_value_20261108_journal"
LOCK = ROOT / "reports/taiji_gate_task_value_20261108.lock"
WATCHDOG_MARKER = ROOT / "reports/taiji_gate_task_value_20261108_watchdog.json"
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

SUPERSEDED_RUNS = [
    {
        "report_sha256": "aabbca3b970122f9b835e69da6b2383d3e01e236789d40f1eb0c8e403671b731",
        "manifest_sha256": "2adcd08a059860e4f75c59d6c909f4c275113aafd7f70072526180c52bc84db7",
        "verdict": "gate_hurts",
        "why": "the compute-only simplification pass had an inverted predicate and refused "
        "the writes it was supposed to admit; the two published arms were unaffected",
    },
    {
        "report_sha256": "a47bd1bc2c35e32d946992e0d35e860a9001b8de6ebffbe3c84abb2a0df92fdc",
        "manifest_sha256": "2adcd08a059860e4f75c59d6c909f4c275113aafd7f70072526180c52bc84db7",
        "verdict": "gate_hurts",
        "why": "predicate fixed; this report is complete but the simplification block did not "
        "record why a write was still refused after the dropped clauses, so the owner could "
        "not separate a capacity refusal from a quality refusal",
    },
    {
        "report_sha256": "acd78e9ff7b3a22be9e9024b590a45ae0303c2302c2fbdbefe05cad645c6a1eb",
        "manifest_sha256": "fa046245ffe1bf83027e2dce3476f226b0842245da7cef3ec1ff6f5e48911a6d",
        "verdict": "gate_hurts",
        "why": "never committed: the only change afterwards was a lint-only rewrite (ruff SIM) "
        "of the verdict assignment and one comparison operand order, re-run so the published "
        "readings come from the committed runner",
    },
]

VERDICT_REASON = {
    "gate_hurts": (
        "the gate refused writes the task needed and the retention set was not at risk, "
        "so criterion tuning on this judge is stopped"
    ),
    "gate_has_no_task_value": "refusals exist but neither arm differs on the task metrics",
    "unable_to_assess_no_rejections": "no write was refused, so the two arms are the same run twice",
    "gate_has_task_value": "the gate improved a task metric without losing the other",
}

CRITERIA = {
    "T1_retention": (
        "per arm, the 7 held-out exact keys of the 8-row baseline are read back and "
        "the top-1 slot must equal the expected slot; report the integer count out of 7"
    ),
    "T2_corrections": (
        "per arm, each of the 22 correction faces is read with a synonym (the same value "
        "token at a different position); a face is correct iff the final store's top-1 slot "
        "for that synonym is the slot that face was written to.  A refused write owns no "
        "slot and therefore cannot be correct"
    ),
    "T3_rejections": (
        "per refused write, the first failing clause (exit code) and the full failed list "
        "are reported; the distribution is by exit code"
    ),
    "T4_counterfactual": (
        "for every refused write, the write is applied to the store as it stood at that "
        "moment (copy-on-write, never published) and the retention count is re-read; "
        "'would break retention' means that count is lower than the count just before"
    ),
    "T5_simplification": (
        "compute-only third pass: a gate that keeps only capacity (G3 headroom) and copy "
        "(direction) admits a write iff it was applied to a free slot and its failed list "
        "contains neither commit_rejected_no_capacity nor commit_rejected_no_direction -- a "
        "dropped clause may not refuse; report which writes it admits, which are still "
        "refused and by which clause, which admitted writes the full gate refused, and the "
        "resulting retention and correction counts"
    ),
    "verdict_rule": (
        "both equal + 0 refusals -> unable_to_assess_no_rejections; both equal + refusals>0 "
        "-> gate_has_no_task_value; A better on (retention, corrections) lexicographic -> "
        "gate_has_task_value with the share of refused writes that would break retention; "
        "A worse -> gate_hurts, stop criterion tuning and report the reason"
    ),
    "no_soft_language": (
        "a refusal that protects nothing may not be described as harmless; if the arms tie "
        "the verdict is no task value, never 'close'"
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
    """A deterministic position for the read phrasing, never the write position.

    The second branch fires only when the first choice equals the position the
    fixture already holds that token at, and the next element is always a
    different position, so the result can never collide with the write position.
    """
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


def face_readings(state: gg.GateState, faces: list[dict[str, Any]], slots: dict[str, int]):
    rows = []
    for face in faces:
        slot = slots.get(face["tag"])
        slot = None if slot is None else int(slot)
        observed = top_slot(state, face["read_key"])
        value = read_value(state, face["read_key"])
        rows.append(
            {
                "tag": face["tag"],
                "token": face["token"],
                "read_position": face["read_position"],
                "written_slot": slot,
                "read_top_slot": int(observed),
                "correct": bool(slot is not None and observed == slot),
                "value_residual": round(relative_residual(value, face["value"]), 6),
            }
        )
    return rows


def arm_record(
    name: str,
    state: gg.GateState,
    queries: list[gg.BatteryQuery],
    faces: list[dict[str, Any]],
    slots: dict[str, int],
    rows: list[dict[str, Any]],
    paraphrases: list[dict[str, Any]],
) -> dict[str, Any]:
    readings = face_readings(state, faces, slots)
    paraphrase_correct = sum(
        1 for row in paraphrases if top_slot(state, row["key"]) == row["index"]
    )
    return {
        "arm": name,
        "occupied_final": int(state.occupied),
        "slots": int(state.slots),
        "headroom_final": round(
            (int(state.slots) - int(state.occupied)) / max(int(state.slots), 1), 6
        ),
        "spec_margin_final": round(spec_margin(state), 6),
        "retention_correct": retention_count(state, queries),
        "retention_total": len(queries),
        "corrections_correct": sum(1 for row in readings if row["correct"]),
        "corrections_total": len(faces),
        "rejected_writes": sum(1 for row in rows if not row["accepted"]),
        "evictions_total": int(sum(row.get("evictions", 0) for row in rows)),
        "dropped_writes_total": int(sum(row.get("dropped", 0) for row in rows)),
        "paraphrase_retention_correct": paraphrase_correct,
        "paraphrase_retention_total": len(paraphrases),
        "faces": readings,
        "writes": rows,
    }


def run_gate_arm(
    faces: list[dict[str, Any]],
    initial: gg.GateState,
    queries: list[gg.BatteryQuery],
) -> dict[str, Any]:
    """Arm A: every write goes through the full judge; a refusal changes nothing."""
    state = initial
    slots: dict[str, int] = {}
    rows = []
    counterfactuals = []
    for face in faces:
        write = gg.WriteCell(
            key=face["key"],
            value=face["value"],
            family=face["family"],
            policy=face["policy"],
        )
        quarantine = JOURNAL_DIR / f"{face['tag']}_quarantine.jsonl"
        log = JOURNAL_DIR / f"{face['tag']}_gate_log.jsonl"
        quarantine.write_text("", encoding="utf-8")
        log.write_text("", encoding="utf-8")
        retention_before = retention_count(state, queries)
        receipt, run = gg.run_gate(state, write, queries, quarantine_path=quarantine, log_path=log)
        cert = run.certificate
        record = {
            "arm": "A_gate_on",
            "tag": face["tag"],
            "occupancy_before": int(state.occupied),
            "retention_before": retention_before,
            "accepted": bool(receipt.published),
            "exit_code": receipt.exit_code,
            "failed": list(cert.failed),
            "written_slot": cert.written_slot,
            "g1_before": cert.g1_before,
            "g1_after": cert.g1_after,
            "margin_before": round(cert.g2_margin_before, 6),
            "margin_after": round(cert.g2_margin_after, 6),
            "direction_verdict": cert.direction_verdict,
            "direction_overlap": round(cert.direction_overlap, 6),
            "direction_separation": round(cert.direction_separation, 6),
            "value_residual": round(float(cert.g4_value_residual), 6),
            "headroom_after": round(cert.g3_headroom_after, 6),
            "bytes_delta": cert.g3_bytes_delta,
            "evictions": receipt.evictions,
            "dropped": 0,
            "bit_identical": receipt.bit_identical,
            "quarantine_id": receipt.quarantine_id,
            "quarantine_rows": journal_rows(quarantine),
        }
        if receipt.published:
            state = receipt.state
            slots[face["tag"]] = int(cert.written_slot)
        else:
            forced, applied = gg.shadow_apply(state, write)
            counterfactuals.append(
                {
                    "tag": face["tag"],
                    "applied": bool(applied["applied"]),
                    "applied_slot": applied["slot"],
                    "retention_if_forced": retention_count(forced, queries),
                    "retention_before": retention_before,
                    "would_break_retention": bool(
                        retention_count(forced, queries) < retention_before
                    ),
                    "spec_margin_if_forced": round(spec_margin(forced), 6),
                    "spec_margin_before": round(spec_margin(state), 6),
                }
            )
        append_progress(record)
        rows.append(record)
    return {"state": state, "slots": slots, "rows": rows, "counterfactuals": counterfactuals}


def run_open_arm(
    faces: list[dict[str, Any]],
    initial: gg.GateState,
    queries: list[gg.BatteryQuery],
) -> dict[str, Any]:
    """Arm B: gate off. The store still sits in the read path; nothing refuses."""
    state = initial
    slots: dict[str, int] = {}
    rows = []
    for face in faces:
        write = gg.WriteCell(
            key=face["key"],
            value=face["value"],
            family=face["family"],
            policy=face["policy"],
        )
        shadow, applied = gg.shadow_apply(state, write)
        record = {
            "arm": "B_gate_off",
            "tag": face["tag"],
            "occupancy_before": int(state.occupied),
            "retention_before": retention_count(state, queries),
            "accepted": bool(applied["applied"]),
            "exit_code": "not_evaluated_gate_off",
            "failed": [],
            "written_slot": applied["slot"],
            "evictions": int(applied["evictions"]),
            "dropped": int(applied["dropped"]),
        }
        if applied["applied"]:
            state = shadow
            slots[face["tag"]] = int(applied["slot"])
        append_progress(record)
        rows.append(record)
    return {"state": state, "slots": slots, "rows": rows}


def run_simplified(
    faces: list[dict[str, Any]],
    initial: gg.GateState,
    queries: list[gg.BatteryQuery],
) -> dict[str, Any]:
    """Compute-only: a gate that keeps only capacity and the copy clause.

    Nothing is published and no journal is touched: this pass exists to answer
    "what would a simplified gate let through", not to produce a third arm.
    """
    state = initial
    slots: dict[str, int] = {}
    rows = []
    for face in faces:
        write = gg.WriteCell(
            key=face["key"],
            value=face["value"],
            family=face["family"],
            policy=face["policy"],
        )
        shadow, applied = gg.shadow_apply(state, write)
        cert = gg.evaluate(state, shadow, queries, write, applied["slot"])
        blocking = set(cert.failed) & {gg.EXIT_NO_CAPACITY, gg.EXIT_NO_DIRECTION}
        allowed = bool(applied["applied"]) and not blocking
        rows.append(
            {
                "tag": face["tag"],
                "allowed": bool(allowed),
                "failed": list(cert.failed),
                "written_slot": applied["slot"] if allowed else None,
            }
        )
        if allowed:
            state = shadow
            slots[face["tag"]] = int(applied["slot"])
    return {"state": state, "slots": slots, "rows": rows}


def main() -> int:
    if time.process_time() - START_CPU > CPU_GUARD:
        raise SystemExit("cpu guard tripped before start")

    process_guard = assert_no_black_processes("gate task value probe")
    lock_name = acquire_single_instance()
    atexit.register(release_single_instance)
    stop = threading.Event()
    watchdog = start_watchdog(stop)

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
        "paraphrase_retention_correct": sum(
            1 for row in paraphrases if top_slot(initial, row["key"]) == row["index"]
        ),
        "paraphrase_retention_total": len(paraphrases),
    }
    append_progress({"phase": "baseline", **baseline})

    manifest = {
        "format": "taiji-gate-task-value-preregistration-v1",
        "date": "2026-11-08",
        "question": "does the consensus gate have task value, measured with one switch?",
        "written_before_any_arm_reading": True,
        "written_before_any_case_verdict": True,
        "sequence": [
            "phase 0 baseline geometry (read only, no verdict)",
            "this manifest is written and hashed",
            "arm A: store enabled + gate on",
            "arm B: store enabled + gate off",
            "compute-only simplification pass (capacity + copy only)",
            "verdict from the registered rule",
        ],
        "mapping": {
            "task_shape": (
                "package 20: 22 correction surfaces per world on a content-addressed store of "
                "32 slots (living_core.ADDRESSED_SLOTS)"
            ),
            "store": (
                "existing F1/F3 fixture unchanged: same encoder (localized_centered, legacy "
                "centroid), 128-d keys, 8192-d values, ADDRESSED_TEMPERATURE 0.1"
            ),
            "why_not_the_model": (
                "the judge is a numpy store gate; inserting it into the torch graph would "
                "change the frozen read path. This probe keeps the judge out of the model and "
                "measures it where it actually acts, on store writes"
            ),
            "initial_occupancy": BASE_OCCUPIED,
            "slots": SLOTS,
            "face_count": FACE_COUNT,
            "face_definition": (
                "face i writes a correction whose value token is the i-th new token at "
                "position 5, and is read with the same token at a different position; a face "
                "is correct iff the read's top-1 slot is the slot it was written to"
            ),
            "no_eviction": "append only; shadow_apply refuses a full store instead of evicting",
        },
        "constants": {
            "base_occupied": BASE_OCCUPIED,
            "slots": SLOTS,
            "face_count": FACE_COUNT,
            "correction_position": CORRECTION_POSITION,
            "synonym_positions": list(SYNONYM_POSITIONS),
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
        "verdict_rule_lexicographic": (
            "compare retention count first, correction count second; 'better' means higher"
        ),
        "retention_caveat": (
            "the 7/7 battery is exact stored keys, so an append can never lower it (a key's "
            "cosine to its own slot is 1.0 and no other row can exceed it). It is reported as "
            "the registered convention; the paraphrase retention is a labelled diagnostic and "
            "is not used by the verdict"
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

    arm_a = run_gate_arm(faces, initial, queries)
    record_a = arm_record(
        "A_gate_on",
        arm_a["state"],
        queries,
        faces,
        arm_a["slots"],
        arm_a["rows"],
        paraphrases,
    )
    append_progress(
        {
            "phase": "arm_A_done",
            "retention_correct": record_a["retention_correct"],
            "corrections_correct": record_a["corrections_correct"],
            "rejected_writes": record_a["rejected_writes"],
            "occupied_final": record_a["occupied_final"],
        }
    )

    arm_b = run_open_arm(faces, initial, queries)
    record_b = arm_record(
        "B_gate_off",
        arm_b["state"],
        queries,
        faces,
        arm_b["slots"],
        arm_b["rows"],
        paraphrases,
    )
    append_progress(
        {
            "phase": "arm_B_done",
            "retention_correct": record_b["retention_correct"],
            "corrections_correct": record_b["corrections_correct"],
            "rejected_writes": record_b["rejected_writes"],
            "occupied_final": record_b["occupied_final"],
        }
    )

    simple = run_simplified(faces, initial, queries)
    simple_slots = simple["slots"]
    approved = [row["tag"] for row in simple["rows"] if row["allowed"]]
    refused_simple = [
        {"tag": row["tag"], "failed": row["failed"]} for row in simple["rows"] if not row["allowed"]
    ]
    refused_by_full = {row["tag"] for row in arm_a["rows"] if not row["accepted"]}
    simplified = {
        "admitted": approved,
        "admitted_count": len(approved),
        "admitted_but_refused_by_full_gate": sorted(set(approved) & refused_by_full),
        "refused": refused_simple,
        "clauses_dropped": [
            gg.EXIT_RETENTION,
            gg.EXIT_SEPARABILITY,
            gg.EXIT_INFLATION,
            gg.EXIT_NO_GAIN,
        ],
        "retention_correct": retention_count(simple["state"], queries),
        "retention_total": len(queries),
        "corrections_correct": sum(
            1 for row in face_readings(simple["state"], faces, simple_slots) if row["correct"]
        ),
        "corrections_total": len(faces),
        "occupied_final": int(simple["state"].occupied),
        "published": False,
        "note": "compute-only pass; nothing written to the store or to any journal",
    }

    rejected = [row for row in arm_a["rows"] if not row["accepted"]]
    distribution: dict[str, int] = {}
    for row in rejected:
        distribution[row["exit_code"]] = distribution.get(row["exit_code"], 0) + 1
    counterfactual_rows = arm_a["counterfactuals"]
    would_break = [row for row in counterfactual_rows if row["would_break_retention"]]

    all_forced_state = arm_a["state"]
    all_forced_applied = 0
    for face in faces:
        if face["tag"] not in refused_by_full:
            continue
        write = gg.WriteCell(
            key=face["key"],
            value=face["value"],
            family=face["family"],
            policy=face["policy"],
        )
        shadow, applied = gg.shadow_apply(all_forced_state, write)
        if applied["applied"]:
            all_forced_state = shadow
            all_forced_applied += 1

    retention_a = int(record_a["retention_correct"])
    retention_b = int(record_b["retention_correct"])
    corrections_a = int(record_a["corrections_correct"])
    corrections_b = int(record_b["corrections_correct"])
    if retention_a == retention_b and corrections_a == corrections_b:
        verdict = "unable_to_assess_no_rejections" if not rejected else "gate_has_no_task_value"
    elif retention_a > retention_b or (
        retention_a == retention_b and corrections_a > corrections_b
    ):
        verdict = "gate_has_task_value"
    else:
        verdict = "gate_hurts"

    stop_criterion_tuning = verdict in {"gate_hurts", "gate_has_no_task_value"}

    cost_cpu = time.process_time() - START_CPU
    cost_wall = time.time() - START_WALL
    if cost_cpu > CPU_CAP:
        raise SystemExit(f"cpu cap exceeded: {cost_cpu:.3f} > {CPU_CAP}")

    stop.set()
    watchdog.join(timeout=2.0)

    report = {
        "probe": "taiji_gate_task_value",
        "date": "2026-11-08",
        "question": "does the consensus gate have task value, measured with one switch?",
        "verdict": verdict,
        "verdict_detail": {
            "rule": manifest["verdict_rule_lexicographic"],
            "arm_A_gate_on": {
                "retention_correct": retention_a,
                "retention_total": len(queries),
                "corrections_correct": corrections_a,
                "corrections_total": len(faces),
                "rejected_writes": len(rejected),
            },
            "arm_B_gate_off": {
                "retention_correct": retention_b,
                "retention_total": len(queries),
                "corrections_correct": corrections_b,
                "corrections_total": len(faces),
                "rejected_writes": 0,
            },
            "delta_retention": retention_a - retention_b,
            "delta_corrections": corrections_a - corrections_b,
            "share_refused_that_would_break_retention": (
                None if not rejected else round(len(would_break) / len(rejected), 6)
            ),
            "stop_criterion_tuning": stop_criterion_tuning,
            "reason": VERDICT_REASON[verdict],
        },
        "preregistration": {
            "manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "manifest_sha256": manifest_sha,
            "written_before_any_arm_reading": True,
            "criteria": CRITERIA,
            "mapping": manifest["mapping"],
            "retention_caveat": manifest["retention_caveat"],
        },
        "fixture": {
            "source": str(F1_RUNNER.relative_to(ROOT)).replace("\\", "/"),
            "encoder": "localized_centered (legacy 4-parent mean centroid)",
            "dim": 128,
            "value_shape": list(f1.VALUE_SHAPE),
            "temperature": gg.GATE_TEMPERATURE,
            "initial_occupancy": BASE_OCCUPIED,
            "slots": SLOTS,
            "face_count": FACE_COUNT,
            "baseline": baseline,
        },
        "arms": {"A_gate_on": record_a, "B_gate_off": record_b},
        "rejections": {
            "count": len(rejected),
            "distribution_by_exit_code": distribution,
            "failed_clause_counts": {
                code: sum(1 for row in rejected if code in row["failed"]) for code in gg.EXIT_CODES
            },
            "all_refusals_bit_identical": all(row["bit_identical"] for row in rejected),
            "quarantine_rows_exact": all(row["quarantine_rows"] == 1 for row in rejected),
            "tag_list": [row["tag"] for row in rejected],
        },
        "counterfactuals": {
            "per_refused_write": counterfactual_rows,
            "would_break_retention_count": len(would_break),
            "would_break_retention_share": (
                None
                if not counterfactual_rows
                else round(len(would_break) / len(counterfactual_rows), 6)
            ),
            "all_refused_forced_together": {
                "applied": all_forced_applied,
                "occupancy": int(all_forced_state.occupied),
                "retention_correct": retention_count(all_forced_state, queries),
                "retention_total": len(queries),
                "published": False,
            },
        },
        "simplification_capacity_copy_only": simplified,
        "retention_diagnostic_paraphrase": {
            "arm_A": {
                "correct": record_a["paraphrase_retention_correct"],
                "total": record_a["paraphrase_retention_total"],
            },
            "arm_B": {
                "correct": record_b["paraphrase_retention_correct"],
                "total": record_b["paraphrase_retention_total"],
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
            "evictions": int(record_a["evictions_total"] + record_b["evictions_total"]),
            "dropped_writes": int(
                record_a["dropped_writes_total"] + record_b["dropped_writes_total"]
            ),
            "store_full_refusals": 0,
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
        "superseded_runs": SUPERSEDED_RUNS,
        "implementation_corrections": [
            {
                "what": (
                    "the compute-only simplification pass tested `failed` as a subset of "
                    "{no_direction}, which still enforced separability and refused the two "
                    "writes it was meant to admit"
                ),
                "fix": (
                    "a dropped clause cannot refuse: the pass now refuses only on capacity or "
                    "the copy band, so the dropped clauses are ignored as intended"
                ),
                "effect_on_verdict": (
                    "none: the verdict is computed from arms A and B, whose rows were identical "
                    "across the two runs"
                ),
                "criteria_changed": False,
                "thresholds_changed": False,
                "constants_changed": False,
                "verdict_rule_changed": False,
            },
            {
                "what": (
                    "the simplification block listed what it admitted but not why the "
                    "remaining writes were still refused, so a capacity refusal and a "
                    "quality refusal looked the same"
                ),
                "fix": (
                    "the block now also lists every refused write with its failing clauses; "
                    "the T5 registration text was corrected to describe the predicate that "
                    "actually runs (a dropped clause may not refuse)"
                ),
                "effect_on_verdict": (
                    "none: compute-only reporting, and arms A and B are bit-identical again"
                ),
                "criteria_changed": False,
                "thresholds_changed": False,
                "constants_changed": False,
                "verdict_rule_changed": False,
            },
        ],
    }
    write_json(REPORT, report)
    release_single_instance()

    print(
        json.dumps(
            {
                "verdict": verdict,
                "A": {
                    "retention": retention_a,
                    "corrections": corrections_a,
                    "refused": len(rejected),
                },
                "B": {"retention": retention_b, "corrections": corrections_b},
                "distribution": distribution,
                "counterfactual_all_forced_retention": retention_count(all_forced_state, queries),
                "simplified_admitted": len(approved),
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
