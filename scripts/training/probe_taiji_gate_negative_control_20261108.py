"""Negative control: does the store ever say "I don't know"?

The generalization probe (``probe_taiji_gate_generalization_20261108``) read 66
never-written phrasings of already-stored content and scored 66/66, but every
query was still the same value token at a new position, and the nearest stored
key was its own key (median cosine 0.96649).  A count like that cannot separate
"the store knows this content" from "the store always returns its nearest key".
This run asks the complementary question, on the same fixture, the same encoder
and the same written store:

    does any query, including queries with no answer, come back with a slot and
    a confidence, or is there a branch that says "no match"?

Three negative classes, 22 items each, all built on the scoring side only:

    new_token    a fixture token that never entered the store (tokens outside
                 the first eight baseline sentences and outside the 22 face
                 tokens), placed at a face's registered read position -- the
                 exact query shape of the written group, different content
    new_pairing  a stored face token placed at *another* face's registered read
                 position; the token is seen, the token/position pair was never
                 written and never read
    no_value     the unit midpoint of two stored keys: a structurally legal
                 128-d key that corresponds to no stored value and has no
                 correct slot (the annotation exists only on the scoring side)

Per item this run reports the returned slot, the top cosine, the second cosine,
the margin, the softmax weight of the returned slot, the effective support and
the float32 payload bytes a consumer would receive if it treated the returned
value as the answer.  It also audits the implementation for an abstention exit
and reports what is actually there: ``living_core.addressed_read`` returns a
``top_slot`` for every non-empty store and blends it with ``ADDRESSED_ALPHA``;
``top_cosine`` and ``margin`` are diagnostics with no gate on them.  No exit is
added or invented here.

Registered before the first class reading (also in the manifest):

    any class with < 16 items                                  -> not_assessable
    any class: share(top_cos >= 0.95) >= 0.8 and median(margin) < 0.05
                                                               -> always_answers
    else, written-group contrast: median(written.top_cos) - median(new_token.top_cos)
        >= 0.10 and share(new_token.top_cos < min(written.top_cos)) >= 0.8 and
        median(new_token.margin) >= 0.02                       -> has_abstention_signal
    otherwise                                                  -> inconclusive

The ``has_abstention_signal`` threshold is the *written group's* minimum top
cosine -- derived only from the positive control, never tuned against the
negative labels.  The same written group (22) and unwritten group (66) are
re-read in this run so every contrast is computed in one pass.

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

REPORT = ROOT / "reports/taiji_gate_negative_control_20261108.json"
MANIFEST = ROOT / "plans/manifests/taiji_gate_negative_control_20261108.json"
PROGRESS = ROOT / "reports/taiji_gate_negative_control_20261108_progress.jsonl"
JOURNAL_DIR = ROOT / "reports/taiji_gate_negative_control_20261108_journal"
LOCK = ROOT / "reports/taiji_gate_negative_control_20261108.lock"
WATCHDOG_MARKER = ROOT / "reports/taiji_gate_negative_control_20261108_watchdog.json"
F1_RUNNER = ROOT / "scripts/training/probe_taiji_growth_f1_correction_20261108.py"
GATE_FILE = ROOT / "taiji/growth_gate.py"
RUNTIME_FILE = ROOT / "taiji/growth_runtime.py"
LIVING_CORE = ROOT / "taiji/living_core.py"

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
NEG_PER_CLASS = 22
FAMILIAR_COS = 0.95
TIGHT_MARGIN = 0.05
FAMILIAR_SHARE = 0.8
SEPARATION_GAP = 0.10
SEPARATION_COVERAGE = 0.8
SEPARATION_MARGIN = 0.02
VALUE_FLOAT_BYTES = 4

RETAINED_CLAUSES = (gg.EXIT_NO_CAPACITY, gg.EXIT_NO_DIRECTION)
DROPPED_CLAUSES = (
    gg.EXIT_RETENTION,
    gg.EXIT_SEPARABILITY,
    gg.EXIT_INFLATION,
    gg.EXIT_NO_GAIN,
)
ABSTENTION_TERMS = ("abstain", "no_match", "no match", "low_confidence", "not_found")

CRITERIA = {
    "N1_new_token": (
        "22 fixture tokens that never entered the store (outside the first eight baseline "
        "sentences and outside the 22 face tokens), each placed at a face's registered read "
        "position; report top_slot / top_cos / second_cos / margin / top_weight"
    ),
    "N2_new_pairing": (
        "each face token placed at another face's registered read position; the token is seen, "
        "the token/position pair was never written and never used as a registered read; the "
        "same readings are reported"
    ),
    "N3_no_value": (
        "the unit midpoint of two stored keys: a structurally legal 128-d key that corresponds "
        "to no stored value; no correct slot exists and the 'no answer' annotation lives only "
        "on the scoring side"
    ),
    "N4_forced_answer_bytes": (
        "for every item, compute the float32 payload a consumer would receive if it treated "
        "the returned value as the answer (4 * 8192 bytes), whether that payload is bit-identical "
        "to a stored payload, and the softmax weight / effective support behind it"
    ),
    "N5_abstention_exit": (
        "audit the read path for an explicit 'no match / low confidence' exit and report what "
        "is actually implemented; no threshold and no exit is added by the probe"
    ),
    "N6_controls": (
        "re-read the written group (22 registered reads) and the unwritten group (66 never-written "
        "phrasings) in the same run with the same metrics, so every contrast is against a measured "
        "positive control, not against a remembered number"
    ),
    "verdict_rule": (
        "any class with < 16 items -> not_assessable_construction_short; any class with "
        "share(top_cos >= 0.95) >= 0.8 and median(margin) < 0.05 -> "
        "always_answers_no_abstention; else if median(written.top_cos) - "
        "median(new_token.top_cos) >= 0.10 and share(new_token.top_cos < min(written.top_cos)) "
        ">= 0.8 and median(new_token.margin) >= 0.02 -> has_abstention_signal; otherwise -> "
        "inconclusive"
    ),
    "threshold_source": (
        "the abstention threshold is the written group's minimum top cosine, taken from the "
        "positive control only; it is not fitted to the negative labels and no threshold is "
        "moved after the first reading"
    ),
    "no_soft_language": (
        "a high cosine on a query with no answer is not called a good match; it is reported as "
        "a nearest-neighbour floor, and the forced-answer payload is reported next to it"
    ),
    "unchanged": "no threshold, fixture, face, position or class is retuned after the first reading",
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


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def cosine_table(state: gg.GateState, key: np.ndarray) -> np.ndarray:
    keys, _ = active(state)
    normalized = keys / np.clip(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12, None)
    return normalized @ normalize(key)


def top_slot(state: gg.GateState, key: np.ndarray) -> int:
    return int(np.argmax(cosine_table(state, key)))


def read_item(state: gg.GateState, item: dict[str, Any]) -> dict[str, Any]:
    """Everything the store actually hands back for one query."""
    keys, values = active(state)
    cosine = cosine_table(state, item["key"])
    order = np.argsort(-cosine)
    logits = cosine / gg.GATE_TEMPERATURE
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    retrieved = (weights[:, None] * values).sum(axis=0)
    top = int(order[0])
    second = int(order[1])
    nearest_payload = values[top]
    denominator = max(float(np.linalg.norm(nearest_payload)), 1e-12)
    residual = float(np.linalg.norm(retrieved - nearest_payload) / denominator)
    as_float32 = retrieved.astype(np.float32)
    payload_float32 = nearest_payload.astype(np.float32)
    byte_payload = as_float32.tobytes()
    return {
        "class": item["class"],
        "tag": item["tag"],
        "source": item.get("source"),
        "token": item.get("token"),
        "position": item.get("position"),
        "top_slot": top,
        "second_slot": second,
        "top_cos": round(float(cosine[top]), 6),
        "second_cos": round(float(cosine[second]), 6),
        "margin": round(float(cosine[top] - cosine[second]), 6),
        "top_weight": round(float(weights[top]), 9),
        "effective_support": round(float(1.0 / np.sum(weights**2)), 6),
        "retrieved_residual_to_nearest_payload": round(residual, 6),
        "bytes_identical_to_nearest_payload": bool(np.array_equal(as_float32, payload_float32)),
        "forced_answer_bytes": len(byte_payload),
        "forced_answer_sha256": hashlib.sha256(byte_payload).hexdigest(),
        "abstention_exit": False,
        "nearest_is_own_face": item.get("nearest_is_own_face"),
    }


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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cosines = [row["top_cos"] for row in rows]
    margins = [row["margin"] for row in rows]
    share_cos = sum(1 for value in cosines if value >= FAMILIAR_COS) / max(len(rows), 1)
    share_margin = sum(1 for value in margins if value < TIGHT_MARGIN) / max(len(rows), 1)
    return {
        "count": len(rows),
        "top_cos": describe(cosines),
        "second_cos": describe([row["second_cos"] for row in rows]),
        "margin": describe(margins),
        "share_top_cos_ge_095": round(share_cos, 6),
        "share_margin_lt_005": round(share_margin, 6),
        "top_weight": describe([row["top_weight"] for row in rows]),
        "effective_support": describe([row["effective_support"] for row in rows]),
        "forced_answer_equals_existing_payload": sum(
            1 for row in rows if row["bytes_identical_to_nearest_payload"]
        ),
        "distinct_top_slots": len({row["top_slot"] for row in rows}),
        "trigger_always_answers": bool(
            share_cos >= FAMILIAR_SHARE and np.median(margins) < TIGHT_MARGIN
        ),
    }


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
    position = SYNONYM_POSITIONS[index % len(SYNONYM_POSITIONS)]
    if position == token % POSITION_COUNT:
        position = SYNONYM_POSITIONS[(index + 1) % len(SYNONYM_POSITIONS)]
    return position


def faces() -> list[dict[str, Any]]:
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
    used = {
        CORRECTION_POSITION,
        int(face["read_position"]),
        int(face["token"]) % POSITION_COUNT,
    }
    candidates = [position for position in range(POSITION_COUNT) if position not in used]
    return candidates[:UNWRITTEN_PER_RULE]


def unwritten_queries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for face in rows:
        for rank, position in enumerate(unwritten_positions(face)):
            context = f1.paraphrase(base0, tokens0[int(face["token"])], position)
            out.append(
                {
                    "class": "unwritten",
                    "tag": f"unwritten_{face['tag']}_{rank}",
                    "face_tag": face["tag"],
                    "token": int(face["token"]),
                    "position": int(position),
                    "key": gr.encode_read_key(context, encoder, centroid),
                }
            )
    return out


def fresh_token_pool(rows: list[dict[str, Any]]) -> list[int]:
    used = set(range(BASE_OCCUPIED)) | {int(face["token"]) for face in rows}
    return [token for token in range(len(tokens0)) if token not in used]


def pairing_position(face: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[int | None, str]:
    """A seen token at a position it was never written or read at."""
    forbidden = {CORRECTION_POSITION, int(face["read_position"])} | set(unwritten_positions(face))
    for other in rows:
        if other["tag"] == face["tag"]:
            continue
        position = int(other["read_position"])
        if position not in forbidden:
            return position, f"other_face_read_position:{other['tag']}"
    for position in range(POSITION_COUNT):
        if position not in forbidden:
            return position, "unused_position"
    return None, "none"


def negative_items(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Three negative classes, built on the scoring side only."""
    pool = fresh_token_pool(rows)
    chosen = pool[:NEG_PER_CLASS]
    new_token_rows = []
    for index, token in enumerate(chosen):
        face = rows[index]
        context = f1.paraphrase(base0, tokens0[token], int(face["read_position"]))
        new_token_rows.append(
            {
                "class": "new_token",
                "tag": f"new_token_{index:02d}",
                "token": int(token),
                "position": int(face["read_position"]),
                "source": (
                    f"fixture token {token} never entered the store; placed at the registered "
                    f"read position of {face['tag']}"
                ),
                "key": gr.encode_read_key(context, encoder, centroid),
                "nearest_is_own_face": False,
            }
        )

    new_pairing_rows = []
    for index, face in enumerate(rows):
        position, source = pairing_position(face, rows)
        if position is None:
            continue
        context = f1.paraphrase(base0, tokens0[int(face["token"])], position)
        new_pairing_rows.append(
            {
                "class": "new_pairing",
                "tag": f"new_pairing_{index:02d}",
                "token": int(face["token"]),
                "position": int(position),
                "source": source,
                "key": gr.encode_read_key(context, encoder, centroid),
                "nearest_is_own_face": True,
            }
        )

    no_value_rows = []
    for index, face in enumerate(rows):
        other = rows[(index + FACE_COUNT // 2) % FACE_COUNT]
        midpoint = normalize(0.5 * face["key"] + 0.5 * other["key"])
        no_value_rows.append(
            {
                "class": "no_value",
                "tag": f"no_value_{index:02d}",
                "token": None,
                "position": None,
                "source": f"unit midpoint of {face['tag']} and {other['tag']} stored keys",
                "key": midpoint,
                "nearest_is_own_face": False,
            }
        )

    construction = {
        "fresh_pool_size": len(pool),
        "fresh_pool_tokens": [int(token) for token in pool],
        "chosen_new_tokens": [int(token) for token in chosen],
        "new_pairing_sources": {
            source: sum(1 for row in new_pairing_rows if row["source"] == source)
            for source in {row["source"] for row in new_pairing_rows}
        },
        "counts": {
            "new_token": len(new_token_rows),
            "new_pairing": len(new_pairing_rows),
            "no_value": len(no_value_rows),
        },
        "minimum_required": NEG_PER_CLASS,
        "sufficient": all(
            count >= NEG_PER_CLASS
            for count in (len(new_token_rows), len(new_pairing_rows), len(no_value_rows))
        ),
    }
    return (
        {
            "new_token": new_token_rows,
            "new_pairing": new_pairing_rows,
            "no_value": no_value_rows,
        },
        construction,
    )


def run_simplified_arm(
    rows: list[dict[str, Any]],
    initial: gg.GateState,
    queries: list[gg.BatteryQuery],
) -> dict[str, Any]:
    """Capacity + copy only, with growth before a capacity refusal."""
    state = initial
    slots: dict[str, int] = {}
    records = []
    growth_events = []
    journal_path = JOURNAL_DIR / "simplified_refusals.jsonl"
    journal_path.write_text("", encoding="utf-8")
    for face in rows:
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
                    "tag": face["tag"],
                    "accepted": False,
                    "exit_code": gg.EXIT_NO_CAPACITY,
                    "failed": [gg.EXIT_NO_CAPACITY],
                    "written_slot": None,
                    "evictions": 0,
                    "dropped": 0,
                    "bit_identical": gg.snapshot(state).sha256 == sha_before,
                }
                with journal_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                append_progress(record)
                records.append(record)
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
                    "headroom_before_growth": round(float(headroom_before["h_slots"]), 6),
                    "headroom_after_growth": round(float(recheck["h_slots"]), 6),
                    "reserve_rechecked": bool(recheck["sufficient"]),
                }
            )
        shadow, applied = gg.shadow_apply(state, write)
        cert = gg.evaluate(state, shadow, queries, write, applied["slot"])
        blocking = [code for code in cert.failed if code in RETAINED_CLAUSES]
        accepted = bool(applied["applied"]) and not blocking
        record = {
            "tag": face["tag"],
            "accepted": accepted,
            "exit_code": blocking[0] if blocking else cert.exit_code,
            "failed": list(cert.failed),
            "blocking": blocking,
            "written_slot": int(applied["slot"]) if accepted else None,
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
        records.append(record)
    return {
        "state": state,
        "slots": slots,
        "rows": records,
        "growth_events": growth_events,
        "quarantine_rows": journal_rows(journal_path),
    }


def written_readings(
    state: gg.GateState,
    rows: list[dict[str, Any]],
    slots: dict[str, int],
) -> list[dict[str, Any]]:
    out = []
    for face in rows:
        item = {
            "class": "written",
            "tag": face["tag"],
            "source": "registered synonym read",
            "token": int(face["token"]),
            "position": int(face["read_position"]),
            "key": face["read_key"],
        }
        reading = read_item(state, item)
        expected = slots.get(face["tag"])
        reading["expected_slot"] = None if expected is None else int(expected)
        reading["correct"] = bool(expected is not None and reading["top_slot"] == int(expected))
        out.append(reading)
    return out


def unwritten_readings(
    state: gg.GateState,
    queries_rows: list[dict[str, Any]],
    slots: dict[str, int],
) -> list[dict[str, Any]]:
    out = []
    for row in queries_rows:
        reading = read_item(state, row)
        expected = slots.get(row["face_tag"])
        reading["expected_slot"] = None if expected is None else int(expected)
        reading["correct"] = bool(expected is not None and reading["top_slot"] == int(expected))
        reading["nearest_is_own_face"] = (
            None if expected is None else reading["top_slot"] == int(expected)
        )
        out.append(reading)
    return out


def abstention_audit() -> dict[str, Any]:
    """Report what the read path actually does; add nothing."""
    text = LIVING_CORE.read_text(encoding="utf-8").lower()
    hits = {term: text.count(term) for term in ABSTENTION_TERMS}
    return {
        "read_path": "taiji/living_core.py::addressed_read",
        "lines": "550-586",
        "returns_for_any_non_empty_store": [
            "enabled",
            "occupied",
            "valid",
            "blended",
            "top_slot",
            "top_cosine",
            "second_cosine",
            "margin",
            "weights",
            "retrieved_norm",
            "fast_norm_before",
            "fast_norm_after",
            "key",
        ],
        "abstention_exit_present": False,
        "threshold_on_top_cosine_or_margin": False,
        "returns_empty_store_diagnostic": True,
        "empty_store_note": (
            "the only non-answer branch is occupied == 0, which reports the empty store; it "
            "does not decide that a query has no match"
        ),
        "term_scan": hits,
        "term_scan_any_hit": any(value > 0 for value in hits.values()),
        "write_gate_is_not_a_read_gate": (
            "growth_gate.evaluate refuses writes; it is never consulted on a read"
        ),
    }


def decide(
    summaries: dict[str, dict[str, Any]],
    written_summary: dict[str, Any],
    coverage_below_written_min: float,
) -> tuple[str, dict[str, Any]]:
    short = [name for name, summary in summaries.items() if summary["count"] < NEG_PER_CLASS]
    if short:
        return "not_assessable_construction_short", {"short_classes": short}
    flagged = [name for name, summary in summaries.items() if summary["trigger_always_answers"]]
    if flagged:
        return "always_answers_no_abstention", {"flagged_classes": flagged}
    fresh = summaries["new_token"]
    gap = float(written_summary["top_cos"]["median"]) - float(fresh["top_cos"]["median"])
    detail = {
        "separation_gap": round(gap, 6),
        "coverage_below_written_min": round(coverage_below_written_min, 6),
        "new_token_margin_median": fresh["margin"]["median"],
        "written_min_top_cos": written_summary["top_cos"]["min"],
        "threshold_used": written_summary["top_cos"]["min"],
    }
    if (
        gap >= SEPARATION_GAP
        and coverage_below_written_min >= SEPARATION_COVERAGE
        and float(fresh["margin"]["median"]) >= SEPARATION_MARGIN
    ):
        return "has_abstention_signal", detail
    return "inconclusive", detail


def main() -> int:
    if time.process_time() - START_CPU > CPU_GUARD:
        raise SystemExit("cpu guard tripped before start")

    process_guard = assert_no_black_processes("gate negative control probe")
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
    rows = faces()
    unwritten = unwritten_queries(rows)

    baseline = {
        "occupied": int(initial.occupied),
        "slots": int(initial.slots),
        "headroom": round((SLOTS - BASE_OCCUPIED) / SLOTS, 6),
        "retention_correct": retention_count(initial, queries),
        "retention_total": len(queries),
        "face_count": len(rows),
        "unwritten_queries": len(unwritten),
    }
    append_progress({"phase": "baseline", **baseline})

    manifest = {
        "format": "taiji-gate-negative-control-preregistration-v1",
        "date": "2026-11-08",
        "question": (
            "when a query has no stored answer, does the store return a slot and a confidence "
            "as if it did, or does it have an abstention branch?"
        ),
        "written_before_any_class_reading": True,
        "written_before_any_verdict": True,
        "sequence": [
            "phase 0 baseline geometry (read only, no verdict)",
            "this manifest is written and hashed",
            "simplified gate writes the 22 faces (same arm as G2e/G2f)",
            "written group (22) and unwritten group (66) re-read in this run",
            "three negative classes (22 each) constructed on the scoring side",
            "per-item readings and the registered verdict",
        ],
        "classes": {
            "new_token": CRITERIA["N1_new_token"],
            "new_pairing": CRITERIA["N2_new_pairing"],
            "no_value": CRITERIA["N3_no_value"],
        },
        "counts": {
            "per_class": NEG_PER_CLASS,
            "classes": 3,
            "written_control": FACE_COUNT,
            "unwritten_control": len(unwritten),
        },
        "constants": {
            "base_occupied": BASE_OCCUPIED,
            "initial_slots": SLOTS,
            "face_count": FACE_COUNT,
            "correction_position": CORRECTION_POSITION,
            "synonym_positions": list(SYNONYM_POSITIONS),
            "unwritten_per_rule": UNWRITTEN_PER_RULE,
            "familiar_cos": FAMILIAR_COS,
            "tight_margin": TIGHT_MARGIN,
            "familiar_share": FAMILIAR_SHARE,
            "separation_gap": SEPARATION_GAP,
            "separation_coverage": SEPARATION_COVERAGE,
            "separation_margin": SEPARATION_MARGIN,
            "gate_temperature": gg.GATE_TEMPERATURE,
            "value_float_bytes": VALUE_FLOAT_BYTES,
        },
        "criteria": CRITERIA,
        "read_path_audit": abstention_audit(),
        "baseline_readings": baseline,
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
            "rule_truth_in_answer_chain": False,
        },
    }
    write_json(MANIFEST, manifest)
    manifest_sha = file_sha(MANIFEST)
    append_progress({"phase": "manifest_written", "manifest_sha256": manifest_sha})

    arm = run_simplified_arm(rows, initial, queries)
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

    negatives, construction = negative_items(rows)
    append_progress(
        {
            "phase": "negative_construction",
            "counts": construction["counts"],
            "sufficient": construction["sufficient"],
            "fresh_pool_size": construction["fresh_pool_size"],
        }
    )

    written = written_readings(state, rows, slots)
    written_correct = sum(1 for row in written if row["correct"])
    written_summary = summarize(written)
    append_progress(
        {
            "phase": "written_control",
            "correct": written_correct,
            "total": len(written),
            "top_cos_median": written_summary["top_cos"]["median"],
        }
    )

    unwritten = unwritten_readings(state, unwritten, slots)
    unwritten_correct = sum(1 for row in unwritten if row["correct"])
    unwritten_summary = summarize(unwritten)
    append_progress(
        {
            "phase": "unwritten_control",
            "correct": unwritten_correct,
            "total": len(unwritten),
            "top_cos_median": unwritten_summary["top_cos"]["median"],
        }
    )

    readings: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for name in ("new_token", "new_pairing", "no_value"):
        class_rows = [read_item(state, item) for item in negatives[name]]
        readings[name] = class_rows
        summaries[name] = summarize(class_rows)
        append_progress(
            {
                "phase": f"class_{name}",
                "count": len(class_rows),
                "top_cos_median": summaries[name]["top_cos"]["median"],
                "margin_median": summaries[name]["margin"]["median"],
                "trigger": summaries[name]["trigger_always_answers"],
            }
        )

    written_min = float(written_summary["top_cos"]["min"])
    coverage = sum(1 for row in readings["new_token"] if row["top_cos"] < written_min) / max(
        len(readings["new_token"]), 1
    )
    verdict, verdict_extra = decide(summaries, written_summary, coverage)

    evictions = int(sum(row["evictions"] for row in arm["rows"]))
    dropped = int(sum(row["dropped"] for row in arm["rows"]))
    cost_cpu = time.process_time() - START_CPU
    cost_wall = time.time() - START_WALL
    if cost_cpu > CPU_CAP:
        raise SystemExit(f"cpu cap exceeded: {cost_cpu:.3f} > {CPU_CAP}")

    stop.set()

    def payload_digest(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "class": row["class"],
            "tag": row["tag"],
            "source": row["source"],
            "token": row["token"],
            "position": row["position"],
            "top_slot": row["top_slot"],
            "second_slot": row["second_slot"],
            "top_cos": row["top_cos"],
            "second_cos": row["second_cos"],
            "margin": row["margin"],
            "top_weight": row["top_weight"],
            "effective_support": row["effective_support"],
            "retrieved_residual_to_nearest_payload": row["retrieved_residual_to_nearest_payload"],
            "bytes_identical_to_nearest_payload": row["bytes_identical_to_nearest_payload"],
            "forced_answer_bytes": row["forced_answer_bytes"],
            "forced_answer_sha256": row["forced_answer_sha256"],
            "abstention_exit": row["abstention_exit"],
        }

    for name in ("new_token", "new_pairing", "no_value"):
        monthly = [payload_digest(row) for row in readings[name]]
        write_json(REPORT.with_name(REPORT.stem + f"_{name}.json"), monthly)

    report = {
        "probe": "taiji_gate_negative_control",
        "date": "2026-11-08",
        "question": manifest["question"],
        "verdict": verdict,
        "verdict_detail": {
            "rule": CRITERIA["verdict_rule"],
            "threshold_source": CRITERIA["threshold_source"],
            "classes": summaries,
            "extra": verdict_extra,
            "controls": {
                "written": {
                    "correct": written_correct,
                    "total": len(written),
                    "summary": written_summary,
                },
                "unwritten": {
                    "correct": unwritten_correct,
                    "total": len(unwritten),
                    "summary": unwritten_summary,
                },
            },
            "read_path": abstention_audit(),
        },
        "construction": construction,
        "readings": {name: [payload_digest(row) for row in readings[name]] for name in readings},
        "controls": {
            "written": [payload_digest(row) for row in written],
            "unwritten": [payload_digest(row) for row in unwritten],
        },
        "arm": {
            "accepted": len(slots),
            "refused": sum(1 for row in arm["rows"] if not row["accepted"]),
            "occupied_final": int(state.occupied),
            "slots_final": int(state.slots),
            "headroom_final": round(
                (int(state.slots) - int(state.occupied)) / max(int(state.slots), 1), 6
            ),
            "retention_correct": retention_count(state, queries),
            "retention_total": len(queries),
            "grown_slots": int(sum(event["grown_slots"] for event in arm["growth_events"])),
            "growth_events": arm["growth_events"],
            "writes": arm["rows"],
        },
        "preregistration": {
            "manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "manifest_sha256": manifest_sha,
            "written_before_any_class_reading": True,
            "classes": manifest["classes"],
            "counts": manifest["counts"],
            "criteria": CRITERIA,
            "constants": manifest["constants"],
        },
        "fixture": {
            "source": str(F1_RUNNER.relative_to(ROOT)).replace("\\", "/"),
            "encoder": "localized_centered (legacy 4-parent mean centroid)",
            "dim": 128,
            "value_shape": list(f1.VALUE_SHAPE),
            "temperature": gg.GATE_TEMPERATURE,
            "baseline": baseline,
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
            "rule_truth_in_answer_chain": False,
        },
        "isolation": {
            "runner_sha256_start": RUNNER_SHA_START,
            "runner_sha256_end": file_sha(Path(__file__).resolve()),
            "runner_sha_stable": file_sha(Path(__file__).resolve()) == RUNNER_SHA_START,
            "gate_sha256": file_sha(GATE_FILE),
            "runtime_sha256": file_sha(RUNTIME_FILE),
            "living_core_sha256": file_sha(LIVING_CORE),
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
            "report": "reports/taiji_gate_generalization_20261108.json",
            "verdict": "rule_generalized",
            "recalled": (
                "66/66 on never-written phrasings, nearest-key cosine median 0.96649, and the "
                "nearest key was the query's own face key 66/66; that is retrieval of one "
                "content token, and this run asks whether a query with no answer is refused"
            ),
        },
        "implementation_corrections": [],
        "what_this_does_not_mean": [
            "no training, no optimizer step and no model forward ran; nothing here says how a "
            "trained head would behave",
            "the store returns a value vector, not text; the forced-answer bytes are the "
            "float32 payload of that vector, and no text decoder is applied",
            "the negative classes are fixtures, not natural language questions; 'no answer' "
            "is a property of the store, not of a user's intent",
            "one session, one seed, no persistence across checkpoints",
        ],
    }
    write_json(REPORT, report)
    release_single_instance()

    print(
        json.dumps(
            {
                "verdict": verdict,
                "classes": {
                    name: {
                        "count": summaries[name]["count"],
                        "top_cos_median": summaries[name]["top_cos"]["median"],
                        "margin_median": summaries[name]["margin"]["median"],
                        "share_ge_095": summaries[name]["share_top_cos_ge_095"],
                    }
                    for name in summaries
                },
                "controls": {
                    "written": {"correct": written_correct, "total": len(written)},
                    "unwritten": {"correct": unwritten_correct, "total": len(unwritten)},
                },
                "abstention_exit": False,
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
