"""Wire a read gate into the real ``addressed_read`` and prove it is inert by default.

The negative control (``probe_taiji_gate_negative_control_20261108``) showed that
the store answers every query: ``new_token`` queries came back at cosine 0.18,
``no_value`` midpoint keys at 0.70, but a single cosine floor at the written
group's minimum (0.959857) separated both classes from the written group.  The
one class it cannot separate is ``new_pairing`` (0.960411 minimum, overlapping
the written floor).  This run does three things, in order:

1. **Zero-disturbance proof (must pass before anything else).**  The gate is
   added to ``taiji.living_core.LivingCoreReference.addressed_read`` with the
   default ``read_gate_cosine=None``.  A verbatim copy of the pre-change read
   lives in this runner.  On a store built by the *real* forward (``observe`` +
   ``addressed_write`` over 30 byte sentences), both functions read the same 22
   written faces, 66 never-written phrasings and 66 negative-class queries.
   Reported: byte hashes of the returned ``fast``, ``max|delta|``, every legacy
   info field, and two real post-read forwards per face.  With the gate off the
   requirement is ``max|delta| = 0``, every hash equal, and ``matched=True``
   everywhere.

2. **Gate candidates, anchored only on the positive control.**  The threshold
   forms are ``min(written top cosine)``, that floor minus 0.005 / 0.01, and
   ``min(written margin)`` alone or combined.  For each form the report gives
   the accept rate per group (written, unwritten, new_token, new_pairing,
   no_value) and the gap between the written floor and each class maximum.  A
   class with a non-positive gap is written down as "cannot be separated by this
   read gate"; no threshold is tuned to the negative labels.

3. **Zero-training enabled readings.**  The gate is switched on at the
   positive-control floor.  Reported: the 22 written faces' returned bytes
   against the gate-off bytes, the ``matched=False`` share of each negative
   class, the bytes a consumer receives if it forces an answer, and the number
   of written faces the gate wrongly refuses (the real cost).

Two arms are measured:

* ``real_forward`` -- native float32 ``LivingCoreReference``, a store written by
  ``observe`` + ``addressed_write`` from 30 deterministic two-byte-word
  sentences; queries are raw bytes (``store_query_key`` is exercised).
* ``fixture_continuity`` -- the G2g fixture keys and payloads handed to the real
  ``addressed_read`` through its diagnostic ``key=`` entry, in float64, so the
  22 faces, the 66 unwritten phrasings, the three negative classes and the
  0.960 floor are the same objects as the previous report.

Zero training, zero optimizer steps, zero Adam, zero confirm reads, zero
checkpoints.  The only model operation is the frozen forward used to build the
store and to check that the post-read forward bytes do not move.  numpy and
torch only; no ``python -m black``.
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
    LivingState,
)
from taiji.process_guard import assert_no_black_processes  # noqa: E402

REPORT_ZERO = ROOT / "reports/taiji_gate_read_gate_zero_20261108.json"
REPORT_ON = ROOT / "reports/taiji_gate_read_gate_enabled_20261108.json"
MANIFEST = ROOT / "plans/manifests/taiji_gate_read_gate_20261108.json"
PROGRESS = ROOT / "reports/taiji_gate_read_gate_20261108_progress.jsonl"
JOURNAL_DIR = ROOT / "reports/taiji_gate_read_gate_20261108_journal"
LOCK = ROOT / "reports/taiji_gate_read_gate_20261108.lock"
WATCHDOG_MARKER = ROOT / "reports/taiji_gate_read_gate_20261108_watchdog.json"
F1_RUNNER = ROOT / "scripts/training/probe_taiji_growth_f1_correction_20261108.py"
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
VALUE_FLOAT_BYTES = 4
CANDIDATE_STEPS = (0.005, 0.01)
FORWARD_FACE_LIMIT = 22
RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()

CRITERIA = {
    "Z1_state_bytes": (
        "gate off vs the verbatim pre-change read, same store and same query: the returned "
        "fast tensor has identical sha256 and max|delta| = 0 over every one of the 154 queries"
    ),
    "Z2_info_fields": (
        "every info field the pre-change read returned has the same value; the only added key "
        "is matched=True and it changes no byte"
    ),
    "Z3_post_read_forward": (
        "for each written face, a real score_targets forward from the gate-off read state has "
        "identical probability bytes and max|delta| = 0 against the legacy read state"
    ),
    "Z4_determinism": (
        "the gate-off read is run twice; every returned-state hash is identical across the two "
        "passes"
    ),
    "C1_threshold_source": (
        "every candidate threshold is derived from the written group only (its minimum top "
        "cosine or minimum margin); no negative label enters the value"
    ),
    "C2_separation": (
        "per candidate and per group: accept rate, written false-abstain count, and the gap "
        "written_floor - class_max; a non-positive gap is reported as 'not separable by this "
        "read gate', never repaired by moving the threshold"
    ),
    "E1_enabled_bytes": (
        "with the gate at the written floor, each written face either returns bytes identical "
        "to the gate-off read (matched) or is counted once as a false abstention"
    ),
    "E2_negative_refusal": (
        "per negative class: matched=False share and the size/hash of the payload a consumer "
        "would receive if it forced the gate-off answer"
    ),
    "unchanged": "no fixture, face, position, class or threshold is retuned after the first reading",
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


def tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


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
    for report in (REPORT_ZERO, REPORT_ON):
        leftover = report.with_suffix(report.suffix + ".tmp")
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


def load_f1():
    spec = importlib.util.spec_from_file_location("f1_probe", F1_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {F1_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The pre-change read, verbatim except for the diagnostic ``key`` argument.
# ---------------------------------------------------------------------------
def legacy_addressed_read(subject, state, raw: bytes, key: torch.Tensor | None = None):
    key = subject.store_query_key(state, raw) if key is None else key
    if state.store_keys is None:
        return state, {"enabled": False, "blended": False, "key": key}
    valid = torch.nonzero(state.store_age >= 0, as_tuple=False).flatten()
    if valid.numel() == 0:
        return state, {
            "enabled": True,
            "occupied": 0,
            "valid": 0,
            "blended": False,
            "key": key,
        }
    keys = state.store_keys[valid]
    cosine = (keys @ key) / torch.clamp(keys.norm(dim=-1) * key.norm(), min=1e-12)
    weights = torch.softmax(cosine / ADDRESSED_TEMPERATURE, dim=0)
    retrieved = torch.einsum("i,i...->...", weights, state.store_values[valid])
    fast = (1 - ADDRESSED_ALPHA) * state.fast + ADDRESSED_ALPHA * retrieved
    ordered = torch.sort(cosine.detach(), descending=True).values
    top = int(torch.argmax(cosine.detach()))
    margin = float(ordered[0] - ordered[1]) if ordered.numel() > 1 else None
    info = {
        "enabled": True,
        "occupied": int(valid.numel()),
        "valid": int(valid.numel()),
        "blended": True,
        "top_slot": int(valid[top]),
        "top_cosine": float(cosine[top].detach()),
        "second_cosine": float(ordered[1]) if ordered.numel() > 1 else None,
        "margin": margin,
        "weights": [float(value.detach()) for value in weights],
        "retrieved_norm": float(retrieved.detach().norm()),
        "fast_norm_before": float(state.fast.detach().norm()),
        "fast_norm_after": float(fast.detach().norm()),
        "key": key,
    }
    return replace(state, fast=fast), info


# ---------------------------------------------------------------------------
# Fixture (byte-for-byte the G2g construction).
# ---------------------------------------------------------------------------
def build_fixture() -> None:
    global f1, base0, tokens0, contexts0, payloads, encoder, centroid
    f1 = load_f1()
    base0, tokens0, contexts0 = f1.fixture()
    payloads = f1.value_payloads(BASE_OCCUPIED + FACE_COUNT)
    legacy = gr.EncoderConfig(mode="mean")
    parents = np.stack([gr.encode_read_key(contexts0[i], legacy) for i in range(4)])
    centroid = gr.store_centroid(parents)
    encoder = gr.EncoderConfig(mode="localized_centered", center=True)


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


def fixture_faces() -> list[dict[str, Any]]:
    rows = []
    for index, token in enumerate(new_tokens(FACE_COUNT)):
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
            }
        )
    return rows


def unwritten_positions(face: dict[str, Any]) -> list[int]:
    used = {
        CORRECTION_POSITION,
        int(face["read_position"]),
        int(face["token"]) % POSITION_COUNT,
    }
    return [position for position in range(POSITION_COUNT) if position not in used][
        :UNWRITTEN_PER_RULE
    ]


def fixture_unwritten(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def fixture_negatives(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
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
                "key": gr.encode_read_key(context, encoder, centroid),
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
            }
        )
    no_value_rows = []
    for index, face in enumerate(rows):
        other = rows[(index + FACE_COUNT // 2) % FACE_COUNT]
        midpoint = face["key"] + other["key"]
        midpoint = midpoint / max(float(np.linalg.norm(midpoint)), 1e-12)
        no_value_rows.append(
            {
                "class": "no_value",
                "tag": f"no_value_{index:02d}",
                "token": None,
                "position": None,
                "source": f"unit midpoint of {face['tag']} and {other['tag']} stored keys",
                "key": midpoint,
            }
        )
    return {
        "new_token": new_token_rows,
        "new_pairing": new_pairing_rows,
        "no_value": no_value_rows,
    }


def continuity_state(subject: LivingCoreReference, rows: list[dict[str, Any]]) -> LivingState:
    """The G2g store handed to the real read: 8 base keys + 22 face keys."""
    keys = np.zeros((SLOTS, int(f1.DIM)), dtype=np.float64)
    values = np.zeros((SLOTS, 4, 2, 32, 32), dtype=np.float64)
    for slot in range(BASE_OCCUPIED):
        keys[slot] = gr.encode_read_key(contexts0[slot], encoder, centroid)
        values[slot] = payloads[slot].reshape(4, 2, 32, 32)
    for index, face in enumerate(rows):
        keys[BASE_OCCUPIED + index] = face["key"]
        values[BASE_OCCUPIED + index] = face["value"].reshape(4, 2, 32, 32)
    age = np.array(
        [*range(BASE_OCCUPIED + FACE_COUNT), *([-1] * (SLOTS - BASE_OCCUPIED - FACE_COUNT))]
    )
    fast = torch.tensor(payloads[-1], dtype=subject.W.dtype).reshape(4, 2, 32, 32)
    return LivingState(
        subject_id="read-gate-continuity",
        revision=0,
        step=0,
        h=subject.W.new_zeros((4, 128)),
        fast=fast,
        nu_ema=subject.W.new_zeros(8),
        keys=(),
        coverage=subject.W.new_empty(0),
        prior=subject.W.new_empty(0),
        store_keys=torch.tensor(keys, dtype=subject.W.dtype),
        store_values=torch.tensor(values, dtype=subject.W.dtype),
        store_age=torch.tensor(age, dtype=torch.long),
        store_count=torch.tensor(BASE_OCCUPIED + FACE_COUNT, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Real-forward arm: 30 two-byte-word sentences, written through the real store.
# ---------------------------------------------------------------------------
WORDS = [bytes([65 + index % 26, 97 + index // 26]) for index in range(64)]
BASE_WORDS = [WORDS[index] for index in range(BASE_OCCUPIED)]


def sentence(token: int, position: int) -> bytes:
    words = list(BASE_WORDS)
    words[position] = WORDS[token]
    return b"".join(words)


def real_face_token(index: int) -> int:
    return FIRST_NEW_TOKEN + index


def real_read_position(token: int, index: int) -> int:
    return synonym_position(token, index)


def build_real_store(subject: LivingCoreReference) -> tuple[LivingState, list[dict[str, Any]]]:
    state = subject.attach_store(subject.initial_state("read-gate-real"))
    allowed = torch.ones((subject.config.groups, subject.config.fast_heads), dtype=torch.bool)
    for token in range(BASE_OCCUPIED):
        text = sentence(token, token % POSITION_COUNT)
        state = subject.observe(state, text, goal=text, allowed=allowed)
        state, _ = subject.addressed_write(state, text)
    rows = []
    for index in range(FACE_COUNT):
        token = real_face_token(index)
        write_text = sentence(token, CORRECTION_POSITION)
        read_position = real_read_position(token, index)
        read_text = sentence(token, read_position)
        state = subject.observe(state, write_text, goal=write_text, allowed=allowed)
        state, info = subject.addressed_write(state, write_text)
        rows.append(
            {
                "index": index,
                "tag": f"face_{index:02d}",
                "token": int(token),
                "read_position": int(read_position),
                "slot": int(info["slot"]),
                "write_text": write_text,
                "read_text": read_text,
                "key": subject.store_query_key(state, write_text),
                "read_key": subject.store_query_key(state, read_text),
            }
        )
    return state, rows


def real_unwritten(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for face in rows:
        for rank, position in enumerate(unwritten_positions(face)):
            text = sentence(int(face["token"]), position)
            out.append(
                {
                    "class": "unwritten",
                    "tag": f"unwritten_{face['tag']}_{rank}",
                    "face_tag": face["tag"],
                    "token": int(face["token"]),
                    "position": int(position),
                    "raw": text,
                }
            )
    return out


def real_negatives(
    subject: LivingCoreReference, state: LivingState, rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    pool = [token for token in range(FIRST_NEW_TOKEN + FACE_COUNT, len(WORDS))]
    chosen = pool[:NEG_PER_CLASS]
    new_token_rows = []
    for index, token in enumerate(chosen):
        face = rows[index]
        text = sentence(token, int(face["read_position"]))
        new_token_rows.append(
            {
                "class": "new_token",
                "tag": f"new_token_{index:02d}",
                "token": int(token),
                "position": int(face["read_position"]),
                "raw": text,
            }
        )
    new_pairing_rows = []
    for index, face in enumerate(rows):
        position, source = pairing_position(face, rows)
        if position is None:
            continue
        text = sentence(int(face["token"]), position)
        new_pairing_rows.append(
            {
                "class": "new_pairing",
                "tag": f"new_pairing_{index:02d}",
                "token": int(face["token"]),
                "position": int(position),
                "source": source,
                "raw": text,
            }
        )
    no_value_rows = []
    for index, face in enumerate(rows):
        other = rows[(index + FACE_COUNT // 2) % FACE_COUNT]
        first = face["key"] / max(float(face["key"].norm()), 1e-12)
        second = other["key"] / max(float(other["key"].norm()), 1e-12)
        midpoint = first + second
        midpoint = midpoint / max(float(midpoint.norm()), 1e-12)
        no_value_rows.append(
            {
                "class": "no_value",
                "tag": f"no_value_{index:02d}",
                "token": None,
                "position": None,
                "source": f"unit midpoint of {face['tag']} and {other['tag']} stored keys",
                "raw": b"",
                "key": midpoint,
            }
        )
    return {
        "new_token": new_token_rows,
        "new_pairing": new_pairing_rows,
        "no_value": no_value_rows,
    }


# ---------------------------------------------------------------------------
# Reading and comparison.
# ---------------------------------------------------------------------------
def item_key(subject, item: dict[str, Any]):
    key = item.get("key")
    if key is None:
        return None
    if isinstance(key, np.ndarray):
        return torch.tensor(key, dtype=subject.W.dtype, device=subject.W.device)
    return key.to(dtype=subject.W.dtype, device=subject.W.device)


def read_once(subject, state, item: dict[str, Any], *, gate_off: bool = False):
    key = item_key(subject, item)
    if gate_off:
        return legacy_addressed_read(subject, state, item.get("raw", b""), key=key)
    return subject.addressed_read(state, item.get("raw", b""), key=key)


def numeric_info(info: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in info.items()
        if name != "key" and isinstance(value, int | float | bool | str | list)
    }


def compare_legacy(subject, state, item: dict[str, Any]) -> dict[str, Any]:
    key = item_key(subject, item)
    legacy_state, legacy_info = legacy_addressed_read(subject, state, item.get("raw", b""), key=key)
    new_state, new_info = subject.addressed_read(state, item.get("raw", b""), key=key)
    legacy_bytes = legacy_state.fast.detach().cpu().contiguous().numpy().tobytes()
    new_bytes = new_state.fast.detach().cpu().contiguous().numpy().tobytes()
    delta = float((new_state.fast - legacy_state.fast).abs().max())
    legacy_numeric = numeric_info(legacy_info)
    new_numeric = numeric_info(new_info)
    field_equal = {
        name: bool(new_numeric.get(name, object()) == value)
        for name, value in legacy_numeric.items()
    }
    key_equal = bool(torch.equal(legacy_info["key"], new_info["key"]))
    extra = sorted(set(new_info) - set(legacy_info))
    return {
        "tag": item["tag"],
        "class": item["class"],
        "bytes_identical": legacy_bytes == new_bytes,
        "fast_sha_legacy": hashlib.sha256(legacy_bytes).hexdigest(),
        "fast_sha_new": hashlib.sha256(new_bytes).hexdigest(),
        "max_abs_delta": delta,
        "top_slot_legacy": legacy_info.get("top_slot"),
        "top_slot_new": new_info.get("top_slot"),
        "fields_all_equal": all(field_equal.values()),
        "fields_unequal": [name for name, equal in field_equal.items() if not equal],
        "key_equal": key_equal,
        "extra_keys": extra,
        "matched": new_info.get("matched"),
    }


def post_read_forward(subject, state, item: dict[str, Any]) -> dict[str, Any]:
    """One real forward after each read; the gate must not move a single byte."""
    key = item_key(subject, item)
    legacy_state, _ = legacy_addressed_read(subject, state, item.get("raw", b""), key=key)
    new_state, _ = subject.addressed_read(state, item.get("raw", b""), key=key)
    prompt = item.get("prompt") or item.get("raw") or b"read-gate-probe"
    legacy_probs, legacy_after = subject.score_targets(
        legacy_state, prompt=prompt, target=b"!", goal=prompt
    )
    new_probs, new_after = subject.score_targets(new_state, prompt=prompt, target=b"!", goal=prompt)
    legacy_bytes = legacy_probs.detach().cpu().contiguous().numpy().tobytes()
    new_bytes = new_probs.detach().cpu().contiguous().numpy().tobytes()
    return {
        "tag": item["tag"],
        "probability_bytes_identical": legacy_bytes == new_bytes,
        "max_abs_delta": float((new_probs - legacy_probs).abs().max()),
        "fast_after_identical": tensor_sha(legacy_after.fast) == tensor_sha(new_after.fast),
    }


def group_items(arm: str, rows, unwritten, negatives) -> list[dict[str, Any]]:
    items = []
    for face in rows:
        if arm == "real_forward":
            items.append({"class": "written", "tag": face["tag"], "raw": face["read_text"]})
        else:
            items.append(
                {"class": "written", "tag": face["tag"], "raw": b"", "key": face["read_key"]}
            )
    for row in unwritten:
        items.append(dict(row))
    for name in ("new_token", "new_pairing", "no_value"):
        for row in negatives[name]:
            items.append(dict(row))
    return items


def g0(subject, state, arm: str, rows, unwritten, negatives) -> dict[str, Any]:
    items = group_items(arm, rows, unwritten, negatives)
    readings = {}
    for item in items:
        _, info = read_once(subject, state, item)
        readings[item["tag"]] = {
            "class": item["class"],
            "tag": item["tag"],
            "top_cosine": info.get("top_cosine"),
            "second_cosine": info.get("second_cosine"),
            "margin": info.get("margin"),
            "top_slot": info.get("top_slot"),
            "weights": info.get("weights"),
            "matched": info.get("matched"),
        }
    return readings


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


def classify(readings: dict[str, Any], readings_off: dict[str, Any]) -> dict[str, Any]:
    groups = ("written", "unwritten", "new_token", "new_pairing", "no_value")
    out = {}
    for group in groups:
        rows = [row for row in readings.values() if row["class"] == group]
        off_rows = [row for row in readings_off.values() if row["class"] == group]
        out[group] = {
            "count": len(rows),
            "top_cosine": describe(
                [row["top_cosine"] for row in rows if row["top_cosine"] is not None]
            ),
            "margin": describe([row["margin"] for row in rows if row["margin"] is not None]),
            "top_cosines": [row["top_cosine"] for row in rows],
            "margins": [row["margin"] for row in rows],
            "off_top_cosines": [row["top_cosine"] for row in off_rows],
            "off_margins": [row["margin"] for row in off_rows],
        }
    return out


def candidate_table(groups: dict[str, Any], floor: float, margin_floor: float) -> dict[str, Any]:
    candidates = {
        "cos_floor": {"kind": "cosine", "value": floor},
        "cos_floor_minus_0.005": {"kind": "cosine", "value": floor - CANDIDATE_STEPS[0]},
        "cos_floor_minus_0.01": {"kind": "cosine", "value": floor - CANDIDATE_STEPS[1]},
        "margin_floor": {"kind": "margin", "value": margin_floor},
        "cos_and_margin": {
            "kind": "cosine_and_margin",
            "cosine": floor,
            "margin": margin_floor,
        },
    }
    table = {}
    for name, candidate in candidates.items():
        row = {"threshold": candidate, "groups": {}}
        for group, summary in groups.items():
            cosines = summary["off_top_cosines"]
            margins = summary["off_margins"]
            if candidate["kind"] == "cosine":
                accepted = [value >= candidate["value"] for value in cosines]
            elif candidate["kind"] == "margin":
                accepted = [value >= candidate["value"] for value in margins]
            else:
                accepted = [
                    cosines[index] >= candidate["cosine"] and margins[index] >= candidate["margin"]
                    for index in range(len(cosines))
                ]
            row["groups"][group] = {
                "accept": int(sum(accepted)),
                "total": len(accepted),
                "accept_rate": round(sum(accepted) / max(len(accepted), 1), 6),
                "false_abstain": (
                    int(len(accepted) - sum(accepted)) if group == "written" else None
                ),
            }
            if group in ("new_token", "new_pairing", "no_value"):
                row["groups"][group]["gap_written_floor_minus_class_max"] = round(
                    floor - float(max(cosines)), 9
                )
                row["groups"][group]["separable_by_cosine"] = bool(floor - float(max(cosines)) > 0)
        table[name] = row
    return table


def enabled_run(
    subject, state, arm: str, rows, unwritten, negatives, threshold: float
) -> dict[str, Any]:
    items = group_items(arm, rows, unwritten, negatives)
    matched_counts = {"written": 0, "unwritten": 0, "new_token": 0, "new_pairing": 0, "no_value": 0}
    totals = dict.fromkeys(matched_counts, 0)
    face_identical = 0
    face_abstained = 0
    forced_bytes = 0
    forced_identical = 0
    per_item = []
    for item in items:
        before_sha = tensor_sha(state.fast)
        after_state, info = read_once(subject, state, item)
        matched = bool(info.get("matched"))
        totals[item["class"]] += 1
        if matched:
            matched_counts[item["class"]] += 1
        if item["class"] == "written":
            if matched and tensor_sha(after_state.fast) == before_sha:
                raise SystemExit(f"matched read did not change fast for {item['tag']}")
            if not matched:
                face_abstained += 1
            if matched:
                legacy_state, _ = legacy_addressed_read(
                    subject, state, item.get("raw", b""), key=item_key(subject, item)
                )
                if tensor_sha(after_state.fast) == tensor_sha(legacy_state.fast):
                    face_identical += 1
        if not matched and item["class"] in ("new_token", "new_pairing", "no_value"):
            forced_state, forced_info = read_once(subject, state, item, gate_off=True)
            payload = forced_state.fast.detach().cpu().contiguous().numpy().tobytes()
            forced_bytes += len(payload)
            nearest = int(forced_info["top_slot"])
            stored = state.store_values[nearest].detach().cpu().contiguous().numpy().tobytes()
            if payload == stored:
                forced_identical += 1
        per_item.append(
            {
                "class": item["class"],
                "tag": item["tag"],
                "matched": matched,
                "reason": info.get("reason"),
                "top_slot": info.get("top_slot"),
                "top_cosine": info.get("top_cosine"),
                "margin": info.get("margin"),
                "returned_fast_sha": tensor_sha(after_state.fast),
                "returned_fast_bytes": int(
                    after_state.fast.numel() * after_state.fast.element_size()
                ),
                "forced_answer_bytes": (
                    None if matched else int(state.fast.numel() * state.fast.element_size())
                ),
            }
        )
    return {
        "threshold": round(float(threshold), 9),
        "groups": {
            group: {
                "matched": matched_counts[group],
                "total": totals[group],
                "matched_rate": round(matched_counts[group] / max(totals[group], 1), 6),
                "refused": totals[group] - matched_counts[group],
                "refusal_rate": round(
                    (totals[group] - matched_counts[group]) / max(totals[group], 1), 6
                ),
            }
            for group in matched_counts
        },
        "written_false_abstain": face_abstained,
        "written_matched_bytes_identical_to_off": face_identical,
        "forced_answer_bytes_total": forced_bytes,
        "forced_answer_identical_to_stored": forced_identical,
        "items": per_item,
    }


def zero_report(
    real_arm: dict[str, Any], continuity_arm: dict[str, Any], process_guard: str, lock_name: str
) -> dict[str, Any]:
    passed = all(
        arm["bytes_identical"] == arm["total"]
        and arm["max_abs_delta"] == 0.0
        and arm["fields_all_equal"] == arm["total"]
        and arm["key_equal"] == arm["total"]
        and arm["extra_keys"] == [("matched",)]
        and arm["matched_true"] == arm["total"]
        and arm["forward_equal"] == arm["forward_total"]
        and arm["forward_max_abs_delta"] == 0.0
        and arm["deterministic"]
        for arm in (real_arm, continuity_arm)
    )
    return {
        "probe": "taiji_gate_read_gate_zero",
        "date": "2026-11-08",
        "question": "with the read gate off, is the real addressed_read bit-identical to the pre-change read?",
        "verdict": "zero_disturbance" if passed else "disturbance_detected",
        "passed": bool(passed),
        "criteria": CRITERIA,
        "arms": {"real_forward": real_arm, "fixture_continuity": continuity_arm},
        "isolation": {
            "runner_sha256_start": RUNNER_SHA_START,
            "runner_sha256_end": file_sha(Path(__file__).resolve()),
            "living_core_sha256": file_sha(LIVING_CORE),
            "process_guard": process_guard,
            "single_instance_lock": lock_name,
            "watchdog": {
                "seconds": WATCHDOG_SECONDS,
                "fired": bool(WATCHDOG_FIRED[0]),
                "marker_exists": WATCHDOG_MARKER.exists(),
            },
            "write_then_rename": True,
        },
        "discipline": {
            "training_updates": 0,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "confirm_reads": 0,
            "checkpoints_written": 0,
            "black_invoked": False,
            "black_gate": "in-process only",
            "cpu_cap_seconds": CPU_CAP,
        },
        "cost": {
            "cpu_seconds": round(time.process_time() - START_CPU, 3),
            "wall_seconds": round(time.time() - START_WALL, 3),
        },
    }


def parameter_sha(subject) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(subject.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def arm_zero(subject, state, arm: str, rows, unwritten, negatives) -> dict[str, Any]:
    items = group_items(arm, rows, unwritten, negatives)
    compared = []
    for item in items:
        compared.append(compare_legacy(subject, state, item))
    forward = []
    for face in rows[:FORWARD_FACE_LIMIT]:
        item = (
            {"class": "written", "tag": face["tag"], "raw": face["read_text"]}
            if arm == "real_forward"
            else {
                "class": "written",
                "tag": face["tag"],
                "raw": b"",
                "prompt": b"read-gate-probe",
                "key": face["read_key"],
            }
        )
        forward.append(post_read_forward(subject, state, item))
        append_progress({"phase": f"zero_forward_{arm}", "tag": item["tag"]})
    second = [compare_legacy(subject, state, item) for item in items]
    deterministic = all(
        left["fast_sha_new"] == right["fast_sha_new"]
        for left, right in zip(compared, second, strict=True)
    )
    result = {
        "arm": arm,
        "total": len(compared),
        "bytes_identical": sum(1 for row in compared if row["bytes_identical"]),
        "max_abs_delta": max(row["max_abs_delta"] for row in compared),
        "fields_all_equal": sum(1 for row in compared if row["fields_all_equal"]),
        "key_equal": sum(1 for row in compared if row["key_equal"]),
        "matched_true": sum(1 for row in compared if row["matched"] is True),
        "extra_keys": sorted({tuple(row["extra_keys"]) for row in compared}),
        "unequal_fields": sorted({name for row in compared for name in row["fields_unequal"]}),
        "forward_total": len(forward),
        "forward_equal": sum(1 for row in forward if row["probability_bytes_identical"]),
        "forward_max_abs_delta": max((row["max_abs_delta"] for row in forward), default=0.0),
        "deterministic": bool(deterministic),
        "rows": compared,
        "forward_rows": forward,
    }
    append_progress(
        {
            "phase": f"zero_arm_{arm}",
            "bytes_identical": result["bytes_identical"],
            "total": result["total"],
            "max_abs_delta": result["max_abs_delta"],
            "forward_equal": result["forward_equal"],
        }
    )
    return result


def main() -> int:
    if time.process_time() - START_CPU > CPU_GUARD:
        raise SystemExit("cpu guard tripped before start")

    process_guard = assert_no_black_processes("read gate probe")
    lock_name = acquire_single_instance()
    atexit.register(release_single_instance)
    stop = threading.Event()
    start_watchdog(stop)

    reset_run_logs()
    torch.set_grad_enabled(False)
    build_fixture()
    torch.manual_seed(20261108)

    manifest = {
        "format": "taiji-read-gate-preregistration-v1",
        "date": "2026-11-08",
        "question": (
            "can the real addressed_read refuse a query with no stored answer, and is the gate "
            "provably inert when it is off?"
        ),
        "written_before_any_reading": True,
        "sequence": [
            "build the real-forward store (30 byte sentences) and the continuity store",
            "zero-disturbance pass on both arms (must pass before the enabled pass)",
            "gate candidates from the written group only",
            "enabled readings at the positive-control floor",
        ],
        "criteria": CRITERIA,
        "constants": {
            "base_occupied": BASE_OCCUPIED,
            "slots": SLOTS,
            "face_count": FACE_COUNT,
            "correction_position": CORRECTION_POSITION,
            "synonym_positions": list(SYNONYM_POSITIONS),
            "unwritten_per_rule": UNWRITTEN_PER_RULE,
            "negative_per_class": NEG_PER_CLASS,
            "candidate_steps": list(CANDIDATE_STEPS),
            "temperature": ADDRESSED_TEMPERATURE,
            "alpha": ADDRESSED_ALPHA,
            "value_float_bytes": VALUE_FLOAT_BYTES,
        },
        "discipline": {
            "training_updates": 0,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "confirm_reads": 0,
            "black_invoked": False,
            "cpu_cap_seconds": CPU_CAP,
            "watchdog_seconds": WATCHDOG_SECONDS,
            "rule_truth_in_answer_chain": False,
        },
    }
    write_json(MANIFEST, manifest)
    append_progress({"phase": "manifest_written", "sha256": file_sha(MANIFEST)})

    real_subject = LivingCoreReference(LivingCoreConfig())
    real_state, real_rows = build_real_store(real_subject)
    real_unwritten_rows = real_unwritten(real_rows)
    real_negative_rows = real_negatives(real_subject, real_state, real_rows)
    append_progress(
        {
            "phase": "real_store_built",
            "occupied": int((real_state.store_age >= 0).sum()),
            "slots": int(real_state.store_keys.shape[0]),
            "faces": len(real_rows),
        }
    )

    continuity_subject = LivingCoreReference(LivingCoreConfig()).double()
    fixture_rows = fixture_faces()
    fixture_unwritten_rows = fixture_unwritten(fixture_rows)
    fixture_negative_rows = fixture_negatives(fixture_rows)
    continuity = continuity_state(continuity_subject, fixture_rows)
    append_progress({"phase": "continuity_store_built", "faces": len(fixture_rows)})

    zero_real = arm_zero(
        real_subject, real_state, "real_forward", real_rows, real_unwritten_rows, real_negative_rows
    )
    zero_continuity = arm_zero(
        continuity_subject,
        continuity,
        "fixture_continuity",
        fixture_rows,
        fixture_unwritten_rows,
        fixture_negative_rows,
    )
    report_zero = zero_report(zero_real, zero_continuity, process_guard, lock_name)
    write_json(REPORT_ZERO, report_zero)
    append_progress({"phase": "zero_report_written", "verdict": report_zero["verdict"]})
    if not report_zero["passed"]:
        stop.set()
        print(json.dumps({"zero_verdict": report_zero["verdict"], "passed": False}))
        return 2

    arms = {}
    for arm, subject, state, rows, unwritten, negatives in (
        (
            "real_forward",
            real_subject,
            real_state,
            real_rows,
            real_unwritten_rows,
            real_negative_rows,
        ),
        (
            "fixture_continuity",
            continuity_subject,
            continuity,
            fixture_rows,
            fixture_unwritten_rows,
            fixture_negative_rows,
        ),
    ):
        readings_off = g0(subject, state, arm, rows, unwritten, negatives)
        groups = classify(readings_off, readings_off)
        floor = float(min(groups["written"]["off_top_cosines"]))
        margin_floor = float(min(groups["written"]["off_margins"]))
        candidates = candidate_table(groups, floor, margin_floor)
        on_subject = LivingCoreReference(replace(subject.config, read_gate_cosine=floor))
        if arm == "fixture_continuity":
            on_subject = on_subject.double()
        params_equal = parameter_sha(on_subject) == parameter_sha(subject)
        enabled = enabled_run(on_subject, state, arm, rows, unwritten, negatives, floor)
        negative_names = ("new_token", "new_pairing", "no_value")
        gaps = {
            name: round(floor - float(max(groups[name]["off_top_cosines"])), 9)
            for name in negative_names
        }
        separable = [name for name in negative_names if gaps[name] > 0]
        separation = {
            "gaps_written_floor_minus_class_max": gaps,
            "separable_by_cosine": separable,
            "not_separable_by_cosine": [name for name in negative_names if name not in separable],
            "verdict": (
                "no_separation_on_real_keys"
                if not separable
                else (
                    "partial_separation"
                    if len(separable) < len(negative_names)
                    else "full_separation"
                )
            ),
            "unwritten_false_refusal": enabled["groups"]["unwritten"]["refused"],
            "unwritten_total": enabled["groups"]["unwritten"]["total"],
        }
        arms[arm] = {
            "gate_off_groups": groups,
            "candidates": candidates,
            "positive_control_floor": floor,
            "positive_control_margin_floor": margin_floor,
            "params_equal_to_off": bool(params_equal),
            "separation": separation,
            "enabled": enabled,
        }
        append_progress(
            {
                "phase": f"enabled_{arm}",
                "floor": floor,
                "written_false_abstain": enabled["written_false_abstain"],
                "new_token_refusal": enabled["groups"]["new_token"]["refusal_rate"],
                "new_pairing_refusal": enabled["groups"]["new_pairing"]["refusal_rate"],
                "no_value_refusal": enabled["groups"]["no_value"]["refusal_rate"],
            }
        )

    cost_cpu = time.process_time() - START_CPU
    cost_wall = time.time() - START_WALL
    if cost_cpu > CPU_CAP:
        raise SystemExit(f"cpu cap exceeded: {cost_cpu:.3f} > {CPU_CAP}")
    stop.set()

    report_on = {
        "probe": "taiji_gate_read_gate_enabled",
        "date": "2026-11-08",
        "question": manifest["question"],
        "verdict_source": (
            "a class is separable only if the written floor is strictly above that class's "
            "maximum top cosine; the threshold itself is always the written floor"
        ),
        "arms": {
            arm: {
                "positive_control_floor": data["positive_control_floor"],
                "positive_control_margin_floor": data["positive_control_margin_floor"],
                "params_equal_to_off": data["params_equal_to_off"],
                "separation": data["separation"],
                "gate_off_groups": {
                    group: {
                        "count": summary["count"],
                        "top_cosine": summary["top_cosine"],
                        "margin": summary["margin"],
                    }
                    for group, summary in data["gate_off_groups"].items()
                },
                "candidates": data["candidates"],
                "enabled": data["enabled"],
            }
            for arm, data in arms.items()
        },
        "preregistration": {
            "manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "manifest_sha256": file_sha(MANIFEST),
            "written_before_any_reading": True,
            "criteria": CRITERIA,
            "constants": manifest["constants"],
        },
        "zero_disturbance": {
            "report": str(REPORT_ZERO.relative_to(ROOT)).replace("\\", "/"),
            "verdict": report_zero["verdict"],
            "passed": report_zero["passed"],
        },
        "isolation": {
            "runner_sha256_start": RUNNER_SHA_START,
            "runner_sha256_end": file_sha(Path(__file__).resolve()),
            "living_core_sha256": file_sha(LIVING_CORE),
            "process_guard": process_guard,
            "single_instance_lock": lock_name,
            "watchdog": {
                "seconds": WATCHDOG_SECONDS,
                "fired": bool(WATCHDOG_FIRED[0]),
                "marker_exists": WATCHDOG_MARKER.exists(),
            },
            "write_then_rename": True,
        },
        "discipline": {
            "training_updates": 0,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "confirm_reads": 0,
            "checkpoints_written": 0,
            "black_invoked": False,
            "black_gate": "in-process only",
        },
        "cost": {
            "cpu_seconds": round(cost_cpu, 3),
            "wall_seconds": round(cost_wall, 3),
            "cpu_cap_seconds": CPU_CAP,
        },
        "what_this_does_not_mean": [
            "a refused read is not a trained decision; it is a fixed cosine floor on one fixture",
            "the forced-answer bytes are the float32 payload of the store value, not text",
            "new_pairing overlaps the written group, so the read gate cannot refuse it; that is "
            "reported as not separable, not repaired with a fitted threshold",
        ],
    }
    write_json(REPORT_ON, report_on)
    print(
        json.dumps(
            {
                "zero_verdict": report_zero["verdict"],
                "zero_passed": report_zero["passed"],
                "real_floor": arms["real_forward"]["positive_control_floor"],
                "continuity_floor": arms["fixture_continuity"]["positive_control_floor"],
                "real_false_abstain": arms["real_forward"]["enabled"]["written_false_abstain"],
                "continuity_false_abstain": arms["fixture_continuity"]["enabled"][
                    "written_false_abstain"
                ],
                "cpu_seconds": round(cost_cpu, 3),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
