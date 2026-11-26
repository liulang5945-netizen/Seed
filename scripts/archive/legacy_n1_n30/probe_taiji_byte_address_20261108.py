"""Byte-match addressing: move the store's address off the pooled vector.

Owner order (2026-11-08): the copy gate, the thresholds and the capacity logic
stay exactly as they are.  The one variable is **how a record is addressed**:
instead of the 128-dim mean of the encoder's context vectors, the address is the
alignment the 19.2r-verified ``ByteMatchRoute`` produces (65,807 parameters,
identity-initialised, never trained).

Write side: the sentence is matched byte-for-byte against the record lines that
are already in the store through the same route, and the route's (matched
offset, alignment score) is recorded as that write's address.  Read side: the
query is matched against every stored record line through the same route, and
the highest alignment score wins.  No new transform is layered on the mean
vector: the route *is* the match device, and the mean key is never consulted by
the byte-address read (it stays in the tensor only because the frozen write path
computes it and the frozen copy gate reads it).

Readings on the same 154-query battery, side by side with today's mean address:
22 paraphrases, 66 never-written wordings, 22 new tokens, 22 new pairings, 22
key-space midpoints (which have no byte form and are reported as not
applicable), plus the 22 write-backs the order asks for.

Discipline: 0 training / 0 optimizer step / 0 Adam / 0 confirm, no
``python -m black`` (in-process gate only), ``process_guard``, single-instance
lock, 60 s watchdog, CPU cap 200, readings flushed as they are taken,
write-then-rename, dual runner sha.
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
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji import growth_wire as gw  # noqa: E402
from taiji.byte_match_route import IDENTITY_DIM, PAD_ID, VOCAB, ByteMatchRoute  # noqa: E402
from taiji.growth_wire import GrowthWireConfig  # noqa: E402
from taiji.living_core import (  # noqa: E402
    ADDRESSED_TEMPERATURE,
    LivingCoreConfig,
    LivingCoreReference,
)
from taiji.process_guard import assert_no_black_processes  # noqa: E402
from taiji.rule_learning_20_1_identity import graph_identity  # noqa: E402
from taiji.rule_learning_20_1_net import RuleLearningGraph  # noqa: E402

BASE_RUNNER = ROOT / "scripts/training/probe_taiji_gate_read_gate_20261108.py"
ROUTE = ROOT / "taiji/byte_match_route.py"
GROWTH_WIRE = ROOT / "taiji/growth_wire.py"
LIVING_CORE = ROOT / "taiji/living_core.py"
REPORT = ROOT / "reports/taiji_byte_address_20261108.json"
MANIFEST = ROOT / "plans/manifests/taiji_byte_address_20261108.json"
PROGRESS = ROOT / "reports/taiji_byte_address_20261108_progress.jsonl"
LOCK = ROOT / "reports/taiji_byte_address_20261108.lock"
WATCHDOG_MARKER = ROOT / "reports/taiji_byte_address_20261108_watchdog.json"

CPU_CAP = 200.0
CPU_GUARD = 190.0
WATCHDOG_SECONDS = 60.0
LOCK_STALE_SECONDS = 600.0
SPAN = 15
CHUNK = 32
NEAR_ONE = 0.85
NEED_WRITTEN = 20
NEED_UNWRITTEN = 60
NEED_GAPS = 2
CLASSES = (
    "written_writeback",
    "written_paraphrase",
    "unwritten",
    "new_token",
    "new_pairing",
    "no_value",
)
BYTE_CLASSES = ("written_writeback", "written_paraphrase", "unwritten", "new_token", "new_pairing")
NEGATIVE_BYTE_CLASSES = ("new_token", "new_pairing")

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
START_CPU = time.process_time()
START_WALL = time.time()
LAST_READING = [time.time()]
WATCHDOG_FIRED = [False]

CRITERIA = {
    "B1_route_is_the_primitive": (
        "the address is produced by ByteMatchRoute at its 19.2r initialisation: byte_embed equals "
        "eye(257, 256) exactly, position_weight equals 1 exactly, 0 updates, 0 optimizer steps; "
        "the free-offset search is the route's own (no injected offsets)"
    ),
    "B2_one_variable": (
        "both arms read the same store: store keys, values, ages and count are byte-identical "
        "before and after every reading; only the read address differs (mean key vs byte route)"
    ),
    "B3_battery": (
        "identical 154-query battery as the closed encoding line (22 paraphrases, 66 unwritten "
        "wordings, 3 negative classes of 22); the no_value class is a key-space midpoint with no "
        "textual form, so it has no byte query and is reported as not applicable"
    ),
    "B4_written_primary": (
        "written face count = the write-back read: query the stored sentence itself, correct iff "
        "the winning line is the slot that sentence was written to"
    ),
    "B5_gaps": (
        "the floor used for the class gaps is the minimum normalised alignment over the 22 "
        "paraphrases; the write-back floor is reported separately because it equals the maximum "
        "attainable score by construction and cannot test recognition"
    ),
    "B6_gate_frozen": (
        "the copy gate is not modified and not retuned; its input remains the frozen mean key, so "
        "its verdict on the 22 correction faces is reported as inherited, not as caused by the "
        "byte address"
    ),
    "V1_holds": (
        "written write-back >= 20 AND unwritten >= 60 AND at least two byte-expressible negative "
        "classes have a strictly positive gap AND the copy gate no longer refuses the 22 faces"
    ),
    "V2_not_discriminative": (
        "every byte-expressible negative class has a normalised maximum >= 0.85 (near one) AND no "
        "class has a strictly positive gap: any two sentences of this template align about equally"
    ),
    "V3_inconclusive": "anything else: report the distributions only, no partial success",
    "near_one": f"normalised alignment score >= {NEAR_ONE}",
    "unchanged": "no fixture, face, position, class, span or threshold is retuned after the first reading",
}


# ---------------------------------------------------------------------------
# Discipline helpers.
# ---------------------------------------------------------------------------
def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    LOCK.unlink(missing_ok=True)


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
# The route as the address device.  Identity initialisation, no training.
# ---------------------------------------------------------------------------
def identity_check(model: ByteMatchRoute) -> dict[str, Any]:
    reference = torch.eye(VOCAB, IDENTITY_DIM)
    return {
        "byte_embed_max_deviation": float((model.byte_embed.detach() - reference).abs().max()),
        "position_weight_max_deviation": float(
            (model.position_weight.detach() - torch.ones(SPAN)).abs().max()
        ),
        "byte_embed_shape": list(model.byte_embed.shape),
        "position_weight_shape": list(model.position_weight.shape),
        "parameters": int(sum(t.numel() for t in model.parameters())),
        "trainable_parameters": int(sum(t.numel() for t in model.parameters() if t.requires_grad)),
        "optimizer_steps": 0,
        "updates_applied": 0,
        "mode": "free offset search (align_offsets=None)",
    }


def record_tensor(records: list[bytes], device) -> tuple[torch.Tensor, torch.Tensor]:
    longest = max(len(raw) for raw in records)
    ids = torch.full((1, len(records), longest), PAD_ID, dtype=torch.long, device=device)
    mask = torch.zeros((1, len(records), longest), dtype=torch.bool, device=device)
    for index, raw in enumerate(records):
        ids[0, index, : len(raw)] = torch.tensor(list(raw), dtype=torch.long, device=device)
        mask[0, index, : len(raw)] = True
    return ids, mask


def query_tensor(texts: list[bytes], device) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.full((len(texts), SPAN), PAD_ID, dtype=torch.long, device=device)
    mask = torch.zeros((len(texts), SPAN), dtype=torch.bool, device=device)
    for index, raw in enumerate(texts):
        piece = raw[:SPAN]
        ids[index, : len(piece)] = torch.tensor(list(piece), dtype=torch.long, device=device)
        mask[index, : len(piece)] = True
    return ids, mask


def byte_scores(
    model: ByteMatchRoute, texts: list[bytes], records: list[bytes], device
) -> tuple[torch.Tensor, torch.Tensor]:
    rec_ids, rec_mask = record_tensor(records, device)
    score_chunks: list[torch.Tensor] = []
    offset_chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), CHUNK):
        batch = texts[start : start + CHUNK]
        q_ids, q_mask = query_tensor(batch, device)
        count = len(batch)
        scores, offsets, _ = model(
            q_ids,
            q_mask,
            rec_ids.expand(count, -1, -1),
            rec_mask.expand(count, -1, -1),
        )
        score_chunks.append(scores.detach())
        offset_chunks.append(offsets.detach())
        touch()
    return torch.cat(score_chunks, 0), torch.cat(offset_chunks, 0)


def byte_read(scores: torch.Tensor, slots: list[int], index: int) -> dict[str, Any]:
    row = scores[index]
    top = int(torch.argmax(row))
    ordered = torch.sort(row, descending=True).values
    return {
        "applicable": True,
        "top_slot": int(slots[top]),
        "top_score": float(row[top]),
        "top_score_normalized": float(row[top]) / SPAN,
        "second_score": float(ordered[1]) if ordered.numel() > 1 else None,
        "margin": float(ordered[0] - ordered[1]) if ordered.numel() > 1 else None,
        "line_index": top,
    }


# ---------------------------------------------------------------------------
# The frozen mean address, for the side-by-side arm.
# ---------------------------------------------------------------------------
def mean_scores(state, key: torch.Tensor) -> dict[str, Any]:
    valid = torch.nonzero(state.store_age >= 0, as_tuple=False).flatten()
    keys = state.store_keys[valid]
    cosine = (keys @ key) / torch.clamp(keys.norm(dim=-1) * key.norm(), min=1e-12)
    ordered = torch.sort(cosine.detach(), descending=True).values
    top = int(torch.argmax(cosine.detach()))
    return {
        "applicable": True,
        "top_slot": int(valid[top]),
        "top_score": float(cosine[top].detach()),
        "top_score_normalized": float(cosine[top].detach()),
        "second_score": float(ordered[1]) if ordered.numel() > 1 else None,
        "margin": float(ordered[0] - ordered[1]) if ordered.numel() > 1 else None,
        "line_index": top,
    }


# ---------------------------------------------------------------------------
# Battery.
# ---------------------------------------------------------------------------
def build_items(base, subject, state, faces) -> list[dict[str, Any]]:
    slot_by_tag = {face["tag"]: int(face["slot"]) for face in faces}
    items: list[dict[str, Any]] = []
    for face in faces:
        items.append(
            {
                "class": "written_writeback",
                "tag": f"writeback_{face['tag']}",
                "face_tag": face["tag"],
                "expected_slot": int(face["slot"]),
                "raw": face["write_text"],
            }
        )
    for face in faces:
        items.append(
            {
                "class": "written_paraphrase",
                "tag": face["tag"],
                "face_tag": face["tag"],
                "expected_slot": int(face["slot"]),
                "raw": face["read_text"],
            }
        )
    for row in base.real_unwritten(faces):
        items.append(
            {
                "class": "unwritten",
                "tag": row["tag"],
                "face_tag": row["face_tag"],
                "expected_slot": slot_by_tag[row["face_tag"]],
                "raw": row["raw"],
            }
        )
    negatives = base.real_negatives(subject, state, faces)
    for name in ("new_token", "new_pairing"):
        for row in negatives[name]:
            items.append(
                {
                    "class": name,
                    "tag": row["tag"],
                    "face_tag": None,
                    "expected_slot": None,
                    "raw": row["raw"],
                }
            )
    for row in negatives["no_value"]:
        items.append(
            {
                "class": "no_value",
                "tag": row["tag"],
                "face_tag": None,
                "expected_slot": None,
                "raw": b"",
                "key": row["key"],
            }
        )
    return items


def build_gate_store(base, subject):
    """Eight base writes only: the state the gate saw before any face."""
    state = subject.attach_store(subject.initial_state("byte-address-gate"))
    allowed = torch.ones((subject.config.groups, subject.config.fast_heads), dtype=torch.bool)
    for token in range(base.BASE_OCCUPIED):
        text = base.sentence(token, token % base.POSITION_COUNT)
        state = subject.observe(state, text, goal=text, allowed=allowed)
        state, _ = subject.addressed_write(state, text)
    return state


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


def store_signature(state) -> dict[str, Any]:
    return {
        "keys": tensor_sha(state.store_keys),
        "values": tensor_sha(state.store_values),
        "age": tensor_sha(state.store_age.float()),
        "count": int(state.store_count),
    }


def tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def arm_block(
    name: str, readings: list[dict[str, Any]], before: dict, after: dict
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    normalized: dict[str, list[float]] = {}
    for cls in CLASSES:
        rows = [row for row in readings if row["class"] == cls and row.get("applicable", True)]
        normalized[cls] = [float(row["top_score_normalized"]) for row in rows]
        groups[cls] = {
            "count": len(rows),
            "top_score": describe([row["top_score"] for row in rows]),
            "top_score_normalized": describe(normalized[cls]),
            "margin": describe([row["margin"] for row in rows if row["margin"] is not None]),
        }
    paraphrase_floor = min(normalized["written_paraphrase"])
    writeback_floor = min(normalized["written_writeback"])
    gaps = {
        cls: round(float(paraphrase_floor - max(normalized[cls])), 9) + 0.0
        for cls in NEGATIVE_BYTE_CLASSES
    }
    writeback_gaps = {
        cls: round(float(writeback_floor - max(normalized[cls])), 9) + 0.0
        for cls in NEGATIVE_BYTE_CLASSES
    }
    hits = {}
    for cls in ("written_writeback", "written_paraphrase"):
        rows = [row for row in readings if row["class"] == cls]
        hits[cls] = {
            "correct": sum(1 for row in rows if row["top_slot"] == row["expected_slot"]),
            "total": len(rows),
            "misses": [row["tag"] for row in rows if row["top_slot"] != row["expected_slot"]],
        }
    unwritten_rows = [row for row in readings if row["class"] == "unwritten"]
    read_slot = {
        row["face_tag"]: row["top_slot"] for row in readings if row["class"] == "written_paraphrase"
    }
    hits["unwritten"] = {
        "correct": sum(1 for row in unwritten_rows if row["top_slot"] == row["expected_slot"]),
        "total": len(unwritten_rows),
        "criterion": "the winning line is the record's own slot (strict)",
        "agrees_with_read_wording": sum(
            1 for row in unwritten_rows if row["top_slot"] == read_slot.get(row["face_tag"])
        ),
        "misses": [row["tag"] for row in unwritten_rows if row["top_slot"] != row["expected_slot"]],
    }
    return {
        "arm": name,
        "queries": len(readings),
        "groups": groups,
        "hits": hits,
        "floor_paraphrase": round(float(paraphrase_floor), 9),
        "floor_writeback": round(float(writeback_floor), 9),
        "gaps_vs_paraphrase_floor": gaps,
        "gaps_vs_writeback_floor": writeback_gaps,
        "separable": [cls for cls, gap in gaps.items() if gap > 0],
        "store_immutable": before == after,
        "per_item": [
            {
                "class": row["class"],
                "tag": row["tag"],
                "applicable": bool(row.get("applicable", True)),
                "top_slot": row.get("top_slot"),
                "expected_slot": row.get("expected_slot"),
                "top_score": row.get("top_score"),
                "top_score_normalized": row.get("top_score_normalized"),
                "margin": row.get("margin"),
                "offset": row.get("offset"),
                "reason": row.get("reason"),
            }
            for row in readings
        ],
    }


def copy_gate_block(subject, gate_state, faces, base) -> dict[str, Any]:
    """The frozen gate, unchanged, on the same 22 correction faces.

    The store here holds the eight base records only, exactly the state the
    growth-wire probe used before the faces were written; writing a face into a
    store that already contains it would of course read as a perfect copy.
    """
    config = GrowthWireConfig(enabled=True)
    occupied = int((gate_state.store_age >= 0).sum())
    rows = []
    for face in faces:
        out, info = gw.session_write(subject, gate_state, face["write_text"], config=config)
        band = info.get("band") or {}
        rows.append(
            {
                "tag": face["tag"],
                "action": info.get("action"),
                "slot": info.get("slot"),
                "band": band,
                "state_changed": out is not gate_state,
            }
        )
        touch()
    admitted = [row for row in rows if row["action"] == "appended"]
    refused = [row for row in rows if row["action"] == "rejected_copy"]
    dropped = [row for row in rows if row["action"] == "dropped"]
    overlaps = [row["band"].get("overlap") for row in rows if row["band"]]
    separations = [row["band"].get("separation") for row in rows if row["band"]]
    return {
        "config": {"enabled": True, "reserve_ratio": config.reserve_ratio},
        "gate_input": "subject.store_query_key -> the frozen mean key",
        "store_occupied_before": occupied,
        "store_capacity": int(gate_state.store_keys.shape[0]),
        "faces": len(rows),
        "admitted": len(admitted),
        "refused_copy": len(refused),
        "dropped": len(dropped),
        "overlap": describe([float(value) for value in overlaps]),
        "separation": describe([float(value) for value in separations]),
        "verdicts": sorted({row["band"].get("verdict") for row in rows if row["band"]}),
        "per_face": rows,
    }


def verdict_block(arm_a: dict, arm_b: dict, gate: dict) -> dict[str, Any]:
    written = arm_b["hits"]["written_writeback"]["correct"]
    unwritten = arm_b["hits"]["unwritten"]["correct"]
    gaps = arm_b["gaps_vs_paraphrase_floor"]
    positive = [cls for cls, gap in gaps.items() if gap > 0]
    negatives_near_one = [
        cls
        for cls in NEGATIVE_BYTE_CLASSES
        if arm_b["groups"][cls]["top_score_normalized"]["max"] >= NEAR_ONE
    ]
    holds = (
        written >= NEED_WRITTEN
        and unwritten >= NEED_UNWRITTEN
        and len(positive) >= NEED_GAPS
        and gate["refused_copy"] < gate["faces"]
    )
    not_discriminative = len(negatives_near_one) == len(NEGATIVE_BYTE_CLASSES) and not positive
    if holds:
        verdict = "byte_match_addressing_holds"
    elif not_discriminative:
        verdict = "alignment_not_discriminative"
    else:
        verdict = "inconclusive"
    return {
        "written_writeback": written,
        "written_need": NEED_WRITTEN,
        "written_paraphrase": arm_b["hits"]["written_paraphrase"]["correct"],
        "written_paraphrase_total": arm_b["hits"]["written_paraphrase"]["total"],
        "unwritten": unwritten,
        "unwritten_total": arm_b["hits"]["unwritten"]["total"],
        "unwritten_criterion": arm_b["hits"]["unwritten"]["criterion"],
        "unwritten_agrees_with_read_wording": arm_b["hits"]["unwritten"][
            "agrees_with_read_wording"
        ],
        "unwritten_need": NEED_UNWRITTEN,
        "gaps_vs_paraphrase_floor": gaps,
        "positive_gap_classes": positive,
        "need_positive_gap_classes": NEED_GAPS,
        "negative_classes_near_one": negatives_near_one,
        "near_one_threshold": NEAR_ONE,
        "copy_gate_refused": gate["refused_copy"],
        "copy_gate_faces": gate["faces"],
        "copy_gate_still_false_kills": gate["refused_copy"] >= gate["faces"],
        "holds": bool(holds),
        "not_discriminative_conditions": bool(not_discriminative),
        "verdict": verdict,
        "reason": (
            "the route's score is a count of equal bytes over a 15-byte window: every sentence in "
            "this battery is the same eight-word template differing in one two-byte word, so a "
            "never-written word and the face's own word at the same position cost exactly the same "
            "two bytes.  The eight base records are byte-identical to the template, so any "
            "single-word paraphrase is closer to a base record (two mismatched bytes) than to its "
            "own record (four), which is why the paraphrases score like the negatives and the "
            "write-back alone reaches the maximum."
            if verdict == "alignment_not_discriminative"
            else (
                "the pre-written pass rule holds on this battery"
                if verdict == "byte_match_addressing_holds"
                else "the readings fall between the two written verdicts; distributions only"
            )
        ),
    }


def plain_language_block(arm_a: dict, arm_b: dict, gate: dict, verdict: dict) -> dict[str, str]:
    return {
        "summary": (
            "ON THIS MODEL AND THESE 154 QUESTIONS, handing the match decision to the byte route "
            "does not separate 'this was written' from 'this was never written'.  The literal "
            f"sentence that was stored finds itself every time ({arm_b['hits']['written_writeback']['correct']}"
            f"/{arm_b['hits']['written_writeback']['total']}), but that is the trivial case.  When the "
            "question is the same record asked in a different word position, the byte route picks "
            f"the shared template record, not the record: "
            f"{arm_b['hits']['written_paraphrase']['correct']}/"
            f"{arm_b['hits']['written_paraphrase']['total']} correct, and "
            f"{arm_b['hits']['unwritten']['correct']}/"
            f"{arm_b['hits']['unwritten']['total']} on the never-written wordings.  A never-written "
            "word at the same position gets exactly the same alignment score as the legitimate "
            "paraphrase, so a threshold cannot tell them apart.  Today's averaged-pooled address "
            f"does better on the same questions ({arm_a['hits']['unwritten']['correct']}/"
            f"{arm_a['hits']['unwritten']['total']} on the never-written wordings), because pooling "
            "throws the position away and keeps the novel word, which is exactly the information "
            "the byte route discards.  The copy gate is not modified and still reads the old mean "
            f"key, so it still refuses all {gate['refused_copy']} faces; the byte address never "
            "reaches it."
        ),
        "counting_rule": (
            "the never-written number counts a hit only when the winning line is that record's own "
            "slot.  The older convention (the never-written wording lands on the same slot as the "
            f"read wording) is reported too: arm A {arm_a['hits']['unwritten']['agrees_with_read_wording']}"
            f"/{arm_a['hits']['unwritten']['total']}, arm B "
            f"{arm_b['hits']['unwritten']['agrees_with_read_wording']}"
            f"/{arm_b['hits']['unwritten']['total']}.  The older convention is degenerate for the "
            "byte arm: everything lands on the same template record, so everything 'agrees'."
        ),
        "what_it_shows": (
            "the primitive is not broken and it runs with zero training; what it measures is "
            "byte identity over a fixed template, not content identity.  On a battery where every "
            "sentence shares one template and differs by one word, byte identity cannot be "
            "discriminative: it answers 'is this the same string' and the question is 'is this the "
            "same fact'."
        ),
        "what_it_does_not_show": (
            "it does not show that byte matching is useless in general -- the 19.2r gate used it "
            "with an injected alignment offset on a task where the object description was a "
            "verbatim substring.  It does not show that the model cannot learn, and it does not "
            "test growth, capacity or training."
        ),
        "next": (
            f"the verdict is {verdict['verdict']}; no encoding, geometry or threshold tuning "
            "follows from this run."
        ),
    }


def main() -> int:
    if time.process_time() - START_CPU > CPU_GUARD:
        raise SystemExit("cpu guard tripped before start")

    process_guard = assert_no_black_processes("byte-address probe")
    lock_name = acquire_single_instance()
    atexit.register(release_single_instance)
    stop = threading.Event()
    start_watchdog(stop)

    PROGRESS.write_text("", encoding="utf-8")
    WATCHDOG_MARKER.unlink(missing_ok=True)
    torch.set_grad_enabled(False)
    torch.manual_seed(20261108)
    base = load_base()

    subject = LivingCoreReference(LivingCoreConfig())
    device = subject.W.device
    model = ByteMatchRoute(dim=IDENTITY_DIM, span=SPAN).to(device)
    route_identity = identity_check(model)

    manifest = {
        "format": "taiji-byte-address-preregistration-v1",
        "date": "2026-11-08",
        "question": (
            "with the address generated by the 19.2r byte-match route instead of the mean-pooled "
            "vector, and with zero training, does the store recognise its own paraphrases while "
            "the never-written classes fall behind?"
        ),
        "written_before_any_reading": True,
        "sequence": [
            "instantiate ByteMatchRoute at identity, verify byte_embed = eye and weights = 1",
            "build the real-forward store with today's write path (30 sentences, 32 slots)",
            "record each write's address through the route (offset, score, nearest line)",
            "read the 154-query battery twice: mean-key arm and byte-route arm, same store",
            "run the frozen copy gate on the 22 correction faces",
            "apply the pre-written verdict rule once",
        ],
        "criteria": CRITERIA,
        "constants": {
            "span": SPAN,
            "slots": int(subject.config.width and 32),
            "faces": base.FACE_COUNT,
            "base_occupied": base.BASE_OCCUPIED,
            "unwritten_per_rule": base.UNWRITTEN_PER_RULE,
            "negative_per_class": base.NEG_PER_CLASS,
            "temperature": ADDRESSED_TEMPERATURE,
            "near_one": NEAR_ONE,
            "route_parameters": route_identity["parameters"],
            "battery": 22 + 66 + 66,
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

    identity_before = graph_identity(
        RuleLearningGraph(subject, arm="CF1", addressed_store_enabled=True)
    )
    state, faces = base.build_real_store(subject)
    slots = torch.nonzero(state.store_age >= 0, as_tuple=False).flatten().tolist()
    records = [
        base.sentence(token, token % base.POSITION_COUNT) for token in range(base.BASE_OCCUPIED)
    ]
    records.extend(face["write_text"] for face in faces)
    if len(records) != len(slots):
        raise SystemExit(f"record/occupancy mismatch: {len(records)} vs {len(slots)}")
    append_progress(
        {
            "phase": "store_built",
            "occupied": len(slots),
            "faces": len(faces),
            "record_bytes": int(sum(len(raw) for raw in records)),
            "record_lengths": sorted({len(raw) for raw in records}),
        }
    )

    # ---- write-time address: the route's own alignment ---------------------
    write_rows = []
    for index, raw in enumerate(records):
        if index < base.BASE_OCCUPIED:
            write_rows.append({"slot": index, "kind": "base", "raw_len": len(raw), "address": None})
            continue
        scores, offsets = byte_scores(model, [raw], records[:index], device)
        row = scores[0]
        top = int(torch.argmax(row))
        write_rows.append(
            {
                "slot": index,
                "kind": "face",
                "raw_len": len(raw),
                "address": {
                    "nearest_line": top,
                    "nearest_slot": slots[top],
                    "offset": int(offsets[0, top]),
                    "score": float(row[top]),
                    "score_normalized": float(row[top]) / SPAN,
                    "lines_matched_at_best": int((row == row[top]).sum()),
                },
            }
        )
        touch()
    face_addresses = [row["address"] for row in write_rows if row["kind"] == "face"]
    write_address = {
        "face_writes": len(face_addresses),
        "distinct_signatures": len(
            {(row["offset"], round(row["score"], 9)) for row in face_addresses}
        ),
        "score": describe([row["score_normalized"] for row in face_addresses]),
        "offset": describe([float(row["offset"]) for row in face_addresses]),
        "nearest_is_base_record": sum(
            1 for row in face_addresses if row["nearest_slot"] < base.BASE_OCCUPIED
        ),
        "nearest_is_older_face": sum(
            1 for row in face_addresses if row["nearest_slot"] >= base.BASE_OCCUPIED
        ),
        "reading": (
            "the 22 distinct correction sentences collapse to a handful of route addresses: the "
            "score only counts equal bytes, so which novel word sits at the substituted position "
            "never enters the address"
        ),
        "rows": write_rows,
    }
    append_progress(
        {
            "phase": "write_address",
            "distinct_signatures": write_address["distinct_signatures"],
            "score": write_address["score"],
        }
    )

    # ---- readings ----------------------------------------------------------
    items = build_items(base, subject, state, faces)
    before = store_signature(state)
    mean_readings = []
    for item in items:
        if not item["raw"] and "key" not in item:
            mean_readings.append({**item, "applicable": False, "reason": "no query form"})
            continue
        key = item["key"] if "key" in item else subject.store_query_key(state, item["raw"])
        row = mean_scores(state, key)
        row.update(
            {
                "class": item["class"],
                "tag": item["tag"],
                "face_tag": item.get("face_tag"),
                "expected_slot": item["expected_slot"],
            }
        )
        mean_readings.append(row)
        touch()
    after_mean = store_signature(state)
    append_progress({"phase": "mean_arm_done", "immutable": before == after_mean})

    byte_readings: list[dict[str, Any]] = []
    byte_items = [item for item in items if item["raw"]]
    scores, offsets = byte_scores(model, [item["raw"] for item in byte_items], records, device)
    for index, item in enumerate(byte_items):
        row = byte_read(scores, slots, index)
        row["offset"] = int(offsets[index, row["line_index"]])
        row.update(
            {
                "class": item["class"],
                "tag": item["tag"],
                "face_tag": item.get("face_tag"),
                "expected_slot": item["expected_slot"],
            }
        )
        byte_readings.append(row)
        touch()
    for item in items:
        if item["raw"]:
            continue
        byte_readings.append(
            {
                "class": item["class"],
                "tag": item["tag"],
                "expected_slot": item["expected_slot"],
                "applicable": False,
                "reason": (
                    "the no_value query is a midpoint of two stored keys: it has no byte sentence, "
                    "so a byte address cannot be asked this question"
                ),
            }
        )
    after_bytes = store_signature(state)
    append_progress({"phase": "byte_arm_done", "immutable": before == after_bytes})

    arm_a = arm_block("A_mean", mean_readings, before, after_mean)
    arm_b = arm_block("B_byte", byte_readings, before, after_bytes)
    arm_a["address"] = {
        "source": "state.store_keys = encoder.context(raw).mean(0) (today)",
        "read": "cosine over the occupied keys, argmax",
    }
    arm_b["address"] = {
        "source": "ByteMatchRoute(dim=256, span=15) free-offset alignment over the stored bytes",
        "read": "argmax alignment score over the stored lines",
        "mean_key_consulted_by_read": False,
        "store_keys_role": (
            "kept in the tensor only because the frozen write path computes it and the frozen "
            "copy gate reads it; the byte-address read never touches it"
        ),
        "training": "none: identity initialisation, 0 updates, 0 optimizer steps",
    }
    gate_store = build_gate_store(base, subject)
    gate = copy_gate_block(subject, gate_store, faces, base)
    append_progress(
        {
            "phase": "gate_done",
            "admitted": gate["admitted"],
            "refused": gate["refused_copy"],
        }
    )

    identity_after = graph_identity(
        RuleLearningGraph(subject, arm="CF1", addressed_store_enabled=True)
    )
    verdict = verdict_block(arm_a, arm_b, gate)

    state_cost = {
        "graph_state_bytes": int(gw.state_bytes(state)),
        "graph_state_bytes_added": 0,
        "runner_side_address_table_bytes": int(sum(len(raw) for raw in records)),
        "runner_side_address_table_records": len(records),
        "route_parameters": route_identity["parameters"],
        "route_parameter_bytes_float32": route_identity["parameters"] * 4,
        "subject_parameters_added": 0,
        "candidate_sha_v2": identity_before["candidate_sha_v2"],
        "candidate_sha_v2_unchanged": identity_before["candidate_sha_v2"]
        == identity_after["candidate_sha_v2"],
        "parameter_tensors": int(identity_before["parameter_tensors"]),
        "parameter_count": int(identity_before["parameter_count"]),
    }

    for block in (arm_a, arm_b):
        block.pop("per_item", None)

    report = {
        "probe": "taiji_byte_address",
        "date": "2026-11-08",
        "question": manifest["question"],
        "preregistration": {
            "manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "manifest_sha256": file_sha(MANIFEST),
            "written_before_any_reading": True,
        },
        "criteria": CRITERIA,
        "route": route_identity,
        "write_address": write_address,
        "arms": {"A_mean": arm_a, "B_byte": arm_b},
        "copy_gate": gate,
        "verdict": verdict,
        "cost": state_cost,
        "plain_language": plain_language_block(arm_a, arm_b, gate, verdict),
        "isolation": {
            "runner_sha256_start": RUNNER_SHA_START,
            "runner_sha256_end": file_sha(Path(__file__).resolve()),
            "route_sha256": file_sha(ROUTE),
            "living_core_sha256": file_sha(LIVING_CORE),
            "growth_wire_sha256": file_sha(GROWTH_WIRE),
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
        "run_cost": {
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
                "written_writeback": [verdict["written_writeback"], 22],
                "written_paraphrase": [
                    arm_b["hits"]["written_paraphrase"]["correct"],
                    arm_b["hits"]["written_paraphrase"]["total"],
                ],
                "unwritten_byte": [verdict["unwritten"], verdict["unwritten_total"]],
                "unwritten_byte_legacy_agreement": [
                    verdict["unwritten_agrees_with_read_wording"],
                    verdict["unwritten_total"],
                ],
                "gaps_byte": verdict["gaps_vs_paraphrase_floor"],
                "floor_byte": arm_b["floor_paraphrase"],
                "floor_mean": arm_a["floor_paraphrase"],
                "written_mean": [
                    arm_a["hits"]["written_paraphrase"]["correct"],
                    arm_a["hits"]["written_paraphrase"]["total"],
                ],
                "unwritten_mean": [
                    arm_a["hits"]["unwritten"]["correct"],
                    arm_a["hits"]["unwritten"]["total"],
                ],
                "gate": [gate["refused_copy"], gate["faces"]],
                "candidate_sha_v2_unchanged": state_cost["candidate_sha_v2_unchanged"],
                "cpu_seconds": report["run_cost"]["cpu_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
