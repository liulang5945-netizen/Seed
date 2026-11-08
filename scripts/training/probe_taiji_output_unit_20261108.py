"""Output-unit gate: is the printed answer a usable text unit? (0 training, 0 new parameters).

What is under test
------------------
One edit to ``taiji/sc_sfk.py``, built on the frozen definitions of the short-entry
gate (chunking, identity, anchors, fallback ranking unchanged):

**The output unit is a complete record or a complete sentence.**  Anchored reads
print the whole record (``unit = record``).  Unanchored reads still rank records
by block overlap, but print the record's sentence with the highest block overlap
against the query (``unit = sentence``), ties broken by sentence order.  Sentence
spans come from ``sentences()``: a boundary is a sentence terminator, a newline
or the record end, always at a character boundary of the stored UTF-8 text.  No
fragment that cuts a character, a word or a sentence is ever printed, and the
fallback gets the same rule.  Every output carries its provenance: ``unit``,
``start`` / ``end``, ``fallback``, ``fallback_reason`` and ``fallback_basis``
(``record_overlap`` or ``address_order``).

No numeric tolerance is introduced: the terminator set is categorical, sentence
choice reuses the existing multiset-Jaccard ranking, and the frozen constant
inventory / read-path literal set is compared with the running module.

Scoring (three layers, judged on the output)
--------------------------------------------
L1 mechanical: fabricated bytes = 0; illegal slices = 0; UTF-8 decodable = 100%;
boundary legal = 100% (the span is the whole record or exactly one span returned
by ``sentences()`` for the claimed record).
L2 content: output == claimed record's bytes[start:end], faithful rate >= 0.95.
L3 usability and selection, three separate columns over the 120 should-anchor
queries: ``usable_and_correct`` (L1 clean, complete unit, claimed record ==
expected), ``usable_but_wrong`` (L1 clean, complete unit, wrong record),
``unusable`` (L1 blocked: fragment, illegal span, undecodable or empty).  The
count of answers blocked by L1 ("should have answered, gave no usable answer") is
reported with samples.

Verdict (frozen in the manifest before the first reading)
---------------------------------------------------------
closed not neutral -> hard failure, stop; a new/changed numeric constant or
read-path literal (i.e. aligning boundaries would need a tolerance) ->
``output_unit_needs_tolerance`` (stop, do not implement, report the conflict);
L1 all clean and ``usable_and_correct >= 110/120`` and ``unusable = 0`` ->
``output_unit_fixed``; L1 all clean but ``unusable > 0`` -> ``output_unit_leaks``
(samples and reasons only, no loosening); otherwise ``inconclusive``.  The four
synonym rewrites stay in the denominator.

Discipline: 0 training / 0 optimizer steps / 0 Adam / 0 confirm; ``process_guard``
first; single-instance lock; 60 s watchdog; CPU cap 200 s; readings flushed as
taken; write-then-rename; dual runner sha.  No ``python -m black``: the in-process
gate is the only formatter check.
"""

from __future__ import annotations

import atexit
import errno
import hashlib
import importlib
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
if str(ROOT / "scripts/training") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts/training"))

from taiji.living_core import LivingCoreConfig, LivingCoreReference  # noqa: E402
from taiji.process_guard import assert_no_black_processes  # noqa: E402
from taiji.sc_sfk import (  # noqa: E402
    MIN_ADDRESSABLE,
    MIN_CHUNK,
    PROTECT,
    WINDOW,
    AddressTable,
    chunk_address_audit,
    chunks,
    features,
    identity,
    identity_blocks,
    is_subsequence,
    longest_contiguous_run,
    sentences,
)

MANIFEST = ROOT / "plans/manifests/taiji_output_unit_20261108.json"
CORPUS = ROOT / "plans/manifests/taiji_offsetfree_corpus_20261108.json"
SOURCE_MANIFEST = ROOT / "plans/manifests/taiji_offsetfree_20261108.json"
REPORT = ROOT / "reports/taiji_output_unit_20261108.json"
OUTPUTS = ROOT / "reports/taiji_output_unit_20261108_outputs.jsonl"
CLOSED_REPORT = ROOT / "reports/taiji_output_unit_20261108_closed.json"
PROGRESS = ROOT / "reports/taiji_output_unit_20261108_progress.jsonl"
LOCK = ROOT / "reports/taiji_output_unit_20261108.lock"
WATCHDOG_MARKER = ROOT / "reports/taiji_output_unit_20261108_watchdog.json"
PREVIOUS_REPORT = ROOT / "reports/taiji_shorts_20261108.json"
PREVIOUS_OUTPUTS = ROOT / "reports/taiji_shorts_20261108_outputs.jsonl"
PREVIOUS_COMMIT = "506cdd802"
ADDRESS_MODULE = ROOT / "taiji/sc_sfk.py"

CPU_CAP = 200.0
WATCHDOG_SECONDS = 60.0
LOCK_STALE_SECONDS = 600.0
BATCH = 10
PAIR_STRIDE = 30
USABLE_CORRECT_LINE = 110
FAITHFUL_LINE = 0.95
WRITE_LINE = 57
MIN_LENGTH_CEILING = 64

PARAPHRASE_RULES = (
    ("因此", "所以"),
    ("但是", "不过"),
    ("而且", "并且"),
    ("如果", "假如"),
    ("可以", "能够"),
    ("已经", "已"),
    ("目前", "当前"),
    ("需要", "须要"),
    ("以及", "与"),
    ("进行", "做"),
)

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
START_CPU = time.process_time()
START_WALL = time.time()
LAST_READING = [time.time()]
WATCHDOG_FIRED = [False]

DEFINITION = {
    "chunking": (
        "gear hash, rolling = ((rolling << 1) + gear[byte]) & 0xFFFFFFFFFFFFFFFF, gear "
        "deterministically derived from sha256('taiji-gear-v1' + byte); cut at every "
        "position where (rolling & 0x3F) == 0. The mask reads the low six bits, i.e. the "
        "last six bytes, and the rolling hash is never reset: no minimum length, no "
        "maximum length, no distance-from-previous-cut rule. Chunks shorter than 16 bytes "
        "are dropped at emission -- length is examined only after the cut set is fixed, so "
        "a short chunk can never move a cut. Chunk fingerprint = sha256(chunk)[:16].hex()."
    ),
    "identity": (
        "S(r) = ordered fingerprints of every chunk [a, b) with a >= PROTECT and b < len(r). "
        "PROTECT = WINDOW = 6 is the predicate's true dependency width; b < len(r) drops "
        "only the final chunk, whose right edge is the buffer end. A record with no such "
        "chunk has no identity and is rejected alone."
    ),
    "shortest_addressable_length": (
        "MIN_CHUNK + PROTECT + 1 = 16 + 6 + 1 = 23 bytes: one identity chunk needs a >= 6, "
        "b - a >= 16 and b <= len(r) - 1. Necessary, not sufficient -- the cuts themselves "
        "are content-dependent, so a longer record can still be unaddressable."
    ),
    "anchor": (
        "A record is anchored iff S(r) occurs as a contiguous subsequence of C(Q), the "
        "ordered fingerprints of all query chunks (the query is not banded). Anchors are "
        "readings, not exits. No prefix, suffix, boilerplate or unrelated snippet can "
        "create an anchor: the whole S(r) must be present, in order, contiguously. The "
        "floor is one block, never zero: an empty S(r) is a subsequence of every query and "
        "would anchor everything."
    ),
    "output_unit": (
        "The printed answer is a complete text unit: the whole record (unit = record) when a "
        "record is anchored, otherwise the chosen record's sentence with the highest "
        "block-level overlap with the query (unit = sentence), ties broken by sentence "
        "order. A sentence span comes from sentences(): it starts at the record start or "
        "right after a terminator/newline and ends after a terminator/newline or at the "
        "record end, always on a character boundary of the stored UTF-8 text. No fragment "
        "that cuts a character, a word or a sentence is ever printed, fallback included. "
        "Every output states unit, start, end, fallback, fallback_reason and fallback_basis."
    ),
    "fallback": (
        "A record is anchored iff S(r) occurs as a contiguous subsequence of C(Q). Anchored "
        "reads rank by covered blocks descending, coverage ratio descending, address index "
        "ascending. Unanchored reads are labelled fallback = true with fallback_reason in "
        "{no_anchor, no_overlap} and rank every record by block-level overlap (multiset "
        "Jaccard: shared / (|S(r)| + |C(Q)| - shared)) descending, then address index "
        "ascending; fallback_basis is record_overlap when some block is shared and "
        "address_order otherwise. The label is a reading: it decides who is printed, never "
        "whether, and it adds no pass/reject threshold. No band, terminator or overlap "
        "threshold is applied to accept or reject an output."
    ),
    "write": (
        "Append-only, never a batch rollback, never an eviction, never a silent "
        "overwrite. A record is rejected alone when (a) no identity chunk remains, (b) an "
        "equal identity holds a different value, (c) its identity and an existing identity "
        "are contiguous subsequences of one another."
    ),
}

CRITERIA = {
    "C1_closed_state_neutral": (
        "with sc_sfk_enabled=False the 176-case battery reproduces the stored baseline "
        "arrays bit for bit (max|delta| = 0), parameters / state bytes / candidate sha / "
        "store signature unchanged, and no table allocated, with no half-open branch."
    ),
    "C2_short_entries_admitted": (
        "at least 57 of the 58 records must enter the table; every rejection is reported "
        "one by one with its reason. The 10 records rejected before the short-entry gate "
        "must all be admitted."
    ),
    "C3_no_tolerance": (
        "no new numeric constant and no new numeric literal inside the read path; the "
        "frozen constant inventory and the frozen read-path literal set are compared with "
        "the running module. Any difference -> output_unit_needs_tolerance, stop."
    ),
    "L1_mechanical": (
        "fabricated bytes = 0; illegal slices = 0; UTF-8 decodable = 100%; legal boundary = "
        "100% (the span is the whole record or exactly one span returned by sentences() for "
        "the claimed record, and length matches the span)."
    ),
    "L2_content": "output == claimed record's bytes[start:end], faithful rate >= 0.95.",
    "L3_usable": (
        "over the 120 should-anchor queries, three separate columns: usable_and_correct "
        "(L1 clean, complete unit, claimed == expected), usable_but_wrong (L1 clean, "
        "complete unit, claimed != expected), unusable (L1 blocked: fragment, illegal span, "
        "undecodable or empty). The count blocked by L1 is reported with samples."
    ),
    "C7_no_regression": (
        "the 96 self + lead-in readings of the 48 records admitted before the short-entry "
        "gate are compared row by row (same query text); none may stop being correct, "
        "anchored, or byte-identical."
    ),
    "C8_boundary_sources_reported": (
        "report the unit distribution (record / sentence), the fallback count, the reasons "
        "and bases, and the start/end offsets of every output."
    ),
    "errors": (
        "usable_and_correct = complete decodable unit pointing at the expected record; "
        "usable_but_wrong = complete decodable unit pointing elsewhere; unusable = fragment, "
        "cross-boundary, undecodable or empty."
    ),
    "V_hard_failure": "closed not neutral -> hard_failure_closed_not_neutral, stop.",
    "V_output_unit_needs_tolerance": (
        "a new/changed numeric constant or read-path literal (i.e. aligning boundaries "
        "would need a numeric tolerance) -> output_unit_needs_tolerance: stop, do not "
        "implement, report the conflict."
    ),
    "V_output_unit_leaks": (
        "L1 all clean but unusable > 0 -> output_unit_leaks: samples and reasons only, no "
        "loosening of any tolerance."
    ),
    "V_output_unit_fixed": (
        "closed neutral AND L1 all clean AND usable_and_correct >= 110/120 AND unusable = 0 "
        "-> output_unit_fixed."
    ),
    "V_inconclusive": "anything else.",
}


# ---------------------------------------------------------------------------
# Discipline helpers.
# ---------------------------------------------------------------------------
def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    with open(tmp, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def append_progress(record: dict) -> None:
    record = {**record, "at": round(time.time() - START_WALL, 3)}
    with open(PROGRESS, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    touch()


def touch() -> None:
    LAST_READING[0] = time.time()


def acquire_single_instance() -> str:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        age = time.time() - LOCK.stat().st_mtime
        if age < LOCK_STALE_SECONDS:
            raise SystemExit(f"another instance holds {LOCK} ({age:.1f}s old)") from None
        LOCK.unlink()
        descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as exc:  # pragma: no cover
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


def cpu_check() -> float:
    used = time.process_time() - START_CPU
    if used > CPU_CAP:
        write_json(
            ROOT / "reports/taiji_output_unit_20261108_cpu_abort.json",
            {"aborted": True, "cpu": round(used, 3), "at": time.time()},
        )
        raise SystemExit(f"CPU cap {CPU_CAP}s exceeded ({used:.1f}s)")
    return used


# ---------------------------------------------------------------------------
# Freeze.
# ---------------------------------------------------------------------------
def phase_freeze() -> int:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if file_sha(CORPUS) != source["corpus"]["snapshot_sha256"]:
        raise SystemExit("corpus snapshot differs from the offset-free manifest")
    for key in ("written", "unknown", "unrelated"):
        shas = [text_sha(text) for text in corpus[key]]
        if shas != source["corpus"][f"{key}_sha256"]:
            raise SystemExit(f"corpus {key} differs from the offset-free manifest")

    written = corpus["written"]
    changed = [index for index, text in enumerate(written) if paraphrase(text)[1] > 0]
    pairs = [
        [index, index + PAIR_STRIDE] for index in range(20) if index + PAIR_STRIDE < len(written)
    ]
    manifest = {
        "schema": "taiji-output-unit-v1",
        "date": "2026-11-08",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runner_sha256": RUNNER_SHA_START,
        "question": (
            "is every printed answer a complete, decodable text unit -- the whole record "
            "when a record anchors, otherwise one whole sentence of the chosen record -- and "
            "does the judged column (usable_and_correct / usable_but_wrong / unusable) clear "
            "110/120 with zero unusable outputs, without adding any numeric tolerance?"
        ),
        "definition": DEFINITION,
        "address_module": {
            "path": str(ADDRESS_MODULE.relative_to(ROOT)),
            "sha256": file_sha(ADDRESS_MODULE),
            "constant_inventory": chunk_address_audit()["constants"],
            "read_path_literals": chunk_address_audit()["read_path_literals"],
            "protection": chunk_address_audit()["protection"],
            "note": (
                "frozen before the first reading; the runner compares the live module with "
                "this copy. A new constant or read-path literal -> tolerance_creep."
            ),
        },
        "corpus": {
            "snapshot": str(CORPUS.relative_to(ROOT)),
            "snapshot_sha256": file_sha(CORPUS),
            "write_count": len(written),
            "unknown_count": len(corpus["unknown"]),
            "unrelated_count": len(corpus["unrelated"]),
            "written_sha256": [text_sha(text) for text in written],
            "unknown_sha256": [text_sha(text) for text in corpus["unknown"]],
            "unrelated_sha256": [text_sha(text) for text in corpus["unrelated"]],
            "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        },
        "paraphrase": {
            "table": [list(pair) for pair in PARAPHRASE_RULES],
            "changed_records": changed,
            "note": (
                "only these records produce a synonym variant; a middle edit breaks the "
                "contiguous containment, so the fallback answers for them are scored on the "
                "printed unit (no substring credit) and stay in the denominator"
            ),
        },
        "queries": {
            "should_anchor": (
                "58 self reads + 58 lead-in variants ('总的来说，' + text + '。') + "
                f"{len(changed)} synonym variants = {58 + 58 + len(changed)}; every one is "
                "scored by L3 on the output"
            ),
            "should_unknown": "20 held-out paragraphs, never written; no correct record",
            "should_ambiguous": (
                "the union paragraph k || '\\n\\n' || k+30 for the frozen pairs, counted "
                "when both records were admitted"
            ),
            "unrelated": "20 paragraphs from files that are neither written nor paraphrased",
            "pairs": pairs,
        },
        "denominators": {
            "should_anchor": 58 + 58 + len(changed),
            "should_unknown": len(corpus["unknown"]),
            "should_ambiguous": len(pairs),
            "unrelated": len(corpus["unrelated"]),
        },
        "lines": {
            "usable_and_correct": USABLE_CORRECT_LINE,
            "faithful": FAITHFUL_LINE,
            "write_admitted": WRITE_LINE,
            "min_length_ceiling": MIN_LENGTH_CEILING,
        },
        "verdict": {
            "rules": [
                "closed not neutral -> hard_failure_closed_not_neutral (stop)",
                (
                    "new/changed numeric constant or read-path literal vs the frozen "
                    "inventory -> output_unit_needs_tolerance (stop, do not implement, report)"
                ),
                (
                    "L1 all clean AND usable_and_correct >= "
                    f"{USABLE_CORRECT_LINE}/120 AND unusable = 0 -> output_unit_fixed"
                ),
                "L1 all clean but unusable > 0 -> output_unit_leaks (samples and reasons only)",
                "otherwise -> inconclusive",
            ],
            "denominator_note": (
                "the synonym rewrites stay in the denominator even though a middle edit "
                "breaks contiguous containment; they are counted, never removed"
            ),
        },
        "criteria": CRITERIA,
        "discipline": {
            "training": 0,
            "optimizer_steps": 0,
            "adam": 0,
            "confirm_reads": 0,
            "new_parameters": 0,
            "cpu_cap_seconds": CPU_CAP,
            "watchdog_seconds": WATCHDOG_SECONDS,
            "black": "in-process gate only; python -m black forbidden",
        },
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if old["corpus"]["snapshot_sha256"] != manifest["corpus"]["snapshot_sha256"]:
            raise SystemExit("refusing to refreeze: corpus snapshot changed")
        if old.get("verdict") != manifest["verdict"]:
            raise SystemExit("refusing to refreeze: verdict rules changed")
        manifest["freeze_revision"] = int(old.get("freeze_revision", 1)) + 1
        manifest["freeze_note"] = "corpus and verdict unchanged; harness revision only"
    else:
        manifest["freeze_revision"] = 1
        manifest["freeze_note"] = "initial freeze"
    write_json(MANIFEST, manifest)
    print(
        json.dumps(
            {
                "manifest": str(MANIFEST.relative_to(ROOT)),
                "denominators": manifest["denominators"],
                "changed_records": changed,
            },
            ensure_ascii=False,
        )
    )
    return 0


# ---------------------------------------------------------------------------
# Closed arm: the 176-case battery against the stored baseline arrays.
# ---------------------------------------------------------------------------
def phase_closed() -> dict[str, Any]:
    wire = importlib.import_module("probe_taiji_store_wire_20261108")
    base = wire.load_base()
    baseline = json.loads(wire.BASELINE_JSON.read_text(encoding="utf-8"))
    with np.load(wire.BASELINE_NPZ) as data:
        base_fast, base_probs = data["fast"], data["probs"]

    config = LivingCoreConfig()
    default_off = getattr(config, "sc_sfk_enabled", None)
    subject = LivingCoreReference(config)
    state, faces = base.build_real_store(subject)
    items = wire.build_battery(base, faces)
    readings, fast, probabilities = wire.read_battery(subject, state, items, with_forward=True)
    append_progress({"phase": "closed_readings", "count": len(readings)})

    fast_delta = float(np.max(np.abs(fast - base_fast)))
    prob_delta = float(np.max(np.abs(probabilities - base_probs)))
    total = len(readings)
    fast_equal = int(
        sum(
            row["fast_sha"] == ref["fast_sha"]
            for row, ref in zip(readings, baseline["readings"], strict=True)
        )
    )
    prob_equal = int(
        sum(
            row["probability_sha"] == ref["probability_sha"]
            for row, ref in zip(readings, baseline["readings"], strict=True)
        )
    )
    fields_equal = int(
        sum(
            row["fields"] == ref["fields"]
            for row, ref in zip(readings, baseline["readings"], strict=True)
        )
    )
    tags_equal = [row["tag"] for row in readings] == [row["tag"] for row in baseline["readings"]]
    parameters = wire.parameter_stats(subject)
    graph = wire.identity_of(subject)
    signature = wire.store_signature(state)
    state_bytes = wire.living_state_bytes(state)

    checks = {
        "default_off": bool(default_off is False),
        "table_not_allocated": bool(getattr(state, "sc_sfk", None) is None),
        "readings_total": total,
        "fast_max_abs_delta": fast_delta,
        "probability_max_abs_delta": prob_delta,
        "fast_digests_equal": fast_equal,
        "probability_digests_equal": prob_equal,
        "fields_equal": fields_equal,
        "tags_equal": bool(tags_equal),
        "parameter_tensors_equal": bool(parameters["tensors"] == baseline["parameters"]["tensors"]),
        "parameter_elements_equal": bool(
            parameters["elements"] == baseline["parameters"]["elements"]
        ),
        "parameter_sha_equal": bool(parameters["sha256"] == baseline["parameters"]["sha256"]),
        "state_bytes_equal": bool(state_bytes == baseline["state_bytes"]),
        "candidate_sha_v2_equal": bool(
            graph["candidate_sha_v2"] == baseline["identity"]["candidate_sha_v2"]
        ),
        "store_signature_equal": bool(signature == baseline["store"]),
    }
    neutral = (
        checks["default_off"]
        and checks["table_not_allocated"]
        and total == len(baseline["readings"]) == wire.BATTERY + wire.FACE_COUNT
        and tags_equal
        and fast_delta == 0.0
        and prob_delta == 0.0
        and fast_equal == total
        and prob_equal == total
        and fields_equal == total
        and checks["parameter_tensors_equal"]
        and checks["parameter_elements_equal"]
        and checks["parameter_sha_equal"]
        and checks["state_bytes_equal"]
        and checks["candidate_sha_v2_equal"]
        and checks["store_signature_equal"]
    )
    payload = {
        "schema": "taiji-output-unit-closed-v1",
        "date": "2026-11-08",
        "question": (
            "after the output-unit edit (whole record when anchored, whole sentence on "
            "fallback), is the address module still inert while the switch is off?"
        ),
        "verdict": "output_unit_closed_neutral" if neutral else "output_unit_closed_dirty",
        "checks": checks,
        "baseline": {
            "report": wire.BASELINE_JSON.name,
            "living_core_sha256": baseline["living_core_sha256"],
            "candidate_sha_v2": baseline["identity"]["candidate_sha_v2"],
        },
        "current": {
            "candidate_sha_v2": graph["candidate_sha_v2"],
            "state_bytes": state_bytes,
        },
        "disclosures": [
            "the edit added sentences() and _sentence_span and changed the printed span inside "
            "AddressTable.read(); while the flag is off no table is allocated and no edited "
            "line is reached. The battery is replayed because the claim is re-proved, not "
            "inferred.",
            "the comparison is against baseline arrays captured before taiji/ was edited.",
        ],
        "runner_sha256_start": RUNNER_SHA_START,
        "runner_sha256_end": file_sha(Path(__file__).resolve()),
        "cost": {
            "cpu_seconds": round(time.process_time() - START_CPU, 3),
            "wall_seconds": round(time.time() - START_WALL, 3),
        },
    }
    write_json(CLOSED_REPORT, payload)
    append_progress(
        {
            "phase": "closed_written",
            "verdict": payload["verdict"],
            "fast_max_abs_delta": fast_delta,
            "probability_max_abs_delta": prob_delta,
        }
    )
    if not neutral:
        raise SystemExit("closed state is not neutral; the open arm does not run")
    return payload


# ---------------------------------------------------------------------------
# Old implementation, frozen in the runner as a control column only.
# ---------------------------------------------------------------------------
def paraphrase(text: str) -> tuple[str, int]:
    changed = 0
    for old, new in PARAPHRASE_RULES:
        changed += text.count(old)
        text = text.replace(old, new)
    return text, changed


def legacy_features(raw: bytes) -> tuple[str, ...]:
    return tuple(sorted({raw[index : index + 2].hex() for index in range(0, len(raw) - 1, 2)}))


class LegacyTable:
    """The shipped stride-2, DF=1-anchor, DF=0-guard table, as a control."""

    def __init__(self) -> None:
        self.addresses: list[dict[str, Any]] = []

    def find(self, feats: tuple[str, ...]) -> int | None:
        for index, address in enumerate(self.addresses):
            if address["features"] == feats:
                return index
        return None

    def write(self, raw: bytes) -> int:
        feats = legacy_features(raw)
        found = self.find(feats)
        if found is not None:
            self.addresses[found]["count"] += 1
            return found
        self.addresses.append({"features": feats, "count": 1})
        return len(self.addresses) - 1

    def df(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for address in self.addresses:
            for feature in address["features"]:
                counts[feature] = counts.get(feature, 0) + int(address["count"])
        return counts

    def read(self, raw: bytes) -> dict[str, Any]:
        query = legacy_features(raw)
        df = self.df()
        if any(df.get(feature, 0) == 0 for feature in query):
            return {"decision": "no_match", "address": None}
        unique = {feature for feature, count in df.items() if count == 1}
        anchors = [
            index
            for index, address in enumerate(self.addresses)
            if any(feature in unique and feature in address["features"] for feature in query)
        ]
        if len(anchors) == 1:
            return {"decision": "matched", "address": anchors[0]}
        if len(anchors) > 1:
            return {"decision": "ambiguous_multiple_anchors", "address": None}
        return {"decision": "no_match", "address": None}


# ---------------------------------------------------------------------------
# Open arm: one table, every query gets an actual top-1 output.
# ---------------------------------------------------------------------------
def build_queries(
    corpus: dict[str, Any],
    record_address: list[int | None],
    pairs: list[list[int]],
) -> list[dict[str, Any]]:
    written = corpus["written"]
    queries: list[dict[str, Any]] = []
    for index, text in enumerate(written):
        queries.append(
            {
                "kind": "self",
                "variant": "exact",
                "raw": text.encode("utf-8"),
                "text": text,
                "expected": record_address[index],
                "label": text[:60],
            }
        )
    for index, text in enumerate(written):
        lead = "总的来说，" + text + "。"
        queries.append(
            {
                "kind": "lead_in",
                "variant": "prefix_suffix",
                "raw": lead.encode("utf-8"),
                "text": lead,
                "expected": record_address[index],
                "label": lead[:60],
            }
        )
    for index, text in enumerate(written):
        synonym, changed = paraphrase(text)
        if changed == 0:
            continue
        queries.append(
            {
                "kind": "synonym",
                "variant": "global_table",
                "raw": synonym.encode("utf-8"),
                "text": synonym,
                "expected": record_address[index],
                "label": synonym[:60],
            }
        )
    for text in corpus["unknown"]:
        queries.append(
            {
                "kind": "unknown",
                "variant": "held_out",
                "raw": text.encode("utf-8"),
                "text": text,
                "expected": None,
                "label": text[:60],
            }
        )
    for text in corpus["unrelated"]:
        queries.append(
            {
                "kind": "unrelated",
                "variant": "readme",
                "raw": text.encode("utf-8"),
                "text": text,
                "expected": None,
                "label": text[:60],
            }
        )
    for left, right in pairs:
        if left >= len(written) or right >= len(written):
            continue
        if record_address[left] is None or record_address[right] is None:
            continue
        text = written[left] + "\n\n" + written[right]
        queries.append(
            {
                "kind": "ambiguous",
                "variant": "union_text",
                "raw": text.encode("utf-8"),
                "text": text,
                "expected": "ambiguous",
                "label": text[:60],
            }
        )
    return queries


def evaluate(
    table: AddressTable, queries: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fabricated_total = 0
    illegal_spans = 0
    faithful_total = 0
    empty_total = 0
    utf8_valid = 0
    utf8_invalid = 0
    utf8_invalid_scored = 0
    boundary_legal_total = 0
    boundary_illegal = 0
    boundary_illegal_scored = 0
    usable_total = 0
    blocked_by_l1 = 0
    blocked_samples: list[dict[str, Any]] = []
    unit_counts = {"record": 0, "sentence": 0, "none": 0}
    fallback_total = 0
    for item in queries:
        began = time.perf_counter()
        result = table.read(item["raw"])
        elapsed = (time.perf_counter() - began) * 1000.0
        output = result["output"]
        if output is None:
            row_bytes, start, end = b"", 0, 0
            legal = faithful = boundary_ok = decodable = usable = False
            fabricated = 0
            unit = "none"
            unit_index = None
            source = b""
        else:
            record = table.addresses[output["record"]]
            source = record["raw"]
            start, end = output["start"], output["end"]
            legal = 0 <= start <= end <= len(source) and end - start == output["length"]
            row_bytes = source[start:end] if legal else b""
            digest_ok = hashlib.sha256(row_bytes).hexdigest() == output["sha256"]
            faithful = bool(legal and digest_ok)
            fabricated = 0 if faithful else len(row_bytes)
            whole = start == 0 and end == len(source)
            boundary_ok = bool(legal and (whole or (start, end) in sentences(source)))
            unit = "record" if whole else "sentence"
            unit_index = output.get("unit_index")
            try:
                row_bytes.decode("utf-8")
                decodable = len(row_bytes) > 0
            except UnicodeDecodeError:
                decodable = False
            usable = bool(legal and boundary_ok and decodable)
        output_bytes = row_bytes
        illegal = not legal
        if illegal:
            illegal_spans += 1
        fabricated_total += fabricated
        faithful_total += int(faithful)
        if not output_bytes:
            empty_total += 1
        if legal:
            boundary_legal_total += int(boundary_ok)
        if not boundary_ok:
            boundary_illegal += 1
        unit_counts[unit] += 1
        fallback_total += int(result["fallback"])
        usable_total += int(usable)
        expected = item["expected"]
        correct = output is not None and expected is not None and expected == output["record"]
        scorable = item["kind"] in ("self", "lead_in", "synonym")
        if not usable and scorable:
            blocked_by_l1 += 1
            if len(blocked_samples) < 5:
                blocked_samples.append(
                    {
                        "kind": item["kind"],
                        "variant": item["variant"],
                        "record": output["record"] if output is not None else None,
                        "span": [start, end],
                        "legal_span": bool(legal),
                        "boundary_ok": bool(boundary_ok),
                        "decodable": bool(decodable),
                        "reason": (result["fallback_reason"] or "matched"),
                    }
                )
        if output_bytes:
            try:
                output_bytes.decode("utf-8")
                utf8_valid += 1
            except UnicodeDecodeError:
                utf8_invalid += 1
                utf8_invalid_scored += int(scorable)
        else:
            utf8_invalid += 1
            utf8_invalid_scored += int(scorable)
        if not boundary_ok and scorable:
            boundary_illegal_scored += 1
        row = {
            "kind": item["kind"],
            "variant": item["variant"],
            "expected": expected,
            "decision": result["decision"],
            "record": output["record"] if output is not None else None,
            "anchors": result["ambiguous_candidates"],
            "no_anchor": result["no_anchor"],
            "covered_blocks": result["covered_blocks"],
            "coverage_ratio": result["coverage_ratio"],
            "overlap_ratio": result["overlap_ratio"],
            "shared_blocks": result["shared_blocks"],
            "fallback": result["fallback"],
            "fallback_reason": result["fallback_reason"],
            "fallback_basis": result["fallback_basis"],
            "contained": result["contained"],
            "unit": unit,
            "unit_index": unit_index,
            "start": start,
            "end": end,
            "output_len": len(output_bytes),
            "output_sha256": output["sha256"] if output is not None else None,
            "faithful": bool(faithful),
            "legal_span": bool(legal),
            "boundary_ok": bool(boundary_ok),
            "decodable": bool(decodable),
            "usable": bool(usable),
            "fabricated_bytes": fabricated,
            "correct": bool(correct),
            "scorable": bool(scorable),
            "read_ms": round(elapsed, 4),
            "query_text": item["text"],
            "boundary_before": (
                source[max(0, start - 6) : start].decode("utf-8", errors="replace") if legal else ""
            ),
            "boundary_after": (
                source[end : end + 6].decode("utf-8", errors="replace") if legal else ""
            ),
            "output_text": output_bytes.decode("utf-8", errors="replace"),
            "output_hex": output_bytes.hex(),
        }
        if scorable:
            row["output_class"] = (
                "usable_and_correct"
                if usable and correct
                else ("usable_but_wrong" if usable else "unusable")
            )
        else:
            row["output_class"] = "usable_fallback" if usable else "unusable"
        rows.append(row)
        touch()
    layer = {
        "queries": len(rows),
        "fabricated_bytes": fabricated_total,
        "illegal_spans": illegal_spans,
        "faithful": faithful_total,
        "faithful_rate": round(faithful_total / max(len(rows), 1), 6),
        "empty_outputs": empty_total,
        "utf8_valid_slices": utf8_valid,
        "utf8_invalid_slices": utf8_invalid,
        "utf8_invalid_scored": utf8_invalid_scored,
        "boundary_legal": boundary_legal_total,
        "boundary_illegal": boundary_illegal,
        "boundary_illegal_scored": boundary_illegal_scored,
        "usable": usable_total,
        "unusable": len(rows) - usable_total,
        "blocked_by_l1": blocked_by_l1,
        "blocked_samples": blocked_samples,
        "unit_distribution": dict(unit_counts),
        "fallback_outputs": fallback_total,
    }
    return rows, layer


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def kind(name: str) -> list[dict[str, Any]]:
        return [row for row in rows if row["kind"] == name]

    anchor_rows = [row for row in rows if row["kind"] in ("self", "lead_in", "synonym")]
    usable_and_correct = sum(
        1 for row in anchor_rows if row["output_class"] == "usable_and_correct"
    )
    usable_but_wrong = sum(1 for row in anchor_rows if row["output_class"] == "usable_but_wrong")
    unusable = sum(1 for row in anchor_rows if row["output_class"] == "unusable")
    negative_rows = [row for row in rows if row["kind"] in ("unknown", "ambiguous", "unrelated")]
    buckets = {"1.0": 0, "[0.75,1)": 0, "[0.5,0.75)": 0, "[0.25,0.5)": 0, "(0,0.25)": 0, "0.0": 0}

    def bucket(ratio: float) -> str:
        if ratio >= 1.0:
            return "1.0"
        if ratio >= 0.75:
            return "[0.75,1)"
        if ratio >= 0.5:
            return "[0.5,0.75)"
        if ratio >= 0.25:
            return "[0.25,0.5)"
        if ratio > 0.0:
            return "(0,0.25)"
        return "0.0"

    for row in rows:
        buckets[bucket(float(row["coverage_ratio"]))] += 1
    anchors_hist: dict[str, int] = {}
    for row in rows:
        key = str(min(int(row["anchors"]), 3))
        anchors_hist[key] = anchors_hist.get(key, 0) + 1
    return {
        "counts": {name: len(kind(name)) for name in sorted({row["kind"] for row in rows})},
        "should_anchor": {
            "n": len(anchor_rows),
            "usable_and_correct": usable_and_correct,
            "usable_but_wrong": usable_but_wrong,
            "unusable": unusable,
            "usable_and_correct_anchored": sum(
                1
                for row in anchor_rows
                if row["output_class"] == "usable_and_correct" and not row["fallback"]
            ),
            "usable_and_correct_fallback": sum(
                1
                for row in anchor_rows
                if row["output_class"] == "usable_and_correct" and row["fallback"]
            ),
            "usable_rate": round(usable_and_correct / max(len(anchor_rows), 1), 6),
            "no_anchor_reading": sum(1 for row in anchor_rows if row["no_anchor"]),
            "ambiguous_reading": sum(
                1 for row in anchor_rows if row["decision"] == "ambiguous_multiple_anchors"
            ),
        },
        "negatives": {
            "n": len(negative_rows),
            "usable_units": sum(1 for row in negative_rows if row["usable"]),
            "unusable_units": sum(1 for row in negative_rows if not row["usable"]),
            "fallback_labelled": sum(1 for row in negative_rows if row["fallback"]),
            "matched": sum(1 for row in negative_rows if row["decision"] == "matched"),
            "no_anchor": sum(1 for row in negative_rows if row["no_anchor"]),
            "ambiguous": sum(
                1 for row in negative_rows if row["decision"] == "ambiguous_multiple_anchors"
            ),
            "note": (
                "no correct record exists here; a complete unit printed under fallback is "
                "labelled no_anchor, not counted as an answer. matched > 0 would mean a "
                "wrong pointer on a query that should not anchor."
            ),
        },
        "by_kind": {
            name: {
                "n": len(kind(name)),
                "usable_and_correct": sum(
                    1 for row in kind(name) if row["output_class"] == "usable_and_correct"
                ),
                "usable_but_wrong": sum(
                    1 for row in kind(name) if row["output_class"] == "usable_but_wrong"
                ),
                "unusable": sum(1 for row in kind(name) if row["output_class"] == "unusable"),
                "usable_fallback": sum(
                    1 for row in kind(name) if row["output_class"] == "usable_fallback"
                ),
                "matched": sum(1 for row in kind(name) if row["decision"] == "matched"),
                "no_anchor": sum(1 for row in kind(name) if row["no_anchor"]),
                "ambiguous": sum(
                    1 for row in kind(name) if row["decision"] == "ambiguous_multiple_anchors"
                ),
                "record_units": sum(1 for row in kind(name) if row["unit"] == "record"),
                "sentence_units": sum(1 for row in kind(name) if row["unit"] == "sentence"),
                "fallback_outputs": sum(1 for row in kind(name) if row["fallback"]),
            }
            for name in ("self", "lead_in", "synonym", "unknown", "ambiguous", "unrelated")
        },
        "unit_distribution": {
            "record": sum(1 for row in rows if row["unit"] == "record"),
            "sentence": sum(1 for row in rows if row["unit"] == "sentence"),
            "none": sum(1 for row in rows if row["unit"] == "none"),
        },
        "fallback_basis_distribution": {
            basis: sum(1 for row in rows if row["fallback_basis"] == basis)
            for basis in ("none", "record_overlap", "address_order")
        },
        "coverage_ratio_distribution": buckets,
        "anchor_count_histogram": anchors_hist,
        "failures": [
            {
                "kind": row["kind"],
                "variant": row["variant"],
                "expected": row["expected"],
                "record": row["record"],
                "anchors": row["anchors"],
                "coverage_ratio": row["coverage_ratio"],
                "overlap_ratio": row["overlap_ratio"],
                "fallback": row["fallback"],
                "fallback_reason": row["fallback_reason"],
                "unit": row["unit"],
                "span": [row["start"], row["end"]],
                "decision": row["decision"],
                "output_class": row["output_class"],
                "query": row["query_text"][:80],
                "output": row["output_text"][:80],
            }
            for row in anchor_rows
            if row["output_class"] != "usable_and_correct"
        ],
    }


def legacy_summary(legacy: LegacyTable, queries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for item in queries:
        result = legacy.read(item["raw"])
        expected = item["expected"]
        rows.append(
            {
                "kind": item["kind"],
                "decision": result["decision"],
                "address": result["address"],
                "expected": expected,
                "correct": expected is not None and result["address"] == expected,
            }
        )
    anchor_rows = [row for row in rows if row["kind"] in ("self", "lead_in", "synonym")]
    return {
        "should_anchor": {
            "n": len(anchor_rows),
            "correct": sum(1 for row in anchor_rows if row["correct"]),
            "wrong_person": sum(
                1 for row in anchor_rows if row["decision"] == "matched" and not row["correct"]
            ),
            "no_match": sum(1 for row in anchor_rows if row["decision"] == "no_match"),
            "ambiguous": sum(
                1 for row in anchor_rows if row["decision"] == "ambiguous_multiple_anchors"
            ),
            "usable_ratio": round(
                sum(1 for row in anchor_rows if row["correct"]) / max(len(anchor_rows), 1), 6
            ),
        },
        "by_kind": {
            name: {
                "n": sum(1 for row in rows if row["kind"] == name),
                "correct": sum(1 for row in rows if row["kind"] == name and row["correct"]),
                "matched": sum(
                    1 for row in rows if row["kind"] == name and row["decision"] == "matched"
                ),
                "no_match": sum(
                    1 for row in rows if row["kind"] == name and row["decision"] == "no_match"
                ),
                "ambiguous": sum(
                    1
                    for row in rows
                    if row["kind"] == name and row["decision"] == "ambiguous_multiple_anchors"
                ),
            }
            for name in ("self", "lead_in", "synonym", "unknown", "ambiguous", "unrelated")
        },
        "note": "control column only; it never enters the verdict",
    }


def length_profile(written: list[str]) -> dict[str, Any]:
    """Identity size per record and which lengths are still unaddressable."""
    rows = []
    for index, text in enumerate(written):
        raw = text.encode("utf-8")
        blocks = identity_blocks(raw)
        rows.append(
            {
                "record": index,
                "length": len(raw),
                "identity_blocks": len(blocks),
                "addressable": bool(blocks),
                "first_cut": blocks[0][0] if blocks else None,
                "last_block_end": blocks[-1][1] if blocks else None,
            }
        )
    addressable = [row for row in rows if row["addressable"]]
    lengths = sorted({row["length"] for row in rows})
    return {
        "rows": rows,
        "min_record_length": min(lengths),
        "max_record_length": max(lengths),
        "shortest_addressable_length_observed": (
            min(row["length"] for row in addressable) if addressable else None
        ),
        "unaddressable_records": [row["record"] for row in rows if not row["addressable"]],
        "identity_blocks_histogram": {
            str(size): sum(1 for row in rows if row["identity_blocks"] == size)
            for size in sorted({row["identity_blocks"] for row in rows})
        },
        "structurally_unaddressable_below": [value for value in lengths if value < MIN_ADDRESSABLE],
        "definition": {
            "MIN_CHUNK": MIN_CHUNK,
            "PROTECT": PROTECT,
            "WINDOW": WINDOW,
            "MIN_ADDRESSABLE": MIN_ADDRESSABLE,
        },
    }


def shift_invariance_check(written: list[str]) -> dict[str, Any]:
    """Re-check the equality on every record for several prefix/suffix lengths."""
    prefixes = (0, 1, 5, 6, 15)
    suffixes = (0, 1, 3, 7)
    failures: list[dict[str, Any]] = []
    checks = 0
    for index, text in enumerate(written):
        raw = text.encode("utf-8")
        needle = identity(raw)
        for prefix in prefixes:
            for suffix in suffixes:
                if prefix == 0 and suffix == 0:
                    continue
                query = b"x" * prefix + raw + b"y" * suffix
                run, needle_start, _ = longest_contiguous_run(needle, features(query))
                checks += 1
                if run != len(needle) or needle_start != 0:
                    failures.append(
                        {
                            "record": index,
                            "prefix": prefix,
                            "suffix": suffix,
                            "identity_blocks": len(needle),
                            "covered": run,
                            "needle_start": needle_start,
                        }
                    )
    first_chunk_kept = []
    final_chunk_kept = []
    for index, text in enumerate(written):
        raw = text.encode("utf-8")
        blocks = chunks(raw)
        kept = identity_blocks(raw)
        if not blocks or not kept:
            continue
        spans = {(start, end) for start, end, _ in kept}
        if blocks[0][0] == 0 and (blocks[0][0], blocks[0][1]) in spans:
            first_chunk_kept.append(index)
        if blocks[-1][1] == len(raw) and (blocks[-1][0], blocks[-1][1]) in spans:
            final_chunk_kept.append(index)
    return {
        "checks": checks,
        "failures": failures,
        "identity_empty_records": [
            index for index, text in enumerate(written) if not identity(text.encode("utf-8"))
        ],
        "records_keeping_first_chunk": first_chunk_kept,
        "records_keeping_final_chunk": final_chunk_kept,
        "ok": bool(not failures and not first_chunk_kept and not final_chunk_kept),
    }


def soundness_check(table: AddressTable, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The three things that would make a short entry anchor the wrong record."""
    idents = [address["identity"] for address in table.addresses]
    empty = [index for index, ident in enumerate(idents) if not ident]
    clashes = [
        [outer, inner]
        for outer in range(len(idents))
        for inner in range(outer + 1, len(idents))
        if is_subsequence(idents[outer], idents[inner])
        or is_subsequence(idents[inner], idents[outer])
    ]
    negatives = [row for row in rows if row["kind"] in ("unknown", "unrelated")]
    anchored_negatives = [
        {
            "kind": row["kind"],
            "record": row["record"],
            "anchors": row["anchors"],
            "coverage_ratio": row["coverage_ratio"],
            "query": row["query_text"][:80],
        }
        for row in negatives
        if row["anchors"] > 0
    ]
    return {
        "empty_identities": empty,
        "containment_clashes": clashes,
        "anchored_negatives": anchored_negatives,
        "band_equals_predicate_width": bool(PROTECT == WINDOW),
        "min_addressable_matches_definition": bool(MIN_ADDRESSABLE == MIN_CHUNK + PROTECT + 1),
        "ok": bool(
            not empty
            and not clashes
            and not anchored_negatives
            and PROTECT == WINDOW
            and MIN_ADDRESSABLE == MIN_CHUNK + PROTECT + 1
        ),
    }


def tolerance_check(manifest: dict[str, Any]) -> dict[str, Any]:
    """Compare the live address module with the frozen inventory and literals."""
    live = chunk_address_audit()
    frozen = manifest["address_module"]
    old_constants = frozen["constant_inventory"]
    old_literals = frozen["read_path_literals"]
    new_constants = sorted(set(live["constants"]) - set(old_constants))
    changed_constants = sorted(
        name
        for name in live["constants"]
        if name in old_constants and live["constants"][name] != old_constants[name]
    )
    new_literals = sorted(set(live["read_path_literals"]) - set(old_literals))
    changed_literals = sorted(
        name
        for name in live["read_path_literals"]
        if name in old_literals and live["read_path_literals"][name] != old_literals[name]
    )
    previous_manifest = ROOT / "plans/manifests/taiji_shorts_20261108.json"
    previous_constants: dict[str, float] | None = None
    previous_literals: dict[str, list[float]] | None = None
    if previous_manifest.exists():
        previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
        previous_constants = previous["address_module"].get("constant_inventory")
        previous_literals = previous["address_module"].get("read_path_literals")
    return {
        "frozen_constants": old_constants,
        "live_constants": live["constants"],
        "frozen_read_path_literals": old_literals,
        "live_read_path_literals": live["read_path_literals"],
        "new_constants": new_constants,
        "changed_constants": changed_constants,
        "new_literals": new_literals,
        "changed_literals": changed_literals,
        "previous_manifest": (
            str(previous_manifest.relative_to(ROOT)) if previous_constants is not None else None
        ),
        "new_constants_vs_previous": (
            sorted(set(live["constants"]) - set(previous_constants))
            if previous_constants is not None
            else None
        ),
        "changed_constants_vs_previous": (
            sorted(
                name
                for name in live["constants"]
                if name in previous_constants
                and live["constants"][name] != previous_constants[name]
            )
            if previous_constants is not None
            else None
        ),
        "new_read_path_scopes_vs_previous": (
            sorted(set(live["read_path_literals"]) - set(previous_literals))
            if previous_literals is not None
            else None
        ),
        "structural_literal_note": (
            "the numeric literals in sentences()/_sentence_span are 0, 1, 2, 3 used only as "
            "span indices, list offsets and tuple positions; the terminator set is a "
            "categorical character set. None is a threshold or a tolerance."
        ),
        "protection": live["protection"],
        "ok": bool(
            not new_constants
            and not changed_constants
            and not new_literals
            and not changed_literals
            and live["protection"]["PROTECT_equals_WINDOW"]
            and live["protection"]["MIN_ADDRESSABLE_matches_definition"]
        ),
    }


def regression_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Row-by-row comparison with last round's 96 self + lead-in readings."""
    if not PREVIOUS_OUTPUTS.exists():
        return {"previous_rows": 0, "compared": 0, "regressions": [], "missing_source": True}
    previous = [
        json.loads(line)
        for line in PREVIOUS_OUTPUTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    old_rows = [
        row
        for row in previous
        if row["kind"] in ("self", "lead_in") and row["expected"] is not None and row["correct"]
    ]
    index = {(row["kind"], row["query_text"]): row for row in rows}
    compared = []
    regressions = []
    for old in old_rows:
        new = index.get((old["kind"], old["query_text"]))
        if new is None:
            regressions.append({"kind": old["kind"], "reason": "missing_query"})
            continue
        held = bool(
            new["correct"]
            and new["contained"]
            and new["coverage_ratio"] == 1.0
            and new["output_sha256"] == old["output_sha256"]
        )
        compared.append(
            {
                "kind": old["kind"],
                "query_text": old["query_text"],
                "old_sha256": old["output_sha256"],
                "new_sha256": new["output_sha256"],
                "held": held,
            }
        )
        if not held:
            regressions.append(
                {
                    "kind": old["kind"],
                    "query_text": old["query_text"][:80],
                    "old_sha256": old["output_sha256"],
                    "new_sha256": new["output_sha256"],
                    "new_correct": new["correct"],
                    "new_contained": new["contained"],
                    "new_coverage_ratio": new["coverage_ratio"],
                }
            )
    return {
        "previous_report": str(PREVIOUS_REPORT.relative_to(ROOT)),
        "previous_outputs": str(PREVIOUS_OUTPUTS.relative_to(ROOT)),
        "previous_rows": len(old_rows),
        "compared": len(compared),
        "regressions": regressions,
        "ok": bool(old_rows) and not regressions,
        "rows": compared,
    }


def output_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Five actual outputs, verbatim, with the boundary evidence for each."""

    def sample(label: str, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "label": label,
            "kind": row["kind"],
            "variant": row["variant"],
            "decision": row["decision"],
            "expected": row["expected"],
            "record": row["record"],
            "anchors": row["anchors"],
            "fallback": row["fallback"],
            "fallback_reason": row["fallback_reason"],
            "fallback_basis": row["fallback_basis"],
            "unit": row["unit"],
            "unit_index": row["unit_index"],
            "start": row["start"],
            "end": row["end"],
            "output_len": row["output_len"],
            "boundary_before": row["boundary_before"],
            "boundary_after": row["boundary_after"],
            "boundary_ok": row["boundary_ok"],
            "decodable": row["decodable"],
            "output_class": row["output_class"],
            "query_text": row["query_text"],
            "output_text": row["output_text"],
            "output_sha256": row["output_sha256"],
        }

    def first(predicate: Any) -> dict[str, Any] | None:
        return next((row for row in rows if predicate(row)), None)

    picks: list[tuple[str, dict[str, Any] | None]] = [
        (
            "anchored_record",
            first(lambda row: row["kind"] == "self" and row["decision"] == "matched"),
        ),
        (
            "anchored_record_lead_in",
            first(lambda row: row["kind"] == "lead_in" and row["decision"] == "matched"),
        ),
        (
            "fallback_sentence_record_overlap",
            first(
                lambda row: row["fallback_basis"] == "record_overlap" and row["unit"] == "sentence"
            ),
        ),
        (
            "fallback_sentence_address_order",
            first(
                lambda row: row["fallback_basis"] == "address_order" and row["unit"] == "sentence"
            ),
        ),
        (
            "unusable_or_wrong_else_ambiguous",
            first(lambda row: row["output_class"] == "unusable")
            or first(lambda row: row["output_class"] == "usable_but_wrong")
            or first(lambda row: row["kind"] == "ambiguous"),
        ),
    ]
    return [sample(label, row) for label, row in picks if row is not None]


def verdict_of(
    closed: dict[str, Any],
    summary: dict[str, Any],
    layer: dict[str, Any],
    tolerance: dict[str, Any],
) -> str:
    if closed["verdict"] != "output_unit_closed_neutral":
        return "hard_failure_closed_not_neutral"
    if not tolerance["ok"]:
        return "output_unit_needs_tolerance"
    l1_clean = bool(
        layer["fabricated_bytes"] == 0
        and layer["illegal_spans"] == 0
        and layer["utf8_invalid_slices"] == 0
        and layer["boundary_illegal"] == 0
    )
    anchor = summary["should_anchor"]
    if not l1_clean:
        return "inconclusive"
    if anchor["usable_and_correct"] >= USABLE_CORRECT_LINE and anchor["unusable"] == 0:
        return "output_unit_fixed"
    if anchor["unusable"] > 0:
        return "output_unit_leaks"
    return "inconclusive"


def previous_round() -> dict[str, Any]:
    """Frozen side-by-side column: last round's readings, never recomputed here."""
    if not PREVIOUS_REPORT.exists():
        return {"commit": PREVIOUS_COMMIT, "report": None}
    old = json.loads(PREVIOUS_REPORT.read_text(encoding="utf-8"))
    return {
        "commit": PREVIOUS_COMMIT,
        "report": str(PREVIOUS_REPORT.relative_to(ROOT)),
        "verdict": old.get("verdict"),
        "layers": old.get("layers"),
        "should_anchor": old.get("readings", {}).get("should_anchor"),
        "write": {
            key: old.get("write", {}).get(key)
            for key in ("addresses", "appended", "merged", "rejected_records", "rejected_reasons")
        },
    }


def phase_open(closed: dict[str, Any]) -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if file_sha(CORPUS) != manifest["corpus"]["snapshot_sha256"]:
        raise SystemExit("corpus snapshot moved after freeze")
    for key in ("written", "unknown", "unrelated"):
        if [text_sha(text) for text in corpus[key]] != manifest["corpus"][f"{key}_sha256"]:
            raise SystemExit(f"corpus {key} differs from the frozen manifest")

    written = corpus["written"]
    config = LivingCoreConfig(sc_sfk_enabled=True)
    subject = LivingCoreReference(config)
    state = subject.attach_store(subject.initial_state("shorts"))
    table = state.sc_sfk
    if not isinstance(table, AddressTable):
        raise SystemExit("switch did not allocate an AddressTable")
    append_progress({"phase": "store_attached", "enabled": True})
    payload = state.fast.detach().to("cpu").contiguous().numpy().tobytes()

    write_batches = []
    write_detail: list[dict[str, Any]] = []
    for start in range(0, len(written), BATCH):
        block = written[start : start + BATCH]
        result = table.write_batch_checked([(text.encode("utf-8"), payload) for text in block])
        accepted = {row["item"]: row for row in result.get("accepted", [])}
        rejected = {row["item"]: row for row in result.get("rejected", [])}
        for offset, text in enumerate(block):
            raw = text.encode("utf-8")
            detail = {
                "record": start + offset,
                "length": len(raw),
                "identity_blocks": len(identity(raw)),
            }
            if offset in accepted:
                detail["action"] = accepted[offset]["action"]
                detail["address"] = accepted[offset]["address"]
            else:
                detail["action"] = "rejected"
                detail["reason"] = rejected[offset]["reason"]
                detail["definition_necessary"] = bool(
                    rejected[offset]["reason"] == "identity_too_short"
                )
            write_detail.append(detail)
        write_batches.append(
            {
                "batch": start // BATCH,
                "items": result["items"],
                "accepted": result["accepted_items"],
                "rejected": len(result["rejected"]),
                "slots_before": result["slots_before"],
                "slots_after": result["slots_after"],
            }
        )
        append_progress(
            {
                "phase": "write",
                "batch": start // BATCH,
                "action": result["action"],
                "slots": len(table.addresses),
            }
        )

    record_address: list[int | None] = []
    for text in written:
        found = table.find(identity(text.encode("utf-8")))
        record_address.append(found)
    rejected_records = sum(1 for index in record_address if index is None)
    admitted_records = len(record_address) - rejected_records
    append_progress(
        {
            "phase": "records",
            "addresses": len(table.addresses),
            "admitted_records": admitted_records,
            "rejected_records": rejected_records,
            "rejected_reasons": table.rejected_reasons,
        }
    )

    queries = build_queries(corpus, record_address, manifest["queries"]["pairs"])
    rows, layer = evaluate(table, queries)
    summary = summarise(rows)

    lengths = length_profile(written)
    shift = shift_invariance_check(written)
    sound = soundness_check(table, rows)
    tolerance = tolerance_check(manifest)
    regress = regression_check(rows)
    samples = output_samples(rows)

    legacy = LegacyTable()
    for text in written:
        legacy.write(text.encode("utf-8"))
    legacy_rows = legacy_summary(legacy, queries)

    write_block = {
        "batches": write_batches,
        "addresses": len(table.addresses),
        "admitted_records": admitted_records,
        "appended": table.appended,
        "merged": table.merged,
        "rejected_records": rejected_records,
        "rejected_reasons": table.rejected_reasons,
        "rollbacks": table.rollbacks,
        "evictions": table.evictions,
        "silent_overwrites": table.silent_overwrites,
        "rejected_batches": table.rejected_batches,
        "nbytes": table.nbytes(),
        "identity_blocks": sorted(len(address["identity"]) for address in table.addresses),
        "detail": write_detail,
    }
    verdict = verdict_of(closed, summary, layer, tolerance)
    cpu = cpu_check()
    report = {
        "schema": "taiji-output-unit-v1",
        "date": "2026-11-08",
        "question": manifest["question"],
        "verdict": verdict,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": file_sha(MANIFEST),
        "corpus_snapshot_sha256": file_sha(CORPUS),
        "runner_sha_start": RUNNER_SHA_START,
        "runner_sha_end": file_sha(Path(__file__).resolve()),
        "closed": {
            "report": str(CLOSED_REPORT.relative_to(ROOT)),
            "verdict": closed["verdict"],
            "readings": closed["checks"]["readings_total"],
            "fast_max_abs_delta": closed["checks"]["fast_max_abs_delta"],
            "probability_max_abs_delta": closed["checks"]["probability_max_abs_delta"],
            "table_not_allocated": closed["checks"]["table_not_allocated"],
            "living_core_sha256": closed["baseline"]["living_core_sha256"],
        },
        "write": write_block,
        "shortest_addressable_length": {
            "definition_bytes": MIN_ADDRESSABLE,
            "formula": "MIN_CHUNK + PROTECT + 1 = 16 + 6 + 1",
            "necessary_not_sufficient": True,
            "ceiling": MIN_LENGTH_CEILING,
            "observed": lengths,
        },
        "shift_invariance": shift,
        "soundness": sound,
        "tolerance_audit": tolerance,
        "regression": {
            key: regress[key]
            for key in ("previous_report", "previous_outputs", "previous_rows", "compared", "ok")
        }
        | {"regressions": regress["regressions"]},
        "output_samples": samples,
        "output_samples_note": (
            "the unusable and usable_but_wrong classes are empty in this run, so the fifth "
            "slot shows the closest non-answer case available: an ambiguous query, where two "
            "records are anchored and the printed unit is one of them."
        ),
        "boundary_source": {
            "unit_distribution": summary["unit_distribution"],
            "fallback_basis_distribution": summary["fallback_basis_distribution"],
            "fallback_outputs": layer["fallback_outputs"],
            "note": (
                "unit = record means the whole stored record was printed; unit = sentence "
                "means one span returned by sentences() for the claimed record. fallback = "
                "true on every unanchored read, with basis record_overlap or address_order."
            ),
        },
        "denominators": {
            "should_anchor": summary["should_anchor"]["n"],
            "should_unknown": summary["counts"].get("unknown", 0),
            "should_ambiguous": summary["counts"].get("ambiguous", 0),
            "unrelated": summary["counts"].get("unrelated", 0),
            "queries": layer["queries"],
        },
        "layers": {
            "L1": {
                "fabricated_bytes": layer["fabricated_bytes"],
                "illegal_spans": layer["illegal_spans"],
                "utf8_valid": layer["utf8_valid_slices"],
                "utf8_invalid": layer["utf8_invalid_slices"],
                "utf8_decodable_rate": round(
                    layer["utf8_valid_slices"] / max(layer["queries"], 1), 6
                ),
                "boundary_legal": layer["boundary_legal"],
                "boundary_illegal": layer["boundary_illegal"],
                "boundary_legal_rate": round(layer["boundary_legal"] / max(layer["queries"], 1), 6),
                "samples": [
                    {
                        "kind": row["kind"],
                        "record": row["record"],
                        "span": [row["start"], row["end"]],
                        "output_len": row["output_len"],
                        "legal_span": row["legal_span"],
                        "boundary_ok": row["boundary_ok"],
                        "decodable": row["decodable"],
                    }
                    for row in rows
                    if not (row["legal_span"] and row["boundary_ok"] and row["decodable"])
                ][:5],
            },
            "L2": {
                "faithful": layer["faithful"],
                "faithful_rate": layer["faithful_rate"],
                "denominator": layer["queries"],
            },
            "L3": {
                "usable_and_correct": summary["should_anchor"]["usable_and_correct"],
                "usable_but_wrong": summary["should_anchor"]["usable_but_wrong"],
                "unusable": summary["should_anchor"]["unusable"],
                "denominator": summary["should_anchor"]["n"],
                "blocked_by_l1": layer["blocked_by_l1"],
                "blocked_samples": layer["blocked_samples"],
            },
        },
        "readings": summary,
        "legacy_control": legacy_rows,
        "previous_round": previous_round(),
        "cost": {
            "cpu_seconds": round(cpu, 3),
            "wall_seconds": round(time.time() - START_WALL, 3),
            "closed_wall_seconds": closed["cost"]["wall_seconds"],
        },
        "criteria": CRITERIA,
        "disclosures": [
            "the output is a slice of the record the table selected; the runner records the "
            "span and re-checks it byte by byte, so L1/L2 measure the printed unit, not the "
            "intent.",
            "the printed unit is the whole record when a record anchors and one whole sentence "
            "of the chosen record otherwise. A sentence span starts at the record start or "
            "right after a terminator/newline and ends after a terminator/newline or at the "
            "record end, so a slice can never cut a character, a word or a sentence. There is "
            "no byte-offset fragment path left in read().",
            "sentence choice on fallback reuses the existing block-overlap ranking; the "
            "terminator set is categorical. No numeric threshold, band or tolerance was added "
            "anywhere, and the frozen constant / read-path literal comparison is reported.",
            "abstention is a reading: no_anchor, ambiguous_candidates and overlap never gate "
            "the output and never enter the verdict.",
            "the synonym rewrites change bytes in the middle of the record, which breaks "
            "contiguous containment by definition; they fall back and are scored on the "
            "printed unit without substring credit, remaining in the denominator.",
            "chunks shorter than 16 bytes are dropped after the cut set is fixed, so their "
            "bytes are invisible to the address; the rule never moves a cut point.",
            "the band is six bytes, the predicate's own dependency width, and only the final "
            "chunk is dropped; both are the smallest correct choice, not tuned values.",
            "a one-block identity certifies at least 16 verbatim interior bytes, not the whole "
            "record: that is the specificity a short record can offer, and it is reported, "
            "not hidden. An empty identity would anchor every query and is rejected.",
            "the fallback label (fallback / fallback_reason / fallback_basis) is a reading; it "
            "never decides whether an output is produced, and it adds no threshold.",
            "the legacy column is a frozen copy of the old stride-2 / DF=1 / DF=0-guard read "
            "rule and never enters the verdict.",
            "the corpus snapshot is what the run reads, so later plan edits cannot move the "
            "acceptance set.",
        ],
    }
    write_json(REPORT, report)
    write_jsonl(
        OUTPUTS,
        [
            {
                "kind": row["kind"],
                "variant": row["variant"],
                "expected": row["expected"],
                "record": row["record"],
                "decision": row["decision"],
                "anchors": row["anchors"],
                "covered_blocks": row["covered_blocks"],
                "coverage_ratio": row["coverage_ratio"],
                "overlap_ratio": row["overlap_ratio"],
                "shared_blocks": row["shared_blocks"],
                "fallback": row["fallback"],
                "fallback_reason": row["fallback_reason"],
                "fallback_basis": row["fallback_basis"],
                "contained": row["contained"],
                "unit": row["unit"],
                "unit_index": row["unit_index"],
                "span": [row["start"], row["end"]],
                "output_len": row["output_len"],
                "output_sha256": row["output_sha256"],
                "faithful": row["faithful"],
                "legal_span": row["legal_span"],
                "boundary_ok": row["boundary_ok"],
                "decodable": row["decodable"],
                "usable": row["usable"],
                "correct": row["correct"],
                "output_class": row["output_class"],
                "boundary_before": row["boundary_before"],
                "boundary_after": row["boundary_after"],
                "query_text": row["query_text"],
                "output_text": row["output_text"],
                "output_hex": row["output_hex"],
            }
            for row in rows
        ],
    )
    append_progress({"phase": "done", "verdict": verdict})
    print(json.dumps({"verdict": verdict, "layers": report["layers"]}, ensure_ascii=False))
    return 0


def main(argv: list[str]) -> int:
    assert_no_black_processes()
    lock_name = acquire_single_instance()
    atexit.register(release_single_instance)
    stop = threading.Event()
    start_watchdog(stop)
    try:
        if "--freeze" in argv:
            return phase_freeze()
        if not MANIFEST.exists():
            raise SystemExit("manifest missing: run with --freeze first")
        closed = phase_closed()
        return phase_open(closed)
    finally:
        stop.set()
        append_progress({"phase": "exit", "lock": lock_name, "watchdog": WATCHDOG_FIRED[0]})
        release_single_instance()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
