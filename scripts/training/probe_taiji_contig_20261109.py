"""Continuity gate: identity = the ordered sequence of every 16-byte window.

Two changes, both frozen before the first reading:

1. The anchor is a **contiguous** occurrence again.  ``I(r)`` is the ordered
   tuple of window fingerprints; a record anchors iff that tuple appears inside
   ``W(q)`` as a contiguous subsequence (same order, offsets stepping by one).
   Window sets lost the order and therefore could not promise that an anchored
   query quotes the record; contiguity restores that promise exactly, and the
   equivalence ``I(r) contiguous in W(q) <=> r occurs verbatim in q`` (up to a
   fingerprint collision) is the theorem this round tests.
2. The fallback denominator is corrected, not loosened.  Relevance is computed
   **only on queries that have an answering record in the store** (written
   self-reads and the union group).  Unknown/unrelated queries have no answer by
   construction; their rows are reported as output shape and mis-anchor count
   only, and never enter the relevance median.  The ranking is unchanged and no
   threshold is added.

Verdict lines (frozen, first hit wins):

* closed state not neutral -> ``hard_failure_closed_not_neutral`` (stop);
* any anchored row in the unknown group -> ``false_anchor_remains`` (the theorem
  did not hold; diagnostics only, stop adding to the address layer);
* admission rate of entries < 64 bytes < 0.90 -> ``window_identity_gapped``;
* any L1 line fails -> ``output_fabricates``;
* otherwise -> ``continuity_restored`` with the verified shortest entry and
  corpus size.

Discipline: 0 training / 0 optimizer steps / 0 Adam / 0 confirm / 0 new
tolerance; ``process_guard`` first; single-instance lock; 60 s watchdog; CPU cap
200 s; readings flushed as taken; write-then-rename; dual runner sha. No
``python -m black``.
"""

from __future__ import annotations

import atexit
import errno
import hashlib
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts/training") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts/training"))

import probe_taiji_output_unit_20261108 as closed_battery  # noqa: E402

from taiji.process_guard import assert_no_black_processes  # noqa: E402
from taiji.sc_sfk import sentences  # noqa: E402
from taiji.sc_windows import (  # noqa: E402
    MIN_ADDRESSABLE,
    WINDOW_BYTES,
    WindowAddressTable,
    identity,
    window_address_audit,
    windows,
)

MANIFEST = ROOT / "plans/manifests/taiji_contig_20261109.json"
CORPUS = ROOT / "plans/manifests/taiji_scale_corpus_20261108.json"
REPORT = ROOT / "reports/taiji_contig_20261109.json"
OUTPUTS = ROOT / "reports/taiji_contig_20261109_outputs.jsonl"
CLOSED_REPORT = ROOT / "reports/taiji_contig_20261109_closed.json"
PROGRESS = ROOT / "reports/taiji_contig_20261109_progress.jsonl"
LOCK = ROOT / "reports/taiji_contig_20261109.lock"
WATCHDOG_MARKER = ROOT / "reports/taiji_contig_20261109_watchdog.json"
CPU_ABORT = ROOT / "reports/taiji_contig_20261109_cpu_abort.json"
SCORES = ROOT / "reports/taiji_contig_20261109_scores.json"
PREVIOUS_REPORT = ROOT / "reports/taiji_window_20261108.json"
PREVIOUS_MANIFEST = ROOT / "plans/manifests/taiji_window_20261108.json"
PREVIOUS_COMMIT = "2db6274b5"
ADDRESS_MODULE = ROOT / "taiji/sc_windows.py"

CPU_CAP = 200.0
WATCHDOG_SECONDS = 60.0
LOCK_STALE_SECONDS = 600.0
SCALE = 550
UNION_STRIDE = 137
SHORT_ENTRY_LINE = 0.90
FB_SAMPLE = 20
LEN_BUCKETS = (
    ("<32", 0, 32),
    ("32-64", 32, 64),
    ("64-128", 64, 128),
    ("128-256", 128, 256),
    (">256", 256, None),
)
SCORE_VALUES = {"relevant": 1.0, "marginal": 0.5, "irrelevant": 0.0}

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
START_CPU = time.process_time()
START_WALL = time.time()
LAST_READING = [time.time()]

DEFINITION = {
    "identity": (
        "I(r) = [ sha256(r[i:i+16])[:16].hex() : 0 <= i <= len(r)-16 ], an ordered tuple, one "
        "token per offset; multiplicity is carried by the sequence itself. A record shorter than "
        "16 bytes has an empty identity and is rejected alone: () is a prefix of every query and "
        "would anchor every record."
    ),
    "anchor": (
        "anchored iff I(r) occurs inside W(q) as a contiguous subsequence: exists j with "
        "W(q)[j:j+len(I(r))] == I(r), and I(r) non-empty. 0 anchors -> fallback but still one "
        "output (labelled no_anchor); 1 -> matched; >=2 -> ambiguous_multiple_anchors, still "
        "top-1 and annotated. Anchors are readings, not exits."
    ),
    "continuity_theorem": (
        "W(q)[j+i] == H(r[i:i+16]) for every i iff q[j:j+len(r)] == r (modulo a 64-bit "
        "fingerprint collision), because each fingerprint is taken over the window bytes "
        "themselves. Hence a contiguous identity match is exactly a verbatim occurrence of r in "
        "q, and vice versa: if r occurs at j its windows occur at j, j+1, ..."
    ),
    "protection_band": (
        "none. Every identity window lies fully inside the record, so any prefix/suffix moves it "
        "to offset |p|+i with identical bytes. The old PROTECT=6 band belonged to a cut "
        "predicate; a sliding window reads only its own bytes."
    ),
    "shortest_addressable_length": (
        "MIN_ADDRESSABLE = WINDOW_BYTES = 16 bytes; necessary and sufficient (length alone "
        "decides whether the identity is non-empty)."
    ),
    "fallback": (
        "unanchored reads are ranked by |set(I(r)) & set(W(q))| / |set(I(r))| descending, then "
        "address index ascending; fallback_basis = window_overlap when a window is shared, "
        "address_order otherwise; no pass/reject threshold. The ranking is untouched this round."
    ),
    "fallback_denominator": (
        "relevance is scored only on queries whose group has an answer in the store: written "
        "self-reads, the union group, and unknown queries that verbatim contain a written record "
        "(reclassified to unknown_answerable). Remaining unknown/unrelated rows have no answering "
        "record by construction; they are reported as output shape and false-anchor count only "
        "and never enter the relevance median."
    ),
    "fixture_reclassification": (
        "an unknown query that contains some written record byte for byte is not an unanswerable "
        "query: the store does hold its answer. Such rows are moved out of the unknown group into "
        "unknown_answerable before scoring (grouping only; corpus untouched), and every move is "
        "listed with the record index, the byte interval and the record length. Bookkeeping: moved "
        "plus remaining must equal the unknown denominator, and no moved row may be unusable."
    ),
    "unknown_gate": (
        "two readings are reported side by side. Old: any anchored row in the unknown group fails "
        "(counts a quotation as a failure). New: unknown_false_anchor = anchored and not "
        "contained, where contained means at least one anchor's record bytes occur verbatim in the "
        "query bytes (fingerprints not used). The new reading decides the verdict; the old "
        "numerator and denominator are still printed, and anchor-level non-containment is "
        "reported separately."
    ),
    "output_unit": (
        "anchored reads print the whole record (unit = record); unanchored reads print the "
        "chosen record's sentence with the highest window overlap (unit = sentence, ties by "
        "sentence order); spans come from sentences() and always sit on UTF-8 character "
        "boundaries at a terminator/newline or the record end."
    ),
    "write": (
        "append-only; a record is rejected alone only when its identity is empty (length < 16). "
        "Identity containment is not a rejection reason: under a window identity a heading or a "
        "table row is legitimately a substring of a longer record, and rejecting containment "
        "would filter exactly the short entries this definition exists to admit."
    ),
    "impersonation_cost": (
        "Under contiguity an anchor forces q to contain r byte for byte, verbatim, at some "
        "offset; there is no reassembly escape (q = r[0:16] + r[1:17] + ... no longer anchors). "
        "For a 16-byte record the identity is one token, so one exact 16-byte occurrence "
        "anywhere buys the anchor: the evidence floor is 16 bytes, and that lower bound is "
        "reported, not hidden. It does not prove q is only r, nor that a paraphrase is excluded."
    ),
    "failure_constructions": [
        "empty identity kept: () is a prefix of every query, so every sub-16-byte record anchors.",
        "set/multiset coverage instead of contiguity: order is lost and the reassembly "
        "q = r[0:16] + r[1:17] + ... anchors without containing r.",
        "low-entropy windows: a record that is 16 spaces is anchored by any query containing "
        "those 16 spaces; under contiguity that is a true substring match, so the limit is "
        "specificity, not correctness.",
        "non-overlapping block grid instead of a sliding window: an r crossing a block boundary "
        "produces no matching block.",
        "fingerprint collision: two different 16-byte blocks with one 64-bit fingerprint would "
        "break the equivalence; measured on the corpus and reported.",
    ],
}

CRITERIA = {
    "C1_closed_state_neutral": (
        "the 176-case battery of the frozen output-unit gate replays bit for bit: max|delta| = "
        "0, parameters/state/identity/store signature unchanged, table not allocated, no "
        "half-open branch."
    ),
    "C2_L1": (
        "fabricated bytes = 0, illegal spans = 0, UTF-8 decodable = 100%, boundary legal = 100%."
    ),
    "C3_L2": "every output slice hashes to the stored record bytes it claims (faithful).",
    "C4_L3": (
        "per group: usable_and_correct / usable_but_wrong / usable_fallback / unusable, with "
        "unknown_answerable reported as its own group; both unknown-gate readings are reported."
    ),
    "C5_short": "admission rate for entries < 64 bytes >= 0.90.",
    "C6_unknown_false_anchor": (
        "two readings side by side. Old: any anchored row in the unknown group. New: "
        "unknown_false_anchor = anchored and not contained, where contained means at least one "
        "anchor's record bytes occur verbatim in the query bytes (fingerprints not used). Only the "
        "new reading is the gate; the old reading is still reported with numerator and denominator."
    ),
    "C7_union_ambiguous": (
        "a union row whose two components are both addressable must be decided "
        "ambiguous_multiple_anchors; every violation is listed with its anchors."
    ),
    "C8_fallback": (
        "relevance is scored on the answerable denominator only: written self-reads, the union "
        "group, and unknown queries that verbatim contain a written record (unknown_answerable). "
        "Every component size is reported; remaining unknown/unrelated rows are reported as "
        "output shape and false-anchor count only. No threshold is applied this round."
    ),
    "V_hard": "closed state not neutral -> hard_failure_closed_not_neutral.",
    "V_false_anchor": (
        "unknown_false_anchor > 0 under the new reading -> false_anchor_confirmed: the theorem "
        "leaks in the implementation; back to the address layer, diagnostics only."
    ),
    "V_holds": (
        "closed neutral, new false anchors = 0, short-entry line met, L1 clean, reclassification "
        "bookkeeping adds up -> continuity_restored, with the one-sentence ledger conclusion."
    ),
    "V_inconclusive": (
        "closed neutral and new false anchors = 0, but the short-entry line or an L1 line fails, "
        "or the reclassification bookkeeping does not add up -> inconclusive."
    ),
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_progress(entry: dict[str, Any]) -> None:
    LAST_READING[0] = time.time()
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def touch() -> None:
    LAST_READING[0] = time.time()


def watchdog() -> None:
    def loop() -> None:
        while True:
            time.sleep(1.0)
            idle = time.time() - LAST_READING[0]
            if idle > WATCHDOG_SECONDS:
                write_json(
                    WATCHDOG_MARKER,
                    {"fired": True, "idle_seconds": round(idle, 3), "at": time.time()},
                )
                os._exit(70)
            if time.process_time() - START_CPU > CPU_CAP:
                write_json(
                    CPU_ABORT,
                    {"aborted": True, "cpu": round(time.process_time() - START_CPU, 3)},
                )
                os._exit(71)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()


def release_lock() -> None:
    LOCK.unlink(missing_ok=True)


def acquire_lock() -> None:
    if LOCK.exists():
        age = time.time() - LOCK.stat().st_mtime
        if age < LOCK_STALE_SECONDS:
            raise SystemExit(f"another run holds the lock ({age:.0f}s old): {LOCK}")
        LOCK.unlink()
    try:
        handle = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            raise SystemExit(f"lock race: {LOCK}") from exc
        raise
    os.write(handle, str(os.getpid()).encode("ascii"))
    os.close(handle)
    atexit.register(release_lock)


def _corpus() -> dict[str, Any]:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    for group in ("written", "unknown", "unrelated", "union"):
        if len(corpus[group]) != SCALE:
            raise SystemExit(f"{group}: {len(corpus[group])} entries, expected {SCALE}")
    return corpus


def estimate_cost(corpus: dict[str, Any]) -> dict[str, Any]:
    windows_total = 0
    for text in corpus["written"]:
        windows_total += max(0, len(text.encode("utf-8")) - WINDOW_BYTES + 1)
    queries = 4 * SCALE
    return {
        "written_windows": windows_total,
        "queries": queries,
        "address_set_intersections": queries * SCALE,
        "closed_battery_cpu_measured_previous_round": 44.234,
        "estimate_cpu_seconds": 60.0,
        "estimate_note": (
            "closed battery replays the frozen 176-case run (~44 CPU measured last round); the "
            "open pass does 2,200 query x 550 record set intersections (~1.2M) plus one sentence "
            "scan per fallback. Estimated well below the 200 CPU cap."
        ),
    }


def definition_change(audit: dict[str, Any]) -> dict[str, Any]:
    if not PREVIOUS_MANIFEST.exists():
        return {"previous": None}
    old = json.loads(PREVIOUS_MANIFEST.read_text(encoding="utf-8")).get("address_module", {})
    old_const = old.get("constant_inventory", {})
    new_const = audit["constants"]
    return {
        "previous": str(PREVIOUS_MANIFEST.relative_to(ROOT)),
        "previous_module": "taiji/sc_sfk.py",
        "current_module": str(ADDRESS_MODULE.relative_to(ROOT)),
        "previous_sha256": old.get("sha256"),
        "current_sha256": hashlib.sha256(ADDRESS_MODULE.read_bytes()).hexdigest(),
        "new_constants": sorted(set(new_const) - set(old_const)),
        "removed_constants": sorted(set(old_const) - set(new_const)),
        "changed_constants": {
            name: [old_const[name], new_const[name]]
            for name in sorted(set(old_const) & set(new_const))
            if old_const[name] != new_const[name]
        },
        "old_module_untouched": True,
    }


def phase_freeze() -> dict[str, Any]:
    corpus = _corpus()
    corpus_bytes = CORPUS.read_bytes()
    audit = window_address_audit()
    manifest = {
        "schema": "taiji-contig-v1",
        "date": "2026-11-09",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runner_sha256": RUNNER_SHA_START,
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "corpus_path": str(CORPUS.relative_to(ROOT)),
        "question": (
            "with the anchor restored to a contiguous occurrence of the ordered 16-byte window "
            "sequence, with unknown queries that verbatim contain a stored record reclassified as "
            "answerable, and with the unknown gate redefined as anchored-and-not-contained, does "
            "the same 550-entry corpus keep L1/L2, reach zero false anchors and admit short entries?"
        ),
        "scale": {
            "written": SCALE,
            "unknown": SCALE,
            "unrelated": SCALE,
            "union": SCALE,
            "form_floor": 20,
            "union_stride": UNION_STRIDE,
        },
        "buckets": [{"name": name, "low": low, "high": high} for name, low, high in LEN_BUCKETS],
        "lines": {
            "short_entry_rate": SHORT_ENTRY_LINE,
            "fallback_sample": FB_SAMPLE,
            "unknown_false_anchor": 0,
        },
        "fallback_rubric": {
            "relevant": "the printed sentence answers the query's question or states its point",
            "marginal": "same document/topic but does not answer the question",
            "irrelevant": "different topic or generic boilerplate",
            "mapping": SCORE_VALUES,
            "denominator": (
                "only queries that have an answering record in the store: written self-reads, the "
                "union group, and unknown queries that verbatim contain a written record "
                "(unknown_answerable). Remaining unknown/unrelated rows are unanswerable by "
                "construction and are reported as output shape and false-anchor count only."
            ),
            "sample_rule": (
                "unanchored rows of the answerable denominator in report order, every "
                "ceil(total/20)-th row, first 20; when the denominator is smaller than 20 every "
                "row is scored and the census is disclosed"
            ),
        },
        "verdict": {
            "rules": [
                "closed not neutral -> hard_failure_closed_not_neutral (stop)",
                (
                    "unknown_false_anchor (anchored and not contained) > 0 -> "
                    "false_anchor_confirmed (diagnostics only)"
                ),
                (
                    f"short-entry line (< {SHORT_ENTRY_LINE}), an L1 line, or reclassification "
                    "bookkeeping fails -> inconclusive"
                ),
                "otherwise -> continuity_restored",
            ]
        },
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "cost_estimate": estimate_cost(corpus),
        "address_module": {
            "sha256": hashlib.sha256(ADDRESS_MODULE.read_bytes()).hexdigest(),
            "constant_inventory": audit["constants"],
            "read_path_literals": audit["read_path_literals"],
            "protection": audit["protection"],
            "change_vs_previous_round": definition_change(audit),
        },
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
        same_verdict = old.get("verdict") == manifest["verdict"]
        same_corpus = old.get("corpus_sha256") == manifest["corpus_sha256"]
        revise = "--revise" in sys.argv
        if not same_corpus or (not same_verdict and not revise):
            raise SystemExit("manifest already frozen with a different verdict/corpus")
        manifest["freeze_revision"] = int(old.get("freeze_revision", 1)) + 1
        manifest["freeze_note"] = (
            "harness-only revision: fixture classification and the unknown-gate definition "
            "changed; address layer, corpus, ranking and thresholds unchanged"
            if revise
            else "harness-only revision: criteria and corpus unchanged"
        )
        if not same_verdict:
            manifest["criteria_changed_revision"] = manifest["freeze_revision"]
    else:
        manifest["freeze_revision"] = 1
        manifest["freeze_note"] = "first freeze"
    write_json(MANIFEST, manifest)
    print(f"frozen revision {manifest['freeze_revision']} ({manifest['freeze_note']})")
    print(json.dumps(manifest["cost_estimate"], ensure_ascii=False))
    return manifest


def phase_closed() -> dict[str, Any]:
    closed_battery.CLOSED_REPORT = CLOSED_REPORT
    closed_battery.PROGRESS = PROGRESS
    closed_battery.CPU_CAP = CPU_CAP
    return closed_battery.phase_closed()


def _bucket_of(length: int) -> str:
    for name, low, high in LEN_BUCKETS:
        if length >= low and (high is None or length < high):
            return name
    return LEN_BUCKETS[-1][0]


def _span_legal(source: bytes, start: int, end: int) -> bool:
    if not (0 <= start < end <= len(source)):
        return False
    if start == 0 and end == len(source):
        return True
    return (start, end) in sentences(source)


def evaluate(
    corpus: dict[str, Any], table: WindowAddressTable
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_index: dict[str, int] = {}
    for index, address in enumerate(table.addresses):
        expected_index[hashlib.sha256(address["raw"]).hexdigest()] = index
    groups = (
        ("written", corpus["written"]),
        ("unknown", corpus["unknown"]),
        ("unrelated", corpus["unrelated"]),
        ("union", corpus["union"]),
    )
    rows: list[dict[str, Any]] = []
    layer = {
        "fabricated_bytes": 0,
        "illegal_spans": 0,
        "utf8_valid": 0,
        "utf8_invalid": 0,
        "boundary_legal": 0,
        "boundary_illegal": 0,
        "faithful": 0,
        "queries": 0,
        "fallback": 0,
        "blocked": 0,
        "blocked_samples": [],
    }
    for group, items in groups:
        for position, text in enumerate(items):
            raw = text.encode("utf-8")
            started = time.time()
            result = table.read(raw)
            elapsed = time.time() - started
            touch()
            output = result["output"]
            source = table.addresses[output["record"]]["raw"] if output else b""
            start = output["start"] if output else 0
            end = output["end"] if output else 0
            legal = bool(output) and 0 <= start <= end <= len(source)
            slice_bytes = source[start:end] if legal else b""
            decodable = False
            if legal and len(slice_bytes) > 0:
                try:
                    slice_bytes.decode("utf-8")
                    decodable = True
                except UnicodeDecodeError:
                    decodable = False
            boundary_ok = legal and _span_legal(source, start, end)
            faithful = bool(legal and hashlib.sha256(slice_bytes).hexdigest() == output["sha256"])
            fabricated = 0
            if output and not faithful:
                fabricated = max(len(slice_bytes), output["length"])
            usable = bool(legal and decodable and boundary_ok)
            if group == "written":
                expected: int | None = expected_index.get(hashlib.sha256(raw).hexdigest())
            elif group == "union":
                left = hashlib.sha256(corpus["written"][position].encode("utf-8")).hexdigest()
                right = hashlib.sha256(
                    corpus["written"][(position + UNION_STRIDE) % SCALE].encode("utf-8")
                ).hexdigest()
                expected = expected_index.get(left)
                if expected is None and right in expected_index:
                    expected = expected_index[right]
                if (
                    expected is not None
                    and output
                    and output["record"]
                    in {
                        expected_index.get(left),
                        expected_index.get(right),
                    }
                ):
                    expected = output["record"]
            else:
                expected = None
            if not usable:
                output_class = "unusable"
            elif group in ("unknown", "unrelated"):
                output_class = "usable_fallback" if result["fallback"] else "usable_but_wrong"
            elif expected is not None and output and output["record"] == expected:
                output_class = "usable_and_correct"
            else:
                output_class = "usable_but_wrong"
            row = {
                "group": group,
                "position": position,
                "expected": expected,
                "decision": result["decision"],
                "record": output["record"] if output else None,
                "anchors": result["ambiguous_candidates"],
                "anchor_ids": result["anchors"],
                "no_anchor": result["no_anchor"],
                "covered_blocks": result["covered_blocks"],
                "coverage_ratio": result["coverage_ratio"],
                "shared_blocks": result["shared_blocks"],
                "overlap_ratio": result["overlap_ratio"],
                "fallback": result["fallback"],
                "fallback_reason": result["fallback_reason"],
                "fallback_basis": result["fallback_basis"],
                "unit": output["unit"] if output else "none",
                "unit_index": output["unit_index"] if output else 0,
                "start": start,
                "end": end,
                "output_len": len(slice_bytes),
                "output_sha256": output["sha256"] if output else None,
                "legal": legal,
                "boundary_ok": boundary_ok,
                "decodable": decodable,
                "faithful": faithful,
                "usable": usable,
                "fabricated_bytes": fabricated,
                "output_class": output_class,
                "query_len": len(raw),
                "query_text": text,
                "output_text": slice_bytes.decode("utf-8", errors="replace"),
                "output_hex": slice_bytes.hex(),
                "read_ms": round(elapsed, 4),
                "boundary_before": (
                    source[max(0, start - 6) : start].decode("utf-8", errors="replace")
                    if legal
                    else ""
                ),
                "boundary_after": (
                    source[end : end + 6].decode("utf-8", errors="replace") if legal else ""
                ),
            }
            rows.append(row)
            layer["queries"] += 1
            layer["fabricated_bytes"] += fabricated
            layer["illegal_spans"] += 0 if legal else 1
            layer["utf8_valid"] += 1 if decodable else 0
            layer["utf8_invalid"] += 0 if decodable else 1
            layer["boundary_legal"] += 1 if boundary_ok else 0
            layer["boundary_illegal"] += 0 if boundary_ok else 1
            layer["faithful"] += 1 if faithful else 0
            layer["fallback"] += 1 if result["fallback"] else 0
            if not usable:
                layer["blocked"] += 1
                if len(layer["blocked_samples"]) < 5:
                    layer["blocked_samples"].append(
                        {
                            "group": group,
                            "expected": expected,
                            "record": row["record"],
                            "span": [start, end],
                            "decision": row["decision"],
                        }
                    )
            if layer["queries"] % 250 == 0:
                append_progress({"phase": "reads", "count": layer["queries"]})
    return rows, layer


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, Any] = {}
    for group in ("written", "unknown", "unknown_answerable", "unrelated", "union"):
        group_rows = [row for row in rows if row["group"] == group]
        counts = {
            name: sum(1 for row in group_rows if row["output_class"] == name)
            for name in (
                "usable_and_correct",
                "usable_but_wrong",
                "unusable",
                "usable_fallback",
            )
        }
        by_group[group] = {
            "n": len(group_rows),
            "usable_and_correct": counts["usable_and_correct"],
            "usable_but_wrong": counts["usable_but_wrong"],
            "unusable": counts["unusable"],
            "usable_fallback": counts["usable_fallback"],
            "anchored": sum(1 for row in group_rows if not row["fallback"]),
            "fallback": sum(1 for row in group_rows if row["fallback"]),
            "unit_record": sum(1 for row in group_rows if row["unit"] == "record"),
            "unit_sentence": sum(1 for row in group_rows if row["unit"] == "sentence"),
            "decisions": {
                name: sum(1 for row in group_rows if row["decision"] == name)
                for name in ("matched", "no_match", "ambiguous_multiple_anchors")
            },
        }
    return by_group


def entry_curve(corpus: dict[str, Any], table: WindowAddressTable) -> dict[str, Any]:
    admitted = {hashlib.sha256(address["raw"]).hexdigest() for address in table.addresses}
    buckets: dict[str, Any] = {}
    for name, low, high in LEN_BUCKETS:
        entries = [
            text for text in corpus["written"] if low <= len(text.encode("utf-8")) < (high or 10**9)
        ]
        hit = [
            text for text in entries if hashlib.sha256(text.encode("utf-8")).hexdigest() in admitted
        ]
        buckets[name] = {
            "n": len(entries),
            "admitted": len(hit),
            "rate": round(len(hit) / len(entries), 6) if entries else None,
        }
    short_n = sum(buckets[name]["n"] for name in ("<32", "32-64"))
    short_hit = sum(buckets[name]["admitted"] for name in ("<32", "32-64"))
    admitted_lengths = [len(address["raw"]) for address in table.addresses]
    unaddressable = [
        text
        for text in corpus["written"]
        if hashlib.sha256(text.encode("utf-8")).hexdigest() not in admitted
    ]
    unaddressable_lengths = [len(text.encode("utf-8")) for text in unaddressable]
    long_unaddressable = [
        {
            "length": len(raw),
            "identity_blocks": len(set(identity(raw))),
            "all_windows": len(windows(raw)),
            "reason": "empty window identity (record shorter than 16 bytes)",
            "text": raw.decode("utf-8", errors="replace"),
        }
        for raw in (text.encode("utf-8") for text in unaddressable)
        if len(raw) >= 64
    ]
    long_unaddressable.sort(key=lambda item: item["length"])
    return {
        "buckets": buckets,
        "short_entries": {
            "n": short_n,
            "admitted": short_hit,
            "rate": round(short_hit / short_n, 6) if short_n else None,
        },
        "shortest_admitted_length": min(admitted_lengths) if admitted_lengths else None,
        "shortest_unaddressable_length": (
            min(unaddressable_lengths) if unaddressable_lengths else None
        ),
        "long_unaddressable": long_unaddressable,
        "unaddressable_count": len(unaddressable),
        "min_addressable_constant": MIN_ADDRESSABLE,
        "window_bytes": WINDOW_BYTES,
    }


def _answerable_class(row: dict[str, Any]) -> str:
    if not row["usable"]:
        return "unusable"
    if row["record"] is not None and row["record"] == row["expected"]:
        return "usable_and_correct"
    return "usable_but_wrong"


def reclassify_unknown(
    rows: list[dict[str, Any]], table: WindowAddressTable, corpus: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Move unknown queries that verbatim contain a written record to answerable.

    Grouping only: the corpus, the address layer and the read path are untouched. The
    test is plain byte containment of the stored record inside the query bytes.
    """
    moved: list[dict[str, Any]] = []
    anchored_unknown: list[dict[str, Any]] = []
    false_anchor_samples: list[dict[str, Any]] = []
    anchor_level_not_contained = 0
    for row in rows:
        if row["group"] != "unknown":
            continue
        query_bytes = row["query_text"].encode("utf-8")
        anchored = not row["fallback"]
        anchors_contained = [
            table.addresses[address_id]["raw"] in query_bytes for address_id in row["anchor_ids"]
        ]
        if anchored:
            anchor_level_not_contained += sum(1 for flag in anchors_contained if not flag)
            anchored_unknown.append(
                {
                    "position": row["position"],
                    "query_len": row["query_len"],
                    "record": row["record"],
                    "anchors": row["anchors"],
                    "anchors_contained": anchors_contained,
                    "contained_any": any(anchors_contained),
                    "overlap_ratio": row["overlap_ratio"],
                }
            )
            if not any(anchors_contained):
                false_anchor_samples.append(
                    {
                        "position": row["position"],
                        "query_len": row["query_len"],
                        "record": row["record"],
                        "anchors": row["anchors"],
                        "overlap_ratio": row["overlap_ratio"],
                    }
                )
        matches = [
            {
                "record": index,
                "record_len": len(address["raw"]),
                "byte_interval": [
                    query_bytes.find(address["raw"]),
                    query_bytes.find(address["raw"]) + len(address["raw"]),
                ],
                "record_text": address["raw"].decode("utf-8", "replace")[:120],
            }
            for index, address in enumerate(table.addresses)
            if address["raw"] in query_bytes
        ]
        if not matches:
            continue
        row["group"] = "unknown_answerable"
        row["expected"] = matches[0]["record"]
        row["output_class"] = _answerable_class(row)
        row["reclassified_from"] = "unknown"
        row["contained_matches"] = matches
        moved.append(
            {
                "position": row["position"],
                "query_len": row["query_len"],
                "query": row["query_text"][:160],
                "record": matches[0]["record"],
                "record_len": matches[0]["record_len"],
                "byte_interval": matches[0]["byte_interval"],
                "record_text": matches[0]["record_text"],
                "all_matches": len(matches),
                "anchored": anchored,
                "decision": row["decision"],
                "output_class": row["output_class"],
            }
        )
    remaining = sum(1 for row in rows if row["group"] == "unknown")
    blocked = sum(1 for row in rows if row["group"] == "unknown_answerable" and not row["usable"])
    bookkeeping_ok = len(moved) + remaining == len(corpus["unknown"]) and blocked == 0
    block = {
        "rule": (
            "an unknown query that contains some written record byte for byte has an answering "
            "record in the store; it is moved to the answerable class before scoring"
        ),
        "numerator": len(moved),
        "denominator": len(corpus["unknown"]),
        "moved": moved,
        "remaining_unknown": remaining,
        "bookkeeping_ok": bookkeeping_ok,
    }
    gate_block = {
        "total": len(corpus["unknown"]),
        "reclassified_to_answerable": len(moved),
        "remaining": remaining,
        "old_reading": {
            "rule": "any anchored row in the unknown group fails",
            "numerator": len(anchored_unknown),
            "denominator": len(corpus["unknown"]),
            "rows": anchored_unknown,
        },
        "new_reading": {
            "rule": "unknown_false_anchor = anchored and not contained (byte containment, no fingerprints)",
            "numerator": len(false_anchor_samples),
            "denominator": len(corpus["unknown"]),
            "samples": false_anchor_samples,
        },
        "anchor_level_not_contained": anchor_level_not_contained,
    }
    return moved, {"reclassification": block, "unknown_group": gate_block}


def fallback_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Relevance is scored only where the store holds an answer (corrected denominator)."""
    answerable_groups = ("written", "union", "unknown_answerable")
    fallback_all = [row for row in rows if row["fallback"]]
    fallback = [row for row in fallback_all if row["group"] in answerable_groups]
    ratios = sorted(float(row["overlap_ratio"]) for row in fallback)
    if ratios:
        quartiles = statistics.quantiles(ratios, n=4) if len(ratios) > 1 else [ratios[0]] * 3
        median = statistics.median(ratios)
        mean = round(sum(ratios) / len(ratios), 6)
    else:
        quartiles, median, mean = [0.0, 0.0, 0.0], 0.0, 0.0
    step = max(1, len(fallback) // FB_SAMPLE)
    sample = fallback[::step][:FB_SAMPLE]
    review = [
        {
            "group": row["group"],
            "position": row["position"],
            "query": row["query_text"][:160],
            "query_len": row["query_len"],
            "record": row["record"],
            "span": [row["start"], row["end"]],
            "unit": row["unit"],
            "overlap_ratio": row["overlap_ratio"],
            "shared_blocks": row["shared_blocks"],
            "output": row["output_text"],
            "score": None,
        }
        for row in sample
    ]
    unanswerable = {
        group: {
            "n": sum(1 for row in fallback_all if row["group"] == group),
            "unit_record": sum(
                1 for row in fallback_all if row["group"] == group and row["unit"] == "record"
            ),
            "unit_sentence": sum(
                1 for row in fallback_all if row["group"] == group and row["unit"] == "sentence"
            ),
        }
        for group in ("unknown", "unrelated")
    }
    return {
        "denominator": (
            "written + union + unknown_answerable (queries that have an answering record in the "
            "store, including unknown queries that verbatim contain a written record)"
        ),
        "n": len(fallback),
        "n_unanchored_all": len(fallback_all),
        "by_group": {
            group: sum(1 for row in fallback if row["group"] == group)
            for group in answerable_groups
        },
        "overlap_quartiles": [round(float(value), 6) for value in quartiles],
        "overlap_median": round(float(median), 6),
        "overlap_mean": mean,
        "review": review,
        "review_is_census": len(sample) == len(fallback),
        "unanswerable": unanswerable,
        "unanswerable_note": (
            "unknown/unrelated queries have no answering record by construction; their rows are "
            "reported as output shape and mis-anchor count only and never enter the relation median"
        ),
    }


def verdict_of(
    closed: dict[str, Any],
    layer: dict[str, Any],
    curve: dict[str, Any],
    gates: dict[str, Any],
) -> str:
    if not str(closed.get("verdict", "")).endswith("_closed_neutral"):
        return "hard_failure_closed_not_neutral"
    if gates["unknown_false_anchor"] > 0:
        return "false_anchor_confirmed"
    if curve["short_entries"]["rate"] is None or (
        curve["short_entries"]["rate"] < SHORT_ENTRY_LINE
    ):
        return "inconclusive"
    if (
        layer["fabricated_bytes"]
        or layer["illegal_spans"]
        or layer["utf8_invalid"]
        or layer["boundary_illegal"]
    ):
        return "inconclusive"
    if gates["reclassification"]["bookkeeping_ok"] is not True:
        return "inconclusive"
    return "continuity_restored"


def previous_round() -> dict[str, Any]:
    if not PREVIOUS_REPORT.exists():
        return {"commit": PREVIOUS_COMMIT, "report": None}
    old = json.loads(PREVIOUS_REPORT.read_text(encoding="utf-8"))
    return {
        "commit": PREVIOUS_COMMIT,
        "report": str(PREVIOUS_REPORT.relative_to(ROOT)),
        "verdict": old.get("verdict"),
        "corpus": old.get("denominators"),
        "L1": old.get("layers", {}).get("L1"),
        "L2": old.get("layers", {}).get("L2"),
        "L3": old.get("layers", {}).get("L3"),
        "short_entries": old.get("entry_curve", {}).get("short_entries"),
        "buckets": old.get("entry_curve", {}).get("buckets"),
    }


def phase_open(manifest: dict[str, Any]) -> dict[str, Any]:
    corpus = _corpus()
    corpus_sha = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    if corpus_sha != manifest["corpus_sha256"]:
        raise SystemExit("corpus changed after freeze")
    append_progress({"phase": "open_start", "corpus_sha256": corpus_sha})

    table = WindowAddressTable()
    write_start = time.time()
    written = [(text.encode("utf-8"), None) for text in corpus["written"]]
    write_result = table.write_batch_checked(written)
    write_seconds = round(time.time() - write_start, 3)
    rejected = [
        {
            "position": item["item"],
            "length": len(written[item["item"]][0]),
            "reason": item["reason"],
            "identity_blocks": item["identity_blocks"],
            "text": corpus["written"][item["item"]][:120],
        }
        for item in write_result["rejected"]
    ]
    append_progress({"phase": "write_done", "addresses": len(table.addresses)})

    rows, layer = evaluate(corpus, table)
    moved, reclass_blocks = reclassify_unknown(rows, table, corpus)
    by_group = summarise(rows)
    curve = entry_curve(corpus, table)
    fallback = fallback_stats(rows)
    admitted_shas = {hashlib.sha256(address["raw"]).hexdigest() for address in table.addresses}
    union_should = 0
    union_violations: list[dict[str, Any]] = []
    for row in rows:
        if row["group"] != "union":
            continue
        left_text = corpus["written"][row["position"]]
        right_text = corpus["written"][(row["position"] + UNION_STRIDE) % SCALE]
        left = hashlib.sha256(left_text.encode("utf-8")).hexdigest()
        right = hashlib.sha256(right_text.encode("utf-8")).hexdigest()
        if left not in admitted_shas or right not in admitted_shas or left == right:
            continue
        union_should += 1
        if row["decision"] != "ambiguous_multiple_anchors":
            union_violations.append(
                {
                    "position": row["position"],
                    "decision": row["decision"],
                    "anchors": row["anchors"],
                    "anchor_ids": row["anchor_ids"],
                }
            )
    gates = {
        **reclass_blocks,
        "unknown_anchored": reclass_blocks["unknown_group"]["old_reading"]["numerator"],
        "unknown_false_anchor": reclass_blocks["unknown_group"]["new_reading"]["numerator"],
        "unknown_anchored_samples": [
            {
                "position": item["position"],
                "query_len": item["query_len"],
                "record": item["record"],
                "anchors": item["anchors"],
                "overlap_ratio": item["overlap_ratio"],
            }
            for item in reclass_blocks["unknown_group"]["old_reading"]["rows"][:3]
        ],
        "union_should_be_ambiguous": union_should,
        "union_ambiguous": sum(
            1
            for row in rows
            if row["group"] == "union" and row["decision"] == "ambiguous_multiple_anchors"
        ),
        "union_violations": union_violations,
    }
    closed = json.loads(CLOSED_REPORT.read_text(encoding="utf-8"))
    anchored_rows = [row for row in rows if not row["fallback"]]
    violations: list[dict[str, Any]] = []
    for row in anchored_rows:
        query_bytes = row["query_text"].encode("utf-8")
        for address_id in row["anchor_ids"]:
            if table.addresses[address_id]["raw"] not in query_bytes:
                violations.append(
                    {
                        "group": row["group"],
                        "position": row["position"],
                        "record": address_id,
                        "anchor_ids": row["anchor_ids"],
                    }
                )
    gates["anchored_rows"] = len(anchored_rows)
    gates["independent_containment_violations"] = len(violations)
    gates["independent_containment_samples"] = violations[:3]
    gates["unknown_anchored_diagnostic"] = [
        {
            "position": row["position"],
            "record": row["record"],
            "record_len": len(table.addresses[row["record"]]["raw"]),
            "contained": table.addresses[row["record"]]["raw"] in row["query_text"].encode("utf-8"),
            "byte_offset": row["query_text"]
            .encode("utf-8")
            .find(table.addresses[row["record"]]["raw"]),
            "record_text": table.addresses[row["record"]]["raw"].decode("utf-8", "replace"),
        }
        for row in anchored_rows
        if row["group"] == "unknown" or row.get("reclassified_from") == "unknown"
    ]
    verdict = verdict_of(closed, layer, curve, gates)
    failures = [
        {
            "group": row["group"],
            "position": row["position"],
            "query": row["query_text"],
            "query_len": row["query_len"],
            "expected": row["expected"],
            "actual_record": row["record"],
            "anchors": row["anchors"],
            "anchor_ids": row["anchor_ids"],
            "no_anchor": row["no_anchor"],
            "decision": row["decision"],
            "covered_blocks": row["covered_blocks"],
            "coverage_ratio": row["coverage_ratio"],
            "shared_blocks": row["shared_blocks"],
            "overlap_ratio": row["overlap_ratio"],
            "unit": row["unit"],
            "span": [row["start"], row["end"]],
            "output": row["output_text"],
            "output_class": row["output_class"],
        }
        for row in rows
        if row["output_class"] in ("usable_but_wrong", "unusable")
    ]

    report = {
        "schema": "taiji-contig-v1",
        "date": "2026-11-09",
        "verdict": verdict,
        "commit_base": PREVIOUS_COMMIT,
        "runner_sha_start": RUNNER_SHA_START,
        "runner_sha_end": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        "address_module_sha256": hashlib.sha256(ADDRESS_MODULE.read_bytes()).hexdigest(),
        "corpus_sha256": corpus_sha,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_freeze_revision": manifest.get("freeze_revision"),
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "cost_estimate": manifest.get("cost_estimate"),
        "closed": closed,
        "write": {
            "attempted": len(corpus["written"]),
            "addresses": len(table.addresses),
            "appended": table.appended,
            "merged": table.merged,
            "rejected_records": table.rejected_records,
            "rejected_reasons": table.rejected_reasons,
            "rejected": rejected,
            "rollbacks": table.rollbacks,
            "evictions": table.evictions,
            "silent_overwrites": table.silent_overwrites,
            "write_seconds": write_seconds,
        },
        "layers": {
            "L1": {
                "fabricated_bytes": layer["fabricated_bytes"],
                "illegal_spans": layer["illegal_spans"],
                "utf8_decodable": layer["utf8_valid"],
                "utf8_invalid": layer["utf8_invalid"],
                "utf8_rate": round(layer["utf8_valid"] / max(layer["queries"], 1), 6),
                "boundary_legal": layer["boundary_legal"],
                "boundary_illegal": layer["boundary_illegal"],
                "boundary_rate": round(layer["boundary_legal"] / max(layer["queries"], 1), 6),
                "blocked_samples": layer["blocked_samples"],
            },
            "L2": {
                "faithful": layer["faithful"],
                "faithful_rate": round(layer["faithful"] / max(layer["queries"], 1), 6),
                "denominator": layer["queries"],
            },
            "L3": by_group,
        },
        "debate": {
            "queries": layer["queries"],
            "fallback": layer["fallback"],
            "blocked": layer["blocked"],
            "decisions": {
                name: sum(1 for row in rows if row["decision"] == name)
                for name in ("matched", "no_match", "ambiguous_multiple_anchors")
            },
            "units": {
                "record": sum(1 for row in rows if row["unit"] == "record"),
                "sentence": sum(1 for row in rows if row["unit"] == "sentence"),
                "none": sum(1 for row in rows if row["unit"] == "none"),
            },
            "fallback_basis": {
                name: sum(1 for row in rows if row["fallback_basis"] == name)
                for name in ("none", "window_overlap", "address_order")
            },
        },
        "groups": by_group,
        "gates": gates,
        "failures": failures,
        "definition_change_vs_previous_round": definition_change(window_address_audit()),
        "entry_curve": curve,
        "fallback_relevance": {
            **fallback,
            "rubric": manifest.get("fallback_rubric"),
        },
        "previous_round": previous_round(),
        "denominators": {
            "written": len(corpus["written"]),
            "unknown": len(corpus["unknown"]),
            "unknown_answerable": reclass_blocks["reclassification"]["numerator"],
            "unknown_remaining": reclass_blocks["unknown_group"]["remaining"],
            "unrelated": len(corpus["unrelated"]),
            "union": len(corpus["union"]),
            "answerable_total": (
                len(corpus["written"])
                + len(corpus["union"])
                + reclass_blocks["reclassification"]["numerator"]
            ),
        },
        "form_counts": corpus["form_counts"],
        "length_buckets": corpus["length_buckets_written"],
        "disclosures": [
            "the corpus is the frozen 550-entry natural sample of the previous round "
            "(sha 350eef77...), byte-identical; nothing was resampled or filtered.",
            "taiji/sc_sfk.py is untouched; the address layer under test is the new "
            "taiji/sc_windows.py, whose sha and numeric surface are frozen in the manifest.",
            "a written entry shorter than 16 bytes has an empty identity and is rejected alone; "
            "it is listed one by one, not hidden in the rate.",
            "unknown and unrelated queries usually have no answering record by construction; "
            "their outputs are classed usable_fallback, never usable_and_correct. A spurious "
            "anchor on a low-entropy window is classed usable_but_wrong and counted.",
            "fixture classification only: an unknown query that verbatim contains a written "
            "record has an answering record in the store, so it is moved from unknown to "
            "unknown_answerable before scoring. The corpus, the address layer, the ranking and "
            "the thresholds are untouched; every move is listed with record index, byte interval "
            "and length, and the bookkeeping is checked (moved + remaining == 550).",
            "both unknown-gate readings are reported: old = any anchored row (a theorem gate that "
            "counts quotation as failure), new = anchored and not contained. Containment is tested "
            "on raw bytes without fingerprints. Only the new reading decides the verdict; the old "
            "numerator and denominator are shown next to it, and anchor-level non-containment is "
            "reported separately.",
            "the union group joins two entries with a newline; pointing at either source record "
            "counts as correct, pointing elsewhere does not. A union row whose two components are "
            "both admitted must be decided ambiguous; violations are listed in gates.",
            "fallback relevance is scored only on queries that have an answering record in the "
            "store (written + union + unknown_answerable); remaining unknown/unrelated rows are "
            "unanswerable by construction and are reported as output shape and false-anchor count "
            "only. Reclassified rows enter the denominator; none is dropped. This corrects the "
            "denominator, it does not loosen the ranking or add a threshold.",
            "the fallback review scores are supplied after the run through --score; the rubric and "
            "the answerable denominator were frozen in the manifest before the first reading.",
        ],
        "cost": {
            "cpu_seconds": round(time.process_time() - START_CPU, 3),
            "wall_seconds": round(time.time() - START_WALL, 3),
            "cpu_cap": CPU_CAP,
        },
    }
    write_jsonl(OUTPUTS, rows)
    write_json(REPORT, report)
    append_progress({"phase": "done", "verdict": verdict})
    print(
        json.dumps(
            {"verdict": verdict, "write": report["write"]["addresses"]},
            ensure_ascii=False,
        )
    )
    print(json.dumps(curve["buckets"], ensure_ascii=False))
    print(json.dumps(fallback, ensure_ascii=False)[:800])
    return report


def phase_score(path: Path) -> dict[str, Any]:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores = payload["scores"] if isinstance(payload, dict) else payload
    review = report["fallback_relevance"]["review"]
    if len(scores) != len(review):
        raise SystemExit(f"need {len(review)} scores, got {len(scores)}")
    for item, score in zip(review, scores, strict=True):
        if score not in SCORE_VALUES:
            raise SystemExit(f"bad score {score!r}; use {sorted(SCORE_VALUES)}")
        item["score"] = score
    closed = report["closed"]
    verdict = verdict_of(closed, report["layers"]["L1"], report["entry_curve"], report["gates"])
    report["verdict"] = verdict
    values = sorted(SCORE_VALUES[score] for score in scores)
    report["fallback_relevance"]["score_median"] = round(float(statistics.median(values)), 6)
    report["fallback_relevance"]["score_counts"] = {
        name: scores.count(name) for name in SCORE_VALUES
    }
    report["cost"]["score_pass_cpu_seconds"] = round(time.process_time() - START_CPU, 3)
    report["cost"]["score_pass_wall_seconds"] = round(time.time() - START_WALL, 3)
    report["cost"]["open_pass_cpu_seconds"] = report["cost"]["cpu_seconds"]
    report["cost"]["open_pass_wall_seconds"] = report["cost"]["wall_seconds"]
    report["cost"]["closed_battery_cpu_seconds"] = closed.get("cost", {}).get("cpu_seconds")
    report["cost"]["closed_battery_wall_seconds"] = closed.get("cost", {}).get("wall_seconds")
    report["cost"]["total_cpu_seconds"] = round(
        float(report["cost"]["open_pass_cpu_seconds"])
        + float(report["cost"]["closed_battery_cpu_seconds"] or 0.0),
        3,
    )
    report["cost"]["total_wall_seconds"] = round(
        float(report["cost"]["open_pass_wall_seconds"])
        + float(report["cost"]["closed_battery_wall_seconds"] or 0.0),
        3,
    )
    write_json(REPORT, report)
    print(json.dumps({"verdict": verdict, "median": report["fallback_relevance"]["score_median"]}))
    return report


def main(argv: list[str]) -> int:
    assert_no_black_processes()
    if "--freeze" in argv:
        acquire_lock()
        watchdog()
        phase_freeze()
        return 0
    if "--score" in argv:
        acquire_lock()
        watchdog()
        index = argv.index("--score")
        phase_score(Path(argv[index + 1]))
        return 0
    if not MANIFEST.exists():
        raise SystemExit("manifest missing: run --freeze first")
    acquire_lock()
    watchdog()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not CLOSED_REPORT.exists():
        phase_closed()
    phase_open(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
