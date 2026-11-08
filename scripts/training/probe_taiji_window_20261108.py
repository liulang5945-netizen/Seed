"""Window-set identity gate: identity = every 16-byte window of the record.

Frozen question: with the definition replaced (no chunking, no protection band,
minimum 16 bytes) and **nothing else changed**, does the same 550-entry natural
corpus, the same four groups, the same L1/L2/L3 readings and the same fallback
rubric hold?

Verdict lines (frozen before the first reading, first hit wins):

* closed state not neutral -> ``hard_failure_closed_not_neutral`` (stop);
* any L1 line fails -> ``output_fabricates`` (samples only, never a tolerance);
* admission rate of entries < 64 bytes < 0.90 -> ``window_set_still_gapped``
  (write the length limit down, stop adding to the address layer, go to the
  representation layer; diagnostics only);
* fallback relevance median < 0.5 -> ``fallback_still_irrelevant``;
* otherwise -> ``window_set_identity_holds`` with the verified corpus size and
  the shortest admitted entry.

Discipline: 0 training / 0 optimizer steps / 0 Adam / 0 confirm; ``process_guard``
first; single-instance lock; 60 s watchdog; CPU cap 200 s; readings flushed as
taken; write-then-rename; dual runner sha. No ``python -m black``.
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

MANIFEST = ROOT / "plans/manifests/taiji_window_20261108.json"
CORPUS = ROOT / "plans/manifests/taiji_scale_corpus_20261108.json"
REPORT = ROOT / "reports/taiji_window_20261108.json"
OUTPUTS = ROOT / "reports/taiji_window_20261108_outputs.jsonl"
CLOSED_REPORT = ROOT / "reports/taiji_window_20261108_closed.json"
PROGRESS = ROOT / "reports/taiji_window_20261108_progress.jsonl"
LOCK = ROOT / "reports/taiji_window_20261108.lock"
WATCHDOG_MARKER = ROOT / "reports/taiji_window_20261108_watchdog.json"
CPU_ABORT = ROOT / "reports/taiji_window_20261108_cpu_abort.json"
SCORES = ROOT / "reports/taiji_window_20261108_scores.json"
PREVIOUS_REPORT = ROOT / "reports/taiji_scale_20261108.json"
PREVIOUS_MANIFEST = ROOT / "plans/manifests/taiji_scale_20261108.json"
PREVIOUS_COMMIT = "84cb5cee8"
ADDRESS_MODULE = ROOT / "taiji/sc_windows.py"

CPU_CAP = 200.0
WATCHDOG_SECONDS = 60.0
LOCK_STALE_SECONDS = 600.0
SCALE = 550
UNION_STRIDE = 137
SHORT_ENTRY_LINE = 0.90
FB_SAMPLE = 20
FB_LINE = 0.5
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
        "I(r) = multiset { sha256(r[i:i+16])[:16].hex() : 0 <= i <= len(r)-16 }; one token per "
        "offset, so repeated windows keep their multiplicity. A record shorter than 16 bytes has "
        "an empty identity and is rejected alone. An empty identity is never addressable: it "
        "would be covered by every query."
    ),
    "multiplicity_choice": (
        "multiset coverage, frozen before the run: count_q(t) >= count_r(t) for every distinct "
        "window t. Reason: set inclusion lets one query occurrence pay for a window that occurs "
        "twice in the record (repeated table rows/separators); counting is one dict lookup and "
        "adds no numeric constant. Multiset coverage implies set inclusion, never the reverse."
    ),
    "protection_band": (
        "none, and none is kept. Every identity window lies fully inside the record, so any "
        "prefix/suffix leaves every window at offset |p|+i equal to r[i:i+16]. The old PROTECT=6 "
        "band belonged to a cut predicate that read bytes before the cut; a fixed window reads "
        "only its own bytes."
    ),
    "shortest_addressable_length": (
        "MIN_ADDRESSABLE = WINDOW_BYTES = 16 bytes; necessary and sufficient (length alone "
        "decides whether the identity is non-empty)."
    ),
    "anchor": (
        "anchored iff identity(r) is covered by W(q) with multiplicity; 0 anchors -> fallback "
        "but still one output; 1 -> matched; >=2 -> ambiguous_multiple_anchors, still top-1 and "
        "annotated. Anchors are readings, not exits."
    ),
    "fallback": (
        "unanchored reads are ranked by |set(I(r)) & set(W(q))| / |set(I(r))| descending, then "
        "address index ascending; fallback_basis = window_overlap when a window is shared, "
        "address_order otherwise; no pass/reject threshold."
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
        "If identity(r) is covered by W(q), q must contain every 16-byte window of r as a "
        "substring with the record's multiplicity. This does NOT imply q contains r contiguously "
        "and does not imply near-verbatim reproduction; the order is lost. Construction: "
        "q = r[0:16] + r[1:17] + ... covers every window while never containing r as a "
        "contiguous run. Verdict: 'almost verbatim reproduction is required' cannot be derived."
    ),
    "failure_constructions": [
        "empty identity kept: covered by every query, anchors every sub-16-byte record; floor one.",
        "set inclusion instead of multiplicity: a repeated window is paid by one query occurrence.",
        "low-entropy windows: a 16-byte run of spaces is shared by any query containing it, so an "
        "unrelated query can anchor that record; reported per group, not hidden.",
        "order loss: window coverage is not contiguity (see impersonation_cost).",
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
    "C4_L3": "per group: usable_and_correct / usable_but_wrong / usable_fallback / unusable.",
    "C5_short": "admission rate for entries < 64 bytes >= 0.90.",
    "C6_fallback": (
        "20 fallback rows sampled mechanically; each printed sentence is scored relevant / "
        "marginal / irrelevant against its query; median >= 0.5. The denominator is frozen as "
        "**all unanchored rows** (same rule as the previous round); the written-group subset "
        "(queries that do have a record in the store) is reported separately as a diagnostic, "
        "because unknown/unrelated queries have no answering record by construction."
    ),
    "V_hard": "closed state not neutral -> hard_failure_closed_not_neutral.",
    "V_fabricates": "any L1 line fails -> output_fabricates, samples only.",
    "V_gap": (
        "admission rate for entries < 64 bytes < 0.90 -> window_set_still_gapped: write the "
        "length limit down, stop adding to the address layer, go to the representation layer."
    ),
    "V_fallback": "fallback relevance median < 0.5 -> fallback_still_irrelevant.",
    "V_holds": "all lines met -> window_set_identity_holds.",
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
        "schema": "taiji-window-v1",
        "date": "2026-11-08",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runner_sha256": RUNNER_SHA_START,
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "corpus_path": str(CORPUS.relative_to(ROOT)),
        "question": (
            "does replacing the identity with the multiset of all 16-byte windows (no band, "
            "minimum 16 bytes) admit short entries and keep L1/L2 and fallback quality on the "
            "same 550-entry natural corpus?"
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
            "fallback_median": FB_LINE,
            "fallback_sample": FB_SAMPLE,
        },
        "fallback_rubric": {
            "relevant": "the printed sentence answers the query's question or states its point",
            "marginal": "same document/topic but does not answer the question",
            "irrelevant": "different topic or generic boilerplate",
            "mapping": SCORE_VALUES,
            "denominator": (
                "all unanchored rows (frozen); the written-group subset is reported separately"
            ),
            "sample_rule": (
                "unanchored rows in report order, every ceil(total/20)-th row, first 20"
            ),
        },
        "verdict": {
            "rules": [
                "closed not neutral -> hard_failure_closed_not_neutral (stop)",
                "any L1 line fails -> output_fabricates",
                (
                    f"admission rate for entries < 64 bytes < {SHORT_ENTRY_LINE} -> "
                    "window_set_still_gapped"
                ),
                f"fallback relevance median < {FB_LINE} -> fallback_still_irrelevant",
                "otherwise -> window_set_identity_holds",
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
        if not (same_verdict and same_corpus):
            raise SystemExit("manifest already frozen with a different verdict/corpus")
        manifest["freeze_revision"] = int(old.get("freeze_revision", 1)) + 1
        manifest["freeze_note"] = "harness-only revision: criteria and corpus unchanged"
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
    for group in ("written", "unknown", "unrelated", "union"):
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


def fallback_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fallback = [row for row in rows if row["fallback"]]
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
    written = [row for row in fallback if row["group"] == "written"]
    written_ratios = sorted(float(row["overlap_ratio"]) for row in written)
    return {
        "n": len(fallback),
        "overlap_quartiles": [round(float(value), 6) for value in quartiles],
        "overlap_median": round(float(median), 6),
        "overlap_mean": mean,
        "review": review,
        "written_subset": {
            "n": len(written),
            "overlap_median": (
                round(float(__import__("statistics").median(written_ratios)), 6)
                if written_ratios
                else None
            ),
            "note": (
                "queries that do have a record in the store but were not anchored; the "
                "verdict-binding denominator stays all unanchored rows as frozen"
            ),
        },
    }


def verdict_of(
    closed: dict[str, Any], layer: dict[str, Any], curve: dict[str, Any], fb: dict[str, Any]
) -> str:
    if not str(closed.get("verdict", "")).endswith("_closed_neutral"):
        return "hard_failure_closed_not_neutral"
    if (
        layer["fabricated_bytes"]
        or layer["illegal_spans"]
        or layer["utf8_invalid"]
        or layer["boundary_illegal"]
    ):
        return "output_fabricates"
    if curve["short_entries"]["rate"] is None or (
        curve["short_entries"]["rate"] < SHORT_ENTRY_LINE
    ):
        return "window_set_still_gapped"
    scores = [item["score"] for item in fb["review"]]
    if any(score is None for score in scores):
        return "pending_fallback_review"
    values = sorted(SCORE_VALUES[score] for score in scores)
    if statistics.median(values) < FB_LINE:
        return "fallback_still_irrelevant"
    return "window_set_identity_holds"


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
    by_group = summarise(rows)
    curve = entry_curve(corpus, table)
    fallback = fallback_stats(rows)
    closed = json.loads(CLOSED_REPORT.read_text(encoding="utf-8"))
    verdict = verdict_of(closed, layer, curve, fallback)
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
        "schema": "taiji-window-v1",
        "date": "2026-11-08",
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
        "failures": failures,
        "definition_change_vs_previous_round": definition_change(window_address_audit()),
        "entry_curve": curve,
        "fallback_relevance": {
            **fallback,
            "line": FB_LINE,
            "rubric": manifest.get("fallback_rubric"),
        },
        "previous_round": previous_round(),
        "denominators": {
            "written": len(corpus["written"]),
            "unknown": len(corpus["unknown"]),
            "unrelated": len(corpus["unrelated"]),
            "union": len(corpus["union"]),
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
            "the union group joins two entries with a newline; pointing at either source record "
            "counts as correct, pointing elsewhere does not.",
            "the fallback review scores are supplied after the run through --score; the rubric "
            "and the pass line were frozen in the manifest before the first reading.",
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
    verdict = verdict_of(
        closed, report["layers"]["L1"], report["entry_curve"], report["fallback_relevance"]
    )
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
