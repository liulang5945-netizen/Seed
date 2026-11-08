"""Scale recheck: the same frozen address definition on a 10x natural corpus.

What is under test
------------------
Nothing in ``taiji/sc_sfk.py`` changes.  This run swaps the data only: the same
chunking, identity, anchor, output-unit and fallback definitions are replayed on
a corpus sampled mechanically from the repository's own natural text.

Corpus (frozen before the first reading, see ``build_corpus``)
--------------------------------------------------------------
* Written / unknown pool: every non-empty line of every ``plans/**/*.md`` file
  plus every maximal run of ``#`` comment lines of every ``taiji/**/*.py`` file,
  stripped, exact-duplicate removed, ordered by ``sha256(text)``.
* Form floor: 20 entries per form (``heading``, ``list_item``, ``table_row``,
  ``prose``, ``code_comment``) first, then the hash order fills the rest, so the
  realistic forms are present in every run.  No entry is ever filtered by
  length, cut point or addressability.
* Written = 550 picked entries; unknown = the next 550 in hash order (never
  written); unrelated = string values of every eighth ``plans/manifests/*.json``
  file (a different source), exact-duplicates of written/unknown removed;
  union = ``written[i] + "\\n" + written[(i + 137) % 550]``.
* The manifest stores every entry verbatim, the source list with shas, the form
  counts and the length-bucket counts, so the corpus cannot move after freezing.

Scoring (same frozen口径 as the output-unit round)
--------------------------------------------------
L1 fabricated bytes = 0, illegal slices = 0, UTF-8 decodable = 100%, boundary
legal = 100%; L2 output == claimed record's ``bytes[start:end]``; L3
``usable_and_correct`` / ``usable_but_wrong`` / ``unusable``.  For a written
query whose record was never admitted the expected address does not exist, so
any output is ``usable_but_wrong`` (complete and decodable, but it cannot be the
source).  The three columns are reported per group; no denominator is dropped.

The two questions this run must answer
--------------------------------------
1. Length -> admission curve in the frozen buckets (<32 / 32-64 / 64-128 /
   128-256 / >256 bytes), the shortest admitted entry, the shortest observed
   unaddressable entry, and every entry of >= 64 bytes with no identity chunk
   (length + reason, one by one).
2. Fallback relevance: block-overlap quartiles of unanchored reads, plus 20
   sampled fallback outputs scored relevant / marginal / irrelevant by reading
   the query and the printed sentence.  The rubric and the line are frozen in
   the manifest before the run: relevant = 1, marginal = 0.5, irrelevant = 0,
   pass line = median >= 0.5.  Scores are supplied after the run through
   ``--score``; the runner never invents them.

Verdict (frozen)
----------------
closed not neutral -> hard failure, stop; any L1 failure ->
``scale_recheck_unusable``; entry rate below 64 bytes < 0.90 ->
``scale_recheck_short_entries_gap`` (write the limit down, stop adding to the
address layer, go to the representation layer); fallback median below the frozen
line -> ``fallback_irrelevant``; all lines met -> ``scale_recheck_holds``.

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
import re
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
from taiji.sc_sfk import (  # noqa: E402
    MIN_ADDRESSABLE,
    MIN_CHUNK,
    PROTECT,
    AddressTable,
    chunk_address_audit,
    chunks,
    identity,
    sentences,
)

MANIFEST = ROOT / "plans/manifests/taiji_scale_20261108.json"
CORPUS = ROOT / "plans/manifests/taiji_scale_corpus_20261108.json"
REPORT = ROOT / "reports/taiji_scale_20261108.json"
OUTPUTS = ROOT / "reports/taiji_scale_20261108_outputs.jsonl"
CLOSED_REPORT = ROOT / "reports/taiji_scale_20261108_closed.json"
PROGRESS = ROOT / "reports/taiji_scale_20261108_progress.jsonl"
LOCK = ROOT / "reports/taiji_scale_20261108.lock"
WATCHDOG_MARKER = ROOT / "reports/taiji_scale_20261108_watchdog.json"
SCORES = ROOT / "reports/taiji_scale_20261108_scores.json"
PREVIOUS_REPORT = ROOT / "reports/taiji_output_unit_20261108.json"
PREVIOUS_MANIFEST = ROOT / "plans/manifests/taiji_output_unit_20261108.json"
PREVIOUS_COMMIT = "1875cb7f6"
ADDRESS_MODULE = ROOT / "taiji/sc_sfk.py"

CPU_CAP = 200.0
WATCHDOG_SECONDS = 60.0
LOCK_STALE_SECONDS = 600.0
SCALE = 550
FORM_FLOOR = 20
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
FORMS = ("heading", "list_item", "table_row", "prose", "code_comment")
SCORE_VALUES = {"relevant": 1.0, "marginal": 0.5, "irrelevant": 0.0}

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
START_CPU = time.process_time()
START_WALL = time.time()
LAST_READING = [time.time()]

DEFINITION = {
    "chunking": (
        "gear hash, rolling = ((rolling << 1) + gear[byte]) & 0xFFFFFFFFFFFFFFFF; cut at every "
        "position where (rolling & 0x3F) == 0; no minimum length, no maximum length, no "
        "distance rule; chunks shorter than 16 bytes are dropped at emission; fingerprint = "
        "sha256(chunk)[:16].hex()."
    ),
    "identity": (
        "S(r) = ordered fingerprints of every chunk [a, b) with a >= PROTECT = 6 and "
        "b < len(r); an empty S(r) is rejected alone. A record with no such chunk has no "
        "identity and cannot be addressed, however long it is."
    ),
    "shortest_addressable_length": (
        "MIN_CHUNK + PROTECT + 1 = 16 + 6 + 1 = 23 bytes; necessary, not sufficient."
    ),
    "anchor": (
        "anchored iff S(r) is a contiguous subsequence of C(Q), the ordered fingerprints of all "
        "query chunks; floor one block. Anchors are readings, not exits."
    ),
    "output_unit": (
        "anchored reads print the whole record (unit = record); unanchored reads print the "
        "chosen record's sentence with the highest block overlap (unit = sentence, ties by "
        "sentence order); spans come from sentences() and always sit on UTF-8 character "
        "boundaries at a terminator/newline or the record end."
    ),
    "fallback": (
        "unanchored reads are labelled fallback = true and ranked by multiset Jaccard "
        "(shared / (|S(r)| + |C(Q)| - shared)) descending, then address order; "
        "fallback_basis = record_overlap when a block is shared, address_order otherwise; no "
        "pass/reject threshold."
    ),
    "write": (
        "append-only; a record is rejected alone when it has no identity chunk, when an equal "
        "identity holds a different value, or when its identity and an existing identity are "
        "contiguous subsequences of one another."
    ),
}

CRITERIA = {
    "C1_closed_state_neutral": (
        "the 176-case battery of the frozen output-unit gate replays bit for bit: max|delta| = "
        "0 in both the fast and the probability layer, parameters / state / identity / store "
        "signature unchanged, no table allocated, no half-open branch."
    ),
    "C2_corpus": (
        ">= 500 written entries, >= 10x the 58-record fixture; forms and length buckets "
        "reported before the run; nothing filtered by cut point or addressability."
    ),
    "C3_no_tolerance": (
        "no code change at all this round: the frozen constant inventory and read-path "
        "literal set must equal the running module, otherwise scale_recheck_unusable."
    ),
    "L1_mechanical": (
        "fabricated bytes = 0; illegal slices = 0; UTF-8 decodable = 100%; boundary legal = "
        "100% (span is the whole record or exactly one span returned by sentences())."
    ),
    "L2_content": "output == claimed record's bytes[start:end] on every scored row.",
    "L3_selection": (
        "three columns per group: usable_and_correct (complete, decodable, claimed record == "
        "expected), usable_but_wrong (complete and decodable but wrong record; a written query "
        "whose own record was never admitted is always wrong), unusable (L1 blocked)."
    ),
    "Q1_length_curve": (
        "admission rate per frozen length bucket; shortest admitted entry; shortest observed "
        "unaddressable entry; every entry >= 64 bytes with no identity chunk listed with its "
        "length and reason."
    ),
    "Q2_fallback_relevance": (
        "for unanchored outputs: block-overlap quartiles; 20 sampled outputs scored relevant / "
        "marginal / irrelevant against the query by reading the printed sentence. Pass line: "
        "median >= 0.5."
    ),
    "V_hard_failure": "closed state not neutral -> stop.",
    "V_unusable": "any L1 line fails -> scale_recheck_unusable, samples only, no loosening.",
    "V_short_gap": (
        "admission rate for entries shorter than 64 bytes < 0.90 -> "
        "scale_recheck_short_entries_gap: write the length/address limit down, stop adding to "
        "the address layer, go to the representation layer."
    ),
    "V_fallback": "fallback relevance median < 0.5 -> fallback_irrelevant, reported as measured.",
    "V_holds": "all lines met -> scale_recheck_holds, with the verified corpus size and tail.",
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
                    ROOT / "reports/taiji_scale_20261108_cpu_abort.json",
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


def _md_form(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith("#"):
        return "heading"
    if re.match(r"^([-*+]|\d+[.)])\s", stripped):
        return "list_item"
    if stripped.startswith("|"):
        return "table_row"
    return "prose"


def _pool() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sorted((ROOT / "plans").rglob("*.md")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            text = line.strip()
            if text:
                items.append(
                    {
                        "text": text,
                        "form": _md_form(text),
                        "source": str(path.relative_to(ROOT)),
                        "line": line_no,
                    }
                )
    for path in sorted((ROOT / "taiji").rglob("*.py")):
        run: list[str] = []
        start = 0
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines + [""], 1):
            if index <= len(lines) and line.lstrip().startswith("#"):
                if not run:
                    start = index
                run.append(line.strip())
            elif run:
                text = "\n".join(run).strip()
                if text:
                    items.append(
                        {
                            "text": text,
                            "form": "code_comment",
                            "source": str(path.relative_to(ROOT)),
                            "line": start,
                        }
                    )
                run = []
    return items


def _unique(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for item in items:
        key = hashlib.sha256(item["text"].encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        kept.append(item | {"sha256": key})
    kept.sort(key=lambda item: item["sha256"])
    return kept


def _unrelated_pool(exclude: set[str]) -> list[dict[str, Any]]:
    manifests = sorted((ROOT / "plans/manifests").glob("*.json"))
    values: list[dict[str, Any]] = []
    seen: set[str] = set()

    def walk(node: Any, source: str) -> None:
        if isinstance(node, str):
            text = node.strip()
            key = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""
            if text and key not in exclude and key not in seen:
                seen.add(key)
                values.append({"text": text, "form": "json_string", "source": source, "line": 0})
        elif isinstance(node, list):
            for child in node:
                walk(child, source)
        elif isinstance(node, dict):
            for child in node.values():
                walk(child, source)

    for index, path in enumerate(manifests):
        if index % 8:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        walk(payload, str(path.relative_to(ROOT)))
    return values


def build_corpus() -> dict[str, Any]:
    pool = _unique(_pool())
    by_form: dict[str, list[dict[str, Any]]] = {form: [] for form in FORMS}
    for item in pool:
        by_form[item["form"]].append(item)
    picked: list[dict[str, Any]] = []
    picked_keys: set[str] = set()
    floor_by_form: dict[str, int] = {}
    for form in FORMS:
        take = by_form[form][:FORM_FLOOR]
        floor_by_form[form] = len(take)
        for item in take:
            if item["sha256"] not in picked_keys:
                picked_keys.add(item["sha256"])
                picked.append(item)
    for item in pool:
        if len(picked) >= SCALE:
            break
        if item["sha256"] not in picked_keys:
            picked_keys.add(item["sha256"])
            picked.append(item)
    if len(picked) < SCALE:
        raise SystemExit(f"pool too small: {len(picked)} < {SCALE}")
    picked.sort(key=lambda item: item["sha256"])
    remaining = [item for item in pool if item["sha256"] not in picked_keys]
    unknown = remaining[:SCALE]
    if len(unknown) < SCALE:
        raise SystemExit(f"unknown pool too small: {len(unknown)} < {SCALE}")
    written_texts = [item["text"] for item in picked]
    exclude = {item["sha256"] for item in picked} | {item["sha256"] for item in unknown}
    unrelated = _unique(_unrelated_pool(exclude))[:SCALE]
    if len(unrelated) < SCALE:
        raise SystemExit(f"unrelated pool too small: {len(unrelated)} < {SCALE}")
    union = [
        written_texts[index] + "\n" + written_texts[(index + UNION_STRIDE) % len(written_texts)]
        for index in range(SCALE)
    ]
    sources = sorted({item["source"] for item in picked + unknown})
    source_shas = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in sources
        if (ROOT / path).exists()
    }
    form_counts = {
        group: {form: sum(1 for item in items if item["form"] == form) for form in FORMS}
        for group, items in (
            ("written", picked),
            ("unknown", unknown),
            ("unrelated", unrelated),
        )
    }
    form_counts["unrelated"]["json_string"] = len(unrelated)
    length_counts = {
        name: sum(1 for text in written_texts if low <= len(text.encode("utf-8")) < (high or 10**9))
        for name, low, high in LEN_BUCKETS
    }
    return {
        "schema": "taiji-scale-corpus-v1",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rules": {
            "written_unknown_pool": (
                "every non-empty stripped line of plans/**/*.md; every maximal run of # comment "
                "lines of taiji/**/*.py; exact duplicates removed; ordered by sha256(text)"
            ),
            "form_floor": (
                f"first {FORM_FLOOR} entries of each form by hash order, then hash order fills "
                f"the rest; no entry filtered by length, cut point or addressability"
            ),
            "split": (
                f"written = first {SCALE}; unknown = next {SCALE} in hash order, never written"
            ),
            "unrelated": (
                "string values of every eighth plans/manifests/*.json file, exact duplicates "
                "of written/unknown removed, hash order, first "
                f"{SCALE}"
            ),
            "union": f"written[i] + newline + written[(i + {UNION_STRIDE}) % {SCALE}]",
        },
        "sources": source_shas,
        "floor_by_form": floor_by_form,
        "form_counts": form_counts,
        "length_buckets_written": length_counts,
        "written": written_texts,
        "unknown": [item["text"] for item in unknown],
        "unrelated": [item["text"] for item in unrelated],
        "union": union,
    }


def phase_build() -> dict[str, Any]:
    corpus = build_corpus()
    write_json(CORPUS, corpus)
    write_json(
        ROOT / "reports/taiji_scale_20261108_corpus_preview.json",
        {
            "form_counts": corpus["form_counts"],
            "length_buckets_written": corpus["length_buckets_written"],
            "written": len(corpus["written"]),
            "unknown": len(corpus["unknown"]),
            "unrelated": len(corpus["unrelated"]),
        },
    )
    print(json.dumps(corpus["form_counts"], ensure_ascii=False))
    print(json.dumps(corpus["length_buckets_written"], ensure_ascii=False))
    return corpus


def phase_freeze() -> dict[str, Any]:
    if not CORPUS.exists():
        write_json(CORPUS, build_corpus())
    corpus_bytes = CORPUS.read_bytes()
    audit = chunk_address_audit()
    manifest = {
        "schema": "taiji-scale-v1",
        "date": "2026-11-08",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runner_sha256": RUNNER_SHA_START,
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "corpus_path": str(CORPUS.relative_to(ROOT)),
        "question": (
            "on a mechanically sampled natural corpus at least 10x the fixture: do the frozen "
            "address/output definitions hold, how does admission vary with length, and are "
            "fallback sentences relevant to the query?"
        ),
        "scale": {
            "written": SCALE,
            "unknown": SCALE,
            "unrelated": SCALE,
            "union": SCALE,
            "fixture_ratio": round(SCALE / 58, 3),
            "form_floor": FORM_FLOOR,
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
            "sample_rule": (
                "unanchored rows in report order, every ceil(total/20)-th row, first 20"
            ),
        },
        "verdict": {
            "rules": [
                "closed not neutral -> hard_failure_closed_not_neutral (stop)",
                "any L1 line fails -> scale_recheck_unusable",
                (
                    f"admission rate for entries < 64 bytes < {SHORT_ENTRY_LINE} -> "
                    "scale_recheck_short_entries_gap"
                ),
                f"fallback relevance median < {FB_LINE} -> fallback_irrelevant",
                "otherwise -> scale_recheck_holds",
            ]
        },
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "address_module": {
            "sha256": hashlib.sha256(ADDRESS_MODULE.read_bytes()).hexdigest(),
            "constant_inventory": audit["constants"],
            "read_path_literals": audit["read_path_literals"],
            "protection": audit["protection"],
            "drift_vs_previous_round": drift_vs_previous(audit),
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
    return manifest


def phase_closed() -> dict[str, Any]:
    closed_battery.CLOSED_REPORT = CLOSED_REPORT
    closed_battery.PROGRESS = PROGRESS
    closed_battery.CPU_CAP = CPU_CAP
    payload = closed_battery.phase_closed()
    return payload


def _bucket_of(length: int) -> str:
    for name, low, high in LEN_BUCKETS:
        if length >= low and (high is None or length < high):
            return name
    return LEN_BUCKETS[-1][0]


def _span_legal(source: bytes, start: int, end: int) -> bool:
    """The span is the whole record or exactly one span returned by ``sentences()``."""
    if not (0 <= start < end <= len(source)):
        return False
    if start == 0 and end == len(source):
        return True
    return (start, end) in sentences(source)


def evaluate(
    corpus: dict[str, Any], table: AddressTable
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
                output_class = "usable_fallback"
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


def summarise(
    corpus: dict[str, Any], rows: list[dict[str, Any]], layer: dict[str, Any]
) -> dict[str, Any]:
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


def entry_curve(corpus: dict[str, Any], table: AddressTable) -> dict[str, Any]:
    admitted = {
        hashlib.sha256(address["raw"]).hexdigest(): index
        for index, address in enumerate(table.addresses)
    }
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
    long_unaddressable = []
    for text in unaddressable:
        raw = text.encode("utf-8")
        if len(raw) < 64:
            continue
        long_unaddressable.append(
            {
                "length": len(raw),
                "identity_blocks": len(identity(raw)),
                "all_chunks": len(chunks(raw)),
                "reason": "no identity chunk (no cut at offset >= 6 with 16+ bytes before the "
                "final chunk)",
                "text": text,
            }
        )
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
        "band_width": PROTECT,
        "min_chunk": MIN_CHUNK,
    }


def fallback_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fallback = [row for row in rows if row["fallback"]]
    ratios = sorted(float(row["overlap_ratio"]) for row in fallback)
    if ratios:
        import statistics

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
            "output": row["output_text"][:240],
            "score": None,
        }
        for row in sample
    ]
    return {
        "n": len(fallback),
        "overlap_quartiles": [round(float(value), 6) for value in quartiles],
        "overlap_median": round(float(median), 6),
        "overlap_mean": mean,
        "review": review,
    }


def drift_vs_previous(audit: dict[str, Any]) -> dict[str, Any]:
    """Compare the live address module with the previous round's frozen copy."""
    if not PREVIOUS_MANIFEST.exists():
        return {"previous": None}
    old = json.loads(PREVIOUS_MANIFEST.read_text(encoding="utf-8")).get("address_module", {})
    old_const = old.get("constant_inventory", {})
    old_lit = old.get("read_path_literals", {})
    new_const = audit["constants"]
    new_lit = audit["read_path_literals"]
    drift = {
        "previous": str(PREVIOUS_MANIFEST.relative_to(ROOT)),
        "address_sha256_previous": old.get("sha256"),
        "address_sha256_current": hashlib.sha256(ADDRESS_MODULE.read_bytes()).hexdigest(),
        "new_constants": sorted(set(new_const) - set(old_const)),
        "removed_constants": sorted(set(old_const) - set(new_const)),
        "changed_constants": {
            name: [old_const[name], new_const[name]]
            for name in sorted(set(old_const) & set(new_const))
            if old_const[name] != new_const[name]
        },
        "new_literals": sorted(set(new_lit) - set(old_lit)),
        "removed_literals": sorted(set(old_lit) - set(new_lit)),
        "changed_literals": {
            name: [old_lit[name], new_lit[name]]
            for name in sorted(set(old_lit) & set(new_lit))
            if old_lit[name] != new_lit[name]
        },
    }
    drift["clean"] = bool(
        drift["address_sha256_previous"] == drift["address_sha256_current"]
        and not drift["new_constants"]
        and not drift["removed_constants"]
        and not drift["changed_constants"]
        and not drift["new_literals"]
        and not drift["removed_literals"]
        and not drift["changed_literals"]
    )
    return drift


def verdict_of(
    closed: dict[str, Any],
    layer: dict[str, Any],
    curve: dict[str, Any],
    fb: dict[str, Any],
    drift: dict[str, Any] | None = None,
) -> str:
    if not str(closed.get("verdict", "")).endswith("_closed_neutral"):
        return "hard_failure_closed_not_neutral"
    if drift is not None and not drift.get("clean", False):
        return "scale_recheck_unusable"
    if (
        layer["fabricated_bytes"]
        or layer["illegal_spans"]
        or layer["utf8_invalid"]
        or layer["boundary_illegal"]
    ):
        return "scale_recheck_unusable"
    if curve["short_entries"]["rate"] is None or (
        curve["short_entries"]["rate"] < SHORT_ENTRY_LINE
    ):
        return "scale_recheck_short_entries_gap"
    scores = [item["score"] for item in fb["review"]]
    if any(score is None for score in scores):
        return "pending_fallback_review"
    values = sorted(SCORE_VALUES[score] for score in scores)
    import statistics

    if statistics.median(values) < FB_LINE:
        return "fallback_irrelevant"
    return "scale_recheck_holds"


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
    }


def phase_open(manifest: dict[str, Any]) -> dict[str, Any]:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    corpus_sha = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    if corpus_sha != manifest["corpus_sha256"]:
        raise SystemExit("corpus changed after freeze")
    append_progress({"phase": "open_start", "corpus_sha256": corpus_sha})

    table = AddressTable()
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
    by_group = summarise(corpus, rows, layer)
    curve = entry_curve(corpus, table)
    fallback = fallback_stats(rows)
    closed = json.loads(CLOSED_REPORT.read_text(encoding="utf-8"))
    drift = drift_vs_previous(chunk_address_audit())
    verdict = verdict_of(closed, layer, curve, fallback, drift)
    failures = [
        {
            "group": row["group"],
            "position": row["position"],
            "query": row["query_text"],
            "query_len": row["query_len"],
            "expected": row["expected"],
            "actual_record": row["record"],
            "anchors": row["anchors"],
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
        "schema": "taiji-scale-v1",
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
                for name in ("none", "record_overlap", "address_order")
            },
        },
        "groups": by_group,
        "failures": failures,
        "code_drift_vs_previous_round": drift,
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
            "the corpus is sampled mechanically from this repository's own text; it is a "
            "natural but self-referential corpus (technical Chinese prose, tables, headings, "
            "code comments), not a web crawl.",
            "nothing in taiji/ was edited this round; the address module sha is recorded and "
            "the frozen constant/literal audit is compared against the manifest.",
            "a written entry whose record was never admitted has no expected address, so any "
            "output for it is usable_but_wrong by definition; the admission obstacles are "
            "reported one by one, not hidden in the rate.",
            "unknown and unrelated queries usually have no answering record by construction; "
            "their outputs are classed usable_fallback, never usable_and_correct.",
            "the union group joins two admitted records with a newline; pointing at either "
            "source record counts as correct, pointing elsewhere does not.",
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
    curve = report["entry_curve"]
    fb = report["fallback_relevance"]
    closed = report["closed"]
    drift = report.get("code_drift_vs_previous_round") or drift_vs_previous(chunk_address_audit())
    verdict = verdict_of(closed, report["layers"]["L1"], curve, fb, drift)
    report["verdict"] = verdict
    values = sorted(SCORE_VALUES[score] for score in scores)
    import statistics

    report["fallback_relevance"]["score_median"] = round(float(statistics.median(values)), 6)
    report["fallback_relevance"]["score_counts"] = {
        name: scores.count(name) for name in SCORE_VALUES
    }
    report["cost"]["score_pass_cpu_seconds"] = round(time.process_time() - START_CPU, 3)
    report["cost"]["score_pass_wall_seconds"] = round(time.time() - START_WALL, 3)
    write_json(REPORT, report)
    print(json.dumps({"verdict": verdict, "median": report["fallback_relevance"]["score_median"]}))
    return report


def main(argv: list[str]) -> int:
    assert_no_black_processes()
    if "--build" in argv:
        acquire_lock()
        watchdog()
        phase_build()
        return 0
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
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["runner_sha256"] != RUNNER_SHA_START:
        raise SystemExit("runner changed after freeze: re-freeze first")
    if not CLOSED_REPORT.exists():
        acquire_lock()
        watchdog()
        phase_closed()
    acquire_lock()
    watchdog()
    phase_open(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
