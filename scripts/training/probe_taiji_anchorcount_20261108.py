"""Anchor-count gate: decide the read exit from the anchor count alone.

The A1 gate showed the address layer was not limited by A1 (no record needed an
artificial unique word) but by a second structural rule: any query feature with
``DF = 0`` returned ``no_match`` before the anchors were even counted.  This runner
measures the consequence of removing that rule -- the read now exits only on
``|Anchors_T(q)|``.

Scope is fixed in the manifest before the first reading and may not move after it:

* same extraction rule, same source files, the same three denominators
  (``should_anchor`` / ``should_unknown`` / ``should_ambiguous``) and the same
  usable line (0.60);
* the corpus text itself is frozen into a snapshot file, so a later edit to the
  plans cannot silently move the acceptance set;
* one extra reading class (``unrelated``) is recorded and **never** decides the
  verdict: new sentences from a file that is neither written nor a paraphrase.

Two arms, in this order:

``closed``  the flag at its default reproduces the pre-edit forward bit for bit
            against the stored baseline arrays (same 176-case battery).
``open``    the real switch allocates the ``AddressTable``; the frozen corpus is
            written with ``write_batch_checked``; every frozen query is read.

Discipline: 0 training / 0 optimizer steps / 0 Adam / 0 confirm; ``process_guard``
first; single-instance lock; 60 s watchdog; CPU cap 200 s; readings flushed as taken;
write-then-rename; dual runner sha.  No ``python -m black``: the in-process gate is
the only formatter check.

SUPERSEDED.  The variant was rejected (``anchor_count_unsafe``) and the read path was
rolled back to the ``DF = 0`` guard, so re-running this file now measures A2 and does
not reproduce ``reports/taiji_anchorcount_20261108.json``.  It is kept as the evidence
of the rejected variant, behind the usual environment gate.
"""

from __future__ import annotations

import atexit
import errno
import hashlib
import importlib
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

import numpy as np  # noqa: E402

from taiji.living_core import LivingCoreConfig, LivingCoreReference  # noqa: E402
from taiji.process_guard import assert_no_black_processes  # noqa: E402
from taiji.sc_sfk import AddressTable, features  # noqa: E402

MANIFEST = ROOT / "plans/manifests/taiji_anchorcount_20261108.json"
CORPUS = ROOT / "plans/manifests/taiji_anchorcount_corpus_20261108.json"
REPORT = ROOT / "reports/taiji_anchorcount_20261108.json"
CLOSED_REPORT = ROOT / "reports/taiji_anchorcount_20261108_closed.json"
PROGRESS = ROOT / "reports/taiji_anchorcount_20261108_progress.jsonl"
LOCK = ROOT / "reports/taiji_anchorcount_20261108.lock"
WATCHDOG_MARKER = ROOT / "reports/taiji_anchorcount_20261108_watchdog.json"
PREVIOUS_MANIFEST = ROOT / "plans/manifests/taiji_a1_offfixture_20261108.json"

CPU_CAP = 200.0
WATCHDOG_SECONDS = 60.0
LOCK_STALE_SECONDS = 600.0
N_WRITE = 58
N_UNKNOWN = 20
N_PAIR = 20
PAIR_STRIDE = 30
N_UNRELATED = 20
MIN_CHARS = 120
MAX_CHARS = 500
BATCH = 10
USABLE_LINE = 0.60

WRITE_FILES = (
    "plans/MASTER-PLAN.md",
    "plans/reference/CAPABILITY-CARD_20261103.md",
)
UNKNOWN_FILES = ("plans/reference/INVARIANT-LEDGER_20261103.md",)
UNRELATED_FILES = ("README.zh-CN.md", "TAIJI_CODE_WIKI.md", "INTERFACE_REFERENCE.md")
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
SUPERSEDED = (
    "measured the anchor-count variant; the verdict was anchor_count_unsafe and the "
    "read path was rolled back to the DF = 0 guard, so a re-run measures A2"
)
START_CPU = time.process_time()
START_WALL = time.time()
LAST_READING = [time.time()]
WATCHDOG_FIRED = [False]

CRITERIA = {
    "change_under_test": (
        "the read exit is decided by the anchor count alone: |Anchors_T(q)| = 0 -> "
        "no_match, = 1 -> matched, >= 2 -> ambiguous_multiple_anchors.  A query feature "
        "with DF = 0 cannot be an anchor, so it lowers the count; it is reported as a "
        "reading and cannot short-circuit, weight or bias the exit.  The old early-return "
        "path is deleted, not kept as a second criterion."
    ),
    "C1_corpus_frozen_before_run": (
        "the corpus text is written to a snapshot file and its per-paragraph sha256 list "
        "is recorded in the manifest before the first reading.  The run reads the "
        "snapshot, so editing the plan files afterwards cannot change the acceptance set."
    ),
    "C2_same_implementation_same_switch": (
        "the table under test is the AddressTable allocated by LivingCoreConfig("
        "sc_sfk_enabled=True) through attach_store; writes use write_batch_checked and "
        "reads use table.read.  Zero new parameters, zero tensors."
    ),
    "C3_no_threshold_tuning": (
        "no numeric threshold on the read path, no leading margin, no retry with another "
        "address definition.  The usable line is fixed at 0.60 in the manifest."
    ),
    "C4_closed_state_neutral": (
        "with the flag at its default the forward pass must reproduce the stored baseline "
        "arrays bit for bit (max|delta| = 0), with parameters, state bytes and graph "
        "identity unchanged and no table allocated."
    ),
    "V_wrong_pointer": (
        "a should-anchor query answered 'matched' to an address other than its own record"
    ),
    "V_anchor_count_unsafe": (
        "wrong_pointer > 0: the no-wrong-answer property depended on rejecting unseen "
        "features; go back to the address definition, never tune a threshold."
    ),
    "V_anchor_count_holds": (
        "wrong_pointer = 0 AND usable_ratio >= 0.60 AND the held-out set is 20/20 "
        "no_match AND the two-record set is 20/20 ambiguous.  Cost to state: the 'reject "
        "on a word never seen' hard guard is gone; false answers are now bounded only by "
        "anchor uniqueness and the negative controls."
    ),
    "V_anchor_count_insufficient": (
        "wrong_pointer = 0 AND usable_ratio < 0.60: report the distribution; the line may "
        "not be lowered."
    ),
    "V_inconclusive": (
        "wrong_pointer = 0, usable >= 0.60, but a negative control did not hold: report "
        "the distribution only."
    ),
    "R_unrelated_reading": (
        "sentences from a file that is never written and is not in the paraphrase table.  "
        "A 'matched' answer is counted as a hard answer.  This class is a reading only and "
        "never enters the verdict."
    ),
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
            LOCK.parent / "taiji_anchorcount_20261108_cpu_abort.json",
            {"aborted": True, "cpu": round(used, 3), "at": time.time()},
        )
        raise SystemExit(f"CPU cap {CPU_CAP}s exceeded ({used:.1f}s)")
    return used


# ---------------------------------------------------------------------------
# Corpus (mechanical rule, frozen into a snapshot before the first reading).
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def paragraphs_of(text: str) -> list[str]:
    """Blank-line blocks; drop fences and table rows; strip markers; keep 120..500."""
    blocks = re.split(r"\n\s*\n", normalize(text))
    kept: list[str] = []
    for block in blocks:
        if "```" in block:
            continue
        lines = []
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("|"):
                continue
            stripped = re.sub(r"^#{1,6}\s*", "", stripped)
            stripped = re.sub(r"^>\s*", "", stripped)
            stripped = re.sub(r"^(?:[-*+]|\d+\.)\s+", "", stripped)
            lines.append(stripped)
        text = "".join(lines)
        if MIN_CHARS <= len(text) <= MAX_CHARS:
            kept.append(text)
    return kept


def take(files: tuple[str, ...], limit: int, exclude: set[str]) -> list[dict[str, str]]:
    kept: list[dict[str, str]] = []
    for name in files:
        for text in paragraphs_of((ROOT / name).read_text(encoding="utf-8")):
            if text in exclude:
                continue
            kept.append({"text": text, "source": name})
            if len(kept) >= limit:
                return kept
    return kept


def paraphrase(text: str) -> tuple[str, int]:
    changed = 0
    for old, new in PARAPHRASE_RULES:
        changed += text.count(old)
        text = text.replace(old, new)
    return text, changed


def collect() -> dict[str, Any]:
    written = take(WRITE_FILES, N_WRITE, set())
    seen = {item["text"] for item in written}
    unknown = take(UNKNOWN_FILES, N_UNKNOWN, seen)
    seen |= {item["text"] for item in unknown}
    unrelated = take(UNRELATED_FILES, N_UNRELATED, seen)
    return {"written": written, "unknown": unknown, "unrelated": unrelated}


# ---------------------------------------------------------------------------
# Freeze.
# ---------------------------------------------------------------------------
def phase_freeze() -> int:
    collection = collect()
    written = collection["written"]
    unknown = collection["unknown"]
    unrelated = collection["unrelated"]
    if len(written) < N_WRITE or len(unknown) < N_UNKNOWN or len(unrelated) < N_UNRELATED:
        raise SystemExit(
            f"corpus short: written={len(written)} unknown={len(unknown)} "
            f"unrelated={len(unrelated)}"
        )

    written_sha = [text_sha(item["text"]) for item in written]
    unknown_sha = [text_sha(item["text"]) for item in unknown]
    unrelated_sha = [text_sha(item["text"]) for item in unrelated]
    pairs = [
        [index, index + PAIR_STRIDE]
        for index in range(N_PAIR)
        if index + PAIR_STRIDE < len(written)
    ]
    unparaphrased = sum(1 for item in written if paraphrase(item["text"])[1] == 0)

    drift: dict[str, Any] = {
        "previous_manifest": None,
        "shared_written": None,
        "shared_unknown": None,
    }
    if PREVIOUS_MANIFEST.exists():
        old = json.loads(PREVIOUS_MANIFEST.read_text(encoding="utf-8"))
        drift = {
            "previous_manifest": str(PREVIOUS_MANIFEST.relative_to(ROOT)),
            "previous_freeze_revision": old.get("freeze_revision"),
            "shared_written": len(set(old["corpus"]["written_sha256"]) & set(written_sha)),
            "shared_unknown": len(set(old["corpus"]["unknown_sha256"]) & set(unknown_sha)),
            "note": (
                "the plan files are themselves the corpus, and they were edited by the "
                "previous round's own receipt; the extraction rule and the file list are "
                "identical, the overlap is reported instead of hidden"
            ),
        }

    corpus = {
        "schema": "taiji-anchorcount-corpus-v1",
        "extraction_rule": (
            "blank-line blocks; drop fenced code and table rows; strip heading / quote / "
            f"bullet markers; join lines; keep {MIN_CHARS}..{MAX_CHARS} chars; take the "
            f"first {N_WRITE} paragraphs in file order for the write set, the first "
            f"{N_UNKNOWN} in file order from a disjoint file for the unknown set, and the "
            f"first {N_UNRELATED} in file order from files that are neither written nor in "
            "the synonym table for the unrelated reading set."
        ),
        "write_files": list(WRITE_FILES),
        "unknown_files": list(UNKNOWN_FILES),
        "unrelated_files": list(UNRELATED_FILES),
        "written": [item["text"] for item in written],
        "unknown": [item["text"] for item in unknown],
        "unrelated": [item["text"] for item in unrelated],
        "written_sha256": written_sha,
        "unknown_sha256": unknown_sha,
        "unrelated_sha256": unrelated_sha,
    }
    write_json(CORPUS, corpus)

    manifest = {
        "schema": "taiji-anchorcount-v1",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runner_sha256": RUNNER_SHA_START,
        "question": (
            "does the read exit survive when it is decided by the anchor count alone, on "
            "natural prose, with the same corpus, denominators and usable line?"
        ),
        "corpus": {
            "snapshot": str(CORPUS.relative_to(ROOT)),
            "snapshot_sha256": file_sha(CORPUS),
            "write_count": len(written),
            "unknown_count": len(unknown),
            "unrelated_count": len(unrelated),
            "written_sha256": written_sha,
            "unknown_sha256": unknown_sha,
            "unrelated_sha256": unrelated_sha,
            "written_sha256_aggregate": text_sha("".join(written_sha)),
            "unknown_sha256_aggregate": text_sha("".join(unknown_sha)),
            "drift_from_previous": drift,
        },
        "paraphrase": {
            "rule": "fixed global synonym table applied uniformly, in this order",
            "table": [list(pair) for pair in PARAPHRASE_RULES],
            "records_with_zero_replacements": unparaphrased,
        },
        "queries": {
            "should_anchor": (
                f"{len(written)} self-reads + one or two paraphrase variants per record: "
                "V1 = the fixed synonym table (kept only when it changes the text), "
                "V2 = '总的来说，' + V1 + '。' (always applied)."
            ),
            "should_unknown": (
                f"{len(unknown)} held-out paragraphs from a file that is never written; "
                "ground truth is 'no_match'"
            ),
            "should_ambiguous": (
                f"{len(pairs)} two-record queries built as the union of the two feature "
                f"sets (paragraph k and k+{PAIR_STRIDE}), counted only when both records "
                "are addressable"
            ),
            "unrelated": (
                f"{len(unrelated)} paragraphs from a file that is neither written nor in "
                "the synonym table.  Recorded as a reading: how many are hard-answered "
                "('matched') because the anchor happens to be unique.  Never a criterion."
            ),
            "pairs": pairs,
        },
        "denominators": {
            "should_anchor_upper_bound": len(written)
            + len(written)
            + (len(written) - unparaphrased),
            "should_unknown": len(unknown),
            "should_ambiguous_upper_bound": len(pairs),
            "unrelated_reading": len(unrelated),
            "uniquely_anchorable_definition": (
                "queries whose own record has a table-unique word present in the query: "
                "A_i ∩ Q ∩ U(T) ≠ ∅, a denominator, never a criterion"
            ),
        },
        "verdict": {
            "usable_ratio_definition": (
                "should-anchor queries answered 'matched' to their own address, divided by "
                "all should-anchor queries (self + paraphrase).  Abstentions and ambiguous "
                "answers count as unusable; they are reported, not hidden."
            ),
            "usable_line": USABLE_LINE,
            "rules": [
                "wrong_pointer > 0 -> anchor_count_unsafe",
                "wrong_pointer = 0 and usable_ratio >= 0.60 and unknown 20/20 no_match "
                "and ambiguous 20/20 ambiguous_multiple_anchors -> anchor_count_holds",
                "wrong_pointer = 0 and usable_ratio < 0.60 -> anchor_count_insufficient",
                "otherwise -> inconclusive",
            ],
        },
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
        for key in ("written_sha256", "unknown_sha256", "unrelated_sha256"):
            if old["corpus"].get(key) != manifest["corpus"][key]:
                raise SystemExit(f"refusing to refreeze: corpus {key} changed")
        if old.get("verdict") != manifest["verdict"]:
            raise SystemExit("refusing to refreeze: verdict rules changed")
        manifest["freeze_revision"] = int(old.get("freeze_revision", 1)) + 1
        manifest["freeze_note"] = "corpus and verdict rules unchanged; harness revision only"
    else:
        manifest["freeze_revision"] = 1
        manifest["freeze_note"] = "initial freeze"
    write_json(MANIFEST, manifest)
    print(
        json.dumps(
            {
                "manifest": str(MANIFEST.relative_to(ROOT)),
                "written": len(written),
                "unknown": len(unknown),
                "unrelated": len(unrelated),
                "drift": drift,
                "usable_line": USABLE_LINE,
            },
            ensure_ascii=False,
        )
    )
    return 0


# ---------------------------------------------------------------------------
# Closed arm: the same 176-case battery against the stored baseline bytes.
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
    identity = wire.identity_of(subject)
    signature = wire.store_signature(state)
    state_bytes = wire.living_state_bytes(state)
    living_core_now = wire.file_sha(wire.LIVING_CORE)

    checks = {
        "default_off": bool(default_off is False),
        "table_not_allocated": bool(state.sc_sfk is None if hasattr(state, "sc_sfk") else True),
        "readings_total": len(readings),
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
            identity["candidate_sha_v2"] == baseline["identity"]["candidate_sha_v2"]
        ),
        "store_signature_equal": bool(signature == baseline["store"]),
    }
    total = len(readings)
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
        "schema": "taiji-anchorcount-closed-v1",
        "date": "2026-11-08",
        "question": "is the edited read path inert while the flag is at its default?",
        "verdict": (
            "anchor_count_closed_neutral" if neutral else "anchor_count_closed_not_neutral"
        ),
        "checks": checks,
        "baseline": {
            "report": wire.BASELINE_JSON.name,
            "living_core_sha256": baseline["living_core_sha256"],
            "candidate_sha_v2": baseline["identity"]["candidate_sha_v2"],
        },
        "current": {
            "living_core_sha256": living_core_now,
            "candidate_sha_v2": identity["candidate_sha_v2"],
            "state_bytes": state_bytes,
        },
        "disclosures": [
            "the edit is confined to AddressTable.read, which cannot be reached while the "
            "flag is off (table_not_allocated); the battery is replayed anyway because the "
            "closed-state claim is re-proved, not inferred.",
            "the comparison is against baseline arrays captured before taiji/ was edited, "
            "not against a second run of the edited code.",
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
# Open arm.
# ---------------------------------------------------------------------------
def df_histogram(table: AddressTable, query: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for address in table.addresses:
        for feature in address["features"]:
            counts[feature] = counts.get(feature, 0) + int(address["count"])
    buckets = {"df=0": 0, "df=1": 0, "df=2-3": 0, "df>=4": 0}
    for feature in query:
        value = counts.get(feature, 0)
        if value == 0:
            buckets["df=0"] += 1
        elif value == 1:
            buckets["df=1"] += 1
        elif value <= 3:
            buckets["df=2-3"] += 1
        else:
            buckets["df>=4"] += 1
    return buckets


def evaluate(table: AddressTable, items: list[dict[str, Any]]) -> dict[str, Any]:
    per_query = []
    for item in items:
        query = item["feats"]
        began = time.perf_counter()
        result = table.read(query)
        elapsed = (time.perf_counter() - began) * 1000.0
        record = {
            "kind": item["kind"],
            "variant": item.get("variant"),
            "expected": item["expected"],
            "decision": result["decision"],
            "address": result["address"],
            "anchors": len(result["anchors"]),
            "unique_in_query": result["unique_features_in_query"],
            "unknown_features": result["unknown_features"],
            "top1": result["top1"],
            "top2": result["top2"],
            "margin": result["margin"],
            "read_ms": round(elapsed, 4),
        }
        if result["decision"] not in ("matched",) or result["address"] != item["expected"]:
            record["sample"] = item["label"]
            record["df_histogram"] = df_histogram(table, query)
        per_query.append(record)
    return {"per_query": per_query}


def summarise(per_query: list[dict[str, Any]], anchorable: set[int]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in per_query:
        groups.setdefault(record["kind"], []).append(record)

    def rate(rows: list[dict[str, Any]], key: str) -> int:
        return sum(1 for row in rows if row["decision"] == key)

    self_rows = groups.get("self", [])
    para_rows = groups.get("paraphrase", [])
    unknown_rows = groups.get("unknown", [])
    amb_rows = groups.get("ambiguous", [])
    unrelated_rows = groups.get("unrelated", [])
    anchor_rows = self_rows + para_rows
    correct = sum(
        1
        for row in anchor_rows
        if row["decision"] == "matched" and row["address"] == row["expected"]
    )
    wrong = sum(
        1
        for row in anchor_rows
        if row["decision"] == "matched" and row["address"] != row["expected"]
    )
    failures = [
        row
        for row in anchor_rows
        if not (row["decision"] == "matched" and row["address"] == row["expected"])
    ]
    variants: dict[str, dict[str, int]] = {}
    for record in per_query:
        key = f"{record['kind']}:{record['variant']}"
        bucket = variants.setdefault(
            key, {"n": 0, "matched": 0, "no_match": 0, "ambiguous": 0, "wrong": 0}
        )
        bucket["n"] += 1
        if record["decision"] == "matched":
            if record["expected"] in (None, "ambiguous") or record["address"] != record["expected"]:
                bucket["wrong"] += 1
            else:
                bucket["matched"] += 1
        elif record["decision"] == "no_match":
            bucket["no_match"] += 1
        else:
            bucket["ambiguous"] += 1
    return {
        "counts": {
            "self": len(self_rows),
            "paraphrase": len(para_rows),
            "unknown": len(unknown_rows),
            "ambiguous": len(amb_rows),
            "unrelated": len(unrelated_rows),
        },
        "should_anchor": {
            "n": len(anchor_rows),
            "matched_correct": correct,
            "wrong_pointer": wrong,
            "no_match": rate(anchor_rows, "no_match"),
            "ambiguous": rate(anchor_rows, "ambiguous_multiple_anchors"),
            "usable_ratio": (round(correct / len(anchor_rows), 6) if anchor_rows else 0.0),
            "uniquely_anchorable": len(anchorable),
            "uniquely_anchorable_ratio": (
                round(len(anchorable) / len(anchor_rows), 6) if anchor_rows else 0.0
            ),
        },
        "by_kind": {
            "self": {
                "n": len(self_rows),
                "matched_correct": sum(
                    1
                    for row in self_rows
                    if row["decision"] == "matched" and row["address"] == row["expected"]
                ),
                "wrong_pointer": sum(
                    1
                    for row in self_rows
                    if row["decision"] == "matched" and row["address"] != row["expected"]
                ),
                "no_match": rate(self_rows, "no_match"),
                "ambiguous": rate(self_rows, "ambiguous_multiple_anchors"),
            },
            "paraphrase": {
                "n": len(para_rows),
                "matched_correct": sum(
                    1
                    for row in para_rows
                    if row["decision"] == "matched" and row["address"] == row["expected"]
                ),
                "wrong_pointer": sum(
                    1
                    for row in para_rows
                    if row["decision"] == "matched" and row["address"] != row["expected"]
                ),
                "no_match": rate(para_rows, "no_match"),
                "ambiguous": rate(para_rows, "ambiguous_multiple_anchors"),
            },
        },
        "by_variant": variants,
        "should_unknown": {
            "n": len(unknown_rows),
            "no_match": rate(unknown_rows, "no_match"),
            "matched_false_positive": rate(unknown_rows, "matched"),
            "ambiguous": rate(unknown_rows, "ambiguous_multiple_anchors"),
        },
        "should_ambiguous": {
            "n": len(amb_rows),
            "ambiguous": rate(amb_rows, "ambiguous_multiple_anchors"),
            "no_match": rate(amb_rows, "no_match"),
            "matched": rate(amb_rows, "matched"),
        },
        "unrelated_reading": {
            "n": len(unrelated_rows),
            "matched_hard_answer": rate(unrelated_rows, "matched"),
            "no_match": rate(unrelated_rows, "no_match"),
            "ambiguous": rate(unrelated_rows, "ambiguous_multiple_anchors"),
            "hard_answer_samples": [
                {
                    "sample": row.get("sample"),
                    "address": row["address"],
                    "anchors": row["anchors"],
                    "unknown_features": row["unknown_features"],
                    "df_histogram": row.get("df_histogram"),
                }
                for row in unrelated_rows
                if row["decision"] == "matched"
            ][:5],
        },
        "failures": failures,
    }


def verdict_of(summary: dict[str, Any]) -> str:
    anchor = summary["should_anchor"]
    unknown = summary["should_unknown"]
    ambiguous = summary["should_ambiguous"]
    if anchor["wrong_pointer"] > 0:
        return "anchor_count_unsafe"
    if anchor["usable_ratio"] < USABLE_LINE:
        return "anchor_count_insufficient"
    if (
        unknown["n"] == unknown["no_match"]
        and unknown["matched_false_positive"] == 0
        and ambiguous["ambiguous"] == ambiguous["n"]
    ):
        return "anchor_count_holds"
    return "inconclusive"


def phase_open(closed: dict[str, Any]) -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    if file_sha(CORPUS) != manifest["corpus"]["snapshot_sha256"]:
        raise SystemExit("corpus snapshot moved after freeze")
    for key in ("written", "unknown", "unrelated"):
        if [text_sha(text) for text in corpus[key]] != manifest["corpus"][f"{key}_sha256"]:
            raise SystemExit(f"corpus {key} differs from the frozen manifest")

    written = [{"text": text} for text in corpus["written"]]
    unknown = [{"text": text} for text in corpus["unknown"]]
    unrelated = [{"text": text} for text in corpus["unrelated"]]

    config = LivingCoreConfig(sc_sfk_enabled=True)
    subject = LivingCoreReference(config)
    state = subject.attach_store(subject.initial_state("anchorcount"))
    table = state.sc_sfk
    if not isinstance(table, AddressTable):
        raise SystemExit("switch did not allocate an AddressTable")
    append_progress({"phase": "store_attached", "table": True, "enabled": True})

    write_costs = []
    actions: dict[str, int] = {}
    value_payload = state.fast.detach().to("cpu").contiguous().numpy().tobytes()
    for start in range(0, len(written), BATCH):
        chunk = written[start : start + BATCH]
        began = time.perf_counter()
        result = table.write_batch_checked(
            [(features(item["text"].encode("utf-8")), value_payload) for item in chunk]
        )
        elapsed = (time.perf_counter() - began) * 1000.0
        checks = result.get("global_check") or {}
        write_costs.append(
            {
                "batch": start // BATCH,
                "items": len(chunk),
                "action": result["action"],
                "global_ms": round(elapsed, 4),
                "ok": bool(checks.get("ok")),
            }
        )
        actions[result["action"]] = actions.get(result["action"], 0) + 1
        append_progress(
            {
                "phase": "write",
                "batch": start // BATCH,
                "action": result["action"],
                "slots": len(table.addresses),
                "capacity": table.capacity,
                "global_ms": round(elapsed, 4),
            }
        )
        if result["action"] != "accepted":
            raise SystemExit(f"frozen corpus batch rejected: {result['action']}")

    unique = table.unique_features()

    def anchors_own(text: str, target: int) -> bool:
        query = features(text.encode("utf-8"))
        owned = set(table.addresses[target]["features"])
        return any(feature in unique and feature in owned for feature in query)

    addressable = {
        index
        for index, address in enumerate(table.addresses)
        if any(feature in unique for feature in address["features"])
    }
    expected_index = {}
    duplicates = 0
    for index, item in enumerate(written):
        feats = features(item["text"].encode("utf-8"))
        found = table.find(feats)
        if found is None:
            raise SystemExit(f"written record {index} is not in the table")
        if found in expected_index.values():
            duplicates += 1
        expected_index[index] = found

    queries: list[dict[str, Any]] = []
    for index, item in enumerate(written):
        queries.append(
            {
                "kind": "self",
                "variant": "exact",
                "feats": features(item["text"].encode("utf-8")),
                "expected": expected_index[index],
                "anchorable": anchors_own(item["text"], expected_index[index]),
                "label": item["text"][:80],
            }
        )
    paraphrase_zero = 0
    for index, item in enumerate(written):
        text, changed = paraphrase(item["text"])
        if changed == 0:
            paraphrase_zero += 1
        else:
            queries.append(
                {
                    "kind": "paraphrase",
                    "variant": "synonym",
                    "feats": features(text.encode("utf-8")),
                    "expected": expected_index[index],
                    "anchorable": anchors_own(text, expected_index[index]),
                    "label": text[:80],
                }
            )
        lead = "总的来说，" + text + "。"
        queries.append(
            {
                "kind": "paraphrase",
                "variant": "lead_in",
                "feats": features(lead.encode("utf-8")),
                "expected": expected_index[index],
                "anchorable": anchors_own(lead, expected_index[index]),
                "label": lead[:80],
            }
        )
    for item in unknown:
        queries.append(
            {
                "kind": "unknown",
                "variant": "held_out",
                "feats": features(item["text"].encode("utf-8")),
                "expected": None,
                "label": item["text"][:80],
            }
        )
    for item in unrelated:
        queries.append(
            {
                "kind": "unrelated",
                "variant": "readme",
                "feats": features(item["text"].encode("utf-8")),
                "expected": None,
                "label": item["text"][:80],
            }
        )
    pairs = []
    for pair in manifest["queries"]["pairs"]:
        left, right = pair
        if left >= len(written) or right >= len(written):
            continue
        if expected_index[left] not in addressable or expected_index[right] not in addressable:
            continue
        pairs.append(pair)
        union = set(table.addresses[expected_index[left]]["features"]) | set(
            table.addresses[expected_index[right]]["features"]
        )
        queries.append(
            {
                "kind": "ambiguous",
                "variant": "union",
                "feats": tuple(sorted(union)),
                "expected": "ambiguous",
                "label": written[left]["text"][:40] + " ‖ " + written[right]["text"][:40],
            }
        )
    append_progress(
        {
            "phase": "denominators",
            "addresses": len(table.addresses),
            "addressable": len(addressable),
            "unique_features": len(unique),
            "duplicate_records": duplicates,
            "paraphrase_zero_replacements": paraphrase_zero,
            "ambiguous_pairs_eligible": len(pairs),
            "queries": len(queries),
        }
    )

    anchorable = {index for index, item in enumerate(queries) if item.get("anchorable")}
    readings = evaluate(table, queries)
    summary = summarise(readings["per_query"], anchorable)

    switch_rows = []
    for index in range(min(8, len(written))):
        direct = table.read(features(written[index]["text"].encode("utf-8")))
        _, info = subject.addressed_read(state, written[index]["text"].encode("utf-8"))
        switch_rows.append(
            {
                "record": index,
                "direct_decision": direct["decision"],
                "direct_address": direct["address"],
                "switch_decision": info["decision"],
                "switch_address": info["address"],
                "agree": direct["decision"] == info["decision"]
                and direct["address"] == info["address"],
            }
        )
        touch()

    verdict = verdict_of(summary)
    cpu = cpu_check()
    payload = {
        "schema": "taiji-anchorcount-v1",
        "date": "2026-11-08",
        "question": manifest["question"],
        "verdict": verdict,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": file_sha(MANIFEST),
        "corpus_snapshot_sha256": file_sha(CORPUS),
        "runner_sha_start": RUNNER_SHA_START,
        "runner_sha_end": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        "closed": {
            "report": str(CLOSED_REPORT.relative_to(ROOT)),
            "verdict": closed["verdict"],
            "fast_max_abs_delta": closed["checks"]["fast_max_abs_delta"],
            "probability_max_abs_delta": closed["checks"]["probability_max_abs_delta"],
            "readings": closed["checks"]["readings_total"],
            "table_not_allocated": closed["checks"]["table_not_allocated"],
        },
        "switch": {
            "sc_sfk_enabled": True,
            "table_allocated_by_switch": True,
            "new_parameters": 0,
            "tensors_in_table": False,
            "join": "8 self-reads compared between table.read and addressed_read",
            "join_rows": switch_rows,
            "join_agreement": sum(1 for row in switch_rows if row["agree"]),
        },
        "table": {
            "addresses": len(table.addresses),
            "capacity": table.capacity,
            "peak_capacity": table.peak_capacity,
            "generation": table.generation,
            "appended": table.appended,
            "merged": table.merged,
            "replaced": table.replaced,
            "rejected_batches": table.rejected_batches,
            "rollbacks": table.rollbacks,
            "evictions": table.evictions,
            "silent_overwrites": table.silent_overwrites,
            "global_checks": table.global_checks,
            "global_failures": table.global_failures,
            "addressable": len(addressable),
            "unique_features": len(unique),
            "duplicate_records": duplicates,
            "nbytes": table.nbytes(),
        },
        "denominators": {
            "should_anchor": summary["should_anchor"]["n"],
            "should_unknown": summary["counts"]["unknown"],
            "should_ambiguous": summary["counts"]["ambiguous"],
            "unrelated_reading": summary["counts"]["unrelated"],
            "uniquely_anchorable": summary["should_anchor"]["uniquely_anchorable"],
            "paraphrase_zero_replacements": paraphrase_zero,
        },
        "readings": summary,
        "write_costs": write_costs,
        "cost": {
            "cpu_seconds": round(cpu, 3),
            "wall_seconds": round(time.time() - START_WALL, 3),
            "mean_global_ms": round(
                sum(row["global_ms"] for row in write_costs) / max(len(write_costs), 1),
                4,
            ),
            "max_global_ms": round(max((row["global_ms"] for row in write_costs), default=0.0), 4),
            "mean_read_ms": round(
                sum(row["read_ms"] for row in readings["per_query"])
                / max(len(readings["per_query"]), 1),
                4,
            ),
            "closed_wall_seconds": closed["cost"]["wall_seconds"],
        },
        "criteria": CRITERIA,
        "disclosures": [
            "the exit is the anchor count and nothing else; the DF=0 early return is "
            "deleted, so no query is rejected for carrying an unseen word",
            "unseen features are still counted and reported (unknown_features), including "
            "their document-frequency histogram, but they never enter the decision",
            "the unrelated set is a reading: a hard answer there is recorded, not scored",
            "the corpus snapshot, not the plan files, is what the run reads, so a later "
            "plan edit cannot move the acceptance set",
            "similarity columns are diagnostics; no exit reads them",
        ],
    }
    write_json(REPORT, payload)
    append_progress({"phase": "done", "verdict": verdict})
    print(json.dumps({"verdict": verdict, "readings": summary}, ensure_ascii=False))
    return 0


def main(argv: list[str]) -> int:
    if os.environ.get("TAIJI_ALLOW_SUPERSEDED_RUNNER") != "1":
        raise SystemExit(f"SUPERSEDED: {SUPERSEDED}")
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
