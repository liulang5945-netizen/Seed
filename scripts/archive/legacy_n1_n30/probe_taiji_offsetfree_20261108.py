"""Offset-free features gate: two arms, one run, one frozen corpus.

What changed (one place only)
-----------------------------
``taiji.sc_sfk.features`` now takes a 2-byte unit at **every** byte offset instead
of every second offset.  The feature set is therefore invariant to a shift of the
whole text: a prefix cannot move the features the record already has.  Nothing else
in the address layer moved -- same address definition, same anchors, same capacity,
merge, reject and rollback.

Why two arms in one run
-----------------------
The 2026-11-08 anchor-count gate changed *two* things at once (extraction was still
offset-sensitive, and the guard was deleted), so "offset" and "guard" were mixed
together.  This run holds extraction fixed at the offset-free version and reports:

* **arm A** = offset-free features + the ``DF = 0`` guard (the shipped product path);
* **arm B** = offset-free features + anchor counting only (a *shadow* evaluation in
  this harness, never wired into the product; ``AddressTable.read`` keeps the guard
  and there is no second judge in ``taiji/``).

Arm B's exits are computed from public table state only (``table.addresses``,
``document_frequency``, ``unique_features``): 0 anchors -> ``no_match``, 1 ->
``matched``, >= 2 -> ``ambiguous_multiple_anchors``.  Unseen features are counted
and reported, never consulted.

Corpus, denominators and line (frozen before the first reading, unchanged)
--------------------------------------------------------------------------
The corpus is the **byte-identical** corpus of the anchor-count gate, carried over
from ``plans/manifests/taiji_anchorcount_corpus_20261108.json`` and re-hashed into
this run's own snapshot.  Denominators: ``should_anchor`` = 120 (58 self + 62
paraphrase), ``should_unknown`` = 20, ``should_ambiguous`` = 20, ``unrelated`` = 20
(reading only).  Usable line stays 0.60.  No leading margin, no threshold.

Verdict (written into the manifest before the run)
--------------------------------------------------
* primary is arm B: wrong == 0 AND usable >= 0.60 AND unknown 20/20 ``no_match``
  AND ambiguous 20/20 -> ``offset_free_anchor_count_holds``;
* arm B wrong > 0 -> ``offset_not_the_cause`` (stop patching the address layer);
* arm B wrong == 0 AND usable < 0.60 -> ``offset_free_insufficient``;
* otherwise -> ``inconclusive``.
Arm A's usable ratio is reported separately: it prices the guard itself.

Discipline: 0 training / 0 optimizer steps / 0 Adam / 0 confirm; ``process_guard``
first; single-instance lock; 60 s watchdog; CPU cap 200 s for both arms together;
readings flushed as taken; write-then-rename; dual runner sha.  No
``python -m black``: only the repo's in-process gate.

SUPERSEDED.  The verdict was ``offset_not_the_cause`` and the stop-loss fired, so
``taiji/sc_sfk.py`` was rolled back to the stride-2 extraction (sha
``a158f59d…``).  A re-run therefore measures the shipped stride-2 features and does
**not** reproduce ``reports/taiji_offsetfree_20261108.json``; the file is kept only
as evidence, behind the environment gate in ``main``.
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

import probe_taiji_anchorcount_20261108 as ac  # noqa: E402

from taiji.living_core import LivingCoreConfig, LivingCoreReference  # noqa: E402
from taiji.process_guard import assert_no_black_processes  # noqa: E402
from taiji.sc_sfk import (  # noqa: E402
    AddressTable,
    document_frequency,
    features,
    idf_table,
    similarity,
)

MANIFEST = ROOT / "plans/manifests/taiji_offsetfree_20261108.json"
CORPUS = ROOT / "plans/manifests/taiji_offsetfree_corpus_20261108.json"
SOURCE_CORPUS = ROOT / "plans/manifests/taiji_anchorcount_corpus_20261108.json"
SOURCE_MANIFEST = ROOT / "plans/manifests/taiji_anchorcount_20261108.json"
REPORT = ROOT / "reports/taiji_offsetfree_20261108.json"
CLOSED_REPORT = ROOT / "reports/taiji_offsetfree_20261108_closed.json"
PROGRESS = ROOT / "reports/taiji_offsetfree_20261108_progress.jsonl"
LOCK = ROOT / "reports/taiji_offsetfree_20261108.lock"
WATCHDOG_MARKER = ROOT / "reports/taiji_offsetfree_20261108_watchdog.json"

CPU_CAP = 200.0
WATCHDOG_SECONDS = 60.0
LOCK_STALE_SECONDS = 600.0
N_WRITE = 58
N_UNKNOWN = 20
N_PAIR = 20
PAIR_STRIDE = 30
BATCH = 10
USABLE_LINE = 0.60

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
START_CPU = time.process_time()
START_WALL = time.time()
LAST_READING = [time.time()]
WATCHDOG_FIRED = [False]

file_sha = ac.file_sha
text_sha = ac.text_sha
write_json = ac.write_json
paraphrase = ac.paraphrase

CRITERIA = {
    "C1_one_place": (
        "only taiji.sc_sfk.features changed (every byte offset instead of every other "
        "offset).  Address definition, anchor rule, capacity, merge, reject and rollback "
        "are untouched."
    ),
    "C2_arm_a_is_product": (
        "arm A runs AddressTable.read as shipped, i.e. offset-free features plus the "
        "DF=0 guard.  It is the only path wired into taiji/."
    ),
    "C3_arm_b_is_shadow": (
        "arm B is a shadow evaluation in this harness, computed from public table state "
        "(addresses, document frequency, unique features).  It is not wired into taiji/ "
        "and creates no second judge in the product."
    ),
    "C4_closed_neutral": (
        "with LivingCoreConfig() the 176-read battery is compared bit for bit against the "
        "baseline arrays captured before taiji/ was edited; any difference in fast bytes, "
        "probabilities, parameters, state_bytes, candidate_sha_v2 or store signature is a "
        "hard failure and the open arms do not run."
    ),
    "C5_corpus_frozen": (
        "the corpus snapshot is written and hashed before the first reading and is "
        "byte-identical to the anchor-count gate corpus; the run aborts if either moved."
    ),
    "C6_no_tuning": (
        "0.60 usable line, denominators and verdict rules are frozen in the manifest "
        "before the run; no threshold, no leading margin, no retry with another rule."
    ),
    "V_holds": (
        "arm B: wrong == 0 AND usable >= 0.60 AND unknown 20/20 no_match AND ambiguous "
        "20/20 -> offset_free_anchor_count_holds."
    ),
    "V_offset_not_the_cause": (
        "arm B still has a wrong pointer -> offset_not_the_cause: stop patching the "
        "address layer; representation and addressing have to be redefined together."
    ),
    "V_insufficient": (
        "arm B wrong == 0 but usable < 0.60 -> offset_free_insufficient: report the "
        "distribution, do not move the line."
    ),
    "V_inconclusive": "neither branch reaches its condition cleanly.",
}


# ---------------------------------------------------------------------------
# Discipline helpers (same shape as the previous gate runners).
# ---------------------------------------------------------------------------
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
            LOCK.parent / "taiji_offsetfree_20261108_cpu_abort.json",
            {"aborted": True, "cpu": round(used, 3), "at": time.time()},
        )
        raise SystemExit(f"CPU cap {CPU_CAP}s exceeded ({used:.1f}s)")
    return used


# ---------------------------------------------------------------------------
# Freeze: carry the frozen corpus over, byte for byte.
# ---------------------------------------------------------------------------
def phase_freeze() -> int:
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    source_corpus = SOURCE_CORPUS.read_bytes()
    if file_sha(SOURCE_CORPUS) != source_manifest["corpus"]["snapshot_sha256"]:
        raise SystemExit("source corpus snapshot does not match its own manifest")
    corpus = json.loads(source_corpus.decode("utf-8"))

    written = corpus["written"]
    unknown = corpus["unknown"]
    unrelated = corpus["unrelated"]
    if (len(written), len(unknown), len(unrelated)) != (N_WRITE, N_UNKNOWN, 20):
        raise SystemExit(f"corpus shape changed: {len(written)}/{len(unknown)}/{len(unrelated)}")

    written_sha = [text_sha(item) for item in written]
    unknown_sha = [text_sha(item) for item in unknown]
    unrelated_sha = [text_sha(item) for item in unrelated]
    for key, _values, shas in (
        ("written", written, written_sha),
        ("unknown", unknown, unknown_sha),
        ("unrelated", unrelated, unrelated_sha),
    ):
        if shas != source_manifest["corpus"][f"{key}_sha256"]:
            raise SystemExit(f"{key} paragraphs moved between the two snapshots")

    CORPUS.write_text(
        json.dumps(
            {"written": written, "unknown": unknown, "unrelated": unrelated},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    pairs = [
        [index, index + PAIR_STRIDE]
        for index in range(N_PAIR)
        if index + PAIR_STRIDE < len(written)
    ]
    manifest = {
        "schema": "taiji-offsetfree-manifest-v1",
        "date": "2026-11-08",
        "freeze_revision": 1,
        "question": (
            "with offset-free feature extraction, does the anchor-count exit stay safe "
            "(arm B) and how much usable recall does the DF=0 guard cost (arm A)?"
        ),
        "change": {
            "file": "taiji/sc_sfk.py",
            "function": "features",
            "before": "range(0, len(raw) - 1, 2)  # every second offset",
            "after": "range(len(raw) - 1)  # every offset",
            "untouched": [
                "address definition",
                "anchor rule",
                "capacity / growth",
                "merge / replace / reject / rollback",
                "read exit names",
            ],
            "new_parameters": 0,
            "training_steps": 0,
        },
        "arms": {
            "A": "offset-free features + DF=0 guard (product path, AddressTable.read)",
            "B": "offset-free features + anchor count only (shadow in harness, not wired)",
        },
        "corpus": {
            "source_snapshot": str(SOURCE_CORPUS.relative_to(ROOT)),
            "source_snapshot_sha256": file_sha(SOURCE_CORPUS),
            "snapshot": str(CORPUS.relative_to(ROOT)),
            "snapshot_sha256": file_sha(CORPUS),
            "written_count": len(written),
            "unknown_count": len(unknown),
            "unrelated_count": len(unrelated),
            "written_sha256": written_sha,
            "unknown_sha256": unknown_sha,
            "unrelated_sha256": unrelated_sha,
        },
        "queries": {
            "self": "58 exact reads of the written paragraphs",
            "paraphrase_synonym": "fixed synonym table, kept only when it changes the text",
            "paraphrase_lead_in": "fixed prefix '总的来说，' + text + '。'",
            "unknown": "20 held-out paragraphs from a file that is never written",
            "ambiguous": "20 unions of two records' feature sets (pair stride 30)",
            "unrelated": "20 paragraphs from files that are never written (reading only)",
            "pairs": pairs,
        },
        "denominators": {
            "should_anchor": 120,
            "should_anchor_note": (
                "58 self + 58 lead_in + 4 synonym; the synonym count is 4 because the fixed "
                "table changes only 4 of the 58 paragraphs (same corpus as the parent gate)"
            ),
            "should_unknown": N_UNKNOWN,
            "should_ambiguous": len(pairs),
            "unrelated_reading": 20,
        },
        "verdict": {
            "primary_arm": "B",
            "usable_line": USABLE_LINE,
            "rules": {
                "offset_free_anchor_count_holds": (
                    "arm B wrong == 0 AND usable >= 0.60 AND unknown 20/20 no_match "
                    "AND ambiguous 20/20"
                ),
                "offset_not_the_cause": "arm B wrong > 0 (stop patching the address layer)",
                "offset_free_insufficient": "arm B wrong == 0 AND usable < 0.60",
                "inconclusive": "neither branch reaches its condition cleanly",
            },
        },
        "discipline": {
            "training": 0,
            "optimizer_steps": 0,
            "adam": 0,
            "confirm": 0,
            "new_parameters": 0,
            "cpu_cap_seconds": CPU_CAP,
            "watchdog_seconds": WATCHDOG_SECONDS,
            "format_gate": "in-process only; no python -m black",
        },
    }
    write_json(MANIFEST, manifest)
    append_progress({"phase": "frozen", "manifest_sha256": file_sha(MANIFEST)})
    print(json.dumps({"frozen": True, "manifest": str(MANIFEST)}, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# Closed phase: re-prove that the edited module is inert while the flag is off.
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
    identity = wire.identity_of(subject)
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
            identity["candidate_sha_v2"] == baseline["identity"]["candidate_sha_v2"]
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
        "schema": "taiji-offsetfree-closed-v1",
        "date": "2026-11-08",
        "question": "is the offset-free extraction inert while the switch is at its default?",
        "verdict": ("offset_free_closed_neutral" if neutral else "offset_free_closed_not_neutral"),
        "checks": checks,
        "baseline": {
            "report": wire.BASELINE_JSON.name,
            "living_core_sha256": baseline["living_core_sha256"],
            "candidate_sha_v2": baseline["identity"]["candidate_sha_v2"],
        },
        "current": {
            "candidate_sha_v2": identity["candidate_sha_v2"],
            "state_bytes": state_bytes,
        },
        "disclosures": [
            "the edit is confined to features(), which the store cannot reach while the "
            "flag is off (table_not_allocated); the battery is replayed anyway because the "
            "closed-state claim is re-proved, not inferred.",
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
        raise SystemExit("closed state is not neutral; the open arms do not run")
    return payload


# ---------------------------------------------------------------------------
# Open: one table, two arms over the same queries.
# ---------------------------------------------------------------------------
def anchor_read(
    query: tuple[str, ...],
    owner: dict[str, int],
    df: Any,
    unique: set[str],
) -> dict[str, Any]:
    """Arm B, shadow only: exits by anchor count, unseen features counted not used."""
    anchors = sorted({owner[feature] for feature in query if feature in owner})
    unknown = sum(1 for feature in query if df.get(feature, 0) == 0)
    if not anchors:
        decision = "no_match"
    elif len(anchors) > 1:
        decision = "ambiguous_multiple_anchors"
    else:
        decision = "matched"
    return {
        "decision": decision,
        "address": anchors[0] if decision == "matched" else None,
        "anchors": anchors,
        "unknown_features": unknown,
        "unique_features_in_query": sum(1 for feature in query if feature in unique),
    }


def evaluate_arm_a(table: AddressTable, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        began = time.perf_counter()
        result = table.read(item["feats"])
        elapsed = (time.perf_counter() - began) * 1000.0
        rows.append(
            {
                "kind": item["kind"],
                "variant": item.get("variant"),
                "expected": item["expected"],
                "decision": result["decision"],
                "address": result["address"],
                "anchors": len(result["anchors"]),
                "unknown_features": result["unknown_features"],
                "top1": result["top1"],
                "top2": result["top2"],
                "margin": result["margin"],
                "read_ms": round(elapsed, 4),
                "label": item["label"],
                "feats": item["feats"],
            }
        )
        touch()
    return rows


def evaluate_arm_b(
    table: AddressTable,
    items: list[dict[str, Any]],
    owner: dict[str, int],
    df: Any,
    unique: set[str],
) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        began = time.perf_counter()
        result = anchor_read(item["feats"], owner, df, unique)
        elapsed = (time.perf_counter() - began) * 1000.0
        rows.append(
            {
                "kind": item["kind"],
                "variant": item.get("variant"),
                "expected": item["expected"],
                "decision": result["decision"],
                "address": result["address"],
                "anchors": len(result["anchors"]),
                "unknown_features": result["unknown_features"],
                "read_ms": round(elapsed, 4),
                "label": item["label"],
                "feats": item["feats"],
            }
        )
        touch()
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["kind"], []).append(row)
    anchor_rows = groups.get("self", []) + groups.get("paraphrase", [])
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
        {
            "kind": row["kind"],
            "variant": row["variant"],
            "expected": row["expected"],
            "decision": row["decision"],
            "address": row["address"],
            "anchors": row["anchors"],
            "unknown_features": row["unknown_features"],
            "label": row["label"],
        }
        for row in anchor_rows
        if not (row["decision"] == "matched" and row["address"] == row["expected"])
    ]
    variants: dict[str, dict[str, int]] = {}
    for row in rows:
        key = f"{row['kind']}:{row['variant']}"
        bucket = variants.setdefault(
            key, {"n": 0, "matched": 0, "no_match": 0, "ambiguous": 0, "wrong": 0}
        )
        bucket["n"] += 1
        if row["decision"] == "matched":
            if row["expected"] in (None, "ambiguous") or row["address"] != row["expected"]:
                bucket["wrong"] += 1
            else:
                bucket["matched"] += 1
        elif row["decision"] == "no_match":
            bucket["no_match"] += 1
        else:
            bucket["ambiguous"] += 1
    unknown_rows = groups.get("unknown", [])
    amb_rows = groups.get("ambiguous", [])
    unrelated_rows = groups.get("unrelated", [])
    return {
        "should_anchor": {
            "n": len(anchor_rows),
            "matched_correct": correct,
            "wrong_pointer": wrong,
            "no_match": sum(1 for row in anchor_rows if row["decision"] == "no_match"),
            "ambiguous": sum(
                1 for row in anchor_rows if row["decision"] == "ambiguous_multiple_anchors"
            ),
            "usable_ratio": round(correct / len(anchor_rows), 6) if anchor_rows else 0.0,
        },
        "should_unknown": {
            "n": len(unknown_rows),
            "no_match": sum(1 for row in unknown_rows if row["decision"] == "no_match"),
            "matched": sum(1 for row in unknown_rows if row["decision"] == "matched"),
            "ambiguous": sum(
                1 for row in unknown_rows if row["decision"] == "ambiguous_multiple_anchors"
            ),
            "false_positives": sum(1 for row in unknown_rows if row["decision"] == "matched"),
        },
        "should_ambiguous": {
            "n": len(amb_rows),
            "ambiguous": sum(
                1 for row in amb_rows if row["decision"] == "ambiguous_multiple_anchors"
            ),
            "matched": sum(1 for row in amb_rows if row["decision"] == "matched"),
            "no_match": sum(1 for row in amb_rows if row["decision"] == "no_match"),
        },
        "unrelated_reading": {
            "n": len(unrelated_rows),
            "matched_hard_answer": sum(1 for row in unrelated_rows if row["decision"] == "matched"),
            "ambiguous": sum(
                1 for row in unrelated_rows if row["decision"] == "ambiguous_multiple_anchors"
            ),
            "no_match": sum(1 for row in unrelated_rows if row["decision"] == "no_match"),
            "matched_samples": [
                {"label": row["label"], "address": row["address"], "anchors": row["anchors"]}
                for row in unrelated_rows
                if row["decision"] == "matched"
            ][:5],
        },
        "by_kind": {
            key: {
                "n": len(value),
                "matched_correct": sum(
                    1
                    for row in value
                    if row["decision"] == "matched" and row["address"] == row["expected"]
                ),
                "wrong_pointer": sum(
                    1
                    for row in value
                    if row["decision"] == "matched" and row["address"] != row["expected"]
                ),
                "no_match": sum(1 for row in value if row["decision"] == "no_match"),
                "ambiguous": sum(
                    1 for row in value if row["decision"] == "ambiguous_multiple_anchors"
                ),
            }
            for key, value in groups.items()
        },
        "by_variant": variants,
        "failures": failures,
    }


def verdict_of(arm_b: dict[str, Any]) -> str:
    wrong = arm_b["should_anchor"]["wrong_pointer"]
    usable = arm_b["should_anchor"]["usable_ratio"]
    unknown_ok = arm_b["should_unknown"]["no_match"] == arm_b["should_unknown"]["n"]
    amb_ok = arm_b["should_ambiguous"]["ambiguous"] == arm_b["should_ambiguous"]["n"]
    if wrong > 0:
        return "offset_not_the_cause"
    if usable < USABLE_LINE:
        return "offset_free_insufficient"
    if unknown_ok and amb_ok:
        return "offset_free_anchor_count_holds"
    return "inconclusive"


def attribution(table: AddressTable, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify every arm-B wrong pointer: missing own features vs anything else."""
    idf = idf_table(table.addresses)
    address_sets = [set(address["features"]) for address in table.addresses]
    detail = []
    for row in rows:
        if row["kind"] not in ("self", "paraphrase"):
            continue
        if row["decision"] != "matched" or row["address"] == row["expected"]:
            continue
        own = address_sets[row["expected"]]
        other = address_sets[row["address"]]
        scores = sorted(
            (
                (similarity(row["feats"], address_sets[index], idf), index)
                for index in range(len(address_sets))
            ),
            reverse=True,
        )[:2]
        own_unique_present = sum(
            1 for feature in row["feats"] if feature in own and feature in table.unique_features()
        )
        other_unique_present = sum(
            1 for feature in row["feats"] if feature in other and feature in table.unique_features()
        )
        detail.append(
            {
                "kind": row["kind"],
                "variant": row["variant"],
                "expected": row["expected"],
                "address": row["address"],
                "anchors": row["anchors"],
                "own_unique_features_present": own_unique_present,
                "other_unique_features_present": other_unique_present,
                "top1": round(scores[0][0], 6) if scores else 0.0,
                "top2": round(scores[1][0], 6) if len(scores) > 1 else 0.0,
                "margin": round(scores[0][0] - scores[1][0], 6) if len(scores) > 1 else 0.0,
                "label": row["label"],
                "cause": (
                    "own_unique_features_absent"
                    if own_unique_present == 0
                    else "own_features_present_but_not_anchored"
                ),
            }
        )
    return {
        "wrong_total": len(detail),
        "own_unique_absent": sum(
            1 for item in detail if item["cause"] == "own_unique_features_absent"
        ),
        "own_present_not_anchored": sum(
            1 for item in detail if item["cause"] == "own_features_present_but_not_anchored"
        ),
        "note": (
            "anchor-count exits never read a similarity margin, so a near-zero lead cannot "
            "be the mechanism; the margin column is diagnostic only.  If own unique features "
            "are present the own address is an anchor by construction, so 'own present but "
            "not anchored' would signal a harness defect, not a system behaviour."
        ),
        "rows": detail,
    }


def open_table(rollback_on_failure: bool) -> dict[str, Any]:
    """Build the table through the shipped write path; return it with its write log."""
    config = LivingCoreConfig(sc_sfk_enabled=True)
    subject = LivingCoreReference(config)
    state = subject.attach_store(subject.initial_state("offsetfree"))
    table = state.sc_sfk
    if not isinstance(table, AddressTable):
        raise SystemExit("switch did not allocate an AddressTable")
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    written = [{"text": text} for text in corpus["written"]]

    write_costs = []
    actions: dict[str, int] = {}
    rejected: list[dict[str, Any]] = []
    value_payload = state.fast.detach().to("cpu").contiguous().numpy().tobytes()
    for start in range(0, len(written), BATCH):
        chunk = written[start : start + BATCH]
        began = time.perf_counter()
        result = table.write_batch_checked(
            [(features(item["text"].encode("utf-8")), value_payload) for item in chunk],
            rollback_on_failure=rollback_on_failure,
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
            rejected.append(
                {
                    "batch": start // BATCH,
                    "items": [start, min(start + BATCH, len(written))],
                    "action": result["action"],
                    "global_check": checks,
                    "collateral_loss": (result.get("global_check") or {}).get(
                        "collateral_loss", []
                    ),
                }
            )
            if rollback_on_failure:
                break
    return {
        "subject": subject,
        "state": state,
        "table": table,
        "written": written,
        "write_costs": write_costs,
        "actions": actions,
        "rejected": rejected,
    }


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

    built = open_table(rollback_on_failure=True)
    strict_built = built
    subject, state, table = built["subject"], built["state"], built["table"]
    append_progress({"phase": "store_attached", "table": True, "enabled": True})

    blocked = bool(built["rejected"])
    diagnostic = os.environ.get("TAIJI_OFFSETFREE_DIAGNOSTIC") == "1"
    blocker = None
    if blocked:
        blocker = {
            "stage": "write",
            "verdict": "offset_free_write_invariant_broken",
            "rule": "G-B (no collateral loss) rejected the frozen corpus batch",
            "batches": built["rejected"],
            "note": (
                "the shipped write path rolled the batch back, so the frozen 58-record table "
                "cannot be built under offset-free features and the frozen read readings do "
                "not exist under the shipped protocol.  The frozen verdict set has no branch "
                "for a rejected write, so the primary verdict stays whatever the read arms "
                "say and the blocker is recorded separately rather than invented into a new "
                "verdict name."
            ),
        }
    if blocked:
        append_progress(
            {
                "phase": "write_blocked",
                "batch": built["rejected"][0]["batch"],
                "action": built["rejected"][0]["action"],
                "collateral": built["rejected"][0]["collateral_loss"],
            }
        )
        if not diagnostic:
            payload = {
                "schema": "taiji-offsetfree-v1",
                "date": "2026-11-08",
                "question": manifest["question"],
                "verdict": "inconclusive",
                "manifest": str(MANIFEST.relative_to(ROOT)),
                "manifest_sha256": file_sha(MANIFEST),
                "corpus_snapshot_sha256": file_sha(CORPUS),
                "runner_sha_start": RUNNER_SHA_START,
                "runner_sha_end": file_sha(Path(__file__).resolve()),
                "change": manifest["change"],
                "closed": {
                    "report": str(CLOSED_REPORT.relative_to(ROOT)),
                    "verdict": closed["verdict"],
                    "fast_max_abs_delta": closed["checks"]["fast_max_abs_delta"],
                    "probability_max_abs_delta": closed["checks"]["probability_max_abs_delta"],
                    "readings": closed["checks"]["readings_total"],
                    "table_not_allocated": closed["checks"]["table_not_allocated"],
                },
                "blocker": blocker,
                "criteria": CRITERIA,
                "disclosures": [
                    "closed-state neutrality was re-proved before the open arms "
                    "(176/176, max|delta| = 0, no table allocated).",
                    "nothing was retuned: corpus, denominators, 0.60 line and write policy are "
                    "the frozen ones; the rejection is the system's own rollback.",
                ],
            }
            write_json(REPORT, payload)
            append_progress({"phase": "done", "verdict": "inconclusive", "blocked": True})
            print(
                json.dumps(
                    {"verdict": "inconclusive", "blocker": built["rejected"][0]}, ensure_ascii=False
                )
            )
            return 0
        built = open_table(rollback_on_failure=False)
        subject, state, table = built["subject"], built["state"], built["table"]
        append_progress({"phase": "diagnostic_table", "slots": len(table.addresses)})

    unique = table.unique_features()
    df = document_frequency(table.addresses)
    owner = {
        feature: index
        for index, address in enumerate(table.addresses)
        for feature in address["features"]
        if df.get(feature, 0) == 1
    }
    addressable = {
        index
        for index, address in enumerate(table.addresses)
        if any(feature in unique for feature in address["features"])
    }

    expected_index: dict[int, int] = {}
    duplicates = 0
    seen_targets: set[int] = set()
    for index, item in enumerate(written):
        found = table.find(features(item["text"].encode("utf-8")))
        if found is None:
            raise SystemExit(f"written record {index} is not in the table")
        if found in seen_targets:
            duplicates += 1
        seen_targets.add(found)
        expected_index[index] = found

    def anchors_own(text: str, target: int) -> bool:
        query = features(text.encode("utf-8"))
        owned = set(table.addresses[target]["features"])
        return any(feature in unique and feature in owned for feature in query)

    queries: list[dict[str, Any]] = []
    for index, item in enumerate(written):
        queries.append(
            {
                "kind": "self",
                "variant": "exact",
                "feats": features(item["text"].encode("utf-8")),
                "expected": expected_index[index],
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
                    "label": text[:80],
                }
            )
        lead = "总的来说，" + item["text"] + "。"
        queries.append(
            {
                "kind": "paraphrase",
                "variant": "lead_in",
                "feats": features(lead.encode("utf-8")),
                "expected": expected_index[index],
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

    rows_a = evaluate_arm_a(table, queries)
    rows_b = evaluate_arm_b(table, queries, owner, df, unique)
    summary_a = summarise(rows_a)
    summary_b = summarise(rows_b)
    arm_b_attribution = attribution(table, rows_b)

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

    verdict = verdict_of(summary_b)
    cpu = cpu_check()
    payload = {
        "schema": "taiji-offsetfree-v1",
        "date": "2026-11-08",
        "question": manifest["question"],
        "verdict": verdict,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": file_sha(MANIFEST),
        "corpus_snapshot_sha256": file_sha(CORPUS),
        "runner_sha_start": RUNNER_SHA_START,
        "runner_sha_end": file_sha(Path(__file__).resolve()),
        "change": manifest["change"],
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
        "write": {
            "batches": built["write_costs"],
            "actions": built["actions"],
            "rejected": built["rejected"],
            "rollback_on_failure": not diagnostic,
            "strict_attempt": {
                "actions": strict_built["actions"],
                "rejected": strict_built["rejected"],
            },
        },
        "diagnostic_only": bool(blocked and diagnostic),
        "blocker": blocker,
        "denominators": {
            "should_anchor": summary_b["should_anchor"]["n"],
            "should_unknown": summary_b["should_unknown"]["n"],
            "should_ambiguous": summary_b["should_ambiguous"]["n"],
            "unrelated_reading": summary_b["unrelated_reading"]["n"],
            "paraphrase_zero_replacements": paraphrase_zero,
        },
        "arms": {
            "A": {"definition": manifest["arms"]["A"], "readings": summary_a},
            "B": {"definition": manifest["arms"]["B"], "readings": summary_b},
        },
        "attribution_arm_b": arm_b_attribution,
        "cost": {
            "cpu_seconds": round(cpu, 3),
            "wall_seconds": round(time.time() - START_WALL, 3),
            "mean_global_ms": round(
                sum(row["global_ms"] for row in built["write_costs"])
                / max(len(built["write_costs"]), 1),
                4,
            ),
            "max_global_ms": round(
                max((row["global_ms"] for row in built["write_costs"]), default=0.0), 4
            ),
            "mean_read_ms_arm_a": round(
                sum(row["read_ms"] for row in rows_a) / max(len(rows_a), 1), 4
            ),
            "mean_read_ms_arm_b": round(
                sum(row["read_ms"] for row in rows_b) / max(len(rows_b), 1), 4
            ),
            "closed_wall_seconds": closed["cost"]["wall_seconds"],
        },
        "criteria": CRITERIA,
        "disclosures": [
            "arm B is a shadow evaluation in this harness; AddressTable.read (arm A) is the "
            "only judge wired into taiji/, so the product keeps a single decision rule.",
            "both arms read the same table, same writes, same queries; only the exit rule "
            "differs, which is what separates the offset question from the guard question.",
            "unseen features are counted in both arms; in arm B they never enter the exit.",
            "the unrelated set is a reading: a hard answer there is recorded, not scored.",
            "similarity columns (top1/top2/margin) are diagnostics; no exit reads them.",
        ],
    }
    write_json(REPORT, payload)
    append_progress({"phase": "done", "verdict": verdict})
    print(
        json.dumps(
            {
                "verdict": verdict,
                "arm_a": summary_a["should_anchor"],
                "arm_b": summary_b["should_anchor"],
                "arm_b_unknown": summary_b["should_unknown"],
                "arm_b_ambiguous": summary_b["should_ambiguous"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: list[str]) -> int:
    if os.environ.get("TAIJI_ALLOW_SUPERSEDED_RUNNER") != "1":
        raise SystemExit(
            "SUPERSEDED: features() was rolled back to stride 2 after this gate; "
            "set TAIJI_ALLOW_SUPERSEDED_RUNNER=1 only to inspect the evidence"
        )
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
