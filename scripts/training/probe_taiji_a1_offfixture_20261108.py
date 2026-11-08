"""Off-fixture gate for assumption A1 ("every record carries a table-unique word").

A1 was the only fixture property left after the address system closed its loop.  This
runner tests it on text that was **not** authored to satisfy it: continuous project
prose, whose function words repeat heavily by nature.

Two phases, and the order is the point:

``freeze``  (must run before any reading)
    Pick the corpus by a mechanical rule (fixed files, fixed paragraph filters, first
    N in file order), apply one fixed synonym table to derive the paraphrase set, and
    write every choice -- source sha256, paragraph sha256, counts, query construction,
    verdict thresholds -- into the manifest.  After this point nothing is retuned.

``run``
    Re-hash the sources, abort if they moved, then build the table through the real
    switch (``LivingCoreConfig(sc_sfk_enabled=True)`` -> ``attach_store`` -> the
    ``AddressTable`` instance the switch owns), write the frozen corpus with the same
    ``write_batch_checked`` the store uses, and evaluate the frozen queries.

What is measured (denominators fixed before the first reading):

* ``should_anchor``  = 58 self-reads + one or two paraphrase variants per record
  (synonym rewrite when it changes the text, plus a fixed lead-in variant);
* ``should_unknown`` = 20 held-out paragraphs from a file that is never written;
* ``should_ambiguous`` = 20 deterministic two-record queries, built as the union of the
  two records' feature sets, counted only when both records are addressable.

Exits are the system's own ``table.read`` decisions.  No threshold is tuned, no leading
margin is added, and the similarity numbers are reported as diagnostics only.

Discipline: 0 training / 0 optimizer steps / 0 Adam / 0 confirm; ``process_guard``
before anything else; single-instance lock; 60 s watchdog; CPU cap 200 s; readings
flushed as taken; write-then-rename; dual runner sha.  No ``python -m black``: the
repo's in-process gate is the only formatter check.
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

from taiji.living_core import LivingCoreConfig, LivingCoreReference  # noqa: E402
from taiji.process_guard import assert_no_black_processes  # noqa: E402
from taiji.sc_sfk import AddressTable, features  # noqa: E402

MANIFEST = ROOT / "plans/manifests/taiji_a1_offfixture_20261108.json"
REPORT = ROOT / "reports/taiji_a1_offfixture_20261108.json"
PROGRESS = ROOT / "reports/taiji_a1_offfixture_20261108_progress.jsonl"
LOCK = ROOT / "reports/taiji_a1_offfixture_20261108.lock"
WATCHDOG_MARKER = ROOT / "reports/taiji_a1_offfixture_20261108_watchdog.json"

CPU_CAP = 200.0
WATCHDOG_SECONDS = 60.0
LOCK_STALE_SECONDS = 600.0
N_WRITE = 60
N_UNKNOWN = 20
N_PAIR = 20
PAIR_STRIDE = 30
MIN_CHARS = 120
MAX_CHARS = 500
BATCH = 10
USABLE_LINE = 0.60
REVISION_NOTES = {
    2: "value payload is the real fast snapshot, so the switch join check is meaningful",
    3: (
        "query construction corrected: paraphrase V2 (fixed lead-in) always applies; "
        "ambiguous queries are feature-set unions instead of text concatenation; corpus "
        "and verdict rules unchanged; revision 1/2 produced no recorded verdict reading"
    ),
}

WRITE_FILES = (
    "plans/MASTER-PLAN.md",
    "plans/reference/CAPABILITY-CARD_20261103.md",
)
UNKNOWN_FILES = ("plans/reference/INVARIANT-LEDGER_20261103.md",)
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

CRITERIA = {
    "A1_fixture_assumption": (
        "the whole address system assumes each stored record carries at least one word "
        "whose document frequency is 1, so that a query can be anchored to exactly one "
        "address.  This fixture property was never tested on text that was not authored "
        "to satisfy it."
    ),
    "C1_corpus_frozen_before_run": (
        "the manifest is written before the first reading and records the source file "
        "sha256, the paragraph sha256 list, the paraphrase table, the counts and the "
        "verdict thresholds.  The run aborts if any source file hash moved."
    ),
    "C2_same_implementation_same_switch": (
        "the table under test is the AddressTable allocated by LivingCoreConfig("
        "sc_sfk_enabled=True) through attach_store; writes use write_batch_checked and "
        "reads use table.read, the exact methods the store calls.  Zero new parameters, "
        "zero tensors."
    ),
    "C3_no_threshold_tuning": (
        "no numeric threshold on the read path, no leading margin, no retry with another "
        "address definition.  The usable line is fixed in the manifest before the run."
    ),
    "V_wrong_pointer": (
        "a should-anchor query answered 'matched' to an address other than its own record"
    ),
    "V_a1_required": (
        "wrong_pointer > 0: the no-wrong-answer property depends on A1; go back to the "
        "address definition, never add a threshold or a leading margin."
    ),
    "V_address_not_usable_off_fixture": (
        f"wrong_pointer = 0 AND usable_ratio < {USABLE_LINE}: the address definition does "
        "not carry enough unique anchors on natural prose."
    ),
    "V_address_holds_off_fixture": (
        f"wrong_pointer = 0 AND usable_ratio >= {USABLE_LINE}: still has no trainable "
        "degree of freedom; 0 parameters, 0 gradients."
    ),
    "V_inconclusive": "distribution only: neither branch reaches its threshold cleanly.",
}


# ---------------------------------------------------------------------------
# Discipline helpers (same shape as the store-wire runner).
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
            LOCK.parent / "taiji_a1_offfixture_20261108_cpu_abort.json",
            {"aborted": True, "cpu": round(used, 3), "at": time.time()},
        )
        raise SystemExit(f"CPU cap {CPU_CAP}s exceeded ({used:.1f}s)")
    return used


# ---------------------------------------------------------------------------
# Corpus construction (mechanical rule, fixed in the manifest).
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def paragraphs(path: Path) -> list[str]:
    """Split into paragraphs and strip markdown scaffolding.

    Rule: blank-line separated blocks; drop fenced code; drop table rows; strip
    heading / quote / bullet markers; join the remaining lines; keep 120..500 chars.
    No sentence is selected by hand and no sentence is dropped for its content.
    """
    blocks = re.split(r"\n\s*\n", normalize(path.read_text(encoding="utf-8")))
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


def collect() -> dict[str, Any]:
    written: list[dict[str, str]] = []
    for name in WRITE_FILES:
        for text in paragraphs(ROOT / name):
            written.append({"text": text, "source": name})
            if len(written) >= N_WRITE:
                break
        if len(written) >= N_WRITE:
            break
    unknown: list[dict[str, str]] = []
    for name in UNKNOWN_FILES:
        for text in paragraphs(ROOT / name):
            if any(text == item["text"] for item in written):
                continue
            unknown.append({"text": text, "source": name})
            if len(unknown) >= N_UNKNOWN:
                break
        if len(unknown) >= N_UNKNOWN:
            break
    return {"written": written, "unknown": unknown}


def paraphrase(text: str) -> tuple[str, int]:
    changed = 0
    for old, new in PARAPHRASE_RULES:
        changed += text.count(old)
        text = text.replace(old, new)
    return text, changed


def build_manifest(collection: dict[str, Any]) -> dict[str, Any]:
    written = collection["written"]
    unknown = collection["unknown"]
    para_sha = [text_sha(item["text"]) for item in written]
    unknown_sha = [text_sha(item["text"]) for item in unknown]
    pairs = []
    for index in range(N_PAIR):
        if index + PAIR_STRIDE >= len(written):
            break
        pairs.append([index, index + PAIR_STRIDE])
    unparaphrased = sum(1 for item in written if paraphrase(item["text"])[1] == 0)
    return {
        "schema": "taiji-a1-offfixture-v1",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runner_sha256": RUNNER_SHA_START,
        "question": (
            "does the SC-SFK address system keep its no-wrong-answer property, and stay "
            "usable, on text that was not authored to satisfy A1?"
        ),
        "corpus": {
            "write_files": list(WRITE_FILES),
            "write_file_sha256": {name: file_sha(ROOT / name) for name in WRITE_FILES},
            "unknown_files": list(UNKNOWN_FILES),
            "unknown_file_sha256": {name: file_sha(ROOT / name) for name in UNKNOWN_FILES},
            "extraction_rule": (
                "blank-line blocks; drop fenced code and table rows; strip heading / "
                f"quote / bullet markers; join lines; keep {MIN_CHARS}..{MAX_CHARS} chars; "
                f"take the first {N_WRITE} paragraphs in file order (write set) and the "
                f"first {N_UNKNOWN} paragraphs in file order from a disjoint file "
                "(unknown set).  No sentence is hand-picked, and no sentence is dropped "
                "because of its vocabulary."
            ),
            "write_count": len(written),
            "unknown_count": len(unknown),
            "written_sha256": para_sha,
            "unknown_sha256": unknown_sha,
            "written_sha256_aggregate": text_sha("".join(para_sha)),
            "unknown_sha256_aggregate": text_sha("".join(unknown_sha)),
        },
        "paraphrase": {
            "rule": "fixed global synonym table applied uniformly, in this order",
            "table": [list(pair) for pair in PARAPHRASE_RULES],
            "records_with_zero_replacements": unparaphrased,
            "hand_tuning": "none; a record with zero replacements is excluded and counted",
        },
        "queries": {
            "should_anchor": (
                f"{len(written)} self-reads (query = the stored text) + one or two "
                "paraphrase variants per record: V1 = the fixed synonym table (kept only "
                "when it changes the text), V2 = '总的来说，' + V1 + '。' (always applied).  "
                "Ground truth is the record itself."
            ),
            "should_unknown": (
                f"{len(unknown)} held-out paragraphs from a file that is never written; "
                "ground truth is 'no_match'"
            ),
            "should_ambiguous": (
                f"{len(pairs)} two-record queries built as the *union of the two feature "
                "sets* (paragraph k and k+"
                f"{PAIR_STRIDE}); ground truth is 'ambiguous_multiple_anchors', counted only "
                "when both records are addressable.  Text concatenation is not used: the "
                "junction would inject one unseen bigram and the DF=0 guard would answer "
                "'unknown', which tests a different condition than the anchor count."
            ),
            "pairs": pairs,
        },
        "denominators": {
            "should_anchor_upper_bound": (
                len(written) + len(written) + (len(written) - unparaphrased)
            ),
            "should_unknown": len(unknown),
            "should_ambiguous_upper_bound": len(pairs),
            "uniquely_anchorable_definition": (
                "queries whose own record has a table-unique word present in the query: "
                "A_i ∩ Q ∩ U(T) ≠ ∅.  Computed from the frozen corpus after the writes and "
                "before any query is evaluated; it is a denominator, never a criterion."
            ),
        },
        "verdict": {
            "usable_ratio_definition": (
                "should-anchor queries answered 'matched' to their own address, divided by "
                "all should-anchor queries (self + paraphrase).  Abstentions and ambiguous "
                "answers count as unusable; they are reported, not hidden."
            ),
            "usable_line": USABLE_LINE,
            "wrong_pointer": (
                "should-anchor query answered 'matched' to an address that is not its own"
            ),
            "rules": [
                "wrong_pointer > 0 -> a1_required",
                f"wrong_pointer = 0 and usable_ratio < {USABLE_LINE} -> "
                "address_not_usable_off_fixture",
                f"wrong_pointer = 0 and usable_ratio >= {USABLE_LINE} -> "
                "address_holds_off_fixture",
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


# ---------------------------------------------------------------------------
# Run.
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
        },
        "should_anchor": {
            "n": len(anchor_rows),
            "matched_correct": correct,
            "wrong_pointer": wrong,
            "no_match": rate(anchor_rows, "no_match"),
            "ambiguous": rate(anchor_rows, "ambiguous_multiple_anchors"),
            "usable_ratio": round(correct / len(anchor_rows), 6) if anchor_rows else 0.0,
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
        "failures": failures,
    }


def verdict_of(summary: dict[str, Any]) -> str:
    """The manifest's rules, applied without reinterpretation."""
    anchor = summary["should_anchor"]
    if anchor["wrong_pointer"] > 0:
        return "a1_required"
    if anchor["usable_ratio"] < USABLE_LINE:
        return "address_not_usable_off_fixture"
    if anchor["n"] and anchor["usable_ratio"] >= USABLE_LINE:
        return "address_holds_off_fixture"
    return "inconclusive"


def phase_freeze() -> int:
    collection = collect()
    manifest = build_manifest(collection)
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for key in (
            "write_file_sha256",
            "unknown_file_sha256",
            "written_sha256",
            "unknown_sha256",
        ):
            if old.get("corpus", {}).get(key) != manifest["corpus"][key]:
                raise SystemExit(f"refusing to refreeze: corpus {key} changed")
        if old.get("verdict") != manifest["verdict"]:
            raise SystemExit("refusing to refreeze: verdict rules changed")
        manifest["freeze_revision"] = int(old.get("freeze_revision", 1)) + 1
        revision = manifest["freeze_revision"]
        manifest["freeze_note"] = REVISION_NOTES.get(
            revision,
            "corpus and criteria unchanged; harness-only revision, no report reading "
            "existed under the previous revision",
        )
    else:
        manifest["freeze_revision"] = 1
        manifest["freeze_note"] = "initial freeze"
    write_json(MANIFEST, manifest)
    print(
        json.dumps(
            {
                "manifest": str(MANIFEST.relative_to(ROOT)),
                "written": manifest["corpus"]["write_count"],
                "unknown": manifest["corpus"]["unknown_count"],
                "aggregate": manifest["corpus"]["written_sha256_aggregate"][:16],
                "usable_line": USABLE_LINE,
            },
            ensure_ascii=False,
        )
    )
    return 0


def phase_run() -> int:
    if not MANIFEST.exists():
        raise SystemExit("manifest missing: run with --freeze first")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, digest in manifest["corpus"]["write_file_sha256"].items():
        if file_sha(ROOT / name) != digest:
            raise SystemExit(f"corpus moved after freeze: {name}")
    for name, digest in manifest["corpus"]["unknown_file_sha256"].items():
        if file_sha(ROOT / name) != digest:
            raise SystemExit(f"corpus moved after freeze: {name}")

    collection = collect()
    if [text_sha(item["text"]) for item in collection["written"]] != manifest["corpus"][
        "written_sha256"
    ]:
        raise SystemExit("write set differs from the frozen manifest")
    if [text_sha(item["text"]) for item in collection["unknown"]] != manifest["corpus"][
        "unknown_sha256"
    ]:
        raise SystemExit("unknown set differs from the frozen manifest")

    written = collection["written"]
    unknown = collection["unknown"]
    config = LivingCoreConfig(sc_sfk_enabled=True)
    subject = LivingCoreReference(config)
    state = subject.attach_store(subject.initial_state("a1-offfixture"))
    table = state.sc_sfk
    if not isinstance(table, AddressTable):
        raise SystemExit("switch did not allocate an AddressTable")
    append_progress({"phase": "store_attached", "table": True, "enabled": True})

    write_costs = []
    actions: dict[str, int] = {}
    # The value payload is the real one: a byte-exact ``fast`` snapshot.  A1 is a
    # property of the address layer, so the value cannot change any exit, but keeping
    # the real payload makes the switch-path join check (table.read vs addressed_read)
    # meaningful instead of a synthetic tag.
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
        """Is the query carrying a table-unique word that belongs to its own record?"""
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
        "schema": "taiji-a1-offfixture-v1",
        "verdict": verdict,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": file_sha(MANIFEST),
        "runner_sha_start": RUNNER_SHA_START,
        "runner_sha_end": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
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
            "uniquely_anchorable": summary["should_anchor"]["uniquely_anchorable"],
            "paraphrase_zero_replacements": paraphrase_zero,
        },
        "readings": summary,
        "write_costs": write_costs,
        "cost": {
            "cpu_seconds": round(cpu, 3),
            "wall_seconds": round(time.time() - START_WALL, 3),
            "mean_global_ms": round(
                sum(row["global_ms"] for row in write_costs) / max(len(write_costs), 1), 4
            ),
            "max_global_ms": round(max((row["global_ms"] for row in write_costs), default=0.0), 4),
            "mean_read_ms": round(
                sum(row["read_ms"] for row in readings["per_query"])
                / max(len(readings["per_query"]), 1),
                4,
            ),
        },
        "criteria": CRITERIA,
        "disclosures": [
            "the query set is frozen in the manifest before the run; the usable line was "
            "fixed at 0.60 before the first reading",
            "the address layer is exercised through the AddressTable the switch allocates; "
            "each write carries the live fast snapshot as its value, exactly as the store "
            "passes it, so the switch join check exercises the real value path",
            "held-out paragraphs come from a file that is never written; a 'matched' answer "
            "on one of them is reported separately as a false positive (unknown-side), and "
            "is not folded into wrong_pointer",
            "the paraphrase is lexical only (a fixed synonym table); clause reordering is "
            "not used because the address is position-free and a reorder would keep the "
            "feature set identical",
            "similarity columns are diagnostics; no exit reads them",
        ],
    }
    write_json(REPORT, payload)
    append_progress({"phase": "done", "verdict": verdict})
    print(
        json.dumps({"verdict": verdict, "readings": summary["should_anchor"]}, ensure_ascii=False)
    )
    print(
        json.dumps(
            {
                "unknown": summary["should_unknown"],
                "ambiguous": summary["should_ambiguous"],
                "cost": payload["cost"],
            },
            ensure_ascii=False,
        )
    )
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
        return phase_run()
    finally:
        stop.set()
        append_progress({"phase": "exit", "lock": lock_name, "watchdog": WATCHDOG_FIRED[0]})
        release_single_instance()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
