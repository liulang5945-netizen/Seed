"""Read-only comparison of train-schedule and consumed diagnostic answer lengths."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "plans/reference/PLAN-N1-109_train_answer_length_coverage_audit_20261017.md"
SCHEDULE = ROOT / "plans/manifests/n1_hard_neighbor_training_schedule_v1_20261017.json"
FACE = ROOT / "plans/manifests/n1_hard_neighbor_v10_factorial_diagnostic_20261017.json"
REPORT = ROOT / "reports/taiji_n1_train_v10_answer_length_coverage_20261017.json"
CONTRACT_TEST = ROOT / "tests/taiji_native/test_n1_run109_answer_length_coverage_contract.py"

PLAN_SHA256 = "60704ef0802319dd742849c41c5eb19f057fbbf451d4cb8da3a86e53a47c6dd4"
SCHEDULE_SHA256 = "dc3a96ad17d43bd898e403ecd119c12c781317a71ab06ba39cdab6db24c40d50"
FACE_SHA256 = "9b23a8b4ad1508ab0c2b100b9eb98b9c4dcf572b031bb0782b1267737173d5ca"

SCHEMA = "n1-train-v10-answer-length-coverage-v1"
EXPECTED_KIND_COUNTS = {"lookup": 768, "revision": 768, "transition": 768, "unknown": 768}
EXPECTED_CONDITION_COUNTS = {
    "lookup/obs2/d1": 192,
    "lookup/obs2/d2": 192,
    "lookup/obs4/d1": 192,
    "lookup/obs4/d2": 192,
    "revision/obs4/d1": 384,
    "revision/obs4/d2": 384,
    "transition/obs2/dNone": 768,
    "unknown/obs2/d1": 192,
    "unknown/obs2/d2": 192,
    "unknown/obs4/d1": 192,
    "unknown/obs4/d2": 192,
}
CPU_LIMIT_SECONDS = 15.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _condition_key(entry: dict) -> str:
    distance = entry.get("hamming_distance")
    distance_label = "None" if distance is None else str(distance)
    return f"{entry['parent_kind']}/obs{entry['observation_count']}/d{distance_label}"


def _validate_schedule(schedule: dict) -> list[dict]:
    if schedule.get("schema") != "n1-hard-neighbor-training-schedule-v1":
        raise ValueError("unexpected train-derived schedule schema")
    entries = schedule.get("items")
    if not isinstance(entries, list) or len(entries) != 3072:
        raise ValueError("schedule must contain exactly 3,072 frozen train-derived entries")
    if schedule.get("summary", {}).get("schedule_items") != 3072:
        raise ValueError("schedule summary count mismatch")
    if (
        schedule.get("summary", {}).get("dev_queries") != 0
        or schedule.get("summary", {}).get("holdout_queries") != 0
        or schedule.get("summary", {}).get("parameter_updates") != 0
    ):
        raise ValueError("train schedule records prohibited dev/holdout queries or updates")

    kinds = Counter()
    conditions = Counter()
    for entry in entries:
        kind = entry.get("parent_kind")
        if kind not in EXPECTED_KIND_COUNTS:
            raise ValueError(f"unexpected training kind {kind}")
        kinds[kind] += 1
        conditions[_condition_key(entry)] += 1
        for field in ("item", "second_question"):
            nested = entry.get(field)
            if nested is None:
                continue
            if not isinstance(nested, dict) or nested.get("split") != "train":
                raise ValueError(f"non-train payload encountered in {field}")
            if not isinstance(nested.get("answer"), str):
                raise ValueError(f"missing answer text in train-only {field}")
    if kinds != EXPECTED_KIND_COUNTS or conditions != EXPECTED_CONDITION_COUNTS:
        raise ValueError("schedule kind/condition counts differ from frozen PLAN-N1-100")
    if schedule.get("summary", {}).get("kind_counts") != EXPECTED_KIND_COUNTS:
        raise ValueError("schedule's frozen kind summary differs from its entries")
    if schedule.get("summary", {}).get("condition_counts") != EXPECTED_CONDITION_COUNTS:
        raise ValueError("schedule's frozen condition summary differs from its entries")
    return entries


def _validate_face(face: dict) -> list[dict]:
    if (
        face.get("schema") != "n1-hard-neighbor-v10-factorial-diagnostic-v1"
        or face.get("split") != "diagnostic"
        or face.get("items") != 248
    ):
        raise ValueError("unexpected v10 diagnostic face identity/count")
    episodes = face.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 248:
        raise ValueError("v10 face must contain exactly 248 diagnostic items")
    if Counter(item.get("kind") for item in episodes) != {
        "lookup": 96,
        "revision": 88,
        "transition": 44,
        "unknown": 20,
    }:
        raise ValueError("v10 face kind counts differ from PLAN-N1-105")
    if any(item.get("split") != "diagnostic" for item in episodes):
        raise ValueError("unexpected non-diagnostic item in v10 face")
    return episodes


def _histograms(entries: list[dict], episodes: list[dict]) -> dict:
    schedule_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    schedule_condition_counts: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    for entry in entries:
        kind = entry["parent_kind"]
        condition = _condition_key(entry)
        for source_field in ("item", "second_question"):
            nested = entry.get(source_field)
            if nested is None:
                continue
            answer_length = len(nested["answer"])
            schedule_counts[(kind, source_field)][answer_length] += 1
            schedule_condition_counts[(kind, condition, source_field)][answer_length] += 1

    face_counts: dict[str, Counter] = defaultdict(Counter)
    for episode in episodes:
        face_counts[episode["kind"]][len(episode["answer"])] += 1

    lookup_revision_train = {
        kind: sorted(schedule_counts[(kind, "item")]) for kind in ("lookup", "revision")
    }
    lookup_revision_face = {kind: sorted(face_counts[kind]) for kind in ("lookup", "revision")}
    return {
        "schedule_by_kind_source_field": [
            {
                "kind": kind,
                "source_field": source_field,
                "denominator": sum(counts.values()),
                "answer_length_counts": {str(length): counts[length] for length in sorted(counts)},
            }
            for (kind, source_field), counts in sorted(schedule_counts.items())
        ],
        "schedule_by_condition_source_field": [
            {
                "kind": kind,
                "condition": condition,
                "source_field": source_field,
                "denominator": sum(counts.values()),
                "answer_length_counts": {str(length): counts[length] for length in sorted(counts)},
            }
            for (kind, condition, source_field), counts in sorted(schedule_condition_counts.items())
        ],
        "diagnostic_face_by_kind": [
            {
                "kind": kind,
                "denominator": sum(counts.values()),
                "answer_length_counts": {str(length): counts[length] for length in sorted(counts)},
            }
            for kind, counts in sorted(face_counts.items())
        ],
        "known_answer_length_support": {
            kind: {
                "train_schedule_item_answer_lengths": lookup_revision_train[kind],
                "diagnostic_face_answer_lengths": lookup_revision_face[kind],
                "diagnostic_lengths_absent_from_train_schedule": sorted(
                    set(lookup_revision_face[kind]) - set(lookup_revision_train[kind])
                ),
            }
            for kind in ("lookup", "revision")
        },
        "unknown_label_note": "The train schedule's literal unknown target answer is an abstention label, not a known material-answer length.",
    }


def build_report(schedule: dict, face: dict, before: dict[str, str], cpu_seconds: float) -> dict:
    entries = _validate_schedule(schedule)
    episodes = _validate_face(face)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "complete": True,
        "exploratory_train_exposure_audit_only": True,
        "plan_sha256": PLAN_SHA256,
        "schedule_sha256": SCHEDULE_SHA256,
        "face_sha256": FACE_SHA256,
        "source_hashes": {
            "analyzer": sha256_file(Path(__file__)),
            "contract_test": sha256_file(CONTRACT_TEST),
        },
        "input_hashes_before": before,
        "input_hashes_after": {
            "schedule": sha256_file(SCHEDULE),
            "face": sha256_file(FACE),
            "plan109": sha256_file(PLAN),
        },
        "inputs_unchanged": True,
        "schedule_entries": len(entries),
        "face_items": len(episodes),
        **_histograms(entries, episodes),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "cpu_seconds": cpu_seconds,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_or_holdout_access": False,
        "gate_recomputed": False,
        "can_promote": False,
        "interpretation_limit": "counts scheduled train-derived answer character lengths versus a consumed synthetic diagnostic face; no causal, capability, or promotion claim",
    }


def run() -> dict:
    started = time.process_time()
    for path, expected in ((PLAN, PLAN_SHA256), (SCHEDULE, SCHEDULE_SHA256), (FACE, FACE_SHA256)):
        if sha256_file(path) != expected:
            raise ValueError(f"frozen source hash mismatch: {path.relative_to(ROOT)}")
    before = {
        "schedule": sha256_file(SCHEDULE),
        "face": sha256_file(FACE),
        "plan109": sha256_file(PLAN),
    }
    schedule = read_json(SCHEDULE)
    face = read_json(FACE)
    report = build_report(schedule, face, before, time.process_time() - started)
    if before != report["input_hashes_after"]:
        raise ValueError("one or more read-only audit inputs changed during execution")
    report["inputs_unchanged"] = True
    report["cpu_seconds"] = time.process_time() - started
    if report["cpu_seconds"] > CPU_LIMIT_SECONDS:
        raise ValueError("PLAN-N1-109 CPU ceiling exceeded")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if REPORT.exists():
        raise FileExistsError(f"write-once report already exists: {REPORT}")
    report = run()
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with REPORT.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(REPORT.relative_to(ROOT)),
                "schedule_entries": report["schedule_entries"],
                "face_items": report["face_items"],
                "known_answer_length_support": report["known_answer_length_support"],
                "cpu_seconds": report["cpu_seconds"],
                "model_loads": report["model_loads"],
                "model_queries": report["model_queries"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
