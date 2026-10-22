"""PLAN-N1-145：对已消费的v23确认结果作只读逐题归因。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PLAN = ROOT / "plans/reference/PLAN-N1-145_v23_itemwise_error_attribution_draft_20261019.md"
PLAN_SHA256 = "647c70092b6efae729f29849bf77e36032f899b63c05399f8c497e679fa651d9"
N144_PLAN = (
    ROOT / "plans/reference/PLAN-N1-144_v23_long_context_confirmation_after_n143_draft_20261019.md"
)
N144_PLAN_SHA256 = "b5527b5f8a56aa12604a7dbc40dad1fa74c3f4ea8e953a6b2539f1fa3c5fcf42"
FACE = ROOT / "plans/manifests/n1_v23_confirmation_seed123_20261019.json"
FACE_SHA256 = "e0ddf062e3c5f081dd010db3a8951d9aa426d9717de39872527a4eeabb76f6ae"
RESULT = ROOT / "reports/taiji_n1_v23_confirmation_result_20261019.json"
RESULT_SHA256 = "67038a9848b3c7ac088fea1983d9ba40ea153e91c8a4c436db830e8894e80388"
ATTEMPT = ROOT / "reports/taiji_n1_v23_confirmation_attempt_20261019.json"
PROGRESS = ROOT / "reports/taiji_n1_v23_confirmation_progress_20261019.jsonl"
OUTPUT = ROOT / "reports/taiji_n1_v23_itemwise_error_attribution_20261019.json"
TEST = ROOT / "tests/taiji_native/test_n1_plan145_v23_itemwise_error_attribution_contract.py"

CPU_LIMIT_SECONDS = 30.0
KINDS = ("lookup", "revision", "transition", "unknown")
SEEDS = (11, 12, 13)
COMPARATORS = {
    "n132_parent": ("N1-132", "parent", "parent"),
    "n140_block_candidate": ("N1-140", "candidate", "block_candidate"),
    "n141_control": ("N1-141", "control", "control"),
}
EXPECTED_FACE_POSITIONS = {
    "lookup": (0, 8, 16, 24, 31),
    "revision": (0, 8, 15, 23, 30),
    "transition": (0, 8, 15, 23, 30),
    "unknown": (),
}


class Refuse(RuntimeError):
    """冻结输入或逐题配对不符合合同。"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def expected_endpoints() -> dict[str, tuple[str, str, int]]:
    endpoints = {}
    for seed in SEEDS:
        endpoints[f"N1-132/seed{seed}/parent"] = ("N1-132", "parent", seed)
        endpoints[f"N1-140/seed{seed}/block_candidate"] = ("N1-140", "candidate", seed)
        endpoints[f"N1-141/seed{seed}/candidate"] = ("N1-141", "candidate", seed)
        endpoints[f"N1-141/seed{seed}/control"] = ("N1-141", "control", seed)
    return endpoints


def validate_and_join(
    confirm_rows: list[dict], face: dict
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    episodes = [item for item in face.get("episodes", []) if item.get("subset") == "confirm"]
    require(face.get("schema") == "n1-v23-confirmation-face-v1", "v23 face schema mismatch")
    require(
        face.get("seed") == 123
        and face.get("plan_sha256") == N144_PLAN_SHA256
        and len(episodes) == 80
        and all(item.get("split") == "holdout" for item in episodes),
        "v23 confirm face must contain 80 seed123 holdout items bound to N1-144",
    )
    face_by_id = {str(item.get("id")): item for item in episodes}
    require(len(face_by_id) == 80, "v23 confirm face IDs are not unique")
    expected_meta = expected_endpoints()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in confirm_rows:
        require(row.get("subset") == "confirm", "analysis input contains non-confirm row")
        endpoint_id = str(row.get("endpoint_id"))
        require(endpoint_id in expected_meta, f"unexpected endpoint: {endpoint_id}")
        require(
            type(row.get("free_exact")) is bool,
            f"missing exact-match boolean: {endpoint_id}/{row.get('id')}",
        )
        item_id = str(row.get("id"))
        require(item_id in face_by_id, f"confirmation row has no face item: {item_id}")
        item = face_by_id[item_id]
        require(row.get("kind") == item.get("kind"), f"row/face task kind mismatch: {item_id}")
        source, branch, seed = expected_meta[endpoint_id]
        require(
            (row.get("source"), row.get("branch"), row.get("seed")) == (source, branch, seed),
            f"endpoint provenance mismatch: {endpoint_id}",
        )
        grouped[endpoint_id].append(row)
    require(
        len(confirm_rows) == 960 and set(grouped) == set(expected_meta),
        "confirm input must cover 12 endpoints and 960 rows",
    )

    shared_ids: set[str] | None = None
    for endpoint_id, rows in grouped.items():
        ids = [str(row["id"]) for row in rows]
        require(
            len(rows) == 80 and len(set(ids)) == 80,
            f"endpoint must contain 80 unique rows: {endpoint_id}",
        )
        require(
            all(Counter(row.get("kind") for row in rows)[kind] == 20 for kind in KINDS),
            f"task balance differs: {endpoint_id}",
        )
        current_ids = set(ids)
        require(
            shared_ids is None or current_ids == shared_ids,
            "all endpoints must use the same 80 confirmation IDs",
        )
        shared_ids = current_ids

    require(set(face_by_id) == (shared_ids or set()), "face and confirmation row item sets differ")
    for item_id, item in face_by_id.items():
        kind = item.get("kind")
        require(kind in KINDS, f"unexpected task kind: {item_id}")
        start = item.get("target_start_index")
        if kind in ("lookup", "revision", "transition"):
            require(
                start in EXPECTED_FACE_POSITIONS[kind], f"unexpected target position: {item_id}"
            )
        else:
            require(start is None, f"unknown item must have no target position: {item_id}")
        if kind in ("lookup", "revision"):
            require(item.get("answer_length") in (6, 7), f"unexpected answer length: {item_id}")
    for kind, positions in EXPECTED_FACE_POSITIONS.items():
        if positions:
            position_counts = Counter(
                item.get("target_start_index")
                for item in face_by_id.values()
                if item.get("kind") == kind
            )
            require(
                all(position_counts[position] == 4 for position in positions),
                f"target position balance differs for {kind}",
            )
        if kind in ("lookup", "revision"):
            length_counts = Counter(
                item.get("answer_length")
                for item in face_by_id.values()
                if item.get("kind") == kind
            )
            require(
                all(length_counts[length] == 10 for length in (6, 7)),
                f"answer length balance differs for {kind}",
            )
    return dict(grouped), face_by_id


def recompute_confirmation_summary(rows: list[dict]) -> dict:
    """纯JSON逐题重算N1-144确认汇总，不加载其评估器或checkpoint。"""
    expected_meta = expected_endpoints()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["endpoint_id"])].append(row)
    require(set(grouped) == set(expected_meta), "confirmation endpoint set differs")

    per_endpoint = {}
    family_totals: Counter[tuple[str, str]] = Counter()
    shared_ids: set[str] | None = None
    for endpoint_id, selected in sorted(grouped.items()):
        ids = {str(row["id"]) for row in selected}
        require(
            len(selected) == 80 and len(ids) == 80, f"duplicate or incomplete rows: {endpoint_id}"
        )
        require(
            shared_ids is None or ids == shared_ids,
            "confirmation endpoints do not share one item set",
        )
        shared_ids = ids
        kind_counts = Counter(row.get("kind") for row in selected)
        require(
            all(kind_counts[kind] == 20 for kind in KINDS), f"kind counts differ: {endpoint_id}"
        )
        source, branch, seed = expected_meta[endpoint_id]
        require(
            all(
                (row.get("source"), row.get("branch"), row.get("seed")) == (source, branch, seed)
                for row in selected
            ),
            f"provenance differs: {endpoint_id}",
        )
        by_kind = {
            kind: sum(row.get("kind") == kind and row.get("free_exact") is True for row in selected)
            for kind in KINDS
        }
        exact = sum(row.get("free_exact") is True for row in selected)
        family_totals[(source, branch)] += exact
        per_endpoint[endpoint_id] = {
            "source": source,
            "branch": branch,
            "seed": seed,
            "items": len(selected),
            "exact": exact,
            "free_exact_by_kind": by_kind,
        }

    candidate = family_totals[("N1-141", "candidate")]
    n140_block = family_totals[("N1-140", "candidate")]
    control = family_totals[("N1-141", "control")]
    parent = family_totals[("N1-132", "parent")]
    candidates = [per_endpoint[f"N1-141/seed{seed}/candidate"] for seed in SEEDS]
    gates = {
        "candidate_each_seed_at_least_40_of_80": all(row["exact"] >= 40 for row in candidates),
        "candidate_each_seed_each_kind_at_least_8_of_20": all(
            all(row["free_exact_by_kind"][kind] >= 8 for kind in KINDS) for row in candidates
        ),
        "candidate_advantage_over_n141_control_at_least_48_of_240": candidate >= control + 48,
        "candidate_advantage_over_n132_parent_at_least_48_of_240": candidate >= parent + 48,
        "candidate_advantage_over_n140_block_at_least_24_of_240": candidate >= n140_block + 24,
    }
    return {
        "per_endpoint": per_endpoint,
        "aggregate_exact_by_family": {
            f"{source}/{branch}": total for (source, branch), total in sorted(family_totals.items())
        },
        "candidate_exact": candidate,
        "n141_control_exact": control,
        "n132_parent_exact": parent,
        "n140_block_candidate_exact": n140_block,
        "candidate_advantage_over_n141_control": candidate - control,
        "candidate_advantage_over_n132_parent": candidate - parent,
        "candidate_advantage_over_n140_block": candidate - n140_block,
        "gates": gates,
        "confirmation_pass": all(gates.values()),
    }


def _rate_record(selected: list[dict]) -> dict:
    exact = sum(row["free_exact"] is True for row in selected)
    return {
        "items": len(selected),
        "exact": exact,
        "exact_rate": round(exact / len(selected), 6) if selected else None,
    }


def build_attribution(confirm_rows: list[dict], face: dict) -> dict:
    grouped, face_by_id = validate_and_join(confirm_rows, face)
    by_endpoint_kind = {}
    for endpoint_id, rows in sorted(grouped.items()):
        by_endpoint_kind[endpoint_id] = {
            kind: _rate_record([row for row in rows if row["kind"] == kind]) for kind in KINDS
        }

    candidate_positions = []
    candidate_lengths = []
    for seed in SEEDS:
        endpoint_id = f"N1-141/seed{seed}/candidate"
        rows_by_id = {str(row["id"]): row for row in grouped[endpoint_id]}
        for kind in ("lookup", "revision", "transition"):
            by_position: dict[int, list[dict]] = defaultdict(list)
            for item_id, item in face_by_id.items():
                if item["kind"] == kind:
                    by_position[int(item["target_start_index"])].append(rows_by_id[item_id])
            for position, selected in sorted(by_position.items()):
                candidate_positions.append(
                    {
                        "seed": seed,
                        "kind": kind,
                        "target_start_index": position,
                        **_rate_record(selected),
                    }
                )
        for kind in ("lookup", "revision"):
            by_length: dict[int, list[dict]] = defaultdict(list)
            for item_id, item in face_by_id.items():
                if item["kind"] == kind:
                    by_length[int(item["answer_length"])].append(rows_by_id[item_id])
            for length, selected in sorted(by_length.items()):
                candidate_lengths.append(
                    {"seed": seed, "kind": kind, "answer_length": length, **_rate_record(selected)}
                )

    pairwise = []
    matrix_keys = (
        "candidate_correct_comparator_correct",
        "candidate_correct_comparator_wrong",
        "candidate_wrong_comparator_correct",
        "candidate_wrong_comparator_wrong",
    )
    for label, (source, _branch, endpoint_suffix) in COMPARATORS.items():
        for seed in SEEDS:
            candidate_id = f"N1-141/seed{seed}/candidate"
            comparator_id = f"{source}/seed{seed}/{endpoint_suffix}"
            candidate_rows = {str(row["id"]): row for row in grouped[candidate_id]}
            comparator_rows = {str(row["id"]): row for row in grouped[comparator_id]}
            require(
                candidate_rows.keys() == comparator_rows.keys(),
                f"paired IDs differ: {candidate_id}/{comparator_id}",
            )
            for kind in KINDS:
                selected_ids = [
                    str(row["id"]) for row in candidate_rows.values() if row["kind"] == kind
                ]
                counts = Counter()
                for item_id in selected_ids:
                    candidate_ok = candidate_rows[item_id]["free_exact"] is True
                    comparator_ok = comparator_rows[item_id]["free_exact"] is True
                    key = ("candidate_correct_" if candidate_ok else "candidate_wrong_") + (
                        "comparator_correct" if comparator_ok else "comparator_wrong"
                    )
                    counts[key] += 1
                require(
                    len(selected_ids) == 20,
                    f"paired task count differs: {candidate_id}/{kind} ({len(selected_ids)})",
                )
                pairwise.append(
                    {
                        "comparison": label,
                        "seed": seed,
                        "kind": kind,
                        "items": len(selected_ids),
                        "matrix": {key: counts[key] for key in matrix_keys},
                    }
                )

    return {
        "per_endpoint_kind": by_endpoint_kind,
        "n141_candidate_by_target_position": candidate_positions,
        "n141_candidate_by_answer_length": candidate_lengths,
        "n141_candidate_paired_transitions": pairwise,
    }


def validate_frozen_inputs() -> tuple[dict, dict, dict[str, str]]:
    for path, expected in (
        (PLAN, PLAN_SHA256),
        (N144_PLAN, N144_PLAN_SHA256),
        (FACE, FACE_SHA256),
        (RESULT, RESULT_SHA256),
    ):
        require(path.is_file() and sha256(path) == expected, f"frozen input hash mismatch: {path}")
    require(not OUTPUT.exists(), f"write-once attribution output already exists: {OUTPUT}")
    report = read_json(RESULT)
    face = read_json(FACE)
    require(
        report.get("status") == "confirmation_failed"
        and report.get("complete") is True
        and report.get("confirmation_queries") == 960
        and report.get("sacrifice_queries") == 960
        and len(report.get("confirmation_rows", [])) == 960
        and report.get("source_hashes_unchanged") is True
        and report.get("parameter_updates") == 0
        and report.get("optimizer_steps") == 0
        and report.get("dev_queries") == 0
        and report.get("v3_holdout_queries") == 0
        and report.get("can_promote") is False,
        "N1-144 result is not a complete, frozen confirmation failure",
    )
    require(report.get("plan_sha256") == N144_PLAN_SHA256, "N1-144 result is bound to another plan")
    require(report.get("face_sha256") == FACE_SHA256, "N1-144 result is bound to another face")
    attempt_sha = sha256(ATTEMPT)
    progress_sha = sha256(PROGRESS)
    require(
        report.get("attempt_sha256") == attempt_sha, "N1-144 attempt hash does not match result"
    )
    require(
        report.get("progress_sha256") == progress_sha, "N1-144 progress hash does not match result"
    )
    input_hashes = {
        str(path.resolve()): sha256(path)
        for path in (
            PLAN,
            N144_PLAN,
            FACE,
            RESULT,
            ATTEMPT,
            PROGRESS,
            Path(__file__).resolve(),
            TEST.resolve(),
        )
    }
    return report, face, input_hashes


def analyze() -> dict:
    started = time.process_time()
    report, face, input_hashes = validate_frozen_inputs()
    rows = report["confirmation_rows"]
    summary = recompute_confirmation_summary(rows)
    require(
        summary == report.get("confirmation_summary"),
        "recomputed confirmation summary differs from N1-144 result",
    )
    attribution = build_attribution(rows, face)
    require(
        all(sha256(Path(path)) == digest for path, digest in input_hashes.items()),
        "an analysis input changed during attribution",
    )
    cpu_seconds = time.process_time() - started
    require(
        cpu_seconds <= CPU_LIMIT_SECONDS, f"PLAN-N1-145 CPU ceiling exceeded: {cpu_seconds:.6f}s"
    )
    return {
        "schema": "n1-plan145-v23-itemwise-error-attribution-v1",
        "status": "complete",
        "complete": True,
        "plan_sha256": PLAN_SHA256,
        "n144_plan_sha256": N144_PLAN_SHA256,
        "face_sha256": FACE_SHA256,
        "n144_result_sha256": RESULT_SHA256,
        "attempt_sha256": report["attempt_sha256"],
        "progress_sha256": report["progress_sha256"],
        "analysis_source_hashes": input_hashes,
        "analysis_source_hashes_unchanged": True,
        "input_confirmation_rows": len(rows),
        "endpoint_count": 12,
        "confirmation_items_per_endpoint": 80,
        "recomputed_confirmation_summary_matches": True,
        "confirmation_pass": summary["confirmation_pass"],
        "attribution": attribution,
        "cpu_seconds": round(cpu_seconds, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "v3_holdout_queries": 0,
        "causal_claim": False,
        "capability_claim": False,
        "can_promote": False,
        "summary_zh": "N1-144已消费的960条confirm完成逐题/位置/答案长度描述性归因；汇总复算一致，未加载模型或checkpoint、未查询或训练，不作因果或能力主张。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-145 read-only v23 item-level error attribution"
    )
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if not args.analyze:
        parser.error("one-shot invocation requires --analyze")
    result = analyze()
    write_new(OUTPUT, result)
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "input_confirmation_rows",
                    "cpu_seconds",
                    "model_queries",
                    "parameter_updates",
                    "can_promote",
                    "summary_zh",
                )
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
