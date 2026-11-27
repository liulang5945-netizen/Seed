"""N1-154：N1-153A fresh face逐题配对归因；只读结果JSON。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "plans/reference/PLAN-N1-154_fresh_context_itemwise_attribution_20261020.md"
PLAN_SHA256 = "f795d6f8fa363c2c90a05ea2d5e5066d4fa159950c0c71eb3f165564ad35ec11"
N153A_RESULT = (
    ROOT / "reports/taiji_n1_plan153a_fresh_context_result_binding_recovery_result_20261020.json"
)
N153A_RESULT_SHA256 = "af8853b4f4677b469164e3691e429476d02887adc5a47f9981ed5b976926d06a"
N153A_PLAN = ROOT / "plans/reference/PLAN-N1-153A_fresh_context_result_binding_recovery_20261020.md"
N153A_FAILURE = (
    ROOT / "reports/taiji_n1_plan153_v28_fresh_context_length_confirmation_result_20261020.json"
)
N153_PLAN = ROOT / "plans/reference/PLAN-N1-153_fresh_context_length_paired_diagnostic_20261020.md"
N150_RESULT = ROOT / "reports/taiji_n1_plan150_v27_length_result_20261020.json"
N150_RESULT_SHA256 = "e50e4b5aaf244ced4b5ef6232eb08a09206a5957f7d8f85abb1b3d841860ef0b"
RESULT = ROOT / "reports/taiji_n1_plan154_fresh_context_itemwise_attribution_20261020.json"
TEST = ROOT / "tests/taiji_native/test_n1_plan154_fresh_context_itemwise_attribution_contract.py"

SEEDS = (11, 12, 13)
BRANCHES = ("length_mixed", "eight_row_control")
LENGTHS = (8, 16, 32)
KINDS = ("lookup", "revision", "transition", "unknown")
ROWS_PER_ENDPOINT = 480
EXPECTED_ROWS = 2_880
CPU_LIMIT_SECONDS = 30.0
UNKNOWN_HEX = b"unknown".hex()


class Refuse(RuntimeError):
    """N1-154输入或配对计数合同不成立。"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _validate_rows(rows: list[dict]) -> tuple[dict, dict]:
    if len(rows) != EXPECTED_ROWS:
        raise Refuse(f"expected {EXPECTED_ROWS} fresh diagnostic rows, got {len(rows)}")
    keyed = {}
    counts = Counter()
    for row in rows:
        if row.get("event") != "diagnostic_row" or row.get("split") != "diagnostic":
            raise Refuse("N1-154 accepts only diagnostic_row records")
        seed = int(row["seed"])
        branch = str(row["branch"])
        length = int(row["context_length"])
        kind = str(row["kind"])
        item_id = str(row["id"])
        if (
            seed not in SEEDS
            or branch not in BRANCHES
            or length not in LENGTHS
            or kind not in KINDS
        ):
            raise Refuse(f"unexpected N1-153A endpoint/stratum: {seed}/{branch}/{length}/{kind}")
        key = (seed, branch, length, item_id)
        if key in keyed:
            raise Refuse(f"duplicate itemwise diagnostic row: {key}")
        keyed[key] = row
        counts[(seed, branch, length, kind)] += 1
        if (
            kind == "unknown"
            and row.get("free_exact") is True
            and row.get("output_hex") != UNKNOWN_HEX
        ):
            raise Refuse(f"unknown correct-abstention row is not literal unknown: {key}")
    expected = {
        (seed, branch, length, kind): 40
        for seed in SEEDS
        for branch in BRANCHES
        for length in LENGTHS
        for kind in KINDS
    }
    if dict(counts) != expected:
        raise Refuse("N1-153A rows do not have 40 items per endpoint/length/kind")
    for seed in SEEDS:
        for length in LENGTHS:
            mixed = {
                item_id
                for s, b, l, item_id in keyed
                if s == seed and b == "length_mixed" and l == length
            }
            control = {
                item_id
                for s, b, l, item_id in keyed
                if s == seed and b == "eight_row_control" and l == length
            }
            if mixed != control or len(mixed) != 160:
                raise Refuse(f"candidate/control item IDs do not pair: seed{seed}/{length}")
    return keyed, {
        "rows": len(rows),
        "endpoint_length_kind_strata": len(counts),
        "items_per_stratum": 40,
        "complete": True,
    }


def summarize_rows(rows: list[dict]) -> dict:
    keyed, coverage = _validate_rows(rows)
    branch_groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        branch_groups[(str(row["branch"]), int(row["context_length"]), str(row["kind"]))].append(
            row
        )
    by_branch_length_kind = []
    for branch in BRANCHES:
        for length in LENGTHS:
            for kind in KINDS:
                selected = branch_groups[(branch, length, kind)]
                by_branch_length_kind.append(
                    {
                        "branch": branch,
                        "context_length": length,
                        "kind": kind,
                        "items": len(selected),
                        "free_exact": sum(row.get("free_exact") is True for row in selected),
                        "literal_unknown": sum(
                            row.get("output_hex") == UNKNOWN_HEX for row in selected
                        ),
                    }
                )

    outcome_labels = {
        (True, True): "both_correct",
        (True, False): "candidate_only_correct",
        (False, True): "control_only_correct",
        (False, False): "both_incorrect",
    }
    paired_counts = Counter()
    for seed in SEEDS:
        for length in LENGTHS:
            for kind in KINDS:
                for index in range(40):
                    item_id = f"v28-{kind}-{index:03d}-l{length}"
                    mixed = keyed[(seed, "length_mixed", length, item_id)]
                    control = keyed[(seed, "eight_row_control", length, item_id)]
                    if mixed["kind"] != control["kind"]:
                        raise Refuse(f"paired task kind mismatch: {seed}/{length}/{item_id}")
                    paired_counts[
                        (
                            seed,
                            length,
                            kind,
                            bool(mixed.get("free_exact")),
                            bool(control.get("free_exact")),
                        )
                    ] += 1
    candidate_control = []
    for seed in SEEDS:
        for length in LENGTHS:
            for kind in KINDS:
                row = {"seed": seed, "context_length": length, "kind": kind, "pairs": 40}
                for outcome, label in outcome_labels.items():
                    row[label] = paired_counts[(seed, length, kind, outcome[0], outcome[1])]
                candidate_control.append(row)

    transition_labels = {
        (True, True): "both_correct",
        (True, False): "correct_to_incorrect",
        (False, True): "incorrect_to_correct",
        (False, False): "both_incorrect",
    }
    transitions = []
    for seed in SEEDS:
        for branch in BRANCHES:
            for start, end in ((8, 16), (16, 32), (8, 32)):
                for kind in KINDS:
                    counts = Counter()
                    for index in range(40):
                        start_id = f"v28-{kind}-{index:03d}-l{start}"
                        end_id = f"v28-{kind}-{index:03d}-l{end}"
                        first = keyed[(seed, branch, start, start_id)]
                        second = keyed[(seed, branch, end, end_id)]
                        if first["parent_id"] != second["parent_id"]:
                            raise Refuse(
                                f"within-branch parent mismatch: {seed}/{branch}/{kind}/{start}->{end}"
                            )
                        counts[(bool(first.get("free_exact")), bool(second.get("free_exact")))] += 1
                    transitions.append(
                        {
                            "seed": seed,
                            "branch": branch,
                            "kind": kind,
                            "from_length": start,
                            "to_length": end,
                            "pairs": 40,
                            **{
                                label: counts.get(outcome, 0)
                                for outcome, label in transition_labels.items()
                            },
                        }
                    )
    return {
        "coverage": coverage,
        "by_branch_length_kind": by_branch_length_kind,
        "matched_candidate_control_by_seed_length_kind": candidate_control,
        "within_branch_transitions_by_seed_kind": transitions,
    }


def analyze(result: dict) -> dict:
    rows = result.get("diagnostic_rows", [])
    if result.get("status") != "complete" or result.get("complete") is not True:
        raise Refuse("N1-153A fresh-face result is not complete")
    if (
        result.get("new_query_count") != EXPECTED_ROWS
        or result.get("cumulative_query_count") != 3_360
        or result.get("source_hashes_unchanged") is not True
    ):
        raise Refuse("N1-153A query count/source integrity differs")
    summary = summarize_rows(rows)
    if len(result.get("summary_by_endpoint_length_kind", [])) != 72:
        raise Refuse("N1-153A aggregate summary does not contain 72 strata")
    return summary


def run() -> dict:
    if RESULT.exists():
        raise FileExistsError("N1-154 result is write-once")
    started = time.process_time()
    if sha256(PLAN) != PLAN_SHA256:
        raise Refuse("N1-154 plan SHA changed")
    if sha256(N153A_RESULT) != N153A_RESULT_SHA256 or sha256(N150_RESULT) != N150_RESULT_SHA256:
        raise Refuse("N1-153A/N1-150 result SHA changed")
    n153a = json.loads(N153A_RESULT.read_text(encoding="utf-8"))
    n150 = json.loads(N150_RESULT.read_text(encoding="utf-8"))
    if n150.get("complete") is not True or n150.get("can_promote") is not False:
        raise Refuse("N1-150 lineage result is incomplete")
    source_paths = (
        PLAN,
        N153A_PLAN,
        N153A_FAILURE,
        N153A_RESULT,
        N150_RESULT,
        Path(__file__).resolve(),
        TEST,
    )
    source_hashes = {str(Path(path).resolve()): sha256(Path(path)) for path in source_paths}
    analysis = analyze(n153a)
    cpu = time.process_time() - started
    if cpu > CPU_LIMIT_SECONDS:
        raise TimeoutError("N1-154 CPU ceiling exceeded")
    if not all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in source_hashes.items()
    ):
        raise Refuse("N1-154 source changed during offline analysis")
    report = {
        "schema": "n1-plan154-fresh-context-itemwise-attribution-v1",
        "status": "complete",
        "complete": True,
        "plan_sha256": PLAN_SHA256,
        "n153a_result_sha256": sha256(N153A_RESULT),
        "n150_result_sha256": sha256(N150_RESULT),
        "source_hashes": source_hashes,
        "source_hashes_unchanged": True,
        **analysis,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "cpu_seconds": round(cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "can_promote": False,
        "summary_zh": "N1-154完成2,880条fresh synthetic诊断的逐题配对归因；0模型/checkpoint加载、查询、更新，描述性结果不构成晋级。",
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-154 fresh context itemwise paired attribution"
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("one-shot invocation requires --run")
    report = run()
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in (
                    "status",
                    "complete",
                    "coverage",
                    "cpu_seconds",
                    "model_loads",
                    "model_queries",
                    "parameter_updates",
                    "can_promote",
                )
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if report.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
