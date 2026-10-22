"""从N1-151A封存结果离线归因长度分层逐题变化；不加载模型。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "plans/reference/PLAN-N1-152_length_itemwise_attribution_20261020.md"
PLAN_SHA256 = "97612ef71939c0409f306d5b14e053fa63b36d4199f6bd0961403d128be2a1fa"
N151_RESULT = ROOT / "reports/taiji_n1_plan151a_train_score_ledger_recovery_result_20261020.json"
N151_RESULT_SHA256 = "7fb5ffa74b44ffa918d42a508c5d527991445367b3ec22457bb0d21a718ab6d6"
N150_RESULT = ROOT / "reports/taiji_n1_plan150_v27_length_result_20261020.json"
N150_RESULT_SHA256 = "e50e4b5aaf244ced4b5ef6232eb08a09206a5957f7d8f85abb1b3d841860ef0b"
N151A_PLAN = ROOT / "plans/reference/PLAN-N1-151A_source_binding_return_correction_20261020.md"
N151_FAILURE = ROOT / "reports/taiji_n1_plan151_preflight_binding_failure_20261020.json"
N151A_FAILURE = ROOT / "reports/taiji_n1_plan151_preflight_binding_failure_20261020.json"
RESULT = ROOT / "reports/taiji_n1_plan152_length_itemwise_attribution_20261020.json"
TEST = ROOT / "tests/taiji_native/test_n1_plan152_length_itemwise_attribution_contract.py"

SEEDS = (11, 12, 13)
BRANCHES = ("length_mixed", "eight_row_control")
LENGTHS = (8, 16, 32)
KINDS = ("lookup", "revision", "transition", "unknown")
EXPECTED_ROWS = 1_728
CPU_LIMIT_SECONDS = 30.0
UNKNOWN_HEX = b"unknown".hex()


class Refuse(RuntimeError):
    """N1-152离线来源、分母或配对合同不成立。"""


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


def verify_length_rows(rows: list[dict]) -> dict:
    if len(rows) != EXPECTED_ROWS:
        raise Refuse(f"expected {EXPECTED_ROWS} length rows, got {len(rows)}")
    counts = Counter()
    keyed = {}
    for row in rows:
        if row.get("event") != "length_score_row" or row.get("split") != "train":
            raise Refuse("N1-152 accepts only train length_score_row records")
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
            raise Refuse(
                f"unexpected endpoint or descriptor stratum: {seed}/{branch}/{length}/{kind}"
            )
        key = (seed, branch, length, item_id)
        if key in keyed:
            raise Refuse(f"duplicate endpoint/length/item row: {key}")
        keyed[key] = row
        counts[(seed, branch, length, kind)] += 1
        if (
            kind == "unknown"
            and row.get("free_exact") is True
            and row.get("output_hex") != UNKNOWN_HEX
        ):
            raise Refuse(f"unknown exact row is not a literal unknown output: {key}")
    expected_counts = {
        (seed, branch, length, kind): 24
        for seed in SEEDS
        for branch in BRANCHES
        for length in LENGTHS
        for kind in KINDS
    }
    if dict(counts) != expected_counts:
        raise Refuse("length rows do not contain 24 items per endpoint/length/task kind")

    matched_sets = {}
    for seed in SEEDS:
        for length in LENGTHS:
            mixed = {
                item_id
                for s, branch, l, item_id in keyed
                if s == seed and branch == "length_mixed" and l == length
            }
            control = {
                item_id
                for s, branch, l, item_id in keyed
                if s == seed and branch == "eight_row_control" and l == length
            }
            if mixed != control or len(mixed) != 96:
                raise Refuse(f"candidate/control item IDs are not paired: seed{seed}/{length}")
            matched_sets[f"{seed}/{length}"] = len(mixed)
    return {
        "rows": len(rows),
        "endpoint_length_kind_strata": len(counts),
        "items_per_endpoint_length_kind": 24,
        "candidate_control_paired_items_per_seed_length": matched_sets,
        "complete": True,
    }


def aggregate_by_branch_length_kind(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["branch"]), int(row["context_length"]), str(row["kind"]))].append(row)
    result = []
    for branch in BRANCHES:
        for length in LENGTHS:
            for kind in KINDS:
                selected = groups[(branch, length, kind)]
                result.append(
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
    return result


def matched_candidate_control(rows: list[dict]) -> list[dict]:
    by_endpoint_item = {
        (int(row["seed"]), int(row["context_length"]), str(row["id"]), str(row["branch"])): row
        for row in rows
    }
    groups: dict[tuple[int, str], Counter] = defaultdict(Counter)
    for seed in SEEDS:
        for length in LENGTHS:
            for item_id in sorted(
                {key[2] for key in by_endpoint_item if key[0] == seed and key[1] == length}
            ):
                candidate = by_endpoint_item[(seed, length, item_id, "length_mixed")]
                control = by_endpoint_item[(seed, length, item_id, "eight_row_control")]
                if candidate["kind"] != control["kind"]:
                    raise Refuse(f"candidate/control task kind differs: {seed}/{length}/{item_id}")
                outcome = (bool(candidate.get("free_exact")), bool(control.get("free_exact")))
                groups[(length, str(candidate["kind"]))][outcome] += 1
    labels = {
        (True, True): "both_correct",
        (True, False): "candidate_only_correct",
        (False, True): "control_only_correct",
        (False, False): "both_incorrect",
    }
    output = []
    for length in LENGTHS:
        for kind in KINDS:
            counts = groups[(length, kind)]
            output.append(
                {
                    "context_length": length,
                    "kind": kind,
                    "pairs": sum(counts.values()),
                    **{label: counts.get(outcome, 0) for outcome, label in labels.items()},
                }
            )
    return output


def within_branch_transitions(rows: list[dict]) -> list[dict]:
    keyed = {
        (
            int(row["seed"]),
            str(row["branch"]),
            str(row["kind"]),
            str(row["id"]),
            int(row["context_length"]),
        ): row
        for row in rows
    }
    labels = {
        (True, True): "both_correct",
        (True, False): "correct_to_incorrect",
        (False, True): "incorrect_to_correct",
        (False, False): "both_incorrect",
    }
    output = []
    for branch in BRANCHES:
        for first, second in ((8, 16), (16, 32), (8, 32)):
            for kind in KINDS:
                counts = Counter()
                for seed in SEEDS:
                    first_ids = {
                        item_id
                        for s, b, k, item_id, length in keyed
                        if s == seed and b == branch and k == kind and length == first
                    }
                    second_ids = {
                        item_id
                        for s, b, k, item_id, length in keyed
                        if s == seed and b == branch and k == kind and length == second
                    }
                    if first_ids != second_ids or len(first_ids) != 24:
                        raise Refuse(
                            f"within-branch item pairing differs: {seed}/{branch}/{kind}/{first}->{second}"
                        )
                    for item_id in first_ids:
                        a = keyed[(seed, branch, kind, item_id, first)]
                        b = keyed[(seed, branch, kind, item_id, second)]
                        counts[(bool(a.get("free_exact")), bool(b.get("free_exact")))] += 1
                output.append(
                    {
                        "branch": branch,
                        "kind": kind,
                        "from_length": first,
                        "to_length": second,
                        "pairs": sum(counts.values()),
                        **{label: counts.get(outcome, 0) for outcome, label in labels.items()},
                    }
                )
    return output


def analyze(result: dict) -> dict:
    rows = [
        row for row in result.get("itemwise_rows", []) if row.get("event") == "length_score_row"
    ]
    coverage = verify_length_rows(rows)
    return {
        "source_rows": len(rows),
        "coverage": coverage,
        "by_branch_length_kind": aggregate_by_branch_length_kind(rows),
        "matched_candidate_control": matched_candidate_control(rows),
        "within_branch_transitions": within_branch_transitions(rows),
        "can_promote": False,
    }


def run() -> dict:
    if RESULT.exists():
        raise FileExistsError("N1-152 result is write-once")
    started = time.process_time()
    if (
        sha256(PLAN) != PLAN_SHA256
        or sha256(N151A_PLAN) != "4892cc62f197689e0437a104bd2e592696ed67863a7746d4a7cb36a3ca0bc965"
    ):
        raise Refuse("N1-152 plan/source plan SHA changed")
    if sha256(N151_RESULT) != N151_RESULT_SHA256 or sha256(N150_RESULT) != N150_RESULT_SHA256:
        raise Refuse("N1-151A/N1-150 source result SHA changed")
    n151 = read_json(N151_RESULT)
    n150 = read_json(N150_RESULT)
    if (
        n151.get("complete") is not True
        or n151.get("status") != "complete"
        or n151.get("repeated_train_only_queries") != 7_488
        or n151.get("source_hashes_unchanged") is not True
        or n151.get("parameter_updates") != 0
        or n151.get("can_promote") is not False
        or n150.get("complete") is not True
        or n150.get("source_hashes_unchanged") is not True
        or n150.get("can_promote") is not False
    ):
        raise Refuse("N1-151A/N1-150 result boundary is incomplete")
    source_hashes = {
        str(path.resolve()): sha256(path)
        for path in (
            PLAN,
            N151A_PLAN,
            N151_RESULT,
            N150_RESULT,
            N151_FAILURE,
            Path(__file__).resolve(),
            TEST,
        )
    }
    if not all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in source_hashes.items()
    ):
        raise Refuse("N1-152 source hash changed before analysis")
    analysis = analyze(n151)
    cpu = time.process_time() - started
    if cpu > CPU_LIMIT_SECONDS:
        raise TimeoutError("N1-152 CPU ceiling exceeded")
    if not all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in source_hashes.items()
    ):
        raise Refuse("N1-152 source hash changed during analysis")
    report = {
        "schema": "n1-plan152-length-itemwise-attribution-v1",
        "status": "complete",
        "complete": True,
        "plan_sha256": PLAN_SHA256,
        "n151a_result_sha256": sha256(N151_RESULT),
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
        "summary_zh": "N1-152完成1,728条长度评分的逐题配对离线归因；0模型/checkpoint加载、0查询、0更新，描述性train-only，不作泛化或晋级主张。",
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="PLAN-N1-152 offline length itemwise attribution")
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
                    "source_rows",
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
