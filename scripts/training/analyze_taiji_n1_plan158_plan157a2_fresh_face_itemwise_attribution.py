"""PLAN-N1-158：对N1-157A2已消费的fresh face评分做离线逐题配对归因。"""

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

from scripts.training import build_taiji_n1_plan157_seed128_candidate_vs_parent_face as n157_builder
from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder

PLAN = ROOT / "plans/reference/PLAN-N1-158_plan157a2_fresh_face_itemwise_attribution_20261020.md"
PLAN_SHA256 = "d0d0de644cb934468c56c071582b149d92372c963d7190fac1097a968b844db6"
SOURCE_RESULT = n157_builder.RESULT
SOURCE_RESULT_SHA256 = "c502ade59d20bcf05c7d84b898dcbc7b192187c42224c0e20ed342215a9cd1b5"
SOURCE_PROGRESS = n157_builder.PROGRESS
SOURCE_PROGRESS_SHA256 = "95992dc16b58ba32d5ea1de37837601b5eb1fef8d555e9bb24079d1193b2f441"
FACE = n157_builder.FACE
FACE_SHA256 = "cf34e7de874b316581c4c6520786e2bf6406d328d4989b869901eb667e9c9a2f"
REPORT = ROOT / "reports/taiji_n1_plan158_plan157a2_fresh_face_itemwise_attribution_20261020.json"
TEST = (
    ROOT
    / "tests/taiji_native/test_n1_plan158_plan157a2_fresh_face_itemwise_attribution_contract.py"
)

BRANCHES = ("unknown8_rebalanced", "length_mixed_parent")
SEEDS = (11, 12, 13)
LENGTHS = (8, 16, 32)
KINDS = ("lookup", "revision", "transition", "unknown")
EXPECTED_QUERIES = 2_880
PRIOR_ROWS = 480
NEW_ROWS = 2_400
EXPECTED_ROWS = PRIOR_ROWS + NEW_ROWS
CPU_LIMIT_SECONDS = 30.0
PAIR_KEY_FIELDS = ("seed", "parent_id", "context_length", "kind")
GROUP_FIELDS = ("seed", "context_length", "kind")
LENGTH_TRANSITIONS = ((8, 16), (16, 32), (8, 32))


class Refuse(RuntimeError):
    """N1-157A2面、分母、来源或配对合同不成立。"""


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


def paired_summary(rows: list[dict]) -> list[dict]:
    keyed: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        branch = str(row.get("branch"))
        if branch not in BRANCHES:
            raise Refuse(f"unexpected branch in fresh result: {branch}")
        key = tuple(row.get(field) for field in PAIR_KEY_FIELDS)
        arms = keyed.setdefault(key, {})
        if branch in arms:
            raise Refuse(f"duplicate seed128 paired item: {key}/{branch}")
        arms[branch] = row

    groups: dict[tuple, Counter] = defaultdict(Counter)
    for key, arms in keyed.items():
        if set(arms) != set(BRANCHES):
            raise Refuse(f"candidate/parent pair is incomplete: {key}")
        candidate = arms["unknown8_rebalanced"].get("free_exact") is True
        parent = arms["length_mixed_parent"].get("free_exact") is True
        group = tuple(arms["unknown8_rebalanced"].get(field) for field in GROUP_FIELDS)
        counts = groups[group]
        counts["pairs"] += 1
        counts["candidate_exact"] += candidate
        counts["parent_exact"] += parent
        counts["both_correct"] += candidate and parent
        counts["candidate_only_correct"] += candidate and not parent
        counts["parent_only_correct"] += parent and not candidate
        counts["both_wrong"] += not candidate and not parent

    summary = []
    for group in sorted(groups):
        record = dict(zip(GROUP_FIELDS, group, strict=True))
        record.update({key: int(value) for key, value in sorted(groups[group].items())})
        summary.append(record)
    return summary


def length_transition_summary(rows: list[dict]) -> list[dict]:
    by_item: dict[tuple, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = (
            int(row["seed"]),
            str(row["branch"]),
            str(row["kind"]),
            str(row["parent_id"]),
        )
        length = int(row["context_length"])
        if length in by_item[key]:
            raise Refuse(f"duplicate nested-length item: {key}/{length}")
        by_item[key][length] = row

    counts: dict[tuple, Counter] = defaultdict(Counter)
    for (seed, branch, kind, _parent_id), by_length in by_item.items():
        for before, after in LENGTH_TRANSITIONS:
            if before not in by_length or after not in by_length:
                raise Refuse(f"nested-length item is incomplete: {seed}/{branch}/{kind}")
            start = by_length[before].get("free_exact") is True
            end = by_length[after].get("free_exact") is True
            group = (seed, branch, kind, before, after)
            row = counts[group]
            row["items"] += 1
            row["wrong_to_right"] += not start and end
            row["right_to_wrong"] += start and not end
            row["right_both"] += start and end
            row["wrong_both"] += not start and not end

    output = []
    for seed, branch, kind, before, after in sorted(counts):
        record = {
            "seed": seed,
            "branch": branch,
            "kind": kind,
            "from_length": before,
            "to_length": after,
        }
        record.update(
            {
                key: int(value)
                for key, value in sorted(counts[(seed, branch, kind, before, after)].items())
            }
        )
        output.append(record)
    return output


def analyze() -> dict:
    started = time.process_time()
    if REPORT.exists():
        raise FileExistsError(f"N1-158 report is write-once: {REPORT}")
    if sha256(PLAN) != PLAN_SHA256 or sha256(SOURCE_RESULT) != SOURCE_RESULT_SHA256:
        raise Refuse("N1-158 plan or N1-157A2 result SHA changed")
    result = read_json(SOURCE_RESULT)
    if (
        result.get("complete") is not True
        or result.get("fresh_queries") != EXPECTED_QUERIES
        or result.get("new_fresh_queries") != NEW_ROWS
        or result.get("prior_consumed_queries") != PRIOR_ROWS
        or len(result.get("fresh_diagnostic_rows", [])) != EXPECTED_ROWS
        or result.get("source_hashes_unchanged") is not True
        or result.get("can_promote") is not False
        or result.get("dev_queries") != 0
        or result.get("holdout_queries") != 0
        or result.get("confirmation_queries") != 0
    ):
        raise Refuse("N1-157A2 result is incomplete or crosses the diagnostic boundary")
    if sha256(FACE) != FACE_SHA256:
        raise Refuse("seed128 face changed")
    if sha256(SOURCE_PROGRESS) != SOURCE_PROGRESS_SHA256:
        raise Refuse("N1-157A2 progress SHA changed")
    if not n150_builder.verify_hashes(result.get("source_hashes", {})):
        raise Refuse("N1-157A2 bound source/checkpoint hashes changed")

    rows = result["fresh_diagnostic_rows"]
    counts = Counter(
        (int(row["seed"]), str(row["branch"]), str(row["kind"]), int(row["context_length"]))
        for row in rows
    )
    expected = {
        (seed, branch, kind, length): 40
        for seed in SEEDS
        for branch in BRANCHES
        for kind in KINDS
        for length in LENGTHS
    }
    if dict(counts) != expected:
        raise Refuse("N1-158 seed/branch/kind/length denominator differs")
    paired = paired_summary(rows)
    transitions = length_transition_summary(rows)
    if len(paired) != len(SEEDS) * len(KINDS) * len(LENGTHS) or len(transitions) != len(
        SEEDS
    ) * len(BRANCHES) * len(KINDS) * len(LENGTH_TRANSITIONS):
        raise Refuse("N1-158 paired summary strata differ")
    if sha256(SOURCE_RESULT) != SOURCE_RESULT_SHA256 or sha256(FACE) != FACE_SHA256:
        raise Refuse("N1-158 source result or face changed during analysis")

    sources = {
        str(PLAN.resolve()): PLAN_SHA256,
        str(SOURCE_RESULT.resolve()): SOURCE_RESULT_SHA256,
        str(SOURCE_PROGRESS.resolve()): SOURCE_PROGRESS_SHA256,
        str(FACE.resolve()): FACE_SHA256,
        str(Path(__file__).resolve()): sha256(Path(__file__)),
        str(TEST.resolve()): sha256(TEST),
    }
    cpu = round(time.process_time() - started, 6)
    report = {
        "schema": "taiji-n1-plan158-plan157a2-fresh-face-itemwise-attribution-v1",
        "status": "complete",
        "complete": True,
        "plan_sha256": PLAN_SHA256,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "source_progress_sha256": SOURCE_PROGRESS_SHA256,
        "face_sha256": FACE_SHA256,
        "source_hashes": sources,
        "source_hashes_unchanged": n150_builder.verify_hashes(sources),
        "paired_row_count": len(rows),
        "pairs_by_seed_length_kind": paired,
        "nested_length_transitions_by_seed_branch_kind": transitions,
        "cpu_seconds": cpu,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "inference_calls": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
        "summary_zh": (
            f"完成N1-157A2的{len(rows)}条fresh face逐题配对归因；CPU{cpu:.3f}/{CPU_LIMIT_SECONDS:.0f}秒，"
            "0模型/检查点加载、0推理/新查询/训练/更新/dev/holdout/confirmation，can_promote=false。"
        ),
    }
    if cpu > CPU_LIMIT_SECONDS or report["source_hashes_unchanged"] is not True:
        raise Refuse("N1-158 CPU ceiling or output source binding failed")
    write_new(REPORT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-158 fresh-face itemwise paired attribution"
    )
    parser.parse_args()
    report = analyze()
    print(
        json.dumps(
            {
                "status": report["status"],
                "complete": report["complete"],
                "paired_row_count": report["paired_row_count"],
                "pair_strata": len(report["pairs_by_seed_length_kind"]),
                "length_transition_strata": len(
                    report["nested_length_transitions_by_seed_branch_kind"]
                ),
                "cpu_seconds": report["cpu_seconds"],
                "source_hashes_unchanged": report["source_hashes_unchanged"],
                "model_loads": report["model_loads"],
                "checkpoint_loads": report["checkpoint_loads"],
                "model_queries": report["model_queries"],
                "parameter_updates": report["parameter_updates"],
                "can_promote": report["can_promote"],
                "summary_zh": report["summary_zh"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
