"""只读审计N1-132冻结训练流的转向答案和目标行覆盖。"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

from scripts.training import analyze_taiji_n1_v15_after_n132_itemwise_attribution as n133_itemwise
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132
from scripts.training import seal_ast  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
N133_PLAN = (
    ROOT / "plans/reference/PLAN-N1-133_v15_binding_supervision_confirmation_draft_20261019.md"
)
N133_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-133_execution_closeout_20261019.md"
N133_FACE = ROOT / "plans/manifests/n1_hard_neighbor_v15_confirmation_seed115_20261019.json"
N133_CONFIRM = ROOT / "reports/taiji_n1_v15_after_n132_confirmation_20261019.json"
OUTPUT = ROOT / "reports/taiji_n1_plan132_transition_training_coverage_audit_20261019.json"
DIRECTIONS = ("east", "north", "south", "west")
EXPECTED_VIEW_COUNTS = {
    (view, kind): 1152
    for view in ("raw", "hard")
    for kind in ("lookup", "revision", "transition", "unknown")
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _parent_id(item: dict) -> str:
    if item.get("parent_id"):
        return item["parent_id"]
    match = re.match(r"(?:raw|hard)-(train-\d+)", item.get("id", ""))
    return match.group(1) if match else item.get("id", "")


def build_report() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"一次性训练覆盖审计报告已存在：{OUTPUT}")
    started = time.process_time()
    context = n132._build_context(load_inputs=True)
    inputs = context.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("必须读取冻结的N1-132 train-only输入")
    items = inputs.get("items", [])
    if (
        len(items) != 192
        or any(item.get("split") != "train" for item in items)
        or Counter(item.get("kind") for item in items)
        != Counter({"lookup": 48, "revision": 48, "transition": 48, "unknown": 48})
    ):
        raise ValueError("N1-132必须绑定完整且仅含train的192项输入")

    raw_transitions = [item for item in items if item["kind"] == "transition"]
    raw_answer_counts = Counter(item["answer"] for item in raw_transitions)
    raw_position_counts = Counter(
        tuple(n132._target_observation_indices(item)) for item in raw_transitions
    )
    raw_observation_counts = Counter(len(item["observations"]) for item in raw_transitions)
    if (
        len(raw_transitions) != 48
        or set(raw_answer_counts) != set(DIRECTIONS)
        or raw_position_counts != Counter({(0, 1): 48})
        or raw_observation_counts != Counter({2: 48})
    ):
        raise ValueError("N1-132原始train transition分布或目标行位置改变")

    scheduled_by_view_kind = Counter()
    scheduled_by_answer = Counter()
    scheduled_by_view_answer_position = Counter()
    scheduled_by_view_answer_observations = Counter()
    parent_counts = Counter()
    for step in range(n132.ADDITIONAL_UPDATES):
        item, second = context["g"]["scheduled"](step, items, inputs["curriculum"])
        if item.get("split") != "train":
            raise ValueError("N1-132排程出现非train项目")
        if second is not None and (
            second.get("split") != "train" or second.get("observations") != item.get("observations")
        ):
            raise ValueError("paired问题偏离原train观测带")
        view = "hard" if item.get("id", "").startswith("hard-") else "raw"
        kind = item["kind"]
        scheduled_by_view_kind[(view, kind)] += 1
        parent_counts[_parent_id(item)] += 1
        if kind == "transition":
            answer = item["answer"]
            positions = tuple(n132._target_observation_indices(item))
            observations = len(item["observations"])
            scheduled_by_answer[answer] += 1
            scheduled_by_view_answer_position[(view, answer, positions)] += 1
            scheduled_by_view_answer_observations[(view, answer, observations)] += 1

    if (
        scheduled_by_view_kind != EXPECTED_VIEW_COUNTS
        or scheduled_by_answer
        != Counter({answer: count * 48 for answer, count in raw_answer_counts.items()})
        or len(parent_counts) != 192
        or set(parent_counts.values()) != {48}
    ):
        raise ValueError("N1-132冻结9,216步排程覆盖与回执不一致")

    face = read_json(N133_FACE)
    confirm = read_json(N133_CONFIRM)
    itemwise_report = read_json(n133_itemwise.OUTPUT)
    if (
        face.get("seed") != 115
        or face.get("observations_per_item") != 16
        or confirm.get("complete") is not True
        or confirm.get("can_promote") is not False
        or confirm.get("confirmation_queries") != 240
        or not seal_ast.receipt_binding_matches(
            itemwise_report.get("confirmation_report_sha256"), N133_CONFIRM
        )
    ):
        raise ValueError("N1-133已消费结果或v15面绑定失效")
    confirm_ids = set(face["subsets"]["confirm"])
    transition_items = [
        item
        for item in face["episodes"]
        if item["id"] in confirm_ids and item["kind"] == "transition"
    ]
    transition_outcomes = {
        row["id"]: row
        for row in itemwise_report["逐题正确与错误seed"]
        if row["kind"] == "transition"
    }
    if len(transition_items) != 5 or set(transition_outcomes) != {
        item["id"] for item in transition_items
    }:
        raise ValueError("v15 confirm transition题与离线结果不能逐项对应")
    confirm_transition_rows = []
    for item in transition_items:
        outcome = transition_outcomes[item["id"]]
        confirm_transition_rows.append(
            {
                "id": item["id"],
                "answer_direction": item["answer"],
                "target_start_index": item["target_start_index"],
                "candidate_correct_seed_count": len(outcome["N1_132_position_correct_seeds"]),
                "candidate_error_seeds": outcome["N1_132_position_error_seeds"],
            }
        )

    source_hashes = dict(context["source_hashes"])
    for path in (
        N133_PLAN,
        N133_CLOSEOUT,
        N133_FACE,
        N133_CONFIRM,
        n133_itemwise.OUTPUT,
        Path(n133_itemwise.__file__).resolve(),
        Path(__file__).resolve(),
    ):
        source_hashes[str(Path(path).resolve())] = sha256(path)
    for raw_path, digest in context["parent_hashes"].items():
        path = Path(raw_path)
        if sha256(path) != digest:
            raise ValueError(f"N1-120 position父checkpoint哈希发生变化：{path}")
        source_hashes[str(path.resolve())] = digest
    for raw_path, expected in source_hashes.items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"静态覆盖审计来源变化：{path}")

    report = {
        "schema": "n1-plan132-transition-training-coverage-audit-v1",
        "status": "complete",
        "source_hashes": source_hashes,
        "training_plan_sha256": sha256(n132.PLAN),
        "v15_face_sha256": sha256(N133_FACE),
        "v15_confirmation_sha256": sha256(N133_CONFIRM),
        "v15_itemwise_attribution_sha256": sha256(n133_itemwise.OUTPUT),
        "train_item_count": len(items),
        "train_split_only": True,
        "raw_transition_parent_count": len(raw_transitions),
        "raw_transition_answer_counts": dict(sorted(raw_answer_counts.items())),
        "raw_transition_target_positions": {
            str(position): count for position, count in sorted(raw_position_counts.items())
        },
        "raw_transition_observation_counts": {
            str(count): number for count, number in sorted(raw_observation_counts.items())
        },
        "scheduled_updates": n132.ADDITIONAL_UPDATES,
        "scheduled_transition_updates": sum(scheduled_by_answer.values()),
        "scheduled_updates_by_view_and_kind": {
            f"{view}/{kind}": count
            for (view, kind), count in sorted(scheduled_by_view_kind.items())
        },
        "scheduled_transition_updates_by_answer": dict(sorted(scheduled_by_answer.items())),
        "scheduled_transition_by_view_answer_and_position": {
            f"{view}/{answer}/{positions}": count
            for (view, answer, positions), count in sorted(
                scheduled_by_view_answer_position.items()
            )
        },
        "scheduled_transition_by_view_answer_and_observation_count": {
            f"{view}/{answer}/obs{observations}": count
            for (view, answer, observations), count in sorted(
                scheduled_by_view_answer_observations.items()
            )
        },
        "updates_per_train_parent": 48,
        "v15_confirm_transition_items": confirm_transition_rows,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "can_promote": False,
        "interpretation_zh": [
            "48个原始train transition样本四个方向均有覆盖，但频数为east 14、south 13、north 11、west 10；9,216步排程按同一频数重复48轮，没有额外均衡方向。",
            "原始与hard transition训练都只有2条观测，两个目标事实固定在索引0和1；训练排程没有16行上下文，也没有晚位目标行。",
            "N1-133的16行transition confirm在索引4与11上的south、north项目三seed全错，但索引、方向和题目只有单个样本对应，不能分离上下文长度、目标位置、答案方向或具体模板的影响。",
            "因此N1-134应同时保留2行训练匹配对照，并在16行上下文中完全平衡方向与目标位置；所有新查询仍属未批准的新测面。",
        ],
        "summary_zh": "N1-132 train-only transition覆盖静态审计完成；仅读取192条train样本和既定9,216步排程，0模型/ checkpoint加载、0查询、0参数更新。",
        "cpu_seconds": round(time.process_time() - started, 6),
    }
    write_new(OUTPUT, report)
    return report


if __name__ == "__main__":
    result = build_report()
    print(
        json.dumps(
            {
                "status": result["status"],
                "raw_transition_answer_counts": result["raw_transition_answer_counts"],
                "raw_transition_target_positions": result["raw_transition_target_positions"],
                "scheduled_transition_updates": result["scheduled_transition_updates"],
                "model_loads": result["model_loads"],
                "model_queries": result["model_queries"],
                "parameter_updates": result["parameter_updates"],
                "summary_zh": result["summary_zh"],
                "cpu_seconds": result["cpu_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
