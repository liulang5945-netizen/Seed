"""PLAN-N1-139：匹配对照的上下文鲁棒续训校准。

本runner只访问冻结v3 train输入；不读取dev/holdout，不建立或查询新测面。
阶段顺序固定为来源/schedule核验、六个zero bundle独立恢复、固定续训和
7,488条train-only计分。各阶段报告、attempt和checkpoint均write-once。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training import (  # noqa: E402
    build_taiji_n1_v20_neighbor_distance_dose_curve as v20,
)
from scripts.training import eval_taiji_n1_v11_after_n120 as e11  # noqa: E402
from scripts.training import (  # noqa: E402
    run_taiji_n1_plan132_binding_supervision_calibration as n132,
)
from scripts.training import seal_ast  # noqa: E402
from taiji.native_local_relation_evidence_candidate import (  # noqa: E402
    LocalRelationEvidenceCandidate,
)

PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-139_context_robust_matched_continuation_calibration_draft_20261019.md"
)
PLAN_SHA256 = "b28087dbca404517702d0f5850bf97dbe8f9e284de1b1c31c9ada02b73ef434b"
APPROVED_DRAFT_SHA256 = "bf64a126d927708b02f283d6e68357e1d054d358e4c74bf50c42927946298e5e"

N132_RESULT = n132.RESULT
N138_RESULT = v20.RUN_REPORT
N138_MANIFEST = v20.SOURCE_EXCLUSIONS
N138_FACE = v20.FACE
N138_FACE_CHECKS = v20.FACE_CHECKS

READINESS = (
    ROOT
    / "reports/taiji_n1_plan139_context_robust_matched_continuation_calibration_readiness_20261019.json"
)
READINESS_ATTEMPT = (
    ROOT
    / "reports/taiji_n1_plan139_context_robust_matched_continuation_calibration_readiness_attempt_20261019.json"
)
SOURCE_BINDING = (
    ROOT
    / "reports/taiji_n1_plan139_context_robust_matched_continuation_calibration_source_binding_20261019.json"
)
PREFLIGHT = (
    ROOT
    / "reports/taiji_n1_plan139_context_robust_matched_continuation_calibration_checkpoint_preflight_20261019.json"
)
PREFLIGHT_ATTEMPT = (
    ROOT
    / "reports/taiji_n1_plan139_context_robust_matched_continuation_calibration_checkpoint_preflight_attempt_20261019.json"
)
PREFLIGHT_BINDING = (
    ROOT
    / "reports/taiji_n1_plan139_context_robust_matched_continuation_calibration_checkpoint_preflight_binding_20261019.json"
)
ATTEMPT = (
    ROOT
    / "reports/taiji_n1_plan139_context_robust_matched_continuation_calibration_attempt_20261019.json"
)
PROGRESS = (
    ROOT
    / "reports/taiji_n1_plan139_context_robust_matched_continuation_calibration_progress_20261019.jsonl"
)
RESULT = (
    ROOT
    / "reports/taiji_n1_plan139_context_robust_matched_continuation_calibration_result_20261019.json"
)
CHECKPOINT_ROOT = ROOT / "data/checkpoints/n1_plan139_context_robust_matched_continuation_20261019"

SEEDS = (11, 12, 13)
BRANCHES = ("candidate", "control")
PARENT_UPDATES = 33_792
ADDITIONAL_UPDATES = 9_216
TOTAL_UPDATES = PARENT_UPDATES + ADDITIONAL_UPDATES
CPU_LIMIT_SECONDS = 3_600.0
TRAIN_KINDS = ("lookup", "revision", "transition", "unknown")
TRANSITION_CONDITIONS = (
    ("8/d1", 8, 1),
    ("8/d2", 8, 2),
    ("8/d4", 8, 4),
    ("16/d1", 16, 1),
    ("16/d2", 16, 2),
    ("16/d4", 16, 4),
)
BASELINE_CONDITION = "baseline"
EXPECTED_FROZEN_GATE_QUERIES_PER_ENDPOINT = 960
EXPECTED_AUGMENTED_QUERIES_PER_ENDPOINT = 48 * len(TRANSITION_CONDITIONS)
EXPECTED_TOTAL_TRAIN_QUERIES = 6 * (
    EXPECTED_FROZEN_GATE_QUERIES_PER_ENDPOINT + EXPECTED_AUGMENTED_QUERIES_PER_ENDPOINT
)
EXPECTED_PARENT_HASHES = {
    11: "3b8da0988e039bed63f64ffc51e7c5bdd252065a39cfebecf07e4d23a031d292",
    12: "a2430d8ec6a439890224476c4c7b6078abdbce8901adede694a3680351d62899",
    13: "9ab0d63ce0580cf1d96372c53806a592e76c2b1cd68fd712b51c9d2ed1b96d36",
}
EXPECTED_PARENT_FINGERPRINTS = {
    11: "63bec26c87dc0e80c1a858c2421cbb26aab33c9708e9104dc2b9cb1c9da9537f",
    12: "46740f1ab4be3d59332ba8f12a1d7b0893cdf3e01413317a36d308f93e13da56",
    13: "f7b37130f8aa6f09f990d0558d792560de04f5aad7ccd7e67f67ee17b5b2fcb4",
}
DIRECTIONS = tuple(v20.DIRECTIONS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _add_source(source_hashes: dict[str, str], path: Path, expected: str | None = None) -> None:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"冻结来源缺失：{path}")
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"冻结来源哈希改变：{path}")
    key = str(path)
    prior = source_hashes.get(key)
    if prior is not None and prior != actual:
        raise ValueError(f"来源哈希冲突：{path}")
    source_hashes[key] = actual


def _merge_source_hashes(target: dict[str, str], incoming: dict[str, str]) -> None:
    for raw_path, digest in incoming.items():
        path = str(Path(raw_path).resolve())
        prior = target.get(path)
        if prior is not None and prior != digest:
            raise ValueError(f"来源哈希冲突：{path}")
        target[path] = digest


def _verify_source_hashes(source_hashes: dict[str, str]) -> bool:
    for raw_path, expected in source_hashes.items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected:
            return False
    return True


class CpuBudget:
    def __init__(self, limit: float, *, child_cpu: float = 0.0):
        self.start = time.process_time()
        self.limit = float(limit)
        self.children = float(child_cpu)
        self.accounting_complete = True

    def used(self) -> float:
        return time.process_time() - self.start + self.children

    def guard(self, reserve: float = 0.0) -> None:
        if self.used() + reserve >= self.limit:
            raise TimeoutError("PLAN-N1-139累计CPU预算已到停止线")


def _require_plan_identity() -> None:
    if not PLAN.is_file() or sha256(PLAN) != PLAN_SHA256:
        raise ValueError("PLAN-N1-139冻结执行版SHA-256不匹配")


def _scheduled_identity(item: dict) -> tuple[str, str]:
    match = re.match(r"(raw|hard)-(train-\d+)", str(item.get("id", "")))
    if not match:
        raise ValueError(f"无法从schedule item标识raw/hard父项：{item.get('id')!r}")
    route, parent = match.groups()
    declared_parent = item.get("parent_id")
    if declared_parent is not None and declared_parent != parent:
        raise ValueError("schedule item id与parent_id不匹配")
    return parent, route


def _transition_semantics(item: dict) -> tuple[str, tuple[str, ...]]:
    subject = n132._query_subject(item["question"])
    if not re.fullmatch(r"[a-z]{10}", subject):
        raise ValueError(f"schedule transition目标必须为10位小写实体：{subject}")
    if item["question"] != f"direction of {subject}?":
        raise ValueError("transition改名必须只替换问题中的目标实体")
    rows = item.get("observations", [])
    if len(rows) != 2 or any(not row.startswith(subject + " ") for row in rows):
        raise ValueError("transition目标事实必须是同一目标实体的两行事实")
    return item["answer"], tuple(row[len(subject) :] for row in rows)


def _collect_original_schedule(
    g: dict, inputs: dict
) -> tuple[dict, list[dict], dict, list[dict], object]:
    original_scheduled = g["scheduled"]
    transition_records: list[dict] = []

    def record_schedule(step: int, items: list[dict], curriculum: dict):
        item, second = original_scheduled(step, items, curriculum)
        if item.get("kind") == "transition":
            parent, route = _scheduled_identity(item)
            transition_records.append(
                {
                    "step": int(step),
                    "parent_id": parent,
                    "route": route,
                    "item": copy.deepcopy(item),
                    "second": copy.deepcopy(second),
                }
            )
        return item, second

    g["scheduled"] = record_schedule
    try:
        original_summary = n132._simulate_schedule(g, inputs)
    finally:
        g["scheduled"] = original_scheduled

    if len(transition_records) != 2_304:
        raise ValueError("原schedule必须保留2,304个transition更新slot")
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    parent_counts = Counter()
    for record in transition_records:
        grouped[(record["parent_id"], record["route"])].append(record)
        parent_counts[record["parent_id"]] += 1
        item = record["item"]
        if (
            item.get("split") != "train"
            or item.get("kind") != "transition"
            or len(item.get("observations", [])) != 2
            or n132._target_observation_indices(item) != [0, 1]
        ):
            raise ValueError("N1-132 transition slot必须保留train-only两行原始目标事实")
        second = record["second"]
        if second is not None and second.get("observations") != item["observations"]:
            raise ValueError("transition paired query必须共享原始观察")
    if (
        len(parent_counts) != 48
        or set(parent_counts.values()) != {48}
        or set(grouped)
        != {(parent, route) for parent in parent_counts for route in ("raw", "hard")}
        or any(len(rows) != 24 for rows in grouped.values())
    ):
        raise ValueError("每个transition父项须有raw/hard各24个更新slot")

    raw_specimens = {}
    hard_records = []
    for parent in sorted(parent_counts):
        raw_rows = sorted(grouped[(parent, "raw")], key=lambda row: row["step"])
        hard_rows = sorted(grouped[(parent, "hard")], key=lambda row: row["step"])
        raw = raw_rows[0]["item"]
        raw_semantics = _transition_semantics(raw)
        for record in raw_rows:
            current = record["item"]
            if (
                current["question"] != raw["question"]
                or current["answer"] != raw["answer"]
                or current["observations"] != raw["observations"]
            ):
                raise ValueError(f"同一raw父项的原始两行输入不稳定：{parent}")
        raw_specimens[(parent, "raw")] = raw
        for record in hard_rows:
            current = record["item"]
            if _transition_semantics(current) != raw_semantics:
                raise ValueError(
                    f"hard路径必须只改名并保留父项目标事实与答案：{parent}/{record['step']}"
                )
            hard_records.append(record)
    return (
        original_summary,
        transition_records,
        raw_specimens,
        hard_records,
        original_scheduled,
    )


def build_context_assignment(records: list[dict]) -> tuple[dict[int, dict], dict]:
    """为每个transition父项固定2行基线及六种增强条件的raw/hard slot配比。"""
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for record in records:
        key = (record["parent_id"], record["route"])
        grouped[key].append(int(record["step"]))
    parents = sorted({parent for parent, _route in grouped})
    if len(parents) != 48 or len(grouped) != 96:
        raise ValueError("context assignment requires 48 parents with raw/hard routes")
    if any(len(grouped[(parent, route)]) != 24 for parent in parents for route in ("raw", "hard")):
        raise ValueError("each parent/route must expose exactly 24 transition slots")

    assignment: dict[int, dict] = {}
    per_parent = {}
    route_condition_counts = Counter()
    global_counts = Counter()
    for parent_index, parent in enumerate(parents):
        raw_extra = {
            TRANSITION_CONDITIONS[(parent_index * 3 + offset) % len(TRANSITION_CONDITIONS)][0]
            for offset in range(3)
        }
        parent_counts = Counter()
        for route in ("raw", "hard"):
            route_steps = sorted(grouped[(parent, route)])
            route_plan = [BASELINE_CONDITION] * 3
            for condition, _length, _distance in TRANSITION_CONDITIONS:
                if route == "raw":
                    count = 4 if condition in raw_extra else 3
                else:
                    count = 3 if condition in raw_extra else 4
                route_plan.extend([condition] * count)
            if len(route_plan) != 24:
                raise AssertionError("route condition assignment must fill 24 slots")
            for step, condition in zip(route_steps, route_plan, strict=True):
                if step in assignment:
                    raise ValueError("duplicate schedule step in context assignment")
                assignment[step] = {
                    "parent_id": parent,
                    "route": route,
                    "condition": condition,
                }
                parent_counts[condition] += 1
                route_condition_counts[(route, condition)] += 1
                global_counts[condition] += 1
        if parent_counts[BASELINE_CONDITION] != 6 or any(
            parent_counts[condition] != 7 for condition, _length, _distance in TRANSITION_CONDITIONS
        ):
            raise ValueError(f"per-parent context exposure counts differ: {parent}")
        per_parent[parent] = dict(sorted(parent_counts.items()))

    expected_global = Counter(
        {
            BASELINE_CONDITION: 288,
            **{condition: 336 for condition, _length, _distance in TRANSITION_CONDITIONS},
        }
    )
    if global_counts != expected_global:
        raise ValueError("candidate context exposure totals changed")
    if any(
        route_condition_counts[(route, condition)] != expected
        for condition, _length, _distance in TRANSITION_CONDITIONS
        for route, expected in (("raw", 168), ("hard", 168))
    ):
        raise ValueError("raw/hard context exposure balance changed")
    if any(route_condition_counts[(route, BASELINE_CONDITION)] != 144 for route in ("raw", "hard")):
        raise ValueError("raw/hard two-row baseline must split 144/144")

    summary = {
        "transition_slots": len(assignment),
        "parents": len(parents),
        "updates_per_parent": 48,
        "per_parent": per_parent,
        "by_condition": dict(sorted(global_counts.items())),
        "by_route_and_condition": {
            f"{route}/{condition}": route_condition_counts[(route, condition)]
            for route in ("raw", "hard")
            for condition in (BASELINE_CONDITION, *(row[0] for row in TRANSITION_CONDITIONS))
        },
    }
    return assignment, summary


def _hamming(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("Hamming distance requires equal-length entity names")
    return sum(a != b for a, b in zip(left, right, strict=True))


def _new_neighbor(subject: str, distance: int, used: set[str], rng: random.Random) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    if not re.fullmatch(r"[a-z]+", subject) or distance not in (1, 2, 4):
        raise ValueError("transition subject must be lowercase letters and d in {1,2,4}")
    for _ in range(20_000):
        chars = list(subject)
        for position in rng.sample(range(len(subject)), distance):
            chars[position] = rng.choice(letters.replace(chars[position], ""))
        candidate = "".join(chars)
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise ValueError(f"无法生成全新d={distance}实体：{subject}")


def build_context_variants(
    raw_specimens: dict[tuple[str, str], dict],
    hard_records: list[dict],
    assignment: dict[int, dict],
    excluded_names: set[str],
) -> tuple[dict, dict]:
    if len(raw_specimens) != 48 or len(hard_records) != 1_152:
        raise ValueError("N1-139 requires 48 stable raw targets and 1,152 scheduled hard slots")
    used = set(excluded_names)
    for record in hard_records:
        used.update(record["item"].get("entities", []))
        used.add(n132._query_subject(record["item"]["question"]))
    for item in raw_specimens.values():
        used.update(item.get("entities", []))
        used.add(n132._query_subject(item["question"]))
    initial_used = len(used)
    variants: dict = {}
    all_new_names: list[str] = []

    def make_distance_rows(subject: str, key: tuple, distance: int, count: int) -> dict:
        seed_material = "/".join(str(part) for part in (*key, distance)).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        rng = random.Random(seed)
        direction_seed = int.from_bytes(
            hashlib.sha256("/".join(str(part) for part in key).encode("utf-8")).digest()[:8],
            "big",
        )
        directions = [
            DIRECTIONS[(direction_seed + offset) % len(DIRECTIONS)] for offset in range(count)
        ]
        names = [_new_neighbor(subject, distance, used, rng) for _ in range(count)]
        if any(_hamming(subject, name) != distance for name in names):
            raise AssertionError("generated neighbor has incorrect Hamming distance")
        observations = [
            f"{name} faces {direction}." for name, direction in zip(names, directions, strict=True)
        ]
        all_new_names.extend(names)
        return {"entities": names, "observations": observations}

    for key, item in sorted(raw_specimens.items()):
        subject = n132._query_subject(item["question"])
        by_distance = {
            distance: make_distance_rows(subject, (key[0], "raw"), distance, 14)
            for distance in (1, 2, 4)
        }
        variants[key] = {"subject": subject, "by_distance": by_distance}

    hard_requirements: dict[tuple[str, str], Counter] = defaultdict(Counter)
    hard_augmented_slots = 0
    for record in hard_records:
        step = int(record["step"])
        condition = assignment.get(step)
        if (
            condition is None
            or condition.get("parent_id") != record["parent_id"]
            or condition.get("route") != "hard"
        ):
            raise ValueError("hard-slot context assignment differs from the frozen schedule")
        if condition["condition"] == BASELINE_CONDITION:
            continue
        length, distance = next(
            (length, distance)
            for name, length, distance in TRANSITION_CONDITIONS
            if name == condition["condition"]
        )
        subject = n132._query_subject(record["item"]["question"])
        key = (record["parent_id"], subject)
        hard_requirements[key][distance] = max(hard_requirements[key][distance], length - 2)
        hard_augmented_slots += 1
    if hard_augmented_slots != 1_008:
        raise ValueError("each parent must retain 21 nonbaseline hard-route slots")

    hard_distance_variant_count = 0
    for key in sorted(hard_requirements):
        subject = key[1]
        by_distance = {}
        for distance, count in sorted(hard_requirements[key].items()):
            by_distance[distance] = make_distance_rows(subject, (key[0], "hard"), distance, count)
            hard_distance_variant_count += 1
        variants[(key[0], "hard", subject)] = {
            "subject": subject,
            "by_distance": by_distance,
        }

    raw_name_count = 48 * 3 * 14
    hard_name_count = len(all_new_names) - raw_name_count
    if (
        raw_name_count != len(raw_specimens) * 3 * 14
        or len(all_new_names) != len(set(all_new_names))
        or set(all_new_names) & excluded_names
        or hard_name_count <= 0
    ):
        raise ValueError("N1-139 filler entities must be globally unique and source-excluded")
    summary = {
        "raw_parent_target_variants": len(raw_specimens),
        "raw_filler_entity_count": raw_name_count,
        "hard_unique_target_subjects_with_augmented_slots": len(hard_requirements),
        "hard_target_distance_variants": hard_distance_variant_count,
        "hard_augmented_slots": hard_augmented_slots,
        "hard_filler_entity_count": hard_name_count,
        "unique_new_entity_count": len(all_new_names),
        "names_per_raw_variant_per_distance": 14,
        "names_per_hard_subject_distance": "maximum required by scheduled context length",
        "distances": [1, 2, 4],
        "entity_length": 10,
        "excluded_entity_count": len(excluded_names),
        "generated_entity_sha256": hashlib.sha256(
            "\n".join(sorted(all_new_names)).encode("utf-8")
        ).hexdigest(),
        "used_name_pool_size_including_train_entities": initial_used,
    }
    return variants, summary


def validate_context_schedule(
    records: list[dict],
    assignment: dict[int, dict],
    variants: dict,
    excluded_names: set[str],
) -> dict:
    observed = Counter()
    lengths = Counter()
    for record in records:
        step = int(record["step"])
        parent = record["parent_id"]
        route = record["route"]
        item = record["item"]
        spec = assignment.get(step)
        if spec is None or spec["parent_id"] != parent or spec["route"] != route:
            raise ValueError("context validation saw a schedule/assignment mismatch")
        condition = spec["condition"]
        observed[(route, condition)] += 1
        if condition == BASELINE_CONDITION:
            if len(item.get("observations", [])) != 2:
                raise ValueError("baseline context must preserve exactly two target rows")
            continue
        if route == "raw":
            variant = variants[(parent, "raw")]
        else:
            subject = n132._query_subject(item["question"])
            variant = variants[(parent, "hard", subject)]
        changed = _apply_context(item, variant, condition)
        length, distance = next(
            (length, distance)
            for name, length, distance in TRANSITION_CONDITIONS
            if name == condition
        )
        filler_count = length - 2
        filler_names = changed["entities"][-filler_count:]
        filler_rows = changed["observations"][-filler_count:]
        target = n132._query_subject(item["question"])
        if (
            changed["question"] != item["question"]
            or changed["answer"] != item["answer"]
            or changed["observations"][:2] != item["observations"]
            or len(changed["observations"]) != length
            or len(filler_names) != filler_count
            or set(filler_names) & excluded_names
            or any(_hamming(target, name) != distance for name in filler_names)
            or any(
                not row.startswith(name + " faces ")
                or row.rsplit(" ", 1)[-1].removesuffix(".") not in DIRECTIONS
                for name, row in zip(filler_names, filler_rows, strict=True)
            )
        ):
            raise ValueError(
                f"scheduled context violates target/length/distance contract at {step}"
            )
        lengths[str(length)] += 1

    expected = Counter()
    expected[("raw", BASELINE_CONDITION)] = 144
    expected[("hard", BASELINE_CONDITION)] = 144
    for condition, _length, _distance in TRANSITION_CONDITIONS:
        expected[("raw", condition)] = 168
        expected[("hard", condition)] = 168
    if observed != expected:
        raise ValueError("context validation counts differ from the fixed per-parent schedule")
    if sum(lengths.values()) != 2_016:
        raise ValueError("candidate schedule must expose exactly 2,016 augmented transition slots")
    return {
        "complete": True,
        "validated_transition_slots": len(records),
        "validated_augmented_slots": sum(lengths.values()),
        "validated_baseline_slots": sum(
            value
            for (_route, condition), value in observed.items()
            if condition == BASELINE_CONDITION
        ),
        "augmented_context_length_counts": dict(sorted(lengths.items())),
        "by_route_and_condition": {
            f"{route}/{condition}": observed[(route, condition)]
            for route in ("raw", "hard")
            for condition in (BASELINE_CONDITION, *(row[0] for row in TRANSITION_CONDITIONS))
        },
    }


def _apply_context(
    item: dict, variant: dict, condition: str, *, item_id: str | None = None
) -> dict:
    selected = {name: (length, distance) for name, length, distance in TRANSITION_CONDITIONS}
    if condition not in selected:
        raise ValueError(f"unknown augmented context condition: {condition}")
    length, distance = selected[condition]
    filler_count = length - 2
    filler = variant["by_distance"][distance]
    names = filler["entities"][:filler_count]
    rows = filler["observations"][:filler_count]
    if len(item.get("observations", [])) != 2 or len(names) != filler_count:
        raise ValueError("context augmentation requires the original two target rows")
    if (
        item["observations"][0].split()[0] != variant["subject"]
        or item["observations"][1].split()[0] != variant["subject"]
    ):
        raise ValueError("target facts must remain first and unchanged")
    changed = copy.deepcopy(item)
    changed["observations"] = [*item["observations"], *rows]
    merged_entities = list(item.get("entities", []))
    if set(merged_entities) & set(names):
        raise ValueError("fresh filler entity collided with a scheduled train entity")
    changed["entities"] = [*merged_entities, *names]
    if item_id is not None:
        changed["id"] = item_id
    if len(changed["observations"]) != length:
        raise AssertionError("augmented context length differs from its condition")
    return changed


def _validate_n132_parent_report() -> tuple[dict, dict[int, dict]]:
    report = read_json(N132_RESULT)
    models = {
        int(row["seed"]): row for row in report.get("models", []) if row.get("arm") == "position"
    }
    if (
        report.get("complete") is not True
        or report.get("training_updates") != 27_648
        or report.get("parameter_updates") != 27_648
        or report.get("train_score_queries") != 2_880
        or report.get("training_sufficient_all_seeds") is not True
        or report.get("dev_queries") != 0
        or report.get("v13_queries") != 0
        or report.get("v14_queries") != 0
        or report.get("confirmation_queries") != 0
        or report.get("can_promote") is not False
        or set(models) != set(SEEDS)
    ):
        raise ValueError("N1-132完整训练/恢复/训练门结案证据不符")
    for seed, row in models.items():
        path = Path(row["final_checkpoint_path"]).resolve()
        expected_path = (n132.CHECKPOINT_ROOT / f"s{seed}_position_final.pt").resolve()
        digest = row.get("final_checkpoint_sha256")
        if (
            path != expected_path
            or digest != EXPECTED_PARENT_HASHES[seed]
            or not path.is_file()
            or sha256(path) != digest
            or row.get("total_updates") != PARENT_UPDATES
            or row.get("final_history_count") != 52_224
            or row.get("training_sufficient") is not True
            or row.get("final_bundle_fingerprint") != EXPECTED_PARENT_FINGERPRINTS[seed]
        ):
            raise ValueError(f"N1-132 seed{seed} frozen position endpoint changed")
    return report, models


def _validate_n138_exclusions() -> tuple[dict, set[str], set[str]]:
    manifest = read_json(N138_MANIFEST)
    face = read_json(N138_FACE)
    face_checks = read_json(N138_FACE_CHECKS)
    result = read_json(N138_RESULT)
    excluded_names = set(manifest.get("excluded_entities", []))
    excluded_words = set(manifest.get("excluded_word_values", []))
    face_names = {
        entity
        for template in face.get("matched_templates", [])
        for entity in template.get("entities", [])
    }
    if (
        manifest.get("schema") != "n1-plan138-neighbor-distance-dose-source-exclusions-v1"
        or manifest.get("entity_count") != 9_010
        or manifest.get("word_value_count") != 6_288
        or len(excluded_names) != 9_010
        or len(excluded_words) != 6_288
        or face_checks.get("accepted") is not True
        or not seal_ast.receipt_binding_matches(face_checks.get("face_sha256"), N138_FACE)
        or face_checks.get("parameter_updates") != 0
        or result.get("complete") is not True
        or result.get("diagnostic_queries") != 672
        or result.get("parameter_updates") != 0
        or result.get("dev_queries") != 0
        or result.get("source_hashes_unchanged") is not True
        or result.get("can_promote") is not False
        or len(face_names) != 344
        or face_names & excluded_names
        or not set(DIRECTIONS) <= excluded_words
    ):
        raise ValueError("N1-138 v20已消费测面及完整来源排除清单校验失败")
    n138_sources = dict(result.get("source_hashes", {}))
    if not n138_sources or not _verify_source_hashes(n138_sources):
        raise ValueError("N1-138结案报告绑定的原始来源哈希已变化")
    return manifest, excluded_names | face_names, excluded_words


def _build_static_context() -> dict:
    _require_plan_identity()
    parent_report, parent_models = _validate_n132_parent_report()
    manifest, excluded_names, excluded_words = _validate_n138_exclusions()
    bound = n132._build_context(load_inputs=True)
    g, inputs = bound["g"], bound["inputs"]
    if (
        inputs is None
        or len(inputs.get("items", [])) != 192
        or any(item.get("split") != "train" for item in inputs["items"])
        or Counter(item.get("kind") for item in inputs["items"])
        != Counter({kind: 48 for kind in TRAIN_KINDS})
    ):
        raise ValueError("PLAN-N1-139必须沿用192项冻结v3 train输入")

    (
        original_schedule,
        records,
        specimens,
        hard_records,
        base_scheduled,
    ) = _collect_original_schedule(g, inputs)
    assignment, assignment_summary = build_context_assignment(records)
    train_entity_names = {entity for item in inputs["items"] for entity in item.get("entities", [])}
    if not train_entity_names <= excluded_names:
        raise ValueError("N1-138消费清单没有覆盖全部冻结v3 train实体名")
    variants, variant_summary = build_context_variants(
        specimens,
        hard_records,
        assignment,
        excluded_names | train_entity_names,
    )
    context_validation = validate_context_schedule(
        records,
        assignment,
        variants,
        excluded_names | train_entity_names,
    )

    source_hashes = dict(bound["source_hashes"])
    _merge_source_hashes(source_hashes, parent_report.get("source_hashes", {}))
    n138_report = read_json(N138_RESULT)
    _merge_source_hashes(source_hashes, n138_report.get("source_hashes", {}))
    for path in (
        PLAN,
        Path(__file__).resolve(),
        ROOT / "tests/taiji_native/test_n1_plan139_context_robust_continuation_contract.py",
        N132_RESULT,
        n132.PLAN,
        Path(n132.__file__).resolve(),
        N138_RESULT,
        N138_MANIFEST,
        N138_FACE,
        N138_FACE_CHECKS,
        v20.PLAN,
        Path(v20.__file__).resolve(),
        Path(v20.__file__).resolve().with_name("eval_taiji_n1_v20_neighbor_distance_dose_curve.py"),
    ):
        _add_source(source_hashes, path)

    n132_parent_hashes = {}
    for _seed, row in parent_models.items():
        parent_path = Path(row["final_checkpoint_path"]).resolve()
        digest = row["final_checkpoint_sha256"]
        _add_source(source_hashes, parent_path, digest)
        n132_parent_hashes[str(parent_path)] = digest
    for path, digest in bound["parent_hashes"].items():
        _add_source(source_hashes, Path(path), digest)

    schedule_digest_rows = [
        (step, row["parent_id"], row["route"], assignment[step]["condition"])
        for step, row in sorted({record["step"]: record for record in records}.items())
    ]
    schedule_sha = hashlib.sha256(
        json.dumps(schedule_digest_rows, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    return {
        "g": g,
        "inputs": inputs,
        "parent_report": parent_report,
        "parent_models": parent_models,
        "parent_hashes": n132_parent_hashes,
        "manifest": manifest,
        "excluded_names": excluded_names | train_entity_names,
        "excluded_words": excluded_words,
        "source_hashes": source_hashes,
        "original_schedule": original_schedule,
        "base_scheduled": base_scheduled,
        "transition_records": records,
        "specimens": specimens,
        "hard_records": hard_records,
        "context_variants": variants,
        "assignment": assignment,
        "assignment_summary": assignment_summary,
        "variant_summary": variant_summary,
        "context_validation": context_validation,
        "schedule_sha256": schedule_sha,
    }


def _outputs_exist(paths: tuple[Path, ...]) -> bool:
    return any(path.exists() for path in paths)


def source_readiness() -> dict:
    _require_plan_identity()
    outputs = (
        READINESS,
        READINESS_ATTEMPT,
        SOURCE_BINDING,
        PREFLIGHT,
        PREFLIGHT_ATTEMPT,
        PREFLIGHT_BINDING,
        ATTEMPT,
        PROGRESS,
        RESULT,
        CHECKPOINT_ROOT,
    )
    if _outputs_exist(outputs):
        raise FileExistsError("PLAN-N1-139是一次性固定批次；已有产物时不得覆盖或重跑")
    write_new(
        READINESS_ATTEMPT,
        {
            "schema": "n1-plan139-context-robust-readiness-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "model_loads": 0,
            "checkpoint_loads": 0,
            "parameter_updates": 0,
            "train_queries": 0,
            "dev_queries": 0,
            "holdout_queries": 0,
            "can_promote": False,
        },
    )
    started = time.process_time()
    bound = None
    errors = []
    try:
        bound = _build_static_context()
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    cpu = time.process_time() - started
    source_hashes = {} if bound is None else bound["source_hashes"]
    unchanged = _verify_source_hashes(source_hashes) if source_hashes else False
    if source_hashes and not unchanged:
        errors.append("来源核验期间至少一个文件哈希发生变化")
    schedule = (
        {}
        if bound is None
        else {
            "base_schedule": bound["original_schedule"],
            "context_assignment": bound["assignment_summary"],
            "schedule_sha256": bound["schedule_sha256"],
        }
    )
    contexts = {} if bound is None else bound["variant_summary"]
    parent_hashes = {} if bound is None else bound["parent_hashes"]
    complete = (
        not errors
        and bound is not None
        and len(parent_hashes) == 3
        and len(bound["assignment"]) == 2_304
        and bound["variant_summary"]["raw_filler_entity_count"] == 2_016
        and bound["variant_summary"]["hard_augmented_slots"] == 1_008
        and bound["context_validation"]["complete"] is True
        and bound["context_validation"]["validated_transition_slots"] == 2_304
        and bound["context_validation"]["validated_augmented_slots"] == 2_016
        and bound["variant_summary"]["unique_new_entity_count"]
        == (
            bound["variant_summary"]["raw_filler_entity_count"]
            + bound["variant_summary"]["hard_filler_entity_count"]
        )
        and unchanged
        and cpu <= CPU_LIMIT_SECONDS
    )
    report = {
        "schema": "n1-plan139-context-robust-readiness-v1",
        "status": "ready_for_checkpoint_preflight" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "approved_draft_sha256": APPROVED_DRAFT_SHA256,
        "parent_checkpoint_hashes": parent_hashes,
        "parent_bundle_fingerprints": EXPECTED_PARENT_FINGERPRINTS,
        "source_hashes": source_hashes,
        "source_hashes_unchanged": unchanged,
        "schedule": schedule,
        "context_variants": contexts,
        "context_validation": {} if bound is None else bound["context_validation"],
        "parent_train_item_count": 192,
        "transition_parent_count": 48,
        "scheduled_transition_slots_per_seed": 2_304,
        "hard_augmented_slot_count": contexts.get("hard_augmented_slots", 0),
        "total_generated_filler_entities": contexts.get("unique_new_entity_count", 0),
        "frozen_gate_queries_per_endpoint": EXPECTED_FROZEN_GATE_QUERIES_PER_ENDPOINT,
        "augmented_context_queries_per_endpoint": EXPECTED_AUGMENTED_QUERIES_PER_ENDPOINT,
        "planned_train_only_queries": EXPECTED_TOTAL_TRAIN_QUERIES,
        "planned_updates": len(SEEDS) * len(BRANCHES) * ADDITIONAL_UPDATES,
        "cpu_seconds": round(cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "parameter_updates": 0,
        "train_queries": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
        "errors": errors,
        "summary_zh": (
            "来源、schedule及2,304个transition上下文slot静态核验通过；未加载模型或checkpoint，"
            "未训练、未查询，可进入zero bundle保存恢复预检。"
            if complete
            else "静态核验未通过；本批次停止，未授权后续训练。"
        ),
    }
    write_new(READINESS, report)
    write_new(
        SOURCE_BINDING,
        {
            "schema": "n1-plan139-context-robust-source-binding-v1",
            "status": "complete" if complete else "stopped",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_hashes": source_hashes,
            "parent_checkpoint_hashes": parent_hashes,
            "context_schedule_sha256": schedule.get("schedule_sha256"),
        },
    )
    return report


def _verify_readiness() -> tuple[dict, dict]:
    if not READINESS.is_file() or not SOURCE_BINDING.is_file():
        raise FileNotFoundError("必须先完成PLAN-N1-139来源与schedule静态核验")
    readiness = read_json(READINESS)
    binding = read_json(SOURCE_BINDING)
    if (
        readiness.get("complete") is not True
        or readiness.get("plan_sha256") != PLAN_SHA256
        or readiness.get("source_hashes_unchanged") is not True
        or readiness.get("model_loads") != 0
        or readiness.get("checkpoint_loads") != 0
        or readiness.get("parameter_updates") != 0
        or readiness.get("train_queries") != 0
        or readiness.get("dev_queries") != 0
        or readiness.get("holdout_queries") != 0
        or readiness.get("planned_updates") != 55_296
        or readiness.get("planned_train_only_queries") != EXPECTED_TOTAL_TRAIN_QUERIES
        or binding.get("status") != "complete"
        or binding.get("plan_sha256") != PLAN_SHA256
        or not seal_ast.receipt_binding_matches(binding.get("readiness_sha256"), READINESS)
        or binding.get("source_hashes") != readiness.get("source_hashes")
        or binding.get("parent_checkpoint_hashes") != readiness.get("parent_checkpoint_hashes")
    ):
        raise ValueError("PLAN-N1-139 readiness/source binding不完整或已改变")
    _require_plan_identity()
    if not _verify_source_hashes(readiness["source_hashes"]):
        raise ValueError("PLAN-N1-139静态来源或父checkpoint哈希已改变")
    return readiness, binding


def _restore_payload(
    payload: dict,
) -> tuple[LocalRelationEvidenceCandidate, torch.optim.Optimizer]:
    if payload.get("schema") != "n1-plan139-context-robust-checkpoint-v1":
        raise ValueError("N1-139 checkpoint schema不符")
    if payload.get("plan_sha256") != PLAN_SHA256 or payload.get("branch") not in BRANCHES:
        raise ValueError("N1-139 checkpoint plan/branch不符")
    if int(payload.get("parent_updates", -1)) != PARENT_UPDATES:
        raise ValueError("N1-139父checkpoint更新数不符")
    parent_path = Path(payload["parent_checkpoint_path"])
    if not parent_path.is_file() or sha256(parent_path) != payload["parent_checkpoint_sha256"]:
        raise ValueError("N1-132冻结父checkpoint哈希改变")
    candidate = copy.deepcopy(payload["candidate"])
    model = LocalRelationEvidenceCandidate.restore_bundle(candidate)
    optimizer = torch.optim.Adam(model.parameters(), lr=n132.FINAL_LR)
    optimizer.load_state_dict(copy.deepcopy(candidate["optimizer"]))
    signature = e11.scorer.paired.fingerprint(model.bundle(optimizer))
    if signature != payload.get("bundle_fingerprint"):
        raise ValueError("N1-139完整模型/Adam/RNG/history恢复指纹不符")
    additional = int(payload.get("additional_updates", -1))
    expected_updates = PARENT_UPDATES + additional
    if additional < 0 or additional > ADDITIONAL_UPDATES:
        raise ValueError("N1-139 checkpoint续训更新数越界")
    expected_phase = (
        "zero" if additional == 0 else "final" if additional == ADDITIONAL_UPDATES else "partial"
    )
    if (
        payload.get("phase") != expected_phase
        or int(payload.get("total_updates", -1)) != expected_updates
        or int(payload.get("history_count", -1)) != 52_224 + additional
    ):
        raise ValueError("N1-139 checkpoint phase/total/history metadata不符")
    if len(optimizer.state) != 41 or any(
        int(optimizer.state[parameter]["step"]) != expected_updates
        for parameter in model.parameters()
    ):
        raise ValueError("N1-139全部41组Adam step计数不符")
    if len(model._observations) != payload.get("history_count"):
        raise ValueError("N1-139观察history计数不符")
    if payload.get("phase") == "zero" and signature != payload.get("parent_bundle_fingerprint"):
        raise ValueError("N1-139 zero bundle必须逐位复现父checkpoint")
    return model, optimizer


def _restore_worker(path: Path) -> dict:
    started = time.process_time()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model, optimizer = _restore_payload(payload)
    signature = e11.scorer.paired.fingerprint(model.bundle(optimizer))
    return {
        "complete": signature == payload["bundle_fingerprint"],
        "signature": signature,
        "bundle_fingerprint": payload["bundle_fingerprint"],
        "parent_bundle_fingerprint": payload["parent_bundle_fingerprint"],
        "seed": payload["seed"],
        "branch": payload["branch"],
        "phase": payload["phase"],
        "additional_updates": payload["additional_updates"],
        "history_count": len(model._observations),
        "optimizer_parameters": len(optimizer.state),
        "cpu_seconds": round(time.process_time() - started, 6),
    }


def _save_checkpoint(
    model,
    optimizer,
    path: Path,
    *,
    seed: int,
    branch: str,
    parent_path: Path,
    parent_hash: str,
    parent_fingerprint: str,
    additional_updates: int,
    schedule_sha256: str,
) -> dict:
    if path.exists():
        raise FileExistsError(f"N1-139 checkpoint write-once路径已存在：{path}")
    if not Path(parent_path).is_file() or sha256(parent_path) != parent_hash:
        raise ValueError("保存前N1-132父checkpoint哈希改变")
    if branch not in BRANCHES:
        raise ValueError("unsupported N1-139 branch")
    candidate = model.bundle(optimizer)
    payload = {
        "schema": "n1-plan139-context-robust-checkpoint-v1",
        "plan_sha256": PLAN_SHA256,
        "seed": int(seed),
        "branch": branch,
        "phase": (
            "zero"
            if additional_updates == 0
            else "final" if additional_updates == ADDITIONAL_UPDATES else "partial"
        ),
        "parent_checkpoint_path": str(Path(parent_path).resolve()),
        "parent_checkpoint_sha256": parent_hash,
        "parent_bundle_fingerprint": parent_fingerprint,
        "parent_updates": PARENT_UPDATES,
        "additional_updates": int(additional_updates),
        "total_updates": PARENT_UPDATES + int(additional_updates),
        "history_count": len(model._observations),
        "context_schedule_sha256": schedule_sha256,
        "objective": {
            "base": "frozen_n1_132_paired_question_and_observation_fidelity",
            "binding_loss": "mean_negative_log_target_fact_row_alignment_mass",
            "binding_weight": n132.BINDING_WEIGHT,
            "epsilon": n132.BINDING_EPSILON,
            "learning_rate": n132.FINAL_LR,
            "trust_region": "n1_120_expanded_normalized_coupled_rwl_adam",
        },
        "candidate": candidate,
    }
    payload["bundle_fingerprint"] = e11.scorer.paired.fingerprint(candidate)
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)
    return payload


def _independent_restore(path: Path, budget: CpuBudget) -> dict:
    budget.guard(reserve=5)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    timeout = min(120.0, max(1.0, budget.limit - budget.used() - 1.0))
    response = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--restore-worker", str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
        env=env,
    )
    result = json.loads(response.stdout)
    budget.children += float(result["cpu_seconds"])
    budget.guard()
    if result.get("complete") is not True or result.get("signature") != result.get(
        "bundle_fingerprint"
    ):
        raise ValueError(f"N1-139独立完整bundle恢复不匹配：{path.name}")
    return {"path": str(path.resolve()), "sha256": sha256(path), **result, "equal": True}


def checkpoint_preflight() -> dict:
    started = time.process_time()
    readiness, binding = _verify_readiness()
    if _outputs_exist(
        (
            PREFLIGHT,
            PREFLIGHT_ATTEMPT,
            PREFLIGHT_BINDING,
            ATTEMPT,
            PROGRESS,
            RESULT,
            CHECKPOINT_ROOT,
        )
    ):
        raise FileExistsError("PLAN-N1-139 checkpoint preflight已开始或已有产物；禁止重试")
    write_new(
        PREFLIGHT_ATTEMPT,
        {
            "schema": "n1-plan139-context-robust-checkpoint-preflight-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_binding_sha256": sha256(SOURCE_BINDING),
            "parent_checkpoint_hashes": readiness["parent_checkpoint_hashes"],
            "planned_zero_bundles": 6,
            "model_loads": 0,
            "parameter_updates": 0,
            "train_queries": 0,
            "dev_queries": 0,
            "can_promote": False,
        },
    )
    setup_cpu = time.process_time() - started
    budget = CpuBudget(
        max(
            0.0,
            CPU_LIMIT_SECONDS - float(readiness["cpu_seconds"]) - setup_cpu,
        )
    )
    zeros = []
    errors = []
    parent_bundle_fingerprints = {}
    try:
        CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=False)
        _require_plan_identity()
        if not _verify_source_hashes(readiness["source_hashes"]):
            raise ValueError(
                "static readiness source changed before zero-copy checkpoint preflight"
            )
        parent_rows = {
            int(row["seed"]): row
            for row in read_json(N132_RESULT)["models"]
            if row.get("arm") == "position"
        }
        for seed in SEEDS:
            budget.guard(reserve=20)
            parent_row = parent_rows[seed]
            parent_path = Path(parent_row["final_checkpoint_path"]).resolve()
            parent_hash = readiness["parent_checkpoint_hashes"][str(parent_path)]
            model, optimizer, payload = n132._load_n132_checkpoint(parent_path)
            parent_fingerprint = e11.scorer.paired.fingerprint(model.bundle(optimizer))
            if (
                parent_fingerprint != EXPECTED_PARENT_FINGERPRINTS[seed]
                or payload.get("bundle_fingerprint") != parent_fingerprint
                or int(payload.get("total_updates", -1)) != PARENT_UPDATES
                or len(model._observations) != 52_224
            ):
                raise ValueError(f"N1-132 seed{seed}加载后的完整bundle与冻结指纹不符")
            parent_bundle_fingerprints[seed] = parent_fingerprint
            for branch in BRANCHES:
                budget.guard(reserve=10)
                zero_path = CHECKPOINT_ROOT / f"s{seed}_{branch}_zero.pt"
                saved = _save_checkpoint(
                    model,
                    optimizer,
                    zero_path,
                    seed=seed,
                    branch=branch,
                    parent_path=parent_path,
                    parent_hash=parent_hash,
                    parent_fingerprint=parent_fingerprint,
                    additional_updates=0,
                    schedule_sha256=binding["context_schedule_sha256"],
                )
                if saved["bundle_fingerprint"] != parent_fingerprint:
                    raise ValueError("zero checkpoint must be an exact full-bundle copy")
                restore = _independent_restore(zero_path, budget)
                zeros.append(
                    {
                        "seed": seed,
                        "branch": branch,
                        "parent_checkpoint_path": str(parent_path),
                        "parent_checkpoint_sha256": parent_hash,
                        "path": str(zero_path.resolve()),
                        "sha256": sha256(zero_path),
                        "bundle_fingerprint": saved["bundle_fingerprint"],
                        "parent_bundle_fingerprint": parent_fingerprint,
                        "history_count": len(model._observations),
                        "independent_restore": restore,
                        "equal": restore["signature"] == parent_fingerprint,
                    }
                )
            del model, optimizer
        for seed in SEEDS:
            pair = [row for row in zeros if row["seed"] == seed]
            if len(pair) != 2 or pair[0]["bundle_fingerprint"] != pair[1]["bundle_fingerprint"]:
                raise ValueError(f"seed{seed}候选/控制zero bundle未逐位匹配")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    unchanged = _verify_source_hashes(readiness["source_hashes"])
    if not unchanged:
        errors.append("zero-copy预检期间来源或父checkpoint哈希发生变化")
    cpu = time.process_time() - started + budget.children
    complete = (
        not errors
        and len(zeros) == 6
        and all(row["equal"] for row in zeros)
        and unchanged
        and cpu + float(readiness["cpu_seconds"]) <= CPU_LIMIT_SECONDS
    )
    zero_checkpoint_hashes = (
        {
            str(path.resolve()): sha256(path)
            for path in CHECKPOINT_ROOT.glob("s*_zero.pt")
            if path.is_file()
        }
        if CHECKPOINT_ROOT.is_dir()
        else {}
    )
    report = {
        "schema": "n1-plan139-context-robust-checkpoint-preflight-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "readiness_sha256": sha256(READINESS),
        "source_binding_sha256": sha256(SOURCE_BINDING),
        "parent_checkpoint_hashes": readiness["parent_checkpoint_hashes"],
        "parent_bundle_fingerprints": parent_bundle_fingerprints,
        "zero_checkpoints": zeros,
        "zero_checkpoint_hashes": zero_checkpoint_hashes,
        "model_loads": len({row["seed"] for row in zeros}),
        "checkpoint_loads": len(zeros) + len(parent_bundle_fingerprints),
        "independent_restores": len(zeros),
        "parameter_updates": 0,
        "train_queries": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "source_hashes_unchanged": unchanged,
        "cpu_seconds": round(cpu, 6),
        "cumulative_cpu_seconds": round(cpu + float(readiness["cpu_seconds"]), 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "errors": errors,
        "summary_zh": (
            "六个zero bundle均与对应N1-132父bundle逐位一致，并经独立进程恢复；"
            "本阶段零参数更新、零模型查询。"
            if complete
            else "zero bundle保存/独立恢复前置未完整通过；本批次停止，不训练。"
        ),
        "can_promote": False,
    }
    write_new(PREFLIGHT, report)
    write_new(
        PREFLIGHT_BINDING,
        {
            "schema": "n1-plan139-context-robust-preflight-binding-v1",
            "status": "complete" if complete else "stopped",
            "plan_sha256": PLAN_SHA256,
            "preflight_sha256": sha256(PREFLIGHT),
            "preflight_attempt_sha256": sha256(PREFLIGHT_ATTEMPT),
            "readiness_sha256": sha256(READINESS),
            "zero_checkpoint_hashes": report["zero_checkpoint_hashes"],
            "parent_checkpoint_hashes": readiness["parent_checkpoint_hashes"],
        },
    )
    return report


def _verify_preflight() -> tuple[dict, dict, dict, dict]:
    readiness, source_binding = _verify_readiness()
    if not all(path.is_file() for path in (PREFLIGHT, PREFLIGHT_ATTEMPT, PREFLIGHT_BINDING)):
        raise FileNotFoundError("六个zero bundle的独立保存/恢复预检尚未完成")
    preflight = read_json(PREFLIGHT)
    attempt = read_json(PREFLIGHT_ATTEMPT)
    binding = read_json(PREFLIGHT_BINDING)
    if (
        preflight.get("complete") is not True
        or preflight.get("plan_sha256") != PLAN_SHA256
        or preflight.get("independent_restores") != 6
        or preflight.get("parameter_updates") != 0
        or preflight.get("train_queries") != 0
        or preflight.get("source_hashes_unchanged") is not True
        or attempt.get("status") != "started"
        or attempt.get("plan_sha256") != PLAN_SHA256
        or binding.get("status") != "complete"
        or binding.get("plan_sha256") != PLAN_SHA256
        or not seal_ast.receipt_binding_matches(binding.get("preflight_sha256"), PREFLIGHT)
        or not seal_ast.receipt_binding_matches(
            binding.get("preflight_attempt_sha256"), PREFLIGHT_ATTEMPT
        )
        or binding.get("zero_checkpoint_hashes") != preflight.get("zero_checkpoint_hashes")
        or binding.get("parent_checkpoint_hashes") != readiness.get("parent_checkpoint_hashes")
    ):
        raise ValueError("PLAN-N1-139 zero-copy preflight缺失或绑定失效")
    checked = {
        **readiness["source_hashes"],
        **readiness["parent_checkpoint_hashes"],
        **binding["zero_checkpoint_hashes"],
        str(PREFLIGHT.resolve()): sha256(PREFLIGHT),
        str(PREFLIGHT_ATTEMPT.resolve()): sha256(PREFLIGHT_ATTEMPT),
        str(PREFLIGHT_BINDING.resolve()): sha256(PREFLIGHT_BINDING),
    }
    if not _verify_source_hashes(checked):
        raise ValueError("PLAN-N1-139执行前来源/zero checkpoint哈希已变化")
    expected_zero_paths = {
        str((CHECKPOINT_ROOT / f"s{seed}_{branch}_zero.pt").resolve())
        for seed in SEEDS
        for branch in BRANCHES
    }
    if set(binding["zero_checkpoint_hashes"]) != expected_zero_paths:
        raise ValueError("N1-139 zero checkpoint文件集合不等于冻结的六个目标")
    return readiness, source_binding, preflight, binding


def _branch_scheduled(g: dict, bound: dict, branch: str, context_ledger: Counter):
    original = g["scheduled"]
    assignment = bound["assignment"]
    variants = bound["context_variants"]

    def scheduled(step: int, items: list[dict], curriculum: dict):
        item, second = original(step, items, curriculum)
        if item.get("kind") != "transition":
            return item, second
        parent, route = _scheduled_identity(item)
        if branch == "control":
            if len(item.get("observations", [])) != 2:
                raise ValueError("控制臂transition必须始终使用原始两行上下文")
            context_ledger[(parent, route, "control_2row")] += 1
            return item, second

        spec = assignment.get(int(step))
        if spec is None or spec["parent_id"] != parent or spec["route"] != route:
            raise ValueError("候选context schedule与冻结slot assignment不匹配")
        condition = spec["condition"]
        context_ledger[(parent, route, condition)] += 1
        if condition == BASELINE_CONDITION:
            if len(item.get("observations", [])) != 2:
                raise ValueError("候选两行baseline slot发生改变")
            return item, second
        if route == "raw":
            variant = variants[(parent, "raw")]
        else:
            subject = n132._query_subject(item["question"])
            variant = variants[(parent, "hard", subject)]
        changed = _apply_context(item, variant, condition)
        if second is not None:
            if second.get("observations") != item["observations"]:
                raise ValueError("transition paired question观察输入不一致")
            second = copy.deepcopy(second)
            second["observations"] = list(changed["observations"])
            second["entities"] = list(changed["entities"])
        return changed, second

    return scheduled


def _context_ledger_summary(ledger: Counter, branch: str) -> dict:
    expected = Counter()
    if branch == "candidate":
        expected[("raw", BASELINE_CONDITION)] = 144
        expected[("hard", BASELINE_CONDITION)] = 144
        for condition, _length, _distance in TRANSITION_CONDITIONS:
            expected[("raw", condition)] = 168
            expected[("hard", condition)] = 168
    else:
        expected[("raw", "control_2row")] = 1_152
        expected[("hard", "control_2row")] = 1_152
    actual = Counter()
    for (_parent, route, condition), count in ledger.items():
        actual[(route, condition)] += int(count)
    if actual != expected:
        raise ValueError(f"{branch}训练上下文slot读数不符：{dict(actual)}")
    return {
        f"{route}/{condition}": int(actual[(route, condition)])
        for route, condition in sorted(actual)
    }


def _score_frozen_gate(
    g: dict,
    model,
    optimizer,
    seed: int,
    inputs: dict,
    budget: CpuBudget,
    partial: dict,
):
    before = e11.scorer.paired.fingerprint(model.bundle(optimizer))
    evidence = {"rows": [], "score_queries": 0}
    try:
        rows, summary = g["evaluate"](
            model,
            optimizer,
            seed,
            n132.ARM,
            TOTAL_UPDATES,
            inputs["views"],
            budget,
            evidence,
        )
    finally:
        partial["frozen_gate_queries"] = evidence["score_queries"]
        partial["frozen_gate_rows"] = evidence["rows"]
    if evidence["score_queries"] != EXPECTED_FROZEN_GATE_QUERIES_PER_ENDPOINT:
        raise ValueError("N1-132冻结train gate必须计分960条/endpoint")
    if e11.scorer.paired.fingerprint(model.bundle(optimizer)) != before:
        raise ValueError("N1-132冻结充分性计分改变模型/Adam/RNG/history")
    return rows, summary, bool(g["calibration"].sufficient(summary)), evidence["score_queries"]


def _score_augmented_contexts(
    g: dict,
    model,
    optimizer,
    seed: int,
    branch: str,
    bound: dict,
    budget: CpuBudget,
    partial: dict,
) -> list[dict]:
    before = e11.scorer.paired.fingerprint(model.bundle(optimizer))
    core = g["runner"].core
    rows = []
    partial["augmented_context_queries"] = 0
    partial["augmented_context_rows"] = rows
    parents = sorted({parent for parent, _route in bound["specimens"]})
    for parent in parents:
        source_item = bound["specimens"][(parent, "raw")]
        variant = bound["context_variants"][(parent, "raw")]
        for condition, length, distance in TRANSITION_CONDITIONS:
            budget.guard(reserve=5)
            item_id = f"n139-{parent}-{length}-d{distance}"
            item = _apply_context(source_item, variant, condition, item_id=item_id)
            session = f"n139-score-{seed}-{branch}-{item_id}"
            if session in model._observations:
                raise ValueError(f"N1-139 train-only scoring session collision: {session}")
            partial["augmented_context_queries"] += 1
            try:
                row = core.measure(model, item, n132.ARM, session)
            finally:
                model._observations.pop(session, None)
            rows.append(
                {
                    **row,
                    "seed": seed,
                    "branch": branch,
                    "parent_id": parent,
                    "context_length": length,
                    "neighbor_distance": distance,
                    "condition": condition,
                    "answer_direction": source_item["answer"],
                    "target_fact_indices": [0, 1],
                    "split": "train",
                }
            )
            partial["augmented_context_rows"] = rows
    if len(rows) != EXPECTED_AUGMENTED_QUERIES_PER_ENDPOINT:
        raise ValueError("N1-139 augmented context score denominator must be 288/endpoint")
    if e11.scorer.paired.fingerprint(model.bundle(optimizer)) != before:
        raise ValueError("augmented train-only scoring changed model/Adam/RNG/history")
    return rows


def _context_score_summary(rows: list[dict]) -> dict:
    summary = {}
    for condition, length, distance in TRANSITION_CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        if len(selected) != 48:
            raise ValueError(f"上下文条件{condition}必须有48个父项")
        summary[condition] = {
            "items": len(selected),
            "free_exact": sum(row.get("free_exact") is True for row in selected),
            "errors": sum(row.get("free_exact") is not True for row in selected),
            "context_length": length,
            "neighbor_distance": distance,
            "by_answer_direction": {
                direction: {
                    "items": sum(row["answer_direction"] == direction for row in selected),
                    "free_exact": sum(
                        row["answer_direction"] == direction and row.get("free_exact") is True
                        for row in selected
                    ),
                }
                for direction in DIRECTIONS
            },
        }
    return summary


def run_training() -> dict:
    stage_started = time.process_time()
    readiness, _source_binding, preflight, preflight_binding = _verify_preflight()
    if _outputs_exist((ATTEMPT, PROGRESS, RESULT)):
        raise FileExistsError("PLAN-N1-139固定续训/评估阶段一次性，禁止重启")
    bound = _build_static_context()
    g, inputs = bound["g"], bound["inputs"]
    if (
        bound["schedule_sha256"] != readiness["schedule"]["schedule_sha256"]
        or bound["variant_summary"] != readiness["context_variants"]
    ):
        raise ValueError("PLAN-N1-139重建的context/schedule与来源readiness不一致")
    source_hashes = {
        **readiness["source_hashes"],
        str(READINESS.resolve()): sha256(READINESS),
        str(SOURCE_BINDING.resolve()): sha256(SOURCE_BINDING),
        str(PREFLIGHT.resolve()): sha256(PREFLIGHT),
        str(PREFLIGHT_ATTEMPT.resolve()): sha256(PREFLIGHT_ATTEMPT),
        str(PREFLIGHT_BINDING.resolve()): sha256(PREFLIGHT_BINDING),
    }
    source_hashes.update(readiness["parent_checkpoint_hashes"])
    source_hashes.update(preflight_binding["zero_checkpoint_hashes"])
    if not _verify_source_hashes(source_hashes):
        raise ValueError("固定续训开始前有来源/checkpoint哈希变化")
    setup_cpu = time.process_time() - stage_started
    prior_cpu = float(readiness["cpu_seconds"]) + float(preflight["cpu_seconds"]) + setup_cpu
    remaining = CPU_LIMIT_SECONDS - prior_cpu
    if remaining <= 0:
        raise TimeoutError("PLAN-N1-139累计CPU额度在续训开始前已耗尽")
    budget = CpuBudget(remaining)
    execution_started = time.process_time()
    write_new(
        ATTEMPT,
        {
            "schema": "n1-plan139-context-robust-training-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_binding_sha256": sha256(SOURCE_BINDING),
            "checkpoint_preflight_sha256": sha256(PREFLIGHT),
            "checkpoint_preflight_binding_sha256": sha256(PREFLIGHT_BINDING),
            "source_hashes": source_hashes,
            "parent_checkpoint_hashes": readiness["parent_checkpoint_hashes"],
            "zero_checkpoint_hashes": preflight_binding["zero_checkpoint_hashes"],
            "seed_order": list(SEEDS),
            "branch_order": list(BRANCHES),
            "additional_updates_per_model": ADDITIONAL_UPDATES,
            "planned_parameter_updates": 55_296,
            "planned_train_only_queries": EXPECTED_TOTAL_TRAIN_QUERIES,
            "dev_queries": 0,
            "holdout_queries": 0,
            "confirmation_queries": 0,
            "can_promote": False,
            "说明_zh": "固定6分支批次；不访问dev/holdout，不改变原门，不晋升模型。",
        },
    )

    records = []
    updates_completed = 0
    train_score_queries = 0
    model = optimizer = None
    active = None
    errors = []
    partial_score_receipt = None
    try:
        with PROGRESS.open("x", encoding="utf-8", newline="\n") as progress:
            for seed in SEEDS:
                for branch in BRANCHES:
                    budget.guard(reserve=45)
                    zero_path = CHECKPOINT_ROOT / f"s{seed}_{branch}_zero.pt"
                    zero_payload = torch.load(zero_path, map_location="cpu", weights_only=True)
                    model, optimizer = _restore_payload(zero_payload)
                    parent_fingerprint = zero_payload["parent_bundle_fingerprint"]
                    before = e11.scorer.paired.fingerprint(model.bundle(optimizer))
                    if before != parent_fingerprint:
                        raise ValueError(
                            "zero bundle loaded for continuation is not the parent endpoint"
                        )

                    active = {
                        "seed": seed,
                        "branch": branch,
                        "completed_updates": 0,
                        "model": model,
                        "optimizer": optimizer,
                        "parent_path": zero_payload["parent_checkpoint_path"],
                        "parent_hash": zero_payload["parent_checkpoint_sha256"],
                        "parent_fingerprint": parent_fingerprint,
                        "final_saved": False,
                    }
                    initial_history = len(model._observations)
                    if initial_history != 52_224:
                        raise ValueError("N1-139 continuation must start at N1-132 history 52,224")

                    g["length_ledger"].clear()
                    context_ledger = Counter()
                    g["scheduled"] = _branch_scheduled(g, bound, branch, context_ledger)
                    step_metrics = []
                    try:
                        for step in range(ADDITIONAL_UPDATES):
                            budget.guard(reserve=30)
                            item, second = g["scheduled"](
                                step, inputs["items"], inputs["curriculum"]
                            )
                            context = None
                            if item.get("kind") == "transition":
                                spec = bound["assignment"][step]
                                context = (
                                    spec["condition"] if branch == "candidate" else "control_2row"
                                )
                            metrics = n132._training_step(
                                model,
                                optimizer,
                                item,
                                second,
                                f"n1-plan139-{seed}-{branch}-{step}",
                                g,
                            )
                            step_metrics.append(metrics)
                            active["completed_updates"] = step + 1
                            updates_completed += 1
                            progress.write(
                                json.dumps(
                                    {
                                        "seed": seed,
                                        "branch": branch,
                                        "step": step,
                                        "context_condition": context,
                                        **metrics,
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                            if (step + 1) % 64 == 0:
                                progress.flush()
                            if (step + 1) % 512 == 0:
                                progress.flush()
                                print(
                                    f"plan139 seed={seed} branch={branch} "
                                    f"step={step + 1}/{ADDITIONAL_UPDATES} "
                                    f"qa={metrics['qa']:.4f} obs={metrics['observation']:.4f} "
                                    f"bind={metrics['binding']:.4f} cpu={budget.used():.1f}",
                                    flush=True,
                                )
                    finally:
                        g["scheduled"] = bound["base_scheduled"]

                    ledger_summary = _context_ledger_summary(context_ledger, branch)
                    total_steps = {
                        int(optimizer.state[parameter]["step"]) for parameter in model.parameters()
                    }
                    if (
                        total_steps != {TOTAL_UPDATES}
                        or len(model._observations) != initial_history + ADDITIONAL_UPDATES
                        or len(step_metrics) != ADDITIONAL_UPDATES
                        or sum(g["length_ledger"].values()) != ADDITIONAL_UPDATES
                    ):
                        raise ValueError("N1-139固定更新数、Adam或history末状态不符")
                    if e11.scorer.paired.fingerprint(model.bundle(optimizer)) == before:
                        raise ValueError("N1-139训练结束后模型bundle与父checkpoint意外相同")
                    length_ledger = {
                        "/".join(str(part) for part in key): int(value)
                        for key, value in sorted(g["length_ledger"].items())
                    }
                    final_path = CHECKPOINT_ROOT / f"s{seed}_{branch}_final.pt"
                    final_payload = _save_checkpoint(
                        model,
                        optimizer,
                        final_path,
                        seed=seed,
                        branch=branch,
                        parent_path=Path(active["parent_path"]),
                        parent_hash=active["parent_hash"],
                        parent_fingerprint=parent_fingerprint,
                        additional_updates=ADDITIONAL_UPDATES,
                        schedule_sha256=readiness["schedule"]["schedule_sha256"],
                    )
                    active["final_saved"] = True
                    final_restore = _independent_restore(final_path, budget)
                    means = {
                        name: sum(row[name] for row in step_metrics) / len(step_metrics)
                        for name in ("total", "base_total", "qa", "observation", "binding")
                    }
                    records.append(
                        {
                            "seed": seed,
                            "branch": branch,
                            "parent_checkpoint_path": active["parent_path"],
                            "parent_checkpoint_sha256": active["parent_hash"],
                            "zero_checkpoint_path": str(zero_path.resolve()),
                            "zero_checkpoint_sha256": sha256(zero_path),
                            "initial_bundle_fingerprint": before,
                            "additional_updates": ADDITIONAL_UPDATES,
                            "total_updates": TOTAL_UPDATES,
                            "initial_history_count": initial_history,
                            "final_history_count": len(model._observations),
                            "optimizer_step_values": sorted(total_steps),
                            "length_balance_ledger": length_ledger,
                            "context_slot_ledger": ledger_summary,
                            "loss_means": means,
                            "binding_supervised_query_terms": sum(
                                row["binding_queries"] for row in step_metrics
                            ),
                            "loss_step_count": len(step_metrics),
                            "final_checkpoint_path": str(final_path.resolve()),
                            "final_checkpoint_sha256": sha256(final_path),
                            "final_bundle_fingerprint": final_payload["bundle_fingerprint"],
                            "final_independent_restore": final_restore,
                        }
                    )
                    active = None
                    del model, optimizer
                    model = optimizer = None

        if len(records) != 6 or updates_completed != 55_296:
            raise ValueError("N1-139须先完整训练并恢复六个final endpoint，再开始计分")

        # 六个分支全部训练并恢复后，才执行冻结960行门与288个增强train视图。
        for seed in SEEDS:
            for branch in BRANCHES:
                budget.guard(reserve=30)
                record = next(
                    row for row in records if row["seed"] == seed and row["branch"] == branch
                )
                final_path = Path(record["final_checkpoint_path"])
                final_payload = torch.load(final_path, map_location="cpu", weights_only=True)
                model, optimizer = _restore_payload(final_payload)
                partial_score_receipt = {
                    "seed": seed,
                    "branch": branch,
                    "frozen_gate_queries": 0,
                    "frozen_gate_rows": [],
                    "augmented_context_queries": 0,
                    "augmented_context_rows": [],
                }
                original_rows, gate_summary, sufficient, gate_queries = _score_frozen_gate(
                    g, model, optimizer, seed, inputs, budget, partial_score_receipt
                )
                context_rows = _score_augmented_contexts(
                    g,
                    model,
                    optimizer,
                    seed,
                    branch,
                    bound,
                    budget,
                    partial_score_receipt,
                )
                queries = gate_queries + len(context_rows)
                train_score_queries += queries
                record.update(
                    {
                        "train_sufficiency": gate_summary,
                        "training_sufficient": sufficient,
                        "frozen_gate_queries": gate_queries,
                        "frozen_gate_rows": original_rows,
                        "augmented_context_queries": len(context_rows),
                        "augmented_context_summary": _context_score_summary(context_rows),
                        "augmented_context_rows": context_rows,
                        "train_score_queries": queries,
                    }
                )
                partial_score_receipt = None
                del model, optimizer
                model = optimizer = None
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        if partial_score_receipt is not None:
            train_score_queries += int(partial_score_receipt["frozen_gate_queries"])
            train_score_queries += int(partial_score_receipt["augmented_context_queries"])
    finally:
        if (
            active is not None
            and model is not None
            and optimizer is not None
            and not active["final_saved"]
        ):
            try:
                partial_path = CHECKPOINT_ROOT / f"s{active['seed']}_{active['branch']}_partial.pt"
                partial_payload = _save_checkpoint(
                    model,
                    optimizer,
                    partial_path,
                    seed=active["seed"],
                    branch=active["branch"],
                    parent_path=Path(active["parent_path"]),
                    parent_hash=active["parent_hash"],
                    parent_fingerprint=active["parent_fingerprint"],
                    additional_updates=active["completed_updates"],
                    schedule_sha256=readiness["schedule"]["schedule_sha256"],
                )
                active["partial_checkpoint"] = {
                    "path": str(partial_path.resolve()),
                    "sha256": sha256(partial_path),
                    "additional_updates": active["completed_updates"],
                    "bundle_fingerprint": partial_payload["bundle_fingerprint"],
                    "automatic_resume_allowed": False,
                }
            except BaseException as save_error:
                errors.append(f"partial checkpoint save {type(save_error).__name__}: {save_error}")

    unchanged = _verify_source_hashes(source_hashes)
    if not unchanged:
        errors.append("N1-139运行期间至少一个冻结来源/父checkpoint/zero bundle哈希改变")
    checkpoint_hashes = {
        str(path.resolve()): sha256(path) for path in CHECKPOINT_ROOT.glob("*.pt") if path.is_file()
    }
    stage_cpu = setup_cpu + time.process_time() - execution_started + budget.children
    cumulative_cpu = float(readiness["cpu_seconds"]) + float(preflight["cpu_seconds"]) + stage_cpu
    endpoint_records = [row for row in records if "train_sufficiency" in row]
    sufficient_all = len(endpoint_records) == 6 and all(
        row["training_sufficient"] for row in endpoint_records
    )
    scored_queries = sum(int(row.get("train_score_queries", 0)) for row in endpoint_records)
    scored_rows = sum(
        len(row.get("frozen_gate_rows", [])) + len(row.get("augmented_context_rows", []))
        for row in endpoint_records
    )
    complete = (
        not errors
        and updates_completed == 55_296
        and len(records) == 6
        and len(endpoint_records) == 6
        and train_score_queries == EXPECTED_TOTAL_TRAIN_QUERIES
        and scored_queries == EXPECTED_TOTAL_TRAIN_QUERIES
        and scored_rows == EXPECTED_TOTAL_TRAIN_QUERIES
        and len(checkpoint_hashes) == 12
        and all(row.get("final_independent_restore", {}).get("equal") is True for row in records)
        and unchanged
        and cumulative_cpu <= CPU_LIMIT_SECONDS
    )
    report = {
        "schema": "n1-plan139-context-robust-matched-continuation-calibration-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "approved_draft_sha256": APPROVED_DRAFT_SHA256,
        "readiness_sha256": sha256(READINESS),
        "source_binding_sha256": sha256(SOURCE_BINDING),
        "checkpoint_preflight_sha256": sha256(PREFLIGHT),
        "checkpoint_preflight_binding_sha256": sha256(PREFLIGHT_BINDING),
        "attempt_sha256": sha256(ATTEMPT),
        "progress_sha256": sha256(PROGRESS) if PROGRESS.is_file() else None,
        "source_hashes": source_hashes,
        "source_hashes_unchanged": unchanged,
        "parent_checkpoint_hashes": readiness["parent_checkpoint_hashes"],
        "new_checkpoint_hashes": checkpoint_hashes,
        "models": records,
        "partial_checkpoint": None if active is None else active.get("partial_checkpoint"),
        "partial_score_receipt": partial_score_receipt,
        "training_updates": updates_completed,
        "train_score_queries": train_score_queries,
        "frozen_gate_queries_per_endpoint": EXPECTED_FROZEN_GATE_QUERIES_PER_ENDPOINT,
        "augmented_context_queries_per_endpoint": EXPECTED_AUGMENTED_QUERIES_PER_ENDPOINT,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "parameter_updates": updates_completed,
        "optimizer_steps": updates_completed,
        "cpu_seconds": round(stage_cpu, 6),
        "cumulative_cpu_seconds": round(cumulative_cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "training_sufficient_all_endpoints": sufficient_all,
        "can_promote": False,
        "errors": errors,
        "summary_zh": (
            f"批次{'完整结束' if complete else '停止'}；"
            f"完成{updates_completed}/55,296次更新、{train_score_queries}/7,488条train-only计分；"
            f"六个endpoint原门全通过={sufficient_all}；"
            f"累计CPU {cumulative_cpu:.3f}/3,600秒；未访问dev/holdout，can_promote=false。"
        ),
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--checkpoint-preflight", action="store_true")
    modes.add_argument("--execute-approved", action="store_true")
    modes.add_argument("--restore-worker", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.restore_worker is not None:
        result = _restore_worker(args.restore_worker)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result.get("complete") else 1
    if args.validate_only:
        result = source_readiness()
    elif args.checkpoint_preflight:
        result = checkpoint_preflight()
    else:
        result = run_training()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "complete",
                    "status",
                    "cpu_seconds",
                    "cumulative_cpu_seconds",
                    "planned_updates",
                    "training_updates",
                    "planned_train_only_queries",
                    "train_score_queries",
                    "independent_restores",
                    "training_sufficient_all_endpoints",
                    "dev_queries",
                    "holdout_queries",
                    "confirmation_queries",
                    "can_promote",
                    "errors",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
