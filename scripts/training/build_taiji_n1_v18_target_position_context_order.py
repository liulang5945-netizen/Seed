"""为已批准的PLAN-N1-136建立seed118固定长度目标位置诊断面。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import (  # noqa: E402
    build_taiji_n1_hard_neighbor_v9 as v9,
)
from scripts.training import (  # noqa: E402
    build_taiji_n1_v14_after_n123 as v14_builder,
)
from scripts.training import (  # noqa: E402
    build_taiji_n1_v15_after_n132 as n133_builder,
)
from scripts.training import (  # noqa: E402
    build_taiji_n1_v17_matched_context_length as n135_builder,
)
from scripts.training import seal_ast  # noqa: E402

PLAN = (
    ROOT / "plans/reference/PLAN-N1-136_target_position_context_order_diagnostic_draft_20261019.md"
)
PLAN_SHA256 = "bed528c5fe76ccc865b73724e52a807cf7e9e2b8c66883faf26e9cb3ee3e8f2f"
APPROVED_DRAFT_SHA256 = "a6f7252c3c4075bce1064c05a633f7b38ea53b346b6cfee54d210ecc7d8d6e99"

N135_PLAN = n135_builder.PLAN
N135_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-135_execution_closeout_20261019.md"
N135_SOURCE_EXCLUSIONS = n135_builder.SOURCE_EXCLUSIONS
N135_READINESS = n135_builder.READINESS
N135_FACE = n135_builder.FACE
N135_FACE_ATTEMPT = n135_builder.FACE_ATTEMPT
N135_FACE_CHECKS = n135_builder.FACE_CHECKS
N135_RUN_REPORT = n135_builder.RUN_REPORT
N135_RUN_ATTEMPT = n135_builder.RUN_ATTEMPT

SOURCE_EXCLUSIONS = (
    ROOT
    / "plans/manifests/n1_hard_neighbor_v18_target_position_seed118_source_exclusions_20261019.json"
)
READINESS = ROOT / "reports/taiji_n1_plan136_target_position_readiness_20261019.json"
READINESS_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan136_target_position_readiness_attempt_20261019.json"
)
FACE = (
    ROOT
    / "plans/manifests/n1_hard_neighbor_v18_target_position_context_order_seed118_20261019.json"
)
FACE_ATTEMPT = FACE.with_name(FACE.stem + "_attempt.json")
FACE_CHECKS = ROOT / "reports/taiji_n1_plan136_target_position_face_checks_20261019.json"
RUN_REPORT = (
    ROOT / "reports/taiji_n1_plan136_target_position_context_order_diagnostic_20261019.json"
)
RUN_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan136_target_position_context_order_diagnostic_attempt_20261019.json"
)
TEST = ROOT / "tests/taiji_native/test_n1_plan136_target_position_context_order_contract.py"

SEED = 118
TEMPLATE_COUNT = 8
TEMPLATES_PER_DIRECTION = 2
LENGTHS = (2, 8, 16)
ITEMS_PER_TEMPLATE = 5
ITEMS = TEMPLATE_COUNT * ITEMS_PER_TEMPLATE
DISTANCE = 1
ENTITY_LENGTH = v14_builder.ENTITY_LENGTH
DIRECTIONS = tuple(v14_builder.DIRECTIONS)
TURN_WORDS = ("left", "right")
CPU_LIMIT_SECONDS = 600.0
EXPECTED_EXCLUDED_ENTITIES = 8658
EXPECTED_EXCLUDED_WORDS = 6288

WRITE_ONCE_OUTPUTS = (
    SOURCE_EXCLUSIONS,
    READINESS,
    READINESS_ATTEMPT,
    FACE,
    FACE_ATTEMPT,
    FACE_CHECKS,
    RUN_REPORT,
    RUN_ATTEMPT,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _add_source(source_hashes: dict[str, str], path: Path, expected: str | None = None) -> None:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"冻结来源缺失：{path}")
    actual = sha256(path)
    if expected is not None and not seal_ast.recorded_value_attested(path, expected):
        raise ValueError(f"冻结来源哈希改变：{path}")
    prior = source_hashes.get(str(path))
    if prior is not None and prior != actual and not seal_ast.recorded_value_attested(path, prior):
        raise ValueError(f"来源哈希冲突：{path}")
    source_hashes[str(path)] = actual


def _verify_source_hashes(source_hashes: dict[str, str]) -> None:
    for raw_path, expected in source_hashes.items():
        path = Path(raw_path)
        if not path.is_file() or not seal_ast.recorded_value_attested(path, expected):
            raise ValueError(f"N1-136冻结来源哈希改变：{path}")


def _prior_sources() -> tuple[set[str], set[str], dict[str, str], dict, list[dict]]:
    if sha256(PLAN) != PLAN_SHA256:
        raise ValueError("PLAN-N1-136已批准执行版本SHA-256不匹配")
    if sha256(PLAN) == APPROVED_DRAFT_SHA256:
        raise ValueError("PLAN-N1-136尚未记录用户批准状态")

    readiness = n135_builder._verify_readiness()
    prior_manifest = read_json(N135_SOURCE_EXCLUSIONS)
    prior_face = read_json(N135_FACE)
    prior_face_checks = read_json(N135_FACE_CHECKS)
    prior_report = read_json(N135_RUN_REPORT)
    if (
        prior_manifest.get("entity_count") != 8538
        or prior_manifest.get("word_value_count") != 6288
        or len(prior_manifest.get("excluded_entities", [])) != 8538
        or len(prior_manifest.get("excluded_word_values", [])) != 6288
        or prior_face.get("schema") != "n1-transition-matched-context-length-diagnostic-v1"
        or prior_face.get("seed") != 117
        or prior_face.get("items") != 24
        or len(prior_face.get("episodes", [])) != 24
        or prior_face_checks.get("accepted") is not True
        or not seal_ast.receipt_binding_matches(prior_face_checks.get("face_sha256"), N135_FACE)
        or prior_report.get("complete") is not True
        or prior_report.get("diagnostic_queries") != 288
        or prior_report.get("endpoint_count") != 12
        or prior_report.get("parameter_updates") != 0
        or prior_report.get("dev_queries") != 0
        or prior_report.get("source_hashes_unchanged") is not True
        or prior_report.get("can_promote") is not False
    ):
        raise ValueError("N1-135结案报告、seed117面或逐题查询证据与冻结状态不符")

    excluded_names = set(prior_manifest["excluded_entities"])
    excluded_words = set(prior_manifest["excluded_word_values"])
    face_names, face_materials, face_answers = v14_builder._face_entities_and_words(prior_face)
    if (
        len(face_names) != 120
        or face_materials
        or face_answers != set(DIRECTIONS)
        or excluded_names & face_names
        or not face_answers <= excluded_words
    ):
        raise ValueError("v17实体、答案值或N1-135来源排除关系不符")
    excluded_names.update(face_names)
    if (
        len(excluded_names) != EXPECTED_EXCLUDED_ENTITIES
        or len(excluded_words) != EXPECTED_EXCLUDED_WORDS
    ):
        raise ValueError(
            "N1-136来源排除并集改变："
            f"entities={len(excluded_names)}, words={len(excluded_words)}"
        )

    source_hashes = dict(prior_report.get("source_hashes", {}))
    for path in (
        N135_SOURCE_EXCLUSIONS,
        N135_READINESS,
        N135_FACE,
        N135_FACE_ATTEMPT,
        N135_FACE_CHECKS,
        N135_RUN_REPORT,
        N135_RUN_ATTEMPT,
        N135_PLAN,
        N135_CLOSEOUT,
        PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/training/eval_taiji_n1_v17_matched_context_length.py",
        TEST,
    ):
        _add_source(source_hashes, path)

    endpoints = n133_builder._endpoint_specs()
    restored_by_key = {
        (row.get("source"), row.get("seed"), row.get("arm")): row
        for row in readiness.get("endpoint_restores", [])
    }
    if len(endpoints) != 12 or len(restored_by_key) != 12:
        raise ValueError("PLAN-N1-136必须绑定N1-135的12个冻结endpoint")
    for endpoint in endpoints:
        key = (endpoint["source"], endpoint["seed"], endpoint["arm"])
        restored = restored_by_key.get(key)
        if (
            restored is None
            or restored.get("restore_matches") is not True
            or restored.get("sha256") != endpoint["sha256"]
            or Path(restored.get("path", "")).resolve() != Path(endpoint["path"]).resolve()
        ):
            raise ValueError(f"N1-135完整bundle回执与冻结endpoint不符：{key}")
        _add_source(source_hashes, Path(endpoint["path"]), endpoint["sha256"])

    for ledger_path in sorted((ROOT / "plans/manifests").glob("n1_hard_neighbor_*.json")):
        ledger = read_json(ledger_path)
        _add_source(source_hashes, ledger_path)
        if ledger.get("seed") == SEED:
            raise ValueError(f"seed118已在硬邻居账本中使用：{ledger_path}")
    _verify_source_hashes(source_hashes)
    return (
        excluded_names,
        excluded_words,
        source_hashes,
        {
            "n135_excluded_entities": 8538,
            "n135_excluded_words": 6288,
            "v17_face_entities": len(face_names),
            "v17_face_answers": len(face_answers),
            "excluded_entities": len(excluded_names),
            "excluded_words": len(excluded_words),
            "source_hash_count": len(source_hashes),
            "endpoint_count": len(endpoints),
        },
        endpoints,
    )


def _make_episode(
    *,
    ordinal: int,
    template: dict,
    context_length: int,
    placement: str,
) -> dict:
    target_rows = template["target_rows"]
    filler_count = context_length - len(target_rows)
    fillers = template["filler_rows"][:filler_count]
    if placement == "baseline":
        if context_length != 2:
            raise ValueError("baseline只允许使用2行上下文")
        observations = list(target_rows)
        target_start = 0
    elif placement == "early":
        observations = [*target_rows, *fillers]
        target_start = 0
    elif placement == "late":
        observations = [*fillers, *target_rows]
        target_start = context_length - len(target_rows)
    else:
        raise ValueError(f"未知目标位置条件：{placement}")
    return {
        "id": f"v18-trans-{ordinal:03d}",
        "split": "holdout",
        "subset": "diagnostic",
        "kind": "transition",
        "kind_ordinal": ordinal,
        "template_id": template["template_id"],
        "context_length": context_length,
        "placement": placement,
        "answer_direction": template["answer_direction"],
        "entities": [template["subject"], *template["distractor_entities"][:filler_count]],
        "observations": observations,
        "question": template["question"],
        "answer": template["answer_direction"],
        "target_fact_indices": [target_start, target_start + 1],
        "target_start_index": target_start,
        "start_direction": template["start_direction"],
        "turn_word": template["turn_word"],
        "neighbor_distance": DISTANCE,
    }


def build_v18_face(excluded_names: set[str], excluded_words: set[str]) -> dict:
    """构造8模板×(2行基线+8/16行前置/后置)的40题面。"""
    if not set(DIRECTIONS) <= excluded_words:
        raise ValueError("四个方向答案必须已包含在完整来源排除并集中")
    rng = random.Random(SEED)
    used_names = set(excluded_names)
    templates = []
    episodes = []
    for direction_index, answer in enumerate(DIRECTIONS):
        for turn_word in TURN_WORDS:
            turn = 1 if turn_word == "right" else -1
            start_direction = DIRECTIONS[(direction_index - turn) % len(DIRECTIONS)]
            subject = v9._novel(ENTITY_LENGTH, used_names, rng)
            distractors = [
                v9._confusable(subject, used_names, rng) for _ in range(max(LENGTHS) - 2)
            ]
            target_rows = [
                f"{subject} faces {start_direction}.",
                f"{subject} turns {turn_word}.",
            ]
            filler_rows = [
                f"{name} faces {DIRECTIONS[(direction_index + offset) % len(DIRECTIONS)]}."
                for offset, name in enumerate(distractors)
            ]
            template = {
                "template_id": f"v18-{answer}-{turn_word}",
                "answer_direction": answer,
                "turn_word": turn_word,
                "start_direction": start_direction,
                "subject": subject,
                "question": f"direction of {subject}?",
                "target_rows": target_rows,
                "filler_rows": filler_rows,
                "distractor_entities": distractors,
                "entities": [subject, *distractors],
            }
            templates.append(template)
            for context_length, placement in (
                (2, "baseline"),
                (8, "early"),
                (8, "late"),
                (16, "early"),
                (16, "late"),
            ):
                episodes.append(
                    _make_episode(
                        ordinal=len(episodes),
                        template=template,
                        context_length=context_length,
                        placement=placement,
                    )
                )

    ids = [item["id"] for item in episodes]
    return {
        "schema": "n1-transition-target-position-context-order-diagnostic-v1",
        "seed": SEED,
        "items": ITEMS,
        "template_count": TEMPLATE_COUNT,
        "items_per_template": ITEMS_PER_TEMPLATE,
        "context_lengths": list(LENGTHS),
        "placements": ["baseline", "early", "late"],
        "entity_length": ENTITY_LENGTH,
        "name_neighbor_distance": DISTANCE,
        "answer_directions": list(DIRECTIONS),
        "turn_words": list(TURN_WORDS),
        "subset_sizes": {"diagnostic": ITEMS},
        "subsets": {"diagnostic": ids},
        "matched_templates": templates,
        "exclusion_sources": {},
        "episodes": episodes,
    }


def audit_v18_face(face: dict, *, excluded_names: set[str], excluded_words: set[str]) -> dict:
    episodes = face.get("episodes", [])
    templates = face.get("matched_templates", [])
    names, materials, answers = v14_builder._face_entities_and_words(face)
    ids = [item.get("id") for item in episodes]
    by_template: dict[str, list[dict]] = {}
    for item in episodes:
        by_template.setdefault(item.get("template_id", ""), []).append(item)

    expected_conditions = Counter(
        {
            (2, "baseline"): 8,
            (8, "early"): 8,
            (8, "late"): 8,
            (16, "early"): 8,
            (16, "late"): 8,
        }
    )
    observed_conditions = Counter(
        (item.get("context_length"), item.get("placement")) for item in episodes
    )
    template_checks = len(templates) == TEMPLATE_COUNT and len(by_template) == TEMPLATE_COUNT
    placement_checks = True
    position_checks = True
    neighbor_checks = True
    for template in templates:
        template_id = template["template_id"]
        selected = by_template.get(template_id, [])
        keyed = {(item["context_length"], item["placement"]): item for item in selected}
        if len(selected) != ITEMS_PER_TEMPLATE or len(keyed) != ITEMS_PER_TEMPLATE:
            template_checks = False
            placement_checks = False
            continue
        baseline = keyed[(2, "baseline")]
        early8 = keyed[(8, "early")]
        late8 = keyed[(8, "late")]
        early16 = keyed[(16, "early")]
        late16 = keyed[(16, "late")]
        expected_common = (
            template["subject"],
            template["answer_direction"],
            template["turn_word"],
            template["start_direction"],
            template["question"],
        )
        for item in selected:
            actual_common = (
                item.get("entities", [None])[0],
                item.get("answer"),
                item.get("turn_word"),
                item.get("start_direction"),
                item.get("question"),
            )
            template_checks &= actual_common == expected_common
            template_checks &= item.get("answer_direction") == template["answer_direction"]

        target_rows = template["target_rows"]
        filler_rows = template["filler_rows"]
        placement_checks &= (
            baseline["observations"] == target_rows
            and baseline["target_fact_indices"] == [0, 1]
            and early8["observations"] == [*target_rows, *filler_rows[:6]]
            and late8["observations"] == [*filler_rows[:6], *target_rows]
            and early16["observations"] == [*target_rows, *filler_rows[:14]]
            and late16["observations"] == [*filler_rows[:14], *target_rows]
            and early8["observations"][2:] == late8["observations"][:6]
            and early16["observations"][2:] == late16["observations"][:14]
            and early8["observations"][2:] == early16["observations"][2:8]
            and late8["observations"][:6] == late16["observations"][:6]
        )
        position_checks &= (
            baseline["target_start_index"] == 0
            and baseline["target_fact_indices"] == [0, 1]
            and early8["target_start_index"] == 0
            and early8["target_fact_indices"] == [0, 1]
            and late8["target_start_index"] == 6
            and late8["target_fact_indices"] == [6, 7]
            and early16["target_start_index"] == 0
            and early16["target_fact_indices"] == [0, 1]
            and late16["target_start_index"] == 14
            and late16["target_fact_indices"] == [14, 15]
        )
        neighbor_checks &= all(
            sum(left != right for left, right in zip(template["subject"], name, strict=True))
            == DISTANCE
            for name in template["distractor_entities"]
        )

    template_entity_sets = [set(item["entities"]) for item in templates]
    cross_template_disjoint = all(
        not left & right
        for index, left in enumerate(template_entity_sets)
        for right in template_entity_sets[index + 1 :]
    )
    checks = {
        "schema_seed_size_and_single_subset": face.get("schema")
        == "n1-transition-target-position-context-order-diagnostic-v1"
        and face.get("seed") == SEED
        and face.get("items") == ITEMS
        and len(episodes) == ITEMS
        and ids == [f"v18-trans-{index:03d}" for index in range(ITEMS)]
        and face.get("subsets") == {"diagnostic": ids}
        and all(item.get("split") == "holdout" for item in episodes),
        "balanced_template_conditions": observed_conditions == expected_conditions
        and all(
            len(item.get("observations", [])) == item.get("context_length") for item in episodes
        ),
        "eight_direction_turn_templates": template_checks
        and Counter(item.get("answer_direction") for item in templates)
        == Counter({direction: 2 for direction in DIRECTIONS})
        and Counter(item.get("turn_word") for item in templates)
        == Counter({turn: 4 for turn in TURN_WORDS})
        and all(
            {item["turn_word"] for item in templates if item["answer_direction"] == direction}
            == set(TURN_WORDS)
            for direction in DIRECTIONS
        ),
        "same_content_early_late_and_nested_fillers": placement_checks,
        "target_positions_match_condition": position_checks,
        "unique_entities_and_neighbor_distance": len(names) == 120
        and len({name for template in templates for name in template["entities"]}) == 120
        and cross_template_disjoint
        and neighbor_checks,
        "new_entities_disjoint_from_previous_sources": not names & excluded_names,
        "direction_answers_preexcluded_and_no_material_words": answers == set(DIRECTIONS)
        and answers <= excluded_words
        and materials == set(),
        "all_target_facts_compute_answer": all(
            item["observations"][item["target_start_index"]]
            == f"{item['entities'][0]} faces {item['start_direction']}."
            and item["observations"][item["target_start_index"] + 1]
            == f"{item['entities'][0]} turns {item['turn_word']}."
            and DIRECTIONS[
                (
                    DIRECTIONS.index(item["start_direction"])
                    + (1 if item["turn_word"] == "right" else -1)
                )
                % 4
            ]
            == item["answer"]
            for item in episodes
        ),
    }
    return {
        "schema": "n1-plan136-target-position-face-checks-v1",
        "accepted": all(checks.values()),
        "checks": checks,
        "condition_counts": {
            f"{length}/{placement}": count
            for (length, placement), count in sorted(observed_conditions.items())
        },
        "template_count": len(templates),
        "entity_count": len(names),
        "observation_material_count": len(materials),
        "answer_value_count": len(answers),
        "summary_zh": "seed118题面含8个方向×转向模板；每模板设2行基线及8/16行前置/后置条件，固定事实内容与干扰行，仅改变目标事实位置。",
    }


def _inventory() -> tuple[dict, dict]:
    excluded_names, excluded_words, source_hashes, counts, endpoints = _prior_sources()
    return source_hashes, {
        "excluded_entities": sorted(excluded_names),
        "excluded_word_values": sorted(excluded_words),
        "counts": counts,
        "endpoints": endpoints,
    }


def source_readiness() -> dict:
    if any(path.exists() for path in WRITE_ONCE_OUTPUTS):
        raise FileExistsError("PLAN-N1-136 write-once输出路径已占用；不覆盖、不重试")
    started = time.process_time()
    write_new(
        READINESS_ATTEMPT,
        {
            "schema": "n1-plan136-readiness-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "approved_draft_sha256": APPROVED_DRAFT_SHA256,
            "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "new_face_generated": False,
            "model_queries": 0,
            "parameter_updates": 0,
            "说明_zh": "只核验N1-135逐题来源、seed118账本和12个完整bundle；本阶段不生成题面、不查询模型。",
        },
    )
    source_hashes: dict[str, str] = {}
    endpoint_restores: list[dict] = []
    inventory: dict = {}
    errors = []
    try:
        source_hashes, inventory = _inventory()
        _verify_source_hashes(source_hashes)
        write_new(
            SOURCE_EXCLUSIONS,
            {
                "schema": "n1-plan136-target-position-source-exclusions-v1",
                "status": "ready_for_approved_execution",
                "plan_sha256": PLAN_SHA256,
                "approved_draft_sha256": APPROVED_DRAFT_SHA256,
                "excluded_entities": inventory["excluded_entities"],
                "excluded_word_values": inventory["excluded_word_values"],
                "entity_count": inventory["counts"]["excluded_entities"],
                "word_value_count": inventory["counts"]["excluded_words"],
                "provenance": inventory["counts"],
                "source_hashes": source_hashes,
                "summary_zh": "继承N1-135对v3训练及v5-v16的完整排除，并新增排除已消费v17面120个实体；方向答案词已在既有排除集中。",
                "can_promote": False,
            },
        )
        source_hashes[str(SOURCE_EXCLUSIONS.resolve())] = sha256(SOURCE_EXCLUSIONS)
        source_hashes[str(READINESS_ATTEMPT.resolve())] = sha256(READINESS_ATTEMPT)
        n133_builder._restore_endpoints(inventory["endpoints"], endpoint_restores)
        _verify_source_hashes(source_hashes)
        complete = (
            len(endpoint_restores) == 12
            and all(row.get("restore_matches") is True for row in endpoint_restores)
            and inventory["counts"]["excluded_entities"] == EXPECTED_EXCLUDED_ENTITIES
            and inventory["counts"]["excluded_words"] == EXPECTED_EXCLUDED_WORDS
            and time.process_time() - started <= CPU_LIMIT_SECONDS
        )
        if not complete:
            errors.append("12个bundle恢复、来源并集、seed118账本或CPU门未通过")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    cpu = round(time.process_time() - started, 6)
    report = {
        "schema": "n1-plan136-target-position-readiness-v1",
        "status": "ready_for_approved_execution" if not errors else "stopped",
        "complete": not errors,
        "plan_sha256": PLAN_SHA256,
        "approved_draft_sha256": APPROVED_DRAFT_SHA256,
        "source_exclusion_manifest_sha256": (
            sha256(SOURCE_EXCLUSIONS) if SOURCE_EXCLUSIONS.exists() else None
        ),
        "source_hashes": source_hashes,
        "endpoint_count": len(endpoint_restores),
        "endpoint_restores": endpoint_restores,
        "entity_exclusion_count": inventory.get("counts", {}).get("excluded_entities"),
        "word_value_exclusion_count": inventory.get("counts", {}).get("excluded_words"),
        "model_loads": len(endpoint_restores),
        "checkpoint_loads": len(endpoint_restores),
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "new_face_generated": False,
        "cpu_seconds": cpu,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "can_promote": False,
        "summary_zh": (
            "12个冻结endpoint完整bundle恢复一致；v3训练至v17已消费来源排除完成；本阶段0模型查询、0参数更新、未生成题面。"
            if not errors
            else "N1-136 readiness停止；保留失败回执，不生成新题面、不查询、不更新。"
        ),
        "errors": errors,
    }
    report["readiness_attempt_sha256"] = sha256(READINESS_ATTEMPT)
    write_new(READINESS, report)
    return report


def _verify_readiness() -> dict:
    if not READINESS.is_file() or not SOURCE_EXCLUSIONS.is_file():
        raise FileNotFoundError("必须先完成N1-136零查询readiness")
    report = read_json(READINESS)
    manifest = read_json(SOURCE_EXCLUSIONS)
    if (
        report.get("complete") is not True
        or report.get("status") != "ready_for_approved_execution"
        or report.get("plan_sha256") != PLAN_SHA256
        or report.get("approved_draft_sha256") != APPROVED_DRAFT_SHA256
        or not seal_ast.receipt_binding_matches(
            report.get("source_exclusion_manifest_sha256"), SOURCE_EXCLUSIONS
        )
        or report.get("endpoint_count") != 12
        or report.get("entity_exclusion_count") != EXPECTED_EXCLUDED_ENTITIES
        or report.get("word_value_exclusion_count") != EXPECTED_EXCLUDED_WORDS
        or report.get("model_queries") != 0
        or report.get("parameter_updates") != 0
        or report.get("new_face_generated") is not False
        or manifest.get("entity_count") != EXPECTED_EXCLUDED_ENTITIES
        or manifest.get("word_value_count") != EXPECTED_EXCLUDED_WORDS
    ):
        raise ValueError("N1-136 readiness或source exclusions失效")
    _verify_source_hashes(report.get("source_hashes", {}))
    for endpoint in report.get("endpoint_restores", []):
        path = Path(endpoint["path"])
        if sha256(path) != endpoint.get("sha256") or endpoint.get("restore_matches") is not True:
            raise ValueError(f"endpoint readiness后checkpoint发生变化：{path}")
    return report


def _write_face_attempt(readiness: dict) -> None:
    write_new(
        FACE_ATTEMPT,
        {
            "schema": "n1-plan136-target-position-face-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
            "seed": SEED,
            "planned_items": ITEMS,
            "planned_template_count": TEMPLATE_COUNT,
            "planned_context_lengths": list(LENGTHS),
            "planned_queries": ITEMS * 12,
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
            "说明_zh": "一次性构造seed118目标事实位置诊断面；若合同失败不重抽、不重构、不重试。",
        },
    )


def build_face_after_approval(*, owner_approved: bool) -> dict:
    if not owner_approved:
        raise PermissionError("PLAN-N1-136新测面构造需要owner授权")
    readiness = _verify_readiness()
    if any(path.exists() for path in (FACE, FACE_ATTEMPT, FACE_CHECKS, RUN_REPORT, RUN_ATTEMPT)):
        raise FileExistsError("PLAN-N1-136题面/查询输出路径已占用；不覆盖、不重试")
    manifest = read_json(SOURCE_EXCLUSIONS)
    excluded_names = set(manifest["excluded_entities"])
    excluded_words = set(manifest["excluded_word_values"])
    _write_face_attempt(readiness)
    started = time.process_time()
    face = build_v18_face(excluded_names, excluded_words)
    face["exclusion_sources"] = {
        "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
        "source_hashes": manifest["source_hashes"],
    }
    checks = audit_v18_face(face, excluded_names=excluded_names, excluded_words=excluded_words)
    checks.update(
        {
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
            "face_sha256": None,
            "face_attempt_sha256": sha256(FACE_ATTEMPT),
            "model_loads": 0,
            "checkpoint_loads": 0,
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
            "build_cpu_seconds": round(time.process_time() - started, 6),
        }
    )
    write_new(FACE, face)
    checks["face_sha256"] = sha256(FACE)
    write_new(FACE_CHECKS, checks)
    return {
        "accepted": checks["accepted"],
        "face_sha256": sha256(FACE),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "model_queries": 0,
        "parameter_updates": 0,
        "build_cpu_seconds": checks["build_cpu_seconds"],
        "summary_zh": checks["summary_zh"],
        "errors": [] if checks["accepted"] else ["v18诊断面合同未通过"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--readiness", action="store_true")
    mode.add_argument("--build-face", action="store_true")
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    result = (
        source_readiness()
        if args.readiness
        else build_face_after_approval(owner_approved=args.owner_approved)
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "accepted",
                    "endpoint_count",
                    "entity_exclusion_count",
                    "word_value_exclusion_count",
                    "face_sha256",
                    "face_checks_sha256",
                    "model_queries",
                    "parameter_updates",
                    "cpu_seconds",
                    "build_cpu_seconds",
                    "summary_zh",
                    "errors",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("complete", result.get("accepted", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
