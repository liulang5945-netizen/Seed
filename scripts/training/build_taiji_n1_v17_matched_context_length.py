"""为已批准的PLAN-N1-135建立seed117同题匹配上下文长度诊断面。"""

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
    audit_taiji_n1_plan132_transition_training_coverage as n132_coverage,
)
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
    build_taiji_n1_v16_transition_direction_position_diagnostic as n134_builder,
)
from scripts.training import seal_ast  # noqa: E402

PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-135_matched_context_length_transition_diagnostic_draft_20261019.md"
)
PLAN_SHA256 = "ce13538e0cb743c53b625c6ca6aa1448bb3922d1e2ea3013c934f4fb1e3fb85b"
APPROVED_DRAFT_SHA256 = "5ae6a3ad6a59c9d49cca68f6b7a707d76d8813d77a3e11b50e3c1fef6a8e9bab"
TRAINING_COVERAGE_AUDIT = n132_coverage.OUTPUT
TRAINING_COVERAGE_AUDIT_SHA256 = "2aff999b6c745b74030fcb98b3da4bf84d9db56a2958bc916ff7b475d49b0028"

SOURCE_EXCLUSIONS = (
    ROOT
    / "plans/manifests/n1_hard_neighbor_v17_transition_context_length_seed117_source_exclusions_20261019.json"
)
READINESS = ROOT / "reports/taiji_n1_plan135_transition_context_length_readiness_20261019.json"
READINESS_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan135_transition_context_length_readiness_attempt_20261019.json"
)
FACE = (
    ROOT
    / "plans/manifests/n1_hard_neighbor_v17_matched_context_length_transition_seed117_20261019.json"
)
FACE_ATTEMPT = FACE.with_name(FACE.stem + "_attempt.json")
FACE_CHECKS = ROOT / "reports/taiji_n1_plan135_transition_context_length_face_checks_20261019.json"
RUN_REPORT = ROOT / "reports/taiji_n1_plan135_matched_context_length_diagnostic_20261019.json"
RUN_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan135_matched_context_length_diagnostic_attempt_20261019.json"
)
TEST = ROOT / "tests/taiji_native/test_n1_plan135_matched_context_length_contract.py"

N134_PLAN = n134_builder.PLAN
N134_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-134_execution_closeout_20261019.md"
N134_READINESS = n134_builder.READINESS
N134_SOURCE_EXCLUSIONS = n134_builder.SOURCE_EXCLUSIONS
N134_FACE = n134_builder.FACE
N134_FACE_ATTEMPT = n134_builder.FACE_ATTEMPT
N134_FACE_CHECKS = n134_builder.FACE_CHECKS
N134_RUN_REPORT = n134_builder.RUN_REPORT
N134_RUN_ATTEMPT = n134_builder.RUN_ATTEMPT

SEED = 117
LENGTHS = (2, 8, 16)
TEMPLATES_PER_DIRECTION = 2
ITEMS = 24
DISTANCE = 1
ENTITY_LENGTH = v14_builder.ENTITY_LENGTH
DIRECTIONS = tuple(v14_builder.DIRECTIONS)
TURN_WORDS = ("left", "right")
CPU_LIMIT_SECONDS = 600.0
EXPECTED_EXCLUDED_ENTITIES = 8538
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
    if expected is not None and actual != expected:
        raise ValueError(f"冻结来源哈希改变：{path}")
    prior = source_hashes.get(str(path))
    if prior is not None and prior != actual:
        raise ValueError(f"来源哈希冲突：{path}")
    source_hashes[str(path)] = actual


def _verify_source_hashes(source_hashes: dict[str, str]) -> None:
    for raw_path, expected in source_hashes.items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"N1-135冻结来源哈希改变：{path}")


def _prior_sources() -> tuple[set[str], set[str], dict[str, str], dict, list[dict]]:
    if sha256(PLAN) != PLAN_SHA256:
        raise ValueError("PLAN-N1-135已批准执行版本SHA-256不匹配")
    if sha256(PLAN) == APPROVED_DRAFT_SHA256:
        raise ValueError("PLAN-N1-135尚未记录用户批准状态")

    readiness = n134_builder._verify_readiness()
    manifest = read_json(N134_SOURCE_EXCLUSIONS)
    face = read_json(N134_FACE)
    face_checks = read_json(N134_FACE_CHECKS)
    run_report = read_json(N134_RUN_REPORT)
    if (
        manifest.get("entity_count") != 7938
        or manifest.get("word_value_count") != 6288
        or len(manifest.get("excluded_entities", [])) != 7938
        or len(manifest.get("excluded_word_values", [])) != 6288
        or face_checks.get("accepted") is not True
        or not seal_ast.receipt_binding_matches(face_checks.get("face_sha256"), N134_FACE)
        or face.get("seed") != 116
        or face.get("schema") != "n1-transition-direction-position-diagnostic-v1"
        or len(face.get("episodes", [])) != 40
        or run_report.get("complete") is not True
        or run_report.get("diagnostic_queries") != 480
        or run_report.get("parameter_updates") != 0
        or run_report.get("dev_queries") != 0
        or run_report.get("source_hashes_unchanged") is not True
        or run_report.get("can_promote") is not False
        or sha256(TRAINING_COVERAGE_AUDIT) != TRAINING_COVERAGE_AUDIT_SHA256
    ):
        raise ValueError("N1-134结案、训练覆盖审计或v16面证据与批准范围不符")

    excluded_names = set(manifest["excluded_entities"])
    excluded_words = set(manifest["excluded_word_values"])
    face_names, face_materials, face_answers = v14_builder._face_entities_and_words(face)
    if (
        len(face_names) != 600
        or face_materials
        or face_answers != set(DIRECTIONS)
        or excluded_names & face_names
        or not face_answers <= excluded_words
    ):
        raise ValueError("v16实体、答案值或N1-134排除来源关系不符")
    excluded_names.update(face_names)
    if (
        len(excluded_names) != EXPECTED_EXCLUDED_ENTITIES
        or len(excluded_words) != EXPECTED_EXCLUDED_WORDS
    ):
        raise ValueError(
            "N1-135来源排除并集改变："
            f"entities={len(excluded_names)}, words={len(excluded_words)}"
        )

    source_hashes = dict(readiness.get("source_hashes", {}))
    for path in (
        N134_SOURCE_EXCLUSIONS,
        N134_READINESS,
        N134_FACE,
        N134_FACE_ATTEMPT,
        N134_FACE_CHECKS,
        N134_RUN_REPORT,
        N134_RUN_ATTEMPT,
        N134_PLAN,
        N134_CLOSEOUT,
        TRAINING_COVERAGE_AUDIT,
        PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/training/eval_taiji_n1_v16_transition_direction_position_diagnostic.py",
        TEST,
    ):
        _add_source(source_hashes, path)

    endpoints = n133_builder._endpoint_specs()
    restored_by_key = {
        (row.get("source"), row.get("seed"), row.get("arm")): row
        for row in readiness.get("endpoint_restores", [])
    }
    if len(endpoints) != 12 or len(restored_by_key) != 12:
        raise ValueError("PLAN-N1-135必须绑定12个已恢复的冻结endpoint")
    for endpoint in endpoints:
        key = (endpoint["source"], endpoint["seed"], endpoint["arm"])
        receipt = restored_by_key.get(key)
        if (
            receipt is None
            or receipt.get("restore_matches") is not True
            or receipt.get("sha256") != endpoint["sha256"]
            or Path(receipt.get("path", "")).resolve() != Path(endpoint["path"]).resolve()
        ):
            raise ValueError(f"N1-134完整bundle回执与冻结endpoint不符：{key}")
        _add_source(source_hashes, Path(endpoint["path"]), endpoint["sha256"])

    for ledger_path in sorted((ROOT / "plans/manifests").glob("n1_hard_neighbor_*.json")):
        ledger = read_json(ledger_path)
        _add_source(source_hashes, ledger_path)
        if ledger.get("seed") == SEED:
            raise ValueError(f"seed117已在硬邻居账本中使用：{ledger_path}")
    _verify_source_hashes(source_hashes)
    return (
        excluded_names,
        excluded_words,
        source_hashes,
        {
            "n134_excluded_entities": 7938,
            "n134_excluded_words": 6288,
            "v16_face_entities": len(face_names),
            "v16_face_answers": len(face_answers),
            "excluded_entities": len(excluded_names),
            "excluded_words": len(excluded_words),
            "source_hash_count": len(source_hashes),
            "endpoint_count": len(endpoints),
            "training_coverage_audit_sha256": sha256(TRAINING_COVERAGE_AUDIT),
        },
        endpoints,
    )


def build_v17_face(excluded_names: set[str], excluded_words: set[str]) -> dict:
    """构造八个方向×转向模板，各含嵌套2/8/16行上下文。"""
    if not set(DIRECTIONS) <= excluded_words:
        raise ValueError("四个方向答案必须已包含在完整来源排除并集中")
    rng = random.Random(SEED)
    used_names = set(excluded_names)
    episodes = []
    template_rows = []
    for direction_index, answer in enumerate(DIRECTIONS):
        for turn_word in TURN_WORDS:
            template_id = f"v17-{answer}-{turn_word}"
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
            template_rows.append(
                {
                    "template_id": template_id,
                    "answer_direction": answer,
                    "turn_word": turn_word,
                    "start_direction": start_direction,
                    "subject": subject,
                    "target_rows": target_rows,
                    "filler_rows": filler_rows,
                    "entities": [subject, *distractors],
                }
            )
            for context_length in LENGTHS:
                ordinal = len(episodes)
                filler_count = context_length - len(target_rows)
                observations = target_rows + filler_rows[:filler_count]
                entities = [subject, *distractors[:filler_count]]
                episodes.append(
                    {
                        "id": f"v17-trans-{ordinal:03d}",
                        "split": "holdout",
                        "subset": "diagnostic",
                        "kind": "transition",
                        "kind_ordinal": ordinal,
                        "template_id": template_id,
                        "context_length": context_length,
                        "answer_direction": answer,
                        "entities": entities,
                        "observations": observations,
                        "question": f"direction of {subject}?",
                        "answer": answer,
                        "target_fact_indices": [0, 1],
                        "start_direction": start_direction,
                        "turn_word": turn_word,
                        "neighbor_distance": DISTANCE,
                    }
                )

    ids = [item["id"] for item in episodes]
    return {
        "schema": "n1-transition-matched-context-length-diagnostic-v1",
        "seed": SEED,
        "items": ITEMS,
        "template_count": len(template_rows),
        "items_per_template": len(LENGTHS),
        "context_lengths": list(LENGTHS),
        "entity_length": ENTITY_LENGTH,
        "name_neighbor_distance": DISTANCE,
        "answer_directions": list(DIRECTIONS),
        "turn_words": list(TURN_WORDS),
        "subset_sizes": {"diagnostic": ITEMS},
        "subsets": {"diagnostic": ids},
        "matched_templates": template_rows,
        "exclusion_sources": {},
        "episodes": episodes,
    }


def audit_v17_face(face: dict, *, excluded_names: set[str], excluded_words: set[str]) -> dict:
    episodes = face.get("episodes", [])
    templates = face.get("matched_templates", [])
    names, materials, answers = v14_builder._face_entities_and_words(face)
    ids = [item.get("id") for item in episodes]
    by_template: dict[str, list[dict]] = {}
    for item in episodes:
        by_template.setdefault(item.get("template_id", ""), []).append(item)

    template_checks = len(templates) == 8 and len(by_template) == 8
    nested_checks = True
    neighbor_checks = True
    for template in templates:
        template_id = template["template_id"]
        matched = sorted(by_template.get(template_id, []), key=lambda row: row["context_length"])
        if [item.get("context_length") for item in matched] != list(LENGTHS):
            nested_checks = False
            template_checks = False
            continue
        two, eight, sixteen = matched
        template_checks &= all(
            item.get("answer") == template["answer_direction"]
            and item.get("answer_direction") == template["answer_direction"]
            and item.get("turn_word") == template["turn_word"]
            and item.get("start_direction") == template["start_direction"]
            and item.get("question") == two.get("question")
            and item.get("target_fact_indices") == [0, 1]
            and item.get("entities", [None])[0] == template["subject"]
            for item in matched
        )
        nested_checks &= (
            two["observations"] == template["target_rows"]
            and eight["observations"][:2] == two["observations"]
            and sixteen["observations"][:8] == eight["observations"]
            and sixteen["observations"][:2] == two["observations"]
            and eight["observations"][2:] == template["filler_rows"][:6]
            and sixteen["observations"][2:] == template["filler_rows"][:14]
            and eight["entities"] == [template["subject"], *template["entities"][1:7]]
            and sixteen["entities"] == template["entities"]
        )
        neighbor_checks &= all(
            sum(left != right for left, right in zip(template["subject"], name, strict=True))
            == DISTANCE
            for name in template["entities"][1:]
        )

    template_entity_sets = [set(item["entities"]) for item in templates]
    cross_template_disjoint = all(
        not left & right
        for index, left in enumerate(template_entity_sets)
        for right in template_entity_sets[index + 1 :]
    )
    checks = {
        "schema_seed_size_and_single_subset": face.get("schema")
        == "n1-transition-matched-context-length-diagnostic-v1"
        and face.get("seed") == SEED
        and face.get("items") == ITEMS
        and len(episodes) == ITEMS
        and ids == [f"v17-trans-{index:03d}" for index in range(ITEMS)]
        and face.get("subsets") == {"diagnostic": ids}
        and all(item.get("split") == "holdout" for item in episodes),
        "balanced_context_lengths": Counter(item.get("context_length") for item in episodes)
        == Counter({length: 8 for length in LENGTHS})
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
        "same_template_targets_and_nested_context": nested_checks,
        "unique_entities_and_neighbor_distance": len(names) == 120
        and len({name for template in templates for name in template["entities"]}) == 120
        and cross_template_disjoint
        and neighbor_checks,
        "new_entities_disjoint_from_previous_sources": not names & excluded_names,
        "direction_answers_preexcluded_and_no_material_words": answers == set(DIRECTIONS)
        and answers <= excluded_words
        and materials == set(),
        "target_rows_compute_answer_direction": all(
            item["observations"][0] == f"{item['entities'][0]} faces {item['start_direction']}."
            and item["observations"][1] == f"{item['entities'][0]} turns {item['turn_word']}."
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
        "schema": "n1-plan135-matched-context-length-face-checks-v1",
        "accepted": all(checks.values()),
        "checks": checks,
        "context_length_counts": {
            str(length): sum(item.get("context_length") == length for item in episodes)
            for length in LENGTHS
        },
        "template_count": len(templates),
        "entity_count": len(names),
        "observation_material_count": len(materials),
        "answer_value_count": len(answers),
        "summary_zh": "seed117题面含8个方向×转向模板，各有嵌套2/8/16行版本；目标事实固定在索引0/1，8行版的六条干扰行在16行版中原样保留。",
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
        raise FileExistsError("PLAN-N1-135 write-once输出路径已占用；不覆盖、不重试")
    started = time.process_time()
    write_new(
        READINESS_ATTEMPT,
        {
            "schema": "n1-plan135-readiness-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "approved_draft_sha256": APPROVED_DRAFT_SHA256,
            "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "new_face_generated": False,
            "model_queries": 0,
            "parameter_updates": 0,
            "说明_zh": "只核验N1-134完整来源、12个冻结bundle和seed117账本；本阶段不生成题面、不查询模型。",
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
                "schema": "n1-plan135-transition-source-exclusions-v1",
                "status": "ready_for_approved_execution",
                "plan_sha256": PLAN_SHA256,
                "approved_draft_sha256": APPROVED_DRAFT_SHA256,
                "excluded_entities": inventory["excluded_entities"],
                "excluded_word_values": inventory["excluded_word_values"],
                "entity_count": inventory["counts"]["excluded_entities"],
                "word_value_count": inventory["counts"]["excluded_words"],
                "provenance": inventory["counts"],
                "source_hashes": source_hashes,
                "summary_zh": "继承N1-134对v3训练及v5-v15的完整排除，并新增排除已消费v16面600个实体；方向答案词已在既有排除集中。",
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
            errors.append("12个bundle恢复、来源并集、seed117账本或CPU门未通过")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    cpu = round(time.process_time() - started, 6)
    report = {
        "schema": "n1-plan135-transition-readiness-v1",
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
            "12个冻结endpoint完整bundle恢复一致；v3训练至v16已消费来源排除完成；本阶段0模型查询、0参数更新、未生成题面。"
            if not errors
            else "N1-135 readiness停止；保留失败回执，不生成题面、不查询、不更新。"
        ),
        "errors": errors,
    }
    report["readiness_attempt_sha256"] = sha256(READINESS_ATTEMPT)
    write_new(READINESS, report)
    return report


def _verify_readiness() -> dict:
    if not READINESS.is_file() or not SOURCE_EXCLUSIONS.is_file():
        raise FileNotFoundError("必须先完成N1-135零查询readiness")
    report = read_json(READINESS)
    manifest = read_json(SOURCE_EXCLUSIONS)
    if (
        report.get("complete") is not True
        or report.get("status") != "ready_for_approved_execution"
        or report.get("plan_sha256") != PLAN_SHA256
        or report.get("approved_draft_sha256") != APPROVED_DRAFT_SHA256
        or report.get("source_exclusion_manifest_sha256") != sha256(SOURCE_EXCLUSIONS)
        or report.get("endpoint_count") != 12
        or report.get("entity_exclusion_count") != EXPECTED_EXCLUDED_ENTITIES
        or report.get("word_value_exclusion_count") != EXPECTED_EXCLUDED_WORDS
        or report.get("model_queries") != 0
        or report.get("parameter_updates") != 0
        or report.get("new_face_generated") is not False
        or manifest.get("entity_count") != EXPECTED_EXCLUDED_ENTITIES
        or manifest.get("word_value_count") != EXPECTED_EXCLUDED_WORDS
    ):
        raise ValueError("N1-135 readiness或source exclusions失效")
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
            "schema": "n1-plan135-transition-face-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
            "seed": SEED,
            "planned_items": ITEMS,
            "planned_template_count": 8,
            "planned_context_lengths": list(LENGTHS),
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
            "说明_zh": "一次性构造seed117同模板2/8/16行嵌套诊断面；若合同失败不重抽、不重构、不重试。",
        },
    )


def build_face_after_approval(*, owner_approved: bool) -> dict:
    if not owner_approved:
        raise PermissionError("PLAN-N1-135新测面构造需要owner授权")
    readiness = _verify_readiness()
    if any(path.exists() for path in (FACE, FACE_ATTEMPT, FACE_CHECKS, RUN_REPORT, RUN_ATTEMPT)):
        raise FileExistsError("PLAN-N1-135题面/查询输出路径已占用；不覆盖、不重试")
    manifest = read_json(SOURCE_EXCLUSIONS)
    excluded_names = set(manifest["excluded_entities"])
    excluded_words = set(manifest["excluded_word_values"])
    _write_face_attempt(readiness)
    started = time.process_time()
    face = build_v17_face(excluded_names, excluded_words)
    face["exclusion_sources"] = {
        "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
        "source_hashes": manifest["source_hashes"],
    }
    checks = audit_v17_face(face, excluded_names=excluded_names, excluded_words=excluded_words)
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
        "errors": [] if checks["accepted"] else ["v17诊断面合同未通过"],
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
