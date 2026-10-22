"""为已批准的PLAN-N1-134建立seed116转向方向×目标位置诊断面。"""

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
    analyze_taiji_n1_v15_after_n132_itemwise_attribution as n133_itemwise,
)
from scripts.training import (  # noqa: E402
    audit_taiji_n1_plan132_transition_training_coverage as n132_coverage,
)
from scripts.training import (  # noqa: E402
    build_taiji_n1_v14_after_n123 as v14_builder,
)
from scripts.training import (  # noqa: E402
    build_taiji_n1_v15_after_n132 as n133_builder,
)
from scripts.training import seal_ast  # noqa: E402

PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-134_transition_direction_position_counterbalance_diagnostic_draft_20261019.md"
)
PLAN_SHA256 = "845d0b32e623a8f64fff812ea57fe6620d2316a9edc6ec493d51559c119de6e9"
TRAINING_COVERAGE_AUDIT = n132_coverage.OUTPUT
TRAINING_COVERAGE_AUDIT_SHA256 = "2aff999b6c745b74030fcb98b3da4bf84d9db56a2958bc916ff7b475d49b0028"

SOURCE_EXCLUSIONS = (
    ROOT
    / "plans/manifests/n1_hard_neighbor_v16_transition_diagnostic_seed116_source_exclusions_20261019.json"
)
READINESS = ROOT / "reports/taiji_n1_plan134_transition_diagnostic_readiness_20261019.json"
READINESS_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan134_transition_diagnostic_readiness_attempt_20261019.json"
)
FACE = (
    ROOT
    / "plans/manifests/n1_hard_neighbor_v16_transition_direction_position_seed116_20261019.json"
)
FACE_ATTEMPT = (
    ROOT
    / "plans/manifests/n1_hard_neighbor_v16_transition_direction_position_seed116_20261019_attempt.json"
)
FACE_CHECKS = ROOT / "reports/taiji_n1_plan134_transition_face_checks_20261019.json"
RUN_REPORT = (
    ROOT / "reports/taiji_n1_plan134_transition_direction_position_diagnostic_20261019.json"
)
RUN_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan134_transition_direction_position_diagnostic_attempt_20261019.json"
)

N133_READINESS = n133_builder.READINESS
N133_SOURCE_EXCLUSIONS = n133_builder.SOURCE_EXCLUSIONS
N133_FACE = n133_builder.FACE
N133_FACE_CHECKS = n133_builder.FACE_CHECKS
N133_RANGE = n133_builder.RANGE_REPORT
N133_CONFIRM = n133_builder.CONFIRM_REPORT
N133_PLAN = n133_builder.PLAN
N133_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-133_execution_closeout_20261019.md"
N133_CANDIDATE_ITEMWISE = n133_itemwise.OUTPUT
N132_PLAN = n133_builder.N132_PLAN
N132_CLOSEOUT = n133_builder.N132_CLOSEOUT
N132_RESULT = n133_builder.N132_RESULT
N132_RECONCILIATION = n133_builder.N132_RECONCILIATION

SEED = 116
ITEMS = 40
OBSERVATIONS = 16
DISTANCE = 1
ENTITY_LENGTH = v14_builder.ENTITY_LENGTH
DIRECTIONS = tuple(v14_builder.DIRECTIONS)
TARGET_STARTS = (0, 4, 7, 11, 14)
CPU_LIMIT_SECONDS = 600.0

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
            raise ValueError(f"N1-134冻结来源哈希改变：{path}")


def _prior_sources() -> tuple[set[str], set[str], dict[str, str], dict, list[dict]]:
    if sha256(PLAN) != PLAN_SHA256:
        raise ValueError("PLAN-N1-134冻结计划SHA-256不匹配")
    readiness = n133_builder._verify_readiness()
    prior_manifest = read_json(N133_SOURCE_EXCLUSIONS)
    prior_face = read_json(N133_FACE)
    prior_face_checks = read_json(N133_FACE_CHECKS)
    prior_range = read_json(N133_RANGE)
    prior_confirm = read_json(N133_CONFIRM)
    prior_analysis = read_json(N133_CANDIDATE_ITEMWISE)
    training_audit = read_json(TRAINING_COVERAGE_AUDIT)
    if (
        prior_manifest.get("plan_sha256") != n133_builder.PLAN_SHA256
        or prior_manifest.get("entity_count") != 7308
        or prior_manifest.get("word_value_count") != 5808
        or prior_face.get("schema") != "n1-hard-neighbor-v15-confirmation-v1"
        or prior_face.get("seed") != 115
        or len(prior_face.get("episodes", [])) != 40
        or prior_face_checks.get("accepted") is not True
        or not seal_ast.receipt_binding_matches(prior_face_checks.get("face_sha256"), N133_FACE)
        or prior_range.get("complete") is not True
        or prior_range.get("model_queries") != 180
        or prior_range.get("confirmation_queries") != 0
        or prior_range.get("range_gate", {}).get("range_check_passed") is not True
        or prior_confirm.get("complete") is not True
        or prior_confirm.get("confirmation_queries") != 240
        or prior_confirm.get("total_model_queries") != 420
        or prior_confirm.get("parameter_updates") != 0
        or prior_confirm.get("dev_queries") != 0
        or prior_confirm.get("source_hashes_unchanged") is not True
        or prior_confirm.get("can_promote") is not False
        or prior_analysis.get("status") != "complete"
        or prior_analysis.get("model_queries") != 0
        or prior_analysis.get("parameter_updates") != 0
        or not seal_ast.receipt_binding_matches(
            prior_analysis.get("confirmation_report_sha256"), N133_CONFIRM
        )
        or sha256(TRAINING_COVERAGE_AUDIT) != TRAINING_COVERAGE_AUDIT_SHA256
        or training_audit.get("status") != "complete"
        or training_audit.get("train_split_only") is not True
        or training_audit.get("model_loads") != 0
        or training_audit.get("model_queries") != 0
        or training_audit.get("parameter_updates") != 0
        or training_audit.get("dev_queries") != 0
        or training_audit.get("holdout_queries") != 0
    ):
        raise ValueError("N1-133结案、v15面或train-only覆盖审计与批准范围不一致")

    excluded_names = set(prior_manifest.get("excluded_entities", []))
    excluded_words = set(prior_manifest.get("excluded_word_values", []))
    face_names, face_materials, face_answers = v14_builder._face_entities_and_words(prior_face)
    answer_prior_overlap = len(excluded_words & face_answers)
    material_answer_overlap = len(face_materials & face_answers)
    if (
        len(face_names) != 630
        or len(face_materials) != 480
        or len(face_answers) != 25
        or excluded_names & face_names
        or excluded_words & face_materials
        or answer_prior_overlap != 5
        or material_answer_overlap != 20
    ):
        raise ValueError("v15面实体、材料、答案与排除并集关系不符")
    excluded_names.update(face_names)
    excluded_words.update(face_materials)
    excluded_words.update(face_answers)
    if len(excluded_names) != 7938 or len(excluded_words) != 6288:
        raise ValueError(
            f"N1-134来源排除并集改变：entities={len(excluded_names)}, words={len(excluded_words)}"
        )

    source_hashes = dict(readiness.get("source_hashes", {}))
    for raw_path, expected in (
        prior_face.get("exclusion_sources", {}).get("source_hashes", {}).items()
    ):
        _add_source(source_hashes, Path(raw_path), expected)
    for path in (
        N133_SOURCE_EXCLUSIONS,
        N133_READINESS,
        N133_FACE,
        N133_FACE_CHECKS,
        N133_RANGE,
        N133_CONFIRM,
        N133_CANDIDATE_ITEMWISE,
        TRAINING_COVERAGE_AUDIT,
        N133_PLAN,
        N133_CLOSEOUT,
        N132_PLAN,
        N132_CLOSEOUT,
        N132_RESULT,
        N132_RECONCILIATION,
        PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/training/eval_taiji_n1_v16_transition_direction_position_diagnostic.py",
        ROOT / "tests/taiji_native/test_n1_plan134_transition_diagnostic_contract.py",
    ):
        _add_source(source_hashes, path)

    for manifest_path in sorted((ROOT / "plans/manifests").glob("n1_hard_neighbor_*.json")):
        ledger = read_json(manifest_path)
        _add_source(source_hashes, manifest_path)
        if ledger.get("seed") == SEED:
            raise ValueError(f"seed116已出现在硬邻居账本中：{manifest_path}")

    endpoints = n133_builder._endpoint_specs()
    if len(endpoints) != 12:
        raise ValueError("PLAN-N1-134必须绑定N1-133的12个冻结endpoint")
    readiness_endpoints = readiness.get("endpoint_restores", [])
    restored_by_key = {
        (item.get("source"), item.get("seed"), item.get("arm")): item
        for item in readiness_endpoints
    }
    if len(restored_by_key) != 12:
        raise ValueError("N1-133 readiness缺少12个endpoint恢复回执")
    for endpoint in endpoints:
        key = (endpoint["source"], endpoint["seed"], endpoint["arm"])
        restored = restored_by_key.get(key)
        if (
            restored is None
            or restored.get("restore_matches") is not True
            or restored.get("sha256") != endpoint["sha256"]
        ):
            raise ValueError(f"N1-133冻结endpoint恢复/哈希与终点清单不符：{key}")
        _add_source(source_hashes, Path(endpoint["path"]), endpoint["sha256"])
    _verify_source_hashes(source_hashes)
    return (
        excluded_names,
        excluded_words,
        source_hashes,
        {
            "n133_excluded_entities": 7308,
            "n133_excluded_words": 5808,
            "v15_face_entities": len(face_names),
            "v15_face_materials": len(face_materials),
            "v15_face_answers": len(face_answers),
            "v15_answer_overlap_with_prior": answer_prior_overlap,
            "v15_material_answer_overlap": material_answer_overlap,
            "excluded_entities": len(excluded_names),
            "excluded_words": len(excluded_words),
            "source_hash_count": len(source_hashes),
            "endpoint_count": len(endpoints),
            "training_coverage_audit_sha256": sha256(TRAINING_COVERAGE_AUDIT),
        },
        endpoints,
    )


def build_v16_face(excluded_names: set[str], excluded_words: set[str]) -> dict:
    """构造40项完全平衡的方向×目标位置transition面。"""
    rng = random.Random(SEED)
    used_names = set(excluded_names)
    if not set(DIRECTIONS) <= excluded_words:
        raise ValueError("四个方向答案必须已包含在完整来源排除并集中")
    episodes = []
    for target_start in TARGET_STARTS:
        for direction_index, answer in enumerate(DIRECTIONS):
            for replicate in (0, 1):
                ordinal = len(episodes)
                subject = v14_builder.v9._novel(ENTITY_LENGTH, used_names, rng)
                distractors = [
                    v14_builder.v9._confusable(subject, used_names, rng)
                    for _ in range(OBSERVATIONS - 2)
                ]
                turn = 1 if replicate == 0 else 3
                start_direction = DIRECTIONS[(direction_index - turn) % len(DIRECTIONS)]
                turn_word = "right" if turn == 1 else "left"
                target_rows = [
                    f"{subject} faces {start_direction}.",
                    f"{subject} turns {turn_word}.",
                ]
                filler = [
                    f"{name} faces {DIRECTIONS[(ordinal + offset) % len(DIRECTIONS)]}."
                    for offset, name in enumerate(distractors)
                ]
                rows: list[str | None] = [None] * OBSERVATIONS
                rows[target_start] = target_rows[0]
                rows[target_start + 1] = target_rows[1]
                filler_iter = iter(filler)
                for position, row in enumerate(rows):
                    if row is None:
                        rows[position] = next(filler_iter)
                episodes.append(
                    {
                        "id": f"v16-trans-{ordinal:03d}",
                        "split": "holdout",
                        "subset": "diagnostic",
                        "kind": "transition",
                        "kind_ordinal": ordinal,
                        "direction_position_replicate": replicate,
                        "answer_direction": answer,
                        "entities": [subject, *distractors],
                        "observations": [str(row) for row in rows],
                        "question": f"direction of {subject}?",
                        "answer": answer,
                        "target_start_index": target_start,
                        "turn_word": turn_word,
                    }
                )
    ids = [item["id"] for item in episodes]
    return {
        "schema": "n1-transition-direction-position-diagnostic-v1",
        "seed": SEED,
        "items": ITEMS,
        "items_per_direction_position": 2,
        "observations_per_item": OBSERVATIONS,
        "entity_length": ENTITY_LENGTH,
        "name_neighbor_distance": DISTANCE,
        "answer_directions": list(DIRECTIONS),
        "target_start_indices": list(TARGET_STARTS),
        "subset_sizes": {"diagnostic": ITEMS},
        "subsets": {"diagnostic": ids},
        "exclusion_sources": {},
        "episodes": episodes,
    }


def audit_v16_face(face: dict, *, excluded_names: set[str], excluded_words: set[str]) -> dict:
    episodes = face.get("episodes", [])
    ids = [item.get("id") for item in episodes]
    names, materials, answers = v14_builder._face_entities_and_words(face)
    cell_counts = Counter(
        (item.get("answer_direction"), item.get("target_start_index")) for item in episodes
    )
    cell_turns = Counter(
        (
            item.get("answer_direction"),
            item.get("target_start_index"),
            item.get("turn_word"),
        )
        for item in episodes
    )
    row_checks = True
    for item in episodes:
        subject = item["entities"][0]
        start = item["target_start_index"]
        if not 0 <= start < OBSERVATIONS - 1:
            row_checks = False
            continue
        expected_prefixes = (subject + " faces ", subject + " turns ")
        if not all(
            item["observations"][start + offset].startswith(prefix)
            for offset, prefix in enumerate(expected_prefixes)
        ):
            row_checks = False
            continue
        start_direction = (
            item["observations"][start].removeprefix(subject + " faces ").removesuffix(".")
        )
        turn_word = (
            item["observations"][start + 1].removeprefix(subject + " turns ").removesuffix(".")
        )
        turn = 1 if turn_word == "right" else 3 if turn_word == "left" else None
        if start_direction not in DIRECTIONS or turn is None:
            row_checks = False
            continue
        expected_answer = DIRECTIONS[(DIRECTIONS.index(start_direction) + turn) % len(DIRECTIONS)]
        row_checks &= item.get("answer") == expected_answer

    checks = {
        "schema_seed_and_size": face.get("schema")
        == "n1-transition-direction-position-diagnostic-v1"
        and face.get("seed") == SEED
        and face.get("items") == ITEMS
        and len(episodes) == ITEMS,
        "unique_ids_and_single_diagnostic_subset": len(ids) == ITEMS
        and ids == [f"v16-trans-{index:03d}" for index in range(ITEMS)]
        and all(item.get("split") == "holdout" for item in episodes)
        and face.get("subsets") == {"diagnostic": ids},
        "sixteen_observation_rows": all(
            len(item.get("observations", [])) == OBSERVATIONS for item in episodes
        ),
        "complete_direction_position_grid": cell_counts
        == Counter({(direction, start): 2 for direction in DIRECTIONS for start in TARGET_STARTS}),
        "left_right_balanced_in_each_cell": cell_turns
        == Counter(
            {
                (direction, start, turn): 1
                for direction in DIRECTIONS
                for start in TARGET_STARTS
                for turn in ("left", "right")
            }
        ),
        "two_target_rows_and_direction_consistency": row_checks,
        "unique_entities_and_name_neighbor_distance": len(names) == 600
        and len({name for item in episodes for name in item["entities"]}) == 600
        and all(
            sum(left != right for left, right in zip(item["entities"][0], distractor, strict=True))
            == DISTANCE
            for item in episodes
            for distractor in item["entities"][1:]
        ),
        "new_entities_disjoint_from_previous_sources": not names & excluded_names,
        "direction_answers_preexcluded": answers == set(DIRECTIONS) and answers <= excluded_words,
        "no_material_word_observations": materials == set(),
        "target_metadata_matches_rows": all(
            item["observations"][item["target_start_index"]].startswith(
                item["entities"][0] + " faces "
            )
            and item["observations"][item["target_start_index"] + 1].startswith(
                item["entities"][0] + " turns "
            )
            for item in episodes
        ),
    }
    return {
        "schema": "n1-plan134-transition-direction-position-face-checks-v1",
        "accepted": all(checks.values()),
        "checks": checks,
        "direction_position_cell_counts": {
            f"{direction}/{start}": count
            for (direction, start), count in sorted(cell_counts.items())
        },
        "entity_count": len(names),
        "observation_material_count": len(materials),
        "answer_value_count": len(answers),
        "summary_zh": "seed116诊断面40项方向×目标位置完全平衡；每格一左一右两个模板，16行合同、实体距离和来源排除均核验。",
    }


def _inventory() -> tuple[dict, dict]:
    excluded_names, excluded_words, source_hashes, counts, endpoints = _prior_sources()
    inventory = {
        "excluded_entities": sorted(excluded_names),
        "excluded_word_values": sorted(excluded_words),
        "counts": counts,
        "endpoints": endpoints,
    }
    return source_hashes, inventory


def source_readiness() -> dict:
    if any(path.exists() for path in WRITE_ONCE_OUTPUTS):
        raise FileExistsError("PLAN-N1-134输出路径已占用；不覆盖、不重试")
    started = time.process_time()
    write_new(
        READINESS_ATTEMPT,
        {
            "schema": "n1-plan134-readiness-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "new_face_generated": False,
            "model_queries": 0,
            "parameter_updates": 0,
            "说明_zh": "只核验来源、endpoint哈希与完整bundle恢复；若失败保留回执，不生成题面、不查询模型。",
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
                "schema": "n1-plan134-transition-source-exclusions-v1",
                "status": "ready_for_approved_execution",
                "plan_sha256": PLAN_SHA256,
                "excluded_entities": inventory["excluded_entities"],
                "excluded_word_values": inventory["excluded_word_values"],
                "entity_count": inventory["counts"]["excluded_entities"],
                "word_value_count": inventory["counts"]["excluded_words"],
                "provenance": inventory["counts"],
                "source_hashes": source_hashes,
                "summary_zh": "完整排除v3训练及已消费v5-v15面；额外排除v15新面实体、材料和答案值。",
                "can_promote": False,
            },
        )
        source_hashes[str(SOURCE_EXCLUSIONS.resolve())] = sha256(SOURCE_EXCLUSIONS)
        source_hashes[str(READINESS_ATTEMPT.resolve())] = sha256(READINESS_ATTEMPT)
        _verify_source_hashes(
            {
                key: value
                for key, value in source_hashes.items()
                if key != str(SOURCE_EXCLUSIONS.resolve())
            }
        )
        n133_builder._restore_endpoints(inventory["endpoints"], endpoint_restores)
        _verify_source_hashes(source_hashes)
        complete = (
            len(endpoint_restores) == 12
            and all(item.get("restore_matches") is True for item in endpoint_restores)
            and len(inventory["excluded_entities"]) == 7938
            and len(inventory["excluded_word_values"]) == 6288
            and time.process_time() - started <= CPU_LIMIT_SECONDS
        )
        if not complete:
            errors.append("12个bundle恢复、来源并集或CPU门未通过")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    cpu = round(time.process_time() - started, 6)
    report = {
        "schema": "n1-plan134-transition-readiness-v1",
        "status": "ready_for_approved_execution" if not errors else "stopped",
        "complete": not errors,
        "plan_sha256": PLAN_SHA256,
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
            "12个完整checkpoint bundle恢复和v15扩展来源并集已核验；本阶段未生成新面、未查询、未更新。"
            if not errors
            else "N1-134 readiness停止；保留失败收据，不生成新面、不查询、不更新。"
        ),
        "errors": errors,
    }
    report["readiness_attempt_sha256"] = sha256(READINESS_ATTEMPT)
    write_new(READINESS, report)
    return report


def _verify_readiness() -> dict:
    if not READINESS.is_file() or not SOURCE_EXCLUSIONS.is_file():
        raise FileNotFoundError("必须先完成N1-134零查询readiness")
    report = read_json(READINESS)
    manifest = read_json(SOURCE_EXCLUSIONS)
    if (
        report.get("complete") is not True
        or report.get("status") != "ready_for_approved_execution"
        or report.get("plan_sha256") != PLAN_SHA256
        or not seal_ast.receipt_binding_matches(
            report.get("source_exclusion_manifest_sha256"), SOURCE_EXCLUSIONS
        )
        or report.get("endpoint_count") != 12
        or report.get("model_queries") != 0
        or report.get("parameter_updates") != 0
        or report.get("new_face_generated") is not False
        or manifest.get("entity_count") != 7938
        or manifest.get("word_value_count") != 6288
    ):
        raise ValueError("N1-134 readiness或source exclusions失效")
    _verify_source_hashes(
        {
            key: value
            for key, value in report.get("source_hashes", {}).items()
            if key != str(SOURCE_EXCLUSIONS.resolve())
        }
    )
    for endpoint in report.get("endpoint_restores", []):
        path = Path(endpoint["path"])
        if sha256(path) != endpoint.get("sha256") or endpoint.get("restore_matches") is not True:
            raise ValueError(f"endpoint readiness后checkpoint发生变化：{path}")
    return report


def _write_face_attempt(readiness: dict) -> None:
    write_new(
        FACE_ATTEMPT,
        {
            "schema": "n1-plan134-transition-face-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
            "seed": SEED,
            "planned_items": ITEMS,
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
            "说明_zh": "一次性构造seed116方向×位置诊断面；若合同失败不换seed、不重构、不重试。",
        },
    )


def build_face_after_approval(*, owner_approved: bool) -> dict:
    if not owner_approved:
        raise PermissionError("PLAN-N1-134构面需要owner授权")
    readiness = _verify_readiness()
    if any(path.exists() for path in (FACE, FACE_ATTEMPT, FACE_CHECKS, RUN_REPORT, RUN_ATTEMPT)):
        raise FileExistsError("N1-134题面/查询输出路径已占用；不覆盖、不重试")
    manifest = read_json(SOURCE_EXCLUSIONS)
    excluded_names = set(manifest["excluded_entities"])
    excluded_words = set(manifest["excluded_word_values"])
    _write_face_attempt(readiness)
    started = time.process_time()
    face = build_v16_face(excluded_names, excluded_words)
    face["exclusion_sources"] = {
        "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
        "source_hashes": manifest["source_hashes"],
    }
    checks = audit_v16_face(face, excluded_names=excluded_names, excluded_words=excluded_words)
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
        "errors": [] if checks["accepted"] else ["v16诊断面合同未通过"],
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
