"""为已批准的PLAN-N1-137建立seed119近邻距离干扰诊断面。"""

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
    build_taiji_n1_v18_target_position_context_order as n136_builder,
)

PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-137_neighbor_distance_context_interference_diagnostic_draft_20261019.md"
)
PLAN_SHA256 = "d6e75d3251cc5690211b16e649f00e4ee78206308a4805995465a48c0d7d15f0"
APPROVED_DRAFT_SHA256 = "c76fd2ca5c0c19f9c38afe14c1facfe8871bcd7ddb44766a64710cd06bad44df"

N136_PLAN = n136_builder.PLAN
N136_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-136_execution_closeout_20261019.md"
N136_SOURCE_EXCLUSIONS = n136_builder.SOURCE_EXCLUSIONS
N136_READINESS = n136_builder.READINESS
N136_FACE = n136_builder.FACE
N136_FACE_ATTEMPT = n136_builder.FACE_ATTEMPT
N136_FACE_CHECKS = n136_builder.FACE_CHECKS
N136_RUN_REPORT = n136_builder.RUN_REPORT
N136_RUN_ATTEMPT = n136_builder.RUN_ATTEMPT

SOURCE_EXCLUSIONS = (
    ROOT
    / "plans/manifests/n1_hard_neighbor_v19_neighbor_distance_seed119_source_exclusions_20261019.json"
)
READINESS = ROOT / "reports/taiji_n1_plan137_neighbor_distance_readiness_20261019.json"
READINESS_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan137_neighbor_distance_readiness_attempt_20261019.json"
)
FACE = (
    ROOT
    / "plans/manifests/n1_hard_neighbor_v19_neighbor_distance_context_interference_seed119_20261019.json"
)
FACE_ATTEMPT = FACE.with_name(FACE.stem + "_attempt.json")
FACE_CHECKS = ROOT / "reports/taiji_n1_plan137_neighbor_distance_face_checks_20261019.json"
RUN_REPORT = (
    ROOT
    / "reports/taiji_n1_plan137_neighbor_distance_context_interference_diagnostic_20261019.json"
)
RUN_ATTEMPT = (
    ROOT
    / "reports/taiji_n1_plan137_neighbor_distance_context_interference_diagnostic_attempt_20261019.json"
)
TEST = (
    ROOT / "tests/taiji_native/test_n1_plan137_neighbor_distance_context_interference_contract.py"
)

SEED = 119
TEMPLATE_COUNT = 8
TEMPLATES_PER_DIRECTION = 2
LENGTHS = (2, 8, 16)
DISTANCES = (1, 2)
ITEMS_PER_TEMPLATE = 5
ITEMS = TEMPLATE_COUNT * ITEMS_PER_TEMPLATE
ENTITY_LENGTH = v14_builder.ENTITY_LENGTH
DIRECTIONS = tuple(v14_builder.DIRECTIONS)
TURN_WORDS = ("left", "right")
CPU_LIMIT_SECONDS = 600.0
EXPECTED_EXCLUDED_ENTITIES = 8778
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
            raise ValueError(f"N1-137冻结来源哈希改变：{path}")


def _prior_sources() -> tuple[set[str], set[str], dict[str, str], dict, list[dict]]:
    if sha256(PLAN) != PLAN_SHA256:
        raise ValueError("PLAN-N1-137已批准执行版本SHA-256不匹配")
    if sha256(PLAN) == APPROVED_DRAFT_SHA256:
        raise ValueError("PLAN-N1-137尚未记录用户批准状态")

    readiness = n136_builder._verify_readiness()
    prior_manifest = read_json(N136_SOURCE_EXCLUSIONS)
    prior_face = read_json(N136_FACE)
    prior_face_checks = read_json(N136_FACE_CHECKS)
    prior_report = read_json(N136_RUN_REPORT)
    if (
        prior_manifest.get("entity_count") != 8658
        or prior_manifest.get("word_value_count") != 6288
        or len(prior_manifest.get("excluded_entities", [])) != 8658
        or len(prior_manifest.get("excluded_word_values", [])) != 6288
        or prior_face.get("schema") != "n1-transition-target-position-context-order-diagnostic-v1"
        or prior_face.get("seed") != 118
        or prior_face.get("items") != 40
        or len(prior_face.get("episodes", [])) != 40
        or prior_face_checks.get("accepted") is not True
        or prior_face_checks.get("face_sha256") != sha256(N136_FACE)
        or prior_report.get("complete") is not True
        or prior_report.get("diagnostic_queries") != 480
        or prior_report.get("endpoint_count") != 12
        or prior_report.get("parameter_updates") != 0
        or prior_report.get("dev_queries") != 0
        or prior_report.get("source_hashes_unchanged") is not True
        or prior_report.get("can_promote") is not False
    ):
        raise ValueError("N1-136结案报告、seed118面或逐题查询证据与冻结状态不符")

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
        raise ValueError("v18实体、答案值或N1-136来源排除关系不符")
    excluded_names.update(face_names)
    if (
        len(excluded_names) != EXPECTED_EXCLUDED_ENTITIES
        or len(excluded_words) != EXPECTED_EXCLUDED_WORDS
    ):
        raise ValueError(
            "N1-137来源排除并集改变："
            f"entities={len(excluded_names)}, words={len(excluded_words)}"
        )

    source_hashes = dict(prior_report.get("source_hashes", {}))
    for path in (
        N136_SOURCE_EXCLUSIONS,
        N136_READINESS,
        N136_FACE,
        N136_FACE_ATTEMPT,
        N136_FACE_CHECKS,
        N136_RUN_REPORT,
        N136_RUN_ATTEMPT,
        N136_PLAN,
        N136_CLOSEOUT,
        PLAN,
        Path(__file__).resolve(),
        ROOT / "scripts/training/eval_taiji_n1_v18_target_position_context_order.py",
        TEST,
    ):
        _add_source(source_hashes, path)

    endpoints = n133_builder._endpoint_specs()
    restored_by_key = {
        (row.get("source"), row.get("seed"), row.get("arm")): row
        for row in readiness.get("endpoint_restores", [])
    }
    if len(endpoints) != 12 or len(restored_by_key) != 12:
        raise ValueError("PLAN-N1-137必须绑定N1-136的12个冻结endpoint")
    for endpoint in endpoints:
        key = (endpoint["source"], endpoint["seed"], endpoint["arm"])
        restored = restored_by_key.get(key)
        if (
            restored is None
            or restored.get("restore_matches") is not True
            or restored.get("sha256") != endpoint["sha256"]
            or Path(restored.get("path", "")).resolve() != Path(endpoint["path"]).resolve()
        ):
            raise ValueError(f"N1-136完整bundle回执与冻结endpoint不符：{key}")
        _add_source(source_hashes, Path(endpoint["path"]), endpoint["sha256"])

    for ledger_path in sorted((ROOT / "plans/manifests").glob("n1_hard_neighbor_*.json")):
        ledger = read_json(ledger_path)
        _add_source(source_hashes, ledger_path)
        if ledger.get("seed") == SEED:
            raise ValueError(f"seed119已在硬邻居账本中使用：{ledger_path}")
    _verify_source_hashes(source_hashes)
    return (
        excluded_names,
        excluded_words,
        source_hashes,
        {
            "n136_excluded_entities": 8658,
            "n136_excluded_words": 6288,
            "v18_face_entities": len(face_names),
            "v18_face_answers": len(face_answers),
            "excluded_entities": len(excluded_names),
            "excluded_words": len(excluded_words),
            "source_hash_count": len(source_hashes),
            "endpoint_count": len(endpoints),
        },
        endpoints,
    )


def _confusable_at_distance(
    subject: str,
    distance: int,
    used: set[str],
    rng: random.Random,
) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    for _ in range(5000):
        chars = list(subject)
        positions = rng.sample(range(ENTITY_LENGTH), distance)
        for position in positions:
            chars[position] = rng.choice(letters.replace(chars[position], ""))
        candidate = "".join(chars)
        if candidate not in used:
            used.add(candidate)
            return candidate
    raise ValueError(f"无法生成d={distance}邻近实体")


def _make_episode(
    *, ordinal: int, template: dict, context_length: int, distance: int | None
) -> dict:
    if distance is None:
        if context_length != 2:
            raise ValueError("共同基线只允许使用2行上下文")
        distractor_entities = []
        filler_rows = []
        condition = "baseline"
    else:
        distractor_entities = template["distractor_sets"][str(distance)]["entities"]
        filler_rows = template["distractor_sets"][str(distance)]["observations"]
        condition = f"d{distance}"
    filler_count = context_length - 2
    return {
        "id": f"v19-trans-{ordinal:03d}",
        "split": "holdout",
        "subset": "diagnostic",
        "kind": "transition",
        "kind_ordinal": ordinal,
        "template_id": template["template_id"],
        "context_length": context_length,
        "neighbor_distance": distance,
        "condition": condition,
        "answer_direction": template["answer_direction"],
        "entities": [template["subject"], *distractor_entities[:filler_count]],
        "observations": [*template["target_rows"], *filler_rows[:filler_count]],
        "question": template["question"],
        "answer": template["answer_direction"],
        "target_fact_indices": [0, 1],
        "target_start_index": 0,
        "start_direction": template["start_direction"],
        "turn_word": template["turn_word"],
    }


def build_v19_face(excluded_names: set[str], excluded_words: set[str]) -> dict:
    """构造8模板×(2行基线+d1/d2各自8/16行)的40题面。"""
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
            distractor_sets = {}
            for distance in DISTANCES:
                names = [
                    (
                        v9._confusable(subject, used_names, rng)
                        if distance == 1
                        else _confusable_at_distance(subject, distance, used_names, rng)
                    )
                    for _ in range(max(LENGTHS) - 2)
                ]
                observations = [
                    f"{name} faces {DIRECTIONS[(direction_index + offset) % len(DIRECTIONS)]}."
                    for offset, name in enumerate(names)
                ]
                distractor_sets[str(distance)] = {
                    "distance": distance,
                    "entities": names,
                    "observations": observations,
                }
            template = {
                "template_id": f"v19-{answer}-{turn_word}",
                "answer_direction": answer,
                "turn_word": turn_word,
                "start_direction": start_direction,
                "subject": subject,
                "question": f"direction of {subject}?",
                "target_rows": [
                    f"{subject} faces {start_direction}.",
                    f"{subject} turns {turn_word}.",
                ],
                "distractor_sets": distractor_sets,
                "entities": [
                    subject,
                    *distractor_sets["1"]["entities"],
                    *distractor_sets["2"]["entities"],
                ],
            }
            templates.append(template)
            for context_length, distance in (
                (2, None),
                (8, 1),
                (8, 2),
                (16, 1),
                (16, 2),
            ):
                episodes.append(
                    _make_episode(
                        ordinal=len(episodes),
                        template=template,
                        context_length=context_length,
                        distance=distance,
                    )
                )

    ids = [item["id"] for item in episodes]
    return {
        "schema": "n1-transition-neighbor-distance-context-interference-diagnostic-v1",
        "seed": SEED,
        "items": ITEMS,
        "template_count": TEMPLATE_COUNT,
        "items_per_template": ITEMS_PER_TEMPLATE,
        "context_lengths": list(LENGTHS),
        "neighbor_distances": list(DISTANCES),
        "entity_length": ENTITY_LENGTH,
        "answer_directions": list(DIRECTIONS),
        "turn_words": list(TURN_WORDS),
        "subset_sizes": {"diagnostic": ITEMS},
        "subsets": {"diagnostic": ids},
        "matched_templates": templates,
        "exclusion_sources": {},
        "episodes": episodes,
    }


def audit_v19_face(face: dict, *, excluded_names: set[str], excluded_words: set[str]) -> dict:
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
            (8, "d1"): 8,
            (8, "d2"): 8,
            (16, "d1"): 8,
            (16, "d2"): 8,
        }
    )
    observed_conditions = Counter(
        (item.get("context_length"), item.get("condition")) for item in episodes
    )
    template_checks = len(templates) == TEMPLATE_COUNT and len(by_template) == TEMPLATE_COUNT
    content_checks = True
    distance_checks = True
    condition_distance_checks = True
    target_checks = True
    for template in templates:
        template_id = template["template_id"]
        selected = by_template.get(template_id, [])
        keyed = {(item["context_length"], item["condition"]): item for item in selected}
        if len(selected) != ITEMS_PER_TEMPLATE or len(keyed) != ITEMS_PER_TEMPLATE:
            template_checks = False
            content_checks = False
            continue
        baseline = keyed[(2, "baseline")]
        eight_d1 = keyed[(8, "d1")]
        eight_d2 = keyed[(8, "d2")]
        sixteen_d1 = keyed[(16, "d1")]
        sixteen_d2 = keyed[(16, "d2")]
        template_checks &= all(
            item.get("entities", [None])[0] == template["subject"]
            and item.get("answer") == template["answer_direction"]
            and item.get("answer_direction") == template["answer_direction"]
            and item.get("turn_word") == template["turn_word"]
            and item.get("start_direction") == template["start_direction"]
            and item.get("question") == template["question"]
            and item.get("target_fact_indices") == [0, 1]
            and item.get("target_start_index") == 0
            for item in selected
        )
        content_checks &= (
            baseline["observations"] == template["target_rows"]
            and eight_d1["observations"][:2] == template["target_rows"]
            and eight_d2["observations"][:2] == template["target_rows"]
            and sixteen_d1["observations"][:2] == template["target_rows"]
            and sixteen_d2["observations"][:2] == template["target_rows"]
            and eight_d1["observations"][2:] == template["distractor_sets"]["1"]["observations"][:6]
            and sixteen_d1["observations"][2:]
            == template["distractor_sets"]["1"]["observations"][:14]
            and eight_d2["observations"][2:] == template["distractor_sets"]["2"]["observations"][:6]
            and sixteen_d2["observations"][2:]
            == template["distractor_sets"]["2"]["observations"][:14]
            and eight_d1["observations"][2:] == sixteen_d1["observations"][2:8]
            and eight_d2["observations"][2:] == sixteen_d2["observations"][2:8]
        )
        distance_checks &= all(
            sum(left != right for left, right in zip(template["subject"], name, strict=True))
            == distance
            for distance in DISTANCES
            for name in template["distractor_sets"][str(distance)]["entities"]
        )
        condition_distance_checks &= (
            baseline.get("neighbor_distance") is None
            and baseline.get("condition") == "baseline"
            and eight_d1.get("neighbor_distance") == 1
            and eight_d1.get("condition") == "d1"
            and eight_d2.get("neighbor_distance") == 2
            and eight_d2.get("condition") == "d2"
            and sixteen_d1.get("neighbor_distance") == 1
            and sixteen_d1.get("condition") == "d1"
            and sixteen_d2.get("neighbor_distance") == 2
            and sixteen_d2.get("condition") == "d2"
        )
        target_checks &= all(
            item["observations"][0] == template["target_rows"][0]
            and item["observations"][1] == template["target_rows"][1]
            and DIRECTIONS[
                (
                    DIRECTIONS.index(item["start_direction"])
                    + (1 if item["turn_word"] == "right" else -1)
                )
                % 4
            ]
            == item["answer"]
            for item in selected
        )

    template_entity_sets = [set(item["entities"]) for item in templates]
    cross_template_disjoint = all(
        not left & right
        for index, left in enumerate(template_entity_sets)
        for right in template_entity_sets[index + 1 :]
    )
    checks = {
        "schema_seed_size_and_single_subset": face.get("schema")
        == "n1-transition-neighbor-distance-context-interference-diagnostic-v1"
        and face.get("seed") == SEED
        and face.get("items") == ITEMS
        and len(episodes) == ITEMS
        and ids == [f"v19-trans-{index:03d}" for index in range(ITEMS)]
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
        "same_target_and_nested_distance_specific_content": content_checks,
        "exact_hamming_distances": distance_checks,
        "condition_distance_metadata": condition_distance_checks,
        "unique_new_entities_and_source_disjoint": len(names) == 232
        and len({name for template in templates for name in template["entities"]}) == 232
        and cross_template_disjoint
        and not names & excluded_names,
        "direction_answers_preexcluded_and_no_material_words": answers == set(DIRECTIONS)
        and answers <= excluded_words
        and materials == set(),
        "target_facts_and_answer_direction": target_checks,
    }
    return {
        "schema": "n1-plan137-neighbor-distance-face-checks-v1",
        "accepted": all(checks.values()),
        "checks": checks,
        "condition_counts": {
            f"{length}/{condition}": count
            for (length, condition), count in sorted(observed_conditions.items())
        },
        "template_count": len(templates),
        "entity_count": len(names),
        "observation_material_count": len(materials),
        "answer_value_count": len(answers),
        "summary_zh": "seed119题面含8个方向×转向模板；固定同一目标事实和目标位置，对照d=1/d=2近邻干扰及2/8/16行上下文。",
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
        raise FileExistsError("PLAN-N1-137 write-once输出路径已占用；不覆盖、不重试")
    started = time.process_time()
    write_new(
        READINESS_ATTEMPT,
        {
            "schema": "n1-plan137-readiness-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "approved_draft_sha256": APPROVED_DRAFT_SHA256,
            "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "new_face_generated": False,
            "model_queries": 0,
            "parameter_updates": 0,
            "说明_zh": "只核验N1-136来源、seed119账本和12个完整bundle；本阶段不生成题面、不查询模型。",
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
                "schema": "n1-plan137-neighbor-distance-source-exclusions-v1",
                "status": "ready_for_approved_execution",
                "plan_sha256": PLAN_SHA256,
                "approved_draft_sha256": APPROVED_DRAFT_SHA256,
                "excluded_entities": inventory["excluded_entities"],
                "excluded_word_values": inventory["excluded_word_values"],
                "entity_count": inventory["counts"]["excluded_entities"],
                "word_value_count": inventory["counts"]["excluded_words"],
                "provenance": inventory["counts"],
                "source_hashes": source_hashes,
                "summary_zh": "继承N1-136对v3训练及v5-v17的完整排除，并新增排除已消费v18面120个实体；方向答案词已在既有排除集中。",
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
            errors.append("12个bundle恢复、来源并集、seed119账本或CPU门未通过")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    cpu = round(time.process_time() - started, 6)
    report = {
        "schema": "n1-plan137-neighbor-distance-readiness-v1",
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
            "12个冻结endpoint完整bundle恢复一致；v3训练至v18已消费来源排除完成；本阶段0模型查询、0参数更新、未生成题面。"
            if not errors
            else "N1-137 readiness停止；保留失败回执，不生成题面、不查询、不更新。"
        ),
        "errors": errors,
    }
    report["readiness_attempt_sha256"] = sha256(READINESS_ATTEMPT)
    write_new(READINESS, report)
    return report


def _verify_readiness() -> dict:
    if not READINESS.is_file() or not SOURCE_EXCLUSIONS.is_file():
        raise FileNotFoundError("必须先完成N1-137零查询readiness")
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
        raise ValueError("N1-137 readiness或source exclusions失效")
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
            "schema": "n1-plan137-neighbor-distance-face-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
            "seed": SEED,
            "planned_items": ITEMS,
            "planned_template_count": TEMPLATE_COUNT,
            "planned_queries": ITEMS * 12,
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
            "说明_zh": "一次性构造seed119近邻距离诊断面；若合同失败不重抽、不重构、不重试。",
        },
    )


def build_face_after_approval(*, owner_approved: bool) -> dict:
    if not owner_approved:
        raise PermissionError("PLAN-N1-137新测面构造需要owner授权")
    readiness = _verify_readiness()
    if any(path.exists() for path in (FACE, FACE_ATTEMPT, FACE_CHECKS, RUN_REPORT, RUN_ATTEMPT)):
        raise FileExistsError("PLAN-N1-137题面/查询输出路径已占用；不覆盖、不重试")
    manifest = read_json(SOURCE_EXCLUSIONS)
    excluded_names = set(manifest["excluded_entities"])
    excluded_words = set(manifest["excluded_word_values"])
    _write_face_attempt(readiness)
    started = time.process_time()
    face = build_v19_face(excluded_names, excluded_words)
    face["exclusion_sources"] = {
        "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
        "source_hashes": manifest["source_hashes"],
    }
    checks = audit_v19_face(face, excluded_names=excluded_names, excluded_words=excluded_words)
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
        "errors": [] if checks["accepted"] else ["v19诊断面合同未通过"],
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
