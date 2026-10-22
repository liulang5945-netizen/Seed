"""构造并校验PLAN-N1-142的v21新确认面及完整来源排除并集。"""

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

from scripts.training import build_taiji_n1_v20_neighbor_distance_dose_curve as v20
from scripts.training import (
    run_taiji_n1_plan139_context_robust_matched_continuation_calibration as n139,
)
from scripts.training import (
    run_taiji_n1_plan140_baseline_retention_balanced_matched_continuation as n140,
)
from scripts.training import (
    run_taiji_n1_plan141_interleaved_context_schedule_matched_continuation as n141,
)
from scripts.training import seal_ast

PLAN = ROOT / "plans/reference/PLAN-N1-142_v21_fresh_confirmation_after_n141_draft_20261019.md"
PLAN_SHA256 = "6bed0e31a250b01b4a4664f34e904749788dd4d3e07089fb9b84320fa2afad54"
SOURCE_PRECHECK = ROOT / "reports/taiji_n1_v21_source_union_precheck_20261019.json"
SOURCE_EXCLUSIONS = ROOT / "plans/manifests/n1_v21_after_n141_source_exclusions_20261019.json"
FACE = ROOT / "plans/manifests/n1_v21_confirmation_seed121_20261019.json"
FACE_ATTEMPT = ROOT / "plans/manifests/n1_v21_confirmation_seed121_20261019_attempt.json"
FACE_CHECKS = ROOT / "reports/taiji_n1_v21_confirmation_face_checks_20261019.json"
PREPARED = ROOT / "reports/taiji_n1_v21_confirmation_face_readiness_20261019.json"

SEED = 121
ITEMS = 160
ITEMS_PER_KIND = 40
ITEMS_PER_SUBSET_KIND = 20
OBSERVATIONS = 16
ENTITY_LENGTH = 10
DISTANCE = 1
KINDS = ("lookup", "revision", "transition", "unknown")
DIRECTIONS = tuple(v20.DIRECTIONS)
LETTERS = "abcdefghijklmnopqrstuvwxyz"
ABSTENTION = "unknown"
SUBSETS = ("sacrifice", "confirm")
LOOKUP_POSITIONS = (0, 4, 8, 12, 15)
PAIR_POSITIONS = (0, 4, 7, 11, 14)
EXPECTED_BASE_ENTITY_COUNT = 9_354
EXPECTED_SYNTHETIC_FILLER_COUNTS = {"n139": 12_096, "n140": 7_776, "n141": 7_776}
EXPECTED_SYNTHETIC_UNION_COUNT = 17_328
EXPECTED_ENTITY_EXCLUSION_COUNT = 26_682
EXPECTED_WORD_EXCLUSION_COUNT = 6_288


class Refuse(RuntimeError):
    """一项冻结前置失败；不重抽或覆盖write-once输出。"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_strings(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _merge_hashes(target: dict[str, str], incoming: dict[str, str]) -> None:
    for path, digest in incoming.items():
        resolved = str(Path(path).resolve())
        prior = target.get(resolved)
        if prior is None or prior == digest:
            # 不可取证的记录值（历史垃圾）不入表——终验三通道必红，入表只会前移爆点
            if seal_ast.verify_hash_dict({resolved: digest})[0]:
                target[resolved] = digest
            continue
        # S4 重冻结后同文件可有多代合法历史字节：新值可取证取新值；新值不可取证
        # 但先值可取证则弃用新值；双垃圾且不同值同样弃用新值（终验三通道兜底）
        if seal_ast.verify_hash_dict({resolved: digest})[0]:
            target[resolved] = digest


def _verify_hashes(source_hashes: dict[str, str]) -> bool:
    """S3 切换: 经 seal_ast 三通道校验（在位字节 / 在位 AST / git 历史取证）。"""
    return seal_ast.verify_hash_dict(source_hashes)[0]


def _filler_names(bound: dict) -> set[str]:
    names: set[str] = set()
    for variant in bound["context_variants"].values():
        for distance_data in variant["by_distance"].values():
            names.update(str(name) for name in distance_data["entities"])
    return names


def _collect_source_union() -> tuple[set[str], set[str], dict[str, str], dict]:
    n138_manifest, base_names, excluded_words = n141._validate_n138_exclusions()
    require(len(n138_manifest.get("excluded_entities", [])) == 9_010, "N1-138实体排除底账改变")
    require(len(excluded_words) == EXPECTED_WORD_EXCLUSION_COUNT, "N1-138词值排除底账改变")
    require(len(base_names) == EXPECTED_BASE_ENTITY_COUNT, "N1-138+v20实体并集数量改变")

    bound139 = n139._build_static_context()
    bound140 = n140._build_static_context()
    bound141 = n141._build_static_context()
    filler_sets = {
        "n139": _filler_names(bound139),
        "n140": _filler_names(bound140),
        "n141": _filler_names(bound141),
    }
    for name, expected_count in EXPECTED_SYNTHETIC_FILLER_COUNTS.items():
        require(len(filler_sets[name]) == expected_count, f"{name}合成训练实体数量改变")
        require(not (filler_sets[name] & base_names), f"{name}合成实体与旧排除集冲突")

    all_fillers = set().union(*filler_sets.values())
    require(len(all_fillers) == EXPECTED_SYNTHETIC_UNION_COUNT, "三批训练合成实体并集数量改变")
    excluded_entities = base_names | all_fillers
    require(len(excluded_entities) == EXPECTED_ENTITY_EXCLUSION_COUNT, "v21实体排除并集数量改变")
    excluded_words = set(excluded_words)
    require(set(DIRECTIONS) <= excluded_words, "方向答案必须保留在旧词值排除集")

    source_hashes: dict[str, str] = {}
    for bound in (bound139, bound140, bound141):
        _merge_hashes(source_hashes, bound["source_hashes"])
    n139_result = read_json(n139.RESULT)
    n140_result = read_json(n140.RESULT)
    n141_result = read_json(n141.RESULT)
    for label, result in (("n139", n139_result), ("n140", n140_result), ("n141", n141_result)):
        require(result.get("complete") is True, f"{label}训练结果未完整结案")
        require(result.get("source_hashes_unchanged") is True, f"{label}来源守卫失败")
        require(
            result.get("dev_queries") == 0 and result.get("holdout_queries") == 0,
            f"{label}越过train-only边界",
        )
        _merge_hashes(source_hashes, result.get("source_hashes") or {})
        _merge_hashes(source_hashes, result.get("new_checkpoint_hashes") or {})

    itemwise141 = read_json(
        ROOT / "reports/taiji_n1_plan141_itemwise_retention_context_attribution_20261019.json"
    )
    require(itemwise141.get("complete") is True, "N1-141逐题归因报告不完整")
    require(itemwise141.get("source_hashes_unchanged") is True, "N1-141逐题归因来源改变")
    _merge_hashes(source_hashes, itemwise141.get("source_hashes", {}))
    for path in (
        PLAN,
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("eval_taiji_n1_v21_confirmation_after_n141.py"),
        ROOT / "tests/taiji_native/test_n1_plan142_v21_face_builder_contract.py",
        ROOT / "tests/taiji_native/test_n1_plan142_v21_confirmation_runner_contract.py",
        n139.PLAN,
        n139.RESULT,
        ROOT / "plans/reference/PLAN-N1-139_execution_closeout_20261019.md",
        n140.PLAN,
        n140.RESULT,
        ROOT / "plans/reference/PLAN-N1-140_execution_closeout_20261019.md",
        n141.PLAN,
        n141.RESULT,
        ROOT / "plans/reference/PLAN-N1-141_execution_closeout_20261019.md",
        ROOT / "reports/taiji_n1_plan141_itemwise_retention_context_attribution_20261019.json",
        v20.SOURCE_EXCLUSIONS,
        v20.FACE,
        v20.FACE_CHECKS,
        v20.RUN_REPORT,
    ):
        require(Path(path).is_file(), f"冻结来源缺失：{path}")
        _merge_hashes(source_hashes, {str(Path(path).resolve()): sha256(Path(path))})

    require(_verify_hashes(source_hashes), "至少一项冻结来源哈希已变化")
    summary = {
        "base_manifest_entity_count": len(n138_manifest["excluded_entities"]),
        "n138_face_entity_count": len(base_names) - len(n138_manifest["excluded_entities"]),
        "base_entity_union_count": len(base_names),
        "synthetic_filler_counts": {key: len(value) for key, value in filler_sets.items()},
        "synthetic_filler_unique_union_count": len(all_fillers),
        "synthetic_filler_pairwise_overlap": {
            "n139_n140": len(filler_sets["n139"] & filler_sets["n140"]),
            "n139_n141": len(filler_sets["n139"] & filler_sets["n141"]),
            "n140_n141": len(filler_sets["n140"] & filler_sets["n141"]),
        },
        "entity_exclusion_count": len(excluded_entities),
        "entity_exclusion_sha256": digest_strings(excluded_entities),
        "word_exclusion_count": len(excluded_words),
        "word_exclusion_sha256": digest_strings(excluded_words),
        "source_hash_count": len(source_hashes),
    }
    return excluded_entities, excluded_words, source_hashes, summary


def _novel(length: int, pool: set[str], rng: random.Random) -> str:
    for _ in range(100_000):
        value = "".join(rng.choices(LETTERS, k=length))
        if value not in pool:
            pool.add(value)
            return value
    raise Refuse("unique string pool exhausted")


def _confusable(subject: str, pool: set[str], rng: random.Random) -> str:
    for _ in range(100_000):
        chars = list(subject)
        position = rng.randrange(len(chars))
        chars[position] = rng.choice(LETTERS.replace(chars[position], ""))
        value = "".join(chars)
        if value not in pool:
            pool.add(value)
            return value
    raise Refuse("unique d=1 neighbor pool exhausted")


def build_face(excluded_entities: set[str], excluded_words: set[str], seed: int = SEED) -> dict:
    """纯内存、可复现地生成160题v21；不写文件、不查询模型。"""
    rng = random.Random(seed)
    used_entities = set(excluded_entities)
    used_words = set(excluded_words) | set(DIRECTIONS) | {ABSTENTION}
    episodes = []
    answer_lengths = Counter()
    starts = Counter()
    direction_counts = Counter()
    turn_counts = Counter()

    for subset in SUBSETS:
        for kind in KINDS:
            for local in range(ITEMS_PER_SUBSET_KIND):
                index = len(episodes)
                subject = _novel(ENTITY_LENGTH, used_entities, rng)
                answer_length = 6 if local % 2 == 0 else 7
                answer_lengths[(subset, kind, answer_length)] += 1
                start = (
                    LOOKUP_POSITIONS[local % len(LOOKUP_POSITIONS)]
                    if kind == "lookup"
                    else (
                        PAIR_POSITIONS[local % len(PAIR_POSITIONS)]
                        if kind in ("revision", "transition")
                        else None
                    )
                )
                if start is not None:
                    starts[(subset, kind, start)] += 1

                target_rows = (
                    1 if kind == "lookup" else 2 if kind in ("revision", "transition") else 0
                )
                distractors = [
                    _confusable(subject, used_entities, rng)
                    for _ in range(OBSERVATIONS - target_rows)
                ]
                observations: list[str | None] = [None] * OBSERVATIONS
                target_indices: list[int] = []
                answer_direction = None
                turn_template = None
                if kind == "lookup":
                    answer = _novel(answer_length, used_words, rng)
                    observations[start] = f"{subject} uses {answer}."
                    target_indices = [start]
                    question = f"material of {subject}?"
                elif kind == "revision":
                    stale_length = 6 if local % 2 == 0 else 7
                    stale = _novel(stale_length, used_words, rng)
                    answer = _novel(answer_length, used_words, rng)
                    observations[start] = f"{subject} uses {stale}."
                    observations[start + 1] = f"{subject} now uses {answer}."
                    target_indices = [start, start + 1]
                    question = f"material of {subject}?"
                elif kind == "transition":
                    answer_direction = DIRECTIONS[local % len(DIRECTIONS)]
                    turn_parity = (local // len(DIRECTIONS) + local % len(DIRECTIONS)) % 2
                    turn_template = "right" if turn_parity == 0 else "left"
                    delta = 1 if turn_template == "right" else -1
                    initial = DIRECTIONS[
                        (DIRECTIONS.index(answer_direction) - delta) % len(DIRECTIONS)
                    ]
                    answer = answer_direction
                    observations[start] = f"{subject} faces {initial}."
                    observations[start + 1] = f"{subject} turns {turn_template}."
                    target_indices = [start, start + 1]
                    direction_counts[(subset, answer_direction)] += 1
                    turn_counts[(subset, turn_template)] += 1
                    question = f"direction of {subject}?"
                else:
                    answer = ABSTENTION
                    question = f"material of {subject}?"

                filler_indices = [
                    position for position, row in enumerate(observations) if row is None
                ]
                require(
                    len(filler_indices) == len(distractors), f"{kind} filler slot count differs"
                )
                filler_rows = []
                for number, name in enumerate(distractors):
                    if kind == "transition":
                        direction = DIRECTIONS[(local + number + index) % len(DIRECTIONS)]
                        row = f"{name} faces {direction}."
                    else:
                        material_length = 6 if (local + number + index) % 2 == 0 else 7
                        material = _novel(material_length, used_words, rng)
                        row = f"{name} uses {material}."
                    filler_rows.append(row)
                for position, row in zip(filler_indices, filler_rows, strict=True):
                    observations[position] = row

                require(
                    all(row is not None for row in observations), f"{kind} context has empty rows"
                )
                entities = [subject, *distractors]
                require(len(entities) == len(set(entities)), f"{kind} episode repeats an entity")
                require(
                    all(len(name) == ENTITY_LENGTH for name in entities),
                    f"{kind} entity length differs",
                )
                require(
                    all(_hamming(subject, name) == DISTANCE for name in distractors),
                    f"{kind} d=1 differs",
                )
                episodes.append(
                    {
                        "id": f"v21-{subset}-{kind}-{local:03d}",
                        "split": "holdout",
                        "subset": subset,
                        "kind": kind,
                        "entities": entities,
                        "observations": [str(row) for row in observations],
                        "question": question,
                        "answer": answer,
                        "target_fact_indices": target_indices,
                        "answer_direction": answer_direction,
                        "turn_template": turn_template,
                        "target_start_index": start,
                        "answer_length": len(answer) if kind in ("lookup", "revision") else None,
                    }
                )

    return {
        "schema": "n1-v21-confirmation-face-v1",
        "seed": seed,
        "items": ITEMS,
        "items_per_kind": ITEMS_PER_KIND,
        "observations_per_item": OBSERVATIONS,
        "entity_length": ENTITY_LENGTH,
        "name_neighbor_distance": DISTANCE,
        "answer_lengths": [6, 7],
        "target_start_positions": {
            "lookup": list(LOOKUP_POSITIONS),
            "revision": list(PAIR_POSITIONS),
            "transition": list(PAIR_POSITIONS),
            "unknown": [],
        },
        "episodes": episodes,
        "shape_counts": {
            "items_by_subset_and_kind": {
                f"{subset}/{kind}": sum(
                    item["subset"] == subset and item["kind"] == kind for item in episodes
                )
                for subset in SUBSETS
                for kind in KINDS
            },
            "answer_length_counts": {
                f"{subset}/{kind}/{length}": count
                for (subset, kind, length), count in sorted(answer_lengths.items())
            },
            "target_start_counts": {
                f"{subset}/{kind}/{position}": count
                for (subset, kind, position), count in sorted(starts.items())
            },
            "direction_counts": {
                f"{subset}/{direction}": count
                for (subset, direction), count in sorted(direction_counts.items())
            },
            "turn_counts": {
                f"{subset}/{turn}": count for (subset, turn), count in sorted(turn_counts.items())
            },
            "new_entity_count": len(used_entities - excluded_entities),
            "new_word_count": len(used_words - excluded_words - set(DIRECTIONS) - {ABSTENTION}),
        },
    }


def _hamming(first: str, second: str) -> int:
    if len(first) != len(second):
        return max(len(first), len(second))
    return sum(a != b for a, b in zip(first, second, strict=True))


def validate_face(face: dict, excluded_entities: set[str], excluded_words: set[str]) -> dict:
    episodes = face.get("episodes", [])
    require(face.get("schema") == "n1-v21-confirmation-face-v1", "v21 schema mismatch")
    require(face.get("seed") == SEED and len(episodes) == ITEMS, "v21 seed/item count mismatch")
    require(len({item.get("id") for item in episodes}) == ITEMS, "v21 IDs are not unique")
    require(all(item.get("split") == "holdout" for item in episodes), "v21 split must be holdout")
    require(
        all(len(item.get("observations", [])) == OBSERVATIONS for item in episodes),
        "v21 observation counts differ",
    )
    require(
        all(
            len(item.get("entities", []))
            == 1
            + OBSERVATIONS
            - (
                1
                if item["kind"] == "lookup"
                else 2 if item["kind"] in ("revision", "transition") else 0
            )
            for item in episodes
        ),
        "v21 entity shape differs",
    )
    require(
        all(len(set(item["entities"])) == len(item["entities"]) for item in episodes),
        "duplicate entity within an episode",
    )
    all_names = {name for item in episodes for name in item["entities"]}
    require(
        not (all_names & excluded_entities), "v21 entity overlaps consumed/source exclusion union"
    )
    require(
        len(all_names) == sum(len(item["entities"]) for item in episodes),
        "v21 entity repeats across episodes",
    )
    material_values = {
        row.removesuffix(".").split()[-1]
        for item in episodes
        for row in item["observations"]
        if " uses " in row
    } | {item["answer"] for item in episodes if item["kind"] in ("lookup", "revision")}
    require(
        not (material_values & excluded_words),
        "v21 material/answer overlaps consumed word exclusions",
    )
    require(
        all(len(value) in (6, 7) for value in material_values), "v21 materials must be 6/7 chars"
    )
    require(
        all(
            _hamming(item["entities"][0], name) == DISTANCE
            for item in episodes
            for name in item["entities"][1:]
        ),
        "v21 Hamming-distance contract failed",
    )
    counts = Counter((item["subset"], item["kind"]) for item in episodes)
    require(
        all(
            counts[(subset, kind)] == ITEMS_PER_SUBSET_KIND for subset in SUBSETS for kind in KINDS
        ),
        "v21 subset balance differs",
    )
    directions = Counter(
        (item["subset"], item["answer_direction"])
        for item in episodes
        if item["kind"] == "transition"
    )
    turns = Counter(
        (item["subset"], item["turn_template"]) for item in episodes if item["kind"] == "transition"
    )
    require(
        all(directions[(subset, direction)] == 5 for subset in SUBSETS for direction in DIRECTIONS),
        "v21 direction balance differs",
    )
    require(
        all(turns[(subset, turn)] == 10 for subset in SUBSETS for turn in ("left", "right")),
        "v21 turn balance differs",
    )
    return {
        "accepted": True,
        "items": len(episodes),
        "items_by_subset_and_kind": {f"{s}/{k}": counts[(s, k)] for s in SUBSETS for k in KINDS},
        "entity_count": len(all_names),
        "material_word_count": len(material_values),
        "face_seed": face["seed"],
        "schema": face["schema"],
    }


def prepare_face() -> dict:
    if not PLAN.is_file() or sha256(PLAN) != PLAN_SHA256:
        raise Refuse("PLAN-N1-142 frozen SHA-256 mismatch")
    output_paths = (SOURCE_PRECHECK, SOURCE_EXCLUSIONS, FACE, FACE_ATTEMPT, FACE_CHECKS, PREPARED)
    if any(path.exists() for path in output_paths):
        raise FileExistsError("PLAN-N1-142 face preparation is write-once")
    started = time.process_time()
    excluded_entities, excluded_words, source_hashes, union_summary = _collect_source_union()
    source_hashes[str(PLAN.resolve())] = sha256(PLAN)
    source_hashes[str(Path(__file__).resolve())] = sha256(Path(__file__))
    require(_verify_hashes(source_hashes), "source union changed during static preparation")
    precheck = {
        "schema": "n1-plan142-v21-source-union-precheck-v1",
        "status": "complete",
        "plan_sha256": PLAN_SHA256,
        "source_hashes": source_hashes,
        "source_hashes_unchanged": True,
        "source_union": union_summary,
        "new_face_generated": False,
        "model_queries": 0,
        "parameter_updates": 0,
        "cpu_seconds": round(time.process_time() - started, 6),
        "can_promote": False,
    }
    write_new(SOURCE_PRECHECK, precheck)
    manifest = {
        "schema": "n1-plan142-v21-source-exclusions-v1",
        "status": "ready_for_one_time_face_generation",
        "plan_sha256": PLAN_SHA256,
        "seed": SEED,
        "entity_count": len(excluded_entities),
        "word_value_count": len(excluded_words),
        "entity_exclusion_sha256": digest_strings(excluded_entities),
        "word_exclusion_sha256": digest_strings(excluded_words),
        "excluded_entities": sorted(excluded_entities),
        "excluded_word_values": sorted(excluded_words),
        "provenance": union_summary,
        "source_hashes": source_hashes,
        "can_promote": False,
    }
    write_new(SOURCE_EXCLUSIONS, manifest)
    write_new(
        FACE_ATTEMPT,
        {
            "schema": "n1-plan142-v21-face-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "source_union_sha256": sha256(SOURCE_EXCLUSIONS),
            "seed": SEED,
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
        },
    )
    face = build_face(excluded_entities, excluded_words)
    face["plan_sha256"] = PLAN_SHA256
    face["source_exclusion_sha256"] = sha256(SOURCE_EXCLUSIONS)
    face["source_exclusion_entity_sha256"] = manifest["entity_exclusion_sha256"]
    face["source_exclusion_word_sha256"] = manifest["word_exclusion_sha256"]
    face["source_hashes"] = source_hashes
    write_new(FACE, face)
    checks = validate_face(face, excluded_entities, excluded_words)
    checks.update(
        {
            "plan_sha256": PLAN_SHA256,
            "face_sha256": sha256(FACE),
            "source_exclusion_sha256": sha256(SOURCE_EXCLUSIONS),
            "source_hashes_unchanged": _verify_hashes(source_hashes),
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
        }
    )
    if checks["source_hashes_unchanged"] is not True:
        raise Refuse("source changed during v21 face construction")
    write_new(FACE_CHECKS, checks)
    prepared = {
        "schema": "n1-plan142-v21-face-readiness-v1",
        "status": "ready_for_checkpoint_readiness",
        "complete": True,
        "plan_sha256": PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "source_exclusion_sha256": sha256(SOURCE_EXCLUSIONS),
        "source_precheck_sha256": sha256(SOURCE_PRECHECK),
        "source_hashes": source_hashes,
        "source_hashes_unchanged": _verify_hashes(source_hashes),
        "model_queries": 0,
        "parameter_updates": 0,
        "cpu_seconds": round(time.process_time() - started, 6),
        "summary_zh": "v21新面构造及160题合同检查通过；没有加载模型、恢复checkpoint、训练或查询，可进入12端点checkpoint readiness。",
        "can_promote": False,
    }
    write_new(PREPARED, prepared)
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(description="PLAN-N1-142 v21 face builder and source audit")
    parser.add_argument("--prepare-face", action="store_true")
    args = parser.parse_args()
    if not args.prepare_face:
        parser.error("one-shot invocation requires --prepare-face")
    result = prepare_face()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
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
