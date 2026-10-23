"""Prepare or, after approval, build the unused PLAN-N1-121 v11 face."""

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

from scripts.training import (
    build_taiji_n1_hard_neighbor_training_schedule as train_data,
)  # noqa: E402
from scripts.training import build_taiji_n1_hard_neighbor_v9 as v9  # noqa: E402
from scripts.training import seal_ast  # noqa: E402

N100_REPORT = ROOT / "reports/taiji_n1_hard_neighbor_training_20261017.json"
N100_REPORT_SHA256 = "dad10ec72c605367f99da035bab07d522ce1793154915ebcc3c7f1022f411909"
N110_PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-110_length_balanced_training_and_v11_confirmation_draft_20261017.md"
)
N110_PLAN_SHA256 = "16d0cecb325af6463c107973fc0e741478140134a7642cd64c86e045c6069783"
N110_SCHEDULE = ROOT / "plans/manifests/n1_hard_neighbor_length_balanced_schedule_v1_20261018.json"
N110_SCHEDULE_SHA256 = "b2d637fcc1e096d40bba35bbc91bc66b0c27257cd74a4bcac5f5f488f55b6d4c"
N110_VALUE_MAP_SHA256 = "dba1d9eb8fec6d60e6b1ceca496bce3b3386b05d1539ad88696177db0b9e9ba8"
N110_VALUE_MAP_ENTRIES = 288
N121_PLAN = (
    ROOT / "plans/reference/PLAN-N1-121_v11_confirmation_after_n120_training_draft_20261019.md"
)
N121_PLAN_SHA256 = "5c2ac778e8e205e1574a2bd93d77dfc1c597490aced4db7a0b0f6d2efbbfefb8"
N121_SOURCE_CORRECTION = (
    ROOT / "plans/reference/PLAN-N1-121_source_exclusion_correction_20261019.md"
)
N121_SOURCE_CORRECTION_SHA256 = "896dad354e6c81b7a1157746fe4e4bc5dcc8fcd266a7707dc09f476d7bb7aba7"
N120_REPORT = ROOT / "reports/taiji_n1_expanded_normalized_rwl_length_balanced_20261019.json"
N120_REPORT_SHA256 = "873c237623fa7b97de0f4d8d95edaf4a5d8f9bf944c8af301468bd1d0f19ab2f"
N120_REVALIDATION = (
    ROOT
    / "reports/taiji_n1_expanded_normalized_rwl_length_balanced_20261019_receipt_revalidation.json"
)
N120_REVALIDATION_SHA256 = "22055b74784e7c35891098595928af238561405a52cbfb4b94eb87491ba7e03d"
N120_RESTORE_READINESS = ROOT / "reports/taiji_n1_v11_after_n120_restore_readiness_20261019.json"
N120_RESTORE_READINESS_SHA256 = "6212e0413153e2d22d7cde86cc8eb97414d0467ee52ac56cee1d04027ca5f2a3"
V9_FACE = ROOT / "plans/manifests/n1_hard_neighbor_v9_20261017.json"
V9_FACE_SHA256 = "d93dd6c6e0d98125ef1c91ed7ad8aec07a46f6621fb597dbffa6e889f29ac60f"
V9_FACE_CHECKS = ROOT / "reports/taiji_n1_hard_neighbor_v9_face_checks_20261017.json"
V9_FACE_CHECKS_SHA256 = "157422b6534c7b723f76c49ec589bf75600d2e0ed82687141711b90393a5beb8"
V9_RANGE = ROOT / "reports/taiji_n1_hard_neighbor_v9_range_check_20261017.json"
V9_RANGE_SHA256 = "ee229efcbe5c309eaab76705bc38ef5adb0f1d1e47571a459ea217f9af98576c"
V9_RANGE_ATTEMPT = V9_RANGE.with_name(V9_RANGE.stem + "_attempt.json")
V9_RANGE_ATTEMPT_SHA256 = "e830e01e1ec57e890aaea560aa6f3151aed132682d55be9caa2c6f75ce7f212b"
V9_CONFIRM = ROOT / "reports/taiji_n1_hard_neighbor_v9_confirmation_20261017.json"
V9_CONFIRM_SHA256 = "7b9b7e7cccf27a5e9ba6f9e002b1b4df09347fb7264b3827f9eb6cdab44a1802"
V9_CONFIRM_ATTEMPT = V9_CONFIRM.with_name(V9_CONFIRM.stem + "_attempt.json")
V9_CONFIRM_ATTEMPT_SHA256 = "78e8d9d13e90ba235688a4731385255682ee4075afb08ab94c13b172eda149a3"
V10_FACE = ROOT / "plans/manifests/n1_hard_neighbor_v10_factorial_diagnostic_20261017.json"
V10_FACE_SHA256 = "9b23a8b4ad1508ab0c2b100b9eb98b9c4dcf572b031bb0782b1267737173d5ca"
V10_FACE_CHECKS = (
    ROOT / "reports/taiji_n1_hard_neighbor_v10_factorial_diagnostic_checks_20261017.json"
)
V10_FACE_CHECKS_SHA256 = "db3b77437c2f6d11c0fad84c459e44219851452888e5bfcf706cd3d225d95836"
V10_REPORT = ROOT / "reports/taiji_n1_hard_neighbor_v10_factorial_diagnostic_20261017.json"
V10_REPORT_SHA256 = "287cc5c4a38693d8e055f1290d801b1ec6494768386ff3ad1963e3059852a822"
FACE = ROOT / "plans/manifests/n1_hard_neighbor_v11_confirmation_seed111_20261019.json"
FACE_CHECKS = ROOT / "reports/taiji_n1_v11_after_n120_face_checks_20261019.json"
ATTEMPT = FACE.with_name(FACE.stem + "_attempt.json")
SOURCE_READINESS = ROOT / "reports/taiji_n1_v11_after_n120_face_source_readiness_v3_20261019.json"
RANGE_REPORT = ROOT / "reports/taiji_n1_v11_after_n120_range_20261019.json"
RANGE_ATTEMPT = RANGE_REPORT.with_name(RANGE_REPORT.stem + "_attempt.json")
CONFIRM_REPORT = ROOT / "reports/taiji_n1_v11_after_n120_confirmation_20261019.json"
CONFIRM_ATTEMPT = CONFIRM_REPORT.with_name(CONFIRM_REPORT.stem + "_attempt.json")
SEED = 111
ITEMS = 40
ITEMS_PER_KIND = 10
OBSERVATIONS = 12
ENTITY_LENGTH = 10
ANSWER_LENGTHS = (6, 7)
DISTANCE = 1
KINDS = ("lookup", "revision", "transition", "unknown")
DIRECTIONS = ("north", "east", "south", "west")
SUBJECT_ROWS = {"lookup": 1, "revision": 2, "transition": 2, "unknown": 0}
ABSTENTION = "unknown"
LETTERS = v9.LETTERS
CPU_LIMIT_SECONDS = 60


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")


def _bound(path: Path, expected: str) -> dict:
    # 元封印碰撞修复: 清单自身也是重冻结对象，字节钉切 git 历史核验
    if not seal_ast.verify_path_pin(path, expected):
        raise ValueError(f"frozen N1-121 source hash mismatch: {path}")
    return read_json(path)


def _face_entities_and_words(face: dict) -> tuple[set[str], set[str]]:
    names: set[str] = set()
    words: set[str] = {ABSTENTION}
    for episode in face.get("episodes", []):
        names.update(episode["entities"])
        words.add(episode["answer"])
        for line in episode["observations"]:
            if " uses " in line:
                words.add(line.removesuffix(".").split()[-1])
    return names, words


def _source_exclusions() -> tuple[list[dict], set[str], set[str], dict]:
    base_expected = v9.EXPECTED_SHA256
    required = {
        v9.V5: base_expected["v5"],
        v9.V6: base_expected["v6"],
        v9.V7: base_expected["v7"],
        v9.V7_RANGE: base_expected["v7_range"],
        v9.V7_RANGE_ATTEMPT: base_expected["v7_range_attempt"],
        v9.V8: base_expected["v8"],
        v9.V8_RANGE: base_expected["v8_range"],
        v9.V8_RANGE_ATTEMPT: base_expected["v8_range_attempt"],
        N100_REPORT: N100_REPORT_SHA256,
        N110_PLAN: N110_PLAN_SHA256,
        N110_SCHEDULE: N110_SCHEDULE_SHA256,
        N121_PLAN: N121_PLAN_SHA256,
        N121_SOURCE_CORRECTION: N121_SOURCE_CORRECTION_SHA256,
        N120_REPORT: N120_REPORT_SHA256,
        N120_REVALIDATION: N120_REVALIDATION_SHA256,
        N120_RESTORE_READINESS: N120_RESTORE_READINESS_SHA256,
        V9_FACE: V9_FACE_SHA256,
        V9_FACE_CHECKS: V9_FACE_CHECKS_SHA256,
        V9_RANGE: V9_RANGE_SHA256,
        V9_RANGE_ATTEMPT: V9_RANGE_ATTEMPT_SHA256,
        V9_CONFIRM: V9_CONFIRM_SHA256,
        V9_CONFIRM_ATTEMPT: V9_CONFIRM_ATTEMPT_SHA256,
        V10_FACE: V10_FACE_SHA256,
        V10_FACE_CHECKS: V10_FACE_CHECKS_SHA256,
        V10_REPORT: V10_REPORT_SHA256,
    }
    sources = {str(path.resolve()): digest for path, digest in required.items()}
    sources[str(Path(train_data.MANIFEST).resolve())] = train_data.MANIFEST_SHA256
    sources[str(Path(v9.__file__).resolve())] = sha256(Path(v9.__file__).resolve())
    sources[str(Path(train_data.__file__).resolve())] = sha256(Path(train_data.__file__).resolve())
    sources[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())

    n100 = _bound(N100_REPORT, N100_REPORT_SHA256)
    n120 = _bound(N120_REPORT, N120_REPORT_SHA256)
    revalidation = _bound(N120_REVALIDATION, N120_REVALIDATION_SHA256)
    restore_readiness = _bound(N120_RESTORE_READINESS, N120_RESTORE_READINESS_SHA256)
    n110_text = N110_PLAN.read_text(encoding="utf-8")
    n121_text = N121_PLAN.read_text(encoding="utf-8")
    if (
        n100.get("complete") is not True
        or n100.get("training_sufficient") is not True
        or n120.get("status") != "stopped"
        or n120.get("protocol", {}).get("candidate_schedule_sha256") != N110_SCHEDULE_SHA256
        or n120.get("protocol", {}).get("material_value_map_sha256") != N110_VALUE_MAP_SHA256
        or revalidation.get("complete") is not True
        or revalidation.get("training_sufficient") is not True
        or revalidation.get("independent_restore_checks") != 18
        or restore_readiness.get("complete") is not True
        or restore_readiness.get("confirm_face_generated") is not False
        or restore_readiness.get("confirm_face_consumed") is not False
        or n110_text.find("v11 face seed at `111`") < 0
        or n121_text.find("seed `111`") < 0
        or any(
            report.get(key) != 0
            for report in (n100, n120, revalidation, restore_readiness)
            for key in ("dev_queries", "holdout_queries", "confirmation_queries")
        )
    ):
        raise ValueError("N1-100/N1-120 train and gate evidence does not admit v11 preparation")

    v7_names, v7_words, v7_provenance = v9.exclusions_from_consumed_sacrifice(
        label="v7",
        path=v9.V7,
        range_report_path=v9.V7_RANGE,
        range_attempt_path=v9.V7_RANGE_ATTEMPT,
        confirm_report_path=v9.V7_CONFIRM,
        confirm_attempt_path=v9.V7_CONFIRM_ATTEMPT,
    )
    v8_names, v8_words, v8_provenance = v9.exclusions_from_consumed_sacrifice(
        label="v8",
        path=v9.V8,
        range_report_path=v9.V8_RANGE,
        range_attempt_path=v9.V8_RANGE_ATTEMPT,
        confirm_report_path=v9.V8_CONFIRM,
        confirm_attempt_path=v9.V8_CONFIRM_ATTEMPT,
    )
    train_items, train_manifest_sha = train_data.train_items_from_frozen_manifest()
    if train_manifest_sha != train_data.MANIFEST_SHA256:
        raise ValueError("frozen v3 train manifest changed")
    schedule = _bound(N110_SCHEDULE, N110_SCHEDULE_SHA256)
    value_map = schedule.get("value_map", {})
    map_sha = hashlib.sha256(
        json.dumps(value_map, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        map_sha != N110_VALUE_MAP_SHA256
        or len(value_map) != N110_VALUE_MAP_ENTRIES
        or set(value_map) != set(train_data._values(train_items))
        or len(set(value_map.values())) != N110_VALUE_MAP_ENTRIES
        or any(len(source) != 6 or len(mapped) != 7 for source, mapped in value_map.items())
    ):
        raise ValueError("N1-110 frozen length map differs from the expected 288-value bijection")
    names, words = v9.exclusions_from_faces(read_json(v9.V5), read_json(v9.V6))
    names.update(name for item in train_items for name in item["entities"])
    words.update(
        line.removesuffix(".").split()[-1]
        for item in train_items
        for line in item["observations"]
        if " uses " in line
    )
    names.update(v7_names)
    words.update(v7_words)
    names.update(v8_names)
    words.update(v8_words)
    v9_face = read_json(V9_FACE)
    v10_face = read_json(V10_FACE)
    v9_names, v9_words = _face_entities_and_words(v9_face)
    v10_names, v10_words = _face_entities_and_words(v10_face)
    names.update(v9_names)
    names.update(v10_names)
    words.update(v9_words)
    words.update(v10_words)
    map_value_overlap = set(value_map.values()) & words
    if map_value_overlap:
        raise ValueError("N1-110 mapped training values overlap prior confirmation material")
    words.update(value_map)
    words.update(value_map.values())

    v9_checks = _bound(V9_FACE_CHECKS, V9_FACE_CHECKS_SHA256)
    v9_range = _bound(V9_RANGE, V9_RANGE_SHA256)
    v9_confirm = _bound(V9_CONFIRM, V9_CONFIRM_SHA256)
    v10_checks = _bound(V10_FACE_CHECKS, V10_FACE_CHECKS_SHA256)
    v10_report = _bound(V10_REPORT, V10_REPORT_SHA256)
    if (
        v9_face.get("schema") != "n1-hard-neighbor-v9-confirmation-v1"
        or v9_face.get("seed") != 109
        or v9_checks.get("accepted") is not True
        or v9_checks.get("face_sha256") != V9_FACE_SHA256
        or v9_range.get("complete") is not True
        or v9_confirm.get("complete") is not True
        or v9_confirm.get("model_queries") != 360
        or v9_confirm.get("can_promote") is not False
        or v10_face.get("schema") != "n1-hard-neighbor-v10-factorial-diagnostic-v1"
        or len(v10_face.get("episodes", [])) != 248
        or v10_checks.get("face_sha256") != V10_FACE_SHA256
        or v10_report.get("complete") is not True
        or v10_report.get("model_queries") != 2232
        or v10_report.get("parameter_updates") != 0
        or v10_report.get("can_promote") is not False
    ):
        raise ValueError("consumed v9/v10 sources do not match their sealed reports")

    provenance = {
        "v3_train_manifest_sha256": train_manifest_sha,
        "v7_sacrifice": v7_provenance,
        "v8_sacrifice": v8_provenance,
        "v9_face_sha256": V9_FACE_SHA256,
        "v9_confirm_report_sha256": V9_CONFIRM_SHA256,
        "v10_face_sha256": V10_FACE_SHA256,
        "v10_report_sha256": V10_REPORT_SHA256,
        "n110_schedule_sha256": N110_SCHEDULE_SHA256,
        "n110_material_value_map_sha256": N110_VALUE_MAP_SHA256,
        "n110_material_value_map_entries": N110_VALUE_MAP_ENTRIES,
        "n110_mapped_training_values_excluded": len(set(value_map.values()) & words),
        "n121_source_correction_sha256": N121_SOURCE_CORRECTION_SHA256,
        "excluded_entity_count": len(names),
        "excluded_material_count": len(words),
        "v7_confirm_payload_fields_decoded": False,
        "v8_confirm_payload_fields_decoded": False,
        "source_hashes": sources,
    }
    return train_items, names, words, provenance


def source_readiness() -> dict:
    if any(
        path.exists()
        for path in (
            FACE,
            FACE_CHECKS,
            ATTEMPT,
            RANGE_REPORT,
            RANGE_ATTEMPT,
            CONFIRM_REPORT,
            CONFIRM_ATTEMPT,
        )
    ):
        raise FileExistsError("v11 seed111 face or build attempt already exists; do not redraw")
    if SOURCE_READINESS.exists():
        raise FileExistsError("v11 source-readiness report is write-once")
    started = time.process_time()
    train_items, names, words, provenance = _source_exclusions()
    cpu_seconds = round(time.process_time() - started, 6)
    complete = cpu_seconds <= CPU_LIMIT_SECONDS and len(train_items) == 192
    report = {
        "schema": "n1-plan121-v11-face-source-readiness-v3",
        "status": "ready_for_owner_approval" if complete else "stopped",
        "complete": complete,
        "plan_sha256": N121_PLAN_SHA256,
        "n110_plan_sha256": N110_PLAN_SHA256,
        "n100_report_sha256": N100_REPORT_SHA256,
        "n120_report_sha256": N120_REPORT_SHA256,
        "n120_receipt_revalidation_sha256": N120_REVALIDATION_SHA256,
        "n120_restore_readiness_sha256": N120_RESTORE_READINESS_SHA256,
        "source_hashes": provenance["source_hashes"],
        "exclusion_provenance": {
            key: value for key, value in provenance.items() if key != "source_hashes"
        },
        "train_items": len(train_items),
        "excluded_entity_count": len(names),
        "excluded_material_count": len(words),
        "planned_face_seed": SEED,
        "planned_items": ITEMS,
        "planned_items_per_kind": ITEMS_PER_KIND,
        "planned_observations_per_item": OBSERVATIONS,
        "planned_name_distance": DISTANCE,
        "face_generated": False,
        "face_consumed": False,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "confirmation_queries": 0,
        "can_promote": False,
        "cpu_seconds": cpu_seconds,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
    }
    write_new(SOURCE_READINESS, report)
    return report


def build_v11_face(
    *, train_items: list[dict], excluded_names: set[str], excluded_words: set[str]
) -> dict:
    rng = random.Random(SEED)
    used_names = set(excluded_names)
    used_words = set(excluded_words) | {ABSTENTION}
    seen = Counter()
    episodes = []
    for index in range(ITEMS):
        kind = KINDS[index % len(KINDS)]
        ordinal = seen[kind]
        seen[kind] += 1
        subject = v9._novel(ENTITY_LENGTH, used_names, rng)
        distractor_count = OBSERVATIONS - SUBJECT_ROWS[kind]
        distractors = [v9._confusable(subject, used_names, rng) for _ in range(distractor_count)]
        materials = [v9._novel(6, used_words, rng) for _ in range(distractor_count)]
        answer_length = ANSWER_LENGTHS[ordinal % len(ANSWER_LENGTHS)]
        target_rows = []
        if kind == "lookup":
            answer = v9._novel(answer_length, used_words, rng)
            question = f"material of {subject}?"
            target_rows = [f"{subject} uses {answer}."]
        elif kind == "revision":
            stale = v9._novel(answer_length, used_words, rng)
            answer = v9._novel(answer_length, used_words, rng)
            question = f"material of {subject}?"
            target_rows = [f"{subject} uses {stale}.", f"{subject} now uses {answer}."]
        elif kind == "transition":
            answer = DIRECTIONS[ordinal % len(DIRECTIONS)]
            turn = rng.choice((1, 3))
            start = (DIRECTIONS.index(answer) - turn) % len(DIRECTIONS)
            question = f"direction of {subject}?"
            turn_word = "right" if turn == 1 else "left"
            target_rows = [f"{subject} faces {DIRECTIONS[start]}.", f"{subject} turns {turn_word}."]
        else:
            answer = ABSTENTION
            question = f"material of {subject}?"

        if kind == "transition":
            filler = [
                f"{name} faces {DIRECTIONS[(ordinal + offset) % len(DIRECTIONS)]}."
                for offset, name in enumerate(distractors)
            ]
        else:
            filler = [
                f"{name} uses {value}." for name, value in zip(distractors, materials, strict=True)
            ]
        rows: list[str | None] = [None] * OBSERVATIONS
        within_subset_ordinal = ordinal % (ITEMS_PER_KIND // 2)
        target_start = (
            within_subset_ordinal % (OBSERVATIONS - len(target_rows) + 1) if target_rows else 0
        )
        for offset, row in enumerate(target_rows):
            rows[target_start + offset] = row
        filler_iter = iter(filler)
        for position, row in enumerate(rows):
            if row is None:
                rows[position] = next(filler_iter)
        subset = "sacrifice" if ordinal < ITEMS_PER_KIND // 2 else "confirm"
        episodes.append(
            {
                "id": f"v11-hard-{index:03d}",
                "split": "holdout",
                "subset": subset,
                "kind": kind,
                "entities": [subject, *distractors],
                "observations": [str(row) for row in rows],
                "question": question,
                "answer": answer,
            }
        )
    subsets = {
        subset: [episode["id"] for episode in episodes if episode["subset"] == subset]
        for subset in ("sacrifice", "confirm")
    }
    return {
        "schema": "n1-hard-neighbor-v11-confirmation-v1",
        "seed": SEED,
        "items": ITEMS,
        "items_per_kind": ITEMS_PER_KIND,
        "observations_per_item": OBSERVATIONS,
        "entity_length": ENTITY_LENGTH,
        "name_neighbor_distance": DISTANCE,
        "subset_sizes": {name: len(ids) for name, ids in subsets.items()},
        "subsets": subsets,
        "exclusion_sources": {},
        "episodes": episodes,
    }


def audit_v11_face(
    face: dict, *, train_items: list[dict], names: set[str], words: set[str]
) -> dict:
    if face.get("schema") != "n1-hard-neighbor-v11-confirmation-v1" or face.get("seed") != SEED:
        raise ValueError("frozen v11 schema and seed111 required")
    original_schema = face["schema"]
    original_seed = v9.SEED
    face["schema"] = "n1-hard-neighbor-v9-confirmation-v1"
    v9.SEED = SEED
    try:
        result = v9.audit_face(
            face,
            train_items=train_items,
            excluded_names=names,
            excluded_words=words,
        )
    finally:
        face["schema"] = original_schema
        v9.SEED = original_seed
    result["checks"]["frozen_recipe"] = (
        face["schema"] == "n1-hard-neighbor-v11-confirmation-v1"
        and face.get("seed") == 111
        and face.get("items") == 40
        and face.get("observations_per_item") == 12
        and face.get("name_neighbor_distance") == 1
    )
    length_checks = {}
    for kind in ("lookup", "revision"):
        by_subset = {}
        for subset in ("sacrifice", "confirm"):
            values = [
                episode
                for episode in face["episodes"]
                if episode["kind"] == kind and episode["subset"] == subset
            ]
            lengths = Counter(len(episode["answer"]) for episode in values)
            expected = {"sacrifice": {6: 3, 7: 2}, "confirm": {6: 2, 7: 3}}[subset]
            revision_rows_match = True
            if kind == "revision":
                for episode in values:
                    subject = episode["entities"][0]
                    targets = [
                        row for row in episode["observations"] if row.startswith(subject + " ")
                    ]
                    revision_rows_match &= len(targets) == 2 and len(
                        targets[0].removesuffix(".").split()[-1]
                    ) == len(targets[1].removesuffix(".").split()[-1]) == len(episode["answer"])
            by_subset[subset] = {
                "counts": dict(lengths),
                "balanced_as_preregistered": lengths == Counter(expected) and revision_rows_match,
            }
        length_checks[kind] = by_subset
    result["checks"]["sacrifice_confirm_length_split"] = all(
        values[subset]["balanced_as_preregistered"]
        for values in length_checks.values()
        for subset in ("sacrifice", "confirm")
    )
    result["length_split"] = length_checks
    result["accepted"] = all(result["checks"].values())
    return result


def build_face_after_approval() -> dict:
    if any(
        path.exists()
        for path in (
            FACE,
            FACE_CHECKS,
            ATTEMPT,
            RANGE_REPORT,
            RANGE_ATTEMPT,
            CONFIRM_REPORT,
            CONFIRM_ATTEMPT,
        )
    ):
        raise FileExistsError("PLAN-N1-121 v11 face is one-shot; no retry")
    if not SOURCE_READINESS.is_file():
        raise FileNotFoundError(
            "zero-query N1-121 source readiness is required before face generation"
        )
    readiness = read_json(SOURCE_READINESS)
    current_build_sha = sha256(Path(__file__).resolve())
    if (
        readiness.get("complete") is not True
        or readiness.get("status") != "ready_for_owner_approval"
        or readiness.get("source_hashes", {}).get(str(Path(__file__).resolve()))
        != current_build_sha
    ):
        raise ValueError("N1-121 source readiness is stale or incomplete")
    train_items, names, words, provenance = _source_exclusions()
    started = time.process_time()
    write_new(
        ATTEMPT,
        {
            "schema": "n1-plan121-v11-face-attempt-v1",
            "status": "started",
            "plan_sha256": N121_PLAN_SHA256,
            "seed": SEED,
            "model_loads": 0,
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
        },
    )
    face = build_v11_face(train_items=train_items, excluded_names=names, excluded_words=words)
    face["exclusion_sources"] = {
        **provenance,
        "n120_report_sha256": N120_REPORT_SHA256,
        "n120_revalidation_sha256": N120_REVALIDATION_SHA256,
        "plan_sha256": N121_PLAN_SHA256,
        "source_readiness_sha256": sha256(SOURCE_READINESS),
    }
    checks = audit_v11_face(face, train_items=train_items, names=names, words=words)
    checks.update(
        {
            "schema": "n1-plan121-v11-face-checks-v1",
            "face": str(FACE),
            "face_sha256": None,
            "plan_sha256": N121_PLAN_SHA256,
            "model_loads": 0,
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
            "source_readiness_sha256": sha256(SOURCE_READINESS),
            "build_cpu_seconds": round(time.process_time() - started, 6),
        }
    )
    if not checks["accepted"]:
        write_new(FACE_CHECKS, checks)
        raise ValueError("v11 face failed its frozen contract; do not retry or alter seed")
    write_new(FACE, face)
    checks["face_sha256"] = sha256(FACE)
    write_new(FACE_CHECKS, checks)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-readiness", action="store_true")
    mode.add_argument("--build-face", action="store_true")
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    if args.build_face:
        if not args.owner_approved:
            raise PermissionError("face generation requires explicit PLAN-N1-121 owner approval")
        result = build_face_after_approval()
    else:
        result = source_readiness()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "accepted",
                    "source_hashes",
                    "excluded_entity_count",
                    "excluded_material_count",
                    "face",
                    "face_sha256",
                    "model_queries",
                    "confirmation_queries",
                    "can_promote",
                    "cpu_seconds",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete", result.get("accepted", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
