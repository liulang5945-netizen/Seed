"""Build the deterministic train-only PLAN-N1-110 raw/hard length schedule."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import (  # noqa: E402
    build_taiji_n1_hard_neighbor_training_schedule as base_builder,
)
from scripts.training import (  # noqa: E402
    build_taiji_n1_input_factorial_diagnostic as face_builder,
)
from scripts.training import seal_ast  # noqa: E402

SEED = base_builder.SEED
MANIFEST = base_builder.MANIFEST
MANIFEST_SHA256 = base_builder.MANIFEST_SHA256
RENAMING = base_builder.RENAMING
RENAMING_SHA256 = base_builder.RENAMING_SHA256
PAIRS = base_builder.PAIRS
PAIRS_SHA256 = base_builder.PAIRS_SHA256
core = base_builder.core
CONTRACT_TEST = ROOT / "tests/taiji_native/test_n1_plan110_length_balanced_schedule_contract.py"

PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-110_length_balanced_training_and_v11_confirmation_draft_20261017.md"
)
TRAIN_SCHEDULE = ROOT / "plans/manifests/n1_hard_neighbor_length_balanced_schedule_v1_20261018.json"
CHECKS = ROOT / "reports/taiji_n1_length_balanced_schedule_checks_20261018.json"
V10_FACE = face_builder.FACE

PLAN_SHA256 = "16d0cecb325af6463c107973fc0e741478140134a7642cd64c86e045c6069783"
BASE_BUILDER_SHA256 = "efa81a0bbc87f84db11e46cc40bc33e986fcfb6185c12b0404e9fccea4489d79"
V10_FACE_SHA256 = "7fdf769c23e19457adb32892d4ea117ebcca907edf4a1c740799faa091408f85"
LENGTH_ASSIGNMENT_SEED = 11010
VALUE_MAP_SEED = 11011
MATERIAL_LENGTHS = (6, 7)
MATERIAL_KINDS = {"lookup", "revision", "unknown"}
KINDS = ("lookup", "revision", "transition", "unknown")
EXPECTED_CONDITION_COUNTS = {
    "lookup/obs2/d1": 192,
    "lookup/obs2/d2": 192,
    "lookup/obs4/d1": 192,
    "lookup/obs4/d2": 192,
    "revision/obs4/d1": 384,
    "revision/obs4/d2": 384,
    "transition/obs2/dNone": 768,
    "unknown/obs2/d1": 192,
    "unknown/obs2/d2": 192,
    "unknown/obs4/d1": 192,
    "unknown/obs4/d2": 192,
}
LETTERS = string.ascii_lowercase
SCHEMA = "n1-hard-neighbor-length-balanced-training-schedule-v1"

_USES_RE = re.compile(r"^(?P<prefix>[a-z]+(?: now)? uses )(?P<value>[a-z]+)\.$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _collect_disjoint_values(
    train_items: list[dict],
    *,
    excluded_names: set[str],
    excluded_values: set[str],
    seed: int = VALUE_MAP_SEED,
) -> dict[str, str]:
    values = base_builder._values(train_items)
    if not values or any(
        len(value) != 6 or not value.isascii() or not value.islower() for value in values
    ):
        raise ValueError("PLAN-N1-110 requires a nonempty six-character train material pool")
    forbidden = set(excluded_values) | set(excluded_names) | set(values)
    rng = random.Random(seed)
    mapping: dict[str, str] = {}
    generated: set[str] = set()
    for source_value in values:
        for _ in range(10_000):
            candidate = "".join(rng.choice(LETTERS) for _ in range(7))
            if candidate not in forbidden and candidate not in generated:
                mapping[source_value] = candidate
                generated.add(candidate)
                break
        else:
            raise ValueError("could not generate a disjoint seven-character value map")
    if len(set(mapping.values())) != len(mapping) or set(mapping.values()) & forbidden:
        raise ValueError("seven-character material map is not injective/disjoint")
    return mapping


def transform_material_item(
    source: dict,
    length: int | None,
    value_map: dict[str, str],
    *,
    new_id: str | None = None,
) -> dict:
    """Clone one train episode, consistently remapping every `uses` value."""
    item = copy.deepcopy(source)
    if item.get("kind") == "transition":
        if length is not None:
            raise ValueError("transition tasks must remain length-neutral")
        if new_id is not None:
            item["id"] = new_id
        return item
    if length is None:
        raise ValueError("only transition tasks may omit material length")
    if length not in MATERIAL_LENGTHS:
        raise ValueError(f"unsupported material length {length}")
    if item.get("kind") not in MATERIAL_KINDS:
        raise ValueError("material length assignment applied to a non-material task")

    mapped_rows = []
    matched = 0
    for row in item.get("observations", []):
        match = _USES_RE.fullmatch(row)
        if not match:
            mapped_rows.append(row)
            continue
        value = match.group("value")
        if len(value) != 6 or value not in value_map:
            raise ValueError("train `uses` value is missing from the frozen six-to-seven map")
        output = value if length == 6 else value_map[value]
        if len(output) != length:
            raise ValueError("mapped `uses` value does not match its assigned length")
        mapped_rows.append(f"{match.group('prefix')}{output}.")
        matched += 1
    if not matched:
        raise ValueError("material task has no `uses` observations")
    item["observations"] = mapped_rows

    if item.get("kind") in {"lookup", "revision"}:
        answer = item.get("answer")
        if answer not in value_map:
            raise ValueError("known target answer is missing from the frozen value map")
        item["answer"] = answer if length == 6 else value_map[answer]
        if len(item["answer"]) != length:
            raise ValueError("known target answer does not match assigned material length")
    elif item.get("kind") == "unknown" and item.get("answer") != "unknown":
        raise ValueError("unknown task must preserve the literal abstention label")

    if new_id is not None:
        item["id"] = new_id
    return item


def _raw_length_assignments(items: list[dict], *, seed: int = LENGTH_ASSIGNMENT_SEED) -> list[dict]:
    rng = random.Random(seed)
    assignments = []
    for item in items:
        if item["kind"] in MATERIAL_KINDS:
            lengths: list[int | None] = [6] * 8 + [7] * 8
            rng.shuffle(lengths)
        else:
            lengths = [None] * 16
        for raw_slot, length in enumerate(lengths):
            assignments.append(
                {
                    "parent_id": item["id"],
                    "parent_kind": item["kind"],
                    "raw_slot": raw_slot,
                    "material_length_chars": length,
                }
            )
    return assignments


def _materialize_raw_views(
    items: list[dict],
    assignments: list[dict],
    value_map: dict[str, str],
) -> list[dict]:
    by_key = {(row["parent_id"], row["raw_slot"]): row for row in assignments}
    if len(by_key) != 3072:
        raise ValueError("raw length assignment keys are missing or duplicated")
    raw_views = []
    for parent in items:
        for slot in range(16):
            assignment = by_key[parent["id"], slot]
            length = assignment["material_length_chars"]
            item = transform_material_item(
                parent,
                length,
                value_map,
                new_id=f"raw-{parent['id']}-slot{slot:02d}-l{length}",
            )
            if length is not None:
                item["parent_id"] = parent["id"]
            second = (
                base_builder.core.renaming.second_question(item)
                if parent["kind"] in {"lookup", "unknown"}
                else None
            )
            if parent["kind"] in {"lookup", "unknown"}:
                if second is None or second.get("observations") != item.get("observations"):
                    raise ValueError("raw paired question must derive from the transformed view")
                if len(second.get("answer", "")) != length:
                    raise ValueError("raw paired-question answer length differs from its view")
            raw_views.append(
                {
                    "parent_id": parent["id"],
                    "parent_kind": parent["kind"],
                    "raw_slot": slot,
                    "material_length_chars": length,
                    "item": item,
                    "second_question": second,
                }
            )
    return raw_views


def _hard_length_assignments(
    rows: list[dict], *, seed: int = LENGTH_ASSIGNMENT_SEED
) -> dict[int, int | None]:
    rng = random.Random(seed + 1)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    assignments: dict[int, int | None] = {}
    for row in rows:
        kind = row["parent_kind"]
        if kind == "transition":
            assignments[row["schedule_index"]] = None
            continue
        key = (
            row["parent_id"],
            kind,
            row["observation_count"],
            row["hamming_distance"],
        )
        grouped[key].append(row)

    for (parent_id, kind, observation_count, distance), group in sorted(
        grouped.items(), key=lambda entry: tuple(str(part) for part in entry[0])
    ):
        expected_slots = 8 if kind == "revision" else 4
        if len(group) != expected_slots:
            raise ValueError("hard-view parent/cell slot count differs from PLAN-N1-100")
        shuffled = sorted(group, key=lambda row: row["slot"])
        rng.shuffle(shuffled)
        for index, row in enumerate(shuffled):
            assignments[row["schedule_index"]] = 6 if index < len(group) // 2 else 7
    if set(assignments) != {row["schedule_index"] for row in rows}:
        raise ValueError("one or more hard schedule slots lack a length assignment")
    return assignments


def _transform_hard_rows(
    rows: list[dict],
    *,
    length_map: dict[str, str],
    assignments: dict[int, int | None],
) -> list[dict]:
    output = []
    for row in rows:
        length = assignments[row["schedule_index"]]
        if row["parent_kind"] == "transition":
            if length is not None:
                raise ValueError("transition schedule must remain length-neutral")
            transformed = copy.deepcopy(row["item"])
            second = None
        else:
            transformed = transform_material_item(
                row["item"],
                length,
                length_map,
                new_id=f"{row['item']['id']}-l{length}",
            )
            second = (
                base_builder.core.renaming.second_question(transformed)
                if row["parent_kind"] in {"lookup", "unknown"}
                else None
            )
        if row["parent_kind"] in {"lookup", "unknown"}:
            if (
                second is None
                or second.get("observations") != transformed.get("observations")
                or len(second.get("answer", "")) != length
            ):
                raise ValueError("paired question must derive from same-length transformed facts")
        elif second is not None:
            raise ValueError("revision/transition schedule must not add a paired question")
        output.append(
            {
                **copy.deepcopy(row),
                "item": transformed,
                "second_question": second,
                "material_length_chars": length,
            }
        )
    return output


def _source_bundle() -> tuple[list[dict], dict, dict, dict]:
    items, manifest_sha = base_builder.train_items_from_frozen_manifest()
    if manifest_sha != base_builder.MANIFEST_SHA256:
        raise ValueError("frozen train manifest hash mismatch")
    items, curriculum, _ = base_builder.load_training_sources(items)
    excluded_names, excluded_values, provenance, guards = face_builder.collect_exclusions()
    if provenance.get("v7_v8_confirmation_payload_fields_decoded") is not False:
        raise ValueError("prohibited v7/v8 confirmation payload access was recorded")
    if provenance.get("v7_v8_confirmation_artifacts_absent") is not True:
        raise ValueError("unexpected v7/v8 confirmation artifact appeared")
    if sha256_file(V10_FACE) != V10_FACE_SHA256:
        raise ValueError("consumed v10 exclusion face hash changed")
    v10_face = read_json(V10_FACE)
    for episode in v10_face.get("episodes", []):
        excluded_names.update(episode["entities"])
        excluded_values.add(episode["answer"])
        excluded_values.update(
            row.removesuffix(".").split()[-1] for row in episode["observations"] if " uses " in row
        )
    guards[str(V10_FACE.resolve())] = V10_FACE_SHA256
    guards[str(PLAN.resolve())] = PLAN_SHA256
    guards[str(Path(base_builder.__file__).resolve())] = sha256_file(
        Path(base_builder.__file__).resolve()
    )
    contract_test = ROOT / "tests/taiji_native/test_n1_plan110_length_balanced_schedule_contract.py"
    if not contract_test.is_file():
        raise FileNotFoundError("frozen PLAN-N1-110 schedule contract test is required")
    guards[str(Path(__file__).resolve())] = sha256_file(Path(__file__).resolve())
    guards[str(contract_test.resolve())] = sha256_file(contract_test)
    return (
        items,
        curriculum,
        {
            "excluded_names": excluded_names,
            "excluded_values": excluded_values,
            "provenance": provenance,
            "source_guards": guards,
        },
        {"manifest_sha256": manifest_sha},
    )


def build_candidate_schedule(
    items: list[dict],
    curriculum: dict,
    *,
    excluded_names: set[str],
    excluded_values: set[str],
    source_guards: dict[str, str],
) -> dict:
    base_schedule = base_builder.build_schedule(items, curriculum)
    length_map = _collect_disjoint_values(
        items,
        excluded_names=excluded_names,
        excluded_values=excluded_values,
    )
    raw_assignments = _raw_length_assignments(items)
    raw_views = _materialize_raw_views(items, raw_assignments, length_map)
    hard_assignments = _hard_length_assignments(base_schedule["items"])
    hard_rows = _transform_hard_rows(
        base_schedule["items"],
        length_map=length_map,
        assignments=hard_assignments,
    )
    value_mapping = {source: length_map[source] for source in sorted(length_map)}
    condition_counts = copy.deepcopy(base_schedule["summary"]["condition_counts"])
    raw_counts = Counter(
        (items[index // 16]["kind"], assignment["material_length_chars"])
        for index, assignment in enumerate(raw_assignments)
    )
    hard_counts = Counter((row["parent_kind"], row["material_length_chars"]) for row in hard_rows)
    hard_by_condition = Counter(
        (
            row["parent_kind"],
            row["observation_count"],
            row["hamming_distance"],
            row["material_length_chars"],
        )
        for row in hard_rows
    )
    return {
        "schema": SCHEMA,
        "seed": base_builder.SEED,
        "length_assignment_seed": LENGTH_ASSIGNMENT_SEED,
        "value_map_seed": VALUE_MAP_SEED,
        "plan_sha256": PLAN_SHA256,
        "source_manifest_sha256": base_builder.MANIFEST_SHA256,
        "source_renaming_sha256": base_builder.RENAMING_SHA256,
        "source_pair_manifest_sha256": base_builder.PAIRS_SHA256,
        "source_guards": dict(sorted(source_guards.items())),
        "base_builder_sha256": sha256_file(Path(base_builder.__file__).resolve()),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "value_map_sha256": hashlib.sha256(
            json.dumps(value_mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "value_map": value_mapping,
        "raw_length_assignments": raw_assignments,
        "raw_views": raw_views,
        "items": hard_rows,
        "summary": {
            "schedule_items": len(hard_rows),
            "raw_length_slots": len(raw_assignments),
            "parents": len(items),
            "slots_per_parent_per_path": 16,
            "kind_counts": copy.deepcopy(base_schedule["summary"]["kind_counts"]),
            "condition_counts": condition_counts,
            "raw_material_length_counts": {
                f"{kind}/len{length}": count
                for (kind, length), count in sorted(
                    raw_counts.items(), key=lambda entry: tuple(str(part) for part in entry[0])
                )
            },
            "hard_material_length_counts": {
                f"{kind}/len{length}": count
                for (kind, length), count in sorted(
                    hard_counts.items(), key=lambda entry: tuple(str(part) for part in entry[0])
                )
            },
            "hard_condition_length_counts": {
                f"{kind}/obs{observations}/d{distance}/len{length}": count
                for (kind, observations, distance, length), count in sorted(
                    hard_by_condition.items(),
                    key=lambda entry: tuple(str(part) for part in entry[0]),
                )
            },
            "mapping_entries": len(value_mapping),
            "parameter_updates": 0,
            "dev_queries": 0,
            "holdout_queries": 0,
        },
    }


def audit_schedule(schedule: dict, items: list[dict], excluded_values: set[str]) -> dict:
    errors = []
    if schedule.get("schema") != SCHEMA or schedule.get("plan_sha256") != PLAN_SHA256:
        errors.append("plan/schema mismatch")
    if (
        len(schedule.get("items", [])) != 3072
        or len(schedule.get("raw_length_assignments", [])) != 3072
        or len(schedule.get("raw_views", [])) != 3072
    ):
        errors.append("raw/hard slot denominators differ from PLAN-N1-100")
    if schedule.get("summary", {}).get("condition_counts") != EXPECTED_CONDITION_COUNTS:
        errors.append("hard-view observation/distance cell counts changed")

    mapping = schedule.get("value_map", {})
    mapping_values = list(mapping.values())
    if (
        set(mapping) != set(base_builder._values(items))
        or len(mapping_values) != len(set(mapping_values))
        or set(mapping_values) & excluded_values
        or any(
            len(value) != 7 or not value.isascii() or not value.islower()
            for value in mapping_values
        )
    ):
        errors.append("seven-character value map is not injective or source-disjoint")

    raw_assignments = schedule.get("raw_length_assignments", [])
    assignment_by_key = {
        (row.get("parent_id"), row.get("raw_slot")): row for row in raw_assignments
    }
    if len(raw_assignments) != 3072 or len(assignment_by_key) != 3072:
        errors.append("raw length assignment keys are missing or duplicated")
    raw_by_parent: dict[str, Counter] = defaultdict(Counter)
    raw_keys = set()
    for raw_view in schedule.get("raw_views", []):
        parent_id = raw_view["parent_id"]
        slot = raw_view["raw_slot"]
        key = (parent_id, slot)
        if (
            key in raw_keys
            or parent_id not in {item["id"] for item in items}
            or slot not in range(16)
        ):
            errors.append("raw view parent/slot identity is invalid")
        raw_keys.add(key)
        kind = raw_view["parent_kind"]
        length = raw_view["material_length_chars"]
        assignment = assignment_by_key.get(key)
        if (
            assignment is None
            or assignment.get("material_length_chars") != length
            or assignment.get("parent_kind") != kind
        ):
            errors.append("raw view and frozen length assignment disagree")
        raw_by_parent[parent_id][length] += 1
        if kind == "transition":
            if length is not None or raw_view["second_question"] is not None:
                errors.append("raw transition view acquired a material length/second question")
            continue
        raw_item = raw_view["item"]
        uses_lengths = [
            len(match.group("value"))
            for observation in raw_item.get("observations", [])
            if (match := _USES_RE.fullmatch(observation)) is not None
        ]
        if not uses_lengths or set(uses_lengths) != {length}:
            errors.append(f"raw view has mixed/missing uses lengths: {raw_item.get('id')}")
        if kind in {"lookup", "revision"} and len(raw_item.get("answer", "")) != length:
            errors.append(f"raw known answer length mismatch: {raw_item.get('id')}")
        if kind == "unknown" and raw_item.get("answer") != "unknown":
            errors.append(f"raw unknown label changed: {raw_item.get('id')}")
        second = raw_view.get("second_question")
        if kind in {"lookup", "unknown"} and (
            second is None or len(second.get("answer", "")) != length
        ):
            errors.append(f"raw paired question length mismatch: {raw_item.get('id')}")
    source_by_parent = {item["id"]: item for item in items}
    if raw_keys != {(item["id"], slot) for item in items for slot in range(16)}:
        errors.append("raw view parent/slot coverage is incomplete")
    for parent_id, counts in raw_by_parent.items():
        kind = source_by_parent[parent_id]["kind"]
        expected = Counter({6: 8, 7: 8}) if kind in MATERIAL_KINDS else Counter({None: 16})
        if counts != expected:
            errors.append(f"raw length balance failed for {parent_id}")
    expected_path_counts = Counter(
        {
            ("lookup", 6): 384,
            ("lookup", 7): 384,
            ("revision", 6): 384,
            ("revision", 7): 384,
            ("unknown", 6): 384,
            ("unknown", 7): 384,
            ("transition", None): 768,
        }
    )
    raw_counts = Counter(
        (view["parent_kind"], view["material_length_chars"])
        for view in schedule.get("raw_views", [])
    )
    hard_counts = Counter(
        (row["parent_kind"], row["material_length_chars"]) for row in schedule.get("items", [])
    )
    if raw_counts != expected_path_counts or hard_counts != expected_path_counts:
        errors.append("raw/hard aggregate length counts differ from 50/50 contract")
    for label, counts in (
        ("raw_material_length_counts", raw_counts),
        ("hard_material_length_counts", hard_counts),
    ):
        expected_summary = {
            f"{kind}/len{length}": count
            for (kind, length), count in sorted(
                counts.items(), key=lambda entry: tuple(str(part) for part in entry[0])
            )
        }
        if schedule.get("summary", {}).get(label) != expected_summary:
            errors.append(f"{label} differs from materialized schedule")

    hard_by_parent_condition: dict[tuple, Counter] = defaultdict(Counter)
    hard_indices = set()
    for row in schedule.get("items", []):
        if row.get("schedule_index") in hard_indices:
            errors.append("hard schedule index is duplicated")
        hard_indices.add(row.get("schedule_index"))
        item = row["item"]
        length = row["material_length_chars"]
        if row["parent_kind"] == "transition":
            if length is not None:
                errors.append("transition task received a material length")
            continue
        key = (
            row["parent_id"],
            row["parent_kind"],
            row["observation_count"],
            row["hamming_distance"],
        )
        hard_by_parent_condition[key][length] += 1
        uses_lengths = [
            len(match.group("value"))
            for observation in item["observations"]
            if (match := _USES_RE.fullmatch(observation)) is not None
        ]
        if not uses_lengths or set(uses_lengths) != {length}:
            errors.append(f"hard view has mixed/missing uses lengths: {item['id']}")
        if row["parent_kind"] in {"lookup", "revision"} and len(item["answer"]) != length:
            errors.append(f"hard known answer length mismatch: {item['id']}")
        if row["parent_kind"] == "unknown" and item["answer"] != "unknown":
            errors.append(f"unknown abstention label changed: {item['id']}")
        if row["parent_kind"] in {"lookup", "unknown"}:
            if (
                row.get("second_question") is None
                or len(row["second_question"].get("answer", "")) != length
            ):
                errors.append(f"paired question answer length mismatch: {item['id']}")

    for key, counts in hard_by_parent_condition.items():
        expected = (
            Counter({6: 2, 7: 2}) if key[1] in {"lookup", "unknown"} else Counter({6: 4, 7: 4})
        )
        if counts != expected:
            errors.append(f"hard per-parent/condition length balance failed: {key}")
    if hard_indices != set(range(3072)):
        errors.append("hard schedule index coverage is incomplete")
    return {
        "accepted": not errors,
        "errors": errors,
        "raw_parents": len(raw_by_parent),
        "hard_parent_condition_cells": len(hard_by_parent_condition),
        "mapping_entries": len(mapping),
        "raw_length_assignments": len(schedule.get("raw_length_assignments", [])),
        "hard_schedule_entries": len(schedule.get("items", [])),
        "parameter_updates": 0,
        "model_queries": 0,
        "checkpoint_loads": 0,
    }


def build_from_frozen_sources() -> tuple[dict, dict]:
    if sha256_file(PLAN) != PLAN_SHA256:
        raise ValueError("PLAN-N1-110 plan hash changed")
    items, curriculum, exclusions, _ = _source_bundle()
    # S6: base builder 为历史钉点——在位字节或 git 历史 blob 任一核验通过即可
    if not seal_ast.verify_path_pin(Path(base_builder.__file__).resolve(), BASE_BUILDER_SHA256):
        raise ValueError("PLAN-N1-100 base schedule builder changed")
    schedule = build_candidate_schedule(
        items,
        curriculum,
        excluded_names=exclusions["excluded_names"],
        excluded_values=exclusions["excluded_values"],
        source_guards=exclusions["source_guards"],
    )
    checks = audit_schedule(schedule, items, exclusions["excluded_values"])
    if not checks["accepted"]:
        raise ValueError(f"length-balanced schedule rejected: {checks['errors']}")
    checks.update(
        {
            "schema": "n1-length-balanced-schedule-construction-checks-v1",
            "plan_sha256": PLAN_SHA256,
            "schedule_sha256": hashlib.sha256(
                json.dumps(
                    schedule, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "source_guards": schedule["source_guards"],
            "exclusion_provenance": exclusions["provenance"],
        }
    )
    return schedule, checks


def write_candidate_schedule() -> tuple[dict, dict]:
    if TRAIN_SCHEDULE.exists() or CHECKS.exists():
        raise FileExistsError("PLAN-N1-110 schedule/checks are write-once")
    schedule, checks = build_from_frozen_sources()
    schedule_bytes = (
        json.dumps(schedule, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    )
    checks["schedule_file_sha256"] = hashlib.sha256(schedule_bytes.encode("utf-8")).hexdigest()
    checks_bytes = (
        json.dumps(checks, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with TRAIN_SCHEDULE.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(schedule_bytes)
    with CHECKS.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(checks_bytes)
    return schedule, checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        schedule, checks = write_candidate_schedule()
        print(
            json.dumps(
                {
                    "complete": checks["accepted"],
                    "schedule": str(TRAIN_SCHEDULE.relative_to(ROOT)),
                    "checks": str(CHECKS.relative_to(ROOT)),
                    "items": len(schedule["items"]),
                    "raw_length_slots": len(schedule["raw_length_assignments"]),
                    "mapping_entries": len(schedule["value_map"]),
                },
                sort_keys=True,
            )
        )
    else:
        schedule, checks = build_from_frozen_sources()
        print(
            json.dumps(
                {
                    "accepted": checks["accepted"],
                    "errors": checks["errors"],
                    "items": len(schedule["items"]),
                    "raw_length_slots": len(schedule["raw_length_assignments"]),
                    "mapping_entries": len(schedule["value_map"]),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
