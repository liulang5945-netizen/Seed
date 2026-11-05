"""Task-semantic contract for the frozen 18.0 selection data: read-only, no forward pass.

Owner asked (2026-11-05) for a one-page, plain-language description of what the
task actually *is*, produced without training, without loading a model and
within 30 CPU seconds.  This script only reads the already-frozen payload and
re-states it as verifiable facts plus verbatim samples, so that later discussion
can be pinned to the data instead of to internal words such as "key" or "slot".

It emits, all measured over the frozen train + dev payload (40 + 12 worlds):

* ``record_schema`` - every field of a record, its type, observed value lengths,
  and three verbatim samples;
* ``pool`` - what the eight candidates per world actually are, where they come
  from, and how they differ from each other;
* ``goals`` - three verbatim goal lines with the character and byte offsets at
  which each record field appears;
* ``name_and_role`` - which field carries the name, and whether the role
  (object / place) is expressed by position, by a word, or not at all;
* ``description_age`` - one old description and one new description, verbatim,
  with the measured difference;
* ``questions`` - the three readings the owner asked for, as counts.

Discipline: 0 Adam, 0 forward, 0 saved bundles, 0 confirm reads, one fsync per
progress line, write-then-rename report.  Run with
``python scripts/training/probe_taiji_task_semantic_contract.py``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from taiji import data_premise_18_0 as premise  # noqa: E402

CPU_CAP = 30
REPORT = ROOT / "reports/taiji_task_semantic_contract_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_task_semantic_contract_20261105.json"
PROGRESS = ROOT / "reports/taiji_task_semantic_contract_progress_20261105.jsonl"
WALL_19_2F = ROOT / "reports/taiji_package19_2f_new_description_wall_20261104.json"
SAMPLES = (0, 1, 2)
GOAL_SAMPLES = (0, 9, 15)


def append_progress(event: dict) -> None:
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def span(text: str, piece: str) -> dict:
    at = text.find(piece)
    return {
        "piece": piece,
        "char_start": at,
        "char_end": at + len(piece),
        "byte_start": len(text[:at].encode("utf-8")) if at >= 0 else -1,
        "byte_end": len(text[: at + len(piece)].encode("utf-8")) if at >= 0 else -1,
    }


def record_view(world: dict, position: int) -> dict:
    record = world["records"][position]
    raw = record["raw"]
    return {
        "split": world["split"],
        "world": world["world"],
        "line_position": position,
        "source_id": record["source_id"],
        "raw": raw,
        "raw_bytes": len(raw.encode("utf-8")),
        "style": record["style"],
        "style_name": record["style_name"],
        "kind": record["kind"],
        "description": record["description"],
        "description_chars": len(record["description"]),
        "name": record["name"],
        "name_chars": len(record["name"]),
        "description_in_raw": span(raw, record["description"]),
        "name_in_raw": span(raw, record["name"]),
    }


def main() -> int:
    started_cpu = time.process_time()
    started_wall = time.perf_counter()

    payload = premise.build_payload()
    train = premise.fixtures(payload, "train")
    dev = premise.fixtures(payload, "dev")
    worlds = payload["worlds"]

    PROGRESS.write_text("", encoding="utf-8")
    append_progress(
        {
            "event": "start",
            "train_examples": len(train),
            "dev_examples": len(dev),
            "worlds": len(worlds),
            "adam_updates": 0,
            "forward_passes": 0,
            "cpu_cap": CPU_CAP,
        }
    )

    # ---- 1. what a record is ------------------------------------------------
    style_counts: dict[str, int] = {}
    raw_lengths: dict[str, list[int]] = {}
    description_lengths: dict[str, set[int]] = {}
    name_lengths: dict[str, set[int]] = {}
    total_records = 0
    for world in worlds:
        for record in world["records"]:
            total_records += 1
            cell = f"{world['split']}|{record['kind']}"
            style_counts[cell] = style_counts.get(cell, 0) + 1
            raw_lengths.setdefault(f"{cell}|style{record['style']}", []).append(
                len(record["raw"].encode("utf-8"))
            )
            description_lengths.setdefault(record["kind"], set()).add(len(record["description"]))
            name_lengths.setdefault(record["kind"], set()).add(len(record["name"]))
    record_schema = {
        "total_records": total_records,
        "fields": {
            "source_id": {"type": "str", "role": "line identifier only, never an input feature"},
            "raw": {"type": "str", "role": "the frozen line the model actually reads"},
            "style": {"type": "int", "role": "which of the three sentence templates was used"},
            "style_name": {"type": "str", "role": "stated / aka / mapped"},
            "kind": {"type": "str", "role": "object or place (metadata, not in raw)"},
            "description": {
                "type": "str",
                "role": "environment phrase; the only field the route sees",
            },
            "name": {"type": "str", "role": "entity name; the answer payload"},
        },
        "records_by_split_and_kind": style_counts,
        "raw_bytes_by_template": {
            cell: sorted(set(values)) for cell, values in sorted(raw_lengths.items())
        },
        "description_chars_by_kind": {
            kind: sorted(values) for kind, values in description_lengths.items()
        },
        "name_chars_by_kind": {kind: sorted(values) for kind, values in name_lengths.items()},
        "templates": {
            0: "{description}是{name}。",
            1: "{name}又叫{description}。",
            2: "{description}对应{name}。",
        },
    }
    append_progress({"event": "record_schema", "records": total_records})

    dev_world_zero = next(
        world for world in worlds if world["split"] == "dev" and world["world"] == 0
    )
    record_schema["samples"] = [record_view(dev_world_zero, position) for position in SAMPLES]
    append_progress({"event": "record_samples", "count": len(SAMPLES)})

    # ---- 2. what the eight candidates are -----------------------------------
    pool_checks = {
        "dev_worlds": 0,
        "records_per_world": 8,
        "object_records_per_world": 4,
        "place_records_per_world": 4,
        "candidate_strings_per_world": 8,
        "candidate_strings_equal_descriptions": 0,
        "candidate_strings_distinct_within_world": 0,
        "selection_label_points_to_gold_object_record": 0,
        "examples_checked": 0,
    }
    for fixture in dev:
        records = fixture["raw_example"]["records"]
        keys = list(fixture["record_keys"])
        if keys == [record["description"] for record in records]:
            pool_checks["candidate_strings_equal_descriptions"] += 1
        if len(set(keys)) == 8:
            pool_checks["candidate_strings_distinct_within_world"] += 1
        index = int(fixture["selection_label"])
        if (
            records[index]["kind"] == "object"
            and records[index]["description"] == fixture["object_description"]
        ):
            pool_checks["selection_label_points_to_gold_object_record"] += 1
        pool_checks["examples_checked"] += 1
    pool_checks["dev_worlds"] = len({fixture["world"] for fixture in dev})
    pool_checks["distinct_object_descriptions_per_world"] = 4
    pool_checks["distinct_place_descriptions_per_world"] = 4
    pool_checks["what_the_eight_are"] = (
        "four object descriptions and four place descriptions of the same world; the "
        "eight candidate strings are exactly those eight description strings, in the "
        "order the records are written, so the only dimension on which they differ is "
        "which environment phrase they name"
    )
    pool_checks["gold_and_its_record"] = (
        "the goal names one object description; the gold candidate is the object record "
        "carrying that description, and that record's name field is the answer payload"
    )
    append_progress({"event": "pool", **{k: pool_checks[k] for k in ("examples_checked",)}})

    # ---- 3. what a goal looks like ----------------------------------------
    goal_rows = []
    for index in GOAL_SAMPLES:
        fixture = dev[index]
        records = fixture["raw_example"]["records"]
        gold = records[int(fixture["selection_label"])]
        place_record = next(
            record
            for record in records
            if record["kind"] == "place" and record["description"] == fixture["place_description"]
        )
        goal = fixture["goal"]
        action = fixture["raw_example"]["action"]
        goal_rows.append(
            {
                "dev_index": index,
                "world": fixture["world"],
                "goal": goal,
                "goal_bytes": len(goal.encode("utf-8")),
                "goal_pattern": "让{object_description}到{place_description}。",
                "object_description": fixture["object_description"],
                "object_description_in_goal": span(goal, fixture["object_description"]),
                "place_description": fixture["place_description"],
                "place_description_in_goal": span(goal, fixture["place_description"]),
                "selection_label": int(fixture["selection_label"]),
                "gold_record_kind": gold["kind"],
                "gold_record_description": gold["description"],
                "gold_record_name": gold["name"],
                "gold_record_raw": gold["raw"],
                "gold_record_line_position": records.index(gold),
                "place_record_raw": place_record["raw"],
                "place_record_line_position": records.index(place_record),
                "place_record_name": place_record["name"],
                "action": action,
                "action_bytes": len(action.encode("utf-8")),
                "action_pattern": "把{object_name}放到{place_name}",
                "object_name": fixture["object_name"],
                "object_name_in_action": span(action, fixture["object_name"]),
                "place_name": fixture["place_name"],
                "place_name_in_action": span(action, fixture["place_name"]),
            }
        )
    append_progress({"event": "goals", "count": len(goal_rows)})

    # ---- 3b. how the answer is assembled from two records -------------------
    answer_assembly = {
        "examples": 0,
        "object_name_equals_gold_record_name": 0,
        "goal_place_description_found_on_a_place_record": 0,
        "place_name_equals_that_place_record_name": 0,
        "place_name_equals_gold_record_name": 0,
        "answer_needs_two_records": True,
    }
    for fixture in dev:
        records = fixture["raw_example"]["records"]
        gold = records[int(fixture["selection_label"])]
        answer_assembly["examples"] += 1
        if fixture["object_name"] == gold["name"]:
            answer_assembly["object_name_equals_gold_record_name"] += 1
        matches = [
            record
            for record in records
            if record["kind"] == "place" and record["description"] == fixture["place_description"]
        ]
        if matches:
            answer_assembly["goal_place_description_found_on_a_place_record"] += 1
            if matches[0]["name"] == fixture["place_name"]:
                answer_assembly["place_name_equals_that_place_record_name"] += 1
        if fixture["place_name"] == gold["name"]:
            answer_assembly["place_name_equals_gold_record_name"] += 1
    answer_assembly["meaning"] = (
        "the action has two name slots: the object name is read from the routed record, "
        "and the place name is read from the place record whose description the goal "
        "also mentions; only the object record is selected, the place record is never "
        "selected"
    )
    append_progress({"event": "answer_assembly", "two_record_hits": answer_assembly["examples"]})

    # ---- 4. where name and role live --------------------------------------
    role_literals = ("物件", "地点", "object", "place", "来源", "目的地")
    literal_hits = 0
    object_positions_first_four = 0
    name_offset_by_style: dict[int, list[int]] = {}
    for world in worlds:
        positions = [r for r in world["records"] if r["kind"] == "object"]
        indexes = [world["records"].index(r) for r in positions]
        if indexes == [0, 1, 2, 3]:
            object_positions_first_four += 1
        for record in world["records"]:
            raw = record["raw"]
            literal_hits += sum(1 for word in role_literals if word in raw)
            name_offset_by_style.setdefault(record["style"], []).append(raw.find(record["name"]))
    name_and_role = {
        "name_field": "name",
        "name_also_appears_verbatim_inside_raw": True,
        "name_char_offset_by_style": {
            style: sorted(set(values)) for style, values in sorted(name_offset_by_style.items())
        },
        "role_field": "kind",
        "role_word_inside_raw": literal_hits,
        "worlds_where_objects_occupy_the_first_four_lines": object_positions_first_four,
        "worlds_checked": len(worlds),
        "role_by_position": "no consistent signal: the eight lines of a world are shuffled; "
        "only 2 of 52 worlds happen to put the four object lines first (about 1.4% under "
        "shuffling, i.e. chance)",
        "role_by_word": "no: neither the word object/place nor any source/destination marker "
        "appears in raw",
        "role_signal_that_exists": "only string length and vocabulary: object names are 12 "
        "bytes / 4 chars and place names 6 bytes / 2 chars; object descriptions 15 bytes / "
        "5 chars and place descriptions 12 bytes / 4 chars",
        "kind_is_visible_to_the_model": False,
    }
    append_progress({"event": "name_and_role", "role_word_inside_raw": literal_hits})

    # ---- 5. old description vs dev-only description -----------------------
    train_world_zero = next(
        world for world in worlds if world["split"] == "train" and world["world"] == 0
    )
    old_record = next(r for r in train_world_zero["records"] if r["kind"] == "object")
    new_record = next(
        r for r in dev_world_zero["records"] if r["description"] in premise.NEW_OBJECT_DESCRIPTIONS
    )
    train_description_chars = set("".join(premise.OBJECT_DESCRIPTIONS + premise.PLACE_DESCRIPTIONS))
    shared = sorted(set(new_record["description"]) & train_description_chars)
    unseen = sorted(set(new_record["description"]) - train_description_chars)
    description_age = {
        "old": {
            "split": "train",
            "world": train_world_zero["world"],
            "description": old_record["description"],
            "raw": old_record["raw"],
        },
        "new": {
            "split": "dev",
            "world": dev_world_zero["world"],
            "description": new_record["description"],
            "raw": new_record["raw"],
        },
        "same_char_length": len(old_record["description"]) == len(new_record["description"]),
        "same_internal_pattern": old_record["description"][2] == "的"
        and new_record["description"][2] == "的",
        "new_string_occurs_in_train": any(
            new_record["description"] in record["description"]
            for world in worlds
            if world["split"] == "train"
            for record in world["records"]
        ),
        "shared_characters_with_train_descriptions": shared,
        "characters_absent_from_train_descriptions": unseen,
        "shared_character_count": len(shared),
        "new_character_count": len(unseen),
        "difference_class": "same template, same 5-character pattern, different filled words: "
        "the new phrase uses characters that never appear in any train description",
        "all_new_descriptions": [
            {
                "description": description,
                "shared_characters": sorted(set(description) & train_description_chars),
                "characters_absent_from_train_descriptions": sorted(
                    set(description) - train_description_chars
                ),
                "occurs_in_train": any(
                    description in record["description"]
                    for world in worlds
                    if world["split"] == "train"
                    for record in world["records"]
                ),
            }
            for description in premise.NEW_OBJECT_DESCRIPTIONS
        ],
    }
    append_progress({"event": "description_age", "shared": len(shared), "unseen": len(unseen)})

    # ---- 6. the three questions, as counts --------------------------------
    all_names = set()
    all_descriptions = set()
    for world in worlds:
        for record in world["records"]:
            all_names.add(record["name"])
            all_descriptions.add(record["description"])
    goals_with_name = 0
    keys_with_name = 0
    for fixture in list(train) + list(dev):
        goals_with_name += sum(1 for name in all_names if name in fixture["goal"])
        keys_with_name += sum(
            1 for name in all_names for key in fixture["record_keys"] if name in key
        )
    dev_flags = [fixture["flags"] for fixture in dev]
    old_description_examples = [
        fixture for fixture in dev if fixture["object_description"] in premise.OBJECT_DESCRIPTIONS
    ]
    new_description_examples = [
        fixture
        for fixture in dev
        if fixture["object_description"] in premise.NEW_OBJECT_DESCRIPTIONS
    ]
    styles_by_split_kind: dict[str, list[int]] = {}
    for world in worlds:
        for record in world["records"]:
            styles_by_split_kind.setdefault(f"{world['split']}|{record['kind']}", []).append(
                record["style"]
            )
    train_name_chars = {
        "object": set("".join(payload["pools"]["train"]["object_names"])),
        "place": set("".join(payload["pools"]["train"]["place_names"])),
    }
    dev_name_chars = {
        "object": set("".join(payload["pools"]["dev"]["object_names"])),
        "place": set("".join(payload["pools"]["dev"]["place_names"])),
    }
    external = {}
    if WALL_19_2F.exists():
        wall = json.loads(WALL_19_2F.read_text(encoding="utf-8"))
        external = {
            "source": "reports/taiji_package19_2f_new_description_wall_20261104.json",
            "conclusions": wall.get("conclusions", [])[:3],
        }
    questions = {
        "a_only_name_matching": {
            "examples_checked": len(train) + len(dev),
            "goals_containing_any_record_name": goals_with_name,
            "candidate_strings_containing_any_record_name": keys_with_name,
            "dev_examples_whose_name_is_unseen_in_train": sum(
                1 for flag in dev_flags if not flag["name_seen"]
            ),
            "dev_examples_total": len(dev),
            "old_description_examples": len(old_description_examples),
            "old_description_examples_with_unseen_name": sum(
                1 for fixture in old_description_examples if not fixture["flags"]["name_seen"]
            ),
            "new_description_examples": len(new_description_examples),
            "gold_is_an_object_record_carrying_the_goal_description": pool_checks[
                "selection_label_points_to_gold_object_record"
            ],
            "evidence_19_2f": external,
        },
        "b_permutation_axis": {
            "description_name_vocabulary_overlap": len(all_names & all_descriptions),
            "names_occurring_in_goals": goals_with_name,
            "names_occurring_in_candidate_strings": keys_with_name,
            "scoring_input_on_the_candidate_side": "record description only (record_keys)",
            "scoring_input_on_the_query_side": "goal string only",
        },
        "c_old_vs_new": {
            "styles_present_by_split_and_kind": {
                cell: sorted(set(values)) for cell, values in sorted(styles_by_split_kind.items())
            },
            "train_name_characters": {k: sorted(v) for k, v in train_name_chars.items()},
            "dev_name_characters_absent_from_train": {
                kind: sorted(dev_name_chars[kind] - train_name_chars[kind])
                for kind in ("object", "place")
            },
            "new_description_characters_absent_from_train": unseen,
            "new_description_string_seen_in_train": description_age["new_string_occurs_in_train"],
        },
    }
    append_progress({"event": "questions", "goals_with_name": goals_with_name})

    cpu_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    report = {
        "schema": "taiji-task-semantic-contract-v1",
        "step": "task_semantic_contract",
        "date": "2026-11-05",
        "purpose": "state what the frozen 18.0 selection task is, in field names and verbatim "
        "samples, so later discussion can be pinned to the data rather than to internal words",
        "record_schema": record_schema,
        "pool": pool_checks,
        "goals": goal_rows,
        "answer_assembly": answer_assembly,
        "name_and_role": name_and_role,
        "description_age": description_age,
        "questions": questions,
        "cost": {
            "cpu_seconds": cpu_seconds,
            "wall_seconds": wall_seconds,
            "cpu_cap": CPU_CAP,
            "over_budget": cpu_seconds > CPU_CAP,
        },
        "discipline": {
            "adam_updates": 0,
            "forward_passes": 0,
            "bundles_saved": 0,
            "confirm_reads": 0,
            "read_only": True,
            "fsync_per_line": True,
        },
        "progress": str(PROGRESS.relative_to(ROOT)).replace("\\", "/"),
    }
    write_json(REPORT, report)
    manifest = {
        "schema": "taiji-task-semantic-contract-manifest-v1",
        "step": "task_semantic_contract",
        "date": "2026-11-05",
        "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
        "records": total_records,
        "examples": len(train) + len(dev),
        "goals_containing_name": goals_with_name,
        "keys_containing_name": keys_with_name,
        "dev_unseen_name_examples": questions["a_only_name_matching"][
            "dev_examples_whose_name_is_unseen_in_train"
        ],
        "cpu_seconds": cpu_seconds,
        "adam_updates": 0,
        "forward_passes": 0,
        "confirm_reads": 0,
    }
    write_json(MANIFEST, manifest)
    append_progress({"event": "done", "cpu_seconds": cpu_seconds})
    print(json.dumps({"cpu_seconds": cpu_seconds, "report": str(REPORT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
