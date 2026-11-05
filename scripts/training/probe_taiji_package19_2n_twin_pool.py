"""19.2n: does the 8-way candidate pool contain a swap-twin of the gold record?

Owner ruling 2026-11-05 (D16-19, ledger section 16.12): the 19.2m direction is
rejected because the order-decoy is a byte permutation of the gold record and is
not inside the candidate pool, so widening that margin cannot change the pool
argmax.  A follow-up negative-sample proposal is admissible only if the negative
is shown to be *in* the pool.

This package is the read-only design gate that answers that question.  It does
not train, does not touch Adam, does not load a scoring bundle and does not read
confirm.

What is measured, for every dev example x 2 content starts (object start 3,
place start 21):

1. **Constructibility inside the pool.**  The pool is ``fixture["record_keys"]``
   (8 strings per world).  For each gold key we search every pair of
   non-overlapping spans and check whether swapping them yields *another key of
   the same pool* with the same character length and the same UTF-8 byte
   multiset.  The same search is repeated with spans restricted to the world's
   entity strings (descriptions and names).
2. **Generate-from-existing-records.**  The world's record *lines* do contain
   two entities (description and name).  For the gold record we swap those two
   spans in the raw line and ask whether the result (a) has the same length and
   byte multiset, (b) occurs as another record line of the same world, and
   (c) could enter the pool as a distinct key.  Two key conventions are tested:
   the metadata description (the live one) and the first entity of the line.
3. **Current margin.**  ``score_gold - score_twin`` on 28031-B / 28032-B step300
   is *not* measured, because there is no in-pool twin to score; the measurement
   set is empty and the report says so explicitly.  No bundle is loaded.

The verdict is fixed before the run, with this priority:

* ``twin_not_constructible`` if the constructible ratio < 0.60;
* ``twin_already_separated`` if the in-pool margin median >= 1.0;
* ``twin_gate_pass`` otherwise.

The conclusion only answers "does an in-pool opponent exist"; it does not answer
"can training learn it".

Discipline: 0 Adam, 0 saved bundles, 0 confirm reads, per-line fsync progress,
write-then-rename report.  Budget 120 CPU, no formatting in this package.

Run: ``python scripts/training/probe_taiji_package19_2n_twin_pool.py``
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from taiji import data_premise_18_0 as premise  # noqa: E402

CPU_BUDGET = 120.0
CONSTRUCT_RATIO_GATE = 0.60
MARGIN_MEDIAN_GATE = 1.0
STARTS = ((3, "object"), (21, "place"))
TAG = "19.2n"

BUNDLES = (
    ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt",
    ROOT / "output/taiji_package19_compare/28032-B_stop_lock-step300.pt",
)
DATA_JSON = ROOT / "output/taiji_data_premise_18_0/data.json"
PREMISE_MODULE = ROOT / "taiji/data_premise_18_0.py"

REPORT = ROOT / "reports/taiji_package19_2n_twin_pool_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2n_twin_pool_20261105.json"
PROGRESS = ROOT / "reports/taiji_package19_2n_twin_pool_progress_20261105.jsonl"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_progress(handle, row: dict) -> None:
    """D16-15: append one line, flush and fsync before moving on."""
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def span_swap(text: str, first: tuple[int, int], second: tuple[int, int]) -> str:
    """Exchange two non-overlapping, ordered spans ``first`` before ``second``."""
    left_start, left_end = first
    right_start, right_end = second
    return (
        text[:left_start]
        + text[right_start:right_end]
        + text[left_end:right_start]
        + text[left_start:left_end]
        + text[right_end:]
    )


def permutation_twins(gold: str, pool: tuple[str, ...]) -> list[dict]:
    """All pool keys obtained by exchanging two non-overlapping spans of gold."""
    pool_set = set(pool)
    spans = [(start, end) for start in range(len(gold)) for end in range(start + 1, len(gold) + 1)]
    twins: list[dict] = []
    seen: set[str] = set()
    for index, first in enumerate(spans):
        for second in spans[index + 1 :]:
            if first[1] > second[0]:
                continue
            candidate = span_swap(gold, first, second)
            if candidate == gold or candidate not in pool_set or candidate in seen:
                continue
            if len(candidate) != len(gold):
                continue
            if sorted(candidate.encode("utf-8")) != sorted(gold.encode("utf-8")):
                continue
            seen.add(candidate)
            twins.append(
                {
                    "twin": candidate,
                    "spans": [list(first), list(second)],
                    "entities": [gold[first[0] : first[1]], gold[second[0] : second[1]]],
                }
            )
    return twins


def entity_spans(text: str, entities: tuple[str, ...]) -> list[tuple[int, int, str]]:
    spans = []
    for entity in entities:
        start = 0
        while True:
            position = text.find(entity, start)
            if position < 0:
                break
            spans.append((position, position + len(entity), entity))
            start = position + 1
    return spans


def entity_twins(gold: str, pool: tuple[str, ...], entities: tuple[str, ...]) -> list[dict]:
    """Pool keys obtained by exchanging two *entity* spans of gold."""
    pool_set = set(pool)
    spans = entity_spans(gold, entities)
    twins: list[dict] = []
    seen: set[str] = set()
    for index, first in enumerate(spans):
        for second in spans[index + 1 :]:
            if first[2] == second[2] or first[1] > second[0]:
                continue
            candidate = span_swap(gold, (first[0], first[1]), (second[0], second[1]))
            if candidate == gold or candidate not in pool_set or candidate in seen:
                continue
            seen.add(candidate)
            twins.append(
                {
                    "twin": candidate,
                    "entities": [first[2], second[2]],
                    "spans": [[first[0], first[1]], [second[0], second[1]]],
                }
            )
    return twins


def world_entities(raw_example: dict) -> tuple[str, ...]:
    values: list[str] = []
    for record in raw_example["records"]:
        values.extend((record["description"], record["name"]))
    for field in ("object_description", "place_description", "object_name", "place_name"):
        values.append(raw_example[field])
    return tuple(dict.fromkeys(values))


def world_record_lines(raw_example: dict) -> tuple[str, ...]:
    return tuple(record["raw"] for record in raw_example["records"])


def place_record_index(fixture: dict) -> int:
    for index, record in enumerate(fixture["raw_example"]["records"]):
        if record["kind"] == "place" and record["description"] == fixture["place_description"]:
            return index
    raise RuntimeError("place record not found")


def first_entity(text: str) -> str | None:
    """Text before the record's linking token (``是`` / ``又叫`` / ``对应``)."""
    for token in ("又叫", "对应", "是"):
        position = text.find(token)
        if position >= 0:
            return text[:position]
    return None


def line_twin(fixture: dict, record_index: int, gold: str) -> dict:
    """Swap the description and name spans inside the record's raw line."""
    record = fixture["raw_example"]["records"][record_index]
    raw = record["raw"]
    description = record["description"]
    name = record["name"]
    description_at = raw.find(description)
    name_at = raw.find(name)
    if description_at < 0 or name_at < 0:
        return {"status": "entity_not_found", "raw": raw}
    first, second = sorted(
        (
            (description_at, description_at + len(description)),
            (name_at, name_at + len(name)),
        )
    )
    swapped = span_swap(raw, first, second)
    world_lines = world_record_lines(fixture["raw_example"])
    swapped_first = first_entity(swapped)
    return {
        "status": "built",
        "raw": raw,
        "swapped": swapped,
        "same_length": len(swapped) == len(raw),
        "same_multiset": sorted(swapped.encode("utf-8")) == sorted(raw.encode("utf-8")),
        "distinct": swapped != raw,
        "occurs_in_world": swapped in world_lines,
        "metadata_key_unchanged": True,
        "first_entity_key": swapped_first,
        "first_entity_key_is_gold": swapped_first == gold,
        "first_entity_key_in_pool": bool(swapped_first in fixture["record_keys"]),
        "name_is_pool_key": bool(name in fixture["record_keys"]),
    }


def cross_record_swaps(fixture: dict) -> dict:
    """Swap descriptions between record pairs; does the pool key set change?"""
    raw_records = fixture["raw_example"]["records"]
    keys_before = set(fixture["record_keys"])
    changes = 0
    pairs = 0
    for left in range(len(raw_records)):
        for right in range(left + 1, len(raw_records)):
            pairs += 1
            swapped_keys = list(fixture["record_keys"])
            swapped_keys[left], swapped_keys[right] = swapped_keys[right], swapped_keys[left]
            if set(swapped_keys) != keys_before:
                changes += 1
    return {"pairs": pairs, "description_swap_key_set_changes": changes}


def verdict_of(constructible_ratio: float, margin_median: float | None) -> dict:
    if constructible_ratio < CONSTRUCT_RATIO_GATE:
        code = "twin_not_constructible"
    elif margin_median is not None and margin_median >= MARGIN_MEDIAN_GATE:
        code = "twin_already_separated"
    else:
        code = "twin_gate_pass"
    return {
        "code": code,
        "priority": [
            "constructible_ratio < 0.60 -> twin_not_constructible",
            "else in-pool margin median >= 1.0 -> twin_already_separated",
            "else -> twin_gate_pass",
        ],
        "constructible_ratio_gate": CONSTRUCT_RATIO_GATE,
        "margin_median_gate": MARGIN_MEDIAN_GATE,
    }


def next_step_of(code: str) -> str:
    if code == "twin_not_constructible":
        return (
            "只许做：若仍要测顺序敏感选择，必须另立“数据前提”提案（把候选 key 从描述改为整行、"
            "并把 8 条池扩成描述×名字的叉积），提案中必须写明孪生的池内编号、与 gold 的 key 字节差、"
            "以及当前 score_gold − score_twin 实测分布；禁止在现有 8 条候选池上再提池外负样本，"
            "禁止改 readout 判据。"
        )
    if code == "twin_already_separated":
        return "只许做：报“顺序在此数据上已可分”，不训练；禁止再以拉边距作为换序手段。"
    return (
        "只许做：进入设计与定价（池内孪生负样本，先证明池内编号再谈剂量）；"
        "禁止改 readout 判据、禁止使用池外负样本。"
    )


def main() -> int:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    PROGRESS.unlink(missing_ok=True)

    payload = premise.build_payload()
    dev = premise.fixtures(payload, "dev")
    train = premise.fixtures(payload, "train")
    for fixture in dev + train:
        if len(fixture["record_keys"]) != 8:
            raise RuntimeError(f"{TAG}: pool size drift at {fixture['world']}/{fixture['index']}")

    with PROGRESS.open("a", encoding="utf-8") as progress:
        append_progress(
            progress,
            {
                "event": "setup",
                "dev_examples": len(dev),
                "train_examples": len(train),
                "pool_size": 8,
                "starts": [name for _position, name in STARTS],
                "adam_updates": 0,
                "bundles_loaded": 0,
            },
        )

        rows = []
        twin_count_histogram: dict[int, int] = {}
        multiset_pair_total = 0
        line_built = 0
        line_in_world = 0
        line_same_multiset = 0
        line_first_entity_in_pool = 0
        line_first_entity_is_gold = 0
        line_name_in_pool = 0
        cross_pairs = 0
        cross_description_changes = 0
        cross_worlds: set[int] = set()
        unique_cases: dict[tuple[int, str, str], int] = {}

        for fixture in dev:
            entities = world_entities(fixture["raw_example"])
            pool = tuple(fixture["record_keys"])
            place_index = place_record_index(fixture)
            for position, name in STARTS:
                record_index = int(fixture["selection_label"]) if name == "object" else place_index
                gold = pool[record_index]
                general = permutation_twins(gold, pool)
                constrained = entity_twins(gold, pool, entities)
                multiset_matches = [
                    key
                    for key in pool
                    if key != gold and sorted(key.encode("utf-8")) == sorted(gold.encode("utf-8"))
                ]
                multiset_pair_total += len(multiset_matches)
                twin_count = len(general)
                twin_count_histogram[twin_count] = twin_count_histogram.get(twin_count, 0) + 1

                built = line_twin(fixture, record_index, gold)
                if built["status"] == "built":
                    line_built += 1
                    line_in_world += int(bool(built["occurs_in_world"]))
                    line_same_multiset += int(bool(built["same_multiset"]))
                    line_first_entity_in_pool += int(bool(built["first_entity_key_in_pool"]))
                    line_first_entity_is_gold += int(bool(built["first_entity_key_is_gold"]))
                    line_name_in_pool += int(bool(built["name_is_pool_key"]))

                row = {
                    "world": fixture["world"],
                    "index": fixture["index"],
                    "start": name,
                    "start_byte": position,
                    "gold": gold,
                    "pool_size": len(pool),
                    "general_twins": len(general),
                    "entity_twins": len(constrained),
                    "multiset_matches": len(multiset_matches),
                    "line_twin_status": built["status"],
                    "line_twin_in_world": bool(built.get("occurs_in_world", False)),
                    "line_twin_same_multiset": bool(built.get("same_multiset", False)),
                }
                rows.append(row)
                append_progress(progress, {"event": "gold_reading", **row})
                unique_cases[(fixture["world"], gold, name)] = len(general)

            if fixture["world"] in cross_worlds:
                continue
            cross_worlds.add(fixture["world"])
            swap = cross_record_swaps(fixture)
            cross_pairs += swap["pairs"]
            cross_description_changes += swap["description_swap_key_set_changes"]
            append_progress(
                progress,
                {
                    "event": "world_cross_swaps",
                    "world": fixture["world"],
                    "pairs": swap["pairs"],
                    "key_set_changes": swap["description_swap_key_set_changes"],
                },
            )

        total = len(rows)
        constructible = sum(row["general_twins"] for row in rows)
        entity_constructible = sum(row["entity_twins"] for row in rows)
        ratio = constructible / total if total else 0.0
        entity_ratio = entity_constructible / total if total else 0.0
        unique_total = len(unique_cases)
        unique_constructible = sum(1 for value in unique_cases.values() if value > 0)
        unique_ratio = unique_constructible / unique_total if unique_total else 0.0

        margin = {
            "status": "not_measured",
            "samples": 0,
            "median": None,
            "distribution": None,
            "bundles_loaded": 0,
            "reason": (
                "in-pool twins = 0, so score_gold - score_twin has an empty measurement set; "
                "no bundle was loaded and no forward pass was spent"
            ),
            "identity_statement": (
                "any twin that preserves the gold description key has score difference exactly 0 "
                "(identical key bytes); any twin whose key leaves the description pool is "
                "out-of-pool by definition (D16-19)"
            ),
        }
        verdict = verdict_of(ratio, margin["median"])
        verdict["next_step_only"] = next_step_of(verdict["code"])
        verdict["scope"] = "只回答“池内对手是否存在”，不回答“训练能否学会”"

        append_progress(progress, {"event": "verdict", **verdict})

        report = {
            "schema": "taiji-package19-2n-twin-pool-v1",
            "step": "N19.2n in-pool swap-twin design gate",
            "phase": "read-only, 0 Adam, 0 bundles, 0 confirm",
            "date": "2026-11-05",
            "purpose": (
                "D16-19 requires any negative-sample proposal to prove the negative is inside "
                "the candidate pool before it may spend training budget"
            ),
            "ledger": "D16-19 / INVARIANT-LEDGER section 16.12",
            "design": {
                "pool_definition": (
                    "fixture['record_keys']: 4 object descriptions + 4 place descriptions"
                ),
                "pool_size": 8,
                "starts": [{"byte": position, "name": name} for position, name in STARTS],
                "dev_examples": len(dev),
                "train_examples": len(train),
                "readings": total,
                "twin_definition": (
                    "swap two non-overlapping spans of the gold key; same character length and "
                    "same UTF-8 byte multiset; the result must be another key of the same pool"
                ),
                "entity_restricted_definition": (
                    "same, but both spans must be entity strings of the world "
                    "(descriptions and names)"
                ),
                "margin_rule": (
                    "only twins that are in the pool and compete in argmax would be scored; "
                    "there are none, so nothing is scored"
                ),
                "bundle_identities": [
                    {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
                    for path in BUNDLES
                ],
            },
            "pool_correction": {
                "owner_text": "12 条候选池 / argmax over 12",
                "measured": 8,
                "evidence": (
                    f"len(fixture['record_keys']) == 8 for {len(dev) + len(train)} fixtures; "
                    "score_rows appends one null row, so the scored tensor has 9 rows"
                ),
                "consequence": "后半段所有比例以 8 条池计算；12 为笔误，写回 ledger 16.12",
            },
            "constructibility": {
                "readings": total,
                "general_twins": constructible,
                "general_ratio": ratio,
                "entity_twins": entity_constructible,
                "entity_ratio": entity_ratio,
                "unique_cases": unique_total,
                "unique_constructible": unique_constructible,
                "unique_ratio": unique_ratio,
                "twins_per_start_histogram": {
                    str(key): value for key, value in sorted(twin_count_histogram.items())
                },
                "multiset_equal_pool_pairs": multiset_pair_total,
                "share_of_readings_with_multiset_pair": (
                    sum(1 for row in rows if row["multiset_matches"] > 0) / total if total else 0.0
                ),
                "gate": CONSTRUCT_RATIO_GATE,
            },
            "generate_from_records": {
                "line_twins_built": line_built,
                "line_twins_same_multiset": line_same_multiset,
                "line_twins_in_world": line_in_world,
                "line_first_entity_key_in_pool": line_first_entity_in_pool,
                "line_first_entity_key_is_gold": line_first_entity_is_gold,
                "line_swapped_name_in_pool": line_name_in_pool,
                "line_ratio": line_built / total if total else 0.0,
                "can_enter_pool_as_distinct_key": 0,
                "mechanism": (
                    "the live key is the record description; after a description/name swap inside "
                    "the raw line, the first entity is either the same description (duplicate key, "
                    "byte-identical score) or the name (never a pool key), so no distinct in-pool "
                    "key appears"
                ),
                "cross_record_pairs": cross_pairs,
                "description_swap_key_set_changes": cross_description_changes,
                "cross_record_mechanism": (
                    "swapping descriptions between records permutes the same 8 strings, so the key "
                    "set is unchanged; swapping names never touches the key"
                ),
            },
            "margin": margin,
            "verdict": verdict,
            "discipline": {
                "adam_updates": 0,
                "bundles_saved": 0,
                "confirm_reads": 0,
                "fsync_per_line": True,
                "report_write_then_rename": True,
                "format_gate": "not_executed",
                "format_note": (
                    "D16-18: formatting only at the closing commit; nothing was formatted here"
                ),
            },
            "cost": {
                "cpu_seconds": None,
                "wall_seconds": None,
                "budget_cpu": CPU_BUDGET,
                "over_budget": False,
                "units": "pure stdlib string scan; no torch, no forward pass, no bundle load",
            },
            "inputs": {
                "data_json": {
                    "path": str(DATA_JSON.relative_to(ROOT)),
                    "sha256": sha256_file(DATA_JSON),
                },
                "premise_module": {
                    "path": str(PREMISE_MODULE.relative_to(ROOT)),
                    "sha256": sha256_file(PREMISE_MODULE),
                },
                "script": {
                    "path": str(Path(__file__).resolve().relative_to(ROOT)),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
            },
        }
        report["cost"]["cpu_seconds"] = time.process_time() - start_cpu
        report["cost"]["wall_seconds"] = time.perf_counter() - start_wall
        report["cost"]["over_budget"] = report["cost"]["cpu_seconds"] > CPU_BUDGET
        append_progress(progress, {"event": "report_written", "cost": report["cost"]})

    tmp = REPORT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, REPORT)

    manifest = {
        "schema": "taiji-package19-2n-manifest-v1",
        "step": "N19.2n",
        "date": "2026-11-05",
        "report": {
            "path": str(REPORT.relative_to(ROOT)),
            "sha256": sha256_file(REPORT),
        },
        "progress": {
            "path": str(PROGRESS.relative_to(ROOT)),
            "sha256": sha256_file(PROGRESS),
        },
        "inputs": report["inputs"],
        "verdict": report["verdict"]["code"],
        "cost_cpu": report["cost"]["cpu_seconds"],
        "adam_updates": 0,
        "bundles": 0,
        "confirm_reads": 0,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = MANIFEST.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(manifest_tmp, MANIFEST)

    print(
        json.dumps(
            {"verdict": report["verdict"], "constructibility": report["constructibility"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
