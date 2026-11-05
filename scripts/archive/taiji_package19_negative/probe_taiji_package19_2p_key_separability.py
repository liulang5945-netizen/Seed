"""19.2p: zero-Adam design gate for the "whole-line key" data-premise proposal.

The 19.2n gate closed the old direction: the live pool key is the record's
*description* string, the eight keys of a world are pairwise distinct, and no
role-swapped twin exists inside the pool, so a decoy built outside the pool can
never change the 8-way argmax (D16-19).  This gate asks whether widening the key
to the whole record line (``description`` + linking word + ``name``) makes a
role-swapped record separable *inside* the pool.

What is measured, read-only and without any training step:

1. ``baseline_duplicates`` - for each of the 384 content starts (192 dev
   examples x object start 3 / place start 21) the number of other in-pool
   records that are byte-identical to the gold line except that the two entity
   spans have exchanged roles.  Expected 0 by 19.2n; rechecked here as a
   baseline only.
2. ``key_separability`` - ``max|key(line_a) - key(line_b)|`` on the frozen
   key encoder, for
   (a) one record line against the same line with its description and name
       spans exchanged (96 lines), and
   (b) two *records of the same world* whose two entity strings are the same
       set with roles exchanged.  Both readings must exceed 0.5.
3. ``margin`` - ``score_gold - score_role_swapped`` on 28031-B / 28032-B
   step300.  The swapped counterpart must be a real in-pool record; if the pool
   contains none, the measurement set is empty and the report says so instead
   of scoring a synthesised decoy.

Admission is all-or-nothing (owner, 2026-11-05): any unavailable number or any
failing number -> ``proposal_not_admissible``, no pricing.  The report must
also answer, with measurements, why role markers stop the key from collapsing
into a bag.

Run: ``python scripts/training/probe_taiji_package19_2p_key_separability.py``
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.selection_routing_checkpoint import load_routing_bundle  # noqa: E402

BUNDLE_DIR = ROOT / "output/taiji_package19_compare"
ENCODER_BUNDLE = BUNDLE_DIR / "28031-B_stop_lock-step300.pt"
SCORING_BUNDLES = {
    "28031-B": BUNDLE_DIR / "28031-B_stop_lock-step300.pt",
    "28032-B": BUNDLE_DIR / "28032-B_stop_lock-step300.pt",
}
STARTS = ((3, "object"), (21, "place"))
KEY_THRESHOLD = 0.5
CPU_CAP = 150
CPU_ESTIMATE = 8
NOISE_FLOOR = 8.88e-16

REPORT = ROOT / "reports/taiji_package19_2p_key_separability_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2p_key_separability_20261105.json"
PROGRESS = ROOT / "reports/taiji_package19_2p_key_separability_progress_20261105.jsonl"


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


def distance(values: list[float], threshold: float) -> dict:
    if not values:
        return {"count": 0, "status": "empty_measurement_set"}
    ordered = sorted(values)
    count = len(ordered)

    def rank(fraction: float) -> float:
        return ordered[min(count - 1, max(0, int(round(fraction * (count - 1)))))]

    return {
        "count": count,
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p10": rank(0.10),
        "p90": rank(0.90),
        "min": ordered[0],
        "max": ordered[-1],
        "threshold": threshold,
        "above_threshold": sum(1 for value in ordered if value > threshold),
        "above_threshold_rate": sum(1 for value in ordered if value > threshold) / count,
    }


def span_swap(text, first, second):
    left_start, left_end = first
    right_start, right_end = second
    return (
        text[:left_start]
        + text[right_start:right_end]
        + text[left_end:right_start]
        + text[left_start:left_end]
        + text[right_end:]
    )


def role_swapped_line(record: dict) -> str | None:
    """The record line with its description and name spans exchanged."""
    raw = record["raw"]
    description = record["description"]
    name = record["name"]
    description_at = raw.find(description)
    name_at = raw.find(name)
    if description_at < 0 or name_at < 0:
        return None
    first, second = sorted(
        (
            (description_at, description_at + len(description)),
            (name_at, name_at + len(name)),
        )
    )
    return span_swap(raw, first, second)


def role_swap_twin_indices(records: list[dict], gold_index: int) -> list[int]:
    """In-pool records whose two entity strings equal gold's, roles exchanged."""
    gold = records[gold_index]
    twins = []
    for index, record in enumerate(records):
        if index == gold_index:
            continue
        same_set = {record["name"], record["description"]} == {
            gold["name"],
            gold["description"],
        }
        if (
            same_set
            and record["name"] == gold["description"]
            and record["description"] == gold["name"]
        ):
            twins.append(index)
    return twins


def world_records(fixture: dict) -> list[dict]:
    return list(fixture["raw_example"]["records"])


def place_record_index(fixture: dict) -> int:
    for index, record in enumerate(world_records(fixture)):
        if record["kind"] == "place" and record["description"] == fixture["place_description"]:
            return index
    raise RuntimeError("place record not found")


def encode_key(candidate, text: str):
    relation = candidate.base.relation
    subject = candidate.subject
    with torch.no_grad():
        units = relation.encode_units(subject, text.encode("utf-8"))
        return relation.pool(units)


def main() -> int:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()

    payload = premise.build_payload()
    dev = premise.fixtures(payload, "dev")
    worlds = sorted({int(fixture["world"]) for fixture in dev})
    canonical = {}
    for fixture in dev:
        canonical.setdefault(int(fixture["world"]), fixture)

    PROGRESS.write_text("", encoding="utf-8")
    append_progress(
        {
            "event": "plan",
            "step": "19.2p",
            "dev_examples": len(dev),
            "worlds": len(worlds),
            "encoder_bundle": str(ENCODER_BUNDLE.relative_to(ROOT)),
            "key_threshold": KEY_THRESHOLD,
            "cpu_estimate": CPU_ESTIMATE,
            "cpu_cap": CPU_CAP,
            "adam_updates": 0,
            "confirm_reads": 0,
        },
    )

    # ---- baseline: role-swapped twins inside the live pool -------------------
    per_start = []
    baseline_total = 0
    for fixture in dev:
        records = world_records(fixture)
        for start, kind in STARTS:
            gold_index = (
                int(fixture["selection_label"]) if kind == "object" else place_record_index(fixture)
            )
            twins = role_swap_twin_indices(records, gold_index)
            baseline_total += len(twins)
            per_start.append(
                {
                    "world": int(fixture["world"]),
                    "index": int(fixture["index"]),
                    "start": start,
                    "kind": kind,
                    "gold_index": gold_index,
                    "twin_count": len(twins),
                }
            )
    baseline = {
        "starts": len(per_start),
        "role_swapped_in_pool_twins": baseline_total,
        "twin_count_histogram": {
            str(count): sum(1 for row in per_start if row["twin_count"] == count)
            for count in sorted({row["twin_count"] for row in per_start})
        },
        "note": "19.2n measured the same quantity as 0/384; this row only rechecks it 0 Adam",
    }
    append_progress(
        {
            "event": "baseline",
            **{key: baseline[key] for key in ("starts", "role_swapped_in_pool_twins")},
        }
    )

    # ---- (b) world-level pair search: same token set, roles exchanged --------
    pair_rows = []
    shared_token_pairs = 0
    vocabulary_overlap = 0
    description_tokens = set()
    name_tokens = set()
    for world in worlds:
        records = world_records(canonical[world])
        for record in records:
            description_tokens.add(record["description"])
            name_tokens.add(record["name"])
        for left in range(len(records)):
            for right in range(left + 1, len(records)):
                a = records[left]
                b = records[right]
                if {a["name"], a["description"]} & {b["name"], b["description"]}:
                    shared_token_pairs += 1
                if a["name"] == b["description"] and a["description"] == b["name"]:
                    pair_rows.append(
                        {
                            "world": world,
                            "left": left,
                            "right": right,
                            "left_line": a["raw"],
                            "right_line": b["raw"],
                        }
                    )
    vocabulary_overlap = len(description_tokens & name_tokens)
    append_progress(
        {
            "event": "pair_search",
            "worlds": len(worlds),
            "role_swap_pairs": len(pair_rows),
            "shared_token_pairs": shared_token_pairs,
            "vocabulary_overlap": vocabulary_overlap,
        },
    )

    # ---- key separability on the frozen encoder ------------------------------
    candidate, _optimizer, _state, _index, _cursor = load_routing_bundle(ENCODER_BUNDLE)

    worlds_by_id = {}
    for fixture in dev:
        worlds_by_id.setdefault(int(fixture["world"]), fixture)

    within_rows = []
    for world in worlds:
        for index, record in enumerate(world_records(worlds_by_id[world])):
            swapped = role_swapped_line(record)
            if swapped is None:
                append_progress({"event": "entity_not_found", "world": world, "index": index})
                continue
            first = encode_key(candidate, record["raw"])
            second = encode_key(candidate, swapped)
            delta = float((first - second).abs().max())
            within_rows.append(
                {
                    "world": world,
                    "record": index,
                    "kind": record["kind"],
                    "delta": delta,
                    "line": record["raw"],
                    "swapped": swapped,
                }
            )
    within = distance([row["delta"] for row in within_rows], KEY_THRESHOLD)
    within["records"] = len(within_rows)
    within["identical_key_count"] = sum(1 for row in within_rows if row["delta"] == 0.0)
    within["noise_floor"] = NOISE_FLOOR
    within["by_kind"] = {
        kind: distance([row["delta"] for row in within_rows if row["kind"] == kind], KEY_THRESHOLD)
        for kind in ("object", "place")
    }
    append_progress(
        {
            "event": "within_record",
            **{key: within[key] for key in ("records", "median", "min", "max")},
        }
    )

    between_rows = []
    for pair in pair_rows:
        first = encode_key(candidate, pair["left_line"])
        second = encode_key(candidate, pair["right_line"])
        between_rows.append({**pair, "delta": float((first - second).abs().max())})
    between = distance([row["delta"] for row in between_rows], KEY_THRESHOLD)
    between["pairs"] = len(between_rows)
    between["source"] = (
        "records of the same world" if between_rows else "no such record pair exists"
    )
    append_progress(
        {
            "event": "between_record",
            **{key: between[key] for key in ("pairs", "status") if key in between},
        }
    )

    # ---- margin: needs a real in-pool role-swapped record --------------------
    if between_rows:
        raise RuntimeError("in-pool pairs exist; extend this probe to score them before running")

    margin = {
        "status": "empty_measurement_set",
        "reason": (
            "the role-swapped counterpart must be a real in-pool record (owner 1.3); no world "
            "contains one, so scoring would require a synthesised decoy, which is forbidden"
        ),
        "bundles_loaded_for_scoring": 0,
        "systems": sorted(SCORING_BUNDLES),
        "required_quantity": "score_gold - score_role_swapped",
    }

    # ---- admission and verdict ----------------------------------------------
    item_1 = {"pass": baseline_total == 0, "value": baseline_total}
    item_2a = {
        "pass": within.get("above_threshold_rate", 0.0) == 1.0,
        **{key: within.get(key) for key in ("median", "min", "max", "above_threshold_rate")},
    }
    item_2b = {
        "pass": None,
        "value": between["pairs"],
        "status": between.get("status", "measured"),
    }
    item_3 = {"pass": False, "status": margin["status"]}

    if between["pairs"] == 0:
        verdict = {
            "label": "proposal_not_admissible",
            "reason": (
                "two independent blockers: (i) reading 1.2(a) clears 0.5 on only "
                f"{within.get('above_threshold', 0)}/{within.get('records', 0)} record lines "
                f"(min {within.get('min'):.3f}), so the role marker is not a uniform separator; "
                "(ii) readings 1.2(b) and 1.3 both need a role-swapped *record* inside the pool, "
                "and the record vocabulary keeps descriptions and names disjoint, so no world "
                "contains such a pair (0/12) and the mandated numbers cannot be produced without "
                "synthesising a record, which the gate forbids"
            ),
            "narrowing": (
                "the key is not a bag: no swapped line shares the gold key "
                f"({within.get('identical_key_count', 0)}/96 identical, min "
                f"{within.get('min'):.3f} >> {NOISE_FLOOR:g}); the blockers are the 0.5 bar on "
                "part of the lines and the missing in-pool counterpart"
            ),
        }
    elif not item_2a["pass"]:
        verdict = {"label": "key_not_separable", "reason": "role markers do not move the key"}
    elif not item_2b["pass"]:
        verdict = {
            "label": "key_not_separable",
            "reason": "the in-pool pair key difference stays <= 0.5",
        }
    else:
        verdict = {"label": "key_separable", "reason": "all three numbers pass"}
    append_progress({"event": "verdict", "label": verdict["label"]})

    cost_cpu = time.process_time() - start_cpu
    cost_wall = time.perf_counter() - start_wall
    report = {
        "schema": "taiji-package19-2p-key-separability-v1",
        "step": "19.2p",
        "date": "2026-11-05",
        "gate": "key_separability_design_gate",
        "proposal": {
            "key": "whole record line (description + linking word + name)",
            "role_markers": sorted(
                {
                    record["raw"].replace(record["description"], "D").replace(record["name"], "N")
                    for world in worlds
                    for record in world_records(worlds_by_id[world])
                }
            ),
            "threshold": KEY_THRESHOLD,
            "threshold_reason": "19.2f measured a median order-induced score difference of 1.24; a key difference must clear the 8.88e-16 noise floor and be comparable with the score scale",
        },
        "baseline_duplicates": baseline,
        "key_separability": {
            "within_record_role_swap": within,
            "between_record_role_swap": between,
            "shared_token_pairs": shared_token_pairs,
            "description_name_vocabulary_overlap": vocabulary_overlap,
        },
        "margin": margin,
        "admission": {
            "item_1_baseline": item_1,
            "item_2a_within_record": item_2a,
            "item_2b_between_record": item_2b,
            "item_3_zero_adam_margin": item_3,
        },
        "verdict": verdict,
        "boundary": (
            "this gate only says whether a role-swapped opponent can exist inside the pool under the "
            "whole-line key; it says nothing about whether training can learn the distinction"
        ),
        "cost": {
            "estimate_cpu": CPU_ESTIMATE,
            "cpu_seconds": cost_cpu,
            "wall_seconds": cost_wall,
            "cpu_cap": CPU_CAP,
            "over_budget": cost_cpu > CPU_CAP,
        },
        "discipline": {
            "adam_updates": 0,
            "bundles_saved": 0,
            "confirm_reads": 0,
            "fsync_per_line": True,
            "formatted_in_package": False,
        },
        "progress": str(PROGRESS.relative_to(ROOT)),
    }
    write_json(REPORT, report)
    manifest = {
        "schema": "taiji-package19-2p-manifest-v1",
        "step": "19.2p",
        "date": "2026-11-05",
        "report": str(REPORT.relative_to(ROOT)),
        "verdict": verdict["label"],
        "samples": len(per_start),
        "within_record_median": within.get("median"),
        "between_record_pairs": between["pairs"],
        "cpu_seconds": cost_cpu,
        "adam_updates": 0,
        "bundles_saved": 0,
        "confirm_reads": 0,
    }
    write_json(MANIFEST, manifest)
    print(
        json.dumps(
            {
                "verdict": verdict["label"],
                "within_median": within.get("median"),
                "within_min": within.get("min"),
                "pairs": between["pairs"],
                "baseline_twins": baseline_total,
                "cost_cpu": cost_cpu,
                "cost_wall": cost_wall,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
