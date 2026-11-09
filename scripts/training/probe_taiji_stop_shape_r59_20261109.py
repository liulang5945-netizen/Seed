"""R59: frozen-criterion dev probe for where short fields bend the Task1 slots.

R58 showed the structural stop rule generalises across name lengths, but the
two-hop answer content is only byte-exact for 9/9, 12/12 and 12/6 B names; 3/3
and 6/6 B decode to the declared length with wrong content.  This gate asks two
narrow questions with pre-registered answers:

* **A (shape localisation)** - keep the two-hop template, cross object/place
  name lengths over the four cells ``(3,12) (12,3) (6,12) (12,6)`` and classify
  the ``byte_exact`` pattern into ``total_length_threshold`` /
  ``object_slot_short`` / ``place_slot_short`` / ``unexplained``.
* **B (word effect, shape fixed at 12/6)** - object name, object description and
  place description are held constant; only the 2-char place word changes over
  the 30 words published in the R57 trace.  The stop edge at byte 27 reports the
  EOS rank; a deterministic one-sided exact permutation test (C(30,5)=142506
  splits) asks whether the 5 ``*尾`` words rank systematically higher.

Discipline: dev/synthetic only, 0 training, 0 parameter change, the sealed
confirm holdout report is never read, and the 192/192 reading is untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "scripts" / "training"
for entry in (ROOT, TRAINING):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import torch  # noqa: E402

torch.set_num_threads(1)

import probe_taiji_stop_shape_r58_20261109 as r58  # noqa: E402

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.copy_stop_boundary import field_bounds  # noqa: E402

MANIFEST = ROOT / "plans/manifests/taiji_stop_shape_r59_20261109.json"
REPORT = ROOT / "reports/taiji_stop_shape_r59_20261109.json"
PROGRESS = ROOT / "reports/taiji_stop_shape_r59_20261109_progress.jsonl"
BUNDLE = ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt"
WORD_SOURCE = ROOT / "reports/taiji_stop_boundary_20261109.json"

CPU_CAP = 200.0
WALL_CAP = 1800.0
SINGLE_RUN_ESTIMATE = 45.0
HOPS = (1, 2, 3)
NAME_BYTES = (3, 6, 9, 12)
GROUP_A_CELLS = ((3, 12), (12, 3), (6, 12), (12, 6))
# (12, 6) reuses R58's control fixture index 4 so the control is a direct re-read.
GROUP_A_FIXTURE_INDEX = {(3, 12): 0, (12, 3): 1, (6, 12): 2, (12, 6): 4}
GROUP_B_OBJECT_NAME = "乌木罗盘"
GROUP_B_OBJECT_DESCRIPTION = "古旧木制器具"
GROUP_B_PLACE_DESCRIPTION = "旧年石砌场所"
GROUP_B_WORDS = (
    "井前",
    "亭畔",
    "亭角",
    "兰上",
    "圃外",
    "崖前",
    "廊前",
    "松上",
    "松前",
    "池前",
    "池外",
    "沙外",
    "溪前",
    "瓮尾",
    "瓮沿",
    "竹上",
    "舟侧",
    "雾尾",
    "井北",
    "井尾",
    "井畔",
    "兰外",
    "圃下",
    "圃尾",
    "廊北",
    "桥尾",
    "梅角",
    "瓮北",
    "竹沿",
    "雾外",
)
GROUP_B_TAIL = ("瓮尾", "雾尾", "井尾", "圃尾", "桥尾")
FROZEN_FILES = (
    "scripts/training/probe_taiji_stop_shape_r59_20261109.py",
    "scripts/training/probe_taiji_stop_shape_r58_20261109.py",
    "scripts/training/probe_taiji_content_transport_18_1.py",
    "taiji/copy_stop_boundary.py",
    "taiji/binding_copy.py",
    "taiji/record_relation_core.py",
    "taiji/living_core.py",
    "taiji/data_premise_18_0.py",
    "taiji/selection_routing.py",
    "taiji/selection_routing_checkpoint.py",
)

DEFINITION = {
    "gate": "R59 frozen-criterion dev probe: slot localisation + same-shape word effect",
    "group_a": {
        "template": "2-hop 把{object}放到{place}; records written as {desc}是{name}。 and {desc}对应{name}。",
        "names": "R58 GLYPHS slices (each Han char is 3 bytes)",
        "cells": [[3, 12], [12, 3], [6, 12], [12, 6]],
        "fixture_assignment": {
            "3x12": 0,
            "12x3": 1,
            "6x12": 2,
            "12x6": 4,
            "note": "12x6 re-reads R58's control fixture index 4",
        },
        "classifier": {
            "total_length_threshold": "(3,12) fails AND (12,3) fails AND (6,12) passes AND (12,6) passes",
            "object_slot_short": "(3,12) fails AND (12,3) passes AND (12,6) passes",
            "place_slot_short": "(12,3) fails AND (3,12) passes AND (12,6) passes",
            "unexplained": "anything else; all four booleans are reported",
            "note": "(6,12) is reported separately as the 6-byte-short-field reading",
        },
    },
    "group_b": {
        "shape": {"object_bytes": 12, "place_bytes": 6, "answer_bytes": 27},
        "object_name": GROUP_B_OBJECT_NAME,
        "object_description": GROUP_B_OBJECT_DESCRIPTION,
        "place_description": GROUP_B_PLACE_DESCRIPTION,
        "words": list(GROUP_B_WORDS),
        "tail_words": list(GROUP_B_TAIL),
        "word_source": "reports/taiji_stop_boundary_20261109.json trace (R57 published list)",
        "statistic": "EOS rank (1-based, descending) at the place stop edge (byte position 27); larger rank = more desire to continue",
        "decision_rules": [
            "complete separation: min(tail rank) > max(non-tail rank) -> word_effect_systematic",
            "otherwise exact one-sided permutation test on the mean-rank difference, C(30,5)=142506 splits: p < 0.05 and median(tail) > median(non-tail) -> word_effect_systematic",
            "p >= 0.05 and not separated -> word_effect_absent",
            "otherwise (direction right but p undecidable) -> word_effect_weak",
        ],
    },
    "gates": {
        "structural_pass": "12 structural cells exact AND every A/B cell field_lengths_ok",
        "zero_extra_pass": "all four A cells stop at the declared end with zero extra bytes",
        "locate_class": "A classifier output",
        "word_effect": "B decision output",
        "failing_axes": "names of failed boolean gates",
    },
    "scope": "dev/synthetic only; the sealed confirm holdout report is not read; no change to 192/192",
    "estimate_rule": "D-64: single run x 3 (one run plus two replay margins) + report assembly",
}


def append_progress(record: dict) -> None:
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def definition_digest() -> str:
    body = json.dumps(DEFINITION, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def file_hashes() -> dict:
    return {name: sha256_of(ROOT / name) for name in FROZEN_FILES}


def shaped_fixture(
    base_fixture: dict,
    object_description: str,
    place_description: str,
    object_name: str,
    place_name: str,
) -> tuple[dict, bytes, bytes]:
    object_raw = r58.record_for(object_name, object_description, "是").encode()
    place_raw = r58.record_for(place_name, place_description, "对应").encode()
    records = list(base_fixture["records"])
    records[0] = r58.with_raw(records[0], object_raw)
    records[1] = r58.with_raw(records[1], place_raw)
    fixture = dict(base_fixture)
    fixture["records"] = tuple(records)
    fixture["goal"] = f"让{object_description}到{place_description}。"
    fixture["object_description"] = object_description
    fixture["place_description"] = place_description
    fixture["object_name"] = object_name
    fixture["place_name"] = place_name
    keys = list(base_fixture["record_keys"])
    keys[0] = object_description
    keys[1] = place_description
    fixture["record_keys"] = keys
    return fixture, object_raw, place_raw


def shape_for(
    object_name: str,
    place_name: str,
    object_raw: bytes,
    place_raw: bytes,
    object_description: str,
    place_description: str,
) -> dict:
    object_offset, object_copy_stop = field_bounds(object_raw.decode("utf-8"), object_description)
    place_offset, place_copy_stop = field_bounds(place_raw.decode("utf-8"), place_description)
    object_bytes = len(object_name.encode("utf-8"))
    place_bytes = len(place_name.encode("utf-8"))
    object_start = 3  # "把"
    object_stop = object_start + object_bytes
    place_start = object_stop + 6  # "放到"
    place_stop = place_start + place_bytes
    return {
        "object_start": object_start,
        "object_stop": object_stop,
        "place_start": place_start,
        "place_stop": place_stop,
        "stop": place_stop,
        "object_route": 0,
        "place_route": 1,
        "object_name_offset": object_offset,
        "place_name_offset": place_offset,
        "object_copy_stop": object_copy_stop,
        "place_copy_stop": place_copy_stop,
        "field_lengths_ok": (
            object_copy_stop - object_offset == object_bytes
            and place_copy_stop - place_offset == place_bytes
        ),
        "target": f"把{object_name}放到{place_name}".encode(),
    }


def decode_cell(candidate, probe, fixture: dict, shape: dict) -> dict:
    decoded = r58.decode_shape(candidate, probe, fixture, shape, traced=True)
    raw = decoded["raw"]
    return {
        "raw": raw,
        "terminated": decoded["terminated"],
        "traces": decoded["traces"],
        "decoded_text": bytes(raw).decode("utf-8", "replace"),
        "answer_bytes": len(raw),
        "zero_extra": len(raw) == shape["stop"] and decoded["terminated"],
        "byte_exact": tuple(raw) == tuple(shape["target"]),
    }


def edge_or_none(decoded: dict, position: int) -> dict | None:
    trace = decoded["traces"].get(str(position))
    return dict(trace) if trace is not None else None


def classify_a(byte_exact: dict[tuple[int, int], bool]) -> str:
    if (
        not byte_exact[(3, 12)]
        and not byte_exact[(12, 3)]
        and byte_exact[(6, 12)]
        and byte_exact[(12, 6)]
    ):
        return "total_length_threshold"
    if not byte_exact[(3, 12)] and byte_exact[(12, 3)] and byte_exact[(12, 6)]:
        return "object_slot_short"
    if not byte_exact[(12, 3)] and byte_exact[(3, 12)] and byte_exact[(12, 6)]:
        return "place_slot_short"
    return "unexplained"


def exact_permutation_p(ranks: list[int], tail_index: list[int]) -> float:
    """One-sided exact p-value for mean(tail) - mean(non-tail), integer scaled.

    The statistic is multiplied by ``len(ranks) * len(tail)`` so every comparison
    is exact integer arithmetic; the enumeration order of ``combinations`` is
    deterministic, so the p-value is reproducible bit for bit.
    """
    count = len(ranks)
    size = len(tail_index)
    total_sum = sum(ranks)
    tail_sum = sum(ranks[index] for index in tail_index)
    observed = count * tail_sum - size * total_sum
    hits = 0
    total = 0
    for combo in combinations(range(count), size):
        split_sum = sum(ranks[index] for index in combo)
        if count * split_sum - size * total_sum >= observed:
            hits += 1
        total += 1
    return hits / total


def describe_ranks(rows: list[dict]) -> dict:
    ranks = [row["eos_rank"] for row in rows if row["eos_rank"] is not None]
    return {
        "words": len(rows),
        "measured": len(ranks),
        "argmax_is_eos": sum(1 for row in rows if row["argmax_label"] == "<EOS>"),
        "rank_min": min(ranks) if ranks else None,
        "rank_median": statistics.median(ranks) if ranks else None,
        "rank_max": max(ranks) if ranks else None,
        "rank_mean": statistics.fmean(ranks) if ranks else None,
    }


def word_effect_decision(rows: list[dict]) -> dict:
    tail_rows = [row for row in rows if row["group"] == "tail"]
    other_rows = [row for row in rows if row["group"] == "non_tail"]
    summary = {
        "tail": describe_ranks(tail_rows),
        "non_tail": describe_ranks(other_rows),
        "exact_permutation": {"splits": None, "p_value": None},
    }
    ranks = [row["eos_rank"] for row in rows]
    if any(rank is None for rank in ranks):
        summary["verdict"] = "word_effect_unmeasurable"
        summary["separated"] = None
        summary["missing_words"] = [row["place_word"] for row in rows if row["eos_rank"] is None]
        return summary
    tail_index = [index for index, row in enumerate(rows) if row["group"] == "tail"]
    tail_ranks = [ranks[index] for index in tail_index]
    other_ranks = [rank for index, rank in enumerate(ranks) if index not in set(tail_index)]
    separated = min(tail_ranks) > max(other_ranks)
    p_value = exact_permutation_p(ranks, tail_index)
    median_tail = statistics.median(tail_ranks)
    median_other = statistics.median(other_ranks)
    splits = 1
    for step in range(len(tail_index)):
        splits = splits * (len(ranks) - step) // (step + 1)
    summary["exact_permutation"] = {
        "splits": splits,
        "p_value": p_value,
        "mean_difference": statistics.fmean(tail_ranks) - statistics.fmean(other_ranks),
    }
    summary["separated"] = separated
    summary["median_tail"] = median_tail
    summary["median_non_tail"] = median_other
    if separated or (p_value < 0.05 and median_tail > median_other):
        summary["verdict"] = "word_effect_systematic"
    elif p_value >= 0.05:
        summary["verdict"] = "word_effect_absent"
    else:
        summary["verdict"] = "word_effect_weak"
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    started_cpu = time.process_time()
    started_wall = time.perf_counter()

    if args.freeze:
        if MANIFEST.exists():
            raise RuntimeError(f"preregistration already exists: {MANIFEST}")
        write_json(
            MANIFEST,
            {
                "schema": "taiji-stop-shape-r59-manifest-v1",
                "step": "R59",
                "date": "2026-11-09",
                "definition": DEFINITION,
                "definition_digest": definition_digest(),
                "frozen_files": file_hashes(),
                "bundle": str(BUNDLE.relative_to(ROOT)),
                "bundle_sha256": sha256_of(BUNDLE),
                "word_source": {
                    "path": str(WORD_SOURCE.relative_to(ROOT)),
                    "sha256": sha256_of(WORD_SOURCE),
                },
                "cost": {
                    "single_run_estimate": SINGLE_RUN_ESTIMATE,
                    "estimate_with_replays": SINGLE_RUN_ESTIMATE * 3,
                    "cpu_cap": CPU_CAP,
                    "wall_cap": WALL_CAP,
                },
                "immutability": "written once before the run; never rewritten",
                "sealed_split": "not read",
            },
        )
        print(json.dumps({"frozen": str(MANIFEST)}, ensure_ascii=False))
        return 0

    if not MANIFEST.exists():
        raise RuntimeError("preregistration missing; run --freeze first")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["definition_digest"] != definition_digest():
        raise RuntimeError("definition digest drift")
    drifted = [
        name for name, value in manifest["frozen_files"].items() if sha256_of(ROOT / name) != value
    ]
    if drifted:
        raise RuntimeError(f"frozen input drift: {drifted}")
    if manifest["bundle_sha256"] != sha256_of(BUNDLE):
        raise RuntimeError("bundle drift")
    if PROGRESS.exists() and PROGRESS.stat().st_size:
        events = [
            json.loads(line)
            for line in PROGRESS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(event.get("event") == "finish" for event in events):
            raise RuntimeError("R59 already finished")
    else:
        PROGRESS.write_text("", encoding="utf-8")
    append_progress({"event": "start", "definition_digest": definition_digest()})

    structural = [r58.structural_cell(hops, nbytes) for hops in HOPS for nbytes in NAME_BYTES]
    append_progress(
        {
            "event": "structural",
            "cells": len(structural),
            "exact": sum(1 for cell in structural if cell["exact"]),
        }
    )

    base = premise.build_payload()
    pools = premise.split_pools()
    worlds = [
        premise.build_world("dev", world, pools) for world in range(premise.SPLIT_WORLDS["dev"])
    ]
    materialised = dict(base)
    materialised["worlds"] = [
        {key: value for key, value in world.items() if key != "examples"} for world in worlds
    ]
    materialised["examples"] = [example for world in worlds for example in world["examples"]]
    dev = tuple(premise.fixtures(materialised, "dev"))

    candidate, probe = r58.load_candidate()

    group_a_rows: list[dict] = []
    for object_bytes, place_bytes in GROUP_A_CELLS:
        base_fixture = dev[GROUP_A_FIXTURE_INDEX[(object_bytes, place_bytes)]]
        object_name = r58.names_for(1, object_bytes)[0]
        place_name = r58.names_for(1, place_bytes, offset=4)[0]
        fixture, object_raw, place_raw = shaped_fixture(
            base_fixture,
            base_fixture["object_description"],
            base_fixture["place_description"],
            object_name,
            place_name,
        )
        shape = shape_for(
            object_name,
            place_name,
            object_raw,
            place_raw,
            fixture["object_description"],
            fixture["place_description"],
        )
        decoded = decode_cell(candidate, probe, fixture, shape)
        row = {
            "object_bytes": object_bytes,
            "place_bytes": place_bytes,
            "role": "control_task1_shape" if (object_bytes, place_bytes) == (12, 6) else "cell",
            "fixture_index": GROUP_A_FIXTURE_INDEX[(object_bytes, place_bytes)],
            "object_name": object_name,
            "place_name": place_name,
            "object_description": fixture["object_description"],
            "place_description": fixture["place_description"],
            "target_text": shape["target"].decode("utf-8"),
            "decoded_text": decoded["decoded_text"],
            "expected_bytes": len(shape["target"]),
            "answer_bytes": decoded["answer_bytes"],
            "zero_extra": decoded["zero_extra"],
            "byte_exact": decoded["byte_exact"],
            "field_lengths_ok": shape["field_lengths_ok"],
            "object_edge": edge_or_none(decoded, shape["object_stop"]),
            "place_edge": edge_or_none(decoded, shape["place_stop"]),
        }
        group_a_rows.append(row)
        append_progress(
            {
                "event": "group_a",
                "object_bytes": object_bytes,
                "place_bytes": place_bytes,
                "zero_extra": row["zero_extra"],
                "byte_exact": row["byte_exact"],
                "cpu_seconds": round(time.process_time() - started_cpu, 4),
            }
        )

    group_b_rows: list[dict] = []
    base_fixture = dev[0]
    for place_word in GROUP_B_WORDS:
        fixture, object_raw, place_raw = shaped_fixture(
            base_fixture,
            GROUP_B_OBJECT_DESCRIPTION,
            GROUP_B_PLACE_DESCRIPTION,
            GROUP_B_OBJECT_NAME,
            place_word,
        )
        shape = shape_for(
            GROUP_B_OBJECT_NAME,
            place_word,
            object_raw,
            place_raw,
            GROUP_B_OBJECT_DESCRIPTION,
            GROUP_B_PLACE_DESCRIPTION,
        )
        decoded = decode_cell(candidate, probe, fixture, shape)
        trace = edge_or_none(decoded, shape["place_stop"]) or {}
        row = {
            "place_word": place_word,
            "last_char": place_word[-1],
            "group": "tail" if place_word in GROUP_B_TAIL else "non_tail",
            "target_text": shape["target"].decode("utf-8"),
            "decoded_text": decoded["decoded_text"],
            "answer_bytes": decoded["answer_bytes"],
            "zero_extra": decoded["zero_extra"],
            "byte_exact": decoded["byte_exact"],
            "field_lengths_ok": shape["field_lengths_ok"],
            "argmax_byte": trace.get("argmax_byte"),
            "argmax_label": trace.get("argmax_label"),
            "masked": trace.get("masked"),
            "pre_mask_label": trace.get("pre_mask_label"),
            "emitted_label": trace.get("emitted_label"),
            "eos_probability": trace.get("eos_probability"),
            "eos_rank": trace.get("eos_rank"),
            "continuation_desire": trace.get("continuation_desire"),
            "copy_mass_total": trace.get("copy_mass_total"),
        }
        group_b_rows.append(row)
        append_progress(
            {
                "event": "group_b",
                "place_word": place_word,
                "byte_exact": row["byte_exact"],
                "eos_rank": row["eos_rank"],
                "cpu_seconds": round(time.process_time() - started_cpu, 4),
            }
        )

    byte_exact_by_cell = {
        f"{row['object_bytes']}x{row['place_bytes']}": bool(row["byte_exact"])
        for row in group_a_rows
    }
    locate_class = classify_a(
        {(row["object_bytes"], row["place_bytes"]): bool(row["byte_exact"]) for row in group_a_rows}
    )
    short_six_b = next(
        row for row in group_a_rows if (row["object_bytes"], row["place_bytes"]) == (6, 12)
    )
    if short_six_b["byte_exact"]:
        short_six_b_reading = "6 B object field plus 12 B place field is byte exact"
    elif short_six_b["zero_extra"]:
        short_six_b_reading = (
            "6 B object field stops at the declared end but content is not byte exact"
        )
    else:
        short_six_b_reading = "6 B object field does not stop at the declared end"

    word_summary = word_effect_decision(group_b_rows)
    word_effect = word_summary["verdict"]

    structural_pass = all(cell["exact"] for cell in structural) and all(
        row["field_lengths_ok"] for row in group_a_rows + group_b_rows
    )
    zero_extra_pass = all(row["zero_extra"] for row in group_a_rows)
    failing_axes = []
    if not structural_pass:
        failing_axes.append("structural")
    if not zero_extra_pass:
        failing_axes.append("zero_extra")
    if word_effect == "word_effect_unmeasurable":
        failing_axes.append("word_effect_unmeasurable")

    cpu_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    over_budget = cpu_seconds > CPU_CAP or wall_seconds > WALL_CAP

    report = {
        "schema": "taiji-stop-shape-r59-report-v1",
        "step": "R59",
        "date": "2026-11-09",
        "relation_to_main_task": (
            "界定 Task1 的 27 字节答案里，3/6 B 短字段把槽位挤歪发生在哪一槽；"
            "并检验形状固定 12/6、只换地名时停止边是否系统性变化。不改 Task1 的 192/192 读数。"
        ),
        "preregistration": str(MANIFEST.relative_to(ROOT)),
        "preregistration_sha256": sha256_of(MANIFEST),
        "definition": DEFINITION,
        "scope": {
            "sealed_split": "not read",
            "split": "dev/synthetic",
            "fixtures_used": "12 dev worlds materialised; 4 group-A cells + 30 group-B cells",
            "examples_used": len(group_a_rows) + len(group_b_rows),
            "confirm_holdout_report_read": False,
        },
        "structural": {
            "cells": structural,
            "exact": sum(1 for cell in structural if cell["exact"]),
            "total": len(structural),
        },
        "group_a": {
            "cells": group_a_rows,
            "classification": {
                "locate_class": locate_class,
                "byte_exact_by_cell": byte_exact_by_cell,
                "short_six_b_field": short_six_b_reading,
            },
            "word_source_note": "group A uses the dev fixture descriptions, as in R58",
        },
        "group_b": {
            "setup": {
                "object_name": GROUP_B_OBJECT_NAME,
                "object_description": GROUP_B_OBJECT_DESCRIPTION,
                "place_description": GROUP_B_PLACE_DESCRIPTION,
                "goal": f"让{GROUP_B_OBJECT_DESCRIPTION}到{GROUP_B_PLACE_DESCRIPTION}。",
                "words": list(GROUP_B_WORDS),
                "tail_words": list(GROUP_B_TAIL),
                "word_source": str(WORD_SOURCE.relative_to(ROOT)),
                "word_source_sha256": sha256_of(WORD_SOURCE),
                "word_source_note": (
                    "the 30 words are the R57 published trace list; they are also the confirm "
                    "place-name pool, but no confirm fixture or label is generated or read"
                ),
                "description_held_constant": True,
            },
            "cells": group_b_rows,
            "statistics": word_summary,
        },
        "gates": {
            "structural_pass": structural_pass,
            "zero_extra_pass": zero_extra_pass,
            "locate_class": locate_class,
            "word_effect": word_effect,
            "failing_axes": failing_axes,
            "field_lengths_ok": {
                "group_a": all(row["field_lengths_ok"] for row in group_a_rows),
                "group_b": all(row["field_lengths_ok"] for row in group_b_rows),
            },
            "structural_cells_exact": f"{sum(1 for cell in structural if cell['exact'])}/{len(structural)}",
        },
        "outcome": {
            "locate_class": locate_class,
            "word_effect": word_effect,
            "note": (
                "A 组只判短字段把槽位挤歪发生在哪一槽；B 组只判在形状完全相同、只换地点词时"
                "停止边是否系统性变化；两者都不读 confirm、都不改 192/192。"
            ),
        },
        "cost": {
            "cpu_seconds": round(cpu_seconds, 2),
            "wall_seconds": round(wall_seconds, 2),
            "single_run_estimate": SINGLE_RUN_ESTIMATE,
            "estimate_with_replays": SINGLE_RUN_ESTIMATE * 3,
            "cpu_cap": CPU_CAP,
            "wall_cap": WALL_CAP,
            "over_budget": bool(over_budget),
        },
        "discipline": {
            "training_runs": 0,
            "optimizer_steps": 0,
            "parameters_added": 0,
            "graph_changes": 0,
            "confirm_reads": 0,
            "confirm_holdout_report_reads": 0,
        },
        "implementation_choices": [
            "R59 imports the R58 runner module and reuses decode_shape/edge_report/structural_cell/load_candidate instead of copying them.",
            "Group A fixture assignment is (3,12)->dev[0], (12,3)->dev[1], (6,12)->dev[2], (12,6)->dev[4]; dev[4] is R58's control fixture.",
            "Group B uses dev[0] and overwrites records[0]/records[1]; the goal is rebuilt from the constant descriptions.",
            "The exact permutation test compares integer-scaled mean differences, so no floating-point tie handling is involved.",
            "If any group-B stop edge lacks the position-27 probe, the verdict is word_effect_unmeasurable instead of silently dropping the word.",
            "locate_class and word_effect are classification readings, not pass/fail gates; failing_axes lists only failed boolean gates.",
        ],
        "closed_state": {"frozen_files_unchanged": file_hashes() == manifest["frozen_files"]},
    }
    write_json(REPORT, report)
    append_progress(
        {
            "event": "finish",
            "locate_class": locate_class,
            "word_effect": word_effect,
            "cpu_seconds": round(cpu_seconds, 4),
            "wall_seconds": round(wall_seconds, 4),
        }
    )
    print(
        json.dumps(
            {
                "locate_class": locate_class,
                "word_effect": word_effect,
                "byte_exact_by_cell": byte_exact_by_cell,
                "a_zero_extra": [row["zero_extra"] for row in group_a_rows],
                "word_effect_stats": {
                    "p_value": word_summary["exact_permutation"]["p_value"],
                    "separated": word_summary["separated"],
                    "tail": word_summary["tail"],
                    "non_tail": word_summary["non_tail"],
                },
                "structural_exact": f"{report['structural']['exact']}/{report['structural']['total']}",
                "cost": report["cost"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
