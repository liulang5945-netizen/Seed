"""R60: frozen-criterion dev probe for the object-slot byte-length sweep.

R59 localised the short-field failure to the object slot and showed the stop
edge is decided by the whole name string, not by the shape and not by the last
character.  The queue head therefore asks one narrow question: with the place
field held at 6 B and both descriptions held constant, does the object name
need to be exactly the trained 12 B, or is >=9 B already byte exact?

Two stems are swept over 6/9/12/15 B.  Each stem is a nested prefix family, so
the only thing that changes along a stem is the number of name bytes.  The
12 B cell of the first stem re-reads R59's published ``井前`` row and must match
it field for field; a mismatch is an engineering failure, not a reading.

Discipline: dev/synthetic only, 0 training, 0 parameter change, 0 graph change,
the sealed confirm split and its holdout report are never read, and the Task1
192/192 reading is untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "scripts" / "training"
for entry in (ROOT, TRAINING):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import torch  # noqa: E402

torch.set_num_threads(1)

import probe_taiji_stop_shape_r58_20261109 as r58  # noqa: E402
import probe_taiji_stop_shape_r59_20261109 as r59  # noqa: E402

from taiji import data_premise_18_0 as premise  # noqa: E402

MANIFEST = ROOT / "plans/manifests/taiji_object_length_r60_20261109.json"
REPORT = ROOT / "reports/taiji_object_length_r60_20261109.json"
PROGRESS = ROOT / "reports/taiji_object_length_r60_20261109_progress.jsonl"
BUNDLE = r59.BUNDLE
ANCHOR_REPORT = r59.REPORT

CPU_CAP = 200.0
WALL_CAP = 1800.0
SINGLE_RUN_ESTIMATE = 15.0
HOPS = (1, 2, 3)
STRUCTURAL_NAME_BYTES = (3, 6, 9, 12)
OBJECT_LENGTHS = (6, 9, 12, 15)
PLACE_BYTES = 6
PLACE_NAME = r59.GROUP_B_WORDS[0]
OBJECT_DESCRIPTION = r59.GROUP_B_OBJECT_DESCRIPTION
PLACE_DESCRIPTION = r59.GROUP_B_PLACE_DESCRIPTION
ARMS = ("stem_wumu", "stem_glyph")
STEM_WUMU = {6: "乌木", 9: "乌木罗", 12: r59.GROUP_B_OBJECT_NAME, 15: "乌木罗盘座"}
ANCHOR_ARM = "stem_wumu"
ANCHOR_BYTES = 12
ANCHOR_FIELDS = (
    "target_text",
    "decoded_text",
    "answer_bytes",
    "zero_extra",
    "byte_exact",
    "field_lengths_ok",
)
ANCHOR_EDGE_FIELDS = (
    "argmax_byte",
    "argmax_label",
    "masked",
    "pre_mask_label",
    "emitted_label",
    "eos_probability",
    "eos_rank",
    "continuation_desire",
    "copy_mass_total",
)
FROZEN_FILES = (
    "scripts/training/probe_taiji_object_length_r60_20261109.py",
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
    "reports/taiji_stop_shape_r59_20261109.json",
)

DEFINITION = {
    "gate": "R60 frozen-criterion dev probe: object-slot byte-length sweep at fixed place 6 B",
    "held_constant": {
        "base_fixture": "dev[0], as in R59 group B",
        "object_description": OBJECT_DESCRIPTION,
        "place_description": PLACE_DESCRIPTION,
        "place_name": PLACE_NAME,
        "place_bytes": PLACE_BYTES,
        "place_name_rule": "first word in R59's published GROUP_B_WORDS order; not selected on its reading",
        "goal": f"让{OBJECT_DESCRIPTION}到{PLACE_DESCRIPTION}。",
        "template": "2-hop 把{object}放到{place}; records written as {desc}是{name}。 and {desc}对应{name}。",
    },
    "sweep": {
        "object_bytes": list(OBJECT_LENGTHS),
        "arms": {
            "stem_wumu": {str(key): value for key, value in STEM_WUMU.items()},
            "stem_glyph": "r58.names_for(1, nbytes)[0] over the same four lengths",
        },
        "family_shape": "each arm is a nested prefix family: only the name byte count changes",
        "answer_bytes": {str(length): 3 + length + 6 + PLACE_BYTES for length in OBJECT_LENGTHS},
        "cells": len(ARMS) * len(OBJECT_LENGTHS),
    },
    "measurements": {
        "object_edge": "position object_stop = 3 + object_bytes; argmax, EOS probability, EOS rank, copy mass",
        "place_edge": "position place_stop = object_stop + 6 + place_bytes; same fields",
        "content": "decoded_text, answer_bytes, zero_extra, byte_exact, field_lengths_ok",
    },
    "classifier": {
        "pattern": "per arm, the four byte_exact booleans in 6/9/12/15 order",
        "branches_in_order": [
            "both arms (false,true,true,true) -> object_min_nine_sufficient",
            "both arms (false,false,true,false) -> exact_twelve_only",
            "both arms (false,false,true,true) -> object_min_twelve_sufficient",
            "both arms (true,true,true,true) -> all_lengths_exact_contradicts_r59",
            "both arms (false,false,false,false) -> no_object_length_exact",
            "arms disagree -> stem_dependent",
            "same pattern with a true above a false -> non_monotone_length_effect",
            "anything else -> unexplained",
        ],
        "note": "branches are mutually exclusive and evaluated in this order; every pattern is reported",
    },
    "main_task_rule": {
        "object_min_nine_sufficient": "192/192 may be cited for object >=9 B, limited to these two stems, place 6 B and this description pair",
        "object_min_twelve_sufficient": "192/192 may be cited for object >=12 B under the same limits",
        "exact_twelve_only": "192/192 must be cited under the exact trained shape; length variants need retraining",
        "any_other_class": "the capability limit is not relaxed",
    },
    "gates": {
        "structural_pass": "12 structural cells exact AND every cell field_lengths_ok",
        "zero_extra_pass": "all 8 cells stop at the declared end with zero extra bytes",
        "object_edge_measured": "all 8 cells carry a trace at object_stop; missing trace fails closed",
        "anchor_match": "the stem_wumu 12 B cell equals R59's published 井前 row on every declared field",
        "failing_axes": "names of failed boolean gates",
    },
    "anchor": {
        "source": str(ANCHOR_REPORT.relative_to(ROOT)),
        "row": f"group_b.cells[place_word={PLACE_NAME}]",
        "fields": list(ANCHOR_FIELDS),
        "place_edge_fields": list(ANCHOR_EDGE_FIELDS),
        "on_mismatch": "engineering_failure; the sweep is not interpreted",
    },
    "scope": "dev/synthetic only; the sealed confirm split and its holdout report are not read; no change to 192/192",
    "estimate_rule": "D-64: single run x 3 (one run plus two replay margins) + report assembly",
    "unit_price_source": {
        "path": "reports/taiji_stop_shape_r59_20261109_progress.jsonl",
        "setup_structural_and_four_cells_cpu": 7.2969,
        "group_b_last_cpu": 21.8281,
        "group_b_first_cpu": 7.75,
        "group_b_cells_between": 29,
        "derived_cpu_per_cell": 0.4854,
        "r60_single_run_derivation": "5.36 setup+structural + 8 x 0.4854 cells = 9.24; declared 15 with margin",
    },
}


def append_progress(record: dict) -> None:
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def definition_digest() -> str:
    body = json.dumps(DEFINITION, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def file_hashes() -> dict:
    return {name: r59.sha256_of(ROOT / name) for name in FROZEN_FILES}


def object_name_for(arm: str, nbytes: int) -> str:
    if arm == "stem_wumu":
        return STEM_WUMU[nbytes]
    if arm == "stem_glyph":
        return r58.names_for(1, nbytes)[0]
    raise ValueError(f"unknown arm {arm}")


def check_families() -> dict:
    """Both arms must really be nested prefix families of the declared byte lengths."""
    problems = []
    for arm in ARMS:
        names = [object_name_for(arm, length) for length in OBJECT_LENGTHS]
        for length, name in zip(OBJECT_LENGTHS, names, strict=True):
            if len(name.encode("utf-8")) != length:
                problems.append(f"{arm}:{length}:bytes={len(name.encode('utf-8'))}")
        for shorter, longer in zip(names[:-1], names[1:], strict=True):
            if not longer.startswith(shorter):
                problems.append(f"{arm}:{shorter}->longer:{longer}:not_prefix")
    if STEM_WUMU[12] != r59.GROUP_B_OBJECT_NAME:
        problems.append("stem_wumu:12:differs_from_r59_object_name")
    if len(PLACE_NAME.encode("utf-8")) != PLACE_BYTES:
        problems.append(f"place_name_bytes={len(PLACE_NAME.encode('utf-8'))}")
    return {"problems": problems, "ok": not problems}


def classify(patterns: dict[str, tuple[bool, ...]]) -> str:
    values = [patterns[arm] for arm in ARMS]
    first, second = values
    if first == second:
        if first == (False, True, True, True):
            return "object_min_nine_sufficient"
        if first == (False, False, True, False):
            return "exact_twelve_only"
        if first == (False, False, True, True):
            return "object_min_twelve_sufficient"
        if first == (True, True, True, True):
            return "all_lengths_exact_contradicts_r59"
        if first == (False, False, False, False):
            return "no_object_length_exact"
        if any(
            not first[index] and first[later]
            for index in range(len(first))
            for later in range(index + 1, len(first))
        ):
            return "non_monotone_length_effect"
        return "unexplained"
    return "stem_dependent"


def implication(length_class: str) -> str:
    if length_class == "object_min_nine_sufficient":
        return "relax_to_object_min_nine_bytes"
    if length_class == "object_min_twelve_sufficient":
        return "relax_to_object_min_twelve_bytes"
    if length_class == "exact_twelve_only":
        return "exact_trained_shape_only"
    return "limit_not_relaxed"


def anchor_row() -> dict:
    published = json.loads(ANCHOR_REPORT.read_text(encoding="utf-8"))
    rows = [row for row in published["group_b"]["cells"] if row["place_word"] == PLACE_NAME]
    if len(rows) != 1:
        raise RuntimeError(f"anchor row not unique: {len(rows)}")
    return rows[0]


def compare_anchor(cell: dict, published: dict) -> dict:
    mismatches = []
    for field in ANCHOR_FIELDS:
        if cell[field] != published[field]:
            mismatches.append({"field": field, "r60": cell[field], "r59": published[field]})
    edge = cell["place_edge"] or {}
    for field in ANCHOR_EDGE_FIELDS:
        if edge.get(field) != published[field]:
            mismatches.append(
                {"field": f"place_edge.{field}", "r60": edge.get(field), "r59": published[field]}
            )
    return {
        "matched": not mismatches,
        "mismatches": mismatches,
        "fields_compared": len(ANCHOR_FIELDS) + len(ANCHOR_EDGE_FIELDS),
    }


def measure_cell(candidate, probe, base_fixture: dict, arm: str, nbytes: int) -> dict:
    object_name = object_name_for(arm, nbytes)
    fixture, object_raw, place_raw = r59.shaped_fixture(
        base_fixture,
        OBJECT_DESCRIPTION,
        PLACE_DESCRIPTION,
        object_name,
        PLACE_NAME,
    )
    shape = r59.shape_for(
        object_name,
        PLACE_NAME,
        object_raw,
        place_raw,
        OBJECT_DESCRIPTION,
        PLACE_DESCRIPTION,
    )
    decoded = r59.decode_cell(candidate, probe, fixture, shape)
    return {
        "arm": arm,
        "object_bytes": nbytes,
        "place_bytes": PLACE_BYTES,
        "object_name": object_name,
        "place_name": PLACE_NAME,
        "object_description": OBJECT_DESCRIPTION,
        "place_description": PLACE_DESCRIPTION,
        "target_text": shape["target"].decode("utf-8"),
        "decoded_text": decoded["decoded_text"],
        "expected_bytes": len(shape["target"]),
        "answer_bytes": decoded["answer_bytes"],
        "zero_extra": decoded["zero_extra"],
        "byte_exact": decoded["byte_exact"],
        "field_lengths_ok": shape["field_lengths_ok"],
        "object_stop": shape["object_stop"],
        "place_stop": shape["place_stop"],
        "object_edge": r59.edge_or_none(decoded, shape["object_stop"]),
        "place_edge": r59.edge_or_none(decoded, shape["place_stop"]),
    }


def edge_summary(edge: dict | None) -> dict:
    if edge is None:
        return {"measured": False}
    return {
        "measured": True,
        "position": edge.get("position"),
        "argmax_label": edge.get("argmax_label"),
        "eos_probability": edge.get("eos_probability"),
        "eos_rank": edge.get("eos_rank"),
        "copy_mass_total": edge.get("copy_mass_total"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    started_cpu = time.process_time()
    started_wall = time.perf_counter()

    if args.freeze:
        if MANIFEST.exists():
            raise RuntimeError(f"preregistration already exists: {MANIFEST}")
        families = check_families()
        if not families["ok"]:
            raise RuntimeError(f"declared families are not prefix families: {families['problems']}")
        r59.write_json(
            MANIFEST,
            {
                "schema": "taiji-object-length-r60-manifest-v1",
                "step": "R60",
                "date": "2026-11-09",
                "definition": DEFINITION,
                "definition_digest": definition_digest(),
                "frozen_files": file_hashes(),
                "families": families,
                "bundle": str(BUNDLE.relative_to(ROOT)),
                "bundle_sha256": r59.sha256_of(BUNDLE),
                "anchor": {
                    "path": str(ANCHOR_REPORT.relative_to(ROOT)),
                    "sha256": r59.sha256_of(ANCHOR_REPORT),
                    "place_word": PLACE_NAME,
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
        name
        for name, value in manifest["frozen_files"].items()
        if r59.sha256_of(ROOT / name) != value
    ]
    if drifted:
        raise RuntimeError(f"frozen input drift: {drifted}")
    if manifest["bundle_sha256"] != r59.sha256_of(BUNDLE):
        raise RuntimeError("bundle drift")
    if PROGRESS.exists() and PROGRESS.stat().st_size:
        events = [
            json.loads(line)
            for line in PROGRESS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(event.get("event") == "finish" for event in events):
            raise RuntimeError("R60 already finished")
    else:
        PROGRESS.write_text("", encoding="utf-8")
    append_progress({"event": "start", "definition_digest": definition_digest()})

    families = check_families()
    if not families["ok"]:
        raise RuntimeError(f"family drift: {families['problems']}")

    structural = [
        r58.structural_cell(hops, nbytes) for hops in HOPS for nbytes in STRUCTURAL_NAME_BYTES
    ]
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

    cells: list[dict] = []
    for arm in ARMS:
        for nbytes in OBJECT_LENGTHS:
            cell = measure_cell(candidate, probe, dev[0], arm, nbytes)
            cells.append(cell)
            append_progress(
                {
                    "event": "cell",
                    "arm": arm,
                    "object_bytes": nbytes,
                    "byte_exact": cell["byte_exact"],
                    "zero_extra": cell["zero_extra"],
                    "object_edge": edge_summary(cell["object_edge"]),
                    "cpu_seconds": round(time.process_time() - started_cpu, 4),
                }
            )

    anchor_cell = next(
        cell for cell in cells if cell["arm"] == ANCHOR_ARM and cell["object_bytes"] == ANCHOR_BYTES
    )
    anchor = compare_anchor(anchor_cell, anchor_row())
    append_progress({"event": "anchor", "matched": anchor["matched"]})

    patterns = {
        arm: tuple(bool(cell["byte_exact"]) for cell in cells if cell["arm"] == arm) for arm in ARMS
    }
    length_class = classify(patterns)
    object_edge_measured = all(cell["object_edge"] is not None for cell in cells)
    structural_pass = all(cell["exact"] for cell in structural) and all(
        cell["field_lengths_ok"] for cell in cells
    )
    zero_extra_pass = all(cell["zero_extra"] for cell in cells)
    failing_axes = []
    if not structural_pass:
        failing_axes.append("structural")
    if not zero_extra_pass:
        failing_axes.append("zero_extra")
    if not object_edge_measured:
        failing_axes.append("object_edge_unmeasurable")
    if not anchor["matched"]:
        failing_axes.append("anchor_mismatch")

    cpu_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    over_budget = cpu_seconds > CPU_CAP or wall_seconds > WALL_CAP
    if not anchor["matched"]:
        verdict = "engineering_failure"
    elif over_budget:
        verdict = "budget_stop"
    elif not object_edge_measured:
        verdict = "object_edge_unmeasurable"
    else:
        verdict = length_class

    report = {
        "schema": "taiji-object-length-r60-report-v1",
        "step": "R60",
        "date": "2026-11-09",
        "relation_to_main_task": (
            "不改 Task1 的 192/192，只判它的物件槽长度适用范围：地点固定 6 B、描述固定，"
            "物件名 6/9/12/15 B 两条词干各扫一遍，判 >=9 B 即可还是必须精确 12 B。"
        ),
        "preregistration": str(MANIFEST.relative_to(ROOT)),
        "preregistration_sha256": r59.sha256_of(MANIFEST),
        "definition": DEFINITION,
        "scope": {
            "sealed_split": "not read",
            "split": "dev/synthetic",
            "fixtures_used": "12 dev worlds materialised; dev[0] carries all 8 cells",
            "examples_used": len(cells),
            "confirm_holdout_report_read": False,
        },
        "structural": {
            "cells": structural,
            "exact": sum(1 for cell in structural if cell["exact"]),
            "total": len(structural),
        },
        "families": families,
        "sweep": {
            "held_constant": DEFINITION["held_constant"],
            "cells": cells,
            "patterns": {arm: list(patterns[arm]) for arm in ARMS},
            "object_edges": {
                f"{cell['arm']}:{cell['object_bytes']}": edge_summary(cell["object_edge"])
                for cell in cells
            },
        },
        "anchor": anchor,
        "gates": {
            "structural_pass": structural_pass,
            "zero_extra_pass": zero_extra_pass,
            "object_edge_measured": object_edge_measured,
            "anchor_match": anchor["matched"],
            "failing_axes": failing_axes,
            "structural_cells_exact": f"{sum(1 for cell in structural if cell['exact'])}/{len(structural)}",
        },
        "outcome": {
            "verdict": verdict,
            "length_class": length_class,
            "main_task_implication": implication(length_class),
            "note": (
                "分类只在这两条词干、地点 6 B、这一对描述上成立；两向都不读 confirm，"
                "都不改 192/192。锚点不命中即 engineering_failure，不解释扫描结果。"
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
            "R60 imports the R59 runner and reuses shaped_fixture/shape_for/decode_cell/edge_or_none/sha256_of/write_json; R58 supplies names_for/structural_cell/load_candidate.",
            "The place word is R59 GROUP_B_WORDS[0], fixed by list order rather than by its published reading.",
            "Both arms are nested prefix families, so name identity is held as far as a length sweep allows; the two stems are the replication.",
            "The stem_wumu 12 B cell is a direct re-read of R59's published 井前 row and is compared field for field before any classification is interpreted.",
            "classify branches are mutually exclusive and ordered; a missing object edge fails closed instead of dropping the cell.",
        ],
        "closed_state": {"frozen_files_unchanged": file_hashes() == manifest["frozen_files"]},
    }
    r59.write_json(REPORT, report)
    append_progress(
        {
            "event": "finish",
            "verdict": verdict,
            "length_class": length_class,
            "patterns": {arm: list(patterns[arm]) for arm in ARMS},
            "anchor_matched": anchor["matched"],
            "cpu_seconds": round(cpu_seconds, 4),
            "wall_seconds": round(wall_seconds, 4),
        }
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "length_class": length_class,
                "main_task_implication": implication(length_class),
                "patterns": {arm: list(patterns[arm]) for arm in ARMS},
                "anchor_matched": anchor["matched"],
                "anchor_fields": anchor["fields_compared"],
                "structural_exact": report["gates"]["structural_cells_exact"],
                "zero_extra_pass": zero_extra_pass,
                "failing_axes": failing_axes,
                "cost": report["cost"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
