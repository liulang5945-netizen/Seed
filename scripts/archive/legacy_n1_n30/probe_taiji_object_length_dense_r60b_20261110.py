"""R60b: frozen-criterion dev probe for the dense object-slot byte-length ladder.

R60 swept 6/9/12/15 B on two stems and returned ``stem_dependent``.  That grid
could not say where inside a stem a boundary sits (the gaps are coarse), and its
single ``zero_extra`` reading conflated "the answer stopped at the declared end"
with "the answer did not run past the declared end".  This gate therefore

* widens every stem to seven byte lengths 6/7/8/9/10/12/15, with the place field
  fixed at 6 B and both descriptions held constant, and adds a third stem
  (``stem_real``) that R60 never saw;
* splits the stop accounting into ``extra_bytes`` (answer longer than the
  declared answer) and ``missing_bytes`` (answer shorter, i.e. a premature EOS),
  so a premature EOS is counted as missing and never as a passing exception;
* reports the object-field content (``name_exact``) separately from the whole
  sentence content (``sentence_exact``).

The conclusion is pre-registered as exactly one of ``boundary_per_stem`` (every
stem has a monotone threshold X with sentence_exact(L) == (L >= X) on all seven
lengths) or ``no_stable_boundary``.  No "approximately" and no interval is
allowed.

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

MANIFEST = ROOT / "plans/manifests/taiji_object_length_dense_r60b_20261110.json"
REPORT = ROOT / "reports/taiji_object_length_dense_r60b_20261110.json"
PROGRESS = ROOT / "reports/taiji_object_length_dense_r60b_20261110_progress.jsonl"
BUNDLE = r59.BUNDLE
ANCHOR_REPORT = ROOT / "reports/taiji_object_length_r60_20261109.json"

CPU_CAP = 200.0
WALL_CAP = 1800.0
SINGLE_RUN_ESTIMATE = 10.0
ESTIMATE_WITH_REPLAYS = 30.0
HOPS = (1, 2, 3)
STRUCTURAL_NAME_BYTES = (3, 6, 9, 12)
OBJECT_LENGTHS = (6, 7, 8, 9, 10, 12, 15)
PURE_HAN_LENGTHS = (6, 9, 12, 15)
PLACE_BYTES = 6
PLACE_NAME = r59.GROUP_B_WORDS[0]
OBJECT_DESCRIPTION = r59.GROUP_B_OBJECT_DESCRIPTION
PLACE_DESCRIPTION = r59.GROUP_B_PLACE_DESCRIPTION
STEMS = ("stem_wumu", "stem_glyph", "stem_real")
FILLER = "ab"
HAN_NAMES = {
    "stem_wumu": {6: "乌木", 9: "乌木罗", 12: r59.GROUP_B_OBJECT_NAME, 15: "乌木罗盘座"},
    "stem_glyph": {
        6: r58.names_for(1, 6)[0],
        9: r58.names_for(1, 9)[0],
        12: r58.names_for(1, 12)[0],
        15: r58.names_for(1, 15)[0],
    },
    "stem_real": {6: "铜灯", 9: "铜灯架", 12: "铜灯架台", 15: "铜灯架台柱"},
}
CONTENT_FIELDS = (
    "target_text",
    "decoded_text",
    "expected_bytes",
    "answer_bytes",
    "zero_extra",
    "byte_exact",
    "field_lengths_ok",
    "object_stop",
    "place_stop",
)
EDGE_FIELDS = (
    "position",
    "at_declared_end",
    "argmax_byte",
    "argmax_label",
    "masked",
    "pre_mask_label",
    "emitted_label",
    "eos_probability",
    "eos_rank",
    "continuation_desire",
    "copy_mass_total",
    "copy_limit_active",
    "gamma_eff",
    "q_null",
)
FROZEN_FILES = (
    "scripts/training/probe_taiji_object_length_dense_r60b_20261110.py",
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
    "reports/taiji_object_length_r60_20261109.json",
)

DEFINITION = {
    "gate": (
        "R60b frozen-criterion dev probe: seven-point object-length ladder on three stems, "
        "with extra bytes and missing bytes reported separately"
    ),
    "held_constant": {
        "base_fixture": "dev[0], as in R59 group B and R60",
        "object_description": OBJECT_DESCRIPTION,
        "place_description": PLACE_DESCRIPTION,
        "place_name": PLACE_NAME,
        "place_bytes": PLACE_BYTES,
        "place_name_rule": "first word in R59's published GROUP_B_WORDS order; not selected on its reading",
        "goal": f"让{OBJECT_DESCRIPTION}到{PLACE_DESCRIPTION}。",
        "template": "2-hop 把{object}放到{place}; records written as {desc}是{name}。 and {desc}对应{name}。",
    },
    "ladder": {
        "object_bytes": list(OBJECT_LENGTHS),
        "pure_han_object_bytes": list(PURE_HAN_LENGTHS),
        "place_bytes": PLACE_BYTES,
        "stems": {
            stem: {str(length): name for length, name in HAN_NAMES[stem].items()} for stem in STEMS
        },
        "filler_rule": (
            "for a length L not divisible by 3 the name is the Han prefix of 3*floor(L/3) bytes "
            "plus (L mod 3) lower-case ASCII letters a..; 6/9/12/15 are therefore pure Han"
        ),
        "family_shape": (
            "6/9/12/15 form a nested Han prefix chain and 6/7/8 form a nested prefix chain; "
            "the 8->9 step is not a prefix extension, because the ASCII filler is replaced by "
            "the next Han character"
        ),
        "answer_bytes": {str(length): 3 + length + 6 + PLACE_BYTES for length in OBJECT_LENGTHS},
        "cells": len(STEMS) * len(OBJECT_LENGTHS),
    },
    "measurements": {
        "object_edge": (
            "trace at object_stop = 3 + object_bytes; argmax_byte, argmax_label, eos_probability, "
            "eos_rank, masked, emitted_label, plus the full edge report"
        ),
        "place_edge": "trace at place_stop = object_stop + 6 + place_bytes; same fields",
        "name_exact": "decoded raw[3:3+object_bytes] equals the object-name encoding byte for byte",
        "name_field_text": "utf-8 decoding of that same slice",
        "sentence_exact": (
            "raw equals 把{object}放到{place} byte for byte; this is the R60 byte_exact reading"
        ),
        "extra_bytes": "max(0, answer_bytes - expected_bytes)",
        "missing_bytes": "max(0, expected_bytes - answer_bytes)",
        "zero_extra_bytes": "extra_bytes == 0",
        "zero_missing_bytes": "missing_bytes == 0",
        "early_eos_rule": (
            "a premature EOS can only produce missing_bytes; it is never counted as an exception "
            "inside a zero-extra pass"
        ),
        "utf8_char_boundary": (
            "the object-name encoding sliced at object_bytes decodes as strict utf-8 and returns "
            "the declared name; ASCII is one byte, so 7/8/10 also end on a codepoint boundary"
        ),
        "reported_fields": [
            "stem",
            "object_bytes",
            "place_bytes",
            "object_name",
            "place_name",
            "script_mixed",
            "utf8_char_boundary",
            "name_exact",
            "name_field_text",
            "sentence_exact",
            "decoded_text",
            "answer_bytes",
            "expected_bytes",
            "extra_bytes",
            "missing_bytes",
            "zero_extra_bytes",
            "zero_missing_bytes",
            "object_edge",
            "place_edge",
        ],
    },
    "classifier": {
        "stable_boundary": (
            "per stem: there exists X in object_bytes such that sentence_exact(L) == (L >= X) "
            "for every L in the ladder; X is unique when it exists"
        ),
        "branches_in_order": [
            "anchor mismatch, a missing object edge, or a budget overrun -> decision null with "
            "blocked_by listed; no conclusion is issued",
            "all three stems have a stable boundary -> boundary_per_stem",
            "otherwise -> no_stable_boundary",
        ],
        "statements": {
            "boundary_per_stem": "边界随词干不同，引用 192/192 必须按词干标定",
            "no_stable_boundary": "192/192 只能按精确形状＋精确词干引用，任何长度或词干变体须重测",
        },
        "forbidden": ["approximately", "interval"],
        "note": "the two decisions are mutually exclusive and exhaustive; every stem's boolean ladder is reported",
    },
    "main_task_rule": {
        "boundary_per_stem": (
            "192/192 may be cited only when the stem is named and the object length is at or above "
            "that stem's X, under this place 6 B and this description pair"
        ),
        "no_stable_boundary": (
            "192/192 must be cited under the exact trained shape and the exact trained stem; any "
            "length or stem variant has to be re-measured"
        ),
    },
    "gates": {
        "structural_pass": "12 structural cells exact AND every cell field_lengths_ok",
        "object_edge_measured": "all 21 cells carry a trace at object_stop = 3 + object_bytes",
        "zero_extra_bytes_pass": "all 21 cells have extra_bytes == 0",
        "zero_missing_bytes_pass": "all 21 cells have missing_bytes == 0",
        "anchor_match": (
            "the stem_wumu and stem_glyph 12 B cells equal R60's published rows on every declared "
            "field; stem_real has no R60 anchor and is recorded as null"
        ),
        "failing_axes": "names of failed boolean gates",
    },
    "anchor": {
        "source": str(ANCHOR_REPORT.relative_to(ROOT)),
        "rows": "sweep.cells[arm, object_bytes=12] for stem_wumu and stem_glyph",
        "content_fields": list(CONTENT_FIELDS),
        "edge_fields": list(EDGE_FIELDS),
        "edges_compared": ["object_edge", "place_edge"],
        "stem_real": (
            "R60 published no stem_real row, so there is no anchor for the third stem; it is "
            "recorded as null and anchor_match is computed from the two shared stems only"
        ),
        "on_mismatch": "engineering_failure; the ladder is not interpreted",
    },
    "scope": (
        "dev/synthetic only; the sealed confirm split and its holdout report are not read; no "
        "change to 192/192"
    ),
    "estimate_rule": "D-64: single run x 3 (one run plus two replay margins) + report assembly",
    "unit_price_source": {
        "path": "reports/taiji_object_length_r60_20261109_progress.jsonl",
        "r60_setup_structural_and_eight_cells_cpu": 3.31,
        "r60_cells": 8,
        "r60b_single_run_derivation": "3.31 setup+structural+8 cells plus 13 more cells at <=0.5 s",
        "declared": SINGLE_RUN_ESTIMATE,
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


def object_name_for(stem: str, nbytes: int) -> str:
    han_bytes = nbytes - nbytes % 3
    return HAN_NAMES[stem][han_bytes] + FILLER[: nbytes % 3]


def check_ladder() -> dict:
    """Every declared name must be exactly its ladder byte length, and script purity must hold."""
    problems = []
    for stem in STEMS:
        for length in OBJECT_LENGTHS:
            name = object_name_for(stem, length)
            encoded = name.encode("utf-8")
            if len(encoded) != length:
                problems.append(f"{stem}:{length}:bytes={len(encoded)}")
            if encoded[:length].decode("utf-8") != name:
                problems.append(f"{stem}:{length}:utf8_boundary")
            has_ascii = any(ord(char) < 128 for char in name)
            if length in PURE_HAN_LENGTHS and has_ascii:
                problems.append(f"{stem}:{length}:expected_pure_han")
            if length not in PURE_HAN_LENGTHS and not has_ascii:
                problems.append(f"{stem}:{length}:expected_mixed_script")
        if not HAN_NAMES[stem][15].startswith(HAN_NAMES[stem][12]):
            problems.append(f"{stem}:12->15:not_prefix")
        if not HAN_NAMES[stem][12].startswith(HAN_NAMES[stem][9]):
            problems.append(f"{stem}:9->12:not_prefix")
        if not HAN_NAMES[stem][9].startswith(HAN_NAMES[stem][6]):
            problems.append(f"{stem}:6->9:not_prefix")
    if HAN_NAMES["stem_wumu"][12] != r59.GROUP_B_OBJECT_NAME:
        problems.append("stem_wumu:12:differs_from_r59_object_name")
    if any(ord(char) < 128 for char in PLACE_NAME):
        problems.append("place_name_not_pure_han")
    if len(PLACE_NAME.encode("utf-8")) != PLACE_BYTES:
        problems.append(f"place_name_bytes={len(PLACE_NAME.encode('utf-8'))}")
    return {"problems": problems, "ok": not problems, "cells": len(STEMS) * len(OBJECT_LENGTHS)}


def stable_boundary(pattern: dict[int, bool]) -> int | None:
    for threshold in OBJECT_LENGTHS:
        if all(pattern[length] == (length >= threshold) for length in OBJECT_LENGTHS):
            return threshold
    return None


def anchor_rows() -> dict:
    published = json.loads(ANCHOR_REPORT.read_text(encoding="utf-8"))
    rows = {}
    for row in published["sweep"]["cells"]:
        if row["object_bytes"] == 12 and row["arm"] in ("stem_wumu", "stem_glyph"):
            rows[row["arm"]] = row
    missing = {"stem_wumu", "stem_glyph"} - set(rows)
    if missing:
        raise RuntimeError(f"anchor rows missing from R60: {sorted(missing)}")
    return rows


def compare_anchor(cell: dict, published: dict) -> dict:
    mismatches = []
    compared = 0
    for field in CONTENT_FIELDS:
        compared += 1
        if cell[field] != published[field]:
            mismatches.append({"field": field, "r60b": cell[field], "r60": published[field]})
    for edge_name in ("object_edge", "place_edge"):
        edge = cell[edge_name] or {}
        row_edge = published[edge_name] or {}
        for field in EDGE_FIELDS:
            compared += 1
            if edge.get(field) != row_edge.get(field):
                mismatches.append(
                    {
                        "field": f"{edge_name}.{field}",
                        "r60b": edge.get(field),
                        "r60": row_edge.get(field),
                    }
                )
    return {"matched": not mismatches, "fields_compared": compared, "mismatches": mismatches}


def measure_cell(candidate, probe, base_fixture: dict, stem: str, nbytes: int) -> dict:
    object_name = object_name_for(stem, nbytes)
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
    raw = bytes(decoded["raw"])
    name_bytes = object_name.encode("utf-8")
    name_slice = raw[3 : 3 + nbytes]
    expected_bytes = len(shape["target"])
    answer_bytes = len(raw)
    extra_bytes = max(0, answer_bytes - expected_bytes)
    missing_bytes = max(0, expected_bytes - answer_bytes)
    return {
        "stem": stem,
        "object_bytes": nbytes,
        "place_bytes": PLACE_BYTES,
        "object_name": object_name,
        "place_name": PLACE_NAME,
        "object_description": OBJECT_DESCRIPTION,
        "place_description": PLACE_DESCRIPTION,
        "script_mixed": any(ord(char) < 128 for char in object_name),
        "utf8_char_boundary": name_bytes[:nbytes].decode("utf-8") == object_name,
        "target_text": shape["target"].decode("utf-8"),
        "decoded_text": decoded["decoded_text"],
        "expected_bytes": expected_bytes,
        "answer_bytes": answer_bytes,
        "extra_bytes": extra_bytes,
        "missing_bytes": missing_bytes,
        "zero_extra_bytes": extra_bytes == 0,
        "zero_missing_bytes": missing_bytes == 0,
        "zero_extra": decoded["zero_extra"],
        "byte_exact": decoded["byte_exact"],
        "name_exact": name_slice == name_bytes,
        "name_field_text": name_slice.decode("utf-8", "replace"),
        "sentence_exact": raw == shape["target"],
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
        "argmax_byte": edge.get("argmax_byte"),
        "argmax_label": edge.get("argmax_label"),
        "eos_probability": edge.get("eos_probability"),
        "eos_rank": edge.get("eos_rank"),
        "masked": edge.get("masked"),
        "emitted_label": edge.get("emitted_label"),
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
        ladder = check_ladder()
        if not ladder["ok"]:
            raise RuntimeError(f"declared ladder is invalid: {ladder['problems']}")
        r59.write_json(
            MANIFEST,
            {
                "schema": "taiji-object-length-dense-r60b-manifest-v1",
                "step": "R60b",
                "date": "2026-11-10",
                "definition": DEFINITION,
                "definition_digest": definition_digest(),
                "frozen_files": file_hashes(),
                "ladder": ladder,
                "bundle": str(BUNDLE.relative_to(ROOT)),
                "bundle_sha256": r59.sha256_of(BUNDLE),
                "anchor": {
                    "path": str(ANCHOR_REPORT.relative_to(ROOT)),
                    "sha256": r59.sha256_of(ANCHOR_REPORT),
                },
                "cost": {
                    "single_run_estimate": SINGLE_RUN_ESTIMATE,
                    "estimate_with_replays": ESTIMATE_WITH_REPLAYS,
                    "cpu_cap": CPU_CAP,
                    "wall_cap": WALL_CAP,
                },
                "immutability": "written once before the run; never rewritten",
                "sealed_split": "not read",
            },
        )
        print(json.dumps({"frozen": str(MANIFEST), "cells": ladder["cells"]}, ensure_ascii=False))
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
            raise RuntimeError("R60b already finished")
    else:
        PROGRESS.write_text("", encoding="utf-8")
    append_progress({"event": "start", "definition_digest": definition_digest()})

    ladder = check_ladder()
    if not ladder["ok"]:
        raise RuntimeError(f"ladder drift: {ladder['problems']}")

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
    for stem in STEMS:
        for nbytes in OBJECT_LENGTHS:
            cell = measure_cell(candidate, probe, dev[0], stem, nbytes)
            cells.append(cell)
            append_progress(
                {
                    "event": "cell",
                    "stem": stem,
                    "object_bytes": nbytes,
                    "script_mixed": cell["script_mixed"],
                    "name_exact": cell["name_exact"],
                    "sentence_exact": cell["sentence_exact"],
                    "extra_bytes": cell["extra_bytes"],
                    "missing_bytes": cell["missing_bytes"],
                    "object_edge": edge_summary(cell["object_edge"]),
                    "cpu_seconds": round(time.process_time() - started_cpu, 4),
                }
            )

    published = anchor_rows()
    anchor_results = {}
    for stem, published_row in published.items():
        cell = next(row for row in cells if row["stem"] == stem and row["object_bytes"] == 12)
        anchor_results[stem] = compare_anchor(cell, published_row)
    anchor = {
        "source": str(ANCHOR_REPORT.relative_to(ROOT)),
        "source_sha256": r59.sha256_of(ANCHOR_REPORT),
        "rows": {**anchor_results, "stem_real": None},
        "stem_real_note": DEFINITION["anchor"]["stem_real"],
        "matched": all(result["matched"] for result in anchor_results.values()),
        "fields_compared": sum(result["fields_compared"] for result in anchor_results.values()),
    }
    append_progress(
        {"event": "anchor", "matched": anchor["matched"], "fields": anchor["fields_compared"]}
    )

    per_stem = {}
    for stem in STEMS:
        subset = [cell for cell in cells if cell["stem"] == stem]
        pattern = {str(cell["object_bytes"]): bool(cell["sentence_exact"]) for cell in subset}
        boundary = stable_boundary(
            {cell["object_bytes"]: bool(cell["sentence_exact"]) for cell in subset}
        )
        per_stem[stem] = {
            "sentence_exact_by_length": pattern,
            "name_exact_by_length": {
                str(cell["object_bytes"]): bool(cell["name_exact"]) for cell in subset
            },
            "stable_boundary": boundary,
            "monotone_step": boundary is not None,
        }

    structural_pass = all(cell["exact"] for cell in structural) and all(
        cell["field_lengths_ok"] for cell in cells
    )
    object_edge_measured = all(cell["object_edge"] is not None for cell in cells)
    zero_extra_bytes_pass = all(cell["zero_extra_bytes"] for cell in cells)
    zero_missing_bytes_pass = all(cell["zero_missing_bytes"] for cell in cells)
    gates = {
        "structural_pass": structural_pass,
        "object_edge_measured": object_edge_measured,
        "zero_extra_bytes_pass": zero_extra_bytes_pass,
        "zero_missing_bytes_pass": zero_missing_bytes_pass,
        "anchor_match": anchor["matched"],
    }
    failing_axes = [name for name, value in gates.items() if not value]

    cpu_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    over_budget = cpu_seconds > CPU_CAP or wall_seconds > WALL_CAP
    blocked_by = list(failing_axes)
    if over_budget:
        blocked_by.append("budget")

    if blocked_by:
        decision = None
        statement = None
        implication = "no_reading"
    elif all(entry["stable_boundary"] is not None for entry in per_stem.values()):
        decision = "boundary_per_stem"
        statement = DEFINITION["classifier"]["statements"]["boundary_per_stem"]
        implication = "citation_requires_per_stem_calibration"
    else:
        decision = "no_stable_boundary"
        statement = DEFINITION["classifier"]["statements"]["no_stable_boundary"]
        implication = "exact_shape_and_stem_only"

    report = {
        "schema": "taiji-object-length-dense-r60b-report-v1",
        "step": "R60b",
        "date": "2026-11-10",
        "relation_to_main_task": (
            "不改 Task1 的 192/192，只判它的物件槽长度适用范围：地点固定 6 B、描述固定，"
            "物件名 6/7/8/9/10/12/15 B 三条词干各扫一遍；把多出字节与缺少字节分开计，"
            "提前 EOS 只记缺字节，不当成通过里的例外。"
        ),
        "preregistration": str(MANIFEST.relative_to(ROOT)),
        "preregistration_sha256": r59.sha256_of(MANIFEST),
        "definition": DEFINITION,
        "scope": {
            "sealed_split": "not read",
            "split": "dev/synthetic",
            "fixtures_used": "12 dev worlds materialised; dev[0] carries all 21 cells",
            "examples_used": len(cells),
            "confirm_holdout_report_read": False,
        },
        "ladder_notes": [
            (
                "物件名 7 档 6/7/8/9/10/12/15 B，地点名固定为 R59 公开词表第 0 个 井前（6 B），"
                "不按读数挑词。"
            ),
            (
                "7/8/10 档用 ASCII 小写字母补齐：汉字每字 3 B，只能得到 3 的倍数；ASCII 是 1 B，"
                "因此每格结尾仍落在 UTF-8 码点边界上（每格 utf8_char_boundary=true）。"
            ),
            (
                "7/8/10 档的脚本从纯汉字变成汉字+ASCII，这是附加变量；这三档的读数不能当作"
                "纯汉字长度效应的干净证据，结论里如实标注。"
            ),
            (
                "同一词干内 6/7/8 是前缀链，6/9/12/15 是另一条前缀链；8→9 不是前缀延长"
                "（乌木ab→乌木罗），所以这条阶梯不是单一前缀族，而是两条前缀链加混杂脚本档。"
            ),
            (
                "extra_bytes=missing_bytes=0 不代表内容正确：6/9 B 档会把 到 替换成别的字，"
                "字节数正好相等，只有 sentence_exact=false 暴露出来。"
            ),
            (
                "提前 EOS 只算 missing_bytes，不算 zero_extra_bytes 通过里的例外；"
                "两个门分开报，不合并。"
            ),
        ],
        "structural": {
            "cells": structural,
            "exact": sum(1 for cell in structural if cell["exact"]),
            "total": len(structural),
        },
        "ladder_check": ladder,
        "cells": cells,
        "per_stem": per_stem,
        "anchor": anchor,
        "gates": {
            **gates,
            "failing_axes": failing_axes,
            "structural_cells_exact": f"{sum(1 for cell in structural if cell['exact'])}/{len(structural)}",
            "zero_extra_bytes_cells": (
                f"{sum(1 for cell in cells if cell['zero_extra_bytes'])}/{len(cells)}"
            ),
            "zero_missing_bytes_cells": (
                f"{sum(1 for cell in cells if cell['zero_missing_bytes'])}/{len(cells)}"
            ),
            "object_edge_cells": (
                f"{sum(1 for cell in cells if cell['object_edge'] is not None)}/{len(cells)}"
            ),
        },
        "outcome": {
            "decision": decision,
            "statement": statement,
            "per_stem_boundaries": {
                stem: entry["stable_boundary"] for stem, entry in per_stem.items()
            },
            "main_task_implication": implication,
            "blocked_by": blocked_by,
            "note": (
                "结论只在这三条词干、地点 6 B、这一对描述上成立；不读 confirm，不改 192/192。"
                "锚点不命中或物件止边缺失时不发结论，只记 blocked_by。"
            ),
        },
        "cost": {
            "cpu_seconds": round(cpu_seconds, 2),
            "wall_seconds": round(wall_seconds, 2),
            "single_run_estimate": SINGLE_RUN_ESTIMATE,
            "estimate_with_replays": ESTIMATE_WITH_REPLAYS,
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
            "R60b imports the R59/R58 runners and reuses shaped_fixture/shape_for/decode_cell/edge_or_none/names_for/structural_cell/load_candidate/sha256_of/write_json; the generation chain is not re-implemented.",
            "The place word is R59 GROUP_B_WORDS[0], fixed by list order rather than by its published reading.",
            "Names are generated by the declared filler rule (Han prefix of 3*floor(L/3) bytes plus L mod 3 ASCII letters), so 6/9/12/15 stay pure Han and 7/8/10 are the declared mixed-script cells.",
            "R60's report is added to the frozen inputs because the two 12 B cells are compared against it field for field; the content-transport module is frozen because R58 loads its probe context.",
            "zero_extra (reaches the declared end and terminates) is kept only as R60's comparison field; the two new readings extra_bytes and missing_bytes are computed from the lengths and gate separately.",
            "A missing object edge, an anchor mismatch or a budget overrun blocks the decision instead of dropping a cell; no approximate threshold or interval is written.",
        ],
        "closed_state": {"frozen_files_unchanged": file_hashes() == manifest["frozen_files"]},
    }
    r59.write_json(REPORT, report)
    append_progress(
        {
            "event": "finish",
            "decision": decision,
            "per_stem_boundaries": {
                stem: entry["stable_boundary"] for stem, entry in per_stem.items()
            },
            "anchor_matched": anchor["matched"],
            "zero_extra_bytes_pass": zero_extra_bytes_pass,
            "zero_missing_bytes_pass": zero_missing_bytes_pass,
            "cpu_seconds": round(cpu_seconds, 4),
            "wall_seconds": round(wall_seconds, 4),
        }
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "statement": statement,
                "per_stem_boundaries": {
                    stem: entry["stable_boundary"] for stem, entry in per_stem.items()
                },
                "sentence_exact": {
                    stem: entry["sentence_exact_by_length"] for stem, entry in per_stem.items()
                },
                "anchor_matched": anchor["matched"],
                "anchor_fields": anchor["fields_compared"],
                "structural_exact": report["gates"]["structural_cells_exact"],
                "object_edge_cells": report["gates"]["object_edge_cells"],
                "zero_extra_bytes_cells": report["gates"]["zero_extra_bytes_cells"],
                "zero_missing_bytes_cells": report["gates"]["zero_missing_bytes_cells"],
                "failing_axes": failing_axes,
                "cost": report["cost"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
