"""R61: frozen-criterion dev probe for the object-edge argmax predictor.

R60 swept the object name length on two stems and found the outcome is stem
dependent.  After the reading, one separation was visible that had not been
pre-registered: at the object stop edge the argmax byte was 0xE7 (the first byte
of the inserted glyph) in all four failing cells and 0xE6 in all four passing
cells.  This gate turns that observation into a pre-registered prediction and
tests it on stems R60 never saw.

Design: 12 new stems (6 glyph, 6 real-morpheme), each a nested prefix family over
6/9/12/15 B, place fixed at 6 B and both descriptions fixed.  The predictor is
"argmax_byte at the object stop edge == 0xE7 implies the answer is not byte
exact".  Three baselines are scored on the same cells (constant pass, constant
fail, and the length rule "object <= 6 B implies failure"), and an exact McNemar
test compares the predictor with the length rule.  R60's own eight cells are
re-read as a replication anchor and must match field for field.

Revision v3 (2026-11-10): the first run reported one ``zero_extra`` gate that
conflated "the answer reached the declared end" with "the answer did not run
past it", so the early-EOS cell g1@15B was described as an exception inside a
55/56 pass.  The gate is re-signed so that ``extra_bytes`` (answer longer than
declared) and ``missing_bytes`` (answer shorter, i.e. a premature EOS) are
counted separately and gated separately; ``zero_extra_pass`` is removed rather
than aliased, because its old value is False while ``zero_extra_bytes_pass`` is
True on that cell.

Discipline: dev/synthetic only, 0 training, 0 parameter change, 0 graph change,
the sealed confirm split and its holdout report are never read, and the Task1
192/192 reading is untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

import probe_taiji_object_length_r60_20261109 as r60  # noqa: E402
import probe_taiji_stop_shape_r58_20261109 as r58  # noqa: E402
import probe_taiji_stop_shape_r59_20261109 as r59  # noqa: E402

from taiji import data_premise_18_0 as premise  # noqa: E402

MANIFEST = ROOT / "plans/manifests/taiji_object_edge_r61_20261109_v3.json"
REPORT = ROOT / "reports/taiji_object_edge_r61_20261109.json"
PROGRESS = ROOT / "reports/taiji_object_edge_r61_20261109_progress.jsonl"
BUNDLE = r59.BUNDLE
ANCHOR_REPORT = r60.REPORT

CPU_CAP = 200.0
WALL_CAP = 1800.0
SINGLE_RUN_ESTIMATE = 25.0
HOPS = (1, 2, 3)
STRUCTURAL_NAME_BYTES = (3, 6, 9, 12)
OBJECT_LENGTHS = (6, 9, 12, 15)
PLACE_BYTES = 6
PLACE_NAME = r59.GROUP_B_WORDS[0]
OBJECT_DESCRIPTION = r60.OBJECT_DESCRIPTION
PLACE_DESCRIPTION = r60.PLACE_DESCRIPTION
PREDICTOR_BYTE = 0xE7
LENGTH_RULE_MAX_BYTES = 6
MCNEMAR_ALPHA = 0.05
GLYPHS_PER_STEM = 5
NEW_STEMS_PER_FAMILY = 6
RECALIBRATION_DECLARED = {
    "trigger": (
        "the v1/v2 run reported a single zero_extra gate; its 55/56 reading counted the early EOS "
        "of g1@15B as an exception inside a pass, not as a missing byte"
    ),
    "old_manifest": "plans/manifests/taiji_object_edge_r61_20261109_v2.json",
    "old_manifest_sha256": "8ac9f1140dc6fd1b5d4e915db5f9246bc32e018cad2660143f9136a0f05de50a",
    "old_report": "reports/taiji_object_edge_r61_20261109.json",
    "old_report_sha256": "429456f2f9f1ebe141133a07f13417cebbb70199930066fe6af118fbe8f31391",
    "old_cells": 56,
    "old_new_cells": 48,
    "old_byte_exact_new_cells": 29,
    "old_type1_errors": 0,
    "old_type2_errors": 0,
    "old_predictor_accuracy": 1.0,
    "old_zero_extra_cells": 55,
    "old_gate_zero_extra_pass": False,
    "old_verdict": "argmax_predicts_content",
    "old_early_eos_cell": {
        "stem": "g1",
        "object_bytes": 15,
        "answer_bytes": 21,
        "expected_bytes": 30,
    },
}

GLYPH_POOL = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥"
MATERIALS = ("木", "竹", "藤", "棕", "檀", "樟", "楠", "松", "柏", "杉", "榆", "槐")
OBJECTS = (
    "罗盘",
    "算盘",
    "灯笼",
    "茶壶",
    "匣子",
    "香炉",
    "酒盏",
    "茶杯",
    "书盒",
    "花瓶",
    "鼎器",
    "座钟",
)
SUFFIXES = ("底座", "支架", "外壳", "把手", "盖板", "内胆")
SEEDS = {
    "glyph": "r61-glyph|",
    "material": "r61-material|",
    "object": "r61-object|",
    "suffix": "r61-suffix|",
}
CONTENT_FIELDS = (
    "target_text",
    "decoded_text",
    "answer_bytes",
    "zero_extra",
    "byte_exact",
    "field_lengths_ok",
)
EDGE_FIELDS = (
    "argmax_byte",
    "argmax_label",
    "eos_probability",
    "eos_rank",
    "copy_mass_total",
)
KEPT_EDGE_FIELDS = (
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
    "scripts/training/probe_taiji_object_edge_r61_20261109.py",
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
    "gate": "R61 frozen-criterion dev probe: does the object-edge argmax predict content failure on unseen stems",
    "held_constant": {
        "base_fixture": "dev[0], as in R59 group B and R60",
        "object_description": OBJECT_DESCRIPTION,
        "place_description": PLACE_DESCRIPTION,
        "place_name": PLACE_NAME,
        "place_bytes": PLACE_BYTES,
        "goal": f"让{OBJECT_DESCRIPTION}到{PLACE_DESCRIPTION}。",
        "template": "2-hop 把{object}放到{place}; records written as {desc}是{name}。 and {desc}对应{name}。",
    },
    "new_stems": {
        "count": 2 * NEW_STEMS_PER_FAMILY,
        "cells": 2 * NEW_STEMS_PER_FAMILY * len(OBJECT_LENGTHS),
        "object_bytes": list(OBJECT_LENGTHS),
        "family_shape": "nested prefix family: the 6/9/12/15 B names are the first 2/3/4/5 characters of one 5-character stem",
        "glyph_family": {
            "pool": GLYPH_POOL,
            "seed": SEEDS["glyph"],
            "deal": "order the pool by sha256(seed + character); stem i takes ordered[(i*5+k) % len(pool)] for k in 0..4",
            "stems": NEW_STEMS_PER_FAMILY,
        },
        "real_family": {
            "materials": list(MATERIALS),
            "objects": list(OBJECTS),
            "suffixes": list(SUFFIXES),
            "seeds": [SEEDS["material"], SEEDS["object"], SEEDS["suffix"]],
            "deal": "order each list by sha256(seed + item); stem i = materials[i] + objects[i] + suffixes[i] (1+2+2 characters)",
            "stems": NEW_STEMS_PER_FAMILY,
        },
        "selection_rule": "deterministic hash deal declared before the run; no stem is chosen on its reading",
        "collision_rule": "a generated 15 B name equal to either R60 stem name is a freeze-time error",
    },
    "replication": {
        "arms": list(r60.ARMS),
        "cells": len(r60.ARMS) * len(OBJECT_LENGTHS),
        "source": str(ANCHOR_REPORT.relative_to(ROOT)),
        "content_fields": list(CONTENT_FIELDS),
        "edge_fields": list(EDGE_FIELDS),
        "edges_compared": ["object_edge", "place_edge"],
        "on_mismatch": "engineering_failure; the prediction is not interpreted",
    },
    "predictor": {
        "rule": f"predict content failure iff object_edge.argmax_byte == {PREDICTOR_BYTE} (0xE7)",
        "measured_at": "object stop edge = position 3 + object_bytes",
        "outcome": "byte_exact of the whole answer",
        "missing_edge": "fail closed: object_edge_unmeasurable",
        "error_names": {
            "type1": "predicted failure but the answer is byte exact",
            "type2": "predicted pass but the answer is not byte exact",
        },
    },
    "measurements": {
        "content": "decoded_text, answer_bytes, expected_bytes, byte_exact, field_lengths_ok",
        "extra_bytes": "max(0, answer_bytes - expected_bytes): bytes the answer emitted past the declared end",
        "missing_bytes": "max(0, expected_bytes - answer_bytes): declared bytes the answer never emitted",
        "zero_extra_bytes": "extra_bytes == 0",
        "zero_missing_bytes": "missing_bytes == 0",
        "legacy_zero_extra": (
            "kept per cell only because the R60 anchor compares it; it is not a gate in this revision"
        ),
        "early_eos_rule": (
            "a premature EOS only ever produces missing_bytes; it is never counted as an exception "
            "inside a zero-extra pass"
        ),
    },
    "recalibration": {
        "old_reading": RECALIBRATION_DECLARED,
        "rule": (
            "every declared old reading must be reproduced by the v3 run; a mismatch is reported in "
            "recalibration.unchanged instead of being silently overwritten"
        ),
        "new_gates": ["zero_extra_bytes_pass", "zero_missing_bytes_pass"],
        "removed_gate": "zero_extra_pass",
        "removal_reason": (
            "the old gate required len(raw) == stop and terminated, which is False on g1@15B while "
            "zero_extra_bytes_pass is True there; aliasing it would silently change the old reading, "
            "so it is removed and the migration is reported"
        ),
    },
    "baselines": {
        "constant_pass": "predict byte exact for every cell",
        "constant_fail": "predict failure for every cell",
        "length_rule": f"predict failure iff object_bytes <= {LENGTH_RULE_MAX_BYTES}",
        "primary_comparator": "length_rule",
        "scored_on": "the same new-stem cells as the predictor",
    },
    "statistics": {
        "accuracy": "(correct predictions) / (new cells)",
        "mcnemar": "exact two-sided: b = predictor correct and baseline wrong, c = predictor wrong and baseline correct, n = b + c, p = min(1, 2 * sum(C(n,k) for k in 0..min(b,c)) / 2**n); p = 1 when n = 0",
        "alpha": MCNEMAR_ALPHA,
    },
    "classifier": {
        "branches_in_order": [
            "any new or replication cell lacks an object edge -> object_edge_unmeasurable",
            "replication anchor mismatch -> engineering_failure",
            "cpu or wall over cap -> budget_stop",
            "type1 == 0 and type2 == 0 and both outcome classes present overall and in each family and accuracy > max(baselines) -> argmax_predicts_content",
            "type1 == 0 and type2 == 0 otherwise -> separation_without_discrimination",
            "errors > 0 and accuracy > max(baselines) and mcnemar_p_vs_length_rule < alpha -> argmax_partial_predictor",
            "anything else -> argmax_no_better_than_baseline",
        ],
        "note": "branches are mutually exclusive and evaluated in this order; the contingency table and every error cell are reported",
    },
    "main_task_rule": {
        "argmax_predicts_content": "the object-edge argmax is a usable predictor of content failure on these stems; it still does not relax the 192/192 shape limit",
        "argmax_partial_predictor": "the predictor beats the length rule but makes errors; record the error cells and do not write a mechanism conclusion",
        "any_other_class": "the R60 post-hoc separation is not supported as a predictor; the hypothesis is withdrawn",
    },
    "gates": {
        "structural_pass": "12 structural cells exact AND every cell field_lengths_ok",
        "zero_extra_bytes_pass": "every cell has extra_bytes == 0",
        "zero_missing_bytes_pass": "every cell has missing_bytes == 0",
        "object_edge_measured": "every cell carries a trace at object_stop",
        "anchor_match": "all replication cells equal R60's published rows on every declared field",
        "failing_axes": "names of failed boolean gates",
    },
    "scope": "dev/synthetic only; the sealed confirm split and its holdout report are not read; no change to 192/192; conclusions limited to these stems, place 6 B and this description pair",
    "estimate_rule": "D-64: single run x 3 (one run plus two replay margins) + report assembly",
    "unit_price_source": {
        "path": "reports/taiji_object_length_r60_20261109_progress.jsonl",
        "setup_structural_and_first_cell_cpu": 1.5938,
        "measured_cpu_per_cell": 0.2433,
        "r61_single_run_derivation": "1.59 setup + 56 x 0.2433 = 15.21; declared 25 with margin for longer names",
        "caveat": "D-69 logged a 2.0x mismatch between R59 and R60 unit prices; the cap, not the estimate, is the binding budget",
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


def ordered_pool(seed: str, items) -> list[str]:
    return sorted(items, key=lambda item: hashlib.sha256((seed + item).encode("utf-8")).hexdigest())


def prefix_names(full: str) -> dict[int, str]:
    return {nbytes: full[: nbytes // 3] for nbytes in OBJECT_LENGTHS}


def build_stems() -> list[dict]:
    stems: list[dict] = []
    ordered = ordered_pool(SEEDS["glyph"], GLYPH_POOL)
    for index in range(NEW_STEMS_PER_FAMILY):
        full = "".join(
            ordered[(index * GLYPHS_PER_STEM + k) % len(ordered)] for k in range(GLYPHS_PER_STEM)
        )
        stems.append(
            {"stem": f"g{index}", "family": "glyph", "full_name": full, "names": prefix_names(full)}
        )
    materials = ordered_pool(SEEDS["material"], MATERIALS)
    objects = ordered_pool(SEEDS["object"], OBJECTS)
    suffixes = ordered_pool(SEEDS["suffix"], SUFFIXES)
    for index in range(NEW_STEMS_PER_FAMILY):
        full = materials[index] + objects[index] + suffixes[index]
        stems.append(
            {"stem": f"r{index}", "family": "real", "full_name": full, "names": prefix_names(full)}
        )
    for arm in r60.ARMS:
        full = r60.object_name_for(arm, 15)
        stems.append(
            {
                "stem": arm,
                "family": "replication",
                "full_name": full,
                "names": {nbytes: r60.object_name_for(arm, nbytes) for nbytes in OBJECT_LENGTHS},
            }
        )
    return stems


def check_stems(stems: list[dict]) -> dict:
    problems = []
    r60_names = {r60.object_name_for(arm, 15) for arm in r60.ARMS}
    seen = set()
    for stem in stems:
        full = stem["full_name"]
        if len(full.encode("utf-8")) != 15:
            problems.append(f"{stem['stem']}:full_bytes={len(full.encode('utf-8'))}")
        if full in seen:
            problems.append(f"{stem['stem']}:duplicate_full_name")
        seen.add(full)
        if stem["family"] != "replication" and full in r60_names:
            problems.append(f"{stem['stem']}:collides_with_r60")
        for nbytes in OBJECT_LENGTHS:
            name = stem["names"][nbytes]
            if len(name.encode("utf-8")) != nbytes:
                problems.append(f"{stem['stem']}:{nbytes}:bytes={len(name.encode('utf-8'))}")
            if not full.startswith(name):
                problems.append(f"{stem['stem']}:{nbytes}:not_prefix")
    new = [stem for stem in stems if stem["family"] != "replication"]
    if len(new) != DEFINITION["new_stems"]["count"]:
        problems.append(f"new_stem_count={len(new)}")
    if len({stem["family"] for stem in new}) != 2:
        problems.append("new_stems_missing_a_family")
    return {"problems": problems, "ok": not problems, "stems": len(stems), "new_stems": len(new)}


def trim_edge(edge: dict | None) -> dict | None:
    if edge is None:
        return None
    return {field: edge.get(field) for field in KEPT_EDGE_FIELDS}


def measure_cell(candidate, probe, base_fixture: dict, stem: dict, nbytes: int) -> dict:
    object_name = stem["names"][nbytes]
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
    expected_bytes = len(shape["target"])
    answer_bytes = decoded["answer_bytes"]
    extra_bytes = max(0, answer_bytes - expected_bytes)
    missing_bytes = max(0, expected_bytes - answer_bytes)
    return {
        "stem": stem["stem"],
        "family": stem["family"],
        "object_bytes": nbytes,
        "place_bytes": PLACE_BYTES,
        "object_name": object_name,
        "place_name": PLACE_NAME,
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
        "field_lengths_ok": shape["field_lengths_ok"],
        "object_stop": shape["object_stop"],
        "place_stop": shape["place_stop"],
        "object_edge": trim_edge(r59.edge_or_none(decoded, shape["object_stop"])),
        "place_edge": trim_edge(r59.edge_or_none(decoded, shape["place_stop"])),
    }


def predict_failure(cell: dict) -> bool | None:
    edge = cell["object_edge"]
    if edge is None or edge.get("argmax_byte") is None:
        return None
    return int(edge["argmax_byte"]) == PREDICTOR_BYTE


def length_prediction(cell: dict) -> bool:
    return cell["object_bytes"] <= LENGTH_RULE_MAX_BYTES


def accuracy(predictions: list[bool], cells: list[dict]) -> float:
    correct = sum(
        1
        for predicted, cell in zip(predictions, cells, strict=True)
        if predicted != cell["byte_exact"]
    )
    return correct / len(cells)


def mcnemar_exact(first: list[bool], second: list[bool], cells: list[dict]) -> dict:
    b = 0
    c = 0
    for one, two, cell in zip(first, second, cells, strict=True):
        actual_fail = not cell["byte_exact"]
        one_right = one == actual_fail
        two_right = two == actual_fail
        if one_right and not two_right:
            b += 1
        elif two_right and not one_right:
            c += 1
    n = b + c
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
        p = min(1.0, 2.0 * tail / (2**n))
    return {"b_only_first_correct": b, "c_only_second_correct": c, "discordant": n, "p_value": p}


def score(cells: list[dict]) -> dict:
    argmax = [predict_failure(cell) for cell in cells]
    if any(value is None for value in argmax):
        return {"measurable": False}
    argmax = [bool(value) for value in argmax]
    length = [length_prediction(cell) for cell in cells]
    constant_pass = [False] * len(cells)
    constant_fail = [True] * len(cells)
    type1 = [
        cell
        for predicted, cell in zip(argmax, cells, strict=True)
        if predicted and cell["byte_exact"]
    ]
    type2 = [
        cell
        for predicted, cell in zip(argmax, cells, strict=True)
        if not predicted and not cell["byte_exact"]
    ]
    exact_count = sum(1 for cell in cells if cell["byte_exact"])
    baselines = {
        "constant_pass": accuracy(constant_pass, cells),
        "constant_fail": accuracy(constant_fail, cells),
        "length_rule": accuracy(length, cells),
    }
    predictor_accuracy = accuracy(argmax, cells)
    return {
        "measurable": True,
        "cells": len(cells),
        "byte_exact_cells": exact_count,
        "not_byte_exact_cells": len(cells) - exact_count,
        "predicted_fail": sum(argmax),
        "predicted_pass": len(argmax) - sum(argmax),
        "type1_errors": len(type1),
        "type2_errors": len(type2),
        "type1_cells": [
            {
                "stem": cell["stem"],
                "object_bytes": cell["object_bytes"],
                "object_name": cell["object_name"],
            }
            for cell in type1
        ],
        "type2_cells": [
            {
                "stem": cell["stem"],
                "object_bytes": cell["object_bytes"],
                "object_name": cell["object_name"],
            }
            for cell in type2
        ],
        "predictor_accuracy": predictor_accuracy,
        "baselines": baselines,
        "max_baseline_accuracy": max(baselines.values()),
        "mcnemar_vs_length_rule": mcnemar_exact(argmax, length, cells),
        "mcnemar_vs_constant_pass": mcnemar_exact(argmax, constant_pass, cells),
        "mcnemar_vs_constant_fail": mcnemar_exact(argmax, constant_fail, cells),
    }


def classify(stats: dict, families: dict[str, dict], gates: dict, over_budget: bool) -> str:
    if not gates["object_edge_measured"]:
        return "object_edge_unmeasurable"
    if not gates["anchor_match"]:
        return "engineering_failure"
    if over_budget:
        return "budget_stop"
    perfect = stats["type1_errors"] == 0 and stats["type2_errors"] == 0
    both_classes = stats["byte_exact_cells"] > 0 and stats["not_byte_exact_cells"] > 0
    family_classes = all(
        family["byte_exact_cells"] > 0 and family["not_byte_exact_cells"] > 0
        for family in families.values()
    )
    beats = stats["predictor_accuracy"] > stats["max_baseline_accuracy"]
    if perfect and both_classes and family_classes and beats:
        return "argmax_predicts_content"
    if perfect:
        return "separation_without_discrimination"
    if beats and stats["mcnemar_vs_length_rule"]["p_value"] < MCNEMAR_ALPHA:
        return "argmax_partial_predictor"
    return "argmax_no_better_than_baseline"


def implication(verdict: str) -> str:
    if verdict == "argmax_predicts_content":
        return "predictor_supported_on_these_stems"
    if verdict == "argmax_partial_predictor":
        return "predictor_beats_length_rule_with_errors"
    if verdict in ("argmax_no_better_than_baseline", "separation_without_discrimination"):
        return "hypothesis_withdrawn"
    return "no_reading"


def compare_anchor(cells: list[dict], published: dict) -> dict:
    rows = {(row["arm"], row["object_bytes"]): row for row in published["sweep"]["cells"]}
    mismatches = []
    compared = 0
    for cell in cells:
        row = rows.get((cell["stem"], cell["object_bytes"]))
        if row is None:
            mismatches.append(
                {"cell": f"{cell['stem']}:{cell['object_bytes']}", "field": "missing_row"}
            )
            continue
        for field in CONTENT_FIELDS:
            compared += 1
            if cell[field] != row[field]:
                mismatches.append(
                    {
                        "cell": f"{cell['stem']}:{cell['object_bytes']}",
                        "field": field,
                        "r61": cell[field],
                        "r60": row[field],
                    }
                )
        for edge_name in ("object_edge", "place_edge"):
            edge = cell[edge_name] or {}
            for field in EDGE_FIELDS:
                compared += 1
                if edge.get(field) != row[edge_name].get(field):
                    mismatches.append(
                        {
                            "cell": f"{cell['stem']}:{cell['object_bytes']}",
                            "field": f"{edge_name}.{field}",
                            "r61": edge.get(field),
                            "r60": row[edge_name].get(field),
                        }
                    )
    return {
        "matched": not mismatches,
        "fields_compared": compared,
        "cells_compared": len(cells),
        "mismatches": mismatches,
    }


def recalibration_block(cells: list[dict], stats: dict, verdict: str, anchor: dict) -> dict:
    """Compare the v3 readings with the readings declared before this re-sign."""
    observed = {
        "cells": len(cells),
        "new_cells": stats["cells"],
        "byte_exact_new_cells": stats["byte_exact_cells"],
        "type1_errors": stats["type1_errors"],
        "type2_errors": stats["type2_errors"],
        "predictor_accuracy": stats["predictor_accuracy"],
        "zero_extra_cells": sum(1 for cell in cells if cell["zero_extra"]),
        "verdict": verdict,
        "anchor_match": anchor["matched"],
    }
    expected = {
        "cells": RECALIBRATION_DECLARED["old_cells"],
        "new_cells": RECALIBRATION_DECLARED["old_new_cells"],
        "byte_exact_new_cells": RECALIBRATION_DECLARED["old_byte_exact_new_cells"],
        "type1_errors": RECALIBRATION_DECLARED["old_type1_errors"],
        "type2_errors": RECALIBRATION_DECLARED["old_type2_errors"],
        "predictor_accuracy": RECALIBRATION_DECLARED["old_predictor_accuracy"],
        "zero_extra_cells": RECALIBRATION_DECLARED["old_zero_extra_cells"],
        "verdict": RECALIBRATION_DECLARED["old_verdict"],
        "anchor_match": True,
    }
    mismatches = [
        {"field": field, "v3": observed[field], "declared_old": expected[field]}
        for field in expected
        if observed[field] != expected[field]
    ]
    early = [
        {
            "stem": cell["stem"],
            "family": cell["family"],
            "object_bytes": cell["object_bytes"],
            "object_name": cell["object_name"],
            "answer_bytes": cell["answer_bytes"],
            "expected_bytes": cell["expected_bytes"],
            "extra_bytes": cell["extra_bytes"],
            "missing_bytes": cell["missing_bytes"],
            "zero_extra": cell["zero_extra"],
            "decoded_text": cell["decoded_text"],
        }
        for cell in cells
        if cell["missing_bytes"] or cell["extra_bytes"]
    ]
    return {
        "declared_old_reading": RECALIBRATION_DECLARED,
        "observed": observed,
        "unchanged": not mismatches,
        "mismatches": mismatches,
        "counts": {
            "zero_extra_bytes_cells": (
                f"{sum(1 for cell in cells if cell['zero_extra_bytes'])}/{len(cells)}"
            ),
            "zero_missing_bytes_cells": (
                f"{sum(1 for cell in cells if cell['zero_missing_bytes'])}/{len(cells)}"
            ),
            "legacy_zero_extra_cells": (
                f"{sum(1 for cell in cells if cell['zero_extra'])}/{len(cells)}"
            ),
        },
        "early_eos_cells": early,
        "gate_migration": {
            "removed": "zero_extra_pass",
            "replacement": ["zero_extra_bytes_pass", "zero_missing_bytes_pass"],
            "alias_declared": False,
            "reason": DEFINITION["recalibration"]["removal_reason"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    started_cpu = time.process_time()
    started_wall = time.perf_counter()
    stems = build_stems()

    if args.freeze:
        if MANIFEST.exists():
            raise RuntimeError(f"preregistration already exists: {MANIFEST}")
        checked = check_stems(stems)
        if not checked["ok"]:
            raise RuntimeError(f"declared stems are not valid: {checked['problems']}")
        r59.write_json(
            MANIFEST,
            {
                "schema": "taiji-object-edge-r61-manifest-v3",
                "step": "R61",
                "date": "2026-11-10",
                "revision": "v3: extra_bytes and missing_bytes gated separately; zero_extra_pass removed",
                "replaces": {
                    "manifest": RECALIBRATION_DECLARED["old_manifest"],
                    "manifest_sha256": RECALIBRATION_DECLARED["old_manifest_sha256"],
                    "report": RECALIBRATION_DECLARED["old_report"],
                    "report_sha256": RECALIBRATION_DECLARED["old_report_sha256"],
                },
                "definition": DEFINITION,
                "definition_digest": definition_digest(),
                "frozen_files": file_hashes(),
                "stems": stems,
                "stem_check": checked,
                "bundle": str(BUNDLE.relative_to(ROOT)),
                "bundle_sha256": r59.sha256_of(BUNDLE),
                "anchor": {
                    "path": str(ANCHOR_REPORT.relative_to(ROOT)),
                    "sha256": r59.sha256_of(ANCHOR_REPORT),
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
        print(json.dumps({"frozen": str(MANIFEST), "stems": len(stems)}, ensure_ascii=False))
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
    # JSON turns the int byte-length keys of names into strings, so compare in JSON space.
    if manifest["stems"] != json.loads(json.dumps(stems, ensure_ascii=False)):
        raise RuntimeError("stem deal drift")
    if PROGRESS.exists() and PROGRESS.stat().st_size:
        events = [
            json.loads(line)
            for line in PROGRESS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(event.get("event") == "finish" for event in events):
            raise RuntimeError("R61 already finished")
    else:
        PROGRESS.write_text("", encoding="utf-8")
    append_progress({"event": "start", "definition_digest": definition_digest()})

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
    for stem in stems:
        for nbytes in OBJECT_LENGTHS:
            cell = measure_cell(candidate, probe, dev[0], stem, nbytes)
            cells.append(cell)
            edge = cell["object_edge"] or {}
            append_progress(
                {
                    "event": "cell",
                    "stem": stem["stem"],
                    "family": stem["family"],
                    "object_bytes": nbytes,
                    "byte_exact": cell["byte_exact"],
                    "zero_extra": cell["zero_extra"],
                    "extra_bytes": cell["extra_bytes"],
                    "missing_bytes": cell["missing_bytes"],
                    "argmax_byte": edge.get("argmax_byte"),
                    "eos_rank": edge.get("eos_rank"),
                    "cpu_seconds": round(time.process_time() - started_cpu, 4),
                }
            )

    new_cells = [cell for cell in cells if cell["family"] != "replication"]
    replication_cells = [cell for cell in cells if cell["family"] == "replication"]
    published = json.loads(ANCHOR_REPORT.read_text(encoding="utf-8"))
    anchor = compare_anchor(replication_cells, published)
    append_progress(
        {"event": "anchor", "matched": anchor["matched"], "fields": anchor["fields_compared"]}
    )

    stats = score(new_cells)
    families = {}
    for family in ("glyph", "real"):
        subset = [cell for cell in new_cells if cell["family"] == family]
        families[family] = score(subset)
    replication_stats = score(replication_cells)

    object_edge_measured = all(cell["object_edge"] is not None for cell in cells)
    structural_pass = all(cell["exact"] for cell in structural) and all(
        cell["field_lengths_ok"] for cell in cells
    )
    zero_extra_bytes_pass = all(cell["zero_extra_bytes"] for cell in cells)
    zero_missing_bytes_pass = all(cell["zero_missing_bytes"] for cell in cells)
    gates = {
        "structural_pass": structural_pass,
        "zero_extra_bytes_pass": zero_extra_bytes_pass,
        "zero_missing_bytes_pass": zero_missing_bytes_pass,
        "object_edge_measured": object_edge_measured,
        "anchor_match": anchor["matched"],
    }
    failing_axes = [name for name, value in gates.items() if not value]

    cpu_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    over_budget = cpu_seconds > CPU_CAP or wall_seconds > WALL_CAP
    if not stats["measurable"]:
        gates["object_edge_measured"] = False
        failing_axes.append("predictor_unmeasurable")
        verdict = "object_edge_unmeasurable"
    else:
        verdict = classify(stats, families, gates, over_budget)

    if stats["measurable"]:
        recalibration = recalibration_block(cells, stats, verdict, anchor)
    else:
        recalibration = {
            "declared_old_reading": RECALIBRATION_DECLARED,
            "observed": None,
            "unchanged": None,
            "mismatches": [],
            "note": "the predictor is unmeasurable, so no recalibration comparison is issued",
        }

    report = {
        "schema": "taiji-object-edge-r61-report-v2",
        "step": "R61",
        "date": "2026-11-10",
        "revision": "v3: extra_bytes and missing_bytes reported and gated separately",
        "relation_to_main_task": (
            "不改 Task1 的 192/192，只检验 R60 读数后那条未预注册的分离能否当预测子："
            "物件止边 argmax=0xE7 是否预示答案内容败，且是否在 R60 没见过的词干上胜过长度基线。"
        ),
        "preregistration": str(MANIFEST.relative_to(ROOT)),
        "preregistration_sha256": r59.sha256_of(MANIFEST),
        "definition": DEFINITION,
        "scope": {
            "sealed_split": "not read",
            "split": "dev/synthetic",
            "fixtures_used": "12 dev worlds materialised; dev[0] carries all cells",
            "examples_used": len(cells),
            "new_stem_cells": len(new_cells),
            "replication_cells": len(replication_cells),
            "confirm_holdout_report_read": False,
        },
        "structural": {
            "cells": structural,
            "exact": sum(1 for cell in structural if cell["exact"]),
            "total": len(structural),
        },
        "stems": stems,
        "stem_check": check_stems(stems),
        "cells": cells,
        "prediction": {
            "new_stems": stats,
            "families": families,
            "replication": replication_stats,
        },
        "anchor": anchor,
        "recalibration": recalibration,
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
            "legacy_zero_extra_cells": (
                f"{sum(1 for cell in cells if cell['zero_extra'])}/{len(cells)}"
            ),
            "removed_gate": "zero_extra_pass",
            "removed_gate_migration": DEFINITION["recalibration"]["removal_reason"],
        },
        "outcome": {
            "verdict": verdict,
            "main_task_implication": implication(verdict),
            "note": (
                "判据、基线、发牌规则与出口都在读数前冻结；结论只在这 12 条新词干、地点 6 B、"
                "这一对描述上成立。即使预测子成立也不放宽 192/192 的形状限制。"
                "多出字节与缺少字节分列两门：提前 EOS 只算缺字节，不算零多余通过里的例外。"
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
            "R61 imports the R60/R59/R58 runners and reuses shaped_fixture/shape_for/decode_cell/edge_or_none/names_for/structural_cell/load_candidate; the generation chain is not re-implemented.",
            "Stems are dealt by declared sha256 seeds over declared pools; no stem is selected on its reading, and a collision with an R60 stem is a freeze-time error.",
            "Edge traces keep every edge_report field except the top candidate list, which is dropped to keep the card readable; the anchor compares only the declared fields.",
            "The predictor is scored on new stems only; R60's eight cells are a replication anchor and are reported separately, never pooled into the verdict.",
            "McNemar is exact and integer-counted, so no floating-point tie handling is involved.",
            "A missing object edge fails closed instead of dropping the cell.",
            "v3 splits the stop accounting into extra_bytes and missing_bytes and removes zero_extra_pass instead of aliasing it, because that gate is False on g1@15B while zero_extra_bytes_pass is True there.",
            "recalibration.unchanged re-checks every declared old reading (56 cells, 48 new cells, 29 byte-exact new cells, 0/0 errors, accuracy 1.0, 55/56 legacy zero_extra, anchor, verdict) before the reading is interpreted.",
        ],
        "closed_state": {"frozen_files_unchanged": file_hashes() == manifest["frozen_files"]},
    }
    r59.write_json(REPORT, report)
    append_progress(
        {
            "event": "finish",
            "verdict": verdict,
            "new_cells": len(new_cells),
            "type1_errors": stats.get("type1_errors"),
            "type2_errors": stats.get("type2_errors"),
            "predictor_accuracy": stats.get("predictor_accuracy"),
            "baselines": stats.get("baselines"),
            "anchor_matched": anchor["matched"],
            "zero_extra_bytes_pass": zero_extra_bytes_pass,
            "zero_missing_bytes_pass": zero_missing_bytes_pass,
            "zero_extra_bytes_cells": report["gates"]["zero_extra_bytes_cells"],
            "zero_missing_bytes_cells": report["gates"]["zero_missing_bytes_cells"],
            "cpu_seconds": round(cpu_seconds, 4),
            "wall_seconds": round(wall_seconds, 4),
        }
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "main_task_implication": implication(verdict),
                "new_cells": len(new_cells),
                "byte_exact_cells": stats.get("byte_exact_cells"),
                "type1_errors": stats.get("type1_errors"),
                "type2_errors": stats.get("type2_errors"),
                "predictor_accuracy": stats.get("predictor_accuracy"),
                "baselines": stats.get("baselines"),
                "mcnemar_vs_length_rule": stats.get("mcnemar_vs_length_rule"),
                "families": {
                    name: {
                        "cells": family.get("cells"),
                        "byte_exact_cells": family.get("byte_exact_cells"),
                        "type1_errors": family.get("type1_errors"),
                        "type2_errors": family.get("type2_errors"),
                        "predictor_accuracy": family.get("predictor_accuracy"),
                    }
                    for name, family in families.items()
                },
                "anchor_matched": anchor["matched"],
                "anchor_fields": anchor["fields_compared"],
                "structural_exact": report["gates"]["structural_cells_exact"],
                "zero_extra_bytes_pass": zero_extra_bytes_pass,
                "zero_missing_bytes_pass": zero_missing_bytes_pass,
                "zero_extra_bytes_cells": report["gates"]["zero_extra_bytes_cells"],
                "zero_missing_bytes_cells": report["gates"]["zero_missing_bytes_cells"],
                "recalibration_unchanged": recalibration["unchanged"],
                "failing_axes": failing_axes,
                "cost": report["cost"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
