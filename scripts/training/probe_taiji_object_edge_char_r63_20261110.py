"""R63: character-level force at the object stop edge.

R62 showed that the byte emitted at the object stop edge is a causal handle for
content failure in the break direction (29/29 exact cells break when 0xE7 is
forced) but rescues only 18/19 failures when the lead byte 0xE6 is forced.  The
single miss is a lead-byte collision: 0xE6 starts both 放 and 漆, so a one-byte
force does not pin the character.

R63 asks the character-granularity question on the same 48 dev cells: force the
whole three-byte character (放 versus 玉) and report, per cell, whether the edge
character became the forced one, whether the sentence is byte exact, and whether
the lead byte and the two trailing bytes came from the same source.  The same
56-cell bitwise anchor as R61/R62 is a precondition for reading anything.

Discipline: dev/synthetic only, 0 training, 0 parameter change, 0 graph change,
the sealed confirm split and its holdout report are never read, and the Task1
192/192 reading is untouched.  A byte is not a character: forcing a lead byte
never licenses a claim about a character, and that limit is reported as its own
row.
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

import probe_taiji_object_edge_intervention_r62_20261109 as r62  # noqa: E402
import probe_taiji_object_edge_r61_20261109 as r61  # noqa: E402
import probe_taiji_stop_shape_r58_20261109 as r58  # noqa: E402
import probe_taiji_stop_shape_r59_20261109 as r59  # noqa: E402

from taiji import data_premise_18_0 as premise  # noqa: E402

MANIFEST = ROOT / "plans/manifests/taiji_object_edge_char_r63_20261110.json"
REPORT = ROOT / "reports/taiji_object_edge_char_r63_20261110.json"
PROGRESS = ROOT / "reports/taiji_object_edge_char_r63_20261110_progress.jsonl"
BUNDLE = r59.BUNDLE
ANCHOR_REPORT = r61.REPORT
R62_REPORT = r62.REPORT
ANCHOR_FIELDS_EXPECTED = 1890

CPU_CAP = 400.0
WALL_CAP = 1800.0
SINGLE_RUN_ESTIMATE = 110.0
HOPS = (1, 2, 3)
STRUCTURAL_NAME_BYTES = (3, 6, 9, 12)
OBJECT_LENGTHS = r61.OBJECT_LENGTHS
PLACE_BYTES = r61.PLACE_BYTES
PLACE_NAME = r61.PLACE_NAME
OBJECT_DESCRIPTION = r61.OBJECT_DESCRIPTION
PLACE_DESCRIPTION = r61.PLACE_DESCRIPTION
PUT_TEXT = "放"
YU_TEXT = "玉"
PUT_BYTES = PUT_TEXT.encode("utf-8")
YU_BYTES = YU_TEXT.encode("utf-8")
CONTENT_FIELDS = r62.CONTENT_FIELDS
KEPT_EDGE_FIELDS = r62.KEPT_EDGE_FIELDS
ARMS = ("none", "sham", "put_char", "yu_char", "put_lead", "yu_lead")
TARGET_OF = {
    "put_char": PUT_BYTES,
    "yu_char": YU_BYTES,
    "put_lead": PUT_BYTES,
    "yu_lead": YU_BYTES,
}
FROZEN_FILES = (
    "scripts/training/probe_taiji_object_edge_char_r63_20261110.py",
    "scripts/training/probe_taiji_object_edge_intervention_r62_20261109.py",
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
    "reports/taiji_object_edge_r61_20261109.json",
    "reports/taiji_object_edge_intervention_r62_20261109.json",
)

DEFINITION = {
    "gate": "R63 frozen-criterion dev probe: does the whole three-byte character at the object edge control content failure",
    "held_constant": {
        "base_fixture": "dev[0], as in R59/R60/R61/R62",
        "object_description": OBJECT_DESCRIPTION,
        "place_description": PLACE_DESCRIPTION,
        "place_name": PLACE_NAME,
        "place_bytes": PLACE_BYTES,
        "stems": "the same 14 stems R61/R62 dealt (12 new + 2 replication), same sha256 deal, no re-deal",
        "object_bytes": list(OBJECT_LENGTHS),
        "template": "2-hop 把{object}放到{place}; records written as {desc}是{name}。 and {desc}对应{name}。",
    },
    "intervention": {
        "position": "object stop edge = 3 + object_bytes",
        "scope": "character arms force the three bytes of the edge character; lead arms force only its first byte; every other position decodes freely",
        "mechanism": "override the post-mask emitted byte before advance_custom at each forced position, so the state carries the forced bytes",
        "arms": {
            "none": "no override; also the anchor arm",
            "sham": "override with the bytes the none arm already emitted at those positions",
            "put_char": "override with the three bytes of 放 (E6 94 BE)",
            "yu_char": "override with the three bytes of 玉 (E7 8E 89)",
            "put_lead": "override with E6 only, the R62 one-byte arm, kept to read source splitting",
            "yu_lead": "override with E7 only, the R62 one-byte arm, kept to read source splitting",
        },
        "decodes": "48 new cells x 6 arms + 8 replication cells x none = 296",
        "not_applicable": "a cell whose decode never reaches a forced position has that position disclosed, not silently dropped",
    },
    "anchor": {
        "source": str(ANCHOR_REPORT.relative_to(ROOT)),
        "cells": 56,
        "fields_expected": ANCHOR_FIELDS_EXPECTED,
        "content_fields": list(CONTENT_FIELDS),
        "edge_fields": list(KEPT_EDGE_FIELDS),
        "edges_compared": ["object_edge", "place_edge"],
        "on_mismatch": "engineering_failure; no character-level result is interpreted",
    },
    "machinery_guards": {
        "sham_identity": "sham raw bytes and both kept edge traces must equal the none arm bitwise on every applicable new cell",
        "force_applied": "every forced position that the decode actually visits must carry the forced value",
        "no_op_consistency": "where the forced payload equals the natural payload, that arm must equal the sham arm bitwise",
        "on_violation": "engineering_failure",
    },
    "per_cell_report": {
        "char_reached": "the three bytes at the object edge equal the forced target character",
        "sentence_exact": "the whole answer is byte exact against the declared target",
        "same_source": "the lead byte and the two trailing bytes belong to the same character, i.e. the edge bytes equal the target character; reported for lead arms, where the model picks the trailing bytes itself",
        "char_truncated": "the decode emitted fewer than three bytes at the edge, so the character is incomplete and is disclosed",
        "classification": "each new cell is classified by whether the whole character changed and whether the sentence stayed exact, not by rescue counts alone",
    },
    "denominators": {
        "rescue_candidates": "new cells whose none arm is not byte exact and whose natural edge character is not 放",
        "break_candidates": "new cells whose none arm is byte exact and whose natural edge character is not 玉",
        "rescued": "rescue candidates whose put_char arm is byte exact",
        "broken": "break candidates whose yu_char arm is not byte exact",
        "rates": "rescued / rescue candidates, broken / break candidates",
        "char_granularity_undetermined": "cells where the forced character was reached but the sentence is not byte exact; listed as character-granularity evidence, never as a wrong forcing method",
    },
    "classifier": {
        "branches_in_order": [
            "anchor mismatch -> engineering_failure",
            "sham identity, force applied or no-op consistency violated -> engineering_failure",
            "cpu or wall over cap -> budget_stop",
            "either candidate count is zero -> intervention_unmeasurable",
            "both rates == 1 and every char arm reached its target on its candidates -> char_handle_complete",
            "both rates == 1 but some candidate's target character was not reached -> char_handle_partial",
            "exactly one rate == 1 -> char_handle_one_direction",
            "either rate > 0 -> char_handle_partial",
            "both rates == 0 -> char_handle_none",
        ],
        "note": "branches are mutually exclusive and evaluated in this order; every cell is reported with its character-level classification",
    },
    "main_task_rule": {
        "char_handle_complete": "the whole character at the object edge is a causal handle for content in both directions; still not a capability claim and still no relaxation of the 192/192 shape limit",
        "char_handle_one_direction": "the character is causal in the named direction only; the other direction stays a companion readout",
        "char_handle_partial": "counts are reported per cell; no mechanism sentence beyond the counts",
        "char_handle_none": "the character is not a handle; the R61 predictor stays a companion readout and R61 limit (c) stands",
        "engineering_or_unmeasurable": "no reading",
    },
    "gates": {
        "structural_pass": "12 structural cells exact AND every none-arm cell field_lengths_ok",
        "anchor_match": "all 56 none-arm cells equal R61 on every declared field and the compared field count is 1890",
        "sham_identity": "all applicable sham cells bitwise equal none",
        "force_applied": "all visited forced positions carry the forced value",
        "no_op_consistency": "arms whose payload equals the natural payload equal sham bitwise",
        "failing_axes": "names of failed boolean gates",
    },
    "scope": "dev/synthetic only; the sealed confirm split and its holdout report are not read; no change to 192/192; conclusions limited to these stems, place 6 B and this description pair",
    "estimate_rule": "D-64: single run x 3 (one run plus two replay margins) + report assembly",
    "unit_price_source": {
        "path": "reports/taiji_object_edge_intervention_r62_20261109_progress.jsonl",
        "measured_cpu_per_cell": 0.263,
        "setup_cpu": 4.19,
        "r63_single_run_derivation": "4.19 setup + 296 x 0.263 = 82.04; declared 110 with margin",
        "caveat": "D-69/D-71/D-74 logged unattributed unit-price drift across rounds; the cap, not the estimate, binds",
    },
    "reimplementation_disclosure": "the decode loop is r62.decode, which already reproduced all 56 R61 cells bit for bit; R63 only widens the forced payload from one byte to three and does not edit that loop",
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


def edge_slice(raw: tuple, stop: int, length: int) -> bytes:
    return bytes(raw[stop : stop + length])


def measure(
    candidate,
    probe,
    base_fixture: dict,
    stem: dict,
    nbytes: int,
    arm: str,
    forced_payload: bytes | None,
) -> dict:
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
    stop = shape["object_stop"]
    forced = None
    if forced_payload:
        forced = {stop + offset: value for offset, value in enumerate(forced_payload)}
    decoded = r62.decode(candidate, probe, fixture, shape, forced)
    raw = decoded["raw"]
    edge = edge_slice(raw, stop, 3)
    target = TARGET_OF.get(arm)
    natural = None
    natural_char = b""
    if arm == "none":
        natural = raw[stop] if len(raw) > stop else None
        natural_char = edge
        reached = len(edge) == 3
        same_source = False
    elif target is not None:
        reached = edge == target
        same_source = edge == target
    else:
        reached = False
        same_source = False
    return {
        "stem": stem["stem"],
        "family": stem["family"],
        "arm": arm,
        "object_bytes": nbytes,
        "place_bytes": PLACE_BYTES,
        "object_name": object_name,
        "place_name": PLACE_NAME,
        "target_text": shape["target"].decode("utf-8"),
        "decoded_text": bytes(raw).decode("utf-8", "replace"),
        "expected_bytes": len(shape["target"]),
        "answer_bytes": len(raw),
        "zero_extra": len(raw) == shape["stop"] and decoded["terminated"],
        "byte_exact": tuple(raw) == tuple(shape["target"]),
        "field_lengths_ok": shape["field_lengths_ok"],
        "object_stop": stop,
        "place_stop": shape["place_stop"],
        "natural_emitted_byte": natural,
        "natural_edge_hex": natural_char.hex() if arm == "none" else None,
        "natural_edge_text": natural_char.decode("utf-8", "replace") if arm == "none" else None,
        "forced_payload_hex": forced_payload.hex() if forced_payload else None,
        "forced_payload_text": (
            forced_payload.decode("utf-8", "replace") if forced_payload else None
        ),
        "edge_hex": edge.hex(),
        "edge_text": edge.decode("utf-8", "replace"),
        "char_reached": reached,
        "same_source": same_source,
        "char_truncated": len(edge) < 3,
        "applied": decoded["applied"],
        "raw_hex": bytes(raw).hex(),
        "object_edge": r62.trim_edge(decoded["traces"].get(str(stop))),
        "place_edge": r62.trim_edge(decoded["traces"].get(str(shape["place_stop"]))),
    }


def compare_anchor(cells: list[dict], published: dict) -> dict:
    rows = {(row["stem"], row["object_bytes"]): row for row in published["cells"]}
    mismatches = []
    compared = 0
    for cell in cells:
        key = (cell["stem"], cell["object_bytes"])
        row = rows.get(key)
        if row is None:
            mismatches.append({"cell": f"{key[0]}:{key[1]}", "field": "missing_row"})
            continue
        for field in CONTENT_FIELDS:
            compared += 1
            if cell[field] != row[field]:
                mismatches.append(
                    {
                        "cell": f"{key[0]}:{key[1]}",
                        "field": field,
                        "r63": cell[field],
                        "r61": row[field],
                    }
                )
        for edge_name in ("object_edge", "place_edge"):
            mine = cell[edge_name]
            theirs = row[edge_name]
            if (mine is None) != (theirs is None):
                compared += 1
                mismatches.append(
                    {
                        "cell": f"{key[0]}:{key[1]}",
                        "field": f"{edge_name}.presence",
                        "r63": mine is None,
                        "r61": theirs is None,
                    }
                )
                continue
            if mine is None:
                continue
            for field in KEPT_EDGE_FIELDS:
                compared += 1
                if mine.get(field) != theirs.get(field):
                    mismatches.append(
                        {
                            "cell": f"{key[0]}:{key[1]}",
                            "field": f"{edge_name}.{field}",
                            "r63": mine.get(field),
                            "r61": theirs.get(field),
                        }
                    )
    return {
        "matched": not mismatches and compared == ANCHOR_FIELDS_EXPECTED,
        "fields_compared": compared,
        "fields_expected": ANCHOR_FIELDS_EXPECTED,
        "cells_compared": len(cells),
        "mismatches": mismatches[:40],
        "mismatch_count": len(mismatches),
    }


def machinery_guards(by_key: dict) -> dict:
    sham_bad = []
    force_bad = []
    noop_bad = []
    applicable = 0
    for key, arms in by_key.items():
        none = arms["none"]
        natural = none["natural_edge_hex"]
        if natural is None:
            continue
        applicable += 1
        if not r62.same_reading(none, arms["sham"]):
            sham_bad.append(key)
        for arm in ("put_char", "yu_char", "put_lead", "yu_lead"):
            cell = arms[arm]
            payload = bytes.fromhex(cell["forced_payload_hex"] or "")
            raw = bytes.fromhex(cell["raw_hex"])
            stop = cell["object_stop"]
            visited = min(len(raw) - stop, len(payload))
            broken = False
            for offset in range(max(visited, 0)):
                if raw[stop + offset] != payload[offset]:
                    broken = True
            if broken:
                force_bad.append({"cell": key, "arm": arm, "visited": visited})
            if natural == cell["forced_payload_hex"] and not r62.same_reading(cell, arms["sham"]):
                noop_bad.append({"cell": key, "arm": arm})
    return {
        "applicable_cells": applicable,
        "sham_mismatches": sham_bad,
        "force_not_applied": force_bad,
        "no_op_mismatches": noop_bad,
        "sham_identity": not sham_bad,
        "force_applied": not force_bad,
        "no_op_consistency": not noop_bad,
    }


def char_row(key: str, arms: dict) -> dict:
    none = arms["none"]

    def arm_view(name: str) -> dict:
        cell = arms[name]
        return {
            "char_reached": cell["char_reached"],
            "same_source": cell["same_source"],
            "sentence_exact": cell["byte_exact"],
            "char_truncated": cell["char_truncated"],
            "edge_char": cell["edge_text"],
            "edge_hex": cell["edge_hex"],
            "text": cell["decoded_text"],
        }

    if none["byte_exact"]:
        if arms["put_char"]["char_reached"] and not arms["put_char"]["byte_exact"]:
            classification = "exact_office_broken_by_put_char"
        elif arms["yu_char"]["byte_exact"]:
            classification = "exact_office_survives_yu_char"
        else:
            classification = "exact_office_breaks_under_yu_char"
    elif arms["put_char"]["byte_exact"]:
        classification = "failed_office_repaired_by_put_char"
    elif arms["put_char"]["char_reached"]:
        classification = "failed_office_char_set_sentence_wrong"
    else:
        classification = "failed_office_char_not_reached"
    return {
        "cell": key,
        "none_exact": bool(none["byte_exact"]),
        "natural_edge_text": none["natural_edge_text"],
        "natural_edge_hex": none["natural_edge_hex"],
        "put_char": arm_view("put_char"),
        "yu_char": arm_view("yu_char"),
        "put_lead": arm_view("put_lead"),
        "yu_lead": arm_view("yu_lead"),
        "classification": classification,
    }


def effect(by_key: dict) -> dict:
    rescue = []
    broken = []
    rescue_not_applicable = []
    break_not_applicable = []
    undetermined = []
    lead_divergence = []
    for key, arms in by_key.items():
        none = arms["none"]
        natural_hex = none["natural_edge_hex"]
        if not none["byte_exact"]:
            if natural_hex == PUT_BYTES.hex():
                rescue_not_applicable.append({"cell": key, "natural": natural_hex})
            else:
                cell = arms["put_char"]
                rescue.append(
                    {
                        "cell": key,
                        "natural_edge_text": none["natural_edge_text"],
                        "natural_edge_hex": natural_hex,
                        "char_reached": cell["char_reached"],
                        "rescued": bool(cell["byte_exact"]),
                        "forced_text": cell["decoded_text"],
                        "forced_bytes": cell["answer_bytes"],
                        "char_truncated": cell["char_truncated"],
                    }
                )
                if cell["char_reached"] and not cell["byte_exact"]:
                    undetermined.append(
                        {
                            "cell": key,
                            "arm": "put_char",
                            "edge_char": cell["edge_text"],
                            "text": cell["decoded_text"],
                            "note": "target character reached but the sentence is not byte exact",
                        }
                    )
        if none["byte_exact"]:
            if natural_hex == YU_BYTES.hex():
                break_not_applicable.append({"cell": key, "natural": natural_hex})
            else:
                cell = arms["yu_char"]
                broken.append(
                    {
                        "cell": key,
                        "natural_edge_text": none["natural_edge_text"],
                        "natural_edge_hex": natural_hex,
                        "char_reached": cell["char_reached"],
                        "broken": not bool(cell["byte_exact"]),
                        "forced_text": cell["decoded_text"],
                        "forced_bytes": cell["answer_bytes"],
                        "char_truncated": cell["char_truncated"],
                    }
                )
        for arm, target in (("put_lead", PUT_BYTES), ("yu_lead", YU_BYTES)):
            cell = arms[arm]
            if cell["char_truncated"]:
                continue
            if cell["edge_hex"][:2] == target[:1].hex() and not cell["same_source"]:
                lead_divergence.append(
                    {
                        "cell": key,
                        "arm": arm,
                        "forced_lead": target[:1].hex(),
                        "edge_hex": cell["edge_hex"],
                        "edge_char": cell["edge_text"],
                        "note": "lead byte matched but the model chose different trailing bytes",
                    }
                )
    rescued = sum(1 for row in rescue if row["rescued"])
    broke = sum(1 for row in broken if row["broken"])
    return {
        "rescue_candidates": len(rescue),
        "rescued": rescued,
        "rescue_rate": (rescued / len(rescue)) if rescue else None,
        "rescue_rows": rescue,
        "rescue_not_applicable": rescue_not_applicable,
        "break_candidates": len(broken),
        "broken": broke,
        "break_rate": (broke / len(broken)) if broken else None,
        "break_rows": broken,
        "break_not_applicable": break_not_applicable,
        "char_granularity_undetermined": undetermined,
        "lead_source_divergence": lead_divergence,
    }


def summarise_char_level(by_key: dict) -> dict:
    rows = [char_row(key, arms) for key, arms in by_key.items()]
    return {
        "put_char_reached": sum(1 for row in rows if row["put_char"]["char_reached"]),
        "put_char_exact": sum(1 for row in rows if row["put_char"]["sentence_exact"]),
        "yu_char_reached": sum(1 for row in rows if row["yu_char"]["char_reached"]),
        "yu_char_exact": sum(1 for row in rows if row["yu_char"]["sentence_exact"]),
        "put_lead_same_source": sum(1 for row in rows if row["put_lead"]["same_source"]),
        "yu_lead_same_source": sum(1 for row in rows if row["yu_lead"]["same_source"]),
        "cells": len(rows),
        "rows": rows,
    }


def classify(gates: dict, over_budget: bool, stats: dict) -> str:
    if not gates["anchor_match"]:
        return "engineering_failure"
    if not (gates["sham_identity"] and gates["force_applied"] and gates["no_op_consistency"]):
        return "engineering_failure"
    if over_budget:
        return "budget_stop"
    if stats["rescue_candidates"] == 0 or stats["break_candidates"] == 0:
        return "intervention_unmeasurable"
    rescue = stats["rescue_rate"]
    break_ = stats["break_rate"]
    reached = all(row["char_reached"] for row in stats["rescue_rows"] + stats["break_rows"])
    if rescue == 1.0 and break_ == 1.0:
        return "char_handle_complete" if reached else "char_handle_partial"
    if rescue == 1.0 or break_ == 1.0:
        return "char_handle_one_direction"
    if rescue > 0 or break_ > 0:
        return "char_handle_partial"
    return "char_handle_none"


def implication(verdict: str, stats: dict) -> str:
    if verdict == "char_handle_complete":
        return "character_handle_both_directions"
    if verdict == "char_handle_one_direction":
        direction = "rescue" if stats["rescue_rate"] == 1.0 else "break"
        return f"character_handle_{direction}_only"
    if verdict == "char_handle_partial":
        return "character_handle_partial_counts_only"
    if verdict == "char_handle_none":
        return "character_not_a_handle"
    return "no_reading"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    started_cpu = time.process_time()
    started_wall = time.perf_counter()
    stems = r61.build_stems()

    if args.freeze:
        if MANIFEST.exists():
            raise RuntimeError(f"preregistration already exists: {MANIFEST}")
        checked = r61.check_stems(stems)
        if not checked["ok"]:
            raise RuntimeError(f"stems are not valid: {checked['problems']}")
        r59.write_json(
            MANIFEST,
            {
                "schema": "taiji-object-edge-char-r63-manifest-v1",
                "step": "R63",
                "date": "2026-11-10",
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
                "motivation": {
                    "path": str(R62_REPORT.relative_to(ROOT)),
                    "sha256": r59.sha256_of(R62_REPORT),
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
    if manifest["stems"] != json.loads(json.dumps(stems, ensure_ascii=False)):
        raise RuntimeError("stem deal drift")
    if PROGRESS.exists() and PROGRESS.stat().st_size:
        events = [
            json.loads(line)
            for line in PROGRESS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(event.get("event") == "finish" for event in events):
            raise RuntimeError("R63 already finished")
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
    by_key: dict = {}
    decodes = 0
    for stem in stems:
        for nbytes in OBJECT_LENGTHS:
            key = f"{stem['stem']}:{nbytes}"
            none_cell = measure(candidate, probe, dev[0], stem, nbytes, "none", None)
            decodes += 1
            cells.append(none_cell)
            arms = {"none": none_cell}
            if stem["family"] != "replication":
                stop = none_cell["object_stop"]
                raw = bytes.fromhex(none_cell["raw_hex"])
                natural_payload = raw[stop : stop + 3] or None
                payloads = {
                    "sham": natural_payload,
                    "put_char": PUT_BYTES,
                    "yu_char": YU_BYTES,
                    "put_lead": PUT_BYTES[:1],
                    "yu_lead": YU_BYTES[:1],
                }
                for arm in ("sham", "put_char", "yu_char", "put_lead", "yu_lead"):
                    arms[arm] = measure(candidate, probe, dev[0], stem, nbytes, arm, payloads[arm])
                    decodes += 1
                    cells.append(arms[arm])
            by_key[key] = arms
            append_progress(
                {
                    "event": "cell",
                    "stem": stem["stem"],
                    "family": stem["family"],
                    "object_bytes": nbytes,
                    "none_exact": none_cell["byte_exact"],
                    "natural_edge_hex": none_cell["natural_edge_hex"],
                    "decodes": decodes,
                    "cpu_seconds": round(time.process_time() - started_cpu, 4),
                }
            )

    published = json.loads(ANCHOR_REPORT.read_text(encoding="utf-8"))
    none_cells = [arms["none"] for arms in by_key.values()]
    anchor = compare_anchor(none_cells, published)
    append_progress(
        {"event": "anchor", "matched": anchor["matched"], "fields": anchor["fields_compared"]}
    )

    new_keys = {
        key: arms for key, arms in by_key.items() if arms["none"]["family"] != "replication"
    }
    guards = machinery_guards(new_keys)
    stats = effect(new_keys)
    chars = summarise_char_level(new_keys)
    append_progress(
        {
            "event": "guards",
            "sham_identity": guards["sham_identity"],
            "force_applied": guards["force_applied"],
            "no_op_consistency": guards["no_op_consistency"],
            "rescue_candidates": stats["rescue_candidates"],
            "break_candidates": stats["break_candidates"],
            "char_undetermined": len(stats["char_granularity_undetermined"]),
        }
    )

    structural_pass = all(cell["exact"] for cell in structural) and all(
        cell["field_lengths_ok"] for cell in none_cells
    )
    gates = {
        "structural_pass": structural_pass,
        "anchor_match": anchor["matched"],
        "sham_identity": guards["sham_identity"],
        "force_applied": guards["force_applied"],
        "no_op_consistency": guards["no_op_consistency"],
    }
    failing_axes = [name for name, value in gates.items() if not value]

    cpu_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    over_budget = cpu_seconds > CPU_CAP or wall_seconds > WALL_CAP
    verdict = classify(gates, over_budget, stats)

    report = {
        "schema": "taiji-object-edge-char-r63-report-v1",
        "step": "R63",
        "date": "2026-11-10",
        "relation_to_main_task": (
            "不改 Task1 的 192/192、不放宽形状限制；R62 证明物件止边首字节在 break 方向是因果把手，"
            "本轮只把干预从 1 字节升到完整 3 字节字符，按“整字是否改变”分类，检验字符粒度是否把 "
            "rescue 从 18/19 推到全称。"
        ),
        "preregistration": str(MANIFEST.relative_to(ROOT)),
        "preregistration_sha256": r59.sha256_of(MANIFEST),
        "definition": DEFINITION,
        "scope": {
            "sealed_split": "not read",
            "split": "dev/synthetic",
            "fixtures_used": "12 dev worlds materialised; dev[0] carries all cells",
            "decodes": decodes,
            "cells_reported": len(cells),
            "new_stem_cells": len(new_keys),
            "replication_cells": len(by_key) - len(new_keys),
            "confirm_holdout_report_read": False,
        },
        "structural": {
            "cells": structural,
            "exact": sum(1 for cell in structural if cell["exact"]),
            "total": len(structural),
        },
        "stems": stems,
        "cells": cells,
        "char_level": chars,
        "anchor": anchor,
        "machinery_guards": guards,
        "effect": stats,
        "gates": {
            **gates,
            "failing_axes": failing_axes,
            "structural_cells_exact": f"{sum(1 for cell in structural if cell['exact'])}/{len(structural)}",
        },
        "outcome": {
            "verdict": verdict,
            "main_task_implication": implication(verdict, stats),
            "byte_is_not_character": (
                "字节 ≠ 字：在字节原生模型里，强制首字节与定下这个字是两件事；"
                "本轮任何读数都不得用前者声称后者。lead 臂的 same_source 字段就是这条限制的逐格证据。"
            ),
            "char_granularity_undetermined": stats["char_granularity_undetermined"],
            "note": (
                "锚点不命中或机器守卫被违反即 engineering_failure，字符级结果不解释；"
                "即使字符把手成立也只是机制读数，不放宽 192/192 的形状限制。"
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
            "The decode loop is r62.decode, which already reproduced all 56 R61 cells bit for bit; only the forced payload widens from one byte to three.",
            "The override replaces the post-mask emitted byte before advance_custom at each forced position, so the recurrent state carries the forced bytes.",
            "The sham arm forces the exact bytes the none arm already emitted at those positions, which makes it a bitwise identity test of the override machinery.",
            "same_source is computed from the emitted bytes, not from the forced payload: it is the answer to whether the lead byte and the trailing bytes are the same character.",
            "A cell whose decode truncates before three edge bytes is flagged char_truncated and disclosed instead of being counted as a wrong forcing method.",
        ],
        "closed_state": {"frozen_files_unchanged": file_hashes() == manifest["frozen_files"]},
    }
    r59.write_json(REPORT, report)
    append_progress(
        {
            "event": "finish",
            "verdict": verdict,
            "rescue_candidates": stats["rescue_candidates"],
            "rescued": stats["rescued"],
            "break_candidates": stats["break_candidates"],
            "broken": stats["broken"],
            "char_undetermined": len(stats["char_granularity_undetermined"]),
            "anchor_matched": anchor["matched"],
            "decodes": decodes,
            "cpu_seconds": round(cpu_seconds, 4),
            "wall_seconds": round(wall_seconds, 4),
        }
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "main_task_implication": implication(verdict, stats),
                "decodes": decodes,
                "anchor_matched": anchor["matched"],
                "anchor_fields": anchor["fields_compared"],
                "anchor_mismatches": anchor["mismatch_count"],
                "sham_identity": guards["sham_identity"],
                "force_applied": guards["force_applied"],
                "no_op_consistency": guards["no_op_consistency"],
                "rescue": [stats["rescued"], stats["rescue_candidates"], stats["rescue_rate"]],
                "break": [stats["broken"], stats["break_candidates"], stats["break_rate"]],
                "put_char_exact": chars["put_char_exact"],
                "yu_char_exact": chars["yu_char_exact"],
                "put_lead_same_source": chars["put_lead_same_source"],
                "char_undetermined": len(stats["char_granularity_undetermined"]),
                "structural_exact": report["gates"]["structural_cells_exact"],
                "failing_axes": failing_axes,
                "cost": report["cost"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
