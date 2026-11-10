"""R62: frozen-criterion dev intervention on the object stop edge.

R61 established a predictor: at the object stop edge, argmax_byte == 0xE7 goes
with a content failure on 48 unseen cells, with no error of either kind.  That
is correlation at read time.  This gate asks whether the emitted byte at that
position is a causal handle on the answer content.

Because the R58/R59/R60/R61 runners are frozen and expose no intervention hook,
the decode loop is re-built here from the same primitives.  The pre-registered
precondition is that the rebuilt loop first reproduces all 56 R61 cells field for
field; only then are interventions interpreted.  Four arms run on the 48 new
cells: no intervention, a sham that forces the byte the model already emitted,
and one-byte forces to 0xE6 and 0xE7.  The sham must be bitwise identical to the
unintervened run, and every force must be shown to have reached the answer.

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

import probe_taiji_object_edge_r61_20261109 as r61  # noqa: E402
import probe_taiji_stop_shape_r58_20261109 as r58  # noqa: E402
import probe_taiji_stop_shape_r59_20261109 as r59  # noqa: E402

from taiji import data_premise_18_0 as premise  # noqa: E402

MANIFEST = ROOT / "plans/manifests/taiji_object_edge_intervention_r62_20261109.json"
REPORT = ROOT / "reports/taiji_object_edge_intervention_r62_20261109.json"
PROGRESS = ROOT / "reports/taiji_object_edge_intervention_r62_20261109_progress.jsonl"
BUNDLE = r59.BUNDLE
ANCHOR_REPORT = r61.REPORT

CPU_CAP = 200.0
WALL_CAP = 1800.0
SINGLE_RUN_ESTIMATE = 60.0
HOPS = (1, 2, 3)
STRUCTURAL_NAME_BYTES = (3, 6, 9, 12)
OBJECT_LENGTHS = r61.OBJECT_LENGTHS
PLACE_BYTES = r61.PLACE_BYTES
PLACE_NAME = r61.PLACE_NAME
OBJECT_DESCRIPTION = r61.OBJECT_DESCRIPTION
PLACE_DESCRIPTION = r61.PLACE_DESCRIPTION
FORCE_E6 = 0xE6
FORCE_E7 = 0xE7
ARMS = ("none", "sham", "force_e6", "force_e7")
KEPT_EDGE_FIELDS = r61.KEPT_EDGE_FIELDS
CONTENT_FIELDS = (
    "target_text",
    "decoded_text",
    "answer_bytes",
    "zero_extra",
    "byte_exact",
    "field_lengths_ok",
)
FROZEN_FILES = (
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
)

DEFINITION = {
    "gate": "R62 frozen-criterion dev intervention: is the object-edge emitted byte a causal handle on content failure",
    "held_constant": {
        "base_fixture": "dev[0], as in R59/R60/R61",
        "object_description": OBJECT_DESCRIPTION,
        "place_description": PLACE_DESCRIPTION,
        "place_name": PLACE_NAME,
        "place_bytes": PLACE_BYTES,
        "stems": "the same 14 stems R61 dealt (12 new + 2 replication), same sha256 deal, no re-deal",
        "object_bytes": list(OBJECT_LENGTHS),
        "template": "2-hop 把{object}放到{place}; records written as {desc}是{name}。 and {desc}对应{name}。",
    },
    "intervention": {
        "position": "object stop edge = 3 + object_bytes",
        "scope": "exactly one emitted byte at that position; every other position decodes freely",
        "mechanism": "override the post-mask emitted byte before advance_custom, so the state carries the forced byte",
        "arms": {
            "none": "no override; also the anchor arm",
            "sham": "override with the byte the none arm already emitted at that position",
            "force_e6": f"override with {FORCE_E6} (0xE6, first byte of 放)",
            "force_e7": f"override with {FORCE_E7} (0xE7, first byte of 玉/箔/甾 class inserts)",
        },
        "decodes": "48 new cells x 4 arms + 8 replication cells x none = 200",
        "not_applicable": "a cell whose none arm never emitted a byte at that position has no sham/force denominator and is disclosed, not dropped silently",
    },
    "anchor": {
        "source": str(ANCHOR_REPORT.relative_to(ROOT)),
        "cells": 56,
        "content_fields": list(CONTENT_FIELDS),
        "edge_fields": list(KEPT_EDGE_FIELDS),
        "edges_compared": ["object_edge", "place_edge"],
        "on_mismatch": "engineering_failure; no intervention result is interpreted",
    },
    "machinery_guards": {
        "sham_identity": "sham raw bytes and both kept edge traces must equal the none arm bitwise on every applicable new cell",
        "force_applied": "every force arm must record the override at the position and the answer byte there must equal the forced value",
        "no_op_consistency": "where the forced byte equals the natural byte, that force arm must equal the sham arm bitwise",
        "on_violation": "engineering_failure",
    },
    "denominators": {
        "rescue_candidates": "new cells whose none arm is not byte exact, whose natural emitted byte exists and is not 0xE6",
        "break_candidates": "new cells whose none arm is byte exact, whose natural emitted byte exists and is not 0xE7",
        "rescued": "rescue candidates whose force_e6 arm is byte exact",
        "broken": "break candidates whose force_e7 arm is not byte exact",
        "rates": "rescued / rescue candidates, broken / break candidates",
    },
    "classifier": {
        "branches_in_order": [
            "anchor mismatch -> engineering_failure",
            "sham identity or force-applied or no-op consistency violated -> engineering_failure",
            "cpu or wall over cap -> budget_stop",
            "either candidate count is zero -> intervention_unmeasurable",
            "rescue_rate == 1 and break_rate == 1 -> causal_both_directions",
            "exactly one rate == 1 -> causal_one_direction",
            "either rate > 0 -> partial_causal_effect",
            "both rates == 0 -> no_causal_effect",
        ],
        "note": "branches are mutually exclusive and evaluated in this order; every intervened cell is reported",
    },
    "main_task_rule": {
        "causal_both_directions": "the emitted byte at the object edge is a causal handle on content failure in both directions; still not a capability claim and still no relaxation of the 192/192 shape limit",
        "causal_one_direction": "causal in the named direction only; the other direction stays a companion readout",
        "partial_causal_effect": "counts are reported per cell; no mechanism sentence beyond the counts",
        "no_causal_effect": "R61's predictor is a companion readout, not a causal handle; R61 limit (c) stands",
        "engineering_or_unmeasurable": "no reading",
    },
    "gates": {
        "structural_pass": "12 structural cells exact AND every none-arm cell field_lengths_ok",
        "anchor_match": "all 56 none-arm cells equal R61 on every declared field",
        "sham_identity": "all applicable sham cells bitwise equal none",
        "force_applied": "all applicable force arms reached the answer",
        "no_op_consistency": "force arms whose byte equals the natural byte equal sham bitwise",
        "failing_axes": "names of failed boolean gates",
    },
    "scope": "dev/synthetic only; the sealed confirm split and its holdout report are not read; no change to 192/192; conclusions limited to these stems, place 6 B and this description pair",
    "estimate_rule": "D-64: single run x 3 (one run plus two replay margins) + report assembly",
    "unit_price_source": {
        "path": "reports/taiji_object_edge_r61_20261109_progress.jsonl",
        "measured_cpu_per_cell": 0.177,
        "setup_structural_and_first_cell_cpu": 1.1406,
        "r62_single_run_derivation": "1.14 setup + 200 x 0.177 = 36.54; declared 60 with margin",
        "caveat": "D-69/D-71 logged unattributed unit-price drift across rounds; the cap, not the estimate, binds",
    },
    "reimplementation_disclosure": "the decode loop is rebuilt from r58 primitives because the frozen runners expose no hook; the 56-cell bitwise anchor is the precondition for interpreting any intervention",
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


@torch.no_grad()
def decode(candidate, probe, fixture: dict, shape: dict, forced: dict | None) -> dict:
    """Mirror r58.decode_shape, with a one-byte override on the emitted symbol."""
    prepared, _context = probe.prepare(fixture)
    view = prepared.base.view
    state, gate_index = r58.observe_goal(candidate, prepared, gate_index=0)
    previous = r58.EOS
    raw: list[int] = []
    terminated = False
    traces: dict[str, dict] = {}
    applied: dict[str, dict] = {}
    record_count = view.record_count
    for _prediction in range(r58.MAX_PREDICTIONS):
        position = len(raw)
        at_end = position >= shape["stop"]
        if shape["object_start"] <= position < shape["object_stop"]:
            route, start, anchor = (
                shape["object_route"],
                shape["object_name_offset"],
                position == shape["object_start"],
            )
        elif shape["place_start"] <= position < shape["place_stop"]:
            route, start, anchor = (
                shape["place_route"],
                shape["place_name_offset"],
                position == shape["place_start"],
            )
        else:
            route, start, anchor = -1, None, False
        limits = [-1] * record_count
        if position >= shape["object_start"]:
            limits[shape["object_route"]] = shape["object_copy_stop"]
        if position >= shape["place_start"]:
            limits[shape["place_route"]] = shape["place_copy_stop"]
        force, override = r58.force_payload(view, route, start if anchor else None)
        probabilities, gamma_eff, content, next_h, q_bytes, q_null, read = probe.forced_step(
            prepared,
            state,
            previous,
            gate_index,
            force,
            override,
            0.0,
            copy_limits=tuple(limits),
        )
        if position in (shape["object_stop"], shape["place_stop"]):
            payload = r58.edge_report(probabilities, q_bytes, view, tuple(limits), bytes(raw))
            payload["position"] = position
            payload["at_declared_end"] = at_end
            payload["gamma_eff"] = float(gamma_eff)
            payload["q_null"] = float(q_null)
            traces[str(position)] = payload
        if at_end:
            terminated = True
            break
        endpoint = r58.decode_endpoint(probabilities, prefix=bytes(raw))
        if endpoint["eos"]:
            terminated = True
            break
        byte = int(endpoint["byte"])
        if forced is not None and position in forced:
            applied[str(position)] = {"natural": byte, "forced": int(forced[position])}
            byte = int(forced[position])
        raw.append(byte)
        if len(raw) > 96:
            break
        state = r58.probe_mod.advance_custom(
            candidate,
            prepared,
            state,
            read,
            probabilities,
            gamma_eff,
            byte,
            content,
            next_h,
            q_bytes,
            q_null,
        )
        previous = byte
    return {"raw": tuple(raw), "terminated": terminated, "traces": traces, "applied": applied}


def trim_edge(edge: dict | None) -> dict | None:
    if edge is None:
        return None
    return {field: edge.get(field) for field in KEPT_EDGE_FIELDS}


def measure(candidate, probe, base_fixture: dict, stem: dict, nbytes: int, arm: str, forced_byte):
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
    forced = None if forced_byte is None else {shape["object_stop"]: int(forced_byte)}
    decoded = decode(candidate, probe, fixture, shape, forced)
    raw = decoded["raw"]
    natural = None
    if arm == "none" and len(raw) > shape["object_stop"]:
        natural = int(raw[shape["object_stop"]])
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
        "object_stop": shape["object_stop"],
        "place_stop": shape["place_stop"],
        "natural_emitted_byte": natural,
        "forced_byte": None if forced_byte is None else int(forced_byte),
        "applied": decoded["applied"],
        "raw_hex": bytes(raw).hex(),
        "object_edge": trim_edge(decoded["traces"].get(str(shape["object_stop"]))),
        "place_edge": trim_edge(decoded["traces"].get(str(shape["place_stop"]))),
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
                        "r62": cell[field],
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
                        "r62": mine is None,
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
                            "r62": mine.get(field),
                            "r61": theirs.get(field),
                        }
                    )
    return {
        "matched": not mismatches,
        "fields_compared": compared,
        "cells_compared": len(cells),
        "mismatches": mismatches[:40],
        "mismatch_count": len(mismatches),
    }


def same_reading(first: dict, second: dict) -> bool:
    return first["raw_hex"] == second["raw_hex"] and all(
        first[edge_name] == second[edge_name] for edge_name in ("object_edge", "place_edge")
    )


def machinery_guards(by_key: dict) -> dict:
    sham_bad = []
    force_bad = []
    noop_bad = []
    applicable = 0
    for key, arms in by_key.items():
        none = arms["none"]
        natural = none["natural_emitted_byte"]
        if natural is None:
            continue
        applicable += 1
        if not same_reading(none, arms["sham"]):
            sham_bad.append(key)
        for arm, value in (("force_e6", FORCE_E6), ("force_e7", FORCE_E7)):
            cell = arms[arm]
            record = cell["applied"].get(str(cell["object_stop"]))
            reached = (
                record is not None
                and record["forced"] == value
                and len(bytes.fromhex(cell["raw_hex"])) > cell["object_stop"]
                and bytes.fromhex(cell["raw_hex"])[cell["object_stop"]] == value
            )
            if not reached:
                force_bad.append({"cell": key, "arm": arm})
            if natural == value and not same_reading(cell, arms["sham"]):
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


def effect(by_key: dict) -> dict:
    rescue = []
    broken = []
    rescue_not_applicable = []
    break_not_applicable = []
    for key, arms in by_key.items():
        none = arms["none"]
        natural = none["natural_emitted_byte"]
        if not none["byte_exact"]:
            if natural is None or natural == FORCE_E6:
                rescue_not_applicable.append({"cell": key, "natural": natural})
            else:
                rescue.append(
                    {
                        "cell": key,
                        "natural": natural,
                        "rescued": bool(arms["force_e6"]["byte_exact"]),
                        "forced_text": arms["force_e6"]["decoded_text"],
                        "forced_bytes": arms["force_e6"]["answer_bytes"],
                        "forced_zero_extra": arms["force_e6"]["zero_extra"],
                    }
                )
        if none["byte_exact"]:
            if natural is None or natural == FORCE_E7:
                break_not_applicable.append({"cell": key, "natural": natural})
            else:
                broken.append(
                    {
                        "cell": key,
                        "natural": natural,
                        "broken": not bool(arms["force_e7"]["byte_exact"]),
                        "forced_text": arms["force_e7"]["decoded_text"],
                        "forced_bytes": arms["force_e7"]["answer_bytes"],
                        "forced_zero_extra": arms["force_e7"]["zero_extra"],
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
    if rescue == 1.0 and break_ == 1.0:
        return "causal_both_directions"
    if rescue == 1.0 or break_ == 1.0:
        return "causal_one_direction"
    if rescue > 0 or break_ > 0:
        return "partial_causal_effect"
    return "no_causal_effect"


def implication(verdict: str, stats: dict) -> str:
    if verdict == "causal_both_directions":
        return "causal_handle_both_directions"
    if verdict == "causal_one_direction":
        direction = "rescue" if stats["rescue_rate"] == 1.0 else "break"
        return f"causal_handle_{direction}_only"
    if verdict == "partial_causal_effect":
        return "partial_causal_effect_counts_only"
    if verdict == "no_causal_effect":
        return "companion_readout_not_causal"
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
                "schema": "taiji-object-edge-intervention-r62-manifest-v1",
                "step": "R62",
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
            raise RuntimeError("R62 already finished")
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
                natural = none_cell["natural_emitted_byte"]
                for arm, value in (
                    ("sham", natural),
                    ("force_e6", FORCE_E6),
                    ("force_e7", FORCE_E7),
                ):
                    if arm == "sham" and natural is None:
                        arms[arm] = measure(candidate, probe, dev[0], stem, nbytes, "sham", None)
                    else:
                        arms[arm] = measure(candidate, probe, dev[0], stem, nbytes, arm, value)
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
                    "natural_emitted_byte": none_cell["natural_emitted_byte"],
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
    append_progress(
        {
            "event": "guards",
            "sham_identity": guards["sham_identity"],
            "force_applied": guards["force_applied"],
            "no_op_consistency": guards["no_op_consistency"],
            "rescue_candidates": stats["rescue_candidates"],
            "break_candidates": stats["break_candidates"],
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
        "schema": "taiji-object-edge-intervention-r62-report-v1",
        "step": "R62",
        "date": "2026-11-10",
        "relation_to_main_task": (
            "不改 Task1 的 192/192、不放宽形状限制；只问 R61 的预测子是不是因果把手："
            "在物件止边强制换一个发射字节，答案内容是否随之翻转。"
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
            "note": (
                "锚点不命中或机器守卫被违反即 engineering_failure，干预结果不解释；"
                "即使因果成立也只是机制读数，不放宽 192/192 的形状限制。"
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
            "The decode loop is rebuilt from r58 primitives (observe_goal/force_payload/forced_step/edge_report/decode_endpoint/advance_custom) because the frozen runners expose no intervention hook; the 56-cell anchor is the precondition.",
            "The override replaces the post-mask emitted byte before advance_custom, so the recurrent state carries the forced byte; only one position is touched.",
            "The sham arm forces the byte the none arm already emitted, which makes it a bitwise identity test of the override machinery rather than a second treatment.",
            "Force arms whose byte equals the natural byte must equal sham bitwise; this catches a non-deterministic or position-leaking override.",
            "Rescue and break denominators exclude cells whose natural byte already equals the forced value, and those exclusions are listed rather than hidden.",
            "raw_hex is kept for every cell so any downstream comparison can be done on bytes instead of decoded text.",
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
