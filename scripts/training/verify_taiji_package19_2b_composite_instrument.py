"""D16-5 instrument erratum check (0 updates, 0 confirm reads).

The 19.5 stage gate read ``start_hit`` on ``argmax(read.betas)``: the frozen
reader's own position posterior, which never sees the teacher-span pointer.  The
decode does not use that argmax; it lands on

    row_logits = transport_logits(alpha_i * betas_i) + pointer_bias_i
    composite   = argmax(row_logits)

and a committed carry (m peak / explicit cursor) bypasses the sum entirely.
This script verifies that composition on the real 19.5 step300 bundles and then
reports the three start claims side by side on the same examples:

    beta_start_hit      -- route owns the record and argmax(betas) is the first byte
    pointer_start_hit   -- route owns the record and argmax(span bias) is the first byte
    composite_start_hit -- route owns the record and argmax(final row) is the first byte

``composite_start_hit`` is the D16-5 gate object (>= 0.95); ``route_hit`` stays
its own >= 0.95 clause.  The 19.5 numbers are copied into ``legacy_instrument``
verbatim and are not recomputed over the archived report.

Run:
  python scripts/training/verify_taiji_package19_2b_composite_instrument.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

torch.set_num_threads(1)

import composite_start_instrument as instrument  # noqa: E402
import probe_taiji_content_transport_18_1 as probe_mod  # noqa: E402

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.binding_copy import initial_binding_state  # noqa: E402
from taiji.candidate_identity import candidate_identity  # noqa: E402
from taiji.selection_routing_checkpoint import (  # noqa: E402
    build_routing_candidate,
    load_routing_bundle,
    parameter_tensors,
    save_routing_bundle,
)

SEEDS = (28031, 28032)
ARMS = ("A_route_lock", "B_stop_lock", "C_unlocked")
DEFAULT_DIR = ROOT / "output/taiji_package19_compare"
PARENT = ROOT / "output/taiji_core_binding_select_16_1_compare/bundles/28031-A-locked-step300.pt"
LEGACY_REPORT = ROOT / "reports/taiji_package19_probe_step300_20261104.json"
REPORT = ROOT / "reports/taiji_package19_2b_composite_instrument_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2b_composite_instrument_20261104.json"
TEMP_BUNDLE = ROOT / "output/taiji_package19_2b_roundtrip_tmp.pt"

LEGACY = {
    "object": {
        "start_hit_rate": 0.3472222222222222,
        "start_hit_examples": 100,
        "route_hit_rate": 0.7621527777777778,
        "beta_start_hit_rate": 0.3958333333333333,
        "conditional_byte": 0.9409090909090909,
        "unconditional_byte": 0.6777146464646465,
    },
    "place": {
        "start_hit_rate": 0.3402777777777778,
        "start_hit_examples": 98,
        "route_hit_rate": 0.6944444444444444,
        "beta_start_hit_rate": 0.3888888888888889,
        "conditional_byte": 0.9795918367346939,
        "unconditional_byte": 0.63125,
    },
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def scan_cell(path: Path, fixtures: tuple[dict, ...]) -> dict:
    """Read one bundle with the composite instrument and collect composition proof."""
    candidate = load_routing_bundle(path)[0]
    arm = path.name.split("-")[1]
    probe = probe_mod.ProbeContext(candidate)
    probe.arm = arm
    ledgers = [instrument.measure_fixture(probe, fixture) for fixture in fixtures]
    metrics = [instrument.metrics(ledger) for ledger in ledgers]
    summary = instrument.summarize(metrics)

    composition = {
        "parameters_nan": 0,
        "content_rows": 0,
        "transport_rows": 0,
        "carry_rows": 0,
        "forced_rows": 0,
        "start_transport_rows": 0,
        "start_carry_rows": 0,
        "start_decode_rows": 0,
        "start_decode_mismatch": 0,
        "composite_identity_maxdiff": 0.0,
        "pointer_argmax_mismatch": 0,
        "pointer_onehot_mismatch": 0,
        "source_equals_beta_peak_mismatch": 0,
        "decode_copy_peak_mismatch": 0,
    }
    composition["parameters_nan"] = sum(
        int(torch.isnan(parameter).sum()) for parameter in parameter_tensors(candidate).values()
    )
    start_positions = {probe_mod.OBJECT_POSITIONS[0], probe_mod.PLACE_POSITIONS[0]}
    for ledger in ledgers:
        for step in ledger["steps"]:
            if step["record"] < 0:
                continue
            composition["content_rows"] += 1
            source = step["row_source"]
            if source == "transport":
                composition["transport_rows"] += 1
            elif source == "carry":
                composition["carry_rows"] += 1
            elif source == "forced":
                composition["forced_rows"] += 1
            if step["position"] in start_positions:
                composition["start_decode_rows"] += 1
                if step["decode_peak_matches_composite"] is False:
                    composition["start_decode_mismatch"] += 1
                if source == "transport":
                    composition["start_transport_rows"] += 1
                elif source == "carry":
                    composition["start_carry_rows"] += 1
            identity = step["composite_identity_maxdiff"]
            if identity is not None and math.isfinite(identity):
                composition["composite_identity_maxdiff"] = max(
                    composition["composite_identity_maxdiff"], float(identity)
                )
            if step["composite_peak"] != step["pointer_peak"]:
                composition["pointer_argmax_mismatch"] += 1
            if step["pointer_onehot_at_composite"] is False:
                composition["pointer_onehot_mismatch"] += 1
            if step["source_equals_beta_peak"] is False:
                composition["source_equals_beta_peak_mismatch"] += 1
            if step["decode_peak_matches_composite"] is False:
                composition["decode_copy_peak_mismatch"] += 1
    composition["composite_identity_tolerance"] = 1.0e-12
    composition["composite_identity_ok"] = composition["composite_identity_maxdiff"] <= 1.0e-12
    return {"composition": composition, "summary": summary}


def worked_example(path: Path, fixture: dict) -> dict:
    """One concrete object/place start and one carried tail, for the report text."""
    candidate = load_routing_bundle(path)[0]
    probe = probe_mod.ProbeContext(candidate)
    probe.arm = path.name.split("-")[1]
    ledger = instrument.measure_fixture(probe, fixture)
    by_position = {step["position"]: step for step in ledger["steps"]}
    starts = {}
    for name, positions in (
        ("object", instrument.OBJECT_POSITIONS),
        ("place", instrument.PLACE_POSITIONS),
    ):
        step = by_position[positions[0]]
        starts[name] = {
            "position": step["position"],
            "record": step["record"],
            "gold_offset": step["offset"],
            "route_peak": step["route_peak"],
            "beta_peak": step["beta_peak"],
            "source_peak": step["source_peak"],
            "prebias_peak": step["prebias_peak"],
            "pointer_bias_peak": step["pointer_bias_peak"],
            "composite_peak": step["composite_peak"],
            "row_source": step["row_source"],
            "identity_maxdiff": step["composite_identity_maxdiff"],
            "beta_row_hit": step["beta_row_hit"],
            "pointer_row_hit": step["pointer_row_hit"],
            "composite_row_hit": step["composite_row_hit"],
            "route_hit": step["route_hit"],
            "decode_hit": step["decode_hit"],
        }
    carry = next(
        (step for step in ledger["steps"] if step["row_source"] == "carry"),
        None,
    )
    return {
        "fixture_index": fixture["index"],
        "fixture_world": fixture["world"],
        "target": "".join(f"{byte:02x}" for byte in fixture["target_bytes"]),
        "starts": starts,
        "carry_example": (
            None
            if carry is None
            else {
                "position": carry["position"],
                "record": carry["record"],
                "offset": carry["offset"],
                "row_source": carry["row_source"],
                "row_bias_is_none": carry["pointer_bias_peak"] == -1,
                "composite_peak": carry["composite_peak"],
                "pointer_peak": carry["pointer_peak"],
                "note": "carry sets pointer = row_logits = one-hot(carried byte); no bias sum",
            }
        ),
    }


def checkpoint_roundtrip() -> dict:
    """Cheap post-edit save -> restore check (parameters only, no training)."""
    candidate, optimizer = build_routing_candidate(
        PARENT, 28031, "A_route_lock", span_bottleneck=32
    )
    state = initial_binding_state(candidate.binding, "d16-5-roundtrip")
    saved = save_routing_bundle(TEMP_BUNDLE, candidate, optimizer, 0, {}, {}, state)
    restored = load_routing_bundle(TEMP_BUNDLE)[0]
    identity_a = candidate_identity(candidate)
    identity_b = candidate_identity(restored)
    restored_tensors = parameter_tensors(restored)
    same_parameters = all(
        torch.equal(value, restored_tensors[key])
        for key, value in parameter_tensors(candidate).items()
    )
    TEMP_BUNDLE.unlink(missing_ok=True)
    return {
        "bundle_sha256": saved["sha256"],
        "bundle_bytes": saved["bytes"],
        "candidate_sha_v2_prefix": identity_a["candidate_sha_v2"][:8],
        "expected_candidate_sha_v2_prefix": "6c5466ab",
        "identity_semantic_match": identity_a["candidate_sha_v2"] == identity_b["candidate_sha_v2"],
        "identity_state_match": identity_a["candidate_state_sha256"]
        == identity_b["candidate_state_sha256"],
        "parameters_bitwise_equal": same_parameters,
        "span_parameters": sum(parameter.numel() for parameter in candidate.span_head.parameters()),
        "temp_bundle_removed": not TEMP_BUNDLE.exists(),
    }


def main() -> int:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_DIR))
    parser.add_argument("--step", default="step300")
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--manifest", default=str(MANIFEST))
    arguments = parser.parse_args()
    directory = Path(arguments.output_dir)

    payload = premise.build_payload()
    dev = probe_mod.dev_slice(premise.fixtures(payload, "dev"))

    cells = {}
    for seed in SEEDS:
        for arm in ARMS:
            path = directory / f"{seed}-{arm}-{arguments.step}.pt"
            cells[f"{seed}-{arm}"] = scan_cell(path, dev)

    finite_cells = [cell for cell in cells if cells[cell]["composition"]["parameters_nan"] == 0]
    contaminated_cells = [
        cell for cell in cells if cells[cell]["composition"]["parameters_nan"] > 0
    ]
    pooled = {}
    for world in ("object", "place"):
        keys = (
            "beta_start_hit_rate",
            "pointer_start_hit_rate",
            "composite_start_hit_rate",
            "route_hit_rate",
            "beta_row_hit_rate",
            "pointer_row_hit_rate",
            "composite_row_hit_rate",
            "start_byte_hit_rate",
            "decode_start_hit_rate",
            "carry_active_rate",
            "conditional_byte",
            "conditional_byte_beta_diagnostic",
            "conditional_byte_pointer_diagnostic",
            "conditional_byte_composite_row_only_diagnostic",
            "unconditional_byte",
        )
        pooled[world] = {
            key: statistics.fmean(cells[cell]["summary"][world][key] for cell in finite_cells)
            for key in keys
        }
    pooled["route_hit_rate"] = statistics.fmean(
        [pooled["object"]["route_hit_rate"], pooled["place"]["route_hit_rate"]]
    )

    by_arm = {}
    for arm in ARMS:
        arm_cells = [cell for cell in finite_cells if cell.split("-", maxsplit=1)[1] == arm]
        by_arm[arm] = {
            "cells": len(arm_cells),
            "object": {
                "beta_start_hit": statistics.fmean(
                    cells[cell]["summary"]["object"]["beta_start_hit_rate"] for cell in arm_cells
                ),
                "pointer_start_hit": statistics.fmean(
                    cells[cell]["summary"]["object"]["pointer_start_hit_rate"] for cell in arm_cells
                ),
                "composite_start_hit": statistics.fmean(
                    cells[cell]["summary"]["object"]["composite_start_hit_rate"]
                    for cell in arm_cells
                ),
                "route_hit": statistics.fmean(
                    cells[cell]["summary"]["object"]["route_hit_rate"] for cell in arm_cells
                ),
                "decode_start_hit": statistics.fmean(
                    cells[cell]["summary"]["object"]["decode_start_hit_rate"] for cell in arm_cells
                ),
                "conditional_byte": statistics.fmean(
                    cells[cell]["summary"]["object"]["conditional_byte"] for cell in arm_cells
                ),
            },
            "place": {
                "beta_start_hit": statistics.fmean(
                    cells[cell]["summary"]["place"]["beta_start_hit_rate"] for cell in arm_cells
                ),
                "pointer_start_hit": statistics.fmean(
                    cells[cell]["summary"]["place"]["pointer_start_hit_rate"] for cell in arm_cells
                ),
                "composite_start_hit": statistics.fmean(
                    cells[cell]["summary"]["place"]["composite_start_hit_rate"]
                    for cell in arm_cells
                ),
                "route_hit": statistics.fmean(
                    cells[cell]["summary"]["place"]["route_hit_rate"] for cell in arm_cells
                ),
                "decode_start_hit": statistics.fmean(
                    cells[cell]["summary"]["place"]["decode_start_hit_rate"] for cell in arm_cells
                ),
                "conditional_byte": statistics.fmean(
                    cells[cell]["summary"]["place"]["conditional_byte"] for cell in arm_cells
                ),
            },
        }

    composition_pooled = {}
    for key in (
        "content_rows",
        "transport_rows",
        "carry_rows",
        "forced_rows",
        "start_transport_rows",
        "start_carry_rows",
        "start_decode_rows",
        "start_decode_mismatch",
        "composite_identity_maxdiff",
        "pointer_argmax_mismatch",
        "pointer_onehot_mismatch",
        "source_equals_beta_peak_mismatch",
        "decode_copy_peak_mismatch",
    ):
        values = [cells[cell]["composition"][key] for cell in finite_cells]
        composition_pooled[key] = (
            max(values) if key == "composite_identity_maxdiff" else sum(values)
        )
    composition_pooled["composite_identity_tolerance"] = 1.0e-12
    composition_pooled["composite_identity_ok"] = (
        composition_pooled["composite_identity_maxdiff"] <= 1.0e-12
    )
    composition_pooled["cells_identity_ok"] = all(
        cells[cell]["composition"]["composite_identity_ok"] for cell in finite_cells
    )
    composition_pooled["finite_cells"] = finite_cells
    composition_pooled["contaminated_cells"] = contaminated_cells

    example_cell = next(
        (cell for cell in finite_cells if cell.endswith("C_unlocked")), finite_cells[0]
    )
    example = worked_example(directory / f"{example_cell}-{arguments.step}.pt", dev[0])
    roundtrip = checkpoint_roundtrip()

    readings_table = {
        name: {world: pooled[world][f"{name}_start_hit_rate"] for world in ("object", "place")}
        for name in ("beta", "pointer", "composite")
    }
    readings_table["decode"] = {
        world: pooled[world]["decode_start_hit_rate"] for world in ("object", "place")
    }

    report = {
        "schema": "taiji-package19-2b-composite-instrument-v1",
        "step": "D16-5",
        "date": "2026-11-04",
        "kind": "instrument_erratum",
        "instrument": {
            "object": "final transport row argmax used by the decode: "
            "row_logits = transport_logits(alpha_i * betas_i) + pointer_bias_i; "
            "composite = argmax(row_logits); a committed carry sets "
            "pointer = row_logits = one-hot(carried byte) and skips the sum",
            "source": "alpha_i scales the record mass; betas_i picks the transport source "
            "byte, so the source is beta's mode whenever alpha_i > 0",
            "gate_reading": "composite_start_hit, conditioned on the alpha route owning "
            "the gold record; route_hit remains a separate >= 0.95 clause",
            "parallel_reporting": "beta_start_hit / pointer_start_hit / composite_start_hit "
            "are the same route-conditioned quantity on the same examples; no cherry-picking",
            "gamma": "gamma = generation weight / 1 - gamma = copy weight; no training "
            "credit is inferred (M3)",
        },
        "changed": {
            "measurement_object": "argmax(read.betas) -> argmax(row_logits = transport + span bias)",
            "criteria_changed": False,
            "gate_values": {
                "composite_start_hit": 0.95,
                "route_hit": 0.95,
                "conditional_byte": {"object": 0.94, "place": 0.98},
            },
            "seen_before_change": "19.5 readings (start_hit 0.347/0.340, "
            "conditional_byte 0.941/0.980) were already read before this erratum; "
            "the change is registered as D16-5 on 2026-11-04",
            "legacy_rule": "19.5 values are preserved verbatim in legacy_instrument and "
            "are not overwritten or recomputed in the archived report",
        },
        "composition_check": {
            "pooled": composition_pooled,
            "cells": {
                cell: {
                    key: cells[cell]["composition"][key]
                    for key in (
                        "parameters_nan",
                        "transport_rows",
                        "carry_rows",
                        "start_transport_rows",
                        "start_carry_rows",
                        "start_decode_rows",
                        "start_decode_mismatch",
                        "composite_identity_maxdiff",
                        "pointer_argmax_mismatch",
                        "pointer_onehot_mismatch",
                        "source_equals_beta_peak_mismatch",
                        "decode_copy_peak_mismatch",
                    )
                }
                for cell in cells
            },
            "worked_example_cell": example_cell,
            "worked_example": example,
        },
        "readings": {
            "cells": {
                cell: {world: cells[cell]["summary"][world] for world in ("object", "place")}
                for cell in cells
            },
            "pooled": {
                world: {
                    key: pooled[world][key]
                    for key in (
                        "beta_start_hit_rate",
                        "pointer_start_hit_rate",
                        "composite_start_hit_rate",
                        "route_hit_rate",
                        "beta_row_hit_rate",
                        "pointer_row_hit_rate",
                        "composite_row_hit_rate",
                        "start_byte_hit_rate",
                        "decode_start_hit_rate",
                        "carry_active_rate",
                        "conditional_byte",
                        "conditional_byte_beta_diagnostic",
                        "conditional_byte_pointer_diagnostic",
                        "conditional_byte_composite_row_only_diagnostic",
                        "unconditional_byte",
                    )
                }
                for world in ("object", "place")
            },
            "three_readings": readings_table,
            "route_hit_pooled": pooled["route_hit_rate"],
            "by_arm": by_arm,
            "pooled_over": finite_cells,
            "contaminated_cells": contaminated_cells,
        },
        "legacy_instrument": {
            "source_report": str(LEGACY_REPORT.relative_to(ROOT)),
            "label": "legacy_instrument (19.5, beta-conditioned; do not overwrite)",
            "values": LEGACY,
            "note": "the archived 19.5 report still contains these numbers; this run only "
            "reads the bundles and writes a new report.  The pooled 0.347/0.340 includes "
            "the contaminated 28031-A cell and averages three different arms; per-cell and "
            "per-arm values are reported above",
        },
        "parameter_health": {
            "contaminated_cells": contaminated_cells,
            "nan_values": {cell: cells[cell]["composition"]["parameters_nan"] for cell in cells},
            "rule": "cells with any NaN parameter are excluded from pooled statistics and "
            "cannot serve as a baseline",
        },
        "checkpoint_roundtrip_after_edit": roundtrip,
        "cost": {
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.perf_counter() - start_wall,
            "adam_steps": 0,
        },
        "limits": {"zero_updates": True, "read_only": True, "confirm_reads": 0},
    }
    write_json(Path(arguments.report), report)
    write_json(
        Path(arguments.manifest),
        {
            "schema": "taiji-manifest-v1",
            "step": "D16-5",
            "report": str(Path(arguments.report).relative_to(ROOT)),
            "cells": len(cells),
            "examples_per_cell": cells[f"{SEEDS[0]}-{ARMS[0]}"]["summary"]["object"]["examples"],
            "readings_table": readings_table,
            "by_arm": by_arm,
            "contaminated_cells": contaminated_cells,
            "composition_ok": composition_pooled["cells_identity_ok"],
            "roundtrip_ok": roundtrip["identity_semantic_match"]
            and roundtrip["parameters_bitwise_equal"],
            "cost_cpu_seconds": report["cost"]["cpu_seconds"],
        },
    )
    print(
        json.dumps(
            {
                "three_readings": readings_table,
                "route_hit": {
                    world: pooled[world]["route_hit_rate"] for world in ("object", "place")
                },
                "conditional_byte": {
                    world: pooled[world]["conditional_byte"] for world in ("object", "place")
                },
                "composition_ok": composition_pooled["cells_identity_ok"],
                "max_identity_diff": composition_pooled["composite_identity_maxdiff"],
                "start_carry_rows": composition_pooled["start_carry_rows"],
                "report": str(arguments.report),
            }
        )[:1200]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
