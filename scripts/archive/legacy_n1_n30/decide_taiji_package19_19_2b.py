"""19.2b decision: merge the six-cell readings with the step300 ledger probe.

Applies the owner-frozen gates exactly as written in
``plans/manifests/taiji_package19_2b_prereg_20261104.json``:

D16-5 erratum (2026-11-04): the measured object is ``composite_start_hit`` --
route owns the gold record AND ``argmax(row_logits = transport + span bias)``
is the entity's first byte, i.e. the landing spot the decode actually uses.
``beta_start_hit`` (the old 19.5 object) and ``pointer_start_hit`` (the bare span
head) are reported beside it and never replace it; ``conditional_byte`` is
conditioned on the composite hit, with beta/pointer conditions as diagnostics.

stage gate  composite_start_hit object/place >= 0.95, route_hit >= 0.95,
            conditional_byte object >= 0.94 / place >= 0.98 (no regression);
final gate  byte object/place >= 0.95, copied >= 0.60, overall >= 0.80,
            unseen >= 0.80, utf8 frozen path >= 0.98, EOS >= 0.95,
            new_description >= 0.80.

Only if both gates pass does this script read the confirm split, exactly once,
then seal.  Otherwise confirm is never opened.  The verdict keeps magnitudes
(start vs gate vs encoder vs sentence frame) and never infers training credit
from gamma (M3): gamma is the generation weight, 1 - gamma the copy weight.

Run: ``python scripts/training/decide_taiji_package19_19_2b.py``
"""

from __future__ import annotations

import argparse
import importlib.util
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

COMPARE = ROOT / "reports/taiji_package19_2b_compare_20261104.json"
PROBE = ROOT / "reports/taiji_package19_2b_probe_step300_20261104.json"
LOWUPDATE = ROOT / "reports/taiji_package19_2b_lowupdate_composite_20261104.json"
LOWUPDATE_LEGACY = ROOT / "reports/taiji_package19_2b_lowupdate_20261104.json"
PREFLIGHT = ROOT / "reports/taiji_preflight_19_2b_20261104.json"
PRICING = ROOT / "plans/manifests/taiji_pricing_19_2b_20261104.json"
PREREG = ROOT / "plans/manifests/taiji_package19_2b_prereg_20261104.json"
BUNDLES = ROOT / "output/taiji_package19_2b_compare"
CONFIRM_DIR = ROOT / "output/taiji_package19_2b_confirm"
REPORT = ROOT / "reports/taiji_package19_2b_decision_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2b_decision_20261104.json"
COMPARE_MODULE = ROOT / "scripts/training/compare_taiji_selection_routing_17_5.py"
C19_MODULE = ROOT / "scripts/training/compare_taiji_package19.py"

SEEDS = (28031, 28032)
ARMS = ("A_route_lock", "B_stop_lock", "C_unlocked")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def confirm_read(seed: int, arm: str) -> dict:
    """One confirm pass: no seal -> read and seal; sealed -> refuse."""
    CONFIRM_DIR.mkdir(parents=True, exist_ok=True)
    seal = CONFIRM_DIR / "SEALED.json"
    if seal.exists():
        return {"read": False, "reason": "already sealed", "seal": read_json(seal)}
    from taiji import data_premise_18_0 as premise
    from taiji.selection_routing_checkpoint import load_routing_bundle

    cmp_mod = load_module(COMPARE_MODULE, "cmp175_confirm")
    c19 = load_module(C19_MODULE, "cmp19_2b_confirm")
    payload = premise.build_payload()
    fixtures = premise.fixtures(payload, "confirm")
    candidate = load_routing_bundle(BUNDLES / f"{seed}-{arm}-step300.pt")[0]
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    evaluation = cmp_mod.evaluate_dev(candidate, fixtures, arm, "confirm", full=True)
    copy_out = c19.copy_probe(candidate, fixtures, arm, "confirm")
    measured = c19.readings(evaluation, copy_out)
    cpu = time.process_time() - start_cpu
    wall = time.perf_counter() - start_wall
    record = {
        "read": True,
        "cell": f"{seed}-{arm}",
        "examples": len(fixtures),
        "overall": measured["overall"],
        "new_name_copied": measured["new_name_copied"],
        "unseen_output_name": measured["subsets"]["unseen_output_name"],
        "new_description": measured["subsets"]["new_description"],
        "by_world": measured["by_world"],
        "cost": {"cpu_seconds": cpu, "wall_seconds": wall, "adam_steps": 0},
    }
    write_json(CONFIRM_DIR / "confirm_read.json", record)
    write_json(
        seal,
        {
            "sealed": True,
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cell": record["cell"],
            "reads_used": 1,
            "reads_allowed": 1,
            "rule": "a second read refuses; the confirm split is spent",
        },
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-confirm", action="store_true")
    arguments = parser.parse_args()

    compare = read_json(COMPARE)
    probe = read_json(PROBE)
    lowupdate = (
        read_json(LOWUPDATE)
        if LOWUPDATE.exists()
        else (read_json(LOWUPDATE_LEGACY) if LOWUPDATE_LEGACY.exists() else {})
    )
    preflight = read_json(PREFLIGHT) if PREFLIGHT.exists() else {}
    pricing = read_json(PRICING)

    cells = compare["cells"]
    ledger = {f"{cell['seed']}-{cell['arm']}": cell["ledger"] for cell in probe["cells"].values()}
    sweep_model = {
        f"{cell['seed']}-{cell['arm']}": cell["sweep"]["model"] for cell in probe["cells"].values()
    }
    utf8_generate = {
        f"{cell['seed']}-{cell['arm']}": cell["utf8"]["generate_only"]["legal_rate"]
        for cell in probe["cells"].values()
    }

    def healthy(cell_key: str) -> bool:
        values = ledger[cell_key]
        numbers = [
            values[world][field]
            for world in ("object", "place")
            for field in (
                "composite_start_hit_rate",
                "beta_start_hit_rate",
                "pointer_start_hit_rate",
                "route_hit_rate",
                "conditional_byte",
            )
        ]
        return all(isinstance(value, (int, float)) and math.isfinite(value) for value in numbers)

    pooled_cells = [cell_key for cell_key in ledger if healthy(cell_key)]
    excluded_cells = [cell_key for cell_key in ledger if cell_key not in pooled_cells]

    def pooled_field_healthy(world: str, field: str) -> float:
        return statistics.fmean(ledger[cell_key][world][field] for cell_key in pooled_cells)

    stage_readings = {
        "composite_start_hit_object": pooled_field_healthy("object", "composite_start_hit_rate"),
        "composite_start_hit_place": pooled_field_healthy("place", "composite_start_hit_rate"),
        "beta_start_hit_object": pooled_field_healthy("object", "beta_start_hit_rate"),
        "beta_start_hit_place": pooled_field_healthy("place", "beta_start_hit_rate"),
        "pointer_start_hit_object": pooled_field_healthy("object", "pointer_start_hit_rate"),
        "pointer_start_hit_place": pooled_field_healthy("place", "pointer_start_hit_rate"),
        "decode_start_hit_object": pooled_field_healthy("object", "decode_start_hit_rate"),
        "decode_start_hit_place": pooled_field_healthy("place", "decode_start_hit_rate"),
        "route_hit_object": pooled_field_healthy("object", "route_hit_rate"),
        "route_hit_place": pooled_field_healthy("place", "route_hit_rate"),
        "conditional_byte_object": pooled_field_healthy("object", "conditional_byte"),
        "conditional_byte_place": pooled_field_healthy("place", "conditional_byte"),
        "conditional_byte_beta_diagnostic_object": pooled_field_healthy(
            "object", "conditional_byte_beta_diagnostic"
        ),
        "conditional_byte_beta_diagnostic_place": pooled_field_healthy(
            "place", "conditional_byte_beta_diagnostic"
        ),
        "conditional_byte_pointer_diagnostic_object": pooled_field_healthy(
            "object", "conditional_byte_pointer_diagnostic"
        ),
        "conditional_byte_pointer_diagnostic_place": pooled_field_healthy(
            "place", "conditional_byte_pointer_diagnostic"
        ),
        "unconditional_byte_object": pooled_field_healthy("object", "unconditional_byte"),
        "unconditional_byte_place": pooled_field_healthy("place", "unconditional_byte"),
    }
    per_arm = {
        arm: {
            cell_key: {"object": ledger[cell_key]["object"], "place": ledger[cell_key]["place"]}
            for cell_key in ledger
            if cell_key.endswith(f"-{arm}")
        }
        for arm in ARMS
    }
    stage_gate = {
        "composite_start_hit_object_0.95": stage_readings["composite_start_hit_object"] >= 0.95,
        "composite_start_hit_place_0.95": stage_readings["composite_start_hit_place"] >= 0.95,
        "route_hit_0.95": min(stage_readings["route_hit_object"], stage_readings["route_hit_place"])
        >= 0.95,
        "conditional_object_0.94": stage_readings["conditional_byte_object"] >= 0.94,
        "conditional_place_0.98": stage_readings["conditional_byte_place"] >= 0.98,
        "conditional_no_regression": stage_readings["conditional_byte_object"] >= 0.941
        and stage_readings["conditional_byte_place"] >= 0.980,
    }
    stage_pass = all(stage_gate.values())

    final_readings = {
        "byte_object": statistics.fmean(
            sweep_model[cell_key]["object_byte_rate"] for cell_key in sweep_model
        ),
        "byte_place": statistics.fmean(
            sweep_model[cell_key]["place_byte_rate"] for cell_key in sweep_model
        ),
        "new_name_copied": compare["pooled"]["new_name_copied"],
        "overall": compare["pooled"]["overall"],
        "unseen_output_name": compare["pooled"]["unseen_output_name"],
        "utf8_frozen_path": compare["pooled"]["utf8"],
        "utf8_generate_only": statistics.fmean(utf8_generate.values()),
        "eos": compare["pooled"]["eos"],
        "new_description": compare["pooled"]["new_description"],
    }
    final_gate = {
        "byte_object_0.95": final_readings["byte_object"] >= 0.95,
        "byte_place_0.95": final_readings["byte_place"] >= 0.95,
        "copied_0.60": final_readings["new_name_copied"] >= 0.60,
        "overall_0.80": final_readings["overall"] >= 0.80,
        "unseen_0.80": final_readings["unseen_output_name"] >= 0.80,
        "utf8_0.98": final_readings["utf8_frozen_path"] >= 0.98,
        "eos_0.95": final_readings["eos"] >= 0.95,
        "new_description_0.80": final_readings["new_description"] >= 0.80,
    }
    final_pass = all(final_gate.values())

    low_start_deltas = lowupdate.get("deltas", {})
    low_start_rising = all(
        isinstance(low_start_deltas.get(world, {}).get("composite_start_hit_rate"), float)
        and low_start_deltas[world]["composite_start_hit_rate"] > 0.0
        for world in ("object", "place")
    )
    low_row_rising = bool(lowupdate.get("composite_row_rising_on_both_worlds"))

    if not low_start_rising:
        branch = "low_update_not_rising"
        verdict = (
            "D16-5 pre-spend reading did not rise on the gate object "
            "(composite_start_hit); the six cells are not released, no extra dose, no 19.5b"
        )
    elif not stage_pass:
        branch = "stage_gate_failed"
        verdict = "19.2b insufficient at the start pointer; confirm sealed"
    elif stage_pass and final_pass:
        branch = "both_passed"
        verdict = "19.2b passed both gates; confirm released once"
    elif (
        stage_readings["composite_start_hit_object"] >= 0.95
        and final_readings["new_description"] < 0.80
    ):
        branch = "stage_passed_new_description_failed"
        verdict = (
            "composite_start_hit reached the gate but new_description stayed below 0.80; "
            "19.2b insufficient, do not add start dose, switch to sentence-frame attribution"
        )
    else:
        branch = "stage_passed_final_failed"
        verdict = "19.2b insufficient on the final gate; confirm sealed"

    encoder_content = statistics.fmean(
        (
            cell["c_gamma0"]["content_byte_rate"]
            if "content_byte_rate" in cell["c_gamma0"]
            else cell["c_gamma0"]["overall"]
        )
        for cell in probe["cells"].values()
    )
    attribution = {
        "start_gap_magnitude": 1.0
        - min(
            stage_readings["composite_start_hit_object"],
            stage_readings["composite_start_hit_place"],
        ),
        "gate_gap_magnitude": 1.0
        - min(stage_readings["route_hit_object"], stage_readings["route_hit_place"]),
        "encoder_gap_magnitude": 1.0 - encoder_content,
        "sentence_frame_shortfall": max(0.0, 0.80 - final_readings["new_description"]),
        "sentence_shortfall": max(0.0, 0.80 - final_readings["overall"]),
        "copy_weight_note": "gamma is the generation weight, 1 - gamma the copy weight; "
        "gamma never implies training credit (M3)",
    }
    ranked = sorted(
        [
            ("start", attribution["start_gap_magnitude"]),
            ("gate", attribution["gate_gap_magnitude"]),
            ("encoder", attribution["encoder_gap_magnitude"]),
            ("sentence_frame", attribution["sentence_frame_shortfall"]),
        ],
        key=lambda item: item[1],
        reverse=True,
    )

    confirm = {
        "read": False,
        "reason": (
            "pre-spend reading did not rise; the six cells are sealed"
            if branch == "low_update_not_rising"
            else "gates not both passed"
        ),
    }
    if branch == "both_passed" and not arguments.no_confirm:
        candidates = sorted(
            cells.items(),
            key=lambda item: (-item[1]["overall"], ARMS.index(item[1]["arm"]), item[1]["seed"]),
        )
        best = candidates[0][1]
        confirm = confirm_read(best["seed"], best["arm"])

    package_cost = {
        "pricing": 0.0,
        "lowupdate": lowupdate.get("cost", {}).get("cpu_seconds", 0.0),
        "preflight": preflight.get("cost", {}).get("cpu_seconds", 0.0),
        "compare_six_cells": compare["cost"]["cpu_seconds"],
        "probe": probe["cost"]["cpu_seconds"],
        "confirm": confirm.get("cost", {}).get("cpu_seconds", 0.0) if confirm.get("read") else 0.0,
    }
    total = sum(package_cost.values())
    report = {
        "schema": "taiji-package19-2b-decision-v1",
        "step": "N19.2b",
        "date": "2026-11-04",
        "verdict": verdict,
        "branch": branch,
        "stage_gate": {"readings": stage_readings, "checks": stage_gate, "passed": stage_pass},
        "instrument": {
            "gate_object": "composite_start_hit = route owns gold record AND "
            "argmax(row_logits = transport + span bias) is the first byte",
            "side_by_side": {
                "beta": {
                    "object": stage_readings["beta_start_hit_object"],
                    "place": stage_readings["beta_start_hit_place"],
                },
                "pointer": {
                    "object": stage_readings["pointer_start_hit_object"],
                    "place": stage_readings["pointer_start_hit_place"],
                },
                "composite": {
                    "object": stage_readings["composite_start_hit_object"],
                    "place": stage_readings["composite_start_hit_place"],
                },
                "decode": {
                    "object": stage_readings["decode_start_hit_object"],
                    "place": stage_readings["decode_start_hit_place"],
                },
            },
            "legacy_instrument": "19.5 beta-conditioned readings stay archived; "
            "see reports/taiji_package19_2b_composite_instrument_20261104.json",
        },
        "aggregation": {
            "rule": "pooled across healthy cells; per-cell and per-arm reported",
            "pooled_cells": pooled_cells,
            "excluded_cells": excluded_cells,
            "per_arm": per_arm,
        },
        "final_gate": {"readings": final_readings, "checks": final_gate, "passed": final_pass},
        "attribution": attribution,
        "attribution_ranked": ranked,
        "low_update_validation": {
            "report": str(LOWUPDATE.relative_to(ROOT)) if LOWUPDATE.exists() else None,
            "composite_start_rising_on_both_worlds": low_start_rising,
            "composite_row_rising_on_both_worlds": low_row_rising,
            "deltas": lowupdate.get("deltas"),
            "adam_steps": lowupdate.get("accounting", {}).get("adam_steps"),
            "independent_trajectories": lowupdate.get("accounting", {}).get(
                "independent_trajectories"
            ),
            "release_rule": "only a rising composite_start_hit releases the six cells",
            "cost": lowupdate.get("cost"),
        },
        "confirm": confirm,
        "cost": {
            "by_item": package_cost,
            "package_total_cpu": total,
            "pricing_bare_six": pricing["budget_bare_cpu"],
            "pricing_protected_six": pricing["budget_protected_cpu"],
            "measured_19_5_six": pricing["measured_19_5_cpu"],
            "budget_status": (
                "within bare"
                if compare["cost"]["cpu_seconds"] <= pricing["budget_bare_cpu"]
                else "over bare"
            ),
            "watchlines": compare["cost"]["watchlines"],
            "watchline_hits": compare["cost"]["watchline_hits"],
        },
        "per_cell": cells,
        "baseline_19_5": {
            "bundles": "output/taiji_package19_compare",
            "verdict": "negative (overall 0.00-0.63; legacy beta start_hit 0.347/0.340)",
            "contaminated_cell": "28031-A_route_lock (all binding/head parameters NaN)",
            "instrument_note": "19.5 gate read argmax(betas); D16-5 reads the final row argmax",
        },
        "prereg": str(PREREG.relative_to(ROOT)),
    }
    write_json(REPORT, report)
    write_json(
        MANIFEST,
        {
            "schema": "taiji-package19-2b-decision-manifest-v1",
            "step": "N19.2b",
            "report": str(REPORT.relative_to(ROOT)),
            "verdict": verdict,
            "branch": branch,
            "stage_pass": stage_pass,
            "final_pass": final_pass,
            "confirm": confirm,
            "package_total_cpu": total,
            "gate_readings": {**stage_readings, **final_readings},
        },
    )
    print(
        json.dumps(
            {
                "branch": branch,
                "stage": {
                    key: round(value, 3) if isinstance(value, float) else value
                    for key, value in stage_readings.items()
                },
                "stage_pass": stage_pass,
                "final": {
                    key: round(value, 3) if isinstance(value, float) else value
                    for key, value in final_readings.items()
                },
                "final_pass": final_pass,
                "composite_start_rising": low_start_rising,
                "composite_row_rising": low_row_rising,
                "attribution_ranked": [(name, round(value, 3)) for name, value in ranked],
                "confirm": confirm.get("read"),
                "cost_total": round(total, 1),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
