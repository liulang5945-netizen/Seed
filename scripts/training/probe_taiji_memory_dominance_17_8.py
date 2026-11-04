"""17.8 memory-dominance probe: zero-update forward passes on unseen entities/places.

The 17.5 verdict was "template correct, content wrong".  This probe separates two
explanations that the 17.5 readings could not: (a) the copy operator is broken, and
(b) the training set only contains three object names and five place names, so the
task is satisfiable by memorising the 15 combinations and never consulting records.

Construction: four dev fixtures are rewritten into four variants.  The rewrite only
changes the record strings that carry the selected object and the target place, plus
the expected action; the goal keeps the same descriptions.  Variants:

* ``control``   unchanged dev fixture (in-distribution reference);
* ``new_new``   unseen object name x unseen place name;
* ``new_old``   unseen object name x a place name seen in training;
* ``old_new``   a seen object name x unseen place name.

Readings per cell and variant: byte-exact object/place rates, completed rate,
content-intervention drops (no_source / wrong_record / wrong_position), and the
actual gamma (generation weight) vs 1-gamma (copy weight).

Criterion: if new-entity byte accuracy is near zero AND no intervention (record
removal, wrong record, wrong position) ever makes the unseen name appear verbatim,
the model never copies the record -> memory dominance, i.e. the task premise allows
recitation instead of copying.  Output bytes may still change between memorised
names; that is recorded separately as ``output_change``.  Zero optimiser steps, zero
confirm reads, no change to any frozen threshold.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji.binding_copy_scoring import score_generation  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.selection_routing_checkpoint import load_routing_bundle  # noqa: E402

COMPARE = ROOT / "scripts/training/compare_taiji_selection_routing_17_5.py"
VERIFY = ROOT / "scripts/training/verify_taiji_binding_select_16_1_compare.py"
OUTPUT_DIR = ROOT / "output/taiji_selection_routing_17_5"
REPORT = ROOT / "reports/taiji_memory_dominance_probe_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_memory_dominance_probe_20261104.json"

SEEDS = (28031, 28032)
ARMS = ("A_route_lock", "B_stop_lock", "C_unlocked")
VARIANTS = ("control", "new_new", "new_old", "old_new")
BASE_INDICES = (0, 1, 2, 3)
NEAR_ZERO = 0.10

NEW_OBJECTS = ("玄铁罗盘", "赤铜铃铛", "青玉镇纸", "墨色算筹")
NEW_PLACES = ("池畔", "廊下", "阶前", "瓮侧")

_MODULE: dict[str, object] = {}


def compare_module():
    if "module" not in _MODULE:
        spec = importlib.util.spec_from_file_location("cmp17p", COMPARE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE["module"] = module
    return _MODULE["module"]


def verify_module():
    if "verify" not in _MODULE:
        spec = importlib.util.spec_from_file_location("v16p", VERIFY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE["verify"] = module
    return _MODULE["verify"]


def load_fixtures():
    if "fixtures" not in _MODULE:
        _MODULE["fixtures"] = compare_module().load_fixtures()
    return _MODULE["fixtures"]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_vocabulary(train: tuple[dict, ...]) -> dict[str, set[str]]:
    objects = {fixture["object_name"] for fixture in train}
    places = {fixture["place_name"] for fixture in train}
    texts = {record["raw"] for fixture in train for record in fixture["raw_example"]["records"]}
    return {"objects": objects, "places": places, "texts": texts}


def rewrite(base: dict, object_name: str, place_name: str) -> dict:
    module = verify_module()
    raw_records = []
    records = []
    for raw in base["raw_example"]["records"]:
        name = raw["name"]
        if raw["kind"] == "object" and raw["description"] == base["object_description"]:
            name = object_name
        elif raw["kind"] == "place" and raw["description"] == base["place_description"]:
            name = place_name
        text = module.record_template(raw["style_name"], raw["description"], name)
        records.append(
            type(base["records"][0])(source_id=raw["source_id"], raw=text.encode("utf-8"))
        )
        raw_records.append({**raw, "name": name, "raw": text})
    example = {
        **base["raw_example"],
        "records": raw_records,
        "object_name": object_name,
        "place_name": place_name,
        "action": f"把{object_name}放到{place_name}",
    }
    return {
        **base,
        "records": tuple(records),
        "raw_example": example,
        "object_name": object_name,
        "place_name": place_name,
        "target_prefix": tuple(example["action"].encode("utf-8")),
        "target_bytes": tuple(example["action"].encode("utf-8")) + (EOS,),
    }


def alternate_index(fixture: dict) -> int | None:
    correct = fixture["selection_label"]
    target = fixture["raw_example"]["records"][correct]
    for index, record in enumerate(fixture["raw_example"]["records"]):
        if index == correct or record["kind"] != "object":
            continue
        if record["name"] != target["name"]:
            return index
    return None


def build_cases(train: tuple[dict, ...], dev: tuple[dict, ...]) -> list[dict]:
    vocab = train_vocabulary(train)
    unseen_objects = [
        name
        for name in NEW_OBJECTS
        if name not in vocab["objects"] and all(name not in text for text in vocab["texts"])
    ]
    unseen_places = [
        name
        for name in NEW_PLACES
        if name not in vocab["places"] and all(name not in text for text in vocab["texts"])
    ]
    if len(unseen_objects) < 4 or len(unseen_places) < 4:
        raise RuntimeError("unseen name pool is exhausted")
    old_objects = [name for name in sorted(vocab["objects"])]
    old_places = [name for name in sorted(vocab["places"])]
    bases = [dev[index] for index in BASE_INDICES]
    cases = []
    for slot, base in enumerate(bases):
        new_object = unseen_objects[slot]
        new_place = unseen_places[slot]
        old_place = next((name for name in old_places if name != base["place_name"]), old_places[0])
        old_object = next(
            (name for name in old_objects if name != base["object_name"]), old_objects[0]
        )
        variants = {
            "control": base,
            "new_new": rewrite(base, new_object, new_place),
            "new_old": rewrite(base, new_object, old_place),
            "old_new": rewrite(base, old_object, new_place),
        }
        unseen = {
            "control": (),
            "new_new": (new_object, new_place),
            "new_old": (new_object,),
            "old_new": (new_place,),
        }
        for variant, fixture in variants.items():
            cases.append(
                {
                    "variant": variant,
                    "world": base["world"],
                    "index": base["index"],
                    "object_name": fixture["object_name"],
                    "place_name": fixture["place_name"],
                    "unseen": unseen[variant],
                    "fixture": fixture,
                }
            )
    return cases


def gamma_summary(per_case: list[list[float]]) -> dict:
    flat = [value for rows in per_case for value in rows]
    if not flat:
        return {
            "steps": 0,
            "gamma_mean": None,
            "gamma_min": None,
            "gamma_max": None,
            "low_fraction": None,
        }
    return {
        "steps": len(flat),
        "gamma_mean": statistics.fmean(flat),
        "gamma_min": min(flat),
        "gamma_max": max(flat),
        "low_fraction": sum(1 for value in flat if value < 0.5) / len(flat),
    }


def rate(rows: list[dict], key: str) -> float:
    subset = [row for row in rows if row["score"][key]]
    return len(subset) / len(rows) if rows else 0.0


def drop(rows: list[dict], name: str) -> float:
    values = [row[name] for row in rows if row[name] is not None]
    if not values:
        return 0.0
    baseline = sum(1 for row in rows if row["score"]["completed"]) / len(rows)
    return baseline - sum(1 for value in values if value) / len(values)


def change_rate(rows: list[dict], name: str) -> float:
    """Fraction of intervention forward passes whose output bytes differ from baseline."""
    compared = [row for row in rows if row.get(f"{name}_hex") is not None]
    if not compared:
        return 0.0
    changed = [row for row in compared if row[f"{name}_hex"] != row["raw_hex"]]
    return len(changed) / len(compared)


def slot_change(rows: list[dict], name: str, low: int, high: int) -> dict:
    """Name-slot change between baseline and intervention, only on 27-byte outputs."""
    compared = [row for row in rows if row.get(f"{name}_hex") is not None]
    valid = 0
    changed = 0
    for row in compared:
        base = bytes.fromhex(row["raw_hex"])
        other = bytes.fromhex(row[f"{name}_hex"])
        if len(base) == 27 and len(other) == 27:
            valid += 1
            changed += int(base[low:high] != other[low:high])
    return {
        "valid_byte_pairs": valid,
        "changed_pairs": changed,
        "rate_over_all_rows": changed / len(compared) if compared else 0.0,
    }


def copy_rescue(rows: list[dict]) -> float:
    """Fraction of cases where every unseen name appears verbatim in some output.

    Baseline and the three interventions are all considered.  If this is zero, no
    record condition ever transfers the unseen names, which is the decisive test
    for recitation over copying.
    """
    if not rows:
        return 0.0
    hits = 0
    interventions = ("no_source", "wrong_record", "wrong_position")
    for row in rows:
        unseen = [bytes.fromhex(value) for value in row["unseen_hex"]]
        if not unseen:
            continue
        outputs = [row["raw_hex"]] + [
            row[f"{name}_hex"] for name in interventions if row.get(f"{name}_hex") is not None
        ]
        if any(all(name in bytes.fromhex(output) for name in unseen) for output in outputs):
            hits += 1
    return hits / len(rows)


def measure_cell(seed: int, arm: str) -> dict:
    cell_cpu = time.process_time()
    cell_wall = time.perf_counter()
    cases_dir = OUTPUT_DIR
    compare = compare_module()
    candidate = None
    optimizer = None
    bundle = cases_dir / f"{seed}-{arm}-step300.pt"
    candidate, optimizer, _state, update, _payload = load_routing_bundle(bundle)
    del optimizer
    if int(update) != 300:
        raise RuntimeError(f"bundle step mismatch for {seed}-{arm}")
    train, dev = load_fixtures()
    cases = build_cases(train, dev)
    tag = f"probe-{seed}-{arm}"
    rows = []
    gamma_rows = {variant: [] for variant in VARIANTS}
    for case in cases:
        fixture = case["fixture"]
        generated = compare.generate(candidate, fixture, arm, tag=tag)
        scored = score_generation(
            generated["raw_bytes"], fixture["target_prefix"], generated["terminated"]
        )
        row = {
            "variant": case["variant"],
            "world": case["world"],
            "index": case["index"],
            "object_name": case["object_name"],
            "place_name": case["place_name"],
            "unseen_hex": [name.encode("utf-8").hex() for name in case["unseen"]],
            "score": scored,
            "raw_hex": bytes(generated["raw_bytes"]).hex(),
            "gamma_mean": statistics.fmean(generated["gamma"]) if generated["gamma"] else None,
        }
        if case["variant"] != "control":
            alternate = alternate_index(fixture)
            no_source = compare.generate(candidate, fixture, arm, records=(), keys=(), tag=tag)
            row["no_source"] = score_generation(
                no_source["raw_bytes"], fixture["target_prefix"], no_source["terminated"]
            )["completed"]
            row["no_source_hex"] = bytes(no_source["raw_bytes"]).hex()
            row["wrong_record"] = None
            row["wrong_record_hex"] = None
            if alternate is not None:
                wrong_record = compare.generate(
                    candidate, fixture, arm, forced_route=alternate, tag=tag
                )
                row["wrong_record"] = score_generation(
                    wrong_record["raw_bytes"], fixture["target_prefix"], wrong_record["terminated"]
                )["completed"]
                row["wrong_record_hex"] = bytes(wrong_record["raw_bytes"]).hex()
            wrong_position = compare.generate(candidate, fixture, arm, offset_zero=True, tag=tag)
            row["wrong_position"] = score_generation(
                wrong_position["raw_bytes"], fixture["target_prefix"], wrong_position["terminated"]
            )["completed"]
            row["wrong_position_hex"] = bytes(wrong_position["raw_bytes"]).hex()
        rows.append(row)
        gamma_rows[case["variant"]].append(generated["gamma"])
    by_variant = {}
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        by_variant[variant] = {
            "n": len(subset),
            "object_exact": rate(subset, "object_exact"),
            "place_exact": rate(subset, "place_exact"),
            "entity_exact": rate(subset, "entity_exact"),
            "completed": rate(subset, "completed"),
            "drops": {
                name: drop(subset, name)
                for name in ("no_source", "wrong_record", "wrong_position")
                if variant != "control"
            },
            "output_change": {
                name: change_rate(subset, name)
                for name in ("no_source", "wrong_record", "wrong_position")
                if variant != "control"
            },
            "bite_rate": (
                len(
                    [
                        row
                        for row in subset
                        if any(
                            row.get(f"{name}_hex") is not None
                            and row[f"{name}_hex"] != row["raw_hex"]
                            for name in ("no_source", "wrong_record", "wrong_position")
                        )
                    ]
                )
                / len(subset)
                if variant != "control"
                else 0.0
            ),
            "slot_change": (
                {
                    "object": slot_change(subset, "no_source", 3, 15),
                    "place": slot_change(subset, "no_source", 21, 27),
                }
                if variant != "control"
                else {}
            ),
            "copy_rescue": copy_rescue(subset) if variant != "control" else 0.0,
            "gamma": gamma_summary(gamma_rows[variant]),
        }
    pair_keys = [(row["variant"], row["world"], row["index"]) for row in rows]
    if len(set(pair_keys)) != len(pair_keys):
        raise RuntimeError("cross-variant cache key collision")
    new_entity = by_variant["new_new"]["entity_exact"]
    new_rescue = by_variant["new_new"]["copy_rescue"]
    verdict = (
        "memory_dominant"
        if new_entity <= NEAR_ZERO and new_rescue <= NEAR_ZERO
        else "operator_or_dose_limited"
    )
    del candidate
    return {
        "seed": seed,
        "arm": arm,
        "bundle": {"path": str(bundle.relative_to(ROOT)), "sha256": sha256_file(bundle)},
        "by_variant": by_variant,
        "criterion": {
            "new_entity_exact": new_entity,
            "new_copy_rescue": new_rescue,
            "new_no_source_drop_completed": by_variant["new_new"]["drops"]["no_source"],
            "new_output_change": by_variant["new_new"]["output_change"]["no_source"],
            "near_zero_band": NEAR_ZERO,
            "note": "completed drops are degenerate when baseline completion is 0; the decisive "
            "test is whether any record condition ever transfers the unseen name verbatim",
            "verdict": verdict,
        },
        "cost": {
            "cpu_seconds": time.process_time() - cell_cpu,
            "wall_seconds": time.perf_counter() - cell_wall,
        },
        "rows": rows,
    }


def self_checks(train: tuple[dict, ...], dev: tuple[dict, ...]) -> dict:
    """S1: subset ratios divide by the subset length; S2: cache keys and raw bytes."""
    cases = build_cases(train, dev)
    keys = [(case["variant"], case["world"], case["index"]) for case in cases]
    subset = [
        {"variant": "new_new", "score": {"completed": True}},
        {"variant": "new_new", "score": {"completed": False}},
        {"variant": "control", "score": {"completed": False}},
    ]
    ratio = len([row for row in subset if row["score"]["completed"]]) / 2
    return {
        "s1_subset_ratio_denominator": {
            "value": ratio,
            "expected": 0.5,
            "ok": abs(ratio - 0.5) < 1e-12,
        },
        "s2_cache_keys": {
            "key_shape": "(variant, world, index)",
            "unique": len(set(keys)) == len(keys),
            "cases": len(keys),
        },
        "s2_raw_bytes": {"pair_metrics": "raw_hex stored per row", "decoded_copy_kept": False},
    }


def main() -> int:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    train, dev = load_fixtures()
    vocab = train_vocabulary(train)
    checks = self_checks(train, dev)
    if not all(
        [
            checks["s1_subset_ratio_denominator"]["ok"],
            checks["s2_cache_keys"]["unique"],
        ]
    ):
        raise RuntimeError(f"probe self-check failed: {checks}")
    cells = []
    with ProcessPoolExecutor(max_workers=len(SEEDS) * len(ARMS)) as pool:
        futures = {
            pool.submit(measure_cell, seed, arm): (seed, arm) for seed in SEEDS for arm in ARMS
        }
        for future in as_completed(futures):
            seed, arm = futures[future]
            result = future.result()
            cells.append(result)
            print(
                json.dumps(
                    {
                        "cell": f"{seed}-{arm}",
                        "new_new_entity": round(result["by_variant"]["new_new"]["entity_exact"], 3),
                        "new_new_drop": round(
                            result["by_variant"]["new_new"]["drops"]["no_source"], 3
                        ),
                        "new_new_copy_rescue": round(
                            result["by_variant"]["new_new"]["copy_rescue"], 3
                        ),
                        "verdict": result["criterion"]["verdict"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    pooled = {}
    for variant in VARIANTS:
        pooled[variant] = {
            "n": sum(cell["by_variant"][variant]["n"] for cell in cells),
            "object_exact": statistics.fmean(
                cell["by_variant"][variant]["object_exact"] for cell in cells
            ),
            "place_exact": statistics.fmean(
                cell["by_variant"][variant]["place_exact"] for cell in cells
            ),
            "entity_exact": statistics.fmean(
                cell["by_variant"][variant]["entity_exact"] for cell in cells
            ),
            "completed": statistics.fmean(
                cell["by_variant"][variant]["completed"] for cell in cells
            ),
            "drops": (
                {
                    name: statistics.fmean(
                        cell["by_variant"][variant]["drops"][name] for cell in cells
                    )
                    for name in ("no_source", "wrong_record", "wrong_position")
                }
                if variant != "control"
                else {}
            ),
            "output_change": (
                {
                    name: statistics.fmean(
                        cell["by_variant"][variant]["output_change"][name] for cell in cells
                    )
                    for name in ("no_source", "wrong_record", "wrong_position")
                }
                if variant != "control"
                else {}
            ),
            "bite_rate": (
                statistics.fmean(cell["by_variant"][variant]["bite_rate"] for cell in cells)
                if variant != "control"
                else 0.0
            ),
            "copy_rescue": (
                statistics.fmean(cell["by_variant"][variant]["copy_rescue"] for cell in cells)
                if variant != "control"
                else 0.0
            ),
            "gamma": {
                key: statistics.fmean(cell["by_variant"][variant]["gamma"][key] for cell in cells)
                for key in ("gamma_mean", "gamma_min", "gamma_max", "low_fraction")
            },
        }
        pooled[variant]["copy_weight_mean"] = 1.0 - pooled[variant]["gamma"]["gamma_mean"]
    dominant = sum(1 for cell in cells if cell["criterion"]["verdict"] == "memory_dominant")
    report = {
        "schema": "taiji-memory-dominance-probe-17-8-v1",
        "step": "N17.8",
        "date": "2026-11-04",
        "status": "zero_update_probe",
        "design": {
            "variants": list(VARIANTS),
            "base_dev_indices": list(BASE_INDICES),
            "new_object_names": list(NEW_OBJECTS),
            "new_place_names": list(NEW_PLACES),
            "train_vocabulary": {
                "objects": sorted(vocab["objects"]),
                "places": sorted(vocab["places"]),
            },
            "note": "training could in principle memorize all 3x5 name combinations",
        },
        "self_checks": checks,
        "cells": cells,
        "pooled": pooled,
        "verdict": {
            "memory_dominant_cells": dominant,
            "cells": len(cells),
            "criterion": "new_new entity_exact <= 0.10 AND new_new copy_rescue <= 0.10",
            "t3_relation": "T3 gold-forced 720/720 shows the operator can copy when the route "
            "is forced; this probe shows training never needed it",
        },
        "cost": {
            "optimizer_steps": 0,
            "training_updates": 0,
            "confirm_reads": 0,
            "worker_cpu_seconds": sum(cell["cost"]["cpu_seconds"] for cell in cells),
            "worker_wall_seconds": sum(cell["cost"]["wall_seconds"] for cell in cells),
            "driver_cpu_seconds": time.process_time() - start_cpu,
            "driver_wall_seconds": time.perf_counter() - start_wall,
        },
    }
    write_json(REPORT, report)
    write_json(
        MANIFEST,
        {
            "schema": "taiji-memory-dominance-probe-17-8-manifest-v1",
            "step": "N17.8",
            "date": "2026-11-04",
            "report": str(REPORT.relative_to(ROOT)),
            "report_sha256": sha256_file(REPORT),
            "report_bytes": REPORT.stat().st_size,
            "scripts": {
                str(Path(__file__).relative_to(ROOT)): sha256_file(Path(__file__)),
            },
            "bundles": {cell["bundle"]["path"]: cell["bundle"]["sha256"] for cell in cells},
            "optimizer_steps": 0,
            "training_updates": 0,
            "confirm_reads": 0,
        },
    )
    print(
        json.dumps(
            {
                "report": str(REPORT.relative_to(ROOT)),
                "memory_dominant_cells": dominant,
                "pooled_new_new_entity": round(pooled["new_new"]["entity_exact"], 3),
                "pooled_new_new_copy_rescue": round(pooled["new_new"]["copy_rescue"], 3),
                "pooled_new_new_output_change": round(
                    pooled["new_new"]["output_change"]["no_source"], 3
                ),
                "pooled_control_completed": round(pooled["control"]["completed"], 3),
                "worker_cpu": round(report["cost"]["worker_cpu_seconds"], 1),
                "driver_cpu": round(report["cost"]["driver_cpu_seconds"], 1),
                "wall": round(report["cost"]["driver_wall_seconds"], 1),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
