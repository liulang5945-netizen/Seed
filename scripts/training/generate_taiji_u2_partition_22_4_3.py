"""22.4.3 world partition, data recipe, check exposure and final seal.

The partition is a seed-level recipe: 1,280 world groups, each hosting A/B/C/D/J.
Train/fit/check content is regenerable from the 22.4.2.6 generator entry; this
card generates a small train/fit sample receipt, does not generate or read check
content, and reserves final128 under R-SEAL-1 without generating it.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from taiji.u2_task_generator import (
    FAMILIES,
    FEEDBACK_RIGHTS,
    generate,
    load_capsule,
    rebuild,
    save_capsule,
    source_pins,
)

REPORT = ROOT / "reports/taiji_u2_partition_22_4_3_20261117.json"
WORLD_SEED_BASE = 20261110
WORLD_SEED_STRIDE = 1009
SPLIT_ORDER = ("train", "development-fit", "development-check", "final")
COUNTS = {
    "train": 768,
    "development-fit": 192,
    "development-check": 192,
    "final": 128,
}
START = {
    "train": 0,
    "development-fit": 768,
    "development-check": 960,
    "final": 1152,
}
BIRTH_FIT_COUNT = 128
BIRTH_CHECK_COUNT = 64
CPU_CAP = 200.0
WALL_CAP = 400.0
OBSERVATION_SHAPES = {"rgb": [16, 48], "audio": [7, 33], "proprio": [1, 16]}
COORDINATE_SHAPES = {"rgb": [16, 8], "audio": [7, 8], "proprio": [1, 8]}
OUTCOME_SHAPE = [6]
CHECK_ALLOCATIONS = {"22.5": 2, "22.6": 2, "22.7": 3, "22.8": 1}


def seed_at(index: int) -> int:
    return WORLD_SEED_BASE + WORLD_SEED_STRIDE * index


def split_seeds(name: str) -> list[int]:
    start = START[name]
    return [seed_at(index) for index in range(start, start + COUNTS[name])]


def seeds_digest(seeds: list[int]) -> str:
    body = json.dumps(seeds, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def _shape_ok(envelope: dict) -> bool:
    observation = envelope["model_observation"]
    return (
        all(
            observation["observations"][key]["shape"] == shape
            for key, shape in OBSERVATION_SHAPES.items()
        )
        and all(
            observation["coordinates"][key]["shape"] == shape
            for key, shape in COORDINATE_SHAPES.items()
        )
        and observation["outcome"]["shape"] == OUTCOME_SHAPE
    )


def build_manifest() -> dict:
    splits = {}
    for name in SPLIT_ORDER:
        seeds = split_seeds(name)
        entry = {
            "count": len(seeds),
            "first": seeds[0],
            "last": seeds[-1],
            "seeds_sha256": seeds_digest(seeds),
        }
        if name != "final":
            entry["seeds"] = seeds
        splits[name] = entry
    fit = splits["development-fit"]["seeds"]
    birth_fit = fit[:BIRTH_FIT_COUNT]
    birth_check = fit[BIRTH_FIT_COUNT:]
    data_recipe = {
        "schema": "taiji-u2-data-recipe-v1",
        "generator": "taiji/u2_task_generator.py:generate",
        "families": list(FAMILIES),
        "seed_derivation": {
            "base": WORLD_SEED_BASE,
            "stride": WORLD_SEED_STRIDE,
            "split_order": list(SPLIT_ORDER),
            "counts": dict(COUNTS),
            "starts": dict(START),
        },
        "world_group": {
            "unit": "seed",
            "families": list(FAMILIES),
            "cross_split": False,
        },
        "shape_schema": {
            "observations": OBSERVATION_SHAPES,
            "coordinates": COORDINATE_SHAPES,
            "outcome": OUTCOME_SHAPE,
        },
        "feedback_rights": FEEDBACK_RIGHTS,
        "stale_fact_policy": "exclude_superseded_versions",
        "ability_A": {
            "independent_surface": True,
            "fit_worlds": COUNTS["development-fit"],
            "check_worlds": COUNTS["development-check"],
            "final_worlds": COUNTS["final"],
        },
        "source_pins": {
            **source_pins(),
            "generate_taiji_u2_partition_22_4_3.py": hashlib.sha256(
                Path(__file__).read_text(encoding="utf-8-sig").encode()
            ).hexdigest(),
        },
    }
    data_recipe_digest = hashlib.sha256(
        json.dumps(data_recipe, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    final_entry = splits["final"]
    final_seal = {
        "authorization": "R-SEAL-1",
        "status": "reserved_not_generated",
        "count": COUNTS["final"],
        "first": final_entry["first"],
        "last": final_entry["last"],
        "seeds_sha256": final_entry["seeds_sha256"],
        "recipe_digest": data_recipe_digest,
        "generated": False,
        "read": False,
    }
    check_exposure = {
        "max": 8,
        "used": 0,
        "remaining": 8,
        "allocations": dict(CHECK_ALLOCATIONS),
        "exposure_unit": "complete candidate x 3 fixed seeds x all stage arms",
        "irrevocable": True,
        "first_access_consumes": True,
        "aggregate_only": True,
        "labels_to_selector": False,
        "check_content_generated": False,
        "check_labels_read": False,
        "global_check_used_for_birth": False,
    }
    partition = {
        "schema": "taiji-u2-partition-v1",
        "data_recipe_digest": data_recipe_digest,
        "splits": {
            name: {
                "count": entry["count"],
                "seeds_sha256": entry["seeds_sha256"],
            }
            for name, entry in splits.items()
        },
        "birth": {
            "fit_count": len(birth_fit),
            "check_count": len(birth_check),
            "fit_seeds_sha256": seeds_digest(birth_fit),
            "check_seeds_sha256": seeds_digest(birth_check),
            "global_check_used": False,
        },
        "world_group": data_recipe["world_group"],
        "check_exposure": check_exposure,
        "final_seal": final_seal,
    }
    partition_digest = hashlib.sha256(
        json.dumps(partition, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "taiji-u2-partition-manifest-v1",
        "planned_worlds": sum(COUNTS.values()),
        "recipe_worlds": sum(COUNTS[name] for name in SPLIT_ORDER if name != "final"),
        "splits": splits,
        "birth": {
            "fit": {
                "count": len(birth_fit),
                "seeds": birth_fit,
                "seeds_sha256": seeds_digest(birth_fit),
            },
            "check": {
                "count": len(birth_check),
                "seeds": birth_check,
                "seeds_sha256": seeds_digest(birth_check),
            },
            "global_check_used": False,
        },
        "data_recipe": data_recipe,
        "data_recipe_digest": data_recipe_digest,
        "check_exposure": check_exposure,
        "final_seal": final_seal,
        "partition": partition,
        "partition_digest": partition_digest,
    }


def _receipt(
    split: str, index: int, family: str, seed: int, sample, *, save: bool, rebuild_sample: bool
) -> dict:
    entry = {
        "split": split,
        "world_index": index,
        "family": family,
        "seed": seed,
        "envelope_digest": sample.envelope["envelope_digest"],
        "shape_ok": _shape_ok(sample.envelope),
        "generation_cpu": sample.costs["generation_cpu"],
        "encoding_cpu": sample.costs.get("encoding_cpu", 0.0),
    }
    if save:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.capsule"
            size = save_capsule(sample, path)
            loaded = load_capsule(path)
            entry["capsule_bytes"] = size
            entry["load_equal"] = loaded["envelope_digest"] == sample.envelope["envelope_digest"]
    if rebuild_sample:
        started = time.process_time()
        rebuilt = rebuild(sample)
        entry["rebuild_equal"] = rebuilt["equal"]
        entry["rebuild_cpu"] = round(time.process_time() - started, 4)
    return entry


def sample_receipts(manifest: dict) -> list[dict]:
    train_seeds = manifest["splits"]["train"]["seeds"]
    fit_seeds = manifest["splits"]["development-fit"]["seeds"]
    receipts = []
    train_a = generate("A", train_seeds[0])
    receipts.append(
        _receipt("train", 0, "A", train_seeds[0], train_a, save=True, rebuild_sample=True)
    )
    train_d = generate("D", train_seeds[0])
    receipts.append(
        _receipt("train", 0, "D", train_seeds[0], train_d, save=False, rebuild_sample=True)
    )
    train_a2 = generate("A", train_seeds[1])
    receipts.append(
        _receipt("train", 1, "A", train_seeds[1], train_a2, save=False, rebuild_sample=False)
    )
    fit_a = generate("A", fit_seeds[0])
    receipts.append(
        _receipt("development-fit", 0, "A", fit_seeds[0], fit_a, save=True, rebuild_sample=False)
    )
    fit_j = generate("J", fit_seeds[0])
    receipts.append(
        _receipt("development-fit", 0, "J", fit_seeds[0], fit_j, save=False, rebuild_sample=False)
    )
    return receipts


def _problems(manifest: dict, receipts: list[dict]) -> list[str]:
    problems = []
    if sum(COUNTS.values()) != 1280:
        problems.append("planned_world_count")
    seen = set()
    for name in SPLIT_ORDER:
        seeds = split_seeds(name)
        if seeds != manifest["splits"][name].get("seeds", seeds):
            problems.append(f"seed_order:{name}")
        if len(seeds) != COUNTS[name] or len(set(seeds)) != len(seeds):
            problems.append(f"seed_count_or_duplicate:{name}")
        if seen & set(seeds):
            problems.append(f"cross_split_overlap:{name}")
        seen |= set(seeds)
    if len(seen) != 1280:
        problems.append("union_count")
    fit = manifest["splits"]["development-fit"]["seeds"]
    birth_fit = set(manifest["birth"]["fit"]["seeds"])
    birth_check = set(manifest["birth"]["check"]["seeds"])
    if birth_fit | birth_check != set(fit) or birth_fit & birth_check:
        problems.append("birth_subsplit")
    if len(birth_fit) != BIRTH_FIT_COUNT or len(birth_check) != BIRTH_CHECK_COUNT:
        problems.append("birth_counts")
    if sum(CHECK_ALLOCATIONS.values()) != 8:
        problems.append("check_allocation_sum")
    if manifest["check_exposure"]["used"] != 0:
        problems.append("check_exposure_started")
    if (
        manifest["check_exposure"]["check_content_generated"]
        or manifest["check_exposure"]["check_labels_read"]
    ):
        problems.append("check_content_or_labels_used")
    if manifest["final_seal"]["generated"] or manifest["final_seal"]["read"]:
        problems.append("final_generated_or_read")
    if manifest["final_seal"]["authorization"] != "R-SEAL-1":
        problems.append("final_authorization")
    for receipt in receipts:
        if not receipt["shape_ok"]:
            problems.append(f"shape:{receipt['family']}")
        if receipt.get("load_equal") is False:
            problems.append(f"load:{receipt['family']}")
        if receipt.get("rebuild_equal") is False:
            problems.append(f"rebuild:{receipt['family']}")
    return problems


def main() -> int:
    started_cpu, started_wall = time.process_time(), time.monotonic()
    manifest = build_manifest()
    receipts = sample_receipts(manifest)
    problems = _problems(manifest, receipts)
    cpu = round(time.process_time() - started_cpu, 4)
    wall = round(time.monotonic() - started_wall, 4)
    if cpu > CPU_CAP or wall > WALL_CAP:
        verdict = "budget_stop"
    elif problems:
        verdict = "fixture_invalid"
    else:
        verdict = "partition_valid"
    report = {
        "schema": "taiji-u2-22-4-3-partition-v1",
        "card": "22.4.3",
        "phase": "partition_recipe_and_exposure",
        "verdict": verdict,
        "manifest": manifest,
        "sample_receipts": receipts,
        "problems": problems,
        "materialization": {
            "planned_worlds": manifest["planned_worlds"],
            "recipe_worlds": manifest["recipe_worlds"],
            "sample_worlds": 3,
            "sample_families": ["A", "D", "J"],
            "full_materialization": False,
            "regenerable": True,
            "regenerator": "taiji/u2_task_generator.py:generate",
            "check_content_generated": False,
            "final_generated": False,
        },
        "discipline": {
            "model_forwards": 0,
            "optimizer_steps": 0,
            "project_data_reads": 0,
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {"cpu": cpu, "wall": wall, "cpu_cap": CPU_CAP, "wall_cap": WALL_CAP},
    }
    if REPORT.exists():
        raise FileExistsError(str(REPORT))
    REPORT.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    print(
        "partition verdict="
        + verdict
        + " planned="
        + str(manifest["planned_worlds"])
        + " recipe="
        + str(manifest["recipe_worlds"])
        + " problems="
        + str(len(problems))
        + " cpu="
        + str(cpu)
        + " digest="
        + manifest["partition_digest"][:16]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
