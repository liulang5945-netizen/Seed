"""Offline trajectory and parent-error attribution for PLAN-N1-117."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_REPORT = ROOT / "reports/taiji_n1_length_balanced_training_20261018.json"
COUPLED_REPORT = ROOT / "reports/taiji_n1_coupled_rwl_length_balanced_r1_20261018.json"
OUTPUT = ROOT / "reports/taiji_n1_coupled_rwl_trajectory_attribution_20261018.json"
EXPECTED_REPORTS = {
    BASE_REPORT: "e848433de597d378e9e1dc52bf488f296f90c5ce67f3855e19bcfe49268aaa0c",
    COUPLED_REPORT: "1d4f43519790a1e36cb7dfbc580506d697815dccbdecb95fd576ec3a7bc2e391",
}
GROUPS = ("reader", "writer", "local", "context", "relation")
RWL = ("reader", "writer", "local")
SEEDS = (11, 12, 13)
UPDATES = 24_576
CYCLE = 6_144


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def distribution(values: list[float]) -> dict:
    if not values:
        raise ValueError("nonempty scale/loss series required")
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "p10": ordered[max(0, math.ceil(0.10 * len(ordered)) - 1)],
        "p90": ordered[max(0, math.ceil(0.90 * len(ordered)) - 1)],
        "max": ordered[-1],
        "clipped": sum(value < 1.0 for value in ordered),
    }


def summarize_trajectory(record: dict, *, coupled: bool) -> dict:
    measurements = record.get("measurements", [])
    losses = record.get("losses", [])
    if len(measurements) != UPDATES or len(losses) != UPDATES:
        raise ValueError("exact 24,576 measurement/loss rows required")
    result = {}
    for cycle in range(4):
        rows = [row for row in measurements if row.get("cycle") == cycle]
        if len(rows) != CYCLE or any(row.get("step") // CYCLE != cycle for row in rows):
            raise ValueError("cycle/update binding mismatch")
        cycle_result = {}
        if coupled:
            rwl_scales = []
            for row in rows:
                groups = row["groups"]
                values = [groups[group].get("shared_block_scale") for group in RWL]
                if any(value is None for value in values) or max(values) - min(values) > 1e-12:
                    raise ValueError("coupled RWL groups did not use one identical shared scale")
                rwl_scales.append(values[0])
            cycle_result["shared_rwl"] = distribution(rwl_scales)
            for group in ("context", "relation"):
                cycle_result[group] = distribution([row["groups"][group]["scale"] for row in rows])
        else:
            for group in GROUPS:
                cycle_result[group] = distribution([row["groups"][group]["scale"] for row in rows])
        cycle_losses = losses[cycle * CYCLE : (cycle + 1) * CYCLE]
        cycle_result["loss"] = {
            key: statistics.fmean(float(row[key]) for row in cycle_losses)
            for key in ("qa", "observation", "total")
        }
        result[str(cycle)] = cycle_result
    return result


def final_parent_joint(shard_record: dict, evaluation_set: str) -> set[str]:
    rows = [
        row
        for row in shard_record.get("rows", [])
        if row.get("completed_updates") == UPDATES
        and row.get("evaluation_set") == evaluation_set
        and row.get("kind") == "transition"
    ]
    if len(rows) != 192:
        raise ValueError(f"{evaluation_set} requires192 final transition rows")
    grouped: dict[str, list[bool]] = {}
    for row in rows:
        grouped.setdefault(row["id"], []).append(row["free_exact"] is True)
    if len(grouped) != 48 or any(len(values) != 4 for values in grouped.values()):
        raise ValueError(f"{evaluation_set} must contain four conditions for48 parents")
    return {parent for parent, values in grouped.items() if all(values)}


def load_model_shard(report: dict, seed: int) -> tuple[dict, dict]:
    selected = [
        model
        for model in report.get("models", [])
        if (model.get("seed"), model.get("arm")) == (seed, "position")
    ]
    if len(selected) != 1 or selected[0].get("complete") is not True:
        raise ValueError(f"one complete position model required for seed{seed}")
    descriptor = selected[0]["shard"]
    path = Path(descriptor["path"])
    compressed = path.read_bytes()
    if (
        len(compressed) != descriptor["compressed_bytes"]
        or sha256_bytes(compressed) != descriptor["compressed_sha256"]
    ):
        raise ValueError(f"compressed source shard mismatch: {path}")
    payload_bytes = gzip.decompress(compressed)
    if (
        len(payload_bytes) != descriptor["uncompressed_bytes"]
        or sha256_bytes(payload_bytes) != descriptor["uncompressed_sha256"]
    ):
        raise ValueError(f"uncompressed source shard mismatch: {path}")
    shard = json.loads(payload_bytes)
    record = shard.get("record", {})
    if (
        shard.get("schema") != "n1-lossless-training-evidence-shard-v1"
        or len(record.get("measurements", [])) != UPDATES
        or len(record.get("losses", [])) != UPDATES
        or len(record.get("rows", [])) != 4_800
    ):
        raise ValueError(f"sealed source shard schema/count mismatch: {path}")
    return selected[0], shard


def compare_parent_sets(old: dict[str, set[str]], new: dict[str, set[str]]) -> dict:
    result = {}
    for view in ("original", "rename"):
        before, after = old[view], new[view]
        result[view] = {
            "n1_110_joint_four": len(before),
            "n1_117_joint_four": len(after),
            "unchanged_correct": len(before & after),
            "rescued_by_n1_117": sorted(after - before),
            "lost_from_n1_110": sorted(before - after),
            "both_wrong": 48 - len(before | after),
        }
    return result


def analyze() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"write-once attribution report exists: {OUTPUT}")
    reports = {}
    for path, expected in EXPECTED_REPORTS.items():
        if sha256_file(path) != expected:
            raise ValueError(f"sealed training report SHA mismatch: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("complete") is not True
            or value.get("completed_updates") != 9 * UPDATES
            or value.get("schedule_sha256")
            != "b2d637fcc1e096d40bba35bbc91bc66b0c27257cd74a4bcac5f5f488f55b6d4c"
            or any(value.get(k) for k in ("dev_queries", "holdout_queries", "confirmation_queries"))
        ):
            raise ValueError(f"complete matched train-only report required: {path}")
        reports[path] = value
    old_report, new_report = reports[BASE_REPORT], reports[COUPLED_REPORT]
    started = time.process_time()
    results = {}
    parent_comparison = None
    shard_sources = {}
    for seed in SEEDS:
        old_model, old_shard = load_model_shard(old_report, seed)
        old_record = old_shard["record"]
        old_trajectory = summarize_trajectory(old_record, coupled=False)
        old_final_milestone = old_model["milestones"][-1]
        old_parent_sets = None
        if seed == 13:
            old_parent_sets = {
                view: final_parent_joint(old_record, f"factorial_{view}")
                for view in ("original", "rename")
            }
        shard_sources[f"n1_110_seed{seed}"] = old_model["shard"]["compressed_sha256"]
        del old_shard, old_record

        new_model, new_shard = load_model_shard(new_report, seed)
        new_record = new_shard["record"]
        new_trajectory = summarize_trajectory(new_record, coupled=True)
        new_final_milestone = new_model["milestones"][-1]
        results[str(seed)] = {
            "n1_110": {
                "training_sufficient": old_model["training_sufficient"],
                "joint_four": {
                    view: old_final_milestone["factorial"][view]["joint_four_exact"]
                    for view in ("original", "rename")
                },
                "transition_rename": old_final_milestone["rename_free_by_kind"]["transition"],
                "group_scale_by_cycle": old_trajectory,
            },
            "n1_117": {
                "training_sufficient": new_model["training_sufficient"],
                "joint_four": {
                    view: new_final_milestone["factorial"][view]["joint_four_exact"]
                    for view in ("original", "rename")
                },
                "transition_rename": new_final_milestone["rename_free_by_kind"]["transition"],
                "group_scale_by_cycle": new_trajectory,
            },
        }
        shard_sources[f"n1_117_seed{seed}"] = new_model["shard"]["compressed_sha256"]
        if seed == 13:
            new_parent_sets = {
                view: final_parent_joint(new_record, f"factorial_{view}")
                for view in ("original", "rename")
            }
            for view in ("original", "rename"):
                if (
                    len(old_parent_sets[view])
                    != old_final_milestone["factorial"][view]["joint_four_exact"]
                ):
                    raise ValueError(f"N1-110 parent joint-four mismatch: {view}")
                if (
                    len(new_parent_sets[view])
                    != new_final_milestone["factorial"][view]["joint_four_exact"]
                ):
                    raise ValueError(f"N1-117 parent joint-four mismatch: {view}")
            parent_comparison = compare_parent_sets(old_parent_sets, new_parent_sets)
        del new_shard, new_record

    result = {
        "schema": "n1-coupled-rwl-trajectory-attribution-v1",
        "status": "complete",
        "scope": "offline analysis of sealed N1-110 and N1-117 train shards; descriptive only",
        "source_reports": {
            str(path.relative_to(ROOT)): digest for path, digest in EXPECTED_REPORTS.items()
        },
        "source_shards": shard_sources,
        "seed_position_trajectories": results,
        "seed13_position_parent_joint_four_comparison": parent_comparison,
        "source_shards_read": len(shard_sources),
        "checkpoint_payload_reads": 0,
        "model_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "checkpoint_saves": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "cpu_seconds": time.process_time() - started,
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        )
    return result


if __name__ == "__main__":
    result = analyze()
    print(
        json.dumps(
            {
                "status": result["status"],
                "report": str(OUTPUT.relative_to(ROOT)),
                "source_shards_read": result["source_shards_read"],
                "cpu_seconds": result["cpu_seconds"],
                "model_loads": result["model_loads"],
                "model_queries": result["model_queries"],
            },
            ensure_ascii=False,
        )
    )
