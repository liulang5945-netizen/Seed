"""22.4.2.6 generator-entry verification: shape, schema, rights and recovery.

Runs one miniature full-family sample (seed 124) plus a cheap A/B/C seed sweep.
No model forward, no optimizer, no project data. Each family is generated, saved
to a capsule, loaded, rebuilt and probed through the physical executor.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from taiji.u2_history import assert_values_equal
from taiji.u2_task_contract import allow_feedback
from taiji.u2_task_executor import PhysicalExecutor
from taiji.u2_task_generator import (
    FAMILIES,
    FEEDBACK_RIGHTS,
    _zero_action,
    generate,
    load_capsule,
    rebuild,
    save_capsule,
)
from taiji.u2_task_world import PhysicalWorld

REPORT = ROOT / "reports/taiji_u2_generator_22_4_2_20261117.json"
FULL_SEED = 124
SWEEP_SEEDS = (124, 125, 126)
CPU_CAP = 120.0
WALL_CAP = 240.0
OBSERVATION_SHAPES = {"rgb": [16, 48], "audio": [7, 33], "proprio": [1, 16]}
COORDINATE_SHAPES = {"rgb": [16, 8], "audio": [7, 8], "proprio": [1, 8]}
OUTCOME_SHAPE = [6]
ENVELOPE_KEYS = {
    "schema",
    "family",
    "seed",
    "world",
    "executor",
    "model_observation",
    "legal_content",
    "feedback_rights",
    "task",
    "provenance",
    "envelope_digest",
}
LEGAL_KEYS = {
    "goal_raw",
    "requests",
    "model_inputs",
    "observations",
    "coordinates",
    "outcome",
    "action",
}


def _check_shape(envelope: dict) -> list[str]:
    problems = []
    observation = envelope["model_observation"]
    for key, shape in OBSERVATION_SHAPES.items():
        if observation["observations"][key]["shape"] != shape:
            problems.append(f"observation_shape:{key}")
    for key, shape in COORDINATE_SHAPES.items():
        if observation["coordinates"][key]["shape"] != shape:
            problems.append(f"coordinate_shape:{key}")
    if observation["outcome"]["shape"] != OUTCOME_SHAPE:
        problems.append("outcome_shape")
    dtypes = {
        value["dtype"].removeprefix("torch.") for value in observation["observations"].values()
    } | {value["dtype"].removeprefix("torch.") for value in observation["coordinates"].values()}
    if len(dtypes) != 1 or next(iter(dtypes)) not in {"float32", "float64"}:
        problems.append("inconsistent_observation_dtype")
    return problems


def _check_schema(envelope: dict) -> list[str]:
    problems = []
    if set(envelope) != ENVELOPE_KEYS:
        problems.append("envelope_keys")
    if set(envelope["legal_content"]) != LEGAL_KEYS:
        problems.append("legal_keys")
    if set(envelope["world"]) != {
        "world_id",
        "version",
        "tick_index",
        "body_count",
        "modalities",
        "snapshot_sha256",
        "initial_world_sha256",
    }:
        problems.append("world_keys")
    if set(envelope["executor"]) != {"schema", "effect_count", "persistent"}:
        problems.append("executor_keys")
    if envelope["feedback_rights"] != FEEDBACK_RIGHTS:
        problems.append("feedback_rights")
    return problems


def _check_no_leak(sample) -> list[str]:
    problems = []
    legal = json.dumps(sample.envelope["legal_content"], ensure_ascii=False)
    if sample.case is not None:
        target = sample.case.target.decode()
        if target in sample.case.goal_raw.decode():
            problems.append("target_in_goal")
        if target in sample.case.request.raw.decode():
            problems.append("target_in_request")
        if target in legal:
            problems.append("target_in_legal_content")
    else:
        for check in sample.task.get("checks", []):
            expected = check["expected"]
            if expected in legal:
                problems.append(f"expected_in_legal_content:{expected}")
    return problems


def _check_rights() -> list[str]:
    problems = []
    try:
        allow_feedback(
            "observation",
            split="train",
            phase="perception",
            prediction_submitted=False,
            round_index=0,
            release_round=0,
        )
    except ValueError:
        problems.append("observation_right_rejected")
    for kind, split, phase, submitted in (
        ("train_label", "check", "development", False),
        ("score_only", "train", "perception", False),
        ("online_correction", "train", "online", False),
    ):
        try:
            allow_feedback(
                kind,
                split=split,
                phase=phase,
                prediction_submitted=submitted,
                round_index=0,
                release_round=0,
            )
        except ValueError:
            continue
        problems.append(f"right_not_rejected:{kind}")
    try:
        allow_feedback(
            "help_reply",
            split="train",
            phase="online",
            prediction_submitted=True,
            round_index=0,
            release_round=0,
        )
    except ValueError:
        problems.append("help_reply_rejected")
    return problems


def _executor_probe(sample) -> dict:
    started_cpu, started_wall = time.process_time(), time.monotonic()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "executor.bin"
        executor = PhysicalExecutor.create(PhysicalWorld.restore(sample.initial_world), path)
        ack = executor.execute("generator-probe", _zero_action())
        snapshot = executor.snapshot()
        restored = PhysicalExecutor.from_snapshot(snapshot)
        again = restored.snapshot()
        assert_values_equal(snapshot, again, "generator.executor_probe")
        query = restored.query("generator-probe")
        payload = {
            "schema": snapshot["schema"],
            "effect_count": snapshot["world"]["version"],
            "restore_equal": True,
            "query_status": query["status"],
            "world_keys": sorted(snapshot["world"]),
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 4),
            "ack": ack["status"],
        }
    return payload


def _run_family(family: str, seed: int) -> dict:
    sample = generate(family, seed)
    problems = []
    problems += _check_shape(sample.envelope)
    problems += _check_schema(sample.envelope)
    problems += _check_no_leak(sample)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.capsule"
        save_cpu, save_wall = time.process_time(), time.monotonic()
        size = save_capsule(sample, path)
        save_cost = {
            "cpu": round(time.process_time() - save_cpu, 4),
            "wall": round(time.monotonic() - save_wall, 4),
        }
        load_cpu, load_wall = time.process_time(), time.monotonic()
        loaded = load_capsule(path)
        load_cost = {
            "cpu": round(time.process_time() - load_cpu, 4),
            "wall": round(time.monotonic() - load_wall, 4),
        }
        if loaded["envelope_digest"] != sample.envelope["envelope_digest"]:
            problems.append("capsule_digest_mismatch")
    rebuild_cpu, rebuild_wall = time.process_time(), time.monotonic()
    rebuilt = rebuild(sample)
    rebuild_cost = {
        "cpu": round(time.process_time() - rebuild_cpu, 4),
        "wall": round(time.monotonic() - rebuild_wall, 4),
    }
    if not rebuilt["equal"]:
        problems.append("rebuild_not_equal")
    executor = _executor_probe(sample)
    if executor["schema"] != "taiji-u2-physical-executor-v1":
        problems.append("executor_schema")
    if executor["query_status"] != "applied" or executor["ack"] != "applied":
        problems.append("executor_query_status")
    return {
        "family": family,
        "seed": seed,
        "envelope_digest": sample.envelope["envelope_digest"],
        "problems": problems,
        "capsule_bytes": size,
        "generation_cpu": sample.costs["generation_cpu"],
        "encoding_cpu": sample.costs.get("encoding_cpu", 0.0),
        "save": save_cost,
        "load": load_cost,
        "rebuild": rebuild_cost,
        "executor": executor,
        "shapes": {
            "observations": {
                key: value["shape"]
                for key, value in sample.envelope["model_observation"]["observations"].items()
            },
            "coordinates": {
                key: value["shape"]
                for key, value in sample.envelope["model_observation"]["coordinates"].items()
            },
            "outcome": sample.envelope["model_observation"]["outcome"]["shape"],
        },
    }


def main() -> int:
    started_cpu, started_wall = time.process_time(), time.monotonic()
    results = {}
    for family in FAMILIES:
        results[family] = _run_family(family, FULL_SEED)
    sweep = {}
    for family in ("A", "B", "C"):
        sweep[family] = [_run_family(family, seed) for seed in SWEEP_SEEDS]
    rights = _check_rights()
    all_problems = []
    for row in results.values():
        all_problems.extend(f"{row['family']}:{problem}" for problem in row["problems"])
    for rows in sweep.values():
        for row in rows:
            all_problems.extend(f"{row['family']}:{problem}" for problem in row["problems"])
    all_problems.extend(f"rights:{problem}" for problem in rights)
    total_cpu = round(time.process_time() - started_cpu, 4)
    total_wall = round(time.monotonic() - started_wall, 4)
    if total_cpu > CPU_CAP or total_wall > WALL_CAP:
        verdict = "budget_stop"
    elif all_problems:
        verdict = "fixture_invalid"
    else:
        verdict = "generator_entry_valid"
    report = {
        "schema": "taiji-u2-22-4-2-generator-entry-v1",
        "card": "22.4.2.6",
        "phase": "zero_training_generator_entry",
        "verdict": verdict,
        "full_seed": FULL_SEED,
        "sweep_seeds": list(SWEEP_SEEDS),
        "families": results,
        "sweep": sweep,
        "feedback_rights": FEEDBACK_RIGHTS,
        "rights_problems": rights,
        "problems": all_problems,
        "discipline": {
            "model_forwards": 0,
            "optimizer_steps": 0,
            "project_data_reads": 0,
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {
            "cpu": total_cpu,
            "wall": total_wall,
            "cpu_cap": CPU_CAP,
            "wall_cap": WALL_CAP,
        },
    }
    if REPORT.exists():
        raise FileExistsError(str(REPORT))
    REPORT.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    print(
        "generator verdict="
        + verdict
        + " families="
        + str(len(results))
        + " sweep="
        + str(len(sweep))
        + " problems="
        + str(len(all_problems))
        + " cpu="
        + str(total_cpu)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
