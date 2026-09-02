"""Diagnose whether shared-memory capacity is the B5 bottleneck."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_b5_memory import build_corpus  # noqa: E402
from scripts.training.train_taiji_memory import _memory_config  # noqa: E402
from taiji import ContinualMemoryTask, Taiji, TaijiConfig  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-33-memory-capacity-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_33_memory_capacity_20260902.json"
CAPACITIES = (96, 192, 384)
SEEDS = (11, 29, 47)


def _config(seed: int, capacity: int) -> TaijiConfig:
    values = _memory_config(seed).to_dict()
    values["memory_units"] = int(capacity)
    return TaijiConfig.from_dict(values)


def _fresh_process_digest(checkpoint: dict[str, Any]) -> str:
    code = """
import io
import sys
import torch
from taiji import Taiji
from taiji.internalization import content_digest

payload = torch.load(io.BytesIO(sys.stdin.buffer.read()), map_location="cpu", weights_only=False)
model = Taiji.from_checkpoint(payload)
print(content_digest(model.checkpoint()))
"""
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    process = subprocess.run(
        (sys.executable, "-c", code),
        cwd=PROJECT_ROOT,
        input=buffer.getvalue(),
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.decode(errors="replace")[-2000:])
    return process.stdout.decode().strip().splitlines()[-1]


def _seed_metrics(measurement: Any) -> list[dict[str, Any]]:
    raw = next(
        item for item in measurement.evidence if item.startswith("seed_metrics=")
    )
    value = json.loads(raw.split("=", 1)[1])
    if not isinstance(value, list):
        raise ValueError("B5 evidence seed_metrics must be a list")
    return [dict(item) for item in value]


def _capacity_record(capacity: int, corpus: Any) -> dict[str, Any]:
    started = time.perf_counter()
    config = _config(SEEDS[0], capacity)
    measurement = ContinualMemoryTask(
        config,
        seeds=SEEDS,
        replay_learning_targets="all",
    ).evaluate(corpus)
    elapsed = time.perf_counter() - started
    seed_metrics = _seed_metrics(measurement)
    checkpoint_records = []
    for seed in SEEDS:
        model = Taiji(_config(seed, capacity), episode_id=f"m1-33-checkpoint-{capacity}-{seed}")
        checkpoint = deepcopy(model.checkpoint())
        digest = content_digest(checkpoint)
        restored = Taiji.from_checkpoint(deepcopy(checkpoint))
        same_process = content_digest(restored.checkpoint()) == digest
        fresh_digest = _fresh_process_digest(checkpoint)
        checkpoint_records.append(
            {
                "seed": seed,
                "checkpoint_digest": digest,
                "same_process_digest_matches": same_process,
                "fresh_process_digest_matches": fresh_digest == digest,
            }
        )
    model = Taiji(config, episode_id=f"m1-33-budget-{capacity}")
    budget = {
        "memory_units": capacity,
        "identity_organ_enabled": config.identity_organ_enabled,
        "active_parameter_count": model.parameter_count(),
        "planned_active_parameter_count": config.planned_active_parameter_count,
        "parameter_count_matches_plan": (
            model.parameter_count() == config.planned_active_parameter_count
        ),
    }
    return {
        "memory_units": capacity,
        "status": measurement.status,
        "metric_value": measurement.metric_value,
        "baseline_metrics": dict(measurement.baseline_metrics),
        "sample_counts": dict(measurement.sample_counts),
        "holdout_updates": measurement.holdout_updates,
        "seed_metrics": seed_metrics,
        "budget": budget,
        "checkpoint": {
            "records": checkpoint_records,
            "all_same_process_match": all(
                item["same_process_digest_matches"] for item in checkpoint_records
            ),
            "all_fresh_process_match": all(
                item["fresh_process_digest_matches"] for item in checkpoint_records
            ),
        },
        "cpu_seconds": round(elapsed, 3),
    }


def _capacity_passed(record: dict[str, Any]) -> bool:
    return bool(
        record["status"] == "passed"
        and record["holdout_updates"] == 0
        and record["budget"]["identity_organ_enabled"] is False
        and record["budget"]["parameter_count_matches_plan"]
        and record["checkpoint"]["all_same_process_match"]
        and record["checkpoint"]["all_fresh_process_match"]
    )


def run_diagnosis() -> dict[str, Any]:
    corpus = build_corpus(train_count=64, holdout_count=32, retention_count=32)
    records = [_capacity_record(capacity, corpus) for capacity in CAPACITIES]
    for record in records:
        record["capacity_gate_passed"] = _capacity_passed(record)
    highest = records[-1]
    return {
        "format": FORMAT,
        "version": 1,
        "status": "passed" if any(item["capacity_gate_passed"] for item in records) else "failed",
        "variable_changed": "shared memory_units only",
        "identity_organ_enabled": False,
        "learning_rule_changed": False,
        "decoder_changed": False,
        "corpus_digest": corpus.digest,
        "capacities": list(CAPACITIES),
        "seeds": list(SEEDS),
        "records": records,
        "conclusion": {
            "highest_capacity": highest["memory_units"],
            "highest_capacity_b5_status": highest["status"],
            "capacity_alone_sufficient": any(
                item["capacity_gate_passed"] for item in records
            ),
            "next_boundary": (
                "capacity expansion alone did not pass B5; diagnose cue/action"
                " training signal or data curriculum"
                if not any(item["capacity_gate_passed"] for item in records)
                else "a capacity tier passed B5; continue with a held-out scale check"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_diagnosis()
    result["report_path"] = str(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
