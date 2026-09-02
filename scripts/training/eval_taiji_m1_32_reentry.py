"""Aggregate the foundation-first CPU reentry evidence for M1-32."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_32_reentry_20260902.json"
SEEDS = (11, 29, 47)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report must contain an object: {path}")
    return payload


def _model_contract(path: Path) -> dict[str, bool]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = payload.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"training checkpoint is missing model payload: {path}")
    config = model.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"training checkpoint is missing model config: {path}")
    return {
        "identity_organ_enabled": bool(config.get("identity_organ_enabled", False)),
        "identity_payload_absent": "identity_organ" not in model,
    }


def _b1_record(seed: int) -> dict[str, Any]:
    training_path = PROJECT_ROOT / f"reports/taiji_m1_32_foundation_pilot_seed{seed}.json"
    eval_path = PROJECT_ROOT / (
        f"reports/taiji_m1_32_foundation_pilot_seed{seed}_eval_only.json"
    )
    training = _load(training_path)
    evaluated = _load(eval_path)
    checkpoint_paths = training["checkpoint_paths"]
    resolved_paths = {
        name: (PROJECT_ROOT / str(path).replace("\\", "/")).is_file()
        for name, path in checkpoint_paths.items()
    }
    contract = _model_contract(PROJECT_ROOT / str(checkpoint_paths["best_holdout"]).replace("\\", "/"))
    return {
        "seed": seed,
        "dataset_digest": training["dataset_digest"],
        "sample_counts": training["dataset_sample_counts"],
        "parent_holdout_bpb": training["parent_holdout_bpb"],
        "final_holdout_bpb": training["final_holdout_bpb"],
        "final_retention_bpb": training["final_retention_bpb"],
        "child_checkpoint_digest": training["child_checkpoint_digest"],
        "eval_checkpoint_digest": evaluated["checkpoint_digest"],
        "eval_holdout_bpb": evaluated["holdout_bpb"],
        "eval_retention_bpb": evaluated["retention_bpb"],
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_files_exist": resolved_paths,
        "checkpoint_read_only": bool(evaluated["checkpoint_read_only"]),
        "identity_contract": contract,
        "training_status": training["status"],
        "eval_status": evaluated["status"],
    }


def _b2_record(seed: int) -> dict[str, Any]:
    training_path = PROJECT_ROOT / f"reports/taiji_m1_32_memory_pilot_seed{seed}.json"
    eval_path = PROJECT_ROOT / f"reports/taiji_m1_32_memory_pilot_seed{seed}_eval_only.json"
    training = _load(training_path)
    evaluated = _load(eval_path)
    checkpoint_paths = training["checkpoint_paths"]
    resolved_paths = {
        name: (PROJECT_ROOT / str(path).replace("\\", "/")).is_file()
        for name, path in checkpoint_paths.items()
    }
    contract = _model_contract(PROJECT_ROOT / str(checkpoint_paths["best_holdout"]).replace("\\", "/"))
    return {
        "seed": seed,
        "corpus_digest": training["corpus_digest"],
        "sample_counts": training["corpus_sample_counts"],
        "parent_holdout_recall": training["parent_holdout_recall"],
        "best_holdout_recall": training["best_holdout_recall"],
        "final_holdout_recall": training["final_holdout_recall"],
        "final_retention_recall": training["final_retention_recall"],
        "final_memory_lesion_recall": training["final_memory_lesion_recall"],
        "child_checkpoint_digest": training["child_checkpoint_digest"],
        "eval_checkpoint_digest": evaluated["checkpoint_digest"],
        "eval_holdout_recall": evaluated["holdout_recall"],
        "eval_retention_recall": evaluated["retention_recall"],
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_files_exist": resolved_paths,
        "checkpoint_read_only": bool(evaluated["checkpoint_read_only"]),
        "identity_contract": contract,
        "training_status": training["status"],
        "eval_status": evaluated["status"],
    }


def _b5_summary() -> dict[str, Any]:
    report = _load(PROJECT_ROOT / "reports/taiji_m1_32_b5_pilot_20260902.json")
    evidence = report["measurement"]["evidence"]
    seed_metrics = json.loads(next(item for item in evidence if item.startswith("seed_metrics=")).split("=", 1)[1])
    return {
        "corpus_digest": report["corpus_digest"],
        "status": report["measurement"]["status"],
        "primary_metric": report["measurement"]["primary_metric"],
        "metric_value": report["measurement"]["metric_value"],
        "sample_counts": report["measurement"]["sample_counts"],
        "holdout_updates": report["measurement"]["holdout_updates"],
        "seed_metrics": seed_metrics,
    }


def run_reentry() -> dict[str, Any]:
    preflight = _load(PROJECT_ROOT / "reports/taiji_m1_32_checkpoint_preflight_20260902.json")
    b1 = [_b1_record(seed) for seed in SEEDS]
    b2 = [_b2_record(seed) for seed in SEEDS]
    b5 = _b5_summary()
    b1_passed = all(
        item["training_status"] == "completed"
        and item["eval_status"] == "evaluated"
        and item["checkpoint_read_only"]
        and all(item["checkpoint_files_exist"].values())
        and item["identity_contract"]["identity_organ_enabled"] is False
        and item["identity_contract"]["identity_payload_absent"]
        and item["final_holdout_bpb"] < item["parent_holdout_bpb"]
        and abs(item["final_holdout_bpb"] - item["eval_holdout_bpb"]) <= 1e-12
        and abs(item["final_retention_bpb"] - item["eval_retention_bpb"]) <= 1e-12
        for item in b1
    )
    b2_passed = all(
        item["training_status"] == "completed"
        and item["eval_status"] == "evaluated"
        and item["checkpoint_read_only"]
        and all(item["checkpoint_files_exist"].values())
        and item["identity_contract"]["identity_organ_enabled"] is False
        and item["identity_contract"]["identity_payload_absent"]
        and item["best_holdout_recall"] > item["parent_holdout_recall"]
        and item["best_holdout_recall"] > item["final_memory_lesion_recall"]
        and abs(item["best_holdout_recall"] - item["eval_holdout_recall"]) <= 1e-12
        and item["eval_retention_recall"] >= item["parent_holdout_recall"]
        for item in b2
    )
    checkpoint_passed = all(preflight["checks"].values())
    b5_passed = b5["status"] == "passed"
    return {
        "format": "taiji-native-m1-32-reentry-v1",
        "version": 1,
        "status": "blocked" if not (checkpoint_passed and b1_passed and b2_passed and b5_passed) else "passed",
        "identity_organ_enabled": False,
        "preflight": preflight,
        "b1": {"records": b1, "passed": b1_passed},
        "b2": {"records": b2, "passed": b2_passed},
        "b5": {"summary": b5, "passed": b5_passed},
        "gate": {
            "checkpoint_preflight_passed": checkpoint_passed,
            "b1_pilot_passed": b1_passed,
            "b2_pilot_passed": b2_passed,
            "b5_pilot_passed": b5_passed,
            "overall_passed": checkpoint_passed and b1_passed and b2_passed and b5_passed,
            "reason": (
                "B5 replay has causal gain but does not preserve old and new memory"
                " together across seeds; restrict the next change to shared-memory"
                " training signal/data curriculum."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_reentry()
    result["report_path"] = str(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
