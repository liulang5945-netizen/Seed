"""Preflight the bounded M4.R7 formal retention run without training.

The preflight validates inherited checkpoint loading, atomic save/fresh
restore, foundation-profile data capacity, record-disjointness, and a
bounded CPU-time estimate before any formal update is allowed to run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m2r1_phase_c_canary import (  # noqa: E402
    build_disjoint_phase_chain,
)
from scripts.training.eval_taiji_m4r3_update_interference_canary import (  # noqa: E402
    _load_checkpoint,
)
from seed.persistence import atomic_save  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

EXPECTED_SEEDS = (11, 29, 47)
EXPECTED_SCALES = (0.0, 0.5, 1.0)
FORMAL_PROFILE = "foundation"
FORMAL_TRAIN_BYTES = 65_536
FORMAL_EVAL_BYTES = 16_384
MAX_PROJECTED_SECONDS = 7_200.0
FORMAT = "taiji-m4r7-formal-preflight-v1"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report {path} must contain a JSON object")
    return payload


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _checkpoint_preflight(
    report: Mapping[str, Any],
    *,
    artifact_dir: Path,
) -> dict[str, Any]:
    seed = int(report["seed"])
    checkpoint = _resolve_path(str(report["source_checkpoint"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing source checkpoint: {checkpoint}")
    payload, model = _load_checkpoint(checkpoint, expected_seed=seed)
    source_digest = content_digest(payload)
    expected_digest = str(report["source_checkpoint_digest"])
    if source_digest != expected_digest:
        raise ValueError(f"seed{seed} source checkpoint digest mismatch")
    before_digest = content_digest(model.checkpoint())
    round_trip_model = type(model).from_checkpoint(payload)
    round_trip_digest = content_digest(round_trip_model.checkpoint())
    save_path = artifact_dir / f"seed{seed}_preflight.pt"
    atomic_save(
        {
            "format": FORMAT,
            "version": 1,
            "seed": seed,
            "source_checkpoint_digest": source_digest,
            "model": model.checkpoint(),
        },
        save_path,
    )
    saved = torch.load(save_path, map_location="cpu", weights_only=False)
    if not isinstance(saved, Mapping):
        raise ValueError(f"seed{seed} saved preflight artifact is not a mapping")
    restored_model = type(model).from_checkpoint(saved["model"])
    restored_digest = content_digest(restored_model.checkpoint())
    return {
        "seed": seed,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_digest": source_digest,
        "config_seed": int(model.config.seed),
        "source_model_digest": before_digest,
        "round_trip_model_digest": round_trip_digest,
        "saved_artifact": str(save_path),
        "saved_artifact_bytes": save_path.stat().st_size,
        "saved_restore_model_digest": restored_digest,
        "source_digest_matches_report": source_digest == expected_digest,
        "checkpoint_round_trip": before_digest == round_trip_digest == restored_digest,
    }


def preflight(
    reports: list[dict[str, Any]],
    *,
    audit_path: Path | None,
    output: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    seeds = tuple(sorted(int(report["seed"]) for report in reports))
    if seeds != EXPECTED_SEEDS:
        raise ValueError("M4.R7 preflight requires exactly seed11, seed29, and seed47")
    if any(
        report["status"] != "passed"
        or report["technical_gate_all_passed"] is not True
        or report["can_promote"] is not False
        for report in reports
    ):
        raise ValueError("source pilot reports must be technical-passed and unpromoted")
    corpora = {_resolve_path(str(report["corpus"])) for report in reports}
    if len(corpora) != 1:
        raise ValueError("formal preflight requires one shared corpus path")
    corpus = next(iter(corpora))
    if not corpus.is_file():
        raise FileNotFoundError(f"missing formal corpus: {corpus}")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_rows = [
        _checkpoint_preflight(report, artifact_dir=artifact_dir) for report in reports
    ]
    chain = build_disjoint_phase_chain(
        [corpus],
        cohort_seeds=EXPECTED_SEEDS,
        profile=FORMAL_PROFILE,
    )
    phases = (chain.phase_c, chain.phase_c2, chain.phase_c3)
    phase_rows = [
        {
            "phase": name,
            "train_bytes_available": len(phase.train),
            "holdout_bytes_available": len(phase.holdout),
            "record_count": len(phase.selected_record_digests),
            "train_budget_satisfied": len(phase.train) >= FORMAL_TRAIN_BYTES,
            "eval_budget_satisfied": len(phase.holdout) >= FORMAL_EVAL_BYTES,
        }
        for name, phase in zip(("c", "c2", "c3"), phases, strict=True)
    ]
    pilot_seconds = sum(float(report["resources"]["elapsed_seconds"]) for report in reports)
    budget_scale = FORMAL_TRAIN_BYTES / 16_384
    projected_seconds = pilot_seconds * budget_scale
    checks = {
        "source_reports_technical": all(
            report["technical_gate_all_passed"] is True for report in reports
        ),
        "source_reports_unpromoted": all(report["can_promote"] is False for report in reports),
        "same_corpus": len(corpora) == 1,
        "checkpoint_source_digest": all(
            row["source_digest_matches_report"] for row in checkpoint_rows
        ),
        "checkpoint_save_and_fresh_restore": all(
            row["checkpoint_round_trip"] for row in checkpoint_rows
        ),
        "record_disjoint_chain": all(value == 0 for value in chain.overlap_counts.values()),
        "foundation_budget_available": all(
            row["train_budget_satisfied"] and row["eval_budget_satisfied"] for row in phase_rows
        ),
        "fixed_capacity_matrix": EXPECTED_SCALES == (0.0, 0.5, 1.0),
        "projected_cpu_within_bound": projected_seconds <= MAX_PROJECTED_SECONDS,
    }
    if audit_path is not None:
        audit = _load_json(audit_path)
        checks["r6_audit_present"] = (
            audit.get("format") == "taiji-m4r6-cycle-retention-attribution-v1"
            and audit.get("can_promote") is False
        )
    report = {
        "format": FORMAT,
        "version": 1,
        "generated_at_epoch": time.time(),
        "status": "passed" if all(checks.values()) else "failed",
        "can_promote": False,
        "formal_allowed": all(checks.values()),
        "configuration": {
            "profile": FORMAL_PROFILE,
            "train_bytes": FORMAL_TRAIN_BYTES,
            "eval_bytes": FORMAL_EVAL_BYTES,
            "scales": list(EXPECTED_SCALES),
            "fixed_capacity": True,
            "learn_fabric": False,
            "learn_memory": False,
        },
        "corpus": str(corpus),
        "checkpoints": checkpoint_rows,
        "data_chain": {
            "phases": phase_rows,
            "overlap_counts": chain.overlap_counts,
        },
        "cost_estimate": {
            "pilot_elapsed_seconds": pilot_seconds,
            "budget_scale_from_pilot": budget_scale,
            "projected_formal_seconds": projected_seconds,
            "max_projected_seconds": MAX_PROJECTED_SECONDS,
        },
        "checks": checks,
        "technical_gate_all_passed": all(checks.values()),
        "decision_boundary": {
            "next_action": (
                "run bounded M4.R7 formal canary"
                if all(checks.values())
                else "do not start formal; repair the failed preflight Gate"
            ),
            "promotion_allowed": False,
        },
        "resources": {
            "preflight_elapsed_seconds": time.perf_counter() - started,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports",
        nargs=3,
        type=Path,
        default=[
            PROJECT_ROOT / "reports" / "taiji_m4r5_update_scale_pilot_seed11_20260908.json",
            PROJECT_ROOT / "reports" / "taiji_m4r5_update_scale_pilot_seed29_20260908.json",
            PROJECT_ROOT / "reports" / "taiji_m4r5_update_scale_pilot_seed47_20260908.json",
        ],
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m4r6_cycle_retention_attribution_20260908.json",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "taiji-m4r7-formal-preflight",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m4r7_formal_preflight_20260908.json",
    )
    args = parser.parse_args(argv)
    reports = [_load_json(path) for path in args.reports]
    audit_path = args.audit if args.audit.is_absolute() else PROJECT_ROOT / args.audit
    report = preflight(
        reports,
        audit_path=audit_path,
        output=args.output,
        artifact_dir=args.artifact_dir,
    )
    print(
        json.dumps(
            {
                "report": str(args.output),
                "status": report["status"],
                "formal_allowed": report["formal_allowed"],
                "projected_formal_seconds": report["cost_estimate"]["projected_formal_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
