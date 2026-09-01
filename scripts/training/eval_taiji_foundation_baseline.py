"""M0-1 contract canary for the five Taiji foundation abilities.

The task runners are intentionally not hidden behind this command yet.  Until
they produce real measurements, this entry point emits ``not_evaluated`` and
never manufactures a capability pass from a fixture or a provider response.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji.foundation_evaluation import (
    FOUNDATION_REQUIRED_ABILITIES,
    FoundationEvaluation,
    FoundationManifest,
    FoundationMeasurement,
)  # noqa: E402

DEFAULT_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "taiji_foundation_baseline_v1.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_foundation_baseline_20260901.json"


def _checkpoint_gate_status(manifest: FoundationManifest, path: Path | None) -> str:
    if path is None:
        return "not_run"
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("gate") != manifest.checkpoint_gate:
        return "failed"
    if payload.get("status") != "passed":
        return "failed"
    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        return "failed"
    return "passed"


def build_contract_report(
    manifest: FoundationManifest,
    *,
    checkpoint_gate_status: str,
) -> FoundationEvaluation:
    measurements = {
        ability_id: FoundationMeasurement(
            ability_id=ability_id,
            status="not_evaluated",
            primary_metric=manifest.task(ability_id).primary_metric,
            metric_direction=manifest.task(ability_id).metric_direction,
            metric_value=None,
            baseline_metrics={},
            sample_counts={},
            holdout_updates=0,
            evidence=("m0-1-contract-only; task runner pending",),
        )
        for ability_id in FOUNDATION_REQUIRED_ABILITIES
    }
    return FoundationEvaluation.evaluate(
        manifest,
        measurements,
        checkpoint_gate_status=checkpoint_gate_status,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint-report", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    manifest = FoundationManifest.load(args.manifest)
    checkpoint_status = _checkpoint_gate_status(manifest, args.checkpoint_report)
    evaluation = build_contract_report(
        manifest,
        checkpoint_gate_status=checkpoint_status,
    )
    result = evaluation.to_payload()
    result["manifest_path"] = str(args.manifest)
    result["contract_status"] = "validated"
    result["capability_measurements"] = "not_evaluated"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["report_written"] = args.report.is_file()
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["report_written"] and result["contract_status"] == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
