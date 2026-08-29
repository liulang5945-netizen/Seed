"""Run the deterministic W7-R1 provider watchdog S0 Gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (
    ExpressionPlan,
    LanguageEmission,
    LanguageProviderHealthGate,
    LanguageProviderHealthPolicy,
    NativeReadableTextLanguageOrgan,
)

MANIFEST = "plans/manifests/taiji_w7_r1_provider_watchdog_v1.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_w7_r1_provider_watchdog_20260829.json"


class _DegradedLanguageOrgan:
    """Deterministic unhealthy provider that leaks a structured surface."""

    backend_id = "s0-degraded-provider"

    def emit(self, expression: ExpressionPlan) -> LanguageEmission:
        return LanguageEmission(
            expression=expression,
            text_bytes=b'{"semantic_slots": 1}',
            backend=self.backend_id,
        )

    def checkpoint(self) -> dict[str, str]:
        return {"backend": self.backend_id}


def build_report() -> dict[str, object]:
    policy = LanguageProviderHealthPolicy(
        failure_threshold=2,
        cooldown_seconds=30.0,
        minimum_accepted_rate=0.5,
        minimum_rate_probes=8,
    )
    report = LanguageProviderHealthGate(policy).evaluate(
        NativeReadableTextLanguageOrgan(),
        degraded_organ=_DegradedLanguageOrgan(),
        artifact_id="s0-active-artifact",
        artifact_digest="a" * 64,
        rollback_artifact_id=NativeReadableTextLanguageOrgan.BACKEND_ID,
        rollback_artifact_digest=None,
        now=100.0,
    )
    return {
        "manifest": MANIFEST,
        "format": report["format"],
        "gate_id": report["gate_id"],
        "policy": report["policy"],
        "metrics": report["metrics"],
        "health_state": report["health_state"],
        "gate": report["gate"],
        "controls": [
            "healthy provider stays active",
            "structured leakage trips the consecutive-failure threshold",
            "cooldown suppresses a second rollback",
            "health state round-trips through checkpoint",
            "content digest changes reset the health record",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = build_report()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
