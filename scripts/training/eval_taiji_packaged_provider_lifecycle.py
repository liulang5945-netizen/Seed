"""Run the deterministic packaged-provider lifecycle Gate.

The lifecycle tests use an allowlisted deterministic decoder stub.  This keeps
the provider boundary executable on CPU while reporting the real-model asset
requirement separately instead of treating a stub as Qwen production evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_provider_watchdog import (  # noqa: E402
    build_replay_report,
    build_report,
)

REPORT_FORMAT = "taiji-w7-p3-3-packaged-provider-lifecycle-v1"
TEST_NODES = (
    "tests/test_seed_product_smoke.py::test_runtime_watchdog_rolls_back_degraded_provider_after_publish",
    "tests/seed/test_language_provider_runtime.py::test_provider_rotation_stages_canary_and_preserves_live_version_on_failure",
    "tests/seed/test_language_provider_runtime.py::test_provider_health_watchdog_walks_the_full_degradation_ladder",
    "tests/seed/test_language_provider_runtime.py::test_provider_health_probe_folds_live_emissions_and_never_upgrades",
    "tests/seed/test_language_provider_runtime.py::test_missing_explicit_provider_rolls_back_and_is_observable",
    "tests/seed/test_language_provider_runtime.py::test_unknown_provider_is_classified_as_manifest_mismatch",
    "tests/seed/test_language_provider_runtime.py::test_product_chat_first_chat_canary_failure_rolls_back",
    "tests/seed/test_language_provider_runtime.py::test_watchdog_thresholds_round_trip_and_fail_closed",
)


def evaluate() -> dict[str, object]:
    s0 = build_report()
    s1 = build_replay_report()
    pytest_result = pytest.main([*TEST_NODES, "-q"])
    real_assets = tuple(
        path
        for path in (PROJECT_ROOT / "models", PROJECT_ROOT / "artifacts")
        if path.is_dir() and any(path.rglob("*.safetensors"))
    )
    metrics = {
        "provider_watchdog_contract": bool(s0["gate"]["passed"]),
        "provider_artifact_checkpoint_replay": bool(s1["gate"]["passed"]),
        "packaged_rotation_and_fallback_tests": pytest_result == pytest.ExitCode.OK,
        "real_provider_model_assets_available": bool(real_assets),
    }
    return {
        "format": REPORT_FORMAT,
        "test_nodes": list(TEST_NODES),
        "metrics": metrics,
        "watchdog": {"s0": s0["metrics"], "s1_gate": s1["gate"]},
        "real_assets": {
            "available": bool(real_assets),
            "paths": [str(path) for path in real_assets],
            "status": "asset-unverified" if not real_assets else "available-for-next-gate",
        },
        "gate": {
            "passed": all(
                metrics[name]
                for name in (
                    "provider_watchdog_contract",
                    "provider_artifact_checkpoint_replay",
                    "packaged_rotation_and_fallback_tests",
                )
            ),
            "criterion": (
                "the deterministic packaged integration seam must rotate an allowlisted "
                "provider, detect degradation, fall back once, preserve cooldown/health "
                "state through checkpoint, and fail closed on missing or invalid providers"
            ),
        },
        "boundary": (
            "The Gate is passed only for the deterministic integration seam. It does not "
            "claim a real Qwen model was loaded, provider quality was validated, or CUDA, "
            "CI, open-domain gains, or AGI were proven."
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = evaluate()
    report_path = PROJECT_ROOT / "reports" / "taiji_w7_p3_3_packaged_provider_lifecycle_20260831.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["gate"]["passed"] else 1)


if __name__ == "__main__":
    main()
