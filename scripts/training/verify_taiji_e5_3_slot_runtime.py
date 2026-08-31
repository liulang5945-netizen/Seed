"""Verify the Seed-owned declarative Vue slot runtime canary."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _text(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def run_gate() -> dict[str, object]:
    component = _text("frontend/src/components/ClientExtensionSlot.vue")
    app_source = _text("frontend/src/App.vue")
    composable = _text("frontend/src/composables/useClientExtensions.js")
    frontend_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PROJECT_ROOT / "frontend" / "src").rglob("*")
        if path.is_file() and path.suffix in {".js", ".vue", ".ts"}
    )
    frontend_test = PROJECT_ROOT / "frontend" / "src" / "__tests__" / "ClientExtensionSlot.test.js"
    test_run = subprocess.run(
        ["npm.cmd", "run", "test", "--", "--run", "src/__tests__/ClientExtensionSlot.test.js"],
        cwd=PROJECT_ROOT / "frontend",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    test_output = (test_run.stdout + "\n" + test_run.stderr)[-4000:]
    checks = {
        "slot_component_is_present": frontend_test.exists()
        and "<slot" in component
        and "slotManifests" in component,
        "snapshot_entry_is_content_keyed": ":key=\"entryKey(manifest)\"" in component
        and "plugin_digest" in component,
        "failure_state_is_visible": "data-extension-state" in component
        and "quarantined" in component
        and "failed" in component,
        "no_plugin_source_is_loaded": all(
            marker not in component for marker in ("entrypoint", "import_path", "source_path", "executable-source")
        ),
        "app_mounts_route_slot": "ClientExtensionSlot" in app_source
        and 'slot-name="route"' in app_source,
        "facade_remains_declarative": "slotManifests" in composable
        and "clientExtensionsPrepare" in composable,
        "frontend_has_no_legacy_plugin_api": "/api/plugins" not in frontend_source,
        "slot_runtime_canary_passed": test_run.returncode == 0,
    }
    return {
        "gate": "taiji-e5-3-client-slot-runtime",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "test_command": "npm.cmd run test -- --run src/__tests__/ClientExtensionSlot.test.js",
        "test_output_tail": test_output,
        "scope": {
            "renderer_source_execution": False,
            "declarative_snapshot_projection": True,
            "mount_unmount_version_replacement": True,
            "failure_and_quarantine_are_visible": True,
            "taiji_cognition_checkpoint_mutated": False,
            "real_third_party_mcp_or_plugin_enabled": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_e5_3_client_slot_runtime_20260901.json",
    )
    args = parser.parse_args()
    result = run_gate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Windows PowerShell may expose a GBK stdout; keep the persisted report UTF-8
    # while making the terminal rendering encoding-agnostic.
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
