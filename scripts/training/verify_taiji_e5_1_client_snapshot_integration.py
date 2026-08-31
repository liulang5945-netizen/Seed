"""Verify API/Vue integration for the Seed-owned client extension snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import routes_client_extensions  # noqa: E402
from api.app import create_app  # noqa: E402
from seed_platform.client_extension_host import ClientPluginManifest  # noqa: E402


def _manifest() -> dict[str, object]:
    return ClientPluginManifest(
        plugin_id="seed.e5_1.preview",
        version="1.0.0",
        scope="workspace",
        slots=("ide.panel", "route"),
        disposer_id="seed.e5_1.preview.dispose",
        disposer_version="1.0.0",
        metadata={"effect": "read_only"},
    ).to_payload()


def run_gate() -> dict[str, object]:
    routes_client_extensions._host = None
    routes_client_extensions._prepared.clear()
    app = create_app(startup_tasks=False)
    with TestClient(app) as client:
        openapi = client.get("/openapi.json").json()
        status = client.get("/api/client-extensions")
        status_payload = status.json()
        capability_snapshot_id = status_payload["capability_snapshot_id"]
        workbench = client.get("/api/workbench/capabilities").json()
        workbench_snapshot_binding = capability_snapshot_id == workbench["snapshot_id"]

        prepared = client.post(
            "/api/client-extensions/prepare",
            json={
                "capability_snapshot_id": capability_snapshot_id,
                "manifests": [_manifest()],
                "states": {"seed.e5_1.preview": {"open_count": 1}},
            },
        )
        prepared_payload = prepared.json()
        active_before_commit = client.get("/api/client-extensions").json()["active"]
        committed = client.post(
            "/api/client-extensions/commit",
            json={"prepared_id": prepared_payload.get("prepared_id", "")},
        )
        active_after_commit = client.get("/api/client-extensions").json()["active"]

        stale = client.post(
            "/api/client-extensions/prepare",
            json={"capability_snapshot_id": "stale-capability", "manifests": []},
        )

    native_api = (PROJECT_ROOT / "frontend" / "src" / "composables" / "nativeApi.js").read_text(
        encoding="utf-8"
    )
    composable = (
        PROJECT_ROOT / "frontend" / "src" / "composables" / "useClientExtensions.js"
    ).read_text(encoding="utf-8")
    app_source = (PROJECT_ROOT / "frontend" / "src" / "App.vue").read_text(encoding="utf-8")
    checks = {
        "native_api_route_registered": "/api/client-extensions" in openapi.get("paths", {}),
        "client_snapshot_binds_workbench": workbench_snapshot_binding,
        "prepare_is_non_mutating": prepared.status_code == 200 and active_before_commit == [],
        "commit_publishes_one_snapshot": committed.status_code == 200
        and [item["plugin_id"] for item in active_after_commit] == ["seed.e5_1.preview"],
        "stale_capability_rejected": stale.status_code == 409,
        "frontend_native_api_facade": all(
            marker in native_api
            for marker in (
                "clientExtensionsPrepare",
                "clientExtensionsCommit",
                "nativeApiPaths.clientExtensions",
            )
        ),
        "frontend_slot_projection_is_declarative": all(
            marker in composable for marker in ("slotManifests", "manifest.slots", "nativeApi.clientExtensions")
        ),
        "app_injects_client_body_state": all(
            marker in app_source
            for marker in ("useClientExtensions", "provide('clientExtensions'", "clientExtensions.refresh")
        ),
    }
    return {
        "gate": "taiji-e5-1-client-snapshot-integration",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "scope": {
            "workbench_capability_snapshot_is_source_of_truth": True,
            "frontend_executes_plugin_source": False,
            "legacy_plugin_manager_migrated": False,
            "taiji_cognition_checkpoint_mutated": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_e5_1_client_snapshot_integration_20260901.json",
    )
    args = parser.parse_args()
    result = run_gate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
