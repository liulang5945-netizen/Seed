"""Verify that the former Legacy plugin surface is retired fail-closed."""

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


def _text(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def run_gate() -> dict[str, object]:
    routes_client_extensions._host = None
    routes_client_extensions._prepared.clear()
    app = create_app(startup_tasks=False)

    legacy_requests = (
        ("/api/plugins", "get"),
        ("/api/plugins/demo/enable", "post"),
        ("/api/plugins/demo/disable", "post"),
        ("/api/plugins/demo", "delete"),
        ("/api/plugins/install", "post"),
        ("/api/plugins/marketplace", "get"),
        ("/api/plugins/marketplace/refresh", "post"),
        ("/api/plugins/upload", "post"),
    )
    with TestClient(app) as client:
        responses = [getattr(client, method)(path) for path, method in legacy_requests]
        client_extensions_status = client.get("/api/client-extensions")
        openapi = client.get("/openapi.json").json()

    details = [response.json().get("detail", {}) for response in responses]
    legacy_source = _text("api/routes_plugins.py")
    agent_mcp_source = _text("api/routes_agent_mcp.py")
    agent_workspace_source = _text("api/routes_agent_workspace.py")
    frontend_root = PROJECT_ROOT / "frontend" / "src"
    frontend_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in frontend_root.rglob("*")
        if path.is_file() and path.suffix in {".js", ".vue", ".ts"}
    )
    checks = {
        "legacy_routes_return_410": all(response.status_code == 410 for response in responses),
        "retirement_detail_is_stable": all(
            detail.get("code") == "legacy_plugin_surface_retired"
            and detail.get("replacement") == "/api/client-extensions"
            for detail in details
        ),
        "legacy_manager_is_removed": "neuroplex.core.plugin_manager" not in legacy_source
        and "PluginManager" not in legacy_source,
        "tombstone_is_seed_owned": "legacy_plugin_surface_retired" in legacy_source,
        "client_extension_surface_remains_live": client_extensions_status.status_code == 200,
        "client_extension_openapi_is_present": "/api/client-extensions" in openapi.get("paths", {}),
        "duplicate_marketplace_routes_removed": "/api/plugins/marketplace" not in agent_mcp_source,
        "duplicate_upload_route_removed": "@router.post(\"/api/plugins/upload\")" not in agent_workspace_source,
        "frontend_does_not_call_legacy_surface": "/api/plugins" not in frontend_text,
    }
    return {
        "gate": "taiji-e5-2-legacy-plugin-surface",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "scope": {
            "legacy_plugin_manager_executable": False,
            "legacy_plugin_routes_publicly_mutate_state": False,
            "seed_owned_client_extension_surface": True,
            "mcp_cognition_and_client_body_are_unchanged": True,
            "real_third_party_mcp_or_plugin_enabled": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_e5_2_legacy_plugin_surface_20260901.json",
    )
    args = parser.parse_args()
    result = run_gate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
