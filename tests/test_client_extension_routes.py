from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import routes_client_extensions
from api.app import create_app
from seed_platform.client_extension_host import ClientPluginManifest


@pytest.fixture
def client():
    routes_client_extensions._host = None
    routes_client_extensions._prepared.clear()
    app = create_app(startup_tasks=False)
    with TestClient(app) as test_client:
        yield test_client
    routes_client_extensions._host = None
    routes_client_extensions._prepared.clear()


def _manifest(*, service_dependencies: dict[str, str] | None = None) -> dict:
    return ClientPluginManifest(
        plugin_id="seed.api.preview",
        version="1.0.0",
        scope="workspace",
        slots=("ide.panel",),
        service_dependencies=service_dependencies or {},
        disposer_id="seed.api.preview.dispose",
        disposer_version="1.0.0",
        metadata={"effect": "read_only"},
    ).to_payload()


def test_client_extension_api_exposes_native_snapshot_and_two_phase_commit(client):
    status = client.get("/api/client-extensions")
    assert status.status_code == 200
    status_payload = status.json()
    assert status_payload["status"] == "ok"
    capability_snapshot_id = status_payload["capability_snapshot_id"]

    prepared = client.post(
        "/api/client-extensions/prepare",
        json={
            "capability_snapshot_id": capability_snapshot_id,
            "manifests": [_manifest()],
            "states": {"seed.api.preview": {"open_count": 1}},
        },
    )
    assert prepared.status_code == 200
    prepared_payload = prepared.json()
    assert prepared_payload["status"] == "prepared"
    assert client.get("/api/client-extensions").json()["active"] == []

    committed = client.post(
        "/api/client-extensions/commit",
        json={"prepared_id": prepared_payload["prepared_id"]},
    )
    assert committed.status_code == 200
    active = client.get("/api/client-extensions").json()
    assert [item["plugin_id"] for item in active["active"]] == ["seed.api.preview"]
    assert active["snapshot"]["snapshot_id"] == committed.json()["snapshot"]["snapshot_id"]


def test_client_extension_api_fails_closed_on_stale_or_executable_payload(client):
    status_payload = client.get("/api/client-extensions").json()
    stale = client.post(
        "/api/client-extensions/prepare",
        json={"capability_snapshot_id": "stale", "manifests": []},
    )
    assert stale.status_code == 409

    forbidden = client.post(
        "/api/client-extensions/prepare",
        json={
            "capability_snapshot_id": status_payload["capability_snapshot_id"],
            "manifests": [{**_manifest(), "module": "untrusted.plugin"}],
        },
    )
    assert forbidden.status_code == 400
    assert "executable-source" in forbidden.json()["detail"]


def test_client_extension_api_dependency_quarantine_and_explicit_recovery(client):
    status_payload = client.get("/api/client-extensions").json()
    manifest = _manifest(service_dependencies={"workbench": "1.0"})
    prepared = client.post(
        "/api/client-extensions/prepare",
        json={
            "capability_snapshot_id": status_payload["capability_snapshot_id"],
            "manifests": [manifest],
            "dependency_health": {"workbench": True},
        },
    )
    assert prepared.status_code == 200
    assert client.post(
        "/api/client-extensions/commit",
        json={"prepared_id": prepared.json()["prepared_id"]},
    ).status_code == 200

    lost = client.post(
        "/api/client-extensions/dependency",
        json={"service": "workbench", "healthy": False},
    )
    assert lost.status_code == 200
    assert lost.json()["affected"] == ["seed.api.preview"]
    assert client.get("/api/client-extensions").json()["active"] == []

    recovered = client.post(
        "/api/client-extensions/dependency",
        json={"service": "workbench", "healthy": True},
    )
    assert recovered.status_code == 200
    assert client.get("/api/client-extensions").json()["active"] == []
