"""R4: api 路由层覆盖率补齐——无副作用的轻量 GET/POST 端点。

目标：把 api/ 路由层从 12-40% 的覆盖盲区拉起来。所有用例只触碰：
- 纯读端点（硬件/设置/模型清单/运行时状态）
- 无副作用 POST（返回固定错误消息或校验失败 400）
不启动 startup_tasks，不加载 Legacy 运行时。
"""

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.legacy_bridge import legacy_available


@pytest.fixture(scope="module")
def client():
    app = create_app(startup_tasks=False)
    with TestClient(app) as test_client:
        yield test_client


# ======================== runtime / auth / health ========================


def test_runtime_bootstrap_public(client):
    response = client.get("/api/runtime/bootstrap")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_runtime_status(client):
    response = client.get("/api/runtime/status")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_auth_status_public(client):
    response = client.get("/api/auth/status")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    assert "enabled" in payload


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"loading", "ok"}


def test_openapi_schema_stable(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "Taiji API"
    assert "/api/health" in schema["paths"]


# ======================== system ========================


def test_system_hardware(client):
    response = client.get("/api/system/hardware")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "error"}
    assert "cpu" in payload


def test_system_validate_path_existing_folder(client, tmp_path):
    response = client.post(
        "/api/system/validate_path", json={"path": str(tmp_path), "type": "folder"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_system_validate_path_missing(client):
    response = client.post(
        "/api/system/validate_path", json={"path": "/nonexistent/seed/r4", "type": "folder"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "error"


def test_system_validate_path_type_mismatch(client, tmp_path):
    response = client.post(
        "/api/system/validate_path", json={"path": str(tmp_path), "type": "file"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "error"


def test_native_workbench_workspace_contract(client, tmp_path, monkeypatch):
    import api.routes_workbench as routes_workbench

    updates = {}
    monkeypatch.setattr(
        routes_workbench,
        "update_settings",
        lambda payload: updates.update(payload),
    )
    response = client.post("/api/workbench/workspace", json={"path": str(tmp_path)})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["path"] == str(tmp_path.resolve())
    assert updates["workspace_path"] == str(tmp_path.resolve())


def test_native_workbench_workspace_rejects_relative_path(client):
    response = client.post("/api/workbench/workspace", json={"path": "agent_workspace"})
    assert response.status_code == 400
    assert "absolute" in response.json()["detail"]


def test_native_system_quick_paths_shape(client):
    response = client.get("/api/system/quick_paths")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["paths"], list)
    assert all(set(item) == {"label", "path"} for item in payload["paths"])


# ======================== system reset（安全子集） ========================
# 语义边界：仅支持 scope=chat_sessions（清空对话会话文件），
# 不触及模型权重/检查点/配置。用例通过 monkeypatch 隔离目录，不碰真实 user_data。


def test_system_reset_rejects_unknown_scope(client):
    response = client.post("/api/system/reset", json={"scope": "model_weights"})
    assert response.status_code == 400
    assert "重置范围" in response.json()["detail"]


def test_system_reset_rejects_missing_scope(client):
    response = client.post("/api/system/reset", json={})
    assert response.status_code == 400


def test_system_reset_chat_sessions_clears_history(client, tmp_path, monkeypatch):
    import api.routes_system as routes_system

    history_dir = tmp_path / "user_data" / "chat_history"
    history_dir.mkdir(parents=True)
    (history_dir / "s1.json").write_text("{}", encoding="utf-8")
    (history_dir / "s2.json").write_text("{}", encoding="utf-8")
    (history_dir / "notes.txt").write_text("keep", encoding="utf-8")

    monkeypatch.setattr(routes_system, "get_external_path", lambda rel: str(tmp_path / rel))

    response = client.post("/api/system/reset", json={"scope": "chat_sessions"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["scope"] == "chat_sessions"
    assert payload["removed_sessions"] == 2
    # 非 .json 文件不受影响（语义边界：只清会话历史）
    assert (history_dir / "notes.txt").exists()
    assert not (history_dir / "s1.json").exists()


def test_system_reset_chat_sessions_missing_dir_is_noop(client, tmp_path, monkeypatch):
    import api.routes_system as routes_system

    monkeypatch.setattr(routes_system, "get_external_path", lambda rel: str(tmp_path / rel))
    response = client.post("/api/system/reset", json={"scope": "chat_sessions"})
    assert response.status_code == 200
    assert response.json()["removed_sessions"] == 0


# ======================== settings ========================


def test_settings_get(client):
    response = client.get("/api/settings")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_settings_model_is_retired(client):
    response = client.post("/api/settings/model", json={"model_name": "  "})
    assert response.status_code == 410
    assert response.json()["code"] == "legacy_endpoint_deprecated"


def test_runtime_settings_shape(client):
    response = client.get("/api/settings/runtime")
    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    assert payload["runtime_kind"] == "taiji"


def test_current_runtime_shape(client):
    response = client.get("/api/system/current_runtime")
    assert response.status_code == 200
    payload = response.json()
    assert payload["runtime_kind"] == "taiji"
    assert "model_type" not in payload


def test_current_model_is_retired(client):
    response = client.get("/api/system/current_model")
    assert response.status_code == 410


def test_memory_status(client):
    response = client.get("/api/system/memory")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_memory_refresh(client):
    response = client.post("/api/system/memory/refresh")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


# ======================== models（Cortex 单架构语义） ========================


@pytest.mark.parametrize(
    "path,key",
    [
        ("/api/models/installed", "models"),
        ("/api/models/list", "models"),
        ("/api/models/downloaded", "models"),
        ("/api/models/recommend", "models"),
        ("/api/models/tags", "tags"),
        ("/api/models/families", "families"),
    ],
)
def test_models_catalog_endpoints_are_retired(client, path, key):
    response = client.get(path)
    assert response.status_code == 410
    assert response.json()["code"] == "legacy_endpoint_deprecated"


def test_models_info_is_retired(client):
    response = client.get("/api/models/info")
    assert response.status_code == 410


def test_gguf_quants_are_retired(client):
    response = client.get("/api/model/gguf_quants")
    assert response.status_code == 410


def test_download_progress_is_retired(client):
    response = client.get("/api/models/download_progress")
    assert response.status_code == 410


def test_external_download_routes_are_retired(client):
    for path in ("/api/models/download_hf", "/api/models/download", "/api/models/download_resume"):
        response = client.post(path, json={})
        assert response.status_code == 410


def test_download_cancel_and_pause_are_retired(client):
    for path in ("/api/models/download_cancel", "/api/models/download_pause"):
        response = client.post(path, json={})
        assert response.status_code == 410


def test_artifact_inventory_is_native(client):
    response = client.get("/api/artifacts")
    assert response.status_code == 200
    payload = response.json()
    assert payload["artifact_types"] == [
        "taiji_checkpoint",
        "language_provider_artifact",
        "legacy_benchmark_artifact",
    ]
    assert payload["runtime"]["kind"] == "taiji"


def test_native_openapi_hides_retired_product_contracts(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/artifacts" in paths
    assert "/api/runtime/activate" in paths
    assert "/api/settings/runtime" in paths
    for retired in (
        "/api/models/download_hf",
        "/api/settings/gguf",
        "/api/system/current_model",
        "/api/system/switch_model",
    ):
        assert retired not in paths


# ======================== chat（只读） ========================


def test_chat_sessions_list(client):
    response = client.get("/api/chat/sessions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_chat_history_missing_session_404(client):
    response = client.get("/api/chat/history/no-such-session-r4")
    assert response.status_code == 404


def test_chat_history_rejects_path_traversal(client):
    # _safe_session_id 应阻止路径穿越，找不到会话仍为 404 而非读到其他文件
    response = client.get("/api/chat/history/..%2F..%2Fsettings")
    assert response.status_code == 404


# ======================== model switch（只读） ========================


def test_switch_status_is_retired(client):
    response = client.get("/api/system/switch_status")
    assert response.status_code == 410
    assert response.json()["replacement"] == "/api/runtime/status"


# ======================== auth（只读） ========================


def test_auth_audit_readable(client):
    response = client.get("/api/auth/audit")
    assert response.status_code == 200
    assert isinstance(response.json(), (dict, list))


# ======================== legacy-gated 路由 ========================


@pytest.mark.skipif(not legacy_available(), reason="Legacy workflows disabled")
def test_workflows_remain_legacy_gated(client):
    response = client.get("/api/workflows")
    assert response.status_code == 200
    assert isinstance(response.json(), (dict, list))


def test_legacy_plugin_surface_is_tombstoned(client):
    for path, method in (
        ("/api/plugins", "get"),
        ("/api/plugins/marketplace", "get"),
        ("/api/plugins/marketplace/refresh", "post"),
        ("/api/plugins/upload", "post"),
    ):
        response = getattr(client, method)(path)
        assert response.status_code == 410
        detail = response.json()["detail"]
        assert detail["code"] == "legacy_plugin_surface_retired"
        assert detail["replacement"] == "/api/client-extensions"
