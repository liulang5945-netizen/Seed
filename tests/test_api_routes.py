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


def test_settings_model_rejects_empty_name(client):
    response = client.post("/api/settings/model", json={"model_name": "  "})
    assert response.status_code == 400


def test_current_model_shape(client):
    response = client.get("/api/system/current_model")
    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    if payload["status"] == "ok":
        assert payload["model_type"] == "self"
        assert "pending_settings" in payload


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
def test_models_catalog_endpoints(client, path, key):
    response = client.get(path)
    assert response.status_code == 200
    assert key in response.json()


def test_models_info_cortex(client):
    response = client.get("/api/models/info")
    assert response.status_code == 200
    assert response.json()["info"]["type"] == "cortex"


def test_gguf_quants_unsupported(client):
    response = client.get("/api/model/gguf_quants")
    assert response.status_code == 200
    assert response.json()["options"] == []


def test_download_progress_idle(client):
    response = client.get("/api/models/download_progress")
    assert response.status_code == 200
    assert response.json()["active"] is False


def test_external_download_rejected(client):
    for path in ("/api/models/download_hf", "/api/models/download", "/api/models/download_resume"):
        response = client.post(path, json={})
        assert response.status_code == 200
        assert response.json()["status"] == "error"


def test_download_cancel_and_pause_noop(client):
    for path in ("/api/models/download_cancel", "/api/models/download_pause"):
        response = client.post(path, json={})
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


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


def test_switch_status(client):
    response = client.get("/api/system/switch_status")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    assert "switching" in payload or "status" in payload


# ======================== auth（只读） ========================


def test_auth_audit_readable(client):
    response = client.get("/api/auth/audit")
    assert response.status_code == 200
    assert isinstance(response.json(), (dict, list))


# ======================== legacy-gated 路由 ========================


@pytest.mark.skipif(not legacy_available(), reason="Legacy plugin disabled")
def test_workflows_and_plugins_when_legacy_enabled(client):
    for path in ("/api/workflows", "/api/plugins"):
        response = client.get(path)
        assert response.status_code == 200
        assert isinstance(response.json(), (dict, list))
