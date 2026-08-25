"""工作台重命名端点测试（POST /api/workspace/rename）。

沿用 tests/test_api_routes.py 样板：``create_app(startup_tasks=False)`` +
TestClient，不启动 startup_tasks；通过 monkeypatch 把工作区指向临时目录，
避免触碰真实 ``agent_workspace``。
"""

import pytest
from fastapi.testclient import TestClient

import api.routes_agent_workspace as workspace_routes
from api.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app(startup_tasks=False)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def ws_dir(tmp_path, monkeypatch):
    """把工作区目录指向独立临时目录。"""
    monkeypatch.setattr(workspace_routes, "_get_workspace_dir", lambda: str(tmp_path))
    return tmp_path


def test_rename_file_success(client, ws_dir):
    src = ws_dir / "a.txt"
    src.write_text("hello", encoding="utf-8")

    response = client.post(
        "/api/workspace/rename",
        json={"old_name": "a.txt", "new_name": "b.txt"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["path"] == "b.txt"
    assert not src.exists()
    assert (ws_dir / "b.txt").read_text(encoding="utf-8") == "hello"


def test_rename_directory_success(client, ws_dir):
    subdir = ws_dir / "docs"
    subdir.mkdir()
    (subdir / "note.md").write_text("x", encoding="utf-8")

    response = client.post(
        "/api/workspace/rename",
        json={"old_name": "docs", "new_name": "manual"},
    )
    assert response.status_code == 200
    assert response.json()["path"] == "manual"
    assert not subdir.exists()
    assert (ws_dir / "manual" / "note.md").exists()


def test_rename_nested_file_keeps_parent_dir(client, ws_dir):
    nested = ws_dir / "src"
    nested.mkdir()
    (nested / "main.py").write_text("print(1)", encoding="utf-8")

    response = client.post(
        "/api/workspace/rename",
        json={"old_name": "src/main.py", "new_name": "src/app.py"},
    )
    assert response.status_code == 200
    assert response.json()["path"] in {"src/app.py", "src\\app.py"}
    assert (nested / "app.py").exists()


def test_rename_target_exists_conflict(client, ws_dir):
    (ws_dir / "a.txt").write_text("a", encoding="utf-8")
    (ws_dir / "b.txt").write_text("b", encoding="utf-8")

    response = client.post(
        "/api/workspace/rename",
        json={"old_name": "a.txt", "new_name": "b.txt"},
    )
    assert response.status_code == 409
    assert "已存在" in response.json()["detail"]
    # 两个文件都保持原样
    assert (ws_dir / "a.txt").read_text(encoding="utf-8") == "a"
    assert (ws_dir / "b.txt").read_text(encoding="utf-8") == "b"


def test_rename_source_missing_404(client, ws_dir):
    response = client.post(
        "/api/workspace/rename", json={"old_name": "ghost.txt", "new_name": "x.txt"}
    )
    assert response.status_code == 404
    assert "不存在" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {"old_name": "../escape.txt", "new_name": "ok.txt"},
        {"old_name": "a.txt", "new_name": "../outside.txt"},
    ],
)
def test_rename_traversal_rejected(client, ws_dir, payload):
    (ws_dir / "a.txt").write_text("a", encoding="utf-8")
    response = client.post("/api/workspace/rename", json=payload)
    assert response.status_code == 403
    assert "路径不安全" in response.json()["detail"]
    # 工作区外未被写入
    assert not (ws_dir.parent / "outside.txt").exists()
    assert not (ws_dir.parent / "escape.txt").exists()


def test_rename_empty_names_400(client, ws_dir):
    response = client.post("/api/workspace/rename", json={"old_name": "", "new_name": "x"})
    assert response.status_code == 400
    response = client.post("/api/workspace/rename", json={"old_name": "a", "new_name": "  "})
    assert response.status_code == 400
