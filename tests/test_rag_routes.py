"""S1-1: RAG 知识库路由（routes_rag）TestClient 用例。

沿用 test_api_routes.py 的无副作用样板：``create_app(startup_tasks=False)``。
通过 monkeypatch 将文档目录指向 tmp_path、app_state.rag_kb 替换为 FakeKB，
避免触碰真实外部数据目录与全局设置。
"""

import pytest
from fastapi.testclient import TestClient

import api.routes_rag as routes_rag
from api.app import create_app
from seed_platform.app_state import app_state


@pytest.fixture(scope="module")
def client():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SEED_ENABLE_LEGACY", "1")
        app = create_app(startup_tasks=False)
        with TestClient(app) as test_client:
            yield test_client


class FakeKB:
    """最小化的知识库替身：仅提供路由用到的接口。"""

    def __init__(self, names=("notes.md",)):
        self.documents = {name: f"text of {name}" for name in names}

    def get_doc_names(self):
        return list(self.documents.keys())

    def remove_file(self, filename):
        self.documents.pop(filename, None)

    def rebuild_index(self):
        return "ok"


@pytest.fixture()
def fake_env(tmp_path, monkeypatch):
    """隔离外部路径与全局 rag_kb，测试结束后还原。"""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("hello kb", encoding="utf-8")

    monkeypatch.setattr(routes_rag, "get_external_path", lambda key: str(tmp_path / key))
    old_kb = app_state.rag_kb
    app_state.update_rag_kb(FakeKB())
    yield docs
    app_state.update_rag_kb(old_kb)


# ======================== /api/rag/files ========================


def test_files_returns_metadata(client, fake_env):
    response = client.get("/api/rag/files")
    assert response.status_code == 200
    files = response.json()["files"]
    assert len(files) == 1
    entry = files[0]
    assert entry["name"] == "notes.md"
    assert entry["size"] == len(b"hello kb")
    assert entry["mtime"] > 0
    assert entry["status"] == "indexed"


def test_files_pending_status_for_unindexed_doc(client, fake_env):
    (fake_env / "loose.txt").write_text("not indexed yet", encoding="utf-8")
    response = client.get("/api/rag/files")
    entries = {e["name"]: e for e in response.json()["files"]}
    assert entries["loose.txt"]["status"] == "pending"
    assert entries["notes.md"]["status"] == "indexed"


def test_files_includes_index_only_record(client, fake_env):
    app_state.rag_kb.documents["ghost.md"] = "only in index"
    response = client.get("/api/rag/files")
    entries = {e["name"]: e for e in response.json()["files"]}
    assert entries["ghost.md"]["status"] == "indexed"
    # 无本地文件的记录省略 size/mtime（前端回退为 —）
    assert "size" not in entries["ghost.md"]
    assert "mtime" not in entries["ghost.md"]


# ======================== /api/rag/clear ========================


def test_clear_removes_docs_and_index(client, fake_env):
    response = client.post("/api/rag/clear")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["removed"] == 1
    assert not fake_env.exists()
    assert app_state.rag_kb.get_doc_names() == []


def test_clear_without_kb(client, fake_env):
    app_state.update_rag_kb(None)
    response = client.post("/api/rag/clear")
    assert response.status_code == 200
    assert response.json()["removed"] == 0


# ======================== /api/rag/preview ========================


def test_preview_returns_content(client, fake_env):
    response = client.get("/api/rag/preview/notes.md")
    assert response.status_code == 200
    assert response.json()["content"] == "hello kb"


def test_preview_missing_file(client, fake_env):
    response = client.get("/api/rag/preview/nope.txt")
    assert response.status_code == 200
    assert response.json()["content"] == "(文件不存在)"


# ======================== /api/rag/config ========================


def test_config_get_shape(client):
    response = client.get("/api/rag/config")
    assert response.status_code == 200
    config = response.json()["config"]
    assert "candidate_k" in config
    assert "enable_reranker" in config


def test_config_put_rejects_unknown_fields(client):
    response = client.put("/api/rag/config", json={"not_a_field": 1})
    assert response.status_code == 400
