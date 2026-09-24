"""Taiji 原生记忆（append-only 日志 + 召回面）的回归测试。

覆盖 C3「记忆回流通道」的运行时一端：

  1. 一条上报落一条记录，重复上报按内容幂等（写时不重复、跨进程读时收敛）；
  2. 信封字段的规范化（标签去重截断、importance 夹取、text 截断、空值跳过）；
  3. 召回是「关键词 × importance × 新近度」的确定性排序，查询不匹配即不返回；
  4. 三个原生端点常开可达，且请求校验失败返回 422。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from seed_platform import memory_store


@pytest.fixture()
def journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(memory_store, "get_external_path", lambda rel: str(tmp_path / rel))
    memory_store._SEEN_DIGESTS.clear()
    return tmp_path / "data" / "memory" / "entries.jsonl"


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_record_appends_one_entry_and_status_counts(journal: Path) -> None:
    outcome = memory_store.record(
        "interaction",
        "问：你好\n答：我已收到你的问题。",
        session_id="session-1",
        turn=2,
        tags=["workspace:demo", "provider:taiji-local"],
        importance=0.6,
        source="taiji-harness",
        metadata={"tools": ["bash"]},
    )

    assert outcome["status"] == "recorded"
    entries = _read(journal)
    assert len(entries) == 1
    assert entries[0]["kind"] == "interaction"
    assert entries[0]["session_id"] == "session-1"
    assert entries[0]["turn"] == 2
    assert entries[0]["tags"] == ["workspace:demo", "provider:taiji-local"]
    assert entries[0]["importance"] == 0.6
    assert entries[0]["source"] == "taiji-harness"
    assert entries[0]["metadata"] == {"tools": ["bash"]}
    assert entries[0]["entry_id"] and entries[0]["digest"]
    assert entries[0]["recorded_at"] > 0

    summary = memory_store.status()
    assert summary["available"] is True
    assert summary["entries"] == 1
    assert summary["by_kind"] == {"interaction": 1}
    assert summary["tags"] == 2
    assert summary["sessions"] == 1


def test_repeated_report_is_not_appended_twice(journal: Path) -> None:
    first = memory_store.record("interaction", "同一段文本", session_id="s", turn=1)
    again = memory_store.record("interaction", "同一段文本", session_id="s", turn=1)
    rerun = memory_store.record("interaction", "同一段文本但答案是新的", session_id="s", turn=1)

    assert first["status"] == "recorded"
    assert again["status"] == "duplicate"
    assert rerun["status"] == "recorded"
    assert len(_read(journal)) == 2


def test_cross_process_duplicates_collapse_on_read(journal: Path) -> None:
    memory_store.record("interaction", "重启前的同一条", session_id="s", turn=1)
    # A second process would start with an empty in-process set.
    memory_store._SEEN_DIGESTS.clear()
    memory_store.record("interaction", "重启前的同一条", session_id="s", turn=1)

    assert len(_read(journal)) == 2
    assert len(memory_store.entries()) == 1
    assert memory_store.status()["entries"] == 1


def test_empty_values_are_skipped_not_written(journal: Path) -> None:
    assert memory_store.record("interaction", "   ")["status"] == "skipped"
    assert memory_store.record("", "有文本但没 kind")["status"] == "skipped"
    assert _read(journal) == []


def test_envelope_fields_are_normalized(journal: Path) -> None:
    memory_store.record(
        "note",
        "x" * 5000,
        tags=["a", "a", "  ", "b", "c" * 100],
        importance=2.5,
        session_id="s" * 500,
    )

    entry = _read(journal)[0]
    assert len(entry["text"]) == 4000
    assert entry["tags"] == ["a", "b", "c" * 60]
    assert entry["importance"] == 1.0
    assert len(entry["session_id"]) == 200


def test_importance_clamps_and_defaults(journal: Path) -> None:
    memory_store.record("note", "低", importance=-3)
    memory_store.record("note", "高", importance="not a number")
    memory_store.record("note", "非数", importance=float("nan"))
    memory_store.record("note", "默认")

    values = [entry["importance"] for entry in _read(journal)]
    assert values[0] == 0.0
    assert values[1] == memory_store._DEFAULT_IMPORTANCE
    assert values[2] == memory_store._DEFAULT_IMPORTANCE
    assert values[3] == memory_store._DEFAULT_IMPORTANCE
    assert not any(math.isnan(value) for value in values)


def test_recall_drops_entries_no_query_term_matches(journal: Path) -> None:
    memory_store.record("interaction", "今天讨论了记忆回流通道", tags=["memory"], importance=0.9)
    memory_store.record("interaction", "今天讨论了检查点发布", tags=["training"], importance=0.9)

    hits = memory_store.recall("记忆")
    assert [entry["tags"] for entry in hits] == [["memory"]]
    assert hits[0]["score"] > 0

    assert memory_store.recall("完全不出现的词") == []


def test_recall_ranks_by_importance_and_recency_without_a_query(journal: Path) -> None:
    memory_store.record("note", "低重要度", importance=0.1)
    memory_store.record("note", "高重要度", importance=0.95)

    ranked = memory_store.recall()
    assert [entry["text"] for entry in ranked] == ["高重要度", "低重要度"]


def test_recall_filters_and_limit(journal: Path) -> None:
    memory_store.record("interaction", "对话一", session_id="s1", tags=["a"])
    memory_store.record("interaction", "对话二", session_id="s2", tags=["b"])
    memory_store.record("feedback", "反馈一", session_id="s1", tags=["a"])

    assert [entry["text"] for entry in memory_store.recall(kind="feedback")] == ["反馈一"]
    assert {entry["text"] for entry in memory_store.recall(session_id="s1")} == {"对话一", "反馈一"}
    assert [entry["text"] for entry in memory_store.recall(tag="b")] == ["对话二"]
    assert len(memory_store.recall(limit=1)) == 1
    assert memory_store.recall(limit=0) == []


def test_entries_filtering_is_exact(journal: Path) -> None:
    memory_store.record("interaction", "一", session_id="s1")
    memory_store.record("interaction", "二", session_id="s12")

    assert [entry["text"] for entry in memory_store.entries(session_id="s1")] == ["一"]


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(memory_store, "get_external_path", lambda rel: str(tmp_path / rel))
    memory_store._SEEN_DIGESTS.clear()
    with TestClient(create_app(startup_tasks=False)) as test_client:
        yield test_client


def test_memory_endpoints_are_reachable_and_consistent(client: TestClient) -> None:
    payload = {
        "kind": "interaction",
        "text": "问：现在几点\n答：我无法读取时钟。",
        "session_id": "session-route",
        "turn": 4,
        "tags": ["provider:taiji-local"],
        "importance": 0.5,
        "source": "taiji-harness",
    }

    recorded = client.post("/api/memory/record", json=payload)
    assert recorded.status_code == 200
    assert recorded.json()["status"] == "recorded"
    assert client.post("/api/memory/record", json=payload).json()["status"] == "duplicate"

    status = client.get("/api/memory/status")
    assert status.status_code == 200
    assert status.json()["by_kind"] == {"interaction": 1}

    recalled = client.get("/api/memory/recall", params={"query": "时钟", "limit": 3})
    assert recalled.status_code == 200
    body = recalled.json()
    assert body["query"] == "时钟"
    assert [entry["session_id"] for entry in body["entries"]] == ["session-route"]

    assert client.get("/api/memory/recall", params={"query": "不匹配"}).json()["entries"] == []


def test_memory_record_validates_its_request(client: TestClient) -> None:
    assert client.post("/api/memory/record", json={"text": "缺 kind"}).status_code == 422
    assert client.post("/api/memory/record", json={"kind": "note"}).status_code == 422
    assert client.get("/api/memory/recall", params={"limit": 999}).status_code == 422
