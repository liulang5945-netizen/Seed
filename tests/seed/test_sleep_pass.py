"""原生睡眠巩固 pass（B）的回归测试。

覆盖「数据环」在原生线上的生产者：

  1. 分析只报记录支持得起的东西（计数、失败率、按工具的成功率），且把
     「读不到的证据」写成 notes 而不是编一个数；
  2. 投影把约束种子变成 `问：…\\n答：…` 记录、把记忆里的回合原样重放，
     已投影过的文本与已中止的回合不再投影（跨 pass 增量）；
  3. 产出的语料确实能被原生训练器选中（`seed.datasets` 判定可训练 +
     `api.training.datasets.resolve_dataset_path` 能解析）；
  4. 就绪门是显式的，spec 沿用 legacy 的键名并点名语料文件；
  5. 器官睡眠默认不跑；未挂载模型时如实报 skipped；
  6. 三个 HTTP 端点常开可达。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from seed_platform import memory_store, sleep_pass, turn_records

DAY = 86400.0


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every module that resolves the external root at one temporary tree."""

    from api.training import datasets as training_datasets

    for module in (sleep_pass, turn_records, memory_store, training_datasets):
        monkeypatch.setattr(module, "get_external_path", lambda rel: str(tmp_path / rel))
    turn_records._SEEN_CONSTRAINTS.clear()
    memory_store._SEEN_DIGESTS.clear()
    return tmp_path


def _seed_constraint(text: str) -> None:
    turn_records.record_constraint(text)


def _seed_task(success: bool, *, session_id: str = "s1", task: str = "问：你好") -> None:
    turn_records.record_task_outcome(
        task, success, final_answer="答：在" if success else "", session_id=session_id
    )


def _seed_interaction(
    text: str, *, session_id: str = "s1", turn: int = 1, aborted: bool = False
) -> None:
    memory_store.record(
        "interaction",
        text,
        session_id=session_id,
        turn=turn,
        source="taiji-harness",
        metadata={"aborted": aborted},
    )


def _corpus_records(workspace: Path, relative: str) -> list[dict]:
    """Read the corpus a report named; the name is relative to the data root."""

    path = workspace / "data" / relative
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


# ======================== 分析 ========================


def test_analysis_counts_only_what_the_rings_hold(workspace: Path) -> None:
    _seed_task(True)
    _seed_task(False)
    turn_records.record_strategy("prompt", "policy", "问：你好", True, 1.0)
    turn_records.record_strategy("tool_choice", "bash", "问：你好", False, 0.2)
    turn_records.record_strategy("tool_choice", "bash", "问：你好", False, 0.2)

    metrics = sleep_pass.analyze()

    assert metrics["tasks"] == {"total": 2, "succeeded": 1, "failed": 1, "failure_rate": 0.5}
    assert metrics["strategies"]["by_type"] == {"prompt": 1, "tool_choice": 2}
    assert metrics["strategies"]["tools"]["bash"] == {"calls": 2, "successes": 0}
    assert metrics["constraints"] == {"total": 0, "unprojected": 0}
    assert metrics["memory"]["interactions"] == 0


def test_weaknesses_carry_their_counts_and_respect_thresholds(workspace: Path) -> None:
    for _ in range(19):
        _seed_task(False)
    _seed_task(True)

    # 19/20 failures is a measured weakness; a single failed tool call is not.
    metrics = sleep_pass.analyze()
    assert metrics["tasks"]["failure_rate"] == 0.95
    turn_records.record_strategy("tool_choice", "read", "问：你好", False, 0.2)
    metrics = sleep_pass.analyze()

    weaknesses = sleep_pass.weaknesses_of(metrics)
    assert any("失败率 95%" in line and "（19/20" in line for line in weaknesses)
    assert not any("工具 read" in line for line in weaknesses)
    assert sleep_pass.notes_of(metrics)


def test_low_tool_success_rate_becomes_a_weakness_once_it_has_enough_calls(workspace: Path) -> None:
    for _ in range(3):
        turn_records.record_strategy("tool_choice", "read", "问：你好", False, 0.2)

    assert any(
        "工具 read 成功率 0%" in line for line in sleep_pass.weaknesses_of(sleep_pass.analyze())
    )


# ======================== 投影 ========================


def test_projection_writes_constraint_and_interaction_records(workspace: Path) -> None:
    _seed_constraint("你是Taiji，必须只回答可读文本。")
    _seed_interaction("问：你好\n答：你好，我是 Taiji。")

    report = sleep_pass.run(reason="test")

    assert report["projection"]["records"] == 2
    assert report["projection"]["by_source"] == {
        "constraints": 1,
        "interactions": 1,
        "workbench_capabilities": 0,
    }  # C6 P1: the source is always reported, even at zero
    records = _corpus_records(workspace, report["projection"]["corpus"])
    assert (
        records[0]["text"]
        == f"问：{sleep_pass.CONSTRAINT_QUESTION}\n答：你是Taiji，必须只回答可读文本。"
    )
    assert records[1]["text"] == "问：你好\n答：你好，我是 Taiji。"


def test_projection_is_incremental_and_skips_aborted_turns(workspace: Path) -> None:
    _seed_interaction("问：一\n答：一", turn=1)
    _seed_interaction("问：二\n答：二", turn=2, aborted=True)
    first = sleep_pass.run(reason="first")
    assert first["projection"]["records"] == 1
    assert first["projection"]["skipped"] == {"aborted_interactions": 1}

    # The same journal produces nothing new: already-projected text is not repeated,
    # and a pass with nothing new writes no corpus (an empty one is not a dataset,
    # and would truncate what the first pass produced).
    second = sleep_pass.run(reason="second")
    assert second["projection"]["records"] == 0
    assert second["projection"]["corpus"] == ""
    assert second["projection"]["manifest"] == ""
    assert sleep_pass.status()["last_corpus"] == first["projection"]["corpus"]
    assert len(list((workspace / "data" / "consolidated").glob("*.jsonl"))) == 1

    _seed_interaction("问：三\n答：三", turn=3)
    third = sleep_pass.run(reason="third")
    assert third["projection"]["records"] == 1
    assert _corpus_records(workspace, third["projection"]["corpus"])[0]["text"] == "问：三\n答：三"


def test_a_capped_corpus_is_still_a_valid_native_dataset(workspace: Path) -> None:
    from api.training.datasets import resolve_dataset_path
    from seed.datasets import inspect_native_dataset

    for index in range(5):
        _seed_interaction(f"问：{index}\n答：{index}", turn=index + 1)

    report = sleep_pass.run(reason="cap", max_records=3)

    assert report["projection"]["records"] == 3
    relative = report["projection"]["corpus"]
    path = workspace / "data" / relative
    assert path.exists()
    # The native trainer selects it by name with no registration step.
    assert resolve_dataset_path(relative) == str(path)
    assert resolve_dataset_path(path.name) == str(path)
    inspected = inspect_native_dataset(path)
    assert inspected.documents == 3
    assert inspected.native_trainable is True


# ======================== 就绪门与 spec ========================


def test_spec_is_written_only_when_the_data_ring_is_ready(workspace: Path) -> None:
    quiet = sleep_pass.run(reason="quiet")
    assert quiet["spec"]["written"] is False
    assert quiet["spec"]["reason"].startswith("not ready")
    assert not (workspace / "data" / "consolidation" / sleep_pass._SPEC_FILE).exists()

    _seed_constraint("你是Taiji。")
    ready = sleep_pass.run(reason="ready")
    assert ready["spec"]["written"] is True
    spec_path = workspace / "data" / "consolidation" / sleep_pass._SPEC_FILE
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    # Legacy key names, native content, and a named corpus the trainer accepts.
    for key in ("timestamp", "reason", "metrics", "weaknesses", "training_recommendations"):
        assert key in spec
    assert spec["native"] is True
    assert spec["datasets"] == [ready["projection"]["corpus"]]
    assert spec["training_recommendations"][0]["kind"] == "internalise_constraints"
    assert "constraint seed" in spec["reason"]


def test_recommendations_name_workbench_projection_without_inventing_one() -> None:
    """C6 ⑥：折算了几条工作台能力必须成为一条建议，零折算则一条都不许生成。"""

    projected = {"datasets": ["consolidated/night.jsonl"]}
    sleep_pass._add_recommendations(
        projected,
        {"by_source": {"constraints": 2, "interactions": 1, "workbench_capabilities": 16}},
    )
    assert [item["kind"] for item in projected["training_recommendations"]] == [
        "internalise_constraints",
        "rehearse_interactions",
        "internalise_workbench_capabilities",
    ]
    assert (
        "16 declared workbench capability" in projected["training_recommendations"][-1]["rationale"]
    )

    empty = {"datasets": ["consolidated/night.jsonl"]}
    sleep_pass._add_recommendations(
        empty,
        {"by_source": {"constraints": 0, "interactions": 3, "workbench_capabilities": 0}},
    )
    assert [item["kind"] for item in empty["training_recommendations"]] == ["rehearse_interactions"]


def test_state_rolls_forward_across_passes(workspace: Path) -> None:
    _seed_interaction("问：一\n答：一", turn=1)
    sleep_pass.run(reason="first")
    state = json.loads(
        (workspace / "data" / "consolidation" / sleep_pass._STATE_FILE).read_text(encoding="utf-8")
    )
    assert state["passes"] == 1
    assert state["last_corpus"].startswith("consolidated/corpus-")
    assert len(state["projected"]) == 1

    sleep_pass.run(reason="second")
    state = json.loads(
        (workspace / "data" / "consolidation" / sleep_pass._STATE_FILE).read_text(encoding="utf-8")
    )
    assert state["passes"] == 2
    assert len(state["projected"]) == 1
    status = sleep_pass.status()
    assert status["passes"] == 2
    # The journal the pass reads from is reported in the same status, live.
    assert status["journal"]["entries"] == 1
    assert status["journal"]["by_kind"] == {"interaction": 1}


# ======================== 器官睡眠（默认不跑） ========================


def test_organs_are_off_unless_requested(workspace: Path) -> None:
    _seed_interaction("问：一\n答：一")

    report = sleep_pass.run(reason="no organs")

    assert report["organs"] == {"ran": False, "reason": "organs not requested"}


def test_requested_organs_report_honestly_without_a_substrate(workspace: Path) -> None:
    _seed_interaction("问：一\n答：一")

    report = sleep_pass.run(reason="organs", organs=True)

    assert report["organs"]["ran"] is False
    assert report["organs"]["reason"] == "no runtime attached"


def test_a_runtime_without_a_seed_model_is_reported_not_guessed(workspace: Path) -> None:
    _seed_interaction("问：一\n答：一")

    class StubRuntime:
        model = object()

    report = sleep_pass.run(reason="organs", organs=True, runtime=StubRuntime())

    assert report["organs"]["ran"] is False
    assert report["organs"]["reason"].startswith("attached model is not a seed.Seed")
    assert report["projection"]["records"] == 1


# ======================== HTTP 面 ========================


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    for module in (sleep_pass, turn_records, memory_store):
        monkeypatch.setattr(module, "get_external_path", lambda rel: str(tmp_path / rel))
    turn_records._SEEN_CONSTRAINTS.clear()
    memory_store._SEEN_DIGESTS.clear()
    with TestClient(create_app(startup_tasks=False)) as test_client:
        yield test_client


def test_consolidation_endpoints_are_reachable(client: TestClient) -> None:
    client.post(
        "/api/memory/record",
        json={"kind": "interaction", "text": "问：你好\n答：在", "session_id": "http-1", "turn": 1},
    )

    status = client.get("/api/consolidation/status")
    assert status.status_code == 200
    assert status.json()["passes"] == 0
    assert client.get("/api/consolidation/spec").json() == {"spec": None}

    report = client.post("/api/consolidate", json={"reason": "http"})
    assert report.status_code == 200
    body = report.json()
    assert body["reason"] == "http"
    assert body["projection"]["records"] == 1
    assert body["organs"]["ran"] is False

    after = client.get("/api/consolidation/status").json()
    assert after["passes"] == 1
    assert after["last_report"]["reason"] == "http"
    assert after["last_corpus"].startswith("consolidated/corpus-")
    assert after["journal"]["entries"] == 1


def test_consolidate_rejects_an_impossible_budget(client: TestClient) -> None:
    response = client.post("/api/consolidate", json={"max_texts": "not a number"})
    assert response.status_code == 422
