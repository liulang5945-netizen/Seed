"""C6 P1（PLAN-M6-01）工作台能力快照投影的回归测试。

覆盖数据环第四块（工作台能力 → 原生可训语料）的三层：

  1. 适配器把能力快照投影成 `workbench_artifact` 语料单元（每条声明一个
     affordance，内容带 id/风险/可逆/类别/参数）——白名单放行是本测试的
     前提，白名单缺这一条时构造即抛；
  2. no_prose 渲染只出标识符与中文标签（风险等级/可逆/类别），运行时的
     英文描述不进训练文本；
  3. 睡眠巩固 pass 的生产者段：快照按 snapshot_id 只投影一次（去重）、
     截断的投影不滚动快照 id、产出的语料是原生训练器认的形状。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seed_platform import memory_store, sleep_pass, turn_records
from seed_platform.evolution_adapters import (
    WORKBENCH_ANSWER_LEAD,
    WORKBENCH_QUESTION,
    WorkbenchCapabilityAdapter,
    render_workbench_no_prose,
    workbench_training_record,
)
from taiji.evolution_experience import EVOLUTION_CONTRACT_VERSION


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from api.training import datasets as training_datasets

    for module in (sleep_pass, turn_records, memory_store, training_datasets):
        monkeypatch.setattr(module, "get_external_path", lambda rel: str(tmp_path / rel))
    turn_records._SEEN_CONSTRAINTS.clear()
    memory_store._SEEN_DIGESTS.clear()
    return tmp_path


SNAPSHOT = {
    "format": "seed-workbench-capability-snapshot-v1",
    "snapshot_id": "snap-abc123",
    "revision": 6,
    "capabilities": [
        {
            "capability_id": "workbench.file.read",
            "description": "Read a workspace file (English prose stays out of the record).",
            "risk": "low",
            "reversible": True,
            "category": "file",
            "parameters": {"path": "string"},
        },
        {
            "capability_id": "workbench.file.delete",
            "description": "Delete a workspace file.",
            "risk": "high",
            "reversible": False,
            "category": "file",
        },
    ],
}


def test_adapter_projects_one_affordance_per_declared_capability() -> None:
    projection = WorkbenchCapabilityAdapter().project(SNAPSHOT)

    assert projection.source_kind == "workbench"
    assert projection.source_id == "snap-abc123"
    assert projection.source_version == "6"
    assert len(projection.corpus) == 2
    first = projection.corpus[0]
    assert first.source_kind == "workbench_artifact"
    assert first.unit_kind == "affordance"
    assert first.content["capability_id"] == "workbench.file.read"
    assert first.content["risk"] == "low"
    assert first.content["reversible"] is True
    # EVOLUTION_CONTRACT_VERSION stays 1: the additive kind must not bump it.
    assert EVOLUTION_CONTRACT_VERSION == 1


def test_adapter_rejects_a_snapshot_without_usable_capabilities() -> None:
    with pytest.raises(ValueError, match="no usable capability"):
        WorkbenchCapabilityAdapter().project({"snapshot_id": "s", "revision": 1})


def test_no_prose_render_carries_labels_but_not_the_english_prose() -> None:
    rendered = render_workbench_no_prose(SNAPSHOT["capabilities"][0])
    assert rendered.startswith("workbench.file.read；")
    assert "风险等级 low" in rendered
    assert "可逆" in rendered
    assert "类别 file" in rendered
    assert "Read a workspace file" not in rendered

    record = workbench_training_record(SNAPSHOT["capabilities"][0])
    assert record["text"].startswith(f"问：{WORKBENCH_QUESTION}\n答：{WORKBENCH_ANSWER_LEAD}")

    with pytest.raises(ValueError, match="capability_id"):
        render_workbench_no_prose({"description": "no identifier here"})


def test_sleep_pass_projects_the_snapshot_once_per_snapshot_id(workspace: Path) -> None:
    state = sleep_pass._load_state()
    first = sleep_pass.project(state, max_records=200, pass_id="t1", workbench_snapshot=SNAPSHOT)
    assert first["by_source"]["workbench_capabilities"] == 2
    assert "workbench_capabilities" in json.dumps(first["by_source"])

    second = sleep_pass.project(state, max_records=200, pass_id="t2", workbench_snapshot=SNAPSHOT)
    assert second["by_source"]["workbench_capabilities"] == 0
    assert "already projected" in second["workbench_note"]


def test_a_capped_projection_does_not_roll_the_snapshot_id(workspace: Path) -> None:
    state = sleep_pass._load_state()
    capped = sleep_pass.project(state, max_records=1, pass_id="t3", workbench_snapshot=SNAPSHOT)
    # Only one of the two capabilities fit; the snapshot id must not roll
    # forward, so the next pass sees the remainder.
    assert capped["by_source"]["workbench_capabilities"] == 1
    assert "workbench_snapshot_id" not in state

    rest = sleep_pass.project(state, max_records=200, pass_id="t4", workbench_snapshot=SNAPSHOT)
    assert rest["by_source"]["workbench_capabilities"] == 1
    assert state["workbench_snapshot_id"] == "snap-abc123"
