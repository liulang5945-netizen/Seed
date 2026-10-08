"""PLAN-N3-09 的 τ 规则判读器（`freeze_taiji_n3_09_tau_rule.py`）契约测。

这台仪器的全部主张是"冻出来的数必须在这张面上够得着"，所以**每一条拒绝支与两条判级支都必须实走**：

* v1 面（缺六道阈与 EMA 自述）⇒ rc=2，**不许**回落到产品默认；
* face 头缺任一条 policy 自述 ⇒ rc=2 并点名缺哪条；
* 观测数（剔掉 warmup 之后）低于样本下限 ⇒ rc=2；
* 逐行合取与产品自述的 `decision_reasons` 不一致 ⇒ rc=2（这条是"我另算一遍"的合法性来源）；
* 判级两支都走：够得着 ⇒ `freezable`；够不着 ⇒ `not_freezable_at_this_grid` 并出版"必须 ≤ X"。

真实在库面只读；被改的副本一律落 pytest 的 `tmp_path`。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/training/freeze_taiji_n3_09_tau_rule.py"
V1_FACE = REPO / "output/n3_04/beta_gate025/pressure.jsonl"
V2_FACE = REPO / "output/n3_08_smoke/pressure.jsonl"


def _load_module():
    spec = importlib.util.spec_from_file_location("n3_09_tau_rule", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


READER = _load_module()


def _records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write(tmp: Path, records: list[dict[str, Any]]) -> str:
    out = tmp / "pressure.jsonl"
    out.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return str(out)


def _run(tmp: Path, face_path: str) -> tuple[int, dict[str, Any]]:
    out = tmp / "verdict.json"
    rc = READER.main(["--face", face_path, "--out", str(out)])
    return rc, json.loads(out.read_text(encoding="utf-8"))


def test_v2_smoke_face_judges_and_table_agrees(tmp_path: Path) -> None:
    """正支之一：真实 v2 面能读完，且 4,999 行合取与产品自述**逐行一致**（mismatch 必须为 0）。"""

    assert V2_FACE.is_file(), "v2 烟测面不在盘上——本测读的是真件"
    rc, payload = _run(tmp_path, str(V2_FACE))
    assert rc == 0, payload
    face = payload["faces"][0]
    assert face["gate_table_rows_checked"] == 4999
    assert face["gate_table_mismatches"] == 0
    assert face["warmup_dropped"] == 17
    assert face["tau_candidate_v2"] == 0.42
    #: 默认关装配（bridge gate 0.0）那档五道分项闸从未同时为真 ⇒ 上界为 None，判"不可冻"。
    assert face["verdict"] == "not_freezable_at_this_grid"
    assert face["five_gate_window_composite_ceiling"] is None


def test_v1_face_refuses_no_fallback(tmp_path: Path) -> None:
    assert V1_FACE.is_file()
    rc, payload = _run(tmp_path, str(V1_FACE))
    assert rc == 2
    error = payload["refused"][0]["error"]
    assert "taiji-n3-pressure-face-v2" in error


def test_missing_policy_self_report_refuses(tmp_path: Path) -> None:
    records = _records(V2_FACE)
    header = next(record for record in records if record.get("kind") == "face")
    del header["policy"]["minimum_activity_saturation"]
    rc, payload = _run(tmp_path, _write(tmp_path, records))
    assert rc == 2
    assert "minimum_activity_saturation" in payload["refused"][0]["error"]


def test_short_face_refuses_after_warmup(tmp_path: Path) -> None:
    records = [record for record in _records(V2_FACE) if record.get("kind") == "face"]
    pressures = [record for record in _records(V2_FACE) if record.get("kind") == "pressure"]
    #: 480 条本身够样本下限，但剔掉 warmup 的 17 条只剩 463 ⇒ 必须拒判（这条钉的是"剔完之后"的口径）。
    rc, payload = _run(tmp_path, _write(tmp_path, records + pressures[:480]))
    assert rc == 2
    error = payload["refused"][0]["error"]
    assert "样本下限" in error
    assert "463" in error


def test_reasons_disagreement_refuses(tmp_path: Path) -> None:
    #: 把一行 EMA 抬到过阈、但保留该产品"说没过阈"的 reasons ⇒ 我的口径与闸不同源，必须闭嘴。
    records = _records(V2_FACE)
    pressures = [record for record in records if record.get("kind") == "pressure"]
    target = pressures[20]
    for key, field in READER.GATES:
        target[key] = float(records[0]["policy"][field]) + 0.05
    assert "pressure_below_threshold" in target["decision_reasons"]
    rc, payload = _run(tmp_path, _write(tmp_path, records))
    assert rc == 2
    assert "合取与产品自述不一致" in payload["refused"][0]["error"]


def _synthetic(header: dict[str, Any], count: int, *, hot_rows: int) -> list[dict[str, Any]]:
    policy = header["policy"]
    rows: list[dict[str, Any]] = [header]
    for index in range(count):
        hot = index >= count - hot_rows
        row: dict[str, Any] = {
            "kind": "pressure",
            "tick": 1000 + index,
            "decision_should_propose": False,
            "decision_reasons": (
                ["persistent_native_pressure"] if hot else ["pressure_below_threshold"]
            ),
        }
        for key, field in READER.GATES:
            threshold = float(policy[field])
            row[key] = threshold + 0.02 if hot else threshold - 0.10
        rows.append(row)
    return rows


def test_freezable_branch_is_reachable(tmp_path: Path) -> None:
    """为真那一支必须也能走：造一张"末尾连续 8 步六道同过"的合面 ⇒ 判 freezable。"""

    records = _records(V2_FACE)
    header = next(record for record in records if record.get("kind") == "face")
    synthetic = _synthetic(header, count=1200, hot_rows=8)
    rc, payload = _run(tmp_path, _write(tmp_path, synthetic))
    assert rc == 0, payload
    face = payload["faces"][0]
    assert face["verdict"] == "freezable"
    assert face["dynamic_range_at_candidate"]["six_gate_longest_run"] == 8
    assert face["dynamic_range_at_candidate"]["passes"] is True
    #: 合成面的上界＝过阈那几行的合成量（这里应恰好是"阈 + 0.02"）。
    assert face["five_gate_window_composite_ceiling"] == round(
        float(header["policy"]["minimum_pressure"]) + 0.02, 6
    )


def test_not_freezable_branch_is_reachable(tmp_path: Path) -> None:
    #: 同一把尺的反面：只连续 2 步（< required_pressure_steps=3）⇒ 必须判"不可冻"并给出上界。
    records = _records(V2_FACE)
    header = next(record for record in records if record.get("kind") == "face")
    synthetic = _synthetic(header, count=1200, hot_rows=2)
    rc, payload = _run(tmp_path, _write(tmp_path, synthetic))
    assert rc == 0, payload
    face = payload["faces"][0]
    assert face["verdict"] == "not_freezable_at_this_grid"
    assert face["dynamic_range_at_candidate"]["six_gate_longest_run"] == 2
    assert face["must_be_at_most_to_be_solvable"] == round(
        float(header["policy"]["minimum_pressure"]) + 0.02, 6
    )
