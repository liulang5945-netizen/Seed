"""PLAN-N3-06 重放仪（`replay_taiji_n3_06_gate_attribution.py`）的契约测。

这台仪器的全部风险是"我把六道闸自己算了一遍，然后当成产品算的"。所以它的**每一条锚点都必须被实走一次为假**：

* **正支**＝真实在库面读出 `single_gate_blocking`，且 A-1/A-2 两条逐行锚点的失配计数为 **0**（这条为真的证据只能是读数）；
* **A-1 支**＝只翻掉面里一行的 `decision_should_propose` ⇒ rc=2，并点名首个失配行；
* **A-2 支**＝只扰动一行原始信号（摘要会变、`should_propose` 未必变）⇒ rc=2 指认摘要不同源；
* **A-3 支**＝把面截到样本下限以下／抽掉一个必需字段 ⇒ 各自 rc=2；
* **A-5 支与表外原因支**＝拿一支假 trigger 替掉产品那支，让 `reasons` 与六条 EMA 不自洽 ⇒ 仪器必须拒判
  （这两支证明的不是产品，是"仪器不会把不自洽的读数放出去"）。

真实面件只读；被改的副本一律落 pytest 的 `tmp_path`。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/training/replay_taiji_n3_06_gate_attribution.py"
FACE = REPO / "output/n3_04/beta_gate025/pressure.jsonl"


def _load_module():
    spec = importlib.util.spec_from_file_location("n3_06_gate_replay", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


READER = _load_module()


def _rows(path: Path = FACE) -> list[dict[str, Any]]:
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


def _run(face_path: str, tmp: Path) -> tuple[int, dict[str, Any]]:
    out = tmp / "verdict.json"
    rc = READER.main(["--face", face_path, "--out-report", str(out)])
    return rc, json.loads(out.read_text(encoding="utf-8"))


def test_real_face_is_faithful_and_indicts_the_composite_gate(tmp_path: Path) -> None:
    assert FACE.is_file(), "在库面缺失——本测读的是真件，不是夹具"
    rc, payload = _run(str(FACE), tmp_path)
    assert rc == 0, payload
    face = payload["faces"][0]
    assert face["g_n3c_a1_rows_compared"] == 3999
    assert face["g_n3c_a1_should_propose_mismatches"] == 0
    assert face["g_n3c_a2_digests_compared"] == 3999
    assert face["g_n3c_a2_digest_mismatches"] == 0
    assert face["verdict"] == "single_gate_blocking"
    assert face["dead_gates"] == ["pressure_ema"]
    assert face["g_n3c_meets_pressure_crosscheck_equal"] is True
    #: 六道阈里只有合成量那道被面头自述过——这条把 DEBT-G53 的"缺自述"钉成可机检事实。
    self_reported = {
        label: gate["threshold_self_reported_in_face"] for label, gate in face["per_gate"].items()
    }
    assert self_reported["pressure_ema"] is True
    assert sum(1 for flag in self_reported.values() if not flag) == 5
    assert face["assumed_from_product_defaults"] == [
        "minimum_residual_error",
        "minimum_fast_slow_conflict",
        "minimum_activity_saturation",
        "minimum_utility_gap",
        "minimum_resource_state",
        "ema_rate",
    ]


def test_a1_flipped_decision_refuses(tmp_path: Path) -> None:
    records = _rows()
    pressures = [record for record in records if record.get("kind") == "pressure"]
    assert pressures, "面上一个观测行都没有——夹具坏了"
    pressures[0]["decision_should_propose"] = not pressures[0]["decision_should_propose"]
    rc, payload = _run(_write(tmp_path, records), tmp_path)
    assert rc == 2
    assert payload["faces_judged"] == 0
    error = payload["refused"][0]["error"]
    assert "A-1" in error
    #: 失配行号必须是 0——仪器只点名**首个**失配，行号算错就等于把证据指到别处。
    assert "第 0 行" in error


def test_a2_perturbed_signal_breaks_the_digest_anchor(tmp_path: Path) -> None:
    records = _rows()
    for record in records:
        if record.get("kind") == "pressure":
            #: 只动一个原始信号：`should_propose` 仍是假（远够不着阈），但产品自己算的 `pressure_digest` 会变
            record["utility_gap"] = float(record["utility_gap"]) + 1e-6
            break
    rc, payload = _run(_write(tmp_path, records), tmp_path)
    assert rc == 2
    error = payload["refused"][0]["error"]
    assert "A-2" in error or "摘要" in error


def test_a3_short_face_and_missing_field_each_refuse(tmp_path: Path) -> None:
    records = _rows()
    header = [record for record in records if record.get("kind") == "face"]
    pressures = [record for record in records if record.get("kind") == "pressure"]
    rc, payload = _run(_write(tmp_path, header + pressures[:200]), tmp_path)
    assert rc == 2
    assert "样本下限" in payload["refused"][0]["error"]

    stripped = json.loads(json.dumps(pressures[0]))
    del stripped["evidence_id"]
    rc, payload = _run(_write(tmp_path, header + [stripped] + pressures[1:]), tmp_path)
    assert rc == 2
    assert "evidence_id" in payload["refused"][0]["error"]


def _fake_trigger(reasons: list[str], meets_all: bool) -> type:
    """造一支"看起来像产品"的假 trigger：六道 EMA 全部拉到 0.9（推导为真），但 `reasons` 说了算。"""

    class FakeTrigger:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def observe(self, observation: Any, *, structural_budget: int) -> Any:
            ema = 0.9 if meets_all else 0.1
            return types.SimpleNamespace(
                should_propose=False,
                reasons=list(reasons),
                consecutive_pressure_steps=0,
                pressure=ema,
                residual_error_ema=ema,
                fast_slow_conflict_ema=ema,
                activity_saturation_ema=ema,
                utility_gap_ema=ema,
                resource_state_ema=ema,
            )

    return FakeTrigger


def _patched(monkeypatch, trigger_type: type) -> None:
    monkeypatch.setattr(READER, "AdaptiveResidualGrowthTrigger", trigger_type)


def test_a5_inconsistent_reasons_refuse(tmp_path: Path, monkeypatch) -> None:
    #: 假 trigger 把六道 EMA 都判为过阈（推导 3,999 步真），却仍在 `reasons` 里发 `pressure_below_threshold`
    #: ⇒ 我的推导与产品的自述不同源，仪器必须闭嘴。
    _patched(monkeypatch, _fake_trigger(["pressure_below_threshold"], meets_all=True))
    rc, payload = _run(str(FACE), tmp_path)
    assert rc == 2
    assert "六道闸合取与产品自述不一致" in payload["refused"][0]["error"]


def test_unknown_reason_string_refuses(tmp_path: Path, monkeypatch) -> None:
    #: 决策侧词表若被改出第五种原因，本件的指认前提（:460-467 四条）就不成立 ⇒ 拒判而不是照算。
    _patched(monkeypatch, _fake_trigger(["brand_new_reason"], meets_all=False))
    rc, payload = _run(str(FACE), tmp_path)
    assert rc == 2
    assert "表外原因" in payload["refused"][0]["error"]


def test_pressure_ema_identity_holds_on_real_face() -> None:
    """钉住 PLAN-N3-07 §2 那条恒等式：`Σ wᵢ·EMAᵢ ≡ EMA(Σ wᵢ·xᵢ)`。

    那条恒等式是"旧面可以重放出 EMA 口径"的唯一依据，所以它**不许只是文档里的论证**：
    这里用产品自己的 trigger 与一支我自己维护的线性 EMA 逐步对表，任一步偏离即红。
    """

    from taiji.adaptive_residual_growth import (
        AdaptiveResidualGrowthPressure,
        AdaptiveResidualGrowthTrigger,
    )

    weights = {
        "residual_error": 0.30,
        "fast_slow_conflict": 0.25,
        "activity_saturation": 0.20,
        "utility_gap": 0.25,
    }
    #: trigger 只接受 `bridge_id` 相同的观测（:405-406 是产品自己的守卫，本测不绕开）⇒ 用面里的桥名。
    face_bridge_id = str(next(row["bridge_id"] for row in _rows() if row.get("kind") == "pressure"))
    trigger = AdaptiveResidualGrowthTrigger(bridge_id=face_bridge_id)
    rate = float(trigger.policy.ema_rate)
    mine = 0.0
    worst = 0.0
    for row in _rows():
        if row.get("kind") != "pressure":
            continue
        observation = AdaptiveResidualGrowthPressure.create(
            bridge_id=str(row["bridge_id"]),
            tick=int(row["tick"]),
            residual_error=float(row["residual_error"]),
            fast_slow_conflict=float(row["fast_slow_conflict"]),
            activity_saturation=float(row["activity_saturation"]),
            utility_gap=float(row["utility_gap"]),
            resource_state=float(row["resource_state"]),
            evidence_id=str(row["evidence_id"]),
            parent_checkpoint_digest=str(row.get("parent_checkpoint_digest") or ""),
        )
        raw_composite = sum(weight * float(row[field]) for field, weight in weights.items())
        mine = (1.0 - rate) * mine + rate * raw_composite
        decision = trigger.observe(observation, structural_budget=1)
        worst = max(worst, abs(float(decision.pressure) - mine))
    assert worst < 1e-12, worst
    #: 反证：权重抄错一位就必须偏离（否则这条测只会恒真）。
    wrong = 0.0
    other = AdaptiveResidualGrowthTrigger(bridge_id=face_bridge_id)
    for row in _rows():
        if row.get("kind") != "pressure":
            continue
        raw_composite = 0.35 * float(row["residual_error"]) + 0.25 * float(
            row["fast_slow_conflict"]
        )
        wrong = (1.0 - float(other.policy.ema_rate)) * wrong + 0.25 * raw_composite
        other_observation = AdaptiveResidualGrowthPressure.create(
            bridge_id=str(row["bridge_id"]),
            tick=int(row["tick"]),
            residual_error=float(row["residual_error"]),
            fast_slow_conflict=float(row["fast_slow_conflict"]),
            activity_saturation=float(row["activity_saturation"]),
            utility_gap=float(row["utility_gap"]),
            resource_state=float(row["resource_state"]),
            evidence_id=str(row["evidence_id"]),
            parent_checkpoint_digest=str(row.get("parent_checkpoint_digest") or ""),
        )
        other.observe(other_observation, structural_budget=1)
    assert abs(float(other.pressure_ema) - wrong) > 1e-6
