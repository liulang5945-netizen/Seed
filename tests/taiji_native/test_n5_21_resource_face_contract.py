"""资源面摘要器（`summarize_taiji_n5_resource_face.py`，PLAN-N5-01 行④ 的执行体）的契约测。

这台仪器存在的理由就是一个取法错：**把"臂里多出来的运行时状态"当成"参数长大了"**。
出厂件只存权重（真树现读 93 枚张量），训练臂连状态一起存（17,726 枚），所以朴素求和会给出
`+4,628,052` 这样一枚假增长。本册钉的就是这条分界：

* 参数量只在**同名集合**上比；异名部分单独作为运行时状态出版，两枚数不许相加；
* 激活统计只覆盖"只在臂里"的状态张量——**同名集合里就算名字带 `activity` 也不许进**（这一支
  是最容易被写错的：它让"激活"与"权重面"混成一笔）；
* 三种判定各自都能为假：真长大／不长大但落了状态／连状态都没落；
* 缺 `unit_count` 必须 rc=2，不猜影子容量；`--arm` 写错形状也要 rc=2。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "summarize_taiji_n5_resource_face.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n5_21_resource_face", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _save(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def _base(tmp_path: Path, weight: int = 4) -> Path:
    return _save(
        tmp_path / "base.pt",
        {"weights": {"readout.w": torch.zeros(weight), "readout.activity_like_w": torch.ones(2)}},
    )


def _arm(
    tmp_path: Path,
    *,
    weight: int = 4,
    with_state: bool = True,
    shadow: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {"weights": {"readout.w": torch.zeros(weight)}}
    if with_state:
        payload["runtime"] = {
            "bridge.region.activity": torch.tensor([0.0, 1.5, 0.0]),
            "bridge.region.membrane": torch.tensor([2.0, 2.0]),
            #: 计数器与激活同名前缀、量级碾压（真树实测 max_abs 149,998 就来自这类张量）——
            #: 它必须进 `counters` 桶，绝不能把 `activation.max_abs_element` 顶成一个步数。
            "predictive_context.recurrent.usage": torch.tensor([149998.0, 3.0]),
        }
    payload["n5_shadow"] = shadow if shadow is not None else {"unit_count": 97, "gate": 1.0}
    return _save(tmp_path / "arm.pt", payload)


def test_runtime_state_is_not_counted_as_parameter_growth(tmp_path: Path) -> None:
    base = _base(tmp_path)
    arm = _arm(tmp_path)
    payload = MODULE.summarize({"treated": arm}, base)
    row = payload["arms"]["treated"]
    assert payload["j_n5res_1"] == "parameters_unchanged_runtime_materialized", payload
    assert row["shared_numel_delta"] == 0, row
    assert row["only_in_arm_entries"] == 3, row
    assert row["only_in_arm_numel"] == 7, row
    #: 激活只数"只在臂里"的那两枚；同名集合里那枚名字带 `activity_like` 的权重**不许**混进来。
    assert row["activation"]["state_tensors"] == 2, row["activation"]
    assert row["activation"]["elements"] == 5, row["activation"]
    assert row["activation"]["nonzero_elements"] == 3, row["activation"]
    assert row["activation"]["max_abs_element"] == 2.0, row["activation"]
    #: 计数器桶：同一批"只在臂里"的名字，`usage` 必须落在这里，且它的大数不许渗进上面那组。
    assert row["counters"]["state_tensors"] == 1, row["counters"]
    assert row["counters"]["max_abs_element"] == 149998.0, row["counters"]
    assert row["shadow_unit_count"] == 97, row


def test_real_parameter_growth_on_the_shared_face_flips_the_verdict(tmp_path: Path) -> None:
    base = _base(tmp_path, weight=4)
    arm = _arm(tmp_path, weight=9)
    payload = MODULE.summarize({"treated": arm}, base)
    assert payload["j_n5res_1"] == "parameters_grew_on_shared_face", payload
    assert payload["arms_with_shared_parameter_growth"] == ["treated"], payload


def test_an_arm_that_persists_no_state_is_named_separately(tmp_path: Path) -> None:
    base = _base(tmp_path)
    arm = _arm(tmp_path, with_state=False)
    payload = MODULE.summarize({"treated": arm}, base)
    assert payload["j_n5res_1"] == "no_runtime_state_present", payload
    assert payload["arms"]["treated"]["activation"]["state_tensors"] == 0, payload


def test_a_missing_unit_count_refuses_instead_of_guessing(tmp_path: Path) -> None:
    base = _base(tmp_path)
    arm = _arm(tmp_path, shadow={"gate": 1.0})
    out = tmp_path / "res.json"
    rc = MODULE.main(["--arm", f"treated={arm}", "--base", str(base), "--out-report", str(out)])
    assert rc == 2, rc
    assert not out.exists(), "拒绝路径不许留下读数件"


def test_a_malformed_arm_spec_refuses(tmp_path: Path) -> None:
    out = tmp_path / "res.json"
    rc = MODULE.main(["--arm", "justapath.pt", "--out-report", str(out)])
    assert rc == 2, rc
    payload = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    assert payload is None, "拒绝路径不许留下读数件"
