"""N3 乙 τ 旗标（`--growth-min-pressure`）的契约测（owner 2026-10-08 裁"按链分别给值"的落地件）。

四支都必须真走到，本件要防的是"以为接上了"与"以为只动了一枚"：

* **默认关那半**＝不给旗标时**连 `policy` 参数都不给**（类层面记账调用形状），面头
  `minimum_pressure_requested=null`、`minimum_pressure_actual` 与 `policy.minimum_pressure` 都是产品默认 0.70；
* **开旗标那半**＝请求值与读回值成对进面头，且**只替 `minimum_pressure` 一枚**——其余五道阈、
  `required_pressure_steps`／`growth_resource_cost`／`ema_rate` 必须仍是产品默认（这条不许靠"看我写的代码"）；
* **读回不等于请求 ⇒ 整张面作废**（`RuntimeError`，不是静默出版请求值）；
* **CLI 两支响亮拒绝**：越界与"单给旗标不给 `--pressure-record`"各自 rc=2。

真实产品件只读；训练跑的是 `tmp_path` 里的 120 符号小面，不碰 `checkpoints/`。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.training import train_seed_corpus as trainer  # noqa: E402


#: 产品默认值（读自产品自己的类，不在测里重抄常量表）。
def _product_defaults() -> dict[str, Any]:
    from taiji import AdaptiveResidualGrowthPolicy

    policy = AdaptiveResidualGrowthPolicy()
    return {
        "ema_rate": policy.ema_rate,
        "minimum_pressure": policy.minimum_pressure,
        "minimum_residual_error": policy.minimum_residual_error,
        "minimum_fast_slow_conflict": policy.minimum_fast_slow_conflict,
        "minimum_activity_saturation": policy.minimum_activity_saturation,
        "minimum_utility_gap": policy.minimum_utility_gap,
        "minimum_resource_state": policy.minimum_resource_state,
        "required_pressure_steps": policy.required_pressure_steps,
        "growth_resource_cost": policy.growth_resource_cost,
    }


class _GrowthCallSpy:
    """记 `enable_adaptive_residual_growth` 的**调用形状**：给了 policy 没有、给了哪些字段。"""

    def __init__(self) -> None:
        from taiji.model import Taiji

        self.calls: list[dict[str, Any]] = []
        self._original = Taiji.enable_adaptive_residual_growth

        def wrapper(self_model: Any, **kwargs: Any) -> Any:
            policy = kwargs.get("policy")
            record: dict[str, Any] = {"policy_given": policy is not None}
            if policy is not None:
                record["fields"] = {name: getattr(policy, name) for name in _product_defaults()}
            self.calls.append(record)
            return self._original(self_model, **kwargs)

        Taiji.enable_adaptive_residual_growth = wrapper

    def restore(self) -> None:
        from taiji.model import Taiji

        Taiji.enable_adaptive_residual_growth = self._original


def _corpus(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.jsonl"
    docs = [
        "问：甲是什么？答：甲是一个符号串。",
        "问：乙呢？答：乙是另一个符号串。",
        "问：丙呢？答：丙还是符号串。",
    ]
    path.write_text(
        "".join(json.dumps({"text": doc}) + "\n" for doc in docs),
        encoding="utf-8",
        newline="\n",
    )
    return path


def _train(tmp_path: Path, **kwargs: Any) -> Path:
    from seed import SeedConfig

    pressure = tmp_path / "pressure.jsonl"
    trainer.run_training(
        corpus_paths=[_corpus(tmp_path)],
        config=SeedConfig(),
        epochs=1,
        checkpoint_path=tmp_path / "checkpoint.pt",
        progress_path=tmp_path / "progress.jsonl",
        checkpoint_every=1_000_000,
        progress_every=1_000_000,
        max_symbols=120,
        readout="predictive",
        pressure_record=pressure,
        **kwargs,
    )
    return pressure


def _header(pressure: Path) -> dict[str, Any]:
    return json.loads(pressure.read_text(encoding="utf-8").splitlines()[0])


def test_default_off_passes_no_policy_and_keeps_product_threshold(tmp_path: Path) -> None:
    defaults = _product_defaults()
    spy = _GrowthCallSpy()
    try:
        pressure = _train(tmp_path)
    finally:
        spy.restore()
    assert len(spy.calls) == 1, spy.calls
    assert spy.calls[0]["policy_given"] is False
    head = _header(pressure)
    dev = head["developmental"]
    assert dev["minimum_pressure_requested"] is None
    assert float(dev["minimum_pressure_actual"]) == defaults["minimum_pressure"]
    assert head["policy"]["minimum_pressure"] == defaults["minimum_pressure"]
    #: 缺旗标时其余五道阈当然也还是默认——这条同时钉住"读的是产品自己的 policy，不是常量表"。
    for name in ("minimum_residual_error", "ema_rate", "required_pressure_steps"):
        assert head["policy"][name] == defaults[name]


def test_flag_on_replaces_only_minimum_pressure(tmp_path: Path) -> None:
    defaults = _product_defaults()
    spy = _GrowthCallSpy()
    try:
        pressure = _train(tmp_path, growth_minimum_pressure=0.65)
    finally:
        spy.restore()
    assert len(spy.calls) == 1, spy.calls
    call = spy.calls[0]
    assert call["policy_given"] is True
    fields = call["fields"]
    assert fields["minimum_pressure"] == 0.65
    #: "只动一枚"必须在**调用边界**上成立，不是在读数上事后声称。
    untouched = {k: v for k, v in fields.items() if k != "minimum_pressure"}
    assert untouched == {k: v for k, v in defaults.items() if k != "minimum_pressure"}
    dev = _header(pressure)["developmental"]
    assert dev["minimum_pressure_requested"] == 0.65
    assert float(dev["minimum_pressure_actual"]) == 0.65
    assert _header(pressure)["policy"]["minimum_pressure"] == 0.65


def _attach_ignoring_policy(self_model: Any, **kwargs: Any) -> dict[str, Any]:
    """负对照：产品挂载点被换成"吞掉请求的 policy、挂一枚默认 policy 的 trigger"。"""

    from taiji import AdaptiveResidualGrowthPolicy
    from taiji.adaptive_residual_growth import AdaptiveResidualGrowthTrigger

    trigger = AdaptiveResidualGrowthTrigger(
        bridge_id="predictive_residual.bridge", policy=AdaptiveResidualGrowthPolicy()
    )
    self_model._adaptive_residual_growth_trigger = trigger
    return {"format": "negative-control", "version": 0}


def test_readback_mismatch_voids_the_face(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """挂载点若把请求值吞掉（读回仍是默认），本件必须响亮作废而不是照请求值出版 τ。"""

    from taiji.model import Taiji

    monkeypatch.setattr(Taiji, "enable_adaptive_residual_growth", _attach_ignoring_policy)
    with pytest.raises(RuntimeError, match="面作废"):
        _train(tmp_path, growth_minimum_pressure=0.65)
    face = tmp_path / "pressure.jsonl"
    #: 作废必须发生在出版之前：面里不许留下一行把请求值当生效值写出去。
    if face.exists():
        assert "minimum_pressure_requested" not in face.read_text(encoding="utf-8")


@pytest.mark.parametrize("bad_value", ["1.5", "-0.01"])
def test_cli_rejects_out_of_range(bad_value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_seed_corpus.py",
            "--smoke",
            "--pressure-record",
            "output/x.jsonl",
            "--growth-min-pressure",
            bad_value,
        ],
    )
    with pytest.raises(SystemExit) as caught:
        trainer.main()
    assert caught.value.code == 2


def test_cli_rejects_flag_without_face(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["train_seed_corpus.py", "--smoke", "--growth-min-pressure", "0.65"]
    )
    with pytest.raises(SystemExit) as caught:
        trainer.main()
    assert caught.value.code == 2
