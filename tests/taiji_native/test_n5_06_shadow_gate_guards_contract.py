"""DEBT-G69 修法①② 的契约测：`--n5-shadow-gate` 的三条守卫＋通电/自述的源码钉。

事实基座（本轮实测）：影子 `_gate` 初值 0.0（`taiji/adaptive_residual_shadow.py:118`），
`forward()`（`:443`）与 `learn()`（`:506`）都带 `if self._gate == 0.0: return` 早退，而
`train_seed_corpus.py` 里原本没有任何 `shadow.set_gate` 调用 ⇒ F 跑虽然把影子物化并落盘
（`unit_count=97`、`gate=0.0`），学习步数是**零**。

守卫侧三支都落在 argparse 校验段（不真训、不写文件）；钉法侧两支是源码顺序钉，
`HEAD` 面上这些串都不存在 ⇒ 加实现之前必红，不是恒真式。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "train_seed_corpus.py"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("trainer_g69_guards", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAINER = _load()
GATE_FLAG = "--n5-shadow-gate"
CHECKPOINT_MSG = "refusing to write the product checkpoint"
BASE = (
    "--readout",
    "predictive",
    "--n5-shadow",
    "--pressure-record",
    "output/x/pressure.jsonl",
    "--developmental-bridge-gate",
    "1",
)


def _err(monkeypatch, capsys, *argv: str) -> str:
    monkeypatch.setattr(sys, "argv", ["train_seed_corpus.py", *argv])
    with pytest.raises(SystemExit) as caught:
        TRAINER.main()
    assert caught.value.code == 2
    return capsys.readouterr().err


def test_gate_flag_without_shadow_flag_is_rejected(monkeypatch, capsys) -> None:
    #: 注意：`parser.error` 会把**整段 usage** 一起打到 stderr，里面本来就含每个旗标名，
    #: 所以断言只能钉「消息本体」，不能钉「旗标名在不在」（那会恒真）。
    err = _err(monkeypatch, capsys, GATE_FLAG, "0.5", "--pressure-record", "output/x/p.jsonl")
    assert "只在 --n5-shadow 打开时生效" in err


def test_gate_value_out_of_range_is_rejected(monkeypatch, capsys) -> None:
    err = _err(monkeypatch, capsys, *BASE, GATE_FLAG, "1.5")
    assert "必须落在 0.0..1.0 之间" in err


def test_in_range_gate_reaches_the_next_guard(monkeypatch, capsys) -> None:
    #: 正例：1.0 合法 ⇒ 这两条守卫必须放过，拒绝来自后面「拒写产品件」那一支（证明不为恒真）。
    err = _err(monkeypatch, capsys, *BASE, GATE_FLAG, "1")
    assert CHECKPOINT_MSG in err
    assert "只在 --n5-shadow 打开时生效" not in err
    assert "必须落在 0.0..1.0 之间" not in err


def test_powering_call_and_self_report_keys_are_wired() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "adaptive_shadow.set_gate(float(n5_shadow_gate))" in source
    for key in (
        '"shadow_gate_requested"',
        '"candidate_gate"',
        '"candidate_utility"',
        '"candidate_counterfactual_utility"',
    ):
        assert key in source, key


def test_gate_defaults_to_unpowered() -> None:
    #: 缺省必须是「不通电」——否则本格悄悄改了默认行为。
    source = SCRIPT.read_text(encoding="utf-8")
    assert "n5_shadow_gate: float | None = None," in source
    assert "if n5_shadow_gate is not None:" in source
