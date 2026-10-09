"""DEBT-G68 修法② 的契约测：`--n5-shadow` 的两条静默空转必须变成响亮拒绝。

事实基座（全部实测，见台账 08 ㊵-587）：`--developmental-bridge-gate` 缺省 `None` 时挂载段落 `0.0`
且**连 setter 都不调**，`activity_saturation`（"活动绝对值 ≥ `dynamics.target_activity` 的单元占比"，
`taiji/adaptive_residual_bridge.py:101-106`）因此恒 0；`should_propose` 是六道 EMA 合取
（`taiji/adaptive_residual_growth.py:437-444`，`minimum_activity_saturation=0.40`）⇒ 60k 符号跑满
10,001 行面**一次提议都没有**，而把 gate 开到 1.0 后**第 264 行就出首提议**。
所以"给了 `--n5-shadow` 却不开 gate"的跑会产出一张 0 提议的面，并被读成"影子无效应"。

四支都要能为假，且四支**都落在 argparse 校验段**（不会真训、不写文件）：

* 只给 `--n5-shadow`（不给 gate）⇒ 拒绝，且拒绝理由点名 gate；
* 给 `--developmental-bridge-gate 0` ⇒ **同样拒绝**（证明这条不是"有没有提过这个旗标"而是取值判断）；
* 只给 `--n5-shadow`（不给 `--pressure-record`）⇒ 先被压强面的守卫拦下（挂载点在那一支里）；
* **正例**：gate=1.0 ＋ `--pressure-record` ⇒ 本条守卫必须放过，让后面"拒写产品件"的守卫接住
  ⇒ 证明拒绝不是恒真，也证明这条守卫排在真正起跑之前。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "train_seed_corpus.py"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("trainer_g68_guards", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAINER = _load()
GATE_MSG = "需要 --developmental-bridge-gate"
PRESSURE_MSG = "--pressure-record"
CHECKPOINT_MSG = "refusing to write the product checkpoint"


def _stderr_of_rejection(monkeypatch, capsys, *argv: str) -> str:
    #: 这支脚本的 `main()` 不收参数（`def main() -> None`），argv 走 `sys.argv`
    #: ⇒ 只能置换 `sys.argv`；四支都在 argparse 校验段退出，不会真训也不会写文件。
    monkeypatch.setattr(sys, "argv", ["train_seed_corpus.py", *argv])
    with pytest.raises(SystemExit) as caught:
        TRAINER.main()
    assert caught.value.code == 2
    captured = capsys.readouterr()
    return captured.err


def test_shadow_without_gate_is_rejected_by_name(monkeypatch, capsys) -> None:
    err = _stderr_of_rejection(
        monkeypatch, capsys, "--n5-shadow", "--pressure-record", "output/x/pressure.jsonl"
    )
    assert GATE_MSG in err


def test_explicit_zero_gate_is_rejected_too(monkeypatch, capsys) -> None:
    #: 关键判别：`0` 是"显式给过这个旗标"，但合取照样不可满足 ⇒ 必须同样拒绝。
    err = _stderr_of_rejection(
        monkeypatch,
        capsys,
        "--n5-shadow",
        "--pressure-record",
        "output/x/pressure.jsonl",
        "--developmental-bridge-gate",
        "0",
    )
    assert GATE_MSG in err


def test_missing_pressure_record_is_caught_first(monkeypatch, capsys) -> None:
    #: 压强面守卫排在前面，所以这里报的是 `--pressure-record` 而不是 gate。
    err = _stderr_of_rejection(monkeypatch, capsys, "--n5-shadow")
    assert PRESSURE_MSG in err
    assert GATE_MSG not in err


def test_opened_gate_passes_this_guard_and_reaches_the_checkpoint_guard(
    monkeypatch, capsys
) -> None:
    #: 正例：本条守卫放过，拒绝来自后面"拒写产品件"——证明它能不为假而存在。
    err = _stderr_of_rejection(
        monkeypatch,
        capsys,
        "--n5-shadow",
        "--pressure-record",
        "output/x/pressure.jsonl",
        "--developmental-bridge-gate",
        "1",
    )
    assert CHECKPOINT_MSG in err
    assert GATE_MSG not in err
