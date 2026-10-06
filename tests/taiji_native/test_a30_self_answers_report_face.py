"""A30 自答生成器的**报告面**守卫（2026-10-03，随 `--out-report` 必给＋`_rel` 两处改动落）。

为什么值得为"落盘规矩"写守卫：`build_taiji_a30_self_answers.py` 原先把缺省报告名硬钉在一份**已入库**的
读数件上（`PROJECT_ROOT / "reports" / "taiji_a30_self_answers_build_20261001.json"`），而 `--out-report`
是可选的 ⇒ 任何人重跑而不带旗标，就会在跑完之后无声覆盖那张表的唯一一次实测。同一文件的
`relative_to(PROJECT_ROOT)` 又把"输出指到仓外"变成**跑完才炸**（一小时的生成成果只在报告那一步丢掉）。
这两条都能在**不训练**的前提下机检，所以本文件全部守卫都不需要 torch：拒绝分支紧跟在 `parse_args()`
之后返回，而模型导入在其后。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTRUMENT_REL = "scripts/training/build_taiji_a30_self_answers.py"
INSTRUMENT = PROJECT_ROOT / INSTRUMENT_REL
BASE = "output/a31_chunked_self/checkpoint.pt"


def _load():
    spec = importlib.util.spec_from_file_location("_a30_self_answers_under_test", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scan_face_is_the_tracked_file() -> None:
    """零命中的另一种成因是"扫错了面"：这里把范围本身报出来。"""

    tracked = (
        subprocess.run(
            ["git", "ls-files", INSTRUMENT_REL],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
        )
        .stdout.decode()
        .strip()
    )
    assert tracked == INSTRUMENT_REL, f"仪器不在版本控制面上（扫到 {tracked!r}）"
    assert len(INSTRUMENT.read_text(encoding="utf-8").splitlines()) > 100


def test_rel_gives_relative_inside_and_absolute_outside_without_raising(tmp_path: Path) -> None:
    """一正一负：仓内→相对 posix；仓外→绝对 posix 且**不抛**（抛就是"跑完才炸"那个缺陷）。"""

    mod = _load()
    assert mod._rel(PROJECT_ROOT / "output" / "x.jsonl") == "output/x.jsonl"
    outside = tmp_path / "scratch.jsonl"
    assert mod._rel(outside) == outside.resolve().as_posix()


def test_report_face_no_longer_pins_a_tracked_default() -> None:
    """缺陷的正身：代码里不许再有"缺省落到 reports/ 下那份已入库件"的表达式。"""

    source = INSTRUMENT.read_text(encoding="utf-8")
    assert 'PROJECT_ROOT / "reports"' not in source, "缺省报告名又指回已入库件了"
    assert "base.relative_to(PROJECT_ROOT)" not in source
    assert "out.relative_to(PROJECT_ROOT)" not in source
    assert "def _rel(" in source


def test_missing_out_report_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """必给性要能为 false：不带 `--out-report` 必须走 argparse 的 2 号用法错。"""

    mod = _load()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_taiji_a30_self_answers.py",
            "--base",
            BASE,
            "--pairs",
            "1",
            "--out",
            str(tmp_path / "o.jsonl"),
        ],
    )
    with pytest.raises(SystemExit) as caught:
        mod.main()
    assert caught.value.code == 2


def test_existing_target_refused_before_any_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """拒绝路：报告目标已存在 ⇒ rc=2、`[拒绝落盘]` 可见、且**没碰过**输出文件。"""

    mod = _load()
    report = tmp_path / "already_there.json"
    report.write_text("{}", encoding="utf-8")
    out = tmp_path / "should_not_be_created.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_taiji_a30_self_answers.py",
            "--base",
            BASE,
            "--pairs",
            "1",
            "--out",
            str(out),
            "--out-report",
            str(report),
        ],
    )
    assert mod.main() == 2
    assert not out.exists(), "拒绝发生在生成之前，不该留下输出"
    assert "拒绝落盘" in capsys.readouterr().err


def test_fresh_target_is_accepted(tmp_path: Path) -> None:
    """反向也必须走到：目标不存在时**不该**返回 2（否则这条守卫在验一个恒红的门）。

    这里只验到"越过拒绝闸"这一步——再往下就要载模型。做法是把 `read_questions` 换成响亮哨兵：
    越过拒绝闸后会立刻撞上哨兵异常，证明拒绝分支是唯一挡路的东西。
    """

    mod = _load()
    called: list[str] = []

    def _sentinel(*_a, **_k):
        called.append("reached")
        raise RuntimeError("past the refusal gate")

    mod.read_questions = _sentinel
    report = tmp_path / "fresh.json"
    monkeypatch_argv = [
        "build_taiji_a30_self_answers.py",
        "--base",
        BASE,
        "--pairs",
        "1",
        "--out",
        str(tmp_path / "o.jsonl"),
        "--out-report",
        str(report),
    ]
    original = sys.argv
    sys.argv = monkeypatch_argv
    try:
        with pytest.raises(RuntimeError, match="past the refusal gate"):
            mod.main()
    finally:
        sys.argv = original
    assert called == ["reached"], "没走到读语料那步 ⇒ 上一支的 rc=2 可能来自别的原因"
