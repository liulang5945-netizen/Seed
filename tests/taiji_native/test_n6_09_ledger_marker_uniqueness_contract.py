"""`audit_taiji_ledger_marker_uniqueness.py` 的契约册（DEBT-G125 修法②的另一半）。

这台仪器量的是**「这一条登记在哪个号下」**，不是「这条提到了谁」：每行只认第一枚圈号。
四支必须能红／能假的：重号必须红、行内引用别处历史号不许算重号、零行必须响亮失败、
乱序只出版信息量而不失败（号位 reused 与顺序回退是两件事，混成一谈会让人以为门在守单调性）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "training" / "audit_taiji_ledger_marker_uniqueness.py"

MARKER = chr(0x32B5)  # ㊵
BRACKET = chr(0x3010)  # 【

UNIQUE = (
    f"{BRACKET}2026-10-13 {MARKER}-1 · 第一条】正文。\n"
    f"{BRACKET}2026-10-13 {MARKER}-2 · 第二条】正文里引用了 {MARKER}-1 那句历史号。\n"
    f"{BRACKET}2026-10-13 {MARKER}-4 · 第四条】乱序在这一档只是信息量。\n"
)
Duplicated = UNIQUE + f"{BRACKET}2026-10-13 {MARKER}-2 · 又一个二号】重号。\n"


def _load():
    spec = importlib.util.spec_from_file_location("n6_09_marker_census", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "ledger.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def test_unique_markers_are_green_and_in_line_references_are_not_counted(tmp_path: Path) -> None:
    module = _load()
    report = module.scan(_write(tmp_path, UNIQUE))
    assert report["duplicates"] == {}, report["duplicates"]
    assert report["verdicts"]["ledger_markers_unique"] is True
    #: 第二行正文里那句 `㊵-1` 只是引用，不许把一号计成两次——否则这台仪器会把"提到过"读成"重复登记"。
    assert report["rows_counted"] == 3
    assert report["distinct_markers"] == 3
    assert report["max_marker"] == 4
    #: 乱序（1、2、4 里缺 3 不算，这里 4 之后无更小值）与重号是两件事：本档 nonmonotonic 只出版计数。
    assert isinstance(report["nonmonotonic_runs"], int)
    #: 引用文本不是登记行：`reading_limit` 必须把这条边界写在件里，而不是只写在注释里。
    assert "每行**第一枚**圈号" in report["reading_limit"]


def test_a_duplicated_marker_turns_the_instrument_red(tmp_path: Path) -> None:
    module = _load()
    path = _write(tmp_path, Duplicated)
    report = module.scan(path)
    assert report["duplicates"] == {"2": 2}, report["duplicates"]
    assert report["duplicate_lines"]["2"] == [2, 4], report["duplicate_lines"]
    assert report["verdicts"]["ledger_markers_unique"] is False
    out = tmp_path / "report.json"
    assert module.main(["--playbook", str(path), "--out-report", str(out)]) == 1
    #: 红了也要留证据：读数件必须先落盘，不许"红就什么都不出版"。
    assert out.is_file()


def test_out_of_order_markers_are_reported_but_do_not_fail(tmp_path: Path) -> None:
    """单调性不是这台仪器的守项：回退只出版 `nonmonotonic_runs`，rc 仍为 0。"""

    module = _load()
    text = (
        f"{BRACKET}2026-10-13 {MARKER}-9 · 前一条】正文。\n"
        f"{BRACKET}2026-10-13 {MARKER}-3 · 后一条】正文。\n"
    )
    path = _write(tmp_path, text)
    report = module.scan(path)
    assert report["nonmonotonic_runs"] == 1
    assert report["duplicates"] == {}
    out = tmp_path / "report.json"
    assert module.main(["--playbook", str(path), "--out-report", str(out)]) == 0


def test_zero_marker_rows_fail_loudly_rather_than_reading_green(tmp_path: Path) -> None:
    module = _load()
    path = _write(tmp_path, "# 只有标题\n\n没有圈号登记行。\n")
    try:
        module.scan(path)
    except ValueError as exc:
        assert "零行" in str(exc)
    else:
        raise AssertionError("零行必须响亮失败——「普查仪报了 0 行」与「圈号已规范化」是两件事")


def test_bracket_rows_without_a_marker_are_counted_separately(tmp_path: Path) -> None:
    """`bracket_rows` 与 `rows_counted` 是两个量：前者是"以中括号起的行"，后者是"其中带号的那些"。"""

    module = _load()
    text = f"{BRACKET}一条没有号的登记】正文。\n{BRACKET}2026-10-13 {MARKER}-7 · 有号的】正文。\n"
    report = module.scan(_write(tmp_path, text))
    assert report["bracket_rows"] == 2
    assert report["rows_counted"] == 1
    assert report["max_marker"] == 7


def test_the_refusal_to_overwrite_a_tracked_report_is_a_third_exit(tmp_path: Path) -> None:
    """落盘护栏：目标已存在而未给 `--overwrite` ⇒ rc=2，且不碰原文件（DEBT-G35 那一族）。"""

    module = _load()
    ledger = _write(tmp_path, UNIQUE)
    out = tmp_path / "existing.json"
    out.write_text('{"sentinel": true}\n', encoding="utf-8", newline="\n")
    before = out.read_bytes()
    assert module.main(["--playbook", str(ledger), "--out-report", str(out)]) == 2
    assert out.read_bytes() == before
