"""DEBT-G115 修法③ 的契约测（`scripts/training/_taiji_report_io.py`）。

这台助手的存在理由是"每台仪器各写一遍 `write_text`，漏一个 `newline=` 就整档 CRLF"。
所以测的重点不是它能写出文件——而是**它必须能为 false**：
每条断言都配一支同形的反向对照（不带 `newline` 或带 `\r\n` 时必须真的产生 CR），
否则"行尾正确"这句话就只是常量而非参数。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
MODULE = REPO / "scripts" / "training" / "_taiji_report_io.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("taiji_report_io", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IO = _load()


def test_the_helper_writes_no_carriage_returns(tmp_path: Path) -> None:
    target = tmp_path / "r.json"
    shape = IO.write_json_report(target, {"note": "第一行", "second": 2})
    raw = target.read_bytes()
    assert shape["cr"] == 0, shape
    assert shape["bytes"] == len(raw)
    assert raw.count(bytes([13])) == 0
    assert json.loads(raw.decode("utf-8"))["second"] == 2


def test_a_control_write_without_the_helper_does_produce_cr(tmp_path: Path) -> None:
    #: 反面对照（这条测的是"测本身有效"）：同一份文本用旧写法落盘，在这台机器上必须真带 CR。
    #: 若这一支也得到 0，那上面那条断言就没有判别力——平台默认不是 LF 时才轮得到本助手做事。
    target = tmp_path / "old.json"
    target.write_text(json.dumps({"a": 1}, indent=2) + "\n", encoding="utf-8")
    assert target.read_bytes().count(bytes([13])) > 0


def test_newline_is_a_parameter_not_a_constant(tmp_path: Path) -> None:
    #: 参数被真的用掉：显式要 `\r\n` 时必须写出 CRLF，否则"带行尾参数"只是装饰。
    target = tmp_path / "crlf.txt"
    shape = IO.write_text_report(target, "一" + chr(10) + "二" + chr(10), newline="\r\n")
    raw = target.read_bytes()
    assert shape["cr"] == shape["lf"] == 2, shape
    assert bytes([13, 10]) in raw


def test_an_existing_target_is_refused_without_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "r.json"
    IO.write_json_report(target, {"keep": 1})
    before = target.read_bytes()
    try:
        IO.write_json_report(target, {"keep": 2})
    except FileExistsError as error:
        assert "overwrite=True" in str(error)
    else:
        raise AssertionError("a silent second write is the failure this guard exists for")
    #: 拒绝必须发生在截断之前：旧字节原样在，且没有留下 tmp 残骸。
    assert target.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


def test_overwrite_replaces_and_reports_the_new_shape(tmp_path: Path) -> None:
    target = tmp_path / "r.json"
    IO.write_json_report(target, {"keep": 1})
    shape = IO.write_json_report(target, {"keep": 2, "more": "x"}, overwrite=True)
    assert json.loads(target.read_text(encoding="utf-8"))["keep"] == 2
    assert shape["cr"] == 0
    assert shape["bytes"] == target.stat().st_size


def test_missing_parent_directory_is_created(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nest" / "r.json"
    IO.write_json_report(target, {"a": 1})
    assert target.is_file()
    assert target.read_bytes().count(bytes([13])) == 0
