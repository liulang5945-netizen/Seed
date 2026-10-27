"""DEBT-G115 修法③ 的契约测（`scripts/training/_taiji_report_io.py`）。

这台助手的存在理由是"每台仪器各写一遍 `write_text`，漏一个 `newline=` 就整档 CRLF"。
所以测的重点不是它能写出文件——而是**它必须能为 false**：
每条断言都配一支同形的反向对照（不带 `newline` 或带 `\r\n` 时必须真的产生 CR），
否则"行尾正确"这句话就只是常量而非参数。
"""

from __future__ import annotations

import importlib.util
import json
import os
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
    #: 反面对照（这条测的是"测本身有效"）。**宿主默认只是两路里的一路**：Windows 不带 `newline`
    #: 会翻出 CR，POSIX 默认不翻——把判别力押在宿主默认面上，Linux 腿必红而判据没变。
    #: 所以这里把两个平台的默认行尾各**显式**走一遍（两支都能在任一宿主上为假），
    #: 宿主自己的默认值只出版不判。
    payload = json.dumps({"a": 1}, indent=2) + "\n"

    windows_style = tmp_path / "old_windows_default.json"
    windows_style.write_text(payload, encoding="utf-8", newline="\r\n")
    assert windows_style.read_bytes().count(bytes([13])) > 0

    posix_style = tmp_path / "old_posix_default.json"
    posix_style.write_text(payload, encoding="utf-8", newline="\n")
    assert posix_style.read_bytes().count(bytes([13])) == 0

    host = tmp_path / "old_host_default.json"
    host.write_text(payload, encoding="utf-8")
    host_cr = host.read_bytes().count(bytes([13]))
    assert (host_cr > 0) == (os.linesep == "\r\n"), (host_cr, repr(os.linesep))


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
