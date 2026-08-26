"""终端输入换行符规范化测试（任务 #15）。

xterm 键盘 Enter 发送裸 `\r`，管道方式启动的 cmd.exe 不认其为行终止符，
`_normalize_terminal_input` 须在 win32 下统一转成 `\r\n`。

被测函数带平台门（非 win32 直接原样返回），因此断言「发生了转换」的用例必须
显式钉住 `sys.platform`，否则它们在 Windows 上绿、在 Linux CI 上红；而断言
「保持原样」的用例若不钉平台，会因为压根没进转换分支而变成恒真的假绿。
"""

from __future__ import annotations

import pytest

import api.routes_terminal as routes_terminal


@pytest.fixture
def win32_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """钉定 win32，使转换分支在任何宿主平台上都被真实执行。"""
    monkeypatch.setattr(routes_terminal.sys, "platform", "win32")


def test_bare_cr_converted_to_crlf(win32_platform: None):
    assert routes_terminal._normalize_terminal_input("echo hello\r") == "echo hello\r\n"


def test_existing_crlf_not_double_converted(win32_platform: None):
    assert routes_terminal._normalize_terminal_input("dir\r\n") == "dir\r\n"


def test_bare_lf_converted_to_crlf(win32_platform: None):
    assert routes_terminal._normalize_terminal_input("dir\n") == "dir\r\n"


def test_multiline_paste_mixed_newlines(win32_platform: None):
    raw = "echo a\recho b\necho c\r\necho d"
    expected = "echo a\r\necho b\r\necho c\r\necho d"
    assert routes_terminal._normalize_terminal_input(raw) == expected


def test_empty_and_no_newline_unchanged(win32_platform: None):
    assert routes_terminal._normalize_terminal_input("") == ""
    assert routes_terminal._normalize_terminal_input("ls -la") == "ls -la"


def test_non_win32_platform_keeps_input_unchanged(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(routes_terminal.sys, "platform", "linux")
    assert routes_terminal._normalize_terminal_input("echo hello\r") == "echo hello\r"
    assert routes_terminal._normalize_terminal_input("a\rb\nc\r\nd") == "a\rb\nc\r\nd"
