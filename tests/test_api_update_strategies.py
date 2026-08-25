"""R6: api.routes_update SSRF 防护与 api.chat_strategies 纯函数覆盖率补齐。

routes_update._validate_update_url 是安全关键路径（防 DNS rebinding 到内网），
chat_strategies 的时间注入/历史构建为聊天前置逻辑，均为无副作用纯函数。
"""

import types

import pytest
from fastapi import HTTPException

from api.chat_strategies import (
    _apply_rag,
    _build_history,
    _get_current_time_str,
    _inject_datetime,
)
from api.routes_update import _validate_update_url

# ======================== routes_update._validate_update_url ========================


def test_update_url_rejects_non_http_scheme():
    for url in ("ftp://example.com/pkg.zip", "file:///etc/passwd", "gopher://x"):
        with pytest.raises(HTTPException) as exc:
            _validate_update_url(url)
        assert exc.value.status_code == 400
        assert "http/https" in exc.value.detail


def test_update_url_rejects_missing_hostname():
    with pytest.raises(HTTPException) as exc:
        _validate_update_url("http:///path/only")
    assert exc.value.status_code == 400


def test_update_url_rejects_localhost():
    for url in ("http://localhost/pkg", "https://foo.localhost/pkg"):
        with pytest.raises(HTTPException) as exc:
            _validate_update_url(url)
        assert exc.value.status_code == 400
        assert "本机" in exc.value.detail


def test_update_url_rejects_private_ip_literals():
    for url in (
        "http://127.0.0.1/pkg",
        "http://10.0.0.5/pkg",
        "http://192.168.1.1/pkg",
        "http://172.16.0.9/pkg",
        "http://169.254.169.254/latest",  # 云元数据端点
        "http://[::1]/pkg",
        "http://0.0.0.0/pkg",
    ):
        with pytest.raises(HTTPException) as exc:
            _validate_update_url(url)
        assert exc.value.status_code == 400, url


def test_update_url_accepts_public_ip_literal():
    # 公网 IP 字面量不需要 DNS 解析，应放行
    _validate_update_url("https://8.8.8.8/pkg.zip")


# ======================== chat_strategies 时间与历史 ========================


def test_current_time_str_shape():
    s = _get_current_time_str()
    assert "年" in s and "月" in s and "日" in s
    assert "星期" in s
    # 形如 "2026年08月24日 星期一 20:30"
    parts = s.split(" ")
    assert len(parts) == 3


def test_inject_datetime_prepends_block():
    result = _inject_datetime("你是一个助手。")
    assert result.startswith("[重要系统信息] 当前时间：")
    assert result.endswith("你是一个助手。")


def test_inject_datetime_empty_prompt():
    result = _inject_datetime("")
    assert result.startswith("[重要系统信息] 当前时间：")
    # 空提示只返回时间块（以换行结尾）
    assert result.endswith("\n\n")


def test_build_history_flattens_pairs():
    req = types.SimpleNamespace(history=[("你好", "你好！"), ("再见", "")])
    history = _build_history(req)
    assert history == [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "再见"},
    ]


def test_build_history_skips_empty_turns():
    req = types.SimpleNamespace(history=[("", ""), (None, None)])
    assert _build_history(req) == []


def test_build_history_no_history_attr():
    req = types.SimpleNamespace(history=None)
    assert _build_history(req) == []


# ======================== chat_strategies._apply_rag ========================


def test_apply_rag_no_kb_returns_prompt_unchanged():
    app_state = types.SimpleNamespace(rag_kb=None)
    assert _apply_rag("问题", app_state) == "问题"


def test_apply_rag_empty_kb_returns_prompt_unchanged():
    kb = types.SimpleNamespace(chunks=[])
    app_state = types.SimpleNamespace(rag_kb=kb)
    assert _apply_rag("问题", app_state) == "问题"


def test_apply_rag_injects_context():
    kb = types.SimpleNamespace(
        chunks=["c1", "c2"], search_with_fallback=lambda q: ["参考A", "参考B"]
    )
    app_state = types.SimpleNamespace(rag_kb=kb)
    result = _apply_rag("我的问题", app_state)
    assert result.startswith("基于以下参考资料回答问题")
    assert "参考A" in result and "参考B" in result
    assert result.endswith("我的问题")
    assert "---" in result


def test_apply_rag_empty_search_result_returns_prompt():
    kb = types.SimpleNamespace(chunks=["c1"], search_with_fallback=lambda q: [])
    app_state = types.SimpleNamespace(rag_kb=kb)
    assert _apply_rag("问题", app_state) == "问题"
