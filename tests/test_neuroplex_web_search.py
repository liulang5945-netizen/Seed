"""Bing 解析器的两条路径：主选择器与改版兜底（结构相似度）。

兜底路径的价值只有一条：页面改版后搜索不坏。因此测试必须证明它真的能在
主选择器失配时救回结果，同时在结构明显不像结果块时不产出垃圾。
"""

import re

import pytest

from neuroplex.tools import web

# 今天实测的 Bing 结构（裁剪）：<li class="b_algo" ...> 内含 h2 标题锚 + p 摘要
BING_HTML = """
<ol id="b_results">
<li class="b_algo" data-id iid=SERP.1><div class="b_tpcn"><a class="tilk" href="https://example.com/a">
<div class="tptt">example.com</div></a></div><h2 class=""><a target="_blank" href="https://example.com/a"
h="ID=SERP.1">Alpha <strong>result</strong></a></h2><div class="b_caption">
<p class="b_lineclamp2" data-x="">Alpha snippet &amp; more</p></div></li>
<li class="b_algo"><h2><a href="https://example.com/b">Beta result</a></h2>
<div class="b_caption"><p class="b_lineclamp4">Beta snippet</p></div></li>
</ol>
"""

# 改版页：class 改名（主选择器失配）+ 标题层级 h2→h3 + 摘要 class 全改。
# 这是"正则易碎、标签序列相似"的典型真实改版。
BING_HTML_REDESIGNED = """
<ol id="b_results">
<li class="b_algo_v2" data-id><div class="tile"><a class="brand" href="https://example.com/a">
<div class="host">example.com</div></a></div><h3 class="hdr"><a href="https://example.com/a"
data-keep="1">Alpha <b>result</b></a></h3><div class="summary"><p class="snippet-x">Alpha result page
explaining the async patterns and the scheduling model in detail.</p></div></li>
<li class="b_algo_v2"><h3><a href="https://example.com/b">Beta result</a></h3>
<div class="summary"><p class="snippet-x">Beta result page with a longer abstract that carries
enough visible text for the length gate.</p></div></li>
</ol>
"""

# 噪声页：导航、页脚版权、老式 table 布局——形状相似度高但不是搜索结果。
BING_HTML_NOISE = """
<ul class="nav">
<li><a href="https://example.com/home">首页</a><span>导航</span></li>
</ul>
<ul class="ft">
<li class="ft-row"><div class="f1"><a href="https://example.com/privacy">隐私政策</a></div>
<div class="f2"><span>Copyright 2026 Example</span></div></li>
<li class="ft-row"><div class="f1"><a href="https://example.com/terms">服务条款</a></div>
<div class="f2"><span>All rights reserved</span></div></li>
</ul>
<li><table><tr><td><h2><a href="https://example.com/legacy">老式布局</a></h2></td></tr></table></li>
"""


@pytest.fixture
def fake_http(monkeypatch):
    """把 _http_get 换成 fixture，避免测试打真网。"""

    def _install(html: str) -> None:
        monkeypatch.setattr(web, "_http_get", lambda url, timeout=10: html)

    return _install


def test_primary_selector_parses_known_structure(fake_http) -> None:
    fake_http(BING_HTML)
    results = web._search_bing("query", max_results=5)
    assert [(r.title, r.url) for r in results] == [
        ("Alpha result", "https://example.com/a"),
        ("Beta result", "https://example.com/b"),
    ]
    assert results[0].snippet == "Alpha snippet & more"
    assert results[0].source == "Bing"


def test_primary_selector_alone_finds_nothing_after_redesign(fake_http) -> None:
    """前提自证：改版页上主选择器确实 0 命中（兜底不是装饰）。"""
    fake_http(BING_HTML_REDESIGNED)
    assert web._BING_BLOCK_RE.findall(BING_HTML_REDESIGNED) == []


def test_primary_selector_matches_token_only(fake_http) -> None:
    """b_algo_v2 这类同前缀新 class 不得被主选择器当成命中（否则兜底永不触发）。"""
    assert web._BING_BLOCK_RE.findall('<li class="b_algo_v2"><h2>x</h2></li>') == []
    assert len(web._BING_BLOCK_RE.findall('<li class="b_algo" data-id><h2>x</h2></li>')) == 1


def test_adaptive_fallback_recovers_results_after_redesign(fake_http) -> None:
    fake_http(BING_HTML_REDESIGNED)
    results = web._search_bing("query", max_results=5)
    assert [(r.title, r.url) for r in results] == [
        ("Alpha result", "https://example.com/a"),
        ("Beta result", "https://example.com/b"),
    ]
    assert results[0].snippet == (
        "Alpha result page explaining the async patterns and the scheduling model in detail."
    )
    assert all(r.source == "Bing(adaptive)" for r in results)


def test_adaptive_fallback_rejects_non_result_blocks(fake_http) -> None:
    """噪声页不得被当成搜索结果：无标题级标签的导航块、以及结构相似度不足的块。"""
    fake_http(BING_HTML_NOISE)
    assert web._search_bing("query", max_results=5) == []


def test_adaptive_fallback_not_used_when_primary_matches(fake_http) -> None:
    """主路径可用时不得改变来源标记（兜底不得篡改正常路径）。"""
    fake_http(BING_HTML)
    results = web._search_bing("query", max_results=5)
    assert all(r.source == "Bing" for r in results)


def test_similarity_threshold_is_not_decorative() -> None:
    """相似度门本身有效：真实/改版块通过，老式 table 布局被挡。"""
    real = (
        '<li><div class="a"><a href="https://x.com/0">host</a></div><h2>'
        '<a href="https://x.com/1">T</a></h2><div class="c"><p>snippet text here</p></div></li>'
    )
    table = '<li><table><tr><td><h2><a href="https://x.com/2">T</a></h2></td></tr></table></li>'
    assert web._block_similarity(real) >= web._BING_ADAPTIVE_THRESHOLD
    assert web._block_similarity(table) < web._BING_ADAPTIVE_THRESHOLD


def test_text_gate_rejects_footer_shaped_blocks() -> None:
    """正文长度门有效：形状像结果块的页脚版权块不得通过。"""
    footer = (
        '<li class="ft"><div class="f1"><a href="https://example.com/privacy">隐私政策</a></div>'
        '<div class="f2"><span>Copyright 2026</span></div></li>'
    )
    # 前提自证：形状相似度高于门槛，仅靠相似度拦不住它。
    assert web._block_similarity(footer) >= web._BING_ADAPTIVE_THRESHOLD
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", footer)).strip()
    assert len(text) < web._MIN_BLOCK_TEXT
