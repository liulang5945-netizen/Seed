"""
态极联网模块 (Web Module)
==========================

统一的联网能力，整合搜索、网页阅读、内容提取。

核心特性：
1. 多引擎搜索（DuckDuckGo, Bing, Baidu）+ 自动降级
2. 智能网页解析（HTML → Markdown，提取正文）
3. 结果缓存（避免重复请求）
4. 内容去重（相似结果合并）
5. 请求限速（防止被封）

使用方式：
    from neuroplex.tools.web import search, fetch, browse
    results = search("Python 3.12 新特性")
    content = fetch("https://example.com")
"""

import concurrent.futures
import difflib
import hashlib
import html
import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger("Taiji.Web")


# ═══════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════


@dataclass
class SearchResult:
    """搜索结果"""

    title: str
    url: str
    snippet: str
    source: str = ""  # 搜索引擎名称

    def to_markdown(self) -> str:
        return f"### {self.title}\n{self.snippet}\n来源: {self.url}"


@dataclass
class WebPage:
    """网页内容"""

    url: str
    title: str
    content: str
    content_type: str = "text"  # text / markdown / html
    fetched_at: float = 0
    size: int = 0

    def to_markdown(self) -> str:
        return f"# {self.title}\n\n{self.content}"


# ═══════════════════════════════════════════════
# 缓存
# ═══════════════════════════════════════════════


class WebCache:
    """联网结果缓存"""

    def __init__(self, cache_dir: str = None, ttl: int = 3600):
        self.cache_dir = cache_dir or os.path.join("taiji_data", "web_cache")
        self.ttl = ttl  # 缓存有效期（秒）
        os.makedirs(self.cache_dir, exist_ok=True)

    def _key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def get(self, url: str) -> str | None:
        """获取缓存"""
        key = self._key(url)
        path = os.path.join(self.cache_dir, f"{key}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if time.time() - data.get("timestamp", 0) > self.ttl:
                os.remove(path)
                return None
            return data.get("content")
        except Exception:
            return None

    def set(self, url: str, content: str):
        """设置缓存"""
        key = self._key(url)
        path = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "url": url,
                        "content": content[:50000],  # 限制缓存大小
                        "timestamp": time.time(),
                    },
                    f,
                    ensure_ascii=False,
                )
        except Exception as e:
            logger.debug("【WebCache.set】处理失败（非致命）: %s", e)

    def clear(self):
        """清空缓存"""
        import shutil

        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)


# 全局缓存
_cache = WebCache()


# ═══════════════════════════════════════════════
# 重试包装器
# ═══════════════════════════════════════════════


def _with_retry(func, max_retries: int = 2, base_delay: float = 1.0):
    """
    指数退避重试：1s → 2s → 4s，最多 2 次重试（共 3 次尝试）。
    加入 ±25% 随机抖动避免惊群。
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            delay = base_delay * (2 ** (attempt - 1))
            jitter = delay * 0.25 * (random.random() * 2 - 1)
            time.sleep(delay + jitter)
        try:
            return func()
        except Exception as e:
            last_exc = e
            logger.debug(f"  Retry {attempt + 1}/{max_retries + 1} for {func.__name__}: {e}")
    raise last_exc


def _search_single_engine(engine_func, query: str, max_results: int, timeout: float = 3.0):
    """
    单引擎搜索，带超时。
    engine_func(query, max_results) -> List[SearchResult]
    """
    try:
        return engine_func(query, max_results)
    except Exception:
        return []


# ═══════════════════════════════════════════════
# 搜索引擎
# ═══════════════════════════════════════════════


def _http_get(url: str, timeout: int = 10) -> str:
    """统一的 HTTP GET 请求（纯 stdlib）"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _search_duckduckgo(query: str, max_results: int = 5) -> list[SearchResult]:
    """DuckDuckGo 搜索（纯 stdlib，通过 HTML 版搜索页抓取）"""
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        html = _http_get(url)
        results = []
        # 解析 DDG HTML 搜索结果
        # 每个结果在 <a class="result__a" ...> 标题 </a> 和 <a class="result__snippet" ...> 摘要 </a>
        blocks = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            html,
            re.DOTALL,
        )
        for href, title, snippet in blocks[:max_results]:
            # DDG 的 href 可能是重定向链接，提取真实 URL
            real_url = href
            if "uddg=" in href:
                m = re.search(r"uddg=([^&]+)", href)
                if m:
                    real_url = urllib.parse.unquote(m.group(1))
            results.append(
                SearchResult(
                    title=re.sub(r"<[^>]+>", "", title).strip(),
                    url=real_url,
                    snippet=re.sub(r"<[^>]+>", "", snippet).strip(),
                    source="DuckDuckGo",
                )
            )
        return results
    except Exception as e:
        logger.debug(f"DuckDuckGo 搜索失败: {e}")
        return []


# 语义化结构指纹：只保留参与"结果块"构成的标签序列，忽略属性与文本。
# 用于页面改版后的相似度定位（借鉴 Scrapling 的自适应选择器思路，去掉了它的
# ML 与持久化存储部分，只保留 difflib 纯标准库的内核）。
_BING_RESULT_SHAPE = "li div div h2 a div p"
# 0.6 由实测校准：真实结果块 0.82、改版块 0.61 通过；老式 table 布局 0.50、
# 导航块 0.40 挡下。页脚链接块形状相似度高达 0.74（与结果块同为 li>div>a>div），
# 单靠形状区分不开，因此再过一道正文长度门。
_BING_ADAPTIVE_THRESHOLD = 0.6
# 去标签后的正文下限（字符）：页脚版权块约 11 字符、导航约 4，真实结果块 38+。
_MIN_BLOCK_TEXT = 30
# 主选择器：class 里含 b_algo 这个 token。\b 保证 b_algo_v2 之类的新变体不命中
# （否则主路径永远命中，改版兜底就成了永不触发的装饰）。
_BING_BLOCK_RE = re.compile(r'<li\s+class="[^"]*\bb_algo\b[^"]*"[^>]*>(.*?)</li>', re.DOTALL)


def _tag_shape(fragment: str) -> str:
    """片段的标签序列指纹（忽略属性、文本与自闭合标签）。"""
    tags = re.findall(r"<([a-z][a-z0-9]*)[\s>/]", fragment, re.I)
    return " ".join(tag.lower() for tag in tags)


def _block_similarity(fragment: str) -> float:
    """片段与期望结果块结构的相似度（difflib 标准库）。"""
    return difflib.SequenceMatcher(None, _tag_shape(fragment), _BING_RESULT_SHAPE).ratio()


def _parse_bing_blocks(
    blocks: list[str], max_results: int, source: str = "Bing"
) -> list[SearchResult]:
    """从结果块 HTML 中抽取 标题/链接/摘要。

    标题取标题级标签内的首个 http 锚（h2 改版成 h1..h4 也能命中），摘要取块内
    首个 <p> 的文本；块内没有 <p> 时摘要留空而不是编造。
    """
    results: list[SearchResult] = []
    for block in blocks[:max_results]:
        title_m = re.search(
            r"<h[1-4][^>]*>\s*<a[^>]*href=\"(https?://[^\"]+)\"[^>]*>(.*?)</a>",
            block,
            re.DOTALL | re.I,
        )
        desc_m = re.search(r"<p[^>]*>(.*?)</p>", block, re.DOTALL | re.I)
        if not title_m:
            continue
        title = html.unescape(re.sub(r"<[^>]+>", "", title_m.group(2)))
        snippet = html.unescape(re.sub(r"<[^>]+>", "", desc_m.group(1))) if desc_m else ""
        # 页面折行会在摘要里留下换行，模型看到的是一句连续文本更合适。
        results.append(
            SearchResult(
                title=re.sub(r"\s+", " ", title).strip(),
                url=title_m.group(1),
                snippet=re.sub(r"\s+", " ", snippet).strip(),
                source=source,
            )
        )
    return results


def _adaptive_bing_blocks(html: str) -> list[str]:
    """自适应兜底：主选择器失配（页面改版）时，按结构相似度挑出仍然像搜索
    结果的 <li> 块。要求块内有 http 链接与标题级标签，避免把导航、页脚等
    相似度不足的块当结果。

    相似度只作过闸，不参与排序——SERP 的结果顺序就是相关性顺序，打乱会让
    最相关的条目掉出 max_results。
    """
    kept: list[str] = []
    for block in re.findall(r"<li\b[^>]*>.*?</li>", html, re.DOTALL | re.I):
        if not re.search(r"https?://", block) or not re.search(r"<h[1-4]\b", block, re.I):
            continue
        if _block_similarity(block) < _BING_ADAPTIVE_THRESHOLD:
            continue
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", block)).strip()
        if len(text) < _MIN_BLOCK_TEXT:
            continue
        kept.append(block)
    return kept


def _search_bing(query: str, max_results: int = 5) -> list[SearchResult]:
    """Bing 搜索（纯 stdlib，网页抓取 + regex 解析）"""
    try:
        url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
        html = _http_get(url)
        # 解析 Bing 搜索结果：<li class="b_algo" ...><h2><a href="..." >标题</a></h2><p>摘要</p>
        # class 属性后常跟 data-id/iid 等额外属性，不能假设引号后立即闭合。
        results = _parse_bing_blocks(_BING_BLOCK_RE.findall(html), max_results)
        if not results:
            # 改版兜底：class 名或标签层级变了，用结构相似度重新定位结果块。
            results = _parse_bing_blocks(
                _adaptive_bing_blocks(html), max_results, source="Bing(adaptive)"
            )
        return results
    except Exception as e:
        logger.debug(f"Bing 搜索失败: {e}")
        return []


def _search_baidu(query: str, max_results: int = 5) -> list[SearchResult]:
    """百度搜索（纯 stdlib，网页抓取 + regex 解析）"""
    try:
        url = f"https://www.baidu.com/s?wd={urllib.parse.quote(query)}"
        html = _http_get(url)
        results = []
        # 解析百度搜索结果
        blocks = re.findall(
            r'<div[^>]*class="[^"]*result[^"]*c-container[^"]*"[^>]*>(.*?)</div>\s*(?=<div[^>]*class="[^"]*result)',
            html,
            re.DOTALL,
        )
        if not blocks:
            blocks = re.findall(r'<div[^>]*id="content_left"[^>]*>(.*)', html, re.DOTALL)
            if blocks:
                blocks = re.findall(
                    r'<div[^>]*class="[^"]*result[^"]*c-container[^"]*"[^>]*>(.*?)(?=<div[^>]*class="[^"]*result[^"]*c-container)',
                    blocks[0],
                    re.DOTALL,
                )
        for block in blocks[:max_results]:
            title_m = re.search(r"<h3[^>]*>\s*<a[^>]*>(.*?)</a>", block, re.DOTALL)
            link_m = re.search(r'<h3[^>]*>\s*<a[^>]*href="([^"]*)"', block, re.DOTALL)
            desc_m = re.search(
                r'<span[^>]*class="[^"]*content-right_8Zs40[^"]*"[^>]*>(.*?)</span>|<div[^>]*class="[^"]*c-abstract[^"]*"[^>]*>(.*?)</div>',
                block,
                re.DOTALL,
            )
            if title_m:
                title = re.sub(r"<[^>]+>", "", title_m.group(1)).strip()
                snippet = ""
                if desc_m:
                    snippet = re.sub(
                        r"<[^>]+>", "", desc_m.group(1) or desc_m.group(2) or ""
                    ).strip()
                results.append(
                    SearchResult(
                        title=title,
                        url=link_m.group(1) if link_m else "",
                        snippet=snippet,
                        source="Baidu",
                    )
                )
        return results
    except Exception as e:
        logger.debug(f"百度搜索失败: {e}")
        return []


def search(query: str, max_results: int = 5, engine: str = "auto") -> list[SearchResult]:
    """
    搜索互联网。

    Args:
        query: 搜索关键词
        max_results: 最大结果数
        engine: 搜索引擎 ("auto", "duckduckgo", "bing", "baidu")

    Returns:
        搜索结果列表
    """
    # 检查缓存
    cache_key = f"search:{query}:{max_results}"
    cached = _cache.get(cache_key)
    if cached:
        try:
            data = json.loads(cached)
            return [SearchResult(**r) for r in data]
        except Exception as e:
            logger.debug("【search】处理失败（非致命）: %s", e)

    results = []

    # 自动模式：依次尝试
    if engine == "auto":
        # 并行竞速：三个引擎同时发，1.5s 后取最早返回的非空结果
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(
                    _with_retry, lambda f=func: f(query, max_results), max_retries=1, base_delay=0.5
                ): name
                for name, func in [
                    ("DuckDuckGo", _search_duckduckgo),
                    ("Bing", _search_bing),
                    ("Baidu", _search_baidu),
                ]
            }
            try:
                for future in concurrent.futures.as_completed(futures, timeout=4.0):
                    name = futures[future]
                    try:
                        r = future.result(timeout=0.5)
                        if r:
                            results = r
                            logger.info(f"搜索引擎 {name} 返回 {len(results)} 条结果")
                            break
                    except Exception:
                        continue
            except concurrent.futures.TimeoutError:
                # 慢引擎（如被墙的 DuckDuckGo）超时不得拖垮已完成的竞速赢家：
                # as_completed 超时抛异常，已收集的非空 results 保留使用。
                logger.info("搜索引擎竞速超时，使用已返回的结果")
            # 取消未完成的任务
            for f in futures:
                f.cancel()
    elif engine == "duckduckgo":
        results = _with_retry(
            lambda: _search_duckduckgo(query, max_results), max_retries=2, base_delay=1.0
        )
    elif engine == "bing":
        results = _with_retry(
            lambda: _search_bing(query, max_results), max_retries=2, base_delay=1.0
        )
    elif engine == "baidu":
        results = _with_retry(
            lambda: _search_baidu(query, max_results), max_retries=2, base_delay=1.0
        )

    # 缓存结果
    if results:
        try:
            _cache.set(
                cache_key,
                json.dumps(
                    [
                        {
                            "title": r.title,
                            "url": r.url,
                            "snippet": r.snippet,
                            "source": r.source,
                        }
                        for r in results
                    ],
                    ensure_ascii=False,
                ),
            )
        except Exception as e:
            logger.debug("【search】处理失败（非致命）: %s", e)

    return results


def search_to_text(query: str, max_results: int = 5) -> str:
    """搜索并返回格式化文本（供工具调用）"""
    results = search(query, max_results)
    if not results:
        return f"搜索 '{query}' 未找到结果。"
    lines = [f"搜索 '{query}' 找到 {len(results)} 条结果:\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title}")
        lines.append(f"   {r.snippet[:200]}")
        lines.append(f"   URL: {r.url}")
        lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════
# 网页抓取
# ═══════════════════════════════════════════════


def _html_to_text(html: str) -> str:
    """HTML 转纯文本（纯 regex，无外部依赖）"""
    # 移除无用标签
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<footer[^>]*>.*?</footer>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<header[^>]*>.*?</header>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<aside[^>]*>.*?</aside>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<iframe[^>]*>.*?</iframe>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 在块级标签前后插入换行
    text = re.sub(r"<(?:br|p|div|h[1-6]|li|tr|blockquote)[^>]*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(?:p|div|h[1-6]|li|tr|blockquote)>", "\n", text, flags=re.IGNORECASE)
    # 移除所有剩余标签
    text = re.sub(r"<[^>]+>", "", text)
    # 解码 HTML 实体
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
    )
    # 清理多余空白
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _html_to_markdown(html: str) -> str:
    """HTML 转 Markdown（纯 regex，无外部依赖）"""
    # 移除无用标签
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<nav[^>]*>.*?</nav>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<footer[^>]*>.*?</footer>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<aside[^>]*>.*?</aside>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<iframe[^>]*>.*?</iframe>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 转换标题
    text = re.sub(r"<h1[^>]*>(.*?)</h1>", r"\n# \1\n", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<h2[^>]*>(.*?)</h2>", r"\n## \1\n", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<h3[^>]*>(.*?)</h3>", r"\n### \1\n", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<h4[^>]*>(.*?)</h4>", r"\n#### \1\n", text, flags=re.DOTALL | re.IGNORECASE)
    # 转换列表
    text = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", text, flags=re.DOTALL | re.IGNORECASE)
    # 转换代码块
    text = re.sub(
        r"<pre[^>]*><code[^>]*>(.*?)</code></pre>",
        r"\n```\n\1\n```\n",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r"<pre[^>]*>(.*?)</pre>", r"\n```\n\1\n```\n", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`", text, flags=re.DOTALL | re.IGNORECASE)
    # 转换引用
    text = re.sub(
        r"<blockquote[^>]*>(.*?)</blockquote>",
        lambda m: "\n> " + m.group(1).strip().replace("\n", "\n> ") + "\n",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # 段落换行
    text = re.sub(r"<(?:br|p|div)[^>]*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(?:p|div)>", "\n", text, flags=re.IGNORECASE)
    # 移除所有剩余标签
    text = re.sub(r"<[^>]+>", "", text)
    # 解码 HTML 实体
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
    )
    # 清理
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch(url: str, as_markdown: bool = True, max_length: int = 10000) -> str:
    """
    抓取网页内容。

    Args:
        url: 网页 URL
        as_markdown: 是否转换为 Markdown
        max_length: 最大内容长度

    Returns:
        网页内容文本
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # 检查缓存
    cached = _cache.get(url)
    if cached:
        return cached[:max_length]

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            html = resp.read().decode("utf-8", errors="ignore")

        # 根据内容类型处理
        if "json" in content_type:
            content = html[:max_length]
        elif as_markdown:
            content = _html_to_markdown(html)
        else:
            content = _html_to_text(html)

        content = content[:max_length]

        # 缓存
        _cache.set(url, content)

        return content
    except urllib.error.HTTPError as e:
        return f"HTTP 错误: {e.code} {e.reason}"
    except Exception as e:
        return f"抓取失败: {e}"


def fetch_to_markdown(url: str, max_length: int = 10000) -> str:
    """抓取网页并转为 Markdown（供工具调用）"""
    content = fetch(url, as_markdown=True, max_length=max_length)
    if content.startswith("抓取失败") or content.startswith("HTTP 错误"):
        return content
    return f"网页内容 ({url}):\n\n{content}"


# ═══════════════════════════════════════════════
# 统一接口（供工具注册）
# ═══════════════════════════════════════════════


def web_search(query: str) -> str:
    """搜索工具的统一接口"""
    return search_to_text(query, max_results=5)


def web_fetch(url: str) -> str:
    """抓取工具的统一接口"""
    return fetch_to_markdown(url, max_length=8000)


def web_browse(url: str) -> str:
    """浏览工具的统一接口（尝试 JS 渲染）"""
    # 先尝试 Playwright
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=15000)
            html = page.content()
            browser.close()
            return _html_to_markdown(html)[:8000]
    except Exception as e:
        logger.debug("【web_browse】处理失败（非致命）: %s", e)

    # 回退到普通抓取
    return fetch_to_markdown(url)


# ═══════════════════════════════════════════════
# 深度搜索：搜索 → 批量抓取 → 结构化摘要
# ═══════════════════════════════════════════════


def search_deep(query: str, max_fetch: int = 3) -> str:
    """
    深度搜索：搜索 + 并行抓取顶部结果 → 结构化 Markdown 摘要。
    模拟 Perplexity/Claude 的 search → fetch → summarize 管道。

    Args:
        query: 搜索关键词
        max_fetch: 最多抓取的网页数

    Returns:
        格式化的 Markdown 搜索结果 + 抓取摘要
    """
    results = search(query, max_results=max(5, max_fetch))
    if not results:
        return f"搜索 '{query}' 未找到结果。"

    lines = [f"## 搜索: {query}\n", f"找到 {len(results)} 条结果:\n"]

    # 并行抓取顶部页面
    fetched = {}
    if max_fetch > 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_fetch, 3)) as pool:
            future_map = {
                pool.submit(
                    _with_retry,
                    lambda u=r.url: fetch(u, as_markdown=True, max_length=3000),
                    max_retries=1,
                    base_delay=0.5,
                ): i
                for i, r in enumerate(results[:max_fetch])
            }
            for future in concurrent.futures.as_completed(future_map, timeout=12.0):
                idx = future_map[future]
                try:
                    content = future.result(timeout=1.0)
                    if content and not content.startswith(("抓取失败", "HTTP 错误")):
                        fetched[idx] = content[:2000]
                except Exception as e:
                    logger.debug("【search_deep】处理失败（非致命）: %s", e)
            for f in future_map:
                f.cancel()

    for i, r in enumerate(results):
        lines.append(f"### {i+1}. {r.title}")
        lines.append(f"**来源**: {r.url}  ({r.source})")
        lines.append(f"> {r.snippet[:300]}")
        if i in fetched:
            lines.append(f"\n**页面摘要**:\n{fetched[i]}")
        lines.append("")

    return "\n".join(lines)


def web_search_deep(query: str) -> str:
    """search_deep 的工具统一接口"""
    return search_deep(query, max_fetch=3)
