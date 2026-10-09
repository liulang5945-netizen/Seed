"""
网页搜索 API 路由
================

把 neuroplex 原生爬虫搜索（三引擎竞速：DuckDuckGo / Bing / Baidu，
见 neuroplex/tools/web.py）暴露为 harness 可调的 HTTP 端点。

客户端 web_search 工具链（taiji-harness 的 scraper 搜索提供方）通过本端点
使用原生爬虫，不依赖任何外部搜索 API 凭证。
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("ApiServer.WebSearch")
router = APIRouter()


class WebSearchRequest(BaseModel):
    """一次网页搜索请求。"""

    query: str = Field(min_length=1, description="搜索关键词")
    max_results: int | None = Field(default=8, ge=1, le=20, description="返回结果数上限")

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class WebSearchSource(BaseModel):
    """一条可引用的搜索来源。"""

    url: str
    title: str | None = None
    snippet: str | None = None
    engine: str | None = None


class WebSearchResponse(BaseModel):
    """搜索结果：来源列表 + 截断标记（与 harness WebSearchResult 对齐）。"""

    sources: list[WebSearchSource]
    truncated: bool = False


def _native_search(query: str, max_results: int) -> list:
    """间接层：测试通过 monkeypatch 本引用替换原生爬虫调用。"""
    from neuroplex.tools.web import search

    return search(query, max_results=max_results)


@router.post("/api/tools/web_search")
def web_search(request: WebSearchRequest) -> WebSearchResponse:
    """用原生爬虫搜索网页（多引擎竞速，无需任何搜索 API 凭证）。"""
    try:
        results = _native_search(request.query, request.max_results)
    except ImportError as exc:  # pragma: no cover - 导入失败属于部署损坏
        logger.error("原生爬虫模块不可用: %s", exc)
        raise HTTPException(status_code=503, detail="native web search module unavailable") from exc
    except Exception as exc:
        logger.warning("网页搜索失败（query=%r）: %s", request.query, exc)
        raise HTTPException(status_code=502, detail=f"web search failed: {exc}") from exc

    sources = [
        WebSearchSource(
            url=r.url,
            title=r.title if r.title else None,
            snippet=r.snippet if r.snippet else None,
            engine=r.source if r.source else None,
        )
        for r in results
    ]
    # web.search 返回的就是截断后的列表；是否发生了截断这里不猜（引擎竞速
    # 只返回一个引擎的结果集），seam 侧仍有自己的 maxResults 上限兜底。
    return WebSearchResponse(sources=sources, truncated=False)
