"""routes_web_search 契约：端点把 neuroplex 原生爬虫结果映射为 harness 对齐的 sources。"""

from fastapi.testclient import TestClient

from api.app import create_app
from neuroplex.tools.web import SearchResult


def _client() -> TestClient:
    return TestClient(create_app(startup_tasks=False))


def test_web_search_maps_native_results(monkeypatch) -> None:
    from api import routes_web_search

    def fake_search(query: str, max_results: int = 5) -> list[SearchResult]:
        assert query == "python async"
        assert max_results == 3
        return [
            SearchResult(
                title="Alpha", url="https://example.com/a", snippet="first", source="Bing"
            ),
            SearchResult(title="", url="https://example.com/b", snippet="", source="Baidu"),
        ]

    monkeypatch.setattr(routes_web_search, "_native_search", fake_search)
    response = _client().post(
        "/api/tools/web_search", json={"query": "python async", "max_results": 3}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["truncated"] is False
    assert body["sources"] == [
        {"url": "https://example.com/a", "title": "Alpha", "snippet": "first", "engine": "Bing"},
        {"url": "https://example.com/b", "title": None, "snippet": None, "engine": "Baidu"},
    ]


def test_web_search_reports_engine_failure_as_bad_gateway(monkeypatch) -> None:
    from api import routes_web_search

    def boom(query: str, max_results: int = 5) -> list[SearchResult]:
        raise RuntimeError("all engines failed")

    monkeypatch.setattr(routes_web_search, "_native_search", boom)
    response = _client().post("/api/tools/web_search", json={"query": "anything"})

    assert response.status_code == 502
    assert "all engines failed" in response.json()["detail"]


def test_web_search_rejects_blank_query() -> None:
    response = _client().post("/api/tools/web_search", json={"query": "   "})
    assert response.status_code == 422
