---
description: "ctx.web 的 Taiji 运行时搜索提供方：项目自带的多引擎竞速爬虫经本地 api 服务执行，无需任何搜索 API 凭证。"
kind: "package-reference"
---

# @taiji/dsh-web-search-scraper

[English](README.md) | 中文

## Summary

有了 `dsh-web-search-scraper`，harness 通过项目自带的原生爬虫搜索网页：提供方向 Taiji 运行时的 `POST /api/tools/web_search` 发起请求，由运行时的多引擎竞速爬虫（`neuroplex/tools/web.py`：DuckDuckGo / Bing / Baidu，重试退避、缓存、正文提取）返回来源。全程不涉及付费搜索 API，也不需要存储任何凭证。结果不带 `content`，只有来源。模型侧的 `web_search` 工具在 `dsh-tool-web`。

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Further Exploration](#further-exploration)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>
## Use this package

把提供方挂进已加载 web 服务的组合；它以 `taiji-search` 之名注册为搜索提供方。当它是唯一可用的搜索后端时 `ctx.web.search()` 自动选中它，也可以用 `searchProvider: taiji-search` 显式钉住。

### When to choose it

部署运行 Taiji 运行时（Seed 后端服务）并希望搜索由项目自有爬虫驱动时选这个后端。提供方完全不需要密钥：只要配置的运行时地址可解析，`available()` 即为真。运行时只有在后端进程存活时才会应答——对已下线运行时的搜索会以指名恢复路径的提供方错误失败。

### Minimal configuration

加载 web 服务与提供方即可；运行时地址默认指向本地 api 服务。

```yaml
- name: '@taiji/dsh-web'
  config:
    searchProvider: taiji-search
- name: '@taiji/dsh-web-search-scraper'
```

| Field | Default | Meaning |
|---|---|---|
| `baseURL` | `http://127.0.0.1:8000` | Taiji 运行时基址；追加 `/api/tools/web_search`。与 Taiji 聊天路由使用同一默认值 |

生成的[配置目录](../../../docs/config-catalog.md#taijidsh-web-search-scraper)是所有字段及其 JSDoc 的穷尽来源。

### What a search returns

运行时爬虫对多引擎竞速并返回其中一个引擎的结果列表；每行映射为一个 `WebSearchSource`，含 `url`、`title`、`snippet`（空白串丢弃）。爬虫摘要无法得出 `publishedAt`，也不生成 `content`。提供方把 `maxResults` 以请求的 `max_results` 透传以在源头控制结果量；缝隙自身的上限仍然生效。搜不到东西按空结果解析，而不是错误。

### Failures and recovery

传输失败（运行时未启动、HTTP 状态、无法处理的响应体）以 `WEB_PROVIDER_ERROR` 浮出，指名运行时端点，并引导模型指引用户启动后端，或在 Settings > Plugins > Plugin configuration > Web search scraper 修改配置的基址。取消以 `WEB_ABORTED` 浮出。

<a id="understand-the-implementation"></a>
## Understand the implementation

- `provider.ts` 拥有整个表面：运行时请求、响应包络规范化（`mapRuntimeResponse`）和提供方类。本侧不做任何 HTML 解析——引擎与其页面结构归爬虫所有。
- 请求是匿名 POST，有意跟随重定向：全仓的拒绝重定向规则保护的是凭证和请求数据，而本提供方在重定向链中不携带任何此类数据。
- 提供方在每次搜索入口做一次选项快照，搜索中途落下的基址变更不会把同一请求的两部分发往两个运行时。

<a id="further-exploration"></a>
## Further Exploration

- [`@taiji/dsh-web`](../web/README.md) — 能力缝隙、提供方选择与规范化结果类型。
- `neuroplex/tools/web.py`（仓库根）— 本提供方委托的原生多引擎竞速爬虫。
- [`@taiji/dsh-tool-web`](../tool-web/README.md) — 模型侧的 `web_search` / `web_fetch` 工具。

<a id="model-experience"></a>
## Model Experience

模型像使用任何提供方一样调用 `web_search`；工具 schema 中不会出现这个后端的名字。结果以标准的 markdown 来源列表渲染，不带日期。失败时错误文本指名运行时端点与设置页，模型因此可以引导用户启动后端或修正基址。

<a id="known-limitations-and-deferred-work"></a>
## Known Limitations and Deferred Work

- 搜索要求 Taiji 运行时（Seed 后端）在运行；运行时下线时所有搜索都会失败，直至重新启动。
- 原生爬虫每次查询只返回一个引擎的结果（竞速赢家），其百度引擎在服务端 HTTP 客户端下常被反爬拦截——大陆网络下的有效覆盖依赖 Bing。
- 无 `publishedAt`：爬虫摘要不携带日期，猜测即撒谎。

<a id="dev-note"></a>
## Dev Note

- 上面的跟随重定向选择是有意为之，并通过 egress spec（代理透明性）测试。
- `tests/scraper.spec.ts` 用本地运行时替身钉住包络规范化与提供方；真实 smoke（`$DSH_WEB_SCRAPER_E2E`）还要求运行时就绪探针应答。
