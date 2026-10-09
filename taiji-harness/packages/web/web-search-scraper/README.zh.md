---
description: "ctx.web 的原生爬虫搜索提供方：零凭证，直接抓取并解析搜索引擎结果页的网页搜索。"
kind: "package-reference"
---

# @taiji/dsh-web-search-scraper

[English](README.md) | 中文

## Summary

有了 `dsh-web-search-scraper`，harness 不依赖任何搜索 API 即可搜索网页：提供方用普通 HTTP 抓取搜索引擎的结果页，把结果块解析为可引用的来源。它面向不希望依赖付费搜索 API 或存储凭证的部署。引擎是部署选择——默认 Bing，备选 DuckDuckGo HTML 端点——结果不带 `content`，只有来源。模型侧的 `web_search` 工具在 `dsh-tool-web`。

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

把提供方挂进已加载 web 服务的组合；它以 `scraper` 之名注册为搜索提供方。当它是唯一可用的搜索后端时 `ctx.web.search()` 自动选中它，也可以用 `searchProvider: scraper` 显式钉住。

### When to choose it

部署没有搜索 API 凭证但仍需要实时网页结果时选这个后端。提供方完全不需要密钥：只要配置的端点地址可解析，`available()` 即为真。默认引擎（Bing）在中国大陆网络无需代理即可访问；DuckDuckGo HTML 端点则不行，所以处于此类网络的部署应保持 `bing`。

### Minimal configuration

加载 web 服务与提供方即可；所有设置都有安全默认值。

```yaml
- name: '@taiji/dsh-web'
  config:
    searchProvider: scraper
- name: '@taiji/dsh-web-search-scraper'
```

| Field | Default | Meaning |
|---|---|---|
| `engine` | `bing` | 抓取并解析哪个结果页：`bing` 或 `duckduckgo` |
| `bingBaseUrl` | `https://www.bing.com` | Bing 端点基址；追加 `/search` |
| `duckduckgoBaseUrl` | `https://html.duckduckgo.com` | DuckDuckGo 端点基址；追加 `/html/` |

生成的[配置目录](../../../docs/config-catalog.md#taijidsh-web-search-scraper)是所有字段及其 JSDoc 的穷尽来源。

### What a search returns

每个解析出的结果映射为一个 `WebSearchSource`：`url`、`title`，以及——当引擎的结果块带有摘要段落时——`snippet`。Bing 的块取其 `p.b_lineclamp*` 段落；DuckDuckGo 的跳转链接在返回前解析到其 `uddg` 目的地。重复 URL 收敛到首次出现，没有可用锚点的块被丢弃，因此一次调用的来源数可能少于请求数。提供方从不设置 `publishedAt`（引擎摘要可能内联携带"1 天前 ·"之类的年龄前缀，保持原样），也从不设置 `content`。没有可识别结果块的页面按空结果解析，而不是错误。

### Failures and recovery

传输失败（DNS、连接、HTTP 状态）以 `WEB_PROVIDER_ERROR` 浮出，指名引擎并引导用户到 Settings > Plugins > Plugin configuration > Web search scraper，在那里切换 Engine。取消以 `WEB_ABORTED` 浮出。布局变化导致解析器失效时退化为零来源（工具层的"No results found."）而不是硬错误——突然连续出现空结果应视为引擎页面结构的信号，而非查询问题。

<a id="understand-the-implementation"></a>
## Understand the implementation

- `provider.ts` 拥有整个表面：两个解析器（`mapBingHtml`、`mapDuckDuckGoHtml`）、实体解码、跳转解析和提供方类。解析是零依赖的索引切片，因此包除 schema 外没有任何运行时依赖。
- 请求是匿名 GET，带浏览器 `User-Agent`（引擎会丢弃纯脚本的 UA）和 harness 归属头。它们有意跟随重定向：全仓的拒绝重定向规则保护的是凭证和请求数据，而本提供方在重定向链中不携带任何此类数据（Bing 会按地域把 `www.bing.com` 重定向到区域主机）。
- 提供方在每次搜索入口做一次选项快照，搜索中途落下的引擎切换不会把两个引擎的基址混进同一个请求。

<a id="further-exploration"></a>
## Further Exploration

- [`@taiji/dsh-web`](../web/README.md) — 能力缝隙、提供方选择与规范化结果类型。
- [`@taiji/dsh-web-search-deepseek`](../web-search-deepseek/README.md) — 带凭证的原生搜索替代。
- [`@taiji/dsh-tool-web`](../tool-web/README.md) — 模型侧的 `web_search` / `web_fetch` 工具。

<a id="model-experience"></a>
## Model Experience

模型像使用任何提供方一样调用 `web_search`；工具 schema 中不会出现这个后端的名字。结果以标准的 markdown 来源列表渲染；摘要可能逐字携带引擎的内联年龄前缀。引擎失败时，错误文本指名引擎与设置页，模型因此可以引导用户切换引擎。

<a id="known-limitations-and-deferred-work"></a>
## Known Limitations and Deferred Work

- 引擎结果页不是契约：页面结构变化可能静默地把结果归零。e2e smoke（`$DSH_WEB_SCRAPER_E2E`）是预警绊线，不是保证。
- 爬取的摘要比 API 搜索的更短更噪，且不猜就无法得出 `publishedAt`。
- 中国大陆网络不经代理无法访问 DuckDuckGo HTML 端点；因此默认 Bing。
- 对公共搜索引擎的高频自动化使用可能触发反爬挑战；本提供方按对话式搜索流量设计，不是批量爬取工具。

<a id="dev-note"></a>
## Dev Note

- 上面的跟随重定向选择是有意为之，并通过 egress spec（代理透明性）测试，对齐 `web-search-exa` 的覆盖形态。
- `tests/scraper.spec.ts` 用从真实引擎页面裁剪的 fixture 钉住解析器，包括促成"先解码再去标签并塌缩空白"（`decodeHtmlEntities` + `stripHtmlTags` 顺序）的实体/空白形态。
