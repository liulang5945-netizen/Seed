---
description: "The native scraper search provider for ctx.web: credential-free web search that fetches and parses a search engine's result page."
kind: "package-reference"
---

# @taiji/dsh-web-search-scraper

English | [中文](README.zh.md)

## Summary

With `dsh-web-search-scraper`, the harness searches the web without any search API: the provider fetches a search engine's result page over plain HTTP and parses the result blocks into citeable sources. It serves deployments that must not depend on a paid search API or a stored credential. The engine is a deployment choice — Bing by default, with the DuckDuckGo HTML endpoint as the alternative — and the provider carries no `content`, only sources. The model-facing `web_search` tool lives in `dsh-tool-web`.

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

Mount the provider in a composition that already loads the web service; it registers as the `scraper` search provider, so `ctx.web.search()` resolves it automatically when it is the only usable search backend — or pin it with `searchProvider: scraper`.

### When to choose it

Choose this backend when a deployment has no search API credential and still needs live web results. The provider needs no key at all: `available()` is true whenever the configured endpoint bases parse. The default engine (Bing) is reachable from mainland-China networks without a proxy; the DuckDuckGo HTML endpoint is not, so deployments behind such networks should keep `bing`.

### Minimal configuration

Load the web service and the provider; every setting has a safe default.

```yaml
- name: '@taiji/dsh-web'
  config:
    searchProvider: scraper
- name: '@taiji/dsh-web-search-scraper'
```

| Field | Default | Meaning |
|---|---|---|
| `engine` | `bing` | Which result page to fetch and parse: `bing` or `duckduckgo` |
| `bingBaseUrl` | `https://www.bing.com` | Bing endpoint base; `/search` is appended |
| `duckduckgoBaseUrl` | `https://html.duckduckgo.com` | DuckDuckGo endpoint base; `/html/` is appended |

The generated [configuration catalog](../../../docs/config-catalog.md#taijidsh-web-search-scraper) is the exhaustive source for every accepted field and its JSDoc.

### What a search returns

Each parsed result maps to a `WebSearchSource`: `url`, `title`, and — when the engine's block carries a snippet paragraph — `snippet`. Bing blocks contribute their `p.b_lineclamp*` paragraph; DuckDuckGo redirect links resolve to their `uddg` destination before they are returned. Duplicate URLs collapse to their first occurrence, and a block without a usable anchor is dropped, so a call can return fewer sources than requested. The provider never sets `publishedAt` (the engines' snippets may carry an age prefix such as "1 天前 ·" inline) and never sets `content`. A page without recognizable result blocks resolves as an empty result, not an error.

### Failures and recovery

Transport failures (DNS, connect, HTTP status) surface as `WEB_PROVIDER_ERROR` naming the engine and pointing the user at Settings > Plugins > Plugin configuration > Web search scraper, where Engine is switchable. Cancellation surfaces as `WEB_ABORTED`. A layout change that breaks the parser degrades to zero sources ("No results found." at the tool layer) rather than a hard error — treat a sudden run of empty results as an engine-markup signal, not a query problem.

<a id="understand-the-implementation"></a>
## Understand the implementation

- `provider.ts` owns the whole surface: the two parsers (`mapBingHtml`, `mapDuckDuckGoHtml`), entity decoding, redirect resolution, and the provider class. Parsing is dependency-free index slicing, so the package has no runtime dependencies beyond the schema.
- Requests are anonymous GETs with a browser `User-Agent` (engines drop script-only UAs) plus the harness attribution header. They intentionally follow redirects: the packages-wide reject-redirect rule protects credentials and request data, of which this provider carries none into the redirect chain (Bing redirects `www.bing.com` to a regional host by geography).
- The provider snapshots its options once per search, so an engine switch landing mid-search cannot mix two engines' base URLs into one request.

<a id="further-exploration"></a>
## Further Exploration

- [`@taiji/dsh-web`](../web/README.md) — the capability seam, provider selection, and the normalized result types.
- [`@taiji/dsh-web-search-deepseek`](../web-search-deepseek/README.md) — the credential-bearing native-search alternative.
- [`@taiji/dsh-tool-web`](../tool-web/README.md) — the model-facing `web_search` / `web_fetch` tools.

<a id="model-experience"></a>
## Model Experience

The model calls `web_search` as with any provider; nothing in the tool schema names this backend. Results render as the standard markdown source list; snippets may carry the engine's inline age prefix verbatim. On engine failure the error text names the engine and the settings page, so the model can guide the user to switch engines.

<a id="known-limitations-and-deferred-work"></a>
## Known Limitations and Deferred Work

- Engine result pages are not a contract: a markup change can silently reduce results to zero. The e2e smoke (`$DSH_WEB_SCRAPER_E2E`) is the early-warning tripwire, not a guarantee.
- Scraped snippets are shorter and noisier than API search snippets, and no `publishedAt` is derivable without guessing.
- The DuckDuckGo HTML endpoint is unreachable from mainland-China networks without a proxy; Bing is the default for that reason.
- Heavy automated use of a public search engine can trigger anti-bot challenges; this provider is sized for conversational search traffic, not bulk crawling.

<a id="dev-note"></a>
## Dev Note

- The redirect-following choice above is deliberate and tested through the egress spec (proxy transparency), matching `web-search-exa`'s coverage shape.
- `tests/scraper.spec.ts` pins the parser against fixtures trimmed from real engine markup, including the entity/whitespace shapes that motivated `decodeHtmlEntities` + `stripHtmlTags` ordering (decode first, then strip and collapse).
