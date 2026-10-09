---
description: "The Taiji-runtime-backed search provider for ctx.web: the project's own multi-engine racing crawler behind the local api service, with no search API credential."
kind: "package-reference"
---

# @taiji/dsh-web-search-scraper

English | [中文](README.zh.md)

## Summary

With `dsh-web-search-scraper`, the harness searches the web through the project's own native crawler: the provider POSTs to the Taiji runtime's `POST /api/tools/web_search`, and the runtime's multi-engine racing crawler (`neuroplex/tools/web.py`: DuckDuckGo / Bing / Baidu, retry with backoff, caching, readability extraction) returns the sources. No paid search API and no stored credential is involved. The provider carries no `content`, only sources. The model-facing `web_search` tool lives in `dsh-tool-web`.

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

Mount the provider in a composition that already loads the web service; it registers as the `taiji-search` search provider, so `ctx.web.search()` resolves it automatically when it is the only usable search backend — or pin it with `searchProvider: taiji-search`.

### When to choose it

Choose this backend when the deployment runs the Taiji runtime (the Seed backend service) and wants search driven by the project's own crawler. The provider needs no key: `available()` is true whenever the configured runtime root parses. The runtime answers only when the backend process is up — a search against a down runtime fails with a provider error that names the recovery path.

### Minimal configuration

Load the web service and the provider; the runtime root defaults to the local api service.

```yaml
- name: '@taiji/dsh-web'
  config:
    searchProvider: taiji-search
- name: '@taiji/dsh-web-search-scraper'
```

| Field | Default | Meaning |
|---|---|---|
| `baseURL` | `http://127.0.0.1:8000` | Taiji runtime root; `/api/tools/web_search` is appended. The same default the Taiji chat route uses |

The generated [configuration catalog](../../../docs/config-catalog.md#taijidsh-web-search-scraper) is the exhaustive source for every accepted field and its JSDoc.

### What a search returns

The runtime's crawler races its engines and returns one engine's result list; each row maps to a `WebSearchSource` with `url`, `title`, and `snippet` (blank strings drop). No `publishedAt` is derivable from the crawler's snippets, and no `content` is generated. The provider enforces `maxResults` by passing it through as the request's `max_results`; the seam's own cap still applies. A search that finds nothing resolves as an empty result, not an error.

### Failures and recovery

Transport failures (runtime down, HTTP status, unprocessable body) surface as `WEB_PROVIDER_ERROR` naming the runtime endpoint and telling the model to guide the user toward starting the backend or changing the configured root under Settings > Plugins > Plugin configuration > Web search scraper. Cancellation surfaces as `WEB_ABORTED`.

<a id="understand-the-implementation"></a>
## Understand the implementation

- `provider.ts` owns the whole surface: the runtime request, envelope normalization (`mapRuntimeResponse`), and the provider class. No HTML parsing happens on this side — the crawler owns the engines and their markup.
- Requests are anonymous POSTs and intentionally follow redirects: the packages-wide reject-redirect rule protects credentials and request data, of which this provider carries none into the redirect chain.
- The provider snapshots its options once per search, so a base-URL change landing mid-search cannot send one request's parts to two runtimes.

<a id="further-exploration"></a>
## Further Exploration

- [`@taiji/dsh-web`](../web/README.md) — the capability seam, provider selection, and the normalized result types.
- `neuroplex/tools/web.py` (repository root) — the native multi-engine racing crawler this provider delegates to.
- [`@taiji/dsh-tool-web`](../tool-web/README.md) — the model-facing `web_search` / `web_fetch` tools.

<a id="model-experience"></a>
## Model Experience

The model calls `web_search` as with any provider; nothing in the tool schema names this backend. Results render as the standard markdown source list without dates. On failure the error text names the runtime endpoint and the settings page, so the model can guide the user to start the backend or fix the root.

<a id="known-limitations-and-deferred-work"></a>
## Known Limitations and Deferred Work

- Searches require the Taiji runtime (Seed backend) to be running; a down runtime makes every search fail until it is started.
- The native crawler returns one engine's results per query (the racing winner), and its Baidu engine is routinely blocked by anti-bot challenges from server-side HTTP clients — effective coverage on mainland-China networks rests on Bing.
- No `publishedAt`: the crawler's snippets carry no dates, and guessing would lie.

<a id="dev-note"></a>
## Dev Note

- The redirect-following choice above is deliberate and tested through the egress spec (proxy transparency).
- `tests/scraper.spec.ts` pins envelope normalization and the provider against a local runtime double; the live smoke (`$DSH_WEB_SCRAPER_E2E`) additionally requires the runtime's readiness probe to answer.
