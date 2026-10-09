/**
 * `ScraperSearchProvider`: a credential-free `WebSearchProvider` that fetches a
 * search engine's result page and parses it into normalized sources. It serves
 * deployments that must not depend on a paid search API: the engine is a public
 * HTML endpoint, so `available()` never involves a key and every request is a
 * plain anonymous GET.
 * @module @taiji/dsh-web-search-scraper/provider
 */

import { WebError } from '@taiji/dsh-web'
import type {
  WebSearchProvider,
  WebSearchRequest,
  WebSearchResult,
  WebSearchSource,
} from '@taiji/dsh-web'

/** Stable id this provider registers under. */
export const SCRAPER_PROVIDER_ID = 'scraper'

/** Default engine. Bing ships first: its result page is reachable from
 * mainland-China networks without a proxy and its markup is server-rendered
 * and stable, while the DuckDuckGo HTML endpoint is blocked there. */
export const SCRAPER_DEFAULT_ENGINE: ScraperEngine = 'bing'

/** Default Bing endpoint base; `/search` is the operation. */
export const BING_DEFAULT_BASE_URL = 'https://www.bing.com'

/** Default DuckDuckGo endpoint base; `/html/` is the operation. */
export const DUCKDUCKGO_DEFAULT_BASE_URL = 'https://html.duckduckgo.com'

/** Attribution header sent on every request. Bump with the package version. */
const USER_AGENT = 'deepseek-harness/0.0.1'

/** A real browser UA is required for the HTML endpoints to serve full markup. */
const BROWSER_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'

/** Engines the provider can parse. The plugin schema pins this union. */
export type ScraperEngine = 'bing' | 'duckduckgo'

/** Bing caps `count` per request; values above it are clamped. */
const BING_MAX_COUNT = 30

/** Resolved provider options (the plugin's `apply` supplies constant defaults). */
export interface ScraperSearchProviderOptions {
  /** Which engine's result page to fetch and parse. */
  engine: ScraperEngine
  /** Bing endpoint base; `/search` is appended. */
  bingBaseUrl: string
  /** DuckDuckGo endpoint base; `/html/` is appended. */
  duckduckgoBaseUrl: string
}

/**
 * Decode the HTML entities a search page actually emits: the named set seen in
 * Bing/DuckDuckGo markup plus numeric references. Unknown named entities pass
 * through unchanged rather than guessing.
 *
 * @param text - raw text possibly containing entities.
 * @returns decoded text.
 */
export function decodeHtmlEntities(text: string): string {
  return text.replace(/&(?:#[xX]([0-9a-fA-F]+)|#([0-9]+)|([a-zA-Z][a-zA-Z0-9]*));/gu, (match, hex, dec, name) => {
    if (hex !== undefined) return safeCodePoint(parseInt(hex, 16))
    if (dec !== undefined) return safeCodePoint(parseInt(dec, 10))
    const named = NAMED_ENTITIES[name]
    return named ?? match
  })
}

const NAMED_ENTITIES: Record<string, string> = {
  amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ',
  ensp: '\u2002', emsp: '\u2003', hellip: '\u2026', middot: '\u00b7',
  mdash: '\u2014', ndash: '\u2013', laquo: '\u00ab', raquo: '\u00bb',
}

/** Map a code point, rejecting surrogate-range garbage to the replacement char. */
function safeCodePoint(code: number): string {
  if (!Number.isFinite(code) || code < 0 || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) return '\ufffd'
  return String.fromCodePoint(code)
}

/**
 * Remove tags from one markup fragment and collapse the whitespace the removed
 * tags leave behind, so nested `<strong>`/`<b>` spans inside titles and
 * snippets yield plain text.
 *
 * @param fragment - one element's inner markup.
 * @returns tag-free, whitespace-collapsed text (NOT entity-decoded).
 */
export function stripHtmlTags(fragment: string): string {
  return fragment.replace(/<[^>]*>/gu, ' ').replace(/\s+/gu, ' ').trim()
}

/** True for href values that cannot name a citeable destination. */
function unusableHref(href: string): boolean {
  return href.length === 0 || /^(?:javascript|data|about|blob):/iu.test(href)
}

/** Normalize one extracted URL, or `undefined` when it is not citeable. */
function normalizeSourceUrl(href: string): string | undefined {
  if (unusableHref(href)) return undefined
  return href
}

/**
 * Resolve a DuckDuckGo redirect link to its destination. The HTML endpoint
 * wraps results in `//duckduckgo.com/l/?uddg=<encoded>&rut=...`; a direct href
 * passes through unchanged.
 *
 * @param href - the raw `result__a` href.
 * @returns the destination URL, or `undefined` when the redirect carries none.
 */
export function resolveDuckDuckGoHref(href: string): string | undefined {
  if (!/^(?:https?:)?\/\/[^\s]*duckduckgo\.com\/l\//iu.test(href) && !href.startsWith('/l/')) return normalizeSourceUrl(href)
  const absolute = href.startsWith('//') ? `https:${href}` : href.startsWith('/') ? `https://duckduckgo.com${href}` : href
  try {
    const uddg = new URL(absolute).searchParams.get('uddg')
    if (uddg === null || uddg.length === 0) return undefined
    return normalizeSourceUrl(uddg)
  } catch {
    return undefined
  }
}

/**
 * Parse a Bing result page into sources. One source per `li.b_algo` block: the
 * `h2` anchor names the URL and title, and a `p.b_lineclamp*` paragraph — when
 * the block carries one — becomes the snippet. Duplicate URLs collapse to
 * their first occurrence; blocks without a usable anchor are dropped.
 *
 * @param html - the fetched result-page markup.
 * @returns the sources in page order.
 */
export function mapBingHtml(html: string): WebSearchSource[] {
  const sources: WebSearchSource[] = []
  const seen = new Set<string>()
  let cursor = html.indexOf('<li class="b_algo')
  while (cursor >= 0) {
    const next = html.indexOf('<li class="b_algo', cursor + 1)
    const block = html.slice(cursor, next === -1 ? undefined : next)
    const source = mapBingBlock(block)
    if (source !== undefined && !seen.has(source.url)) {
      seen.add(source.url)
      sources.push(source)
    }
    cursor = next
  }
  return sources
}

/** Parse one `li.b_algo` block, or `undefined` when it names no usable result. */
function mapBingBlock(block: string): WebSearchSource | undefined {
  const h2Start = block.indexOf('<h2')
  if (h2Start === -1) return undefined
  const h2End = block.indexOf('</h2>', h2Start)
  if (h2End === -1) return undefined
  const heading = block.slice(h2Start, h2End + 5)
  const href = extractAnchorHref(heading)
  if (href === undefined) return undefined
  const url = normalizeSourceUrl(href)
  if (url === undefined) return undefined
  const title = stripHtmlTags(decodeHtmlEntities(heading.replace(/<a[\s][^>]*>/u, '').replace(/<\/a>/u, '')))
  const snippet = extractBingSnippet(block)
  return {
    url,
    ...title.length > 0 ? { title } : {},
    ...snippet !== undefined ? { snippet } : {},
  }
}

/** First `href="..."` on the first `<a` inside the fragment. */
function extractAnchorHref(fragment: string): string | undefined {
  const anchorStart = fragment.indexOf('<a')
  if (anchorStart === -1) return undefined
  const tagEnd = fragment.indexOf('>', anchorStart)
  if (tagEnd === -1) return undefined
  const tag = fragment.slice(anchorStart, tagEnd)
  const href = /(?:\s)href="([^"]*)"/u.exec(tag)?.[1]
  return href === undefined ? undefined : decodeHtmlEntities(href)
}

/** The `p.b_lineclamp*` paragraph's text, or `undefined` when absent. */
function extractBingSnippet(block: string): string | undefined {
  const marker = '<p class="b_lineclamp'
  const start = block.indexOf(marker)
  if (start === -1) return undefined
  const textStart = block.indexOf('>', start)
  const end = block.indexOf('</p>', start)
  if (textStart === -1 || end === -1 || end <= textStart) return undefined
  const text = stripHtmlTags(decodeHtmlEntities(block.slice(textStart + 1, end)))
  return text.length > 0 ? text : undefined
}

/**
 * Parse a DuckDuckGo HTML-endpoint page into sources. Results are the
 * `a.result__a` links; each pairs with the first `a.result__snippet` that
 * follows it and precedes the next result link. Redirect hrefs resolve to
 * their `uddg` destination; duplicate URLs collapse to their first occurrence.
 *
 * @param html - the fetched result-page markup.
 * @returns the sources in page order.
 */
export function mapDuckDuckGoHtml(html: string): WebSearchSource[] {
  const sources: WebSearchSource[] = []
  const seen = new Set<string>()
  const anchors = [...html.matchAll(/<a\s[^>]*class="[^"]*result__a[^"]*"[^>]*>/gu)]
  for (const [index, match] of anchors.entries()) {
    const anchorStart = match.index
    const anchorTagEnd = html.indexOf('>', anchorStart)
    const anchorEnd = html.indexOf('</a>', anchorTagEnd)
    if (anchorTagEnd === -1 || anchorEnd === -1) continue
    const href = /(?:\s)href="([^"]*)"/u.exec(match[0])?.[1]
    if (href === undefined) continue
    const url = resolveDuckDuckGoHref(decodeHtmlEntities(href))
    if (url === undefined || seen.has(url)) continue
    const windowEnd = index + 1 < anchors.length ? anchors[index + 1]!.index : html.length
    const title = stripHtmlTags(decodeHtmlEntities(html.slice(anchorTagEnd + 1, anchorEnd)))
    const snippetStart = html.indexOf('result__snippet', anchorEnd)
    const snippet = snippetStart >= 0 && snippetStart < windowEnd
      ? extractDuckDuckGoSnippet(html, snippetStart)
      : undefined
    seen.add(url)
    sources.push({
      url,
      ...title.length > 0 ? { title } : {},
      ...snippet !== undefined ? { snippet } : {},
    })
  }
  return sources
}

/** Text of the `a.result__snippet` element starting at `markerIndex`. */
function extractDuckDuckGoSnippet(html: string, markerIndex: number): string | undefined {
  const tagEnd = html.indexOf('>', markerIndex)
  const end = html.indexOf('</a>', tagEnd)
  if (tagEnd === -1 || end === -1 || end <= tagEnd) return undefined
  const text = stripHtmlTags(decodeHtmlEntities(html.slice(tagEnd + 1, end)))
  return text.length > 0 ? text : undefined
}

/** Build the engine request URL for one query. */
function endpointFor(options: ScraperSearchProviderOptions, query: string, maxResults: number | undefined): string {
  const encoded = encodeURIComponent(query)
  if (options.engine === 'duckduckgo') {
    return `${options.duckduckgoBaseUrl}/html/?q=${encoded}`
  }
  const count = maxResults !== undefined ? `&count=${String(Math.min(maxResults, BING_MAX_COUNT))}` : ''
  return `${options.bingBaseUrl}/search?q=${encoded}${count}`
}

/**
 * The scraper-backed search provider. Requests are anonymous GETs, so unlike
 * the credentialed providers they follow redirects (the packages-wide
 * reject-redirect rule protects credentials and request data, of which this
 * provider carries none into the redirect chain). Failures name the endpoint
 * and tell the model how the user can switch engines.
 */
export class ScraperSearchProvider implements WebSearchProvider {
  readonly id = SCRAPER_PROVIDER_ID

  constructor(private readonly resolveOptions: () => ScraperSearchProviderOptions) {}

  available(): boolean {
    const options = this.resolveOptions()
    return URL.canParse(`${options.bingBaseUrl}/search`) && URL.canParse(`${options.duckduckgoBaseUrl}/html/`)
  }

  async search(request: WebSearchRequest, signal?: AbortSignal): Promise<WebSearchResult> {
    // One snapshot per operation so an engine switch landing mid-search cannot
    // mix two engines' base URLs into one request.
    const options = this.resolveOptions()
    throwIfSearchAborted(signal)
    const endpoint = endpointFor(options, request.query, request.maxResults)
    let response: Response
    try {
      response = await fetch(endpoint, {
        headers: {
          'user-agent': BROWSER_USER_AGENT,
          'accept': 'text/html,application/xhtml+xml',
          'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
          // The attribution header stays beside the browser UA; some engines
          // drop requests whose only UA names a script.
          'x-harness-client': USER_AGENT,
        },
        ...signal !== undefined ? { signal } : {},
      })
    } catch (error: unknown) {
      if (signal?.aborted === true || isAbortError(error)) throw searchAborted(signal, error)
      throw endpointError(options, `Scraper search request failed: ${String(error)}`, error)
    }
    if (!response.ok) {
      throw endpointError(options, `Scraper search engine returned HTTP ${String(response.status)}`)
    }
    try {
      const html = await response.text()
      const sources = options.engine === 'duckduckgo' ? mapDuckDuckGoHtml(html) : mapBingHtml(html)
      // An empty page is a legitimate empty result (engine found nothing or
      // served a layout we do not recognize), not a provider error.
      return { sources, truncated: false }
    } catch (error: unknown) {
      if (signal?.aborted === true || isAbortError(error)) throw searchAborted(signal, error)
      throw endpointError(options, `Scraper search could not process the result page: ${String(error)}`, error)
    }
  }
}

/** Add engine-recovery instructions to failures that occur around dispatch. */
function endpointError(options: ScraperSearchProviderOptions, message: string, cause?: unknown): WebError {
  return new WebError(
    `${message}\n\nThe web search request used the ${options.engine} result page. `
    + 'If that engine is unreachable from this deployment (the DuckDuckGo HTML endpoint is blocked '
    + 'in mainland-China networks; Bing is not), guide the user to Settings > Plugins > '
    + 'Plugin configuration > Web search scraper, where they can switch Engine. Only the user '
    + 'should choose or change the engine.',
    'WEB_PROVIDER_ERROR',
    cause === undefined ? undefined : { cause },
  )
}

/** Throw the provider's stable cancellation error when the caller already aborted. */
function throwIfSearchAborted(signal?: AbortSignal): void {
  if (signal?.aborted === true) throw searchAborted(signal)
}

/** Build the provider's stable cancellation error while retaining the caller's reason. */
function searchAborted(signal?: AbortSignal, fallback?: unknown): WebError {
  return new WebError('Scraper search aborted', 'WEB_ABORTED', {
    cause: signal?.aborted === true ? signal.reason : fallback,
  })
}

/** True for a fetch/`AbortSignal` abort, surfaced as `WEB_ABORTED`. */
function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}
