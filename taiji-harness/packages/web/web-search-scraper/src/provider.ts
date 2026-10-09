/**
 * `TaijiSearchProvider`: a credential-free `WebSearchProvider` that delegates to
 * the Taiji runtime's native scraper search (`POST /api/tools/web_search` on the
 * local api service). The runtime runs the project's own multi-engine racing
 * crawler (`neuroplex/tools/web.py`: DuckDuckGo / Bing / Baidu), so the model's
 * searches never depend on a paid search API or a stored credential.
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
export const TAIJI_SEARCH_PROVIDER_ID = 'taiji-search'

/** Default Taiji runtime root: the same local api service the chat route uses. */
export const TAIJI_RUNTIME_DEFAULT_BASE_URL = 'http://127.0.0.1:8000'

/** The runtime's native-scraper search operation, appended to the configured root. */
export const TAIJI_SEARCH_PATH = '/api/tools/web_search'

/** Attribution header sent on every request. Bump with the package version. */
const USER_AGENT = 'deepseek-harness/0.0.1'

/** One source row of the runtime's `POST /api/tools/web_search` response. */
interface RuntimeSearchSource {
  readonly url?: unknown
  readonly title?: unknown
  readonly snippet?: unknown
  readonly engine?: unknown
}

/** The runtime's search response envelope. */
interface RuntimeSearchResponse {
  readonly sources?: unknown
  readonly truncated?: unknown
}

/** Resolved provider options (the plugin's `apply` supplies constant defaults). */
export interface TaijiSearchProviderOptions {
  /** Taiji runtime root; the search path is appended. */
  baseURL: string
}

/**
 * Normalize one runtime source row, or `undefined` when it names no usable URL.
 * Blank optional strings drop so the seam never carries empty-string lies.
 *
 * @param row - one `sources[]` entry as parsed from the runtime response.
 * @returns the normalized source, or `undefined` for an unusable row.
 */
export function mapRuntimeSource(row: RuntimeSearchSource): WebSearchSource | undefined {
  if (typeof row.url !== 'string' || row.url.length === 0) return undefined
  const title = typeof row.title === 'string' && row.title.trim().length > 0 ? row.title.trim() : undefined
  const snippet = typeof row.snippet === 'string' && row.snippet.trim().length > 0 ? row.snippet.trim() : undefined
  return {
    url: row.url,
    ...title !== undefined ? { title } : {},
    ...snippet !== undefined ? { snippet } : {},
  }
}

/**
 * Map the runtime's response envelope to a normalized search result. The
 * runtime's multi-engine racing crawler returns one engine's result list; an
 * empty page is a legitimate empty result, not an error.
 *
 * @param payload - the parsed response body.
 * @returns the normalized result.
 * @throws {@link WebError} when the envelope is not a recognizable shape.
 */
export function mapRuntimeResponse(payload: unknown): WebSearchResult {
  if (payload === null || typeof payload !== 'object' || !Array.isArray((payload as RuntimeSearchResponse).sources)) {
    throw new WebError('Taiji runtime returned an unrecognizable search response', 'WEB_PROVIDER_ERROR')
  }
  const sources: WebSearchSource[] = []
  for (const row of (payload as RuntimeSearchResponse).sources as readonly unknown[]) {
    if (row === null || typeof row !== 'object') continue
    const source = mapRuntimeSource(row as RuntimeSearchSource)
    if (source !== undefined) sources.push(source)
  }
  return { sources, truncated: false }
}

/**
 * The Taiji-runtime-backed search provider. Requests are anonymous POSTs to the
 * local api service, so unlike the credentialed providers they follow redirects
 * (the packages-wide reject-redirect rule protects credentials and request
 * data, of which this provider carries none into the redirect chain). Failures
 * tell the model the runtime may be down and name the recovery path.
 */
export class TaijiSearchProvider implements WebSearchProvider {
  readonly id = TAIJI_SEARCH_PROVIDER_ID

  constructor(private readonly resolveOptions: () => TaijiSearchProviderOptions) {}

  available(): boolean {
    const options = this.resolveOptions()
    return URL.canParse(`${options.baseURL}${TAIJI_SEARCH_PATH}`)
  }

  async search(request: WebSearchRequest, signal?: AbortSignal): Promise<WebSearchResult> {
    // One snapshot per operation so a base-URL change landing mid-search cannot
    // send one request's parts to two runtimes.
    const options = this.resolveOptions()
    throwIfSearchAborted(signal)
    const endpoint = `${options.baseURL}${TAIJI_SEARCH_PATH}`
    let response: Response
    try {
      response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'accept': 'application/json',
          'user-agent': USER_AGENT,
        },
        body: JSON.stringify({
          query: request.query,
          ...(request.maxResults !== undefined ? { max_results: request.maxResults } : {}),
        }),
        ...signal !== undefined ? { signal } : {},
      })
    } catch (error: unknown) {
      if (signal?.aborted === true || isAbortError(error)) throw searchAborted(signal, error)
      throw searchEndpointError(options, `Taiji search request failed: ${String(error)}`, error)
    }
    if (!response.ok) {
      throw searchEndpointError(options, `Taiji runtime search returned HTTP ${String(response.status)}`)
    }
    try {
      return mapRuntimeResponse(await response.json())
    } catch (error: unknown) {
      if (signal?.aborted === true || isAbortError(error)) throw searchAborted(signal, error)
      if (error instanceof WebError) throw error
      throw searchEndpointError(options, `Taiji runtime search returned an unprocessable body: ${String(error)}`, error)
    }
  }
}

/** Add runtime-recovery instructions to failures that occur around dispatch. */
function searchEndpointError(options: TaijiSearchProviderOptions, message: string, cause?: unknown): WebError {
  return new WebError(
    `${message}\n\nThe web search request used the Taiji runtime's native scraper search at `
    + `${JSON.stringify(options.baseURL)}${TAIJI_SEARCH_PATH}. If the Seed backend (local runtime) is not `
    + 'running, searches cannot execute — guide the user to start the backend. If a different runtime root '
    + 'is intended, guide the user to Settings > Plugins > Plugin configuration > Web search scraper, where '
    + 'they can change and save Endpoint. Only the user should choose or change the endpoint.',
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
  return new WebError('Taiji search aborted', 'WEB_ABORTED', {
    cause: signal?.aborted === true ? signal.reason : fallback,
  })
}

/** True for a fetch/`AbortSignal` abort, surfaced as `WEB_ABORTED`. */
function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}
