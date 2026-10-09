/**
 * Taiji-runtime-backed `WebSearchProvider` plugin: the model's `web_search`
 * calls the local api service's native scraper search (the project's own
 * multi-engine racing crawler), needing no search API credential. It
 * contributes to the `ctx.web` registry without owning the service.
 * @module @taiji/dsh-web-search-scraper
 */

import type { Context } from '@taiji/cordis'
import z from '@taiji/schemastery'
import type {} from '@taiji/dsh-web'
import {
  TAIJI_RUNTIME_DEFAULT_BASE_URL,
  TaijiSearchProvider,
} from './provider.ts'

export {
  TAIJI_RUNTIME_DEFAULT_BASE_URL,
  TAIJI_SEARCH_PATH,
  TAIJI_SEARCH_PROVIDER_ID,
  TaijiSearchProvider,
  mapRuntimeResponse,
  mapRuntimeSource,
} from './provider.ts'
export type { TaijiSearchProviderOptions } from './provider.ts'

/** Cordis plugin name used by loader diagnostics. */
export const name = 'web-search-scraper'

/** The web seam this provider registers into. */
export const inject = ['web']

/** Plugin config (all optional — `apply` fills constant defaults). */
export interface Config {
  /** Taiji runtime root; `/api/tools/web_search` is appended. Defaults to the local api service. */
  baseURL: string
}

export const Config: z<Config> = z.object({
  baseURL: z.string().default(TAIJI_RUNTIME_DEFAULT_BASE_URL),
})

/** Register the Taiji-runtime search provider with `ctx.web`. */
export function apply(ctx: Context, config: Config): void {
  ctx.web.registerSearchProvider(new TaijiSearchProvider(() => ({
    baseURL: config.baseURL,
  })))
}
