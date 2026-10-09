/**
 * Scraper-backed `WebSearchProvider` plugin: fetches and parses a search
 * engine's result page (no API key, no paid quota). It contributes to the
 * `ctx.web` registry without owning the service.
 * @module @taiji/dsh-web-search-scraper
 */

import type { Context } from '@taiji/cordis'
import z from '@taiji/schemastery'
import type {} from '@taiji/dsh-web'
import {
  BING_DEFAULT_BASE_URL,
  DUCKDUCKGO_DEFAULT_BASE_URL,
  SCRAPER_DEFAULT_ENGINE,
  ScraperSearchProvider,
} from './provider.ts'
import type { ScraperEngine } from './provider.ts'

export {
  BING_DEFAULT_BASE_URL,
  DUCKDUCKGO_DEFAULT_BASE_URL,
  SCRAPER_DEFAULT_ENGINE,
  SCRAPER_PROVIDER_ID,
  ScraperSearchProvider,
} from './provider.ts'
export type { ScraperEngine, ScraperSearchProviderOptions } from './provider.ts'

/** Cordis plugin name used by loader diagnostics. */
export const name = 'web-search-scraper'

/** The web seam this provider registers into. */
export const inject = ['web']

/** Plugin config (all optional — `apply` fills constant defaults). */
export interface Config {
  /** Which engine's result page to fetch and parse. Defaults to Bing. */
  engine: ScraperEngine
  /** Bing endpoint base; `/search` is appended. */
  bingBaseUrl: string
  /** DuckDuckGo endpoint base; `/html/` is appended. */
  duckduckgoBaseUrl: string
}

export const Config: z<Config> = z.object({
  engine: z.union(['bing', 'duckduckgo'] as const).default(SCRAPER_DEFAULT_ENGINE),
  bingBaseUrl: z.string().default(BING_DEFAULT_BASE_URL),
  duckduckgoBaseUrl: z.string().default(DUCKDUCKGO_DEFAULT_BASE_URL),
})

/** Register the scraper search provider with `ctx.web`. */
export function apply(ctx: Context, config: Config): void {
  ctx.web.registerSearchProvider(new ScraperSearchProvider(() => ({
    engine: config.engine,
    bingBaseUrl: config.bingBaseUrl,
    duckduckgoBaseUrl: config.duckduckgoBaseUrl,
  })))
}
