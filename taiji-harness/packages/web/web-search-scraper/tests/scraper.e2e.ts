import { describe, expect, it } from 'vitest'
import { BING_DEFAULT_BASE_URL, DUCKDUCKGO_DEFAULT_BASE_URL, SCRAPER_DEFAULT_ENGINE, ScraperSearchProvider } from '@taiji/dsh-web-search-scraper'

/**
 * Real-network smoke for the scraper provider. Self-skips without an explicit
 * `$DSH_WEB_SCRAPER_E2E` opt-in (CI and ordinary test runs must not depend on
 * a live search engine); the DuckDuckGo arm additionally needs network reach
 * to `html.duckduckgo.com` (blocked in mainland-China networks).
 */
const enabled = process.env.DSH_WEB_SCRAPER_E2E !== undefined && process.env.DSH_WEB_SCRAPER_E2E.length > 0
const ddgEnabled = enabled && process.env.DSH_WEB_SCRAPER_E2E_DDG !== undefined && process.env.DSH_WEB_SCRAPER_E2E_DDG.length > 0
const maybe = enabled ? describe : describe.skip
const maybeDdg = ddgEnabled ? describe : describe.skip

maybe('ScraperSearchProvider real engine (bing)', () => {
  it('returns sources for a live query', async () => {
    const provider = new ScraperSearchProvider(() => ({ engine: 'bing', bingBaseUrl: BING_DEFAULT_BASE_URL, duckduckgoBaseUrl: DUCKDUCKGO_DEFAULT_BASE_URL }))
    const result = await provider.search({ query: 'DeepSeek Harness', maxResults: 5 })
    expect(result.sources.length).toBeGreaterThan(0)
    for (const source of result.sources) expect(source.url).toMatch(/^https?:\/\//u)
  }, 30_000)
})

maybeDdg('ScraperSearchProvider real engine (duckduckgo)', () => {
  it('returns sources for a live query', async () => {
    const provider = new ScraperSearchProvider(() => ({ engine: 'duckduckgo', bingBaseUrl: BING_DEFAULT_BASE_URL, duckduckgoBaseUrl: DUCKDUCKGO_DEFAULT_BASE_URL }))
    const result = await provider.search({ query: 'DeepSeek Harness', maxResults: 5 })
    expect(result.sources.length).toBeGreaterThan(0)
    for (const source of result.sources) expect(source.url).toMatch(/^https?:\/\//u)
  }, 30_000)
})

describe('defaults', () => {
  it('ships bing as the engine and the documented endpoints', () => {
    expect(SCRAPER_DEFAULT_ENGINE).toBe('bing')
    expect(BING_DEFAULT_BASE_URL).toBe('https://www.bing.com')
    expect(DUCKDUCKGO_DEFAULT_BASE_URL).toBe('https://html.duckduckgo.com')
  })
})
