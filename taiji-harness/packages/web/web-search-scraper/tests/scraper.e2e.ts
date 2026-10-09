import { describe, expect, it } from 'vitest'
import {
  TAIJI_RUNTIME_DEFAULT_BASE_URL,
  TAIJI_SEARCH_PATH,
  TAIJI_SEARCH_PROVIDER_ID,
  TaijiSearchProvider,
} from '@taiji/dsh-web-search-scraper'

/**
 * Real-runtime smoke for the Taiji-search provider. Live arms self-skip unless
 * the local runtime actually answers its readiness probe (CI and ordinary test
 * runs must not depend on a running backend); the arm additionally requires the
 * `$DSH_WEB_SCRAPER_E2E` opt-in so a coincidentally-running runtime never turns
 * a test run into live search traffic.
 */
const enabled = process.env.DSH_WEB_SCRAPER_E2E !== undefined && process.env.DSH_WEB_SCRAPER_E2E.length > 0

/** True when the configured runtime root answers its readiness probe. */
async function runtimeUp(): Promise<boolean> {
  const root = process.env.TAIJI_RUNTIME_BASE_URL ?? TAIJI_RUNTIME_DEFAULT_BASE_URL
  try {
    const res = await fetch(`${root}/api/health`, { signal: AbortSignal.timeout(3_000) })
    return res.ok
  } catch {
    return false
  }
}

const runtimeReachable = enabled && await runtimeUp()
const maybe = runtimeReachable ? describe : describe.skip

maybe('TaijiSearchProvider real runtime', () => {
  it('returns sources for a live query', async () => {
    const root = process.env.TAIJI_RUNTIME_BASE_URL ?? TAIJI_RUNTIME_DEFAULT_BASE_URL
    const provider = new TaijiSearchProvider(() => ({ baseURL: root }))
    const result = await provider.search({ query: 'Python asyncio tutorial', maxResults: 5 })
    expect(result.sources.length).toBeGreaterThan(0)
    for (const source of result.sources) expect(source.url).toMatch(/^https?:\/\//u)
  }, 60_000)
})

describe('defaults', () => {
  it('ships the local runtime root, the documented path, and the provider id', () => {
    expect(TAIJI_RUNTIME_DEFAULT_BASE_URL).toBe('http://127.0.0.1:8000')
    expect(TAIJI_SEARCH_PATH).toBe('/api/tools/web_search')
    expect(TAIJI_SEARCH_PROVIDER_ID).toBe('taiji-search')
  })
})
