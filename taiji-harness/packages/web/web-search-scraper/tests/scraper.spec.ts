import { createServer, type Server } from 'node:http'
import type { AddressInfo } from 'node:net'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import {
  ScraperSearchProvider,
  decodeHtmlEntities,
  mapBingHtml,
  mapDuckDuckGoHtml,
  resolveDuckDuckGoHref,
} from '../src/provider.ts'
import { WebError } from '@taiji/dsh-web'

/** A trimmed Bing result list carrying the markup shapes the parser must own. */
const BING_FIXTURE = `<html><body><ol id="b_results">
<li class="b_algo" data-id iid=SERP.5336><div class="b_tpcn"><a class="tilk" href="https://example.com/alpha/"><div class="tptt">example.com</div></a></div><h2 class=""><a target="_blank" href="https://example.com/alpha?a=1&amp;b=2" h="ID=SERP,5130.1">Alpha <strong>Result</strong></a></h2><div class="b_caption"><p class="b_lineclamp2" data-rslinkclamp-iid="">1 天前&ensp;&#0183;&ensp;Alpha snippet text \u2026</p></div></li>
<li class="b_algo"><h2><a href="https://example.com/beta">Beta page</a></h2><div class="b_caption"><p class="b_lineclamp4">Beta snippet &amp; more</p></div></li>
<li class="b_algo"><h2><a href="javascript:void(0)">Script trap</a></h2></li>
<li class="b_algo"><h2><a href="https://example.com/beta">Beta duplicate</a></h2></li>
<li class="b_algo"><h2><a href="https://example.com/gamma">Gamma bare</a></h2><div class="b_caption"><span>no paragraph here</span></div></li>
</ol></body></html>`

/** A trimmed DuckDuckGo HTML-endpoint result list with redirect and direct hrefs. */
const DDG_FIXTURE = `<html><body>
<div class="result results_links results_links_deep web-result"><div class="result__body"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone&amp;rut=abc">One <b>Title</b></a><a class="result__snippet" href="#">One <b>snippet</b> text</a></div></div>
<div class="result results_links results_links_deep web-result"><div class="result__body"><a class="result__a" href="https://example.com/two">Two page</a><a class="result__snippet">Two snippet</a></div></div>
<div class="result results_links results_links_deep web-result"><div class="result__body"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Ftwo&amp;rut=def">Two redirect duplicate</a></div></div>
</body></html>`

describe('decodeHtmlEntities', () => {
  it('decodes named, decimal, and hex entities', () => {
    expect(decodeHtmlEntities('a&amp;b &#0183; &#x2026; &ensp;')).toBe('a&b \u00b7 \u2026 \u2002')
  })
  it('leaves unknown named entities untouched', () => {
    expect(decodeHtmlEntities('&notinmyvocabulary;')).toBe('&notinmyvocabulary;')
  })
  it('maps out-of-range and surrogate code points to the replacement char', () => {
    expect(decodeHtmlEntities('&#xd800; &#999999999;')).toBe('\ufffd \ufffd')
  })
})

describe('mapBingHtml', () => {
  it('parses anchors and line-clamp snippets from b_algo blocks', () => {
    const sources = mapBingHtml(BING_FIXTURE)
    expect(sources).toEqual([
      {
        url: 'https://example.com/alpha?a=1&b=2',
        title: 'Alpha Result',
        snippet: '1 天前 \u00b7 Alpha snippet text \u2026',
      },
      { url: 'https://example.com/beta', title: 'Beta page', snippet: 'Beta snippet & more' },
      { url: 'https://example.com/gamma', title: 'Gamma bare' },
    ])
  })
  it('collapses duplicate URLs to their first occurrence', () => {
    const urls = mapBingHtml(BING_FIXTURE).map(source => source.url)
    expect(urls.filter(url => url === 'https://example.com/beta')).toHaveLength(1)
  })
  it('returns no sources for a page without result blocks', () => {
    expect(mapBingHtml('<html><body>nothing here</body></html>')).toEqual([])
  })
})

describe('mapDuckDuckGoHtml', () => {
  it('resolves redirect hrefs and pairs snippets with their result link', () => {
    expect(mapDuckDuckGoHtml(DDG_FIXTURE)).toEqual([
      { url: 'https://example.com/one', title: 'One Title', snippet: 'One snippet text' },
      { url: 'https://example.com/two', title: 'Two page', snippet: 'Two snippet' },
    ])
  })
})

describe('resolveDuckDuckGoHref', () => {
  it('decodes the uddg destination of a protocol-relative redirect', () => {
    expect(resolveDuckDuckGoHref('//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fx&rut=r')).toBe('https://example.com/x')
  })
  it('passes direct http hrefs through', () => {
    expect(resolveDuckDuckGoHref('https://example.com/direct')).toBe('https://example.com/direct')
  })
  it('rejects redirect links without a uddg destination', () => {
    expect(resolveDuckDuckGoHref('//duckduckgo.com/l/?rut=r')).toBeUndefined()
  })
  it('rejects unusable schemes', () => {
    expect(resolveDuckDuckGoHref('javascript:void(0)')).toBeUndefined()
  })
})

describe('ScraperSearchProvider', () => {
  let server: Server
  let origin: string
  let lastQuery = ''

  beforeAll(async () => {
    server = createServer((request, response) => {
      lastQuery = request.url ?? ''
      if (request.url?.includes('boom')) {
        response.writeHead(500); response.end('engine exploded')
        return
      }
      response.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
      response.end(BING_FIXTURE)
    })
    const a = await new Promise<AddressInfo>((r) => { server.listen(0, '127.0.0.1', () => { r(server.address() as AddressInfo) }) })
    origin = `http://127.0.0.1:${String(a.port)}`
  })
  afterAll(async () => { await new Promise<void>((r) => { server.close(() => { r() }) }) })

  const provider = (): ScraperSearchProvider => new ScraperSearchProvider(() => ({
    engine: 'bing',
    bingBaseUrl: origin,
    duckduckgoBaseUrl: origin,
  }))

  it('is available without any credential', () => {
    expect(provider().available()).toBe(true)
  })

  it('parses the engine result page into sources', async () => {
    const result = await provider().search({ query: 'probe query', maxResults: 8 })
    expect(lastQuery.startsWith('/search?q=probe%20query&count=8')).toBe(true)
    expect(result.sources).toHaveLength(3)
    expect(result.truncated).toBe(false)
  })

  it('reports engine HTTP failures as provider errors', async () => {
    const failure = await provider().search({ query: 'boom' }).catch((error: unknown) => error)
    expect(failure).toBeInstanceOf(WebError)
    expect((failure as WebError).code).toBe('WEB_PROVIDER_ERROR')
  })

  it('surfaces a pre-aborted caller as WEB_ABORTED', async () => {
    const controller = new AbortController()
    controller.abort()
    const failure = await provider().search({ query: 'probe' }, controller.signal).catch((error: unknown) => error)
    expect(failure).toBeInstanceOf(WebError)
    expect((failure as WebError).code).toBe('WEB_ABORTED')
  })
})
