import { createServer, type Server } from 'node:http'
import type { AddressInfo } from 'node:net'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { WebError } from '@taiji/dsh-web'
import {
  TaijiSearchProvider,
  TAIJI_SEARCH_PROVIDER_ID,
  mapRuntimeResponse,
  mapRuntimeSource,
} from '../src/provider.ts'

describe('mapRuntimeSource', () => {
  it('keeps non-blank optionals and drops blank-string lies', () => {
    expect(mapRuntimeSource({ url: 'https://example.com/a', title: 'A', snippet: 'text', engine: 'Bing' })).toEqual({
      url: 'https://example.com/a',
      title: 'A',
      snippet: 'text',
    })
    expect(mapRuntimeSource({ url: 'https://example.com/b', title: '', snippet: '  ' })).toEqual({
      url: 'https://example.com/b',
    })
  })
  it('rejects rows without a usable url', () => {
    expect(mapRuntimeSource({})).toBeUndefined()
    expect(mapRuntimeSource({ url: '' })).toBeUndefined()
    expect(mapRuntimeSource({ url: 42 })).toBeUndefined()
  })
})

describe('mapRuntimeResponse', () => {
  it('normalizes the runtime envelope', () => {
    expect(mapRuntimeResponse({
      sources: [
        { url: 'https://example.com/a', title: 'A', snippet: 's', engine: 'Bing' },
        { url: 'https://example.com/b' },
        'garbage',
        null,
      ],
      truncated: false,
    })).toEqual({
      sources: [{ url: 'https://example.com/a', title: 'A', snippet: 's' }, { url: 'https://example.com/b' }],
      truncated: false,
    })
  })
  it('accepts an empty source list as a legitimate empty result', () => {
    expect(mapRuntimeResponse({ sources: [], truncated: false })).toEqual({ sources: [], truncated: false })
  })
  it('throws a provider error for an unrecognizable envelope', () => {
    expect(() => mapRuntimeResponse({ nope: true })).toThrowError(WebError)
    expect(() => mapRuntimeResponse(null)).toThrowError(WebError)
    expect(() => mapRuntimeResponse({ sources: 'not-a-list' })).toThrowError(WebError)
  })
})

describe('TaijiSearchProvider', () => {
  let server: Server
  let origin: string
  let lastPath = ''
  let lastBody = ''
  let lastMethod = ''

  beforeAll(async () => {
    server = createServer((request, response) => {
      let body = ''
      request.setEncoding('utf8')
      request.on('data', (chunk: string) => { body += chunk })
      request.on('end', () => {
        lastPath = request.url ?? ''
        lastBody = body
        lastMethod = request.method ?? ''
        if (body.includes('"query":"boom"')) {
          response.writeHead(502); response.end('runtime exploded')
          return
        }
        if (body.includes('"query":"garbage"')) {
          response.writeHead(200, { 'content-type': 'application/json' }); response.end('not-json')
          return
        }
        response.writeHead(200, { 'content-type': 'application/json' })
        response.end(JSON.stringify({
          sources: [
            { url: 'https://example.com/composed', title: 'Composed', snippet: 'works end to end', engine: 'Bing' },
          ],
          truncated: false,
        }))
      })
    })
    const a = await new Promise<AddressInfo>((r) => { server.listen(0, '127.0.0.1', () => { r(server.address() as AddressInfo) }) })
    origin = `http://127.0.0.1:${String(a.port)}`
  })
  afterAll(async () => { await new Promise<void>((r) => { server.close(() => { r() }) }) })

  const provider = (): TaijiSearchProvider => new TaijiSearchProvider(() => ({ baseURL: origin }))

  it('registers under the taiji-search id and is available without any credential', () => {
    expect(TAIJI_SEARCH_PROVIDER_ID).toBe('taiji-search')
    expect(provider().id).toBe('taiji-search')
    expect(provider().available()).toBe(true)
  })

  it('posts to the runtime search endpoint and maps the sources', async () => {
    const result = await provider().search({ query: 'compose probe', maxResults: 8 })
    expect(lastMethod).toBe('POST')
    expect(lastPath).toBe('/api/tools/web_search')
    expect(JSON.parse(lastBody)).toEqual({ query: 'compose probe', max_results: 8 })
    expect(result.sources).toEqual([
      { url: 'https://example.com/composed', title: 'Composed', snippet: 'works end to end' },
    ])
    expect(result.truncated).toBe(false)
  })

  it('reports runtime failures as provider errors naming the endpoint', async () => {
    const failure = await provider().search({ query: 'boom' }).catch((error: unknown) => error)
    expect(failure).toBeInstanceOf(WebError)
    expect((failure as WebError).code).toBe('WEB_PROVIDER_ERROR')
    expect((failure as WebError).message).toContain('/api/tools/web_search')
  })

  it('reports an unprocessable body as a provider error', async () => {
    const failure = await provider().search({ query: 'garbage' }).catch((error: unknown) => error)
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
