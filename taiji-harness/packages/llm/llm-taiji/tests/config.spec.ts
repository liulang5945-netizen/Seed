/** Configuration resolution: defaults, bounds, and the provider-owned retry policy. */
import { describe, expect, it } from 'vitest'
import { Config, plainOptions, resolveAdapterOptions } from '../src/config.ts'
import { DEFAULT_BASE_URL, DEFAULT_MODELS } from '../src/defaults.ts'

describe('Taiji configuration resolution', () => {
  it("defaults to the runtime's local address and its single catalog entry", () => {
    const resolved = resolveAdapterOptions({})

    expect(resolved.baseURL).toBe(DEFAULT_BASE_URL)
    expect(resolved.models).toEqual(DEFAULT_MODELS)
    expect(resolved.retryPolicy).toMatchObject({ mode: 'normal', maxRetries: 5 })
  })

  it('keeps a configured endpoint and its catalog', () => {
    const models = [{ id: 'runtime', name: 'Runtime', description: 'local runtime', contextWindow: 4096 }]

    const resolved = resolveAdapterOptions({ baseURL: 'http://127.0.0.1:9000/', models })

    expect(resolved.baseURL).toBe('http://127.0.0.1:9000/')
    expect(resolved.models).toEqual(models)
  })

  it('reads the value behind a configuration reference', () => {
    const parsed = Config({ baseURL: 'http://127.0.0.1:9000', models: [{ id: 'runtime' }] })

    expect(plainOptions(parsed)).toMatchObject({
      baseURL: 'http://127.0.0.1:9000',
      models: [{ id: 'runtime' }],
    })
  })

  it.each([
    { name: 'a non-HTTP scheme', baseURL: 'file:///tmp/taiji' },
    { name: 'embedded credentials', baseURL: 'http://user:pass@127.0.0.1:8000' },
    { name: 'a query', baseURL: 'http://127.0.0.1:8000/?probe=1' },
    { name: 'a fragment', baseURL: 'http://127.0.0.1:8000/#model' },
    { name: 'text that is not a URL', baseURL: 'not a url' },
  ])('refuses $name as an endpoint', ({ baseURL }) => {
    expect(() => resolveAdapterOptions({ baseURL })).toThrow()
  })

  it.each([
    { name: 'an empty id', models: [{ id: '' }] },
    { name: 'an empty name', models: [{ id: 'runtime', name: '' }] },
    { name: 'a zero contextWindow', models: [{ id: 'runtime', contextWindow: 0 }] },
    { name: 'a fractional contextWindow', models: [{ id: 'runtime', contextWindow: 1.5 }] },
    { name: 'a duplicate id', models: [{ id: 'runtime' }, { id: 'runtime' }] },
  ])('refuses $name in the catalog', ({ models }) => {
    expect(() => resolveAdapterOptions({ models })).toThrow(/llm-taiji/)
  })

  it('accepts one configuration the settings editor would store', () => {
    const parsed = Config({ baseURL: 'http://127.0.0.1:8000', retryPolicy: { mode: 'normal', maxRetries: 1 } })

    const resolved = resolveAdapterOptions(plainOptions(parsed))

    expect(resolved.retryPolicy).toMatchObject({ mode: 'normal', maxRetries: 1 })
    expect(resolved.models).toEqual(DEFAULT_MODELS)
  })
})
