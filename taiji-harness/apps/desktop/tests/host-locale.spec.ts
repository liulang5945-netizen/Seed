/** Desktop reads its startup language through the Host's shared settings RPC. */
import { describe, expect, it, vi } from 'vitest'
import { readHostLocalePreference } from '../src/host-locale.ts'

const origin = 'http://127.0.0.1:19387'

/** A settings/describe transport that only answers the locale namespace. */
function transport(namespaceValue: unknown) {
  const send = vi.fn(async (input: string | URL, init?: RequestInit) => {
    expect(String(input)).toBe(`${origin}/api/settings/describe`)
    const { rpcId, method } = JSON.parse(init!.body as string) as { rpcId: string; method: string }
    expect(method).toBe('settings/describe')
    return Response.json({
      type: 'server-response', rpcId,
      result: { ok: true, value: { namespaces: [{ ns: 'locale', value: namespaceValue }] } },
    })
  })
  return send
}

describe('desktop locale preference over the Host settings RPC', () => {
  it('reads the stored preference without querying accounts or model providers', async () => {
    const send = transport({ preference: 'zh' })
    const previousFetch = globalThis.fetch
    globalThis.fetch = send as unknown as typeof fetch
    try {
      expect(await readHostLocalePreference(origin, 'dsh-auth-test')).toBe('zh')
    } finally { globalThis.fetch = previousFetch }
    expect(send).toHaveBeenCalledOnce()
    const [, init] = send.mock.calls[0]!
    expect((init!.headers as Record<string, string>).cookie).toBe('dsh-auth-test')
  })

  it('treats a missing or empty preference as no explicit selection', async () => {
    const previousFetch = globalThis.fetch
    try {
      globalThis.fetch = transport({}) as unknown as typeof fetch
      expect(await readHostLocalePreference(origin, 'c')).toBeNull()
      globalThis.fetch = transport(undefined) as unknown as typeof fetch
      expect(await readHostLocalePreference(origin, 'c')).toBeNull()
    } finally { globalThis.fetch = previousFetch }
  })

  it('rejects a non-text preference and an unmatched RPC envelope', async () => {
    const previousFetch = globalThis.fetch
    try {
      globalThis.fetch = transport({ preference: 7 }) as unknown as typeof fetch
      await expect(readHostLocalePreference(origin, 'c')).rejects.toThrow('invalid locale preference')
      globalThis.fetch = vi.fn(async () =>
        Response.json({ type: 'server-response', rpcId: 'other', result: { ok: true, value: {} } })) as unknown as typeof fetch
      await expect(readHostLocalePreference(origin, 'c')).rejects.toThrow('settings RPC failed')
    } finally { globalThis.fetch = previousFetch }
  })
})