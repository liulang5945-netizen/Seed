/** Desktop reads its startup language through the Host's shared settings RPC. */
import { describe, expect, it } from 'vitest'
import { readHostLocalePreference } from '../src/host-locale.ts'

const origin = 'http://127.0.0.1:19387'

/** One request the shell made through the installed fetch stub. */
interface RecordedRequest {
  readonly input: URL | RequestInfo
  readonly init?: RequestInit
}

/** A settings/describe transport that only answers the locale namespace. */
function transport(namespaceValue: unknown): {
  readonly requests: RecordedRequest[]
  readonly fetch: (input: URL | RequestInfo, init?: RequestInit) => Promise<Response>
} {
  const requests: RecordedRequest[] = []
  return {
    requests,
    fetch: async (input, init) => {
      requests.push({ input, init })
      expect(String(input)).toBe(`${origin}/api/settings/describe`)
      const { rpcId, method } = JSON.parse(String(init?.body)) as { rpcId: string; method: string }
      expect(method).toBe('settings/describe')
      return Response.json({
        type: 'server-response', rpcId,
        result: { ok: true, value: { namespaces: [{ ns: 'locale', value: namespaceValue }] } },
      })
    },
  }
}

describe('desktop locale preference over the Host settings RPC', () => {
  it('reads the stored preference without querying accounts or model providers', async () => {
    const { requests, fetch: fetchStub } = transport({ preference: 'zh' })
    const previousFetch = globalThis.fetch
    globalThis.fetch = fetchStub
    try {
      expect(await readHostLocalePreference(origin, 'dsh-auth-test')).toBe('zh')
    } finally { globalThis.fetch = previousFetch }
    expect(requests).toHaveLength(1)
    const [first] = requests
    expect((first!.init!.headers as Record<string, string>).cookie).toBe('dsh-auth-test')
  })

  it('treats a missing or empty preference as no explicit selection', async () => {
    const previousFetch = globalThis.fetch
    try {
      globalThis.fetch = transport({}).fetch
      expect(await readHostLocalePreference(origin, 'c')).toBeNull()
      globalThis.fetch = transport(undefined).fetch
      expect(await readHostLocalePreference(origin, 'c')).toBeNull()
    } finally { globalThis.fetch = previousFetch }
  })

  it('rejects a non-text preference and an unmatched RPC envelope', async () => {
    const previousFetch = globalThis.fetch
    try {
      globalThis.fetch = transport({ preference: 7 }).fetch
      await expect(readHostLocalePreference(origin, 'c')).rejects.toThrow('invalid locale preference')
      globalThis.fetch = async () => Response.json({
        type: 'server-response', rpcId: 'other', result: { ok: true, value: {} },
      })
      await expect(readHostLocalePreference(origin, 'c')).rejects.toThrow('settings RPC failed')
    } finally { globalThis.fetch = previousFetch }
  })
})
