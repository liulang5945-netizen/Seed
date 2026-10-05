import { describe, expect, it, vi } from 'vitest'
import { Context } from '@taiji/cordis'
import {
  credentialKey, credentialKeyId, credentialKeyScope, credentialRef, isCredentialKeySegment, parseCredentialKey,
} from '../src/index.ts'
import type { CredentialKey, CredentialRef } from '../src/index.ts'
import { MemoryCredentials } from './memory.ts'

const REF = credentialRef('DEEPSEEK_API_KEY')

async function boot(seed: Record<string, string> = {}): Promise<Context> {
  const ctx = new Context()
  await ctx.plugin(MemoryCredentials, seed)
  return ctx
}

/**
 * The memory provider commits through `ctx.emit`, which is cordis's own
 * fan-out; these pulses drive the provider-side `notify*` halves instead, so
 * the containment contract on `CredentialProvider` itself is what runs.
 */
class NotifyingCredentials extends MemoryCredentials {
  pulseReference(ref: CredentialRef): void { this.notifyUpdated(ref) }

  pulseRecord(key: CredentialKey): void { this.notifyRecordUpdated(key) }
}

async function bootNotifying(seed: Record<string, string> = {}): Promise<Context> {
  const ctx = new Context()
  await ctx.plugin(NotifyingCredentials, seed)
  return ctx
}

describe('credentialRef', () => {
  it('brands POSIX shell identifiers', () => {
    expect(credentialRef('DEEPSEEK_API_KEY')).toBe('DEEPSEEK_API_KEY')
    expect(credentialRef('_private')).toBe('_private')
    expect(credentialRef('lower_case9')).toBe('lower_case9')
  })

  it('rejects every other shape', () => {
    for (const invalid of ['', '9LEADING', 'WITH-DASH', 'WITH SPACE', 'ns:key']) {
      expect(() => credentialRef(invalid)).toThrow(TypeError)
    }
  })
})

describe('isCredentialKeySegment', () => {
  it('answers whether credentialKey would accept the segment', () => {
    for (const valid of ['llm-pi-ai', 'openai-codex', 'a', 'z9']) {
      expect(isCredentialKeySegment(valid)).toBe(true)
    }
    // The shapes an arbitrary settings dict key can take that a record id
    // cannot: a consumer asks here instead of learning it from a throw.
    for (const invalid of ['', 'My_Proxy', 'z.ai', 'UPPER', '9leading', 'a/b']) {
      expect(isCredentialKeySegment(invalid)).toBe(false)
    }
  })
})

describe('the credentials seam through the memory provider', () => {
  it('mounts as ctx.credentials and resolves a seeded reference with its source', async () => {
    const ctx = await boot({ DEEPSEEK_API_KEY: 'sk-seeded' })
    expect(await ctx.credentials.resolve(REF)).toEqual({ value: 'sk-seeded', source: 'memory' })
    expect(await ctx.credentials.describe(REF)).toEqual({ configured: true, source: 'memory', writable: true })
  })

  it('treats an empty stored value as absent everywhere', async () => {
    const ctx = await boot({ DEEPSEEK_API_KEY: '' })
    expect(await ctx.credentials.resolve(REF)).toBeUndefined()
    expect(await ctx.credentials.describe(REF)).toEqual({ configured: false, writable: true })
  })

  it('stores through set, removes through unset, and emits the committed change', async () => {
    const ctx = await boot()
    const events: CredentialRef[] = []
    ctx.on('credentials/reference-updated', ref => void events.push(ref))

    await ctx.credentials.set(REF, 'sk-live')
    expect(await ctx.credentials.resolve(REF)).toEqual({ value: 'sk-live', source: 'memory' })
    await ctx.credentials.unset(REF)
    expect(await ctx.credentials.resolve(REF)).toBeUndefined()
    expect(events).toEqual([REF, REF])
  })

  it('rejects an empty set and keeps an absent unset silent', async () => {
    const ctx = await boot()
    const events: CredentialRef[] = []
    ctx.on('credentials/reference-updated', ref => void events.push(ref))

    await expect(ctx.credentials.set(REF, '')).rejects.toThrow(/empty value/)
    await ctx.credentials.unset(REF)
    expect(events).toEqual([])
  })

  it('removes the service with its fiber', async () => {
    const ctx = new Context()
    const fiber = await ctx.plugin(MemoryCredentials)
    expect(ctx.get('credentials')).toBeDefined()
    await fiber.dispose()
    expect(ctx.get('credentials')).toBeUndefined()
  })
})

describe('credentialKey and its read halves', () => {
  it('brands two valid segments and answers scope and id back', () => {
    const key = credentialKey('llm-pi-ai', 'route-1')
    expect(key).toBe('llm-pi-ai/route-1')
    expect(credentialKeyScope(key)).toBe('llm-pi-ai')
    expect(credentialKeyId(key)).toBe('route-1')
  })

  it('rejects a segment outside the grammar, whichever half carries it', () => {
    expect(() => credentialKey('My_Proxy', 'route-1')).toThrow(TypeError)
    expect(() => credentialKey('llm-pi-ai', 'WITH SPACE')).toThrow(TypeError)
  })

  it('parseCredentialKey admits exactly "<scope>/<id>" and rejects the rest', () => {
    expect(parseCredentialKey('llm-pi-ai/route-1')).toBe('llm-pi-ai/route-1')
    for (const invalid of ['abc', 'a/b/c', 'llm-pi-ai/', '/route-1']) {
      expect(() => parseCredentialKey(invalid)).toThrow(TypeError)
    }
  })
})

describe('the notification fan-out containment', () => {
  it('fans credentials/reference-updated to every listener with the ref as subject', async () => {
    const ctx = await bootNotifying()
    const provider = ctx.credentials as NotifyingCredentials
    const seen: CredentialRef[] = []
    ctx.on('credentials/reference-updated', ref => void seen.push(ref))

    provider.pulseReference(REF)
    expect(seen).toEqual([REF])
  })

  it('keeps a throwing listener from changing the committed operation, and later listeners still run', async () => {
    const ctx = await bootNotifying()
    const provider = ctx.credentials as NotifyingCredentials
    ctx.on('credentials/reference-updated', () => {
      throw new Error('watcher boom')
    })
    const second = vi.fn()
    ctx.on('credentials/reference-updated', second)

    expect(() => provider.pulseReference(REF)).not.toThrow()
    expect(second).toHaveBeenCalledWith(REF)
  })

  it('contains an async listener rejection and logs it after the operation returned', async () => {
    const ctx = await bootNotifying()
    const provider = ctx.credentials as NotifyingCredentials
    // An unknown-returning function keeps the typed surface legal while the
    // runtime value is still the rejected promise the containment must handle.
    const boom = (): unknown => Promise.reject(new Error('async watcher boom'))
    ctx.on('credentials/reference-updated', boom)

    expect(() => provider.pulseReference(REF)).not.toThrow()
    await new Promise(resolve => setTimeout(resolve, 10))
  })

  it('rethrows an invariant-coded listener failure after the remaining listeners', async () => {
    const ctx = await bootNotifying()
    const provider = ctx.credentials as NotifyingCredentials
    ctx.on('credentials/reference-updated', () => {
      throw Object.assign(new Error('forged relation'), { code: 'INVARIANT' })
    })
    const second = vi.fn()
    ctx.on('credentials/reference-updated', second)

    expect(() => provider.pulseReference(REF)).toThrow(/forged relation/)
    expect(second).toHaveBeenCalledWith(REF)
  })

  it('fans credentials/record-updated through the same containment with the key as subject', async () => {
    const ctx = await bootNotifying()
    const provider = ctx.credentials as NotifyingCredentials
    const key = credentialKey('test-plugin', 'route-1')
    const seen: CredentialKey[] = []
    ctx.on('credentials/record-updated', k => void seen.push(k))
    ctx.on('credentials/record-updated', () => {
      throw new Error('record watcher boom')
    })

    expect(() => provider.pulseRecord(key)).not.toThrow()
    expect(seen).toEqual([key])
  })
})
