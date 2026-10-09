/** Pure first-run readiness projection over the shared Models join. */
import { describe, expect, it } from 'vitest'
import type { CredentialInfo } from '@taiji/dsh-api-remotes/client'
import type { ModelsSettingsState, OnboardingTarget, ProviderRow } from '../src/client/store.ts'
import { onboardingReadiness, providerUsable } from '../src/client/store.ts'

const missingCredential: CredentialInfo = { configured: false, writable: true }

/** The route the credential step speaks for. */
const DEEPSEEK: OnboardingTarget = { provider: 'deepseek-official', settingsNs: 'llm-deepseek' }
/** The route the local-runtime step speaks for. */
const TAIJI: OnboardingTarget = { provider: 'taiji-local', settingsNs: 'llm-taiji' }

function row(overrides: Partial<ProviderRow> = {}): ProviderRow {
  return {
    entry: {
      provider: 'deepseek-official',
      displayName: 'DeepSeek',
      settingsNs: 'llm-deepseek',
      settingsPath: [],
      active: true,
    },
    configured: true,
    removable: false,
    apiKeyEnv: 'DEEPSEEK_API_KEY',
    credential: missingCredential,
    ...overrides,
  }
}

/** The shipped local runtime: declared, needs no credential, answers only while up. */
function taijiRow(active: boolean): ProviderRow {
  return {
    entry: {
      provider: 'taiji-local',
      displayName: 'Taiji（本地运行时）',
      settingsNs: 'llm-taiji',
      settingsPath: [],
      active,
    },
    configured: true,
    removable: false,
    apiKeyEnv: undefined,
    credential: undefined,
  }
}

/** A second provider the user configured themselves. */
function otherRow(overrides: Partial<ProviderRow> = {}): ProviderRow {
  return {
    entry: {
      provider: 'hfai',
      displayName: 'HFAI',
      settingsNs: 'llm-pi-ai',
      settingsPath: ['providers', 'hfai'],
      active: true,
    },
    configured: true,
    removable: true,
    apiKeyEnv: 'HFAI_API_KEY',
    credential: { configured: true, source: 'file', writable: true },
    ...overrides,
  }
}

function state(overrides: Partial<ModelsSettingsState> = {}): ModelsSettingsState {
  return {
    status: 'ready',
    error: null,
    credentialError: null,
    writable: true,
    rows: [row()],
    namespaces: new Map(),
    ...overrides,
  }
}

describe('providerUsable', () => {
  it('requires a registered route and a stored key for every named reference', () => {
    expect(providerUsable(otherRow())).toBe(true)
    expect(providerUsable(otherRow({ entry: { ...otherRow().entry, active: false } }))).toBe(false)
    expect(providerUsable(otherRow({ credential: missingCredential }))).toBe(false)
    expect(providerUsable(otherRow({ credential: undefined }))).toBe(false)
  })

  it('treats a reference-free registered route as provider-native authentication', () => {
    expect(providerUsable(otherRow({ apiKeyEnv: undefined, credential: undefined }))).toBe(true)
  })
})

describe('onboardingReadiness', () => {
  it('waits for the first join and skips onboarding when the target directory entry is absent', () => {
    expect(onboardingReadiness(state({ status: 'idle', rows: [] }), DEEPSEEK)).toEqual({ kind: 'loading' })
    expect(onboardingReadiness(state({ status: 'loading', rows: [] }), DEEPSEEK)).toEqual({ kind: 'loading' })
    expect(onboardingReadiness(state({ rows: [] }), DEEPSEEK)).toEqual({ kind: 'adapter-absent' })
    expect(onboardingReadiness(state({
      rows: [row({
        entry: {
          ...row().entry,
          settingsNs: '',
        },
      })],
    }), DEEPSEEK)).toEqual({ kind: 'adapter-absent' })
  })

  it('reports a missing writable effective credential', () => {
    expect(onboardingReadiness(state(), DEEPSEEK)).toEqual({ kind: 'credential-missing' })
  })

  it('ends onboarding once any other registered provider can serve requests', () => {
    expect(onboardingReadiness(state({ rows: [row(), otherRow()] }), DEEPSEEK)).toEqual({ kind: 'provider-ready' })
    // A provider the user cannot reach yet leaves the prompt in place.
    expect(onboardingReadiness(state({
      rows: [row(), otherRow({ credential: missingCredential })],
    }), DEEPSEEK)).toEqual({ kind: 'credential-missing' })
  })

  it('accepts file and process-environment credentials without prompting', () => {
    expect(onboardingReadiness(state({
      rows: [row({ credential: { configured: true, source: 'file', writable: true } })],
    }), DEEPSEEK)).toEqual({ kind: 'provider-ready' })
    expect(onboardingReadiness(state({
      rows: [row({ credential: { configured: true, source: 'env', writable: false } })],
    }), DEEPSEEK)).toEqual({ kind: 'provider-ready' })
  })

  it('separates a credential-free route nobody serves from one awaiting a key', () => {
    expect(onboardingReadiness(state({ rows: [taijiRow(false)] }), TAIJI))
      .toEqual({ kind: 'runtime-unreachable' })
    expect(onboardingReadiness(state({ rows: [taijiRow(true)] }), TAIJI))
      .toEqual({ kind: 'provider-ready' })
    // The same inactive row read by the credential step still asks for a key.
    expect(onboardingReadiness(state({
      rows: [row({ entry: { ...row().entry, active: false } })],
    }), DEEPSEEK)).toEqual({ kind: 'unavailable', reason: 'provider-inactive' })
    // One usable provider ends the runtime step too.
    expect(onboardingReadiness(state({ rows: [taijiRow(false), otherRow()] }), TAIJI))
      .toEqual({ kind: 'provider-ready' })
  })

  it('turns missing capabilities into diagnostics that never block the product', () => {
    expect(onboardingReadiness(state({ status: 'error', error: 'settings down' }), DEEPSEEK)).toEqual({
      kind: 'unavailable',
      reason: 'load-failed',
    })
    expect(onboardingReadiness(state({
      credentialError: 'credentials service is absent',
    }), DEEPSEEK)).toEqual({
      kind: 'unavailable',
      reason: 'credentials-unavailable',
    })
    expect(onboardingReadiness(state({
      rows: [row({ credential: undefined })],
    }), DEEPSEEK)).toEqual({ kind: 'unavailable', reason: 'credentials-unavailable' })
    expect(onboardingReadiness(state({
      rows: [row({ credential: { configured: false, writable: false } })],
    }), DEEPSEEK)).toEqual({ kind: 'unavailable', reason: 'credential-read-only' })
    expect(onboardingReadiness(state({ writable: false }), DEEPSEEK)).toEqual({
      kind: 'unavailable',
      reason: 'settings-read-only',
    })
  })
})
