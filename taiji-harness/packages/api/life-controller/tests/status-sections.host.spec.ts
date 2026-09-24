/** Status-section reads: the workbench capability snapshot and the auth state. */
import { afterEach, describe, expect, it } from 'vitest'
import { Context } from '@taiji/cordis'
import LifeController from '../src/index.ts'
import { closeMockRuntimes, mockLifeRuntime, type MockLifeRuntime } from './mock-runtime.ts'

const roots: Context[] = []

afterEach(async () => {
  await Promise.all(roots.splice(0).map(ctx => ctx.fiber.dispose()))
  await closeMockRuntimes()
})

interface Harness {
  readonly controller: LifeController
  readonly runtime: MockLifeRuntime
}

/** Boot the controller over a loopback stand-in at the schema's floor cadence. */
async function harness(): Promise<Harness> {
  const runtime = await mockLifeRuntime()
  const ctx = new Context()
  roots.push(ctx)
  const dispose = (): void => {}
  ctx.provide('typert', {
    lookups: { configure: () => dispose },
    contexts: { configureHost: () => dispose },
  } as never)
  const controller = new LifeController(ctx, {
    baseURL: runtime.url,
    pollIntervalMs: 250,
    activePollIntervalMs: 250,
    requestTimeoutMs: 2_000,
    maxCheckpoints: 5,
  })
  return { controller, runtime }
}

describe('LifeController status sections', () => {
  it('projects the workbench and auth sections from the same always-on status read', async () => {
    const { controller, runtime } = await harness()
    runtime.runtimeStatus = {
      ...runtime.runtimeStatus,
      tools: {
        status: 'ok',
        tools: [],
        count: 16,
        error: '',
        snapshot_id: 'snap-6',
        revision: 6,
        source: 'seed_platform.workbench.CapabilitySnapshot',
        owner: 'Taiji native Workbench',
        observed_at: 1_760_000_000,
      },
      auth: { enabled: true, authenticated: true, token_valid: true, username: 'operator', has_password: true },
    }

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.workbench).toEqual({
      status: 'ok',
      count: 16,
      source: 'seed_platform.workbench.CapabilitySnapshot',
      owner: 'Taiji native Workbench',
      revision: 6,
      error: '',
    })
    expect(snapshot.auth).toEqual({ enabled: true, authenticated: true, tokenValid: true })
    expect(snapshot.unavailable).toEqual([])
  })

  it('keeps absent sections absent instead of zero-filling them', async () => {
    const { controller, runtime } = await harness()
    const { tools: droppedTools, auth: droppedAuth, ...statusWithoutSections } = runtime.runtimeStatus
    expect(droppedTools).toBeDefined()
    expect(droppedAuth).toBeDefined()
    runtime.runtimeStatus = statusWithoutSections

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.workbench).toBeUndefined()
    expect(snapshot.auth).toBeUndefined()
    // The status read itself answered: the runtime is fine, sections are not there.
    expect(snapshot.fresh).toBe(true)
    expect(snapshot.unavailable).toEqual([])
  })

  it('carries a failed capability snapshot verbatim, not as a zero count', async () => {
    const { controller, runtime } = await harness()
    runtime.runtimeStatus = {
      ...runtime.runtimeStatus,
      tools: {
        status: 'error',
        tools: [],
        count: 0,
        error: '快照构建失败: sentencepiece 缺失',
        snapshot_id: '',
        revision: 0,
        source: '',
        owner: '',
        observed_at: 0,
      },
    }

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.workbench?.status).toBe('error')
    expect(snapshot.workbench?.error).toBe('快照构建失败: sentencepiece 缺失')
  })
})
