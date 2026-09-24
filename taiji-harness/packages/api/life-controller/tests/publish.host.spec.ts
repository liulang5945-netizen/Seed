/** Model publish surface: the artifacts read and the activation verb. */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@taiji/cordis'
import { remoteErrorOf } from '@taiji/dsh-typert-protocol'
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

describe('LifeController publish read', () => {
  it('projects which checkpoint answers and which settings names', async () => {
    const { controller, runtime } = await harness()
    runtime.artifacts = {
      status: 'ok',
      artifact_types: ['taiji_checkpoint'],
      artifacts: [{ artifact_type: 'taiji_checkpoint', artifact_id: 'seed_beta.pt', path: 'seed_beta.pt', active: true }],
      runtime: { kind: 'taiji', configured_checkpoint_id: 'seed_beta.pt', active_checkpoint_id: 'resumed_seed_corpus.pt' },
      language_provider: null,
    }

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.artifacts).toEqual({ activeId: 'resumed_seed_corpus.pt', configuredId: 'seed_beta.pt' })
    expect(snapshot.unavailable).toEqual([])
  })

  it('records an unmounted artifacts endpoint as one unavailable line', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/artifacts', { status: 404 })

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.artifacts).toBeUndefined()
    expect(snapshot.unavailable).toContain('artifacts: HTTP 404')
  })

  it('refuses to invent state when the answer carries no runtime block', async () => {
    const { controller, runtime } = await harness()
    runtime.artifacts = { status: 'ok', artifacts: [] }

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.artifacts).toBeUndefined()
    expect(snapshot.unavailable).toContain('artifacts: no runtime block')
  })
})

describe('LifeController activation', () => {
  it('activates a checkpoint with its name and re-reads the publish state', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/runtime/activate', { status: 200, body: { status: 'ok', checkpoint_id: 'seed_beta.pt', active: true, runtime: {} } })

    await expect(controller.activateCheckpoint({ checkpointId: 'seed_beta.pt' }, new AbortController().signal))
      .resolves.toEqual({ message: '' })

    expect(runtime.requests).toContainEqual({ method: 'POST', path: '/api/runtime/activate', body: { checkpoint_id: 'seed_beta.pt' } })
    // The command re-reads after acceptance, so the panel sees the new active
    // state on its own action; the re-read is asynchronous by design.
    await vi.waitFor(() => {
      expect(runtime.requests.filter(entry => entry.method === 'GET' && entry.path === '/api/artifacts').length)
        .toBeGreaterThanOrEqual(1)
    })
  })

  it('sends the empty name for an explicit built-in request', async () => {
    const { controller, runtime } = await harness()

    await controller.activateCheckpoint({ checkpointId: '' }, new AbortController().signal)

    expect(runtime.requests).toContainEqual({ method: 'POST', path: '/api/runtime/activate', body: { checkpoint_id: '' } })
  })

  it('maps a missing checkpoint with the runtime\'s own detail', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/runtime/activate', { status: 404, body: { detail: '检查点不存在: gone.pt' } })

    const failure = remoteErrorOf(await controller.activateCheckpoint({ checkpointId: 'gone.pt' }, new AbortController().signal).catch((error: unknown) => error))

    expect(failure).toMatchObject({ code: 'life/runtime-error' })
    expect(failure?.message).toContain('检查点不存在: gone.pt')
  })

  it('maps a failed activation — the runtime also marks its startup failed', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/runtime/activate', { status: 500, body: { detail: 'Taiji runtime 激活失败: 权重损坏' } })

    const failure = remoteErrorOf(await controller.activateCheckpoint({ checkpointId: 'broken.pt' }, new AbortController().signal).catch((error: unknown) => error))

    expect(failure).toMatchObject({ code: 'life/runtime-error', details: { status: 500 } })
    expect(failure?.message).toContain('权重损坏')
  })
})
