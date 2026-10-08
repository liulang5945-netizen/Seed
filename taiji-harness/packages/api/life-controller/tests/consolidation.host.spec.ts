/** Consolidation face: the status read, its failure lines, and the pass verb. */
import { afterEach, describe, expect, it } from 'vitest'
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

/** One `GET /api/consolidation/status` body, shaped as `sleep_pass.status()` answers. */
const CONSOLIDATION_STATUS = {
  available: true,
  directory: 'data/consolidated',
  passes: 2,
  last_pass_at: 1_760_000_100,
  last_corpus: 'consolidated/corpus-20260923T080000Z-pass-2.jsonl',
  projected_digests: 3,
  running: false,
  spec: {
    reason: 'interaction journal holds 3 entries',
    weaknesses: ['recency'],
    datasets: ['consolidated/night-1.jsonl'],
  },
  last_report: {
    pass_id: 'pass-2',
    reason: 'manual',
    duration_ms: 1_500,
    weaknesses: ['recency'],
    notes: ['corpus written'],
    projection: { by_source: { constraints: 2, interactions: 1, workbench_capabilities: 16 } },
    spec: { reason: 'interaction journal holds 3 entries' },
  },
  journal: { entries: 4, by_kind: { interaction: 3, reflection: 1 }, sessions: 2, last_recorded_at: 1_760_000_000 },
}

describe('LifeController consolidation read', () => {
  it('projects the consolidation status with its journal, spec, and report', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/consolidation/status', { status: 200, body: CONSOLIDATION_STATUS })

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.consolidation).toEqual({
      passes: 2,
      lastPassAt: 1_760_000_100,
      lastCorpus: 'consolidated/corpus-20260923T080000Z-pass-2.jsonl',
      projectedDigests: 3,
      running: false,
      spec: {
        reason: 'interaction journal holds 3 entries',
        datasets: ['consolidated/night-1.jsonl'],
        weaknesses: ['recency'],
      },
      lastReport: {
        reason: 'manual',
        specReason: 'interaction journal holds 3 entries',
        durationMs: 1_500,
        weaknesses: ['recency'],
        notes: ['corpus written'],
        workbenchCapabilities: 16,
      },
      journal: { entries: 4, byKind: { interaction: 3, reflection: 1 }, sessions: 2, lastRecordedAt: 1_760_000_000 },
    })
    expect(snapshot.availability.runtime).toBe('ok')
    expect(snapshot.unavailable).toEqual([])
  })

  it('reads an absent workbench projection as zero rather than guessing one', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/consolidation/status', {
      status: 200,
      body: {
        ...CONSOLIDATION_STATUS,
        last_report: { ...CONSOLIDATION_STATUS.last_report, projection: undefined },
      },
    })

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.consolidation?.lastReport?.workbenchCapabilities).toBe(0)
  })

  it('records an unmounted consolidation endpoint as one unavailable line, not a failure', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/consolidation/status', { status: 404 })

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.consolidation).toBeUndefined()
    expect(snapshot.availability.runtime).toBe('ok')
    expect(snapshot.fresh).toBe(true)
    expect(snapshot.unavailable).toContain('consolidation: HTTP 404')
  })

  it('skips the consolidation read once the runtime is down', async () => {
    const { controller, runtime } = await harness()
    await runtime.close()

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.consolidation).toBeUndefined()
    expect(snapshot.availability.runtime).toBe('down')
    expect(snapshot.unavailable.join(' ')).toContain('life/runtime-unreachable')
  })
})

describe('LifeController consolidation control', () => {
  it('runs one pass with an explicit reason and returns the runtime message', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/consolidate', { status: 200, body: { status: 'ok', message: 'pass 3 written' } })

    await expect(controller.consolidate({ reason: 'panel' }, new AbortController().signal))
      .resolves.toEqual({ message: 'pass 3 written' })

    expect(runtime.requests).toContainEqual({ method: 'POST', path: '/api/consolidate', body: { reason: 'panel' } })
  })

  it('always sends a reason body, defaulting to the runtime’s own `manual`', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/consolidate', { status: 200, body: { status: 'ok', message: 'pass written' } })

    await expect(controller.consolidate({}, new AbortController().signal)).resolves.toEqual({ message: 'pass written' })

    expect(runtime.requests).toContainEqual({ method: 'POST', path: '/api/consolidate', body: { reason: 'manual' } })
  })

  it('maps a runtime refusal with the runtime’s own detail', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/consolidate', { status: 409, body: { detail: '巩固已在运行' } })

    const failure = remoteErrorOf(await controller.consolidate({}, new AbortController().signal).catch((error: unknown) => error))

    expect(failure).toMatchObject({ code: 'life/conflict' })
    expect(failure?.message).toContain('巩固已在运行')
  })
})
