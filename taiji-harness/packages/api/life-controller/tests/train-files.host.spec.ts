/** Trainable roster: `GET /api/train/files` feeds the panel's dataset chooser. */
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

/** One `GET /api/train/files` body, shaped as the runtime answers it. */
const TRAIN_FILES = {
  files: ['consolidated/night-1.jsonl', 'simple_zh/dialogue_extended_clean.jsonl'],
  entries: [
    { path: 'consolidated/night-1.jsonl', size_bytes: 4096 },
    { path: 'simple_zh/dialogue_extended_clean.jsonl', size_bytes: 108_327_171 },
    // A pathless row carries nothing a chooser can offer and is dropped.
    { path: '', size_bytes: 7 },
  ],
}

describe('LifeController training roster', () => {
  it('projects the trainable roster with its runtime-owned order and sizes', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/train/files', { status: 200, body: TRAIN_FILES })

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.training.datasets).toEqual([
      { path: 'consolidated/night-1.jsonl', sizeBytes: 4096 },
      { path: 'simple_zh/dialogue_extended_clean.jsonl', sizeBytes: 108_327_171 },
    ])
    expect(snapshot.availability.runtime).toBe('ok')
    expect(snapshot.unavailable).toEqual([])
  })

  it('records a missing roster endpoint as one unavailable line, not a failure', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/train/files', { status: 404 })

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.training.datasets).toBeUndefined()
    expect(snapshot.availability.runtime).toBe('ok')
    expect(snapshot.fresh).toBe(true)
    expect(snapshot.unavailable).toContain('train-files: HTTP 404')
  })

  it('leaves the roster absent when a 200 answer carried none', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/train/files', { status: 200, body: { status: 'ok' } })

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.training.datasets).toBeUndefined()
    expect(snapshot.unavailable).toEqual([])
  })

  it('skips the roster read once the runtime is down', async () => {
    const { controller, runtime } = await harness()
    await runtime.close()

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.training.datasets).toBeUndefined()
    expect(snapshot.availability.runtime).toBe('down')
    expect(snapshot.unavailable.join(' ')).toContain('life/runtime-unreachable')
  })
})
