/** Resumed training: `POST /api/train/resume_checkpoint` over the shared stream contract. */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@taiji/cordis'
import { remoteErrorOf } from '@taiji/dsh-typert-protocol'
import LifeController from '../src/index.ts'
import {
  closeMockRuntimes,
  completedFrame,
  mockLifeRuntime,
  progressFrame,
  warningFrame,
  type MockLifeRuntime,
} from './mock-runtime.ts'

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

describe('LifeController resumed training', () => {
  it('sends the checkpoint with its datasets and folds the stream', async () => {
    const { controller, runtime } = await harness()
    runtime.resume = { kind: 'frames', frames: [progressFrame({ fraction: 0.5, step: 500 }), completedFrame({ message: 'resumed run complete' })] }

    await expect(controller.trainResumeCheckpoint({ checkpoint: 'seed_beta.pt', datasets: ['consolidated/night-1.jsonl'] }))
      .resolves.toEqual({ message: 'training accepted' })
    expect(runtime.requests).toContainEqual({
      method: 'POST',
      path: '/api/train/resume_checkpoint',
      body: { checkpoint: 'seed_beta.pt', datasets: ['consolidated/night-1.jsonl'] },
    })

    const settled = await vi.waitFor(async () => {
      const { snapshot } = await controller.snapshot(new AbortController().signal)
      expect(snapshot.availability.trainingStream).toBe('idle')
      return snapshot
    })
    expect(settled.training.progress).toBeUndefined()
  })

  it('omits absent optional fields so the runtime keeps its own defaults', async () => {
    const { controller, runtime } = await harness()

    await expect(controller.trainResumeCheckpoint({ checkpoint: 'resumed_seed_corpus.pt' })).resolves.toBeDefined()

    expect(runtime.requests).toContainEqual({
      method: 'POST',
      path: '/api/train/resume_checkpoint',
      body: { checkpoint: 'resumed_seed_corpus.pt' },
    })
  })

  it('carries a corpus-drift warning into the snapshot and keeps it after the run settles', async () => {
    const { controller, runtime } = await harness()
    const drift = '语料已变更：检查点原指纹 abc123，本次续训 def456'
    runtime.resume = { kind: 'frames', frames: [warningFrame(drift), completedFrame()] }

    await controller.trainResumeCheckpoint({ checkpoint: 'seed_beta.pt', maxTicks: 1_000 })

    const warned = await vi.waitFor(async () => {
      const { snapshot } = await controller.snapshot(new AbortController().signal)
      expect(snapshot.training.warnings).toEqual([drift])
      return snapshot
    })
    // The tick cap reaches the runtime as its own field name, not a panel invention.
    expect(runtime.requests).toContainEqual({
      method: 'POST',
      path: '/api/train/resume_checkpoint',
      body: { checkpoint: 'seed_beta.pt', max_ticks: 1_000 },
    })
    // A settled stream still carries the warning: it is a fact about the last run.
    expect(warned.availability.trainingStream).toBe('idle')
  })

  it('clears the previous run\'s warnings when a new run is accepted', async () => {
    const { controller, runtime } = await harness()
    runtime.resume = { kind: 'frames', frames: [warningFrame('第一次的漂移警告'), completedFrame()] }
    await controller.trainResumeCheckpoint({ checkpoint: 'seed_beta.pt' })
    await vi.waitFor(async () => {
      const { snapshot } = await controller.snapshot(new AbortController().signal)
      expect(snapshot.training.warnings).toEqual(['第一次的漂移警告'])
    })

    runtime.training = { kind: 'frames', frames: [progressFrame(), completedFrame()] }
    await controller.trainStart({})
    // The previous run's warning is the only fact distinguishing the runs'
    // frames here (both settle to `idle`), so waiting on its clearing is the
    // observation that a new accepted run resets the warning state.
    const fresh = await vi.waitFor(async () => {
      const { snapshot } = await controller.snapshot(new AbortController().signal)
      expect(snapshot.training.warnings).toBeUndefined()
      return snapshot
    })
    expect(fresh.training.progress).toBeUndefined()
  })

  it('maps a refused resume with the runtime\'s own detail', async () => {
    const { controller, runtime } = await harness()
    runtime.resume = { kind: 'http-error', status: 404, body: JSON.stringify({ detail: '检查点不存在: gone.pt' }) }

    const failure = remoteErrorOf(await controller.trainResumeCheckpoint({ checkpoint: 'gone.pt' }).catch((error: unknown) => error))

    expect(failure).toMatchObject({ code: 'life/runtime-error' })
    expect(failure?.message).toContain('检查点不存在: gone.pt')
    const { snapshot } = await controller.snapshot(new AbortController().signal)
    expect(snapshot.availability.trainingStream).toBe('idle')
  })

  it('refuses a second run while a resumed stream is open, from either verb', async () => {
    const { controller, runtime } = await harness()
    runtime.resume = { kind: 'frames', frames: [progressFrame()], hold: true }

    await expect(controller.trainResumeCheckpoint({ checkpoint: 'seed_beta.pt' })).resolves.toBeDefined()
    const failure = remoteErrorOf(await controller.trainStart({}).catch((error: unknown) => error))
    expect(failure).toMatchObject({ code: 'life/conflict' })
    await runtime.closeTraining()
  })
})
