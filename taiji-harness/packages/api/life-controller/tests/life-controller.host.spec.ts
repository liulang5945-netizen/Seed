import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@taiji/cordis'
import { remoteErrorOf } from '@taiji/dsh-typert-protocol'
import type { LifeFollowFrame } from '../src/types.ts'
import LifeController from '../src/index.ts'
import {
  closeMockRuntimes,
  completedFrame,
  errorFrame,
  mockLifeRuntime,
  progressFrame,
  type MockLifeRuntime,
} from './mock-runtime.ts'

const roots: Context[] = []

afterEach(async () => {
  await Promise.all(roots.splice(0).map(ctx => ctx.fiber.dispose()))
  await closeMockRuntimes()
})

interface Harness {
  readonly controller: LifeController
  readonly ctx: Context
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
  return { controller, ctx, runtime }
}

async function nextFrame(iterator: AsyncIterator<LifeFollowFrame>): Promise<LifeFollowFrame> {
  const next = await iterator.next()
  if (next.done === true) throw new Error('Life stream ended before the expected frame')
  return next.value
}

describe('LifeController snapshot', () => {
  it('reports a native reading with its source, availability, and gated surfaces', async () => {
    const { controller, runtime } = await harness()
    runtime.checkpoints = [
      { filename: 'seed_native.pt', step: 1000, bytes: 2048, modified_utc: '2026-09-23T10:00:00', saved_at_utc: '2026-09-23T09:59:00', num_epochs: 1 },
    ]
    runtime.knowledgeReply = {
      status: 200,
      body: { status: 'ok', doc_count: 12, chunk_count: 340, has_embeddings: true, embed_dim: 384 },
    }
    runtime.knowledgeFilesReply = {
      status: 200,
      body: { files: [{ name: 'handbook.md', size: 4096, mtime: 1_760_000_000, status: 'indexed' }] },
    }

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.source).toBe('native')
    expect(snapshot.fresh).toBe(true)
    expect(snapshot.health).toMatchObject({ state: 'ok', seedActive: true, modelName: 'taiji-native' })
    expect(snapshot.memory).toMatchObject({ totalGb: 32, availableGb: 12, usedPct: 62.5 })
    expect(snapshot.life).toMatchObject({
      isRunning: true,
      native: { tick: 41, mode: 'wake', needs: { curiosity: 42.5 }, drives: { exploration: 40 } },
    })
    expect(snapshot.life?.legacy).toBeUndefined()
    expect(snapshot.training.checkpoints).toEqual([
      {
        filename: 'seed_native.pt',
        step: 1000,
        bytes: 2048,
        modifiedUtc: '2026-09-23T10:00:00',
        savedAtUtc: '2026-09-23T09:59:00',
        numEpochs: 1,
      },
    ])
    expect(snapshot.knowledge).toEqual({
      docCount: 12,
      chunkCount: 340,
      hasEmbeddings: true,
      embedDim: 384,
      files: [{ name: 'handbook.md', sizeBytes: 4096, status: 'indexed' }],
    })
    // The Legacy life surface answered 404 by default: that is "not mounted", not "broken".
    expect(snapshot.availability).toEqual({
      runtime: 'ok',
      legacy: 'disabled',
      knowledge: 'ok',
      trainingStream: 'idle',
    })
    expect(snapshot.unavailable).toEqual([])
  })

  it('reports the Legacy organ when the runtime serves it', async () => {
    const { controller, runtime } = await harness()
    runtime.runtimeStatus = {
      ...runtime.runtimeStatus,
      life: {
        status: 'ok',
        is_running: true,
        needs: { hunger: 12, fatigue: 80, boredom: 3, stress: 1, curiosity: 44 },
        life_state: 'sleeping',
        dominant_need: 'fatigue',
        total_heartbeats: 137,
        total_events: 12,
        last_heartbeat: '2026-09-23T10:00:00',
        last_activity: '2026-09-23T09:59:00',
      },
    }
    runtime.legacyReply = { status: 200, body: { is_running: true } }

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.source).toBe('legacy')
    expect(snapshot.life).toMatchObject({
      isRunning: true,
      legacy: {
        lifeState: 'sleeping',
        dominantNeed: 'fatigue',
        totalHeartbeats: 137,
        totalEvents: 12,
        lastHeartbeat: '2026-09-23T10:00:00',
      },
    })
    expect(snapshot.life?.native).toBeUndefined()
    expect(snapshot.availability.legacy).toBe('ok')
  })

  it('reports an unreachable runtime as down without inventing a reading', async () => {
    const { controller, runtime } = await harness()
    await runtime.close()

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.availability.runtime).toBe('down')
    expect(snapshot.fresh).toBe(false)
    expect(snapshot.health).toBeUndefined()
    expect(snapshot.life).toBeUndefined()
    expect(snapshot.source).toBe('absent')
    expect(snapshot.training).toMatchObject({ isTraining: false, checkpoints: [] })
    expect(snapshot.training.progress).toBeUndefined()
    expect(snapshot.unavailable.join(' ')).toContain('life/runtime-unreachable')
  })
})

describe('LifeController follow', () => {
  it('opens with the current snapshot and publishes one replacement per change', async () => {
    const { controller, runtime } = await harness()
    const abort = new AbortController()
    const iterator = controller.follow(abort.signal)[Symbol.asyncIterator]()

    const baseline = await nextFrame(iterator)
    expect(baseline.type).toBe('baseline')

    runtime.runtimeStatus = {
      ...runtime.runtimeStatus,
      life: { status: 'seed', is_running: true, needs: { curiosity: 90, fatigue: 1, stress: 0 }, mode: 'wake', tick: 99 },
    }

    const next = await vi.waitFor(async () => {
      const frame: LifeFollowFrame = await nextFrame(iterator)
      if (frame.type !== 'snapshot') throw new Error(`expected a replacement frame, received ${frame.type}`)
      return frame
    })
    expect(next.value.life?.native?.tick).toBe(99)
    abort.abort()
  })

  it('keeps publishing to later consumers after one closes its stream early', async () => {
    const { controller, runtime } = await harness()
    const first = controller.follow(new AbortController().signal)[Symbol.asyncIterator]()
    expect((await nextFrame(first)).type).toBe('baseline')

    // The consumer stops before any replacement frame arrives, so its queue is
    // dropped on the way out and the stream reports completion.
    await first.return?.()
    expect((await first.next()).done).toBe(true)

    runtime.runtimeStatus = {
      ...runtime.runtimeStatus,
      life: { status: 'seed', is_running: true, needs: { curiosity: 90, fatigue: 1, stress: 0 }, mode: 'wake', tick: 120 },
    }

    const second = controller.follow(new AbortController().signal)[Symbol.asyncIterator]()
    expect((await nextFrame(second)).type).toBe('baseline')

    runtime.runtimeStatus = {
      ...runtime.runtimeStatus,
      life: { status: 'seed', is_running: true, needs: { curiosity: 90, fatigue: 1, stress: 0 }, mode: 'wake', tick: 130 },
    }

    // Any frame carrying the newest tick proves the closed consumer left no
    // residue in the poll loop's publish path.
    await vi.waitFor(async () => {
      const frame: LifeFollowFrame = await nextFrame(second)
      if (frame.value.life?.native?.tick !== 130) {
        throw new Error(`expected tick 130, received ${String(frame.value.life?.native?.tick)}`)
      }
    })
  })
})

describe('LifeController training control', () => {
  it('sends every training and life control verb and returns the runtime message', async () => {
    const { controller, runtime } = await harness()
    const verbs: Array<() => Promise<{ message: string }>> = [
      () => controller.trainPause(new AbortController().signal),
      () => controller.trainResume(new AbortController().signal),
      () => controller.trainReset(new AbortController().signal),
    ]

    for (const call of verbs) {
      const before = runtime.requests.length
      const value = await call()
      // The stand-in answers every control verb with the path it was reached on,
      // so the reply itself names the request the verb must have sent. The
      // control path re-reads afterwards, which is why the tail is a set here.
      const path = value.message.endsWith(' accepted') ? value.message.slice(0, -' accepted'.length) : ''
      expect(path).not.toBe('')
      expect(runtime.requests.slice(before).map(entry => entry.path)).toContain(path)
    }

    // The Legacy scheduler verb is gated on this runtime: the refusal has to
    // reach the caller as a refusal, after the request went out.
    await expect(controller.lifeStop(new AbortController().signal)).rejects.toThrow(/Legacy life surface/u)
  })

  it('starts a run and folds progress while the stream stays open', async () => {
    const { controller, runtime } = await harness()
    runtime.training = { kind: 'frames', frames: [progressFrame({ fraction: 0.5, step: 500 })], hold: true }

    await expect(controller.trainStart({ parameterBudget: 1_000 })).resolves.toEqual({ message: 'training accepted' })
    expect(runtime.requests.find(entry => entry.path === '/api/train/native')?.body).toEqual({ parameter_budget: 1_000 })

    const streaming = await vi.waitFor(async () => {
      const { snapshot } = await controller.snapshot(new AbortController().signal)
      expect(snapshot.training.progress?.fraction).toBe(0.5)
      return snapshot
    })
    expect(streaming.availability.trainingStream).toBe('streaming')
    expect(streaming.training.progress).toMatchObject({ step: 500, loss: 1.5, totalSteps: 1000 })
    await runtime.closeTraining()
  })

  it('settles on completion and re-reads the checkpoint roster', async () => {
    const { controller, runtime } = await harness()
    runtime.checkpoints = [{ filename: 'resumed.pt', step: 750, bytes: 10, modified_utc: '', saved_at_utc: '', num_epochs: 1 }]
    runtime.training = { kind: 'frames', frames: [progressFrame({ fraction: 0.5 }), completedFrame()] }

    await expect(controller.trainStart({})).resolves.toEqual({ message: 'training accepted' })

    const settled = await vi.waitFor(async () => {
      const { snapshot } = await controller.snapshot(new AbortController().signal)
      expect(snapshot.availability.trainingStream).toBe('idle')
      return snapshot
    })
    expect(settled.training.progress).toBeUndefined()
    expect(settled.training.checkpoints.map(row => row.filename)).toEqual(['resumed.pt'])
  })

  it('refuses a second run while the first stream is open', async () => {
    const { controller, runtime } = await harness()
    runtime.training = { kind: 'frames', frames: [progressFrame()], hold: true }

    await expect(controller.trainStart({})).resolves.toEqual({ message: 'training accepted' })
    const failure = await controller.trainStart({}).catch((error: unknown) => error)

    expect(remoteErrorOf(failure)).toMatchObject({ code: 'life/conflict' })
    await runtime.closeTraining()
  })

  it('surfaces a refused run and clears the stream state', async () => {
    const { controller, runtime } = await harness()
    runtime.training = { kind: 'http-error', status: 409, body: JSON.stringify({ detail: '训练已在运行' }) }

    const failure = remoteErrorOf(await controller.trainStart({}).catch((error: unknown) => error))

    expect(failure).toMatchObject({ code: 'life/conflict' })
    expect(failure?.message).toContain('训练已在运行')
    const { snapshot } = await controller.snapshot(new AbortController().signal)
    expect(snapshot.availability.trainingStream).toBe('idle')
    expect(snapshot.training.progress).toBeUndefined()
  })

  it('maps a runtime failure with the runtime’s own detail', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/train/stop', { status: 500, body: { detail: '锁释放失败' } })

    const failure = await controller.trainStop(new AbortController().signal).catch((error: unknown) => error)

    expect(remoteErrorOf(failure)).toMatchObject({
      code: 'life/runtime-error',
      details: { status: 500, detail: '锁释放失败' },
    })
  })

  it('reports a gated Legacy verb as unavailable rather than broken', async () => {
    const { controller, runtime } = await harness()

    const failure = await controller.lifeStart(new AbortController().signal).catch((error: unknown) => error)

    expect(remoteErrorOf(failure)).toMatchObject({ code: 'life/unavailable', details: { source: 'life' } })
    expect(runtime.requests.some(entry => entry.path === '/api/taiji/life/start')).toBe(true)
  })

  it('sends a forced Legacy activity with its reason', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/taiji/life/action/feed', { status: 200, body: { status: 'ok', message: 'fed' } })

    await expect(controller.lifeAction({ action: 'feed', reason: 'operator' }, new AbortController().signal))
      .resolves.toEqual({ message: 'fed' })

    expect(runtime.requests).toContainEqual({
      method: 'POST',
      path: '/api/taiji/life/action/feed',
      body: { reason: 'operator' },
    })
  })

  it('reports a mid-run stream failure through the snapshot availability', async () => {
    const { controller, runtime } = await harness()
    runtime.training = { kind: 'frames', frames: [progressFrame(), errorFrame('语料缺失')] }

    await expect(controller.trainStart({})).resolves.toEqual({ message: 'training accepted' })

    const failed = await vi.waitFor(async () => {
      const { snapshot } = await controller.snapshot(new AbortController().signal)
      expect(snapshot.availability.trainingStream).toBe('closed')
      return snapshot
    })
    expect(failed.training.progress).toBeUndefined()
    expect(failed.training.isTraining).toBe(false)
  })
})
