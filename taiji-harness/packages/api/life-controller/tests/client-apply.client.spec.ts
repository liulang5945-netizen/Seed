/** Client Life face: `apply` over a captured `remote.life` namespace and a real Gateway stream carrier. */
import { Context } from '@taiji/cordis'
import type { ConnectionHandle } from '@taiji/dsh-client-connection/client'
import { RemoteStream, type RemoteStreamOptions } from '@taiji/dsh-api-gateway/client'
import { afterEach, describe, expect, it, vi } from 'vitest'
import * as LifeClientApi from '../src/client/index.ts'
import type { LifeStreamSink } from '../src/client/index.ts'
import type { LifeFollowFrame } from '../src/types.ts'

const contexts = new Set<Context>()

afterEach(async () => {
  await Promise.all([...contexts].map(async (ctx) => { await ctx.fiber.dispose() }))
  contexts.clear()
})

/** The client model stores whatever snapshot it is handed, so an identity tag is enough. */
function frame(type: LifeFollowFrame['type'], tick: number): LifeFollowFrame {
  return { type, value: { tick } as never }
}

interface MountOptions {
  /** Frames the first opening replays before it ends. */
  readonly frames?: LifeFollowFrame[]
  /** Throw instead of ending cleanly, after the frames already handed over. */
  readonly fail?: Error
  /** Hold every reopening open: a carrier loss must retry once, not forever. */
  readonly retryHolds?: boolean
  /** Refuse every verb with this value instead of accepting it. */
  readonly refused?: unknown
}

/**
 * Mount the Client half over a fake `remote.life`. A stream that ends after its
 * opening snapshot is a carrier loss, and the carrier retries: `retryHolds` keeps
 * each reopening suspended so one loss costs one retry instead of an endless loop.
 * @returns the mounted context, the failure view, and how often follow opened.
 */
async function mount(options: MountOptions = {}): Promise<{ ctx: Context; opened: number[] }> {
  const { frames = [], fail, retryHolds = true, refused } = options
  const ctx = new Context()
  contexts.add(ctx)
  const opened: number[] = []
  const connection: ConnectionHandle = {
    isLoopback: true,
    generation: { getSnapshot: () => ({ id: 1, host: { home: '/home/fixture' } }), subscribe: () => () => {} },
    state: { getSnapshot: () => 'connected' as const, subscribe: () => () => {} },
    rpc: { call: () => Promise.reject(new Error('unexpected generic RPC call')) },
    reconnect: () => {},
    registerGenerationSource: () => () => {},
    start: () => ({ stop: () => {} }),
  }
  const controlReply = () => Promise.resolve(
    refused === undefined
      ? { ok: true as const, value: { message: 'verb accepted' } }
      : { ok: false as const, error: refused as never },
  )
  const verbs = {
    snapshot: () => Promise.resolve(
      refused === undefined
        ? { ok: true as const, value: { snapshot: { tick: 99 } as never } }
        : { ok: false as const, error: refused as never },
    ),
    trainStart: controlReply,
    trainResumeCheckpoint: controlReply,
    trainPause: controlReply,
    trainResume: controlReply,
    trainStop: controlReply,
    trainReset: controlReply,
    uploadDataset: controlReply,
    deleteDataset: controlReply,
    deleteCheckpoint: controlReply,
    uploadKnowledge: controlReply,
    deleteKnowledge: controlReply,
    consolidate: controlReply,
    activateCheckpoint: controlReply,
    lifeStart: controlReply,
    lifeStop: controlReply,
    lifeAction: controlReply,
  }
  const life = {
    ...verbs,
    follow: (signal: AbortSignal): AsyncGenerator<LifeFollowFrame> => {
      void signal
      const attempt = opened.push(opened.length)
      return (async function* (): AsyncGenerator<LifeFollowFrame> {
        if (attempt > 1 && retryHolds) {
          yield* holdOpen(signal)
          return
        }
        for (const value of frames) yield value
        if (fail !== undefined) throw fail
      })()
    },
  }
  ctx.reflect.provide('remote', {
    $stream: <Item>(opts: RemoteStreamOptions<Item>) => new RemoteStream(connection, opts),
    life,
  })
  ctx.reflect.provide('remote.life', life)
  await ctx.plugin(LifeClientApi)
  return { ctx, opened }
}

/**
 * A stream that stays occupied until its signal aborts: the carrier has an open
 * attempt to wait on, so one loss costs one retry and disposal still settles.
 */
async function* holdOpen(signal: AbortSignal): AsyncGenerator<LifeFollowFrame> {
  if (signal.aborted) return
  await new Promise<void>((resolve) => {
    signal.addEventListener('abort', () => { resolve() }, { once: true })
  })
}

/** Read the settled failure the client model reports once its stream has ended or thrown. */
async function failedState(ctx: Context): Promise<{ state: string; message: string; tick: unknown }> {
  return vi.waitFor(async () => {
    const current = ctx.life.getSnapshot()
    if (current.state !== 'error') throw new Error(`expected a failed reading, received ${current.state}`)
    return {
      state: current.state,
      message: current.error?.message ?? '',
      tick: (current.snapshot as { tick?: number } | undefined)?.tick,
    }
  })
}

describe('Life Controller Client apply', () => {
  it('declares the Remote services it binds', () => {
    expect(LifeClientApi.inject).toEqual(['remote', 'remote.life'])
  })

  it('folds the baseline and each replacement into the stored snapshot', async () => {
    const { ctx, opened } = await mount({ frames: [frame('baseline', 1), frame('snapshot', 2)] })
    expect(ctx.life).toBeDefined()

    // The replay ends after being accepted, which is a carrier loss: the failure has
    // to arrive without discarding the newest snapshot, and the carrier retries once.
    const settled = await failedState(ctx)
    expect(settled.state).toBe('error')
    expect(settled.message).toContain('carrier lost')
    expect(settled.tick).toBe(2)
    await vi.waitFor(async () => { expect(opened.length).toBe(2) })
  })

  it('reports a stream that ends before its opening snapshot as a plain failure', async () => {
    const { ctx, opened } = await mount({ frames: [] })

    // Nothing was accepted, so this is a run that never got a baseline rather than a
    // lost carrier, and the model has no snapshot to keep.
    const settled = await failedState(ctx)
    expect(settled.message).toContain('before its opening snapshot')
    expect(settled.tick).toBeUndefined()
    expect(opened.length).toBe(1)
  })

  it('publishes a thrown follow failure through the sink', async () => {
    const { ctx } = await mount({ frames: [frame('baseline', 7)], fail: new Error('follow exploded') })

    const settled = await failedState(ctx)
    expect(settled.message).toContain('follow exploded')
    expect(settled.tick).toBe(7)
  })

  it('hands out a fresh state object when the failure settles', async () => {
    const { ctx } = await mount({ frames: [frame('baseline', 3), frame('snapshot', 4)] })
    const before = ctx.life.getSnapshot()
    await failedState(ctx)
    expect(ctx.life.getSnapshot()).not.toBe(before)
    expect(before.state).toBeOneOf(['idle', 'ready'])
  })

  it('builds a state stream with no carrier observer for callers that omit one', async () => {
    const { ctx } = await mount({ frames: [] })
    const accept: LifeStreamSink = {
      replaceBaseline: () => {},
      replaceSnapshot: () => {},
      handleStreamFailure: () => {},
    }

    // `apply` always observes the carrier, so the carrier-free caller is the factory's
    // own contract: building without that observer must still yield a stream.
    const stream = LifeClientApi.createLifeStateStream(ctx.remote, { accept, failed: () => {} })
    expect(stream).toBeDefined()
  })

  it('routes every control verb through the Remote namespace and keeps state readable', async () => {
    const { ctx } = await mount({ frames: [] })
    const life = ctx.life

    // `refresh` is the one verb that folds its reading back into the stored state.
    await expect(life.refresh()).resolves.toEqual({ tick: 99 })

    const calls: Array<() => Promise<unknown>> = [
      () => life.trainStart(),
      () => life.trainResumeCheckpoint({ filename: 'seed.pt' } as never),
      () => life.trainPause(),
      () => life.trainResume(),
      () => life.trainStop(),
      () => life.trainReset(),
      () => life.uploadDataset({ name: 'corpus.txt', data: '' }),
      () => life.deleteDataset({ filename: 'corpus.txt' } as never),
      () => life.deleteCheckpoint({ filename: 'seed.pt' }),
      () => life.uploadKnowledge({ name: 'notes.md', content: '' } as never),
      () => life.deleteKnowledge({ name: 'notes.md' }),
      () => life.consolidate(),
      () => life.activateCheckpoint({ filename: 'seed.pt' } as never),
      () => life.lifeStart(),
      () => life.lifeStop(),
      () => life.lifeAction({ action: 'feed', reason: 'operator' } as never),
    ]
    for (const call of calls) await expect(call()).resolves.toEqual({ message: 'verb accepted' })

    const notified: number[] = []
    const stop = life.subscribe(() => { notified.push(notified.length) })
    expect(typeof stop).toBe('function')

    // A settled reading wakes every listener that is still attached.
    await life.refresh()
    expect(notified).toEqual([0])

    stop()
    await life.refresh()
    expect(notified).toEqual([0])
  })

  it('raises the structured control failure when the Host refuses a verb', async () => {
    const { ctx } = await mount({ frames: [], refused: { code: 'life/refused', message: 'the host refused' } })
    const life = ctx.life

    await expect(life.refresh()).rejects.toThrow('the host refused')
    await expect(life.trainPause()).rejects.toThrow('the host refused')

    // The same refusal is folded into the stored state with its code intact.
    const state = life.getSnapshot()
    expect(state.state).toBe('error')
    expect(state.error?.code).toBe('life/refused')
  })

  it('normalizes a refusal that is not shaped like a failure', async () => {
    const { ctx } = await mount({ frames: [], refused: 'bare refusal' })

    // The stored state is normalized through the harness's own failure shape, so a
    // refusal that is bare text still arrives with a readable message and a code.
    await expect(ctx.life.refresh()).rejects.toBeInstanceOf(Error)
    const state = ctx.life.getSnapshot()
    expect(state.state).toBe('error')
    expect(state.error?.message).toBe('bare refusal')
    expect(state.error?.code).toBe('life/stream-failed')
  })
})
