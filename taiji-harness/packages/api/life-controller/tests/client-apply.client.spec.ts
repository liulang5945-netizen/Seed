/** Client Life face: `apply` over a captured `remote.life` namespace and a real Gateway stream carrier. */
import { Context } from '@taiji/cordis'
import type { ConnectionHandle } from '@taiji/dsh-client-connection/client'
import { RemoteStream, type RemoteStreamOptions } from '@taiji/dsh-api-gateway/client'
import { afterEach, describe, expect, it, vi } from 'vitest'
import * as LifeClientApi from '../src/client/index.ts'
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
}

/**
 * Mount the Client half over a fake `remote.life`. A stream that ends after its
 * opening snapshot is a carrier loss, and the carrier retries: `retryHolds` keeps
 * each reopening suspended so one loss costs one retry instead of an endless loop.
 * @returns the mounted context, the failure view, and how often follow opened.
 */
async function mount(options: MountOptions = {}): Promise<{ ctx: Context; opened: number[] }> {
  const { frames = [], fail, retryHolds = true } = options
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
  const life = {
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
})
