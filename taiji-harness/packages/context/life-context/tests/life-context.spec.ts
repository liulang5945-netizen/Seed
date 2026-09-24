import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@taiji/cordis'
import { createUserMessage } from '@taiji/dsh-llm'
import type { UserMessage } from '@taiji/dsh-llm'
import { agentEvents } from '@taiji/dsh-agent'
import type { Agent, PreStepDecision } from '@taiji/dsh-agent'
import type { LifeSnapshot } from '@taiji/dsh-api-life-controller/types'
import * as lifeContext from '../src/index.ts'
import type { Config } from '../src/index.ts'

const BASE = Date.parse('2026-09-23T12:00:00.000Z')
const SIGNAL = new AbortController().signal

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(BASE)
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.useRealTimers()
})

function nativeSnapshot(overrides: Partial<LifeSnapshot> = {}): LifeSnapshot {
  return {
    source: 'native',
    observedAt: new Date(BASE).toISOString(),
    fresh: true,
    pollIntervalMs: 5_000,
    life: {
      isRunning: true,
      native: { tick: 41, mode: 'wake', needs: { curiosity: 42.5, fatigue: 10, stress: 1.5 }, drives: { exploration: 40, replay: 10, rest: 0, play: 30 } },
    },
    training: { isTraining: false, pauseRequested: false, stopRequested: false, publishing: false, checkpoints: [] },
    knowledge: { docCount: 12, chunkCount: 340, hasEmbeddings: true, embedDim: 384 },
    availability: { runtime: 'ok', legacy: 'disabled', knowledge: 'ok', trainingStream: 'idle' },
    unavailable: [],
    ...overrides,
  }
}

/**
 * The same snapshot with its optional readings *absent*, which is how the
 * controller reports a source that never answered.  Absence is built by
 * omission rather than by assigning `undefined`: the readings are optional
 * properties under `exactOptionalPropertyTypes`.
 */
function withoutReadings(snapshot: LifeSnapshot): LifeSnapshot {
  const { health: _health, memory: _memory, life: _life, knowledge: _knowledge, ...rest } = snapshot
  return rest
}

interface Mount {
  readonly ctx: Context
  readonly sections: Array<{ readonly name: string; readonly order: number; readonly text: string }>
  readonly reads: { count: number }
  setSnapshot(snapshot: LifeSnapshot): void
}

async function mount(config: Config = {}, snapshot: LifeSnapshot = nativeSnapshot()): Promise<Mount> {
  const ctx = new Context()
  const sections: Mount['sections'] = []
  ctx.provide('systemPrompt', {
    getSectionOrder: () => 700,
    section: (entry: { name: string; order: number; text: string }) => { sections.push(entry) },
  } as never)
  const reads = { count: 0 }
  let current = snapshot
  ctx.provide('lifeController', {
    snapshot: async (signal: AbortSignal) => {
      reads.count += 1
      if (signal.aborted) throw signal.reason
      return { snapshot: current }
    },
  } as never)
  await ctx.plugin(lifeContext, config)
  return { ctx, sections, reads, setSnapshot(next: LifeSnapshot) { current = next } }
}

const AGENT = { id: 'agent', options: {}, status: 'running', ctx: new Context() } as unknown as Agent

async function fire(ctx: Context, signal: AbortSignal = SIGNAL): Promise<readonly UserMessage[]> {
  const proposed = createUserMessage({ content: [{ type: 'text', text: 'request' }], source: { kind: 'user' } })
  const decision: PreStepDecision = await agentEvents(ctx, AGENT).waterfall(
    'agent/pre-step',
    { messages: [proposed], turn: 1, step: 1, signal },
    () => Promise.resolve({ kind: 'enter' as const, messages: [proposed] }),
  )
  if (decision.kind !== 'enter') throw new Error('unexpected reject')
  return decision.messages
}

describe('life-context policy section', () => {
  it('registers the static policy section at the centrally allocated placement', async () => {
    const { sections } = await mount()
    expect(sections).toHaveLength(1)
    expect(sections[0]).toMatchObject({ name: 'life:policy', order: 700 })
    expect(sections[0]!.text).toContain('internal telemetry')
  })

  it('registers nothing when disabled', async () => {
    const { sections } = await mount({ enabled: false })
    expect(sections).toEqual([])
  })

  it('rejects an interval or budget that cannot carry a reading', async () => {
    await expect(mount({ refreshIntervalMs: -1 })).rejects.toThrow('refreshIntervalMs')
    await expect(mount({ maxChars: 10 })).rejects.toThrow('maxChars')
  })
})

describe('life-context pre-step injection', () => {
  it('appends one durable life-context message on the first eligible step', async () => {
    const { ctx } = await mount()
    const messages = await fire(ctx)
    expect(messages).toHaveLength(2)
    const reading = messages[1]
    expect(reading?.source.kind).toBe('life-context')
    expect(reading?.content).toEqual([{
      type: 'text',
      text: 'life-state age=0s source=native tick=41 mode=wake'
        + ' needs[curiosity=42.5 fatigue=10 stress=1.5]'
        + ' drives[exploration=40 replay=10 rest=0 play=30]'
        + ' training[off] knowledge[12 docs 340 chunks]',
    }])
  })

  it('stays quiet within the refresh interval and injects again after it', async () => {
    const { ctx } = await mount()
    expect(await fire(ctx)).toHaveLength(2)
    vi.setSystemTime(BASE + 1_000)
    expect(await fire(ctx)).toHaveLength(1)
    vi.setSystemTime(BASE + 31_000)
    expect(await fire(ctx)).toHaveLength(2)
  })

  it('injects immediately on a significant change inside the interval', async () => {
    const harness = await mount()
    expect(await fire(harness.ctx)).toHaveLength(2)
    vi.setSystemTime(BASE + 2_000)
    harness.setSnapshot(nativeSnapshot({
      life: {
        isRunning: true,
        native: { tick: 42, mode: 'wake', needs: { curiosity: 80, fatigue: 10, stress: 1.5 }, drives: { exploration: 40, replay: 10, rest: 0, play: 30 } },
      },
    }))
    const messages = await fire(harness.ctx)
    expect(messages).toHaveLength(2)
    if (messages[1] === undefined) throw new Error('expected a reading')
    expect(messages[1].source.kind).toBe('life-context')
  })

  it('omits an unreachable runtime without inventing a reading', async () => {
    const harness = await mount({}, withoutReadings(nativeSnapshot({
      availability: { runtime: 'down', legacy: 'down', knowledge: 'down', trainingStream: 'idle' },
    })))
    expect(await fire(harness.ctx)).toHaveLength(1)
  })

  it('omits a stale reading and keeps the context clean', async () => {
    const harness = await mount()
    vi.setSystemTime(BASE + 61_000)
    expect(await fire(harness.ctx)).toHaveLength(1)
  })
})
