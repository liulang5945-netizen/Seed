import { describe, expect, it } from 'vitest'
import type { LifeSnapshot } from '@taiji/dsh-api-life-controller/types'
import { NEED_DRIFT, STALE_AFTER_MS, renderLifeState, significantChange } from '../src/reading.ts'

const BASE = Date.parse('2026-09-23T12:00:00.000Z')

function nativeSnapshot(overrides: Partial<LifeSnapshot> = {}): LifeSnapshot {
  return {
    source: 'native',
    observedAt: new Date(BASE).toISOString(),
    fresh: true,
    pollIntervalMs: 5_000,
    health: { state: 'ok', modelLoaded: true, modelName: 'taiji-native', seedActive: true, startupComplete: true },
    memory: { totalGb: 32, availableGb: 12, usedPct: 62.5 },
    life: {
      isRunning: true,
      native: {
        tick: 41,
        mode: 'wake',
        needs: { curiosity: 42.5, fatigue: 10, stress: 1.5 },
        drives: { exploration: 40, replay: 10, rest: 0, play: 30 },
      },
    },
    training: { isTraining: false, pauseRequested: false, stopRequested: false, publishing: false, checkpoints: [] },
    knowledge: { docCount: 12, chunkCount: 340, hasEmbeddings: true, embedDim: 384 },
    availability: { runtime: 'ok', legacy: 'disabled', knowledge: 'ok', trainingStream: 'idle' },
    unavailable: [],
    ...overrides,
  }
}

describe('renderLifeState', () => {
  it('renders a native reading with needs, drives, training, and knowledge', () => {
    const reading = renderLifeState(nativeSnapshot(), BASE + 5_000, 400)
    expect(reading).toEqual({
      kind: 'inject',
      text: 'life-state age=5s source=native tick=41 mode=wake'
        + ' needs[curiosity=42.5 fatigue=10 stress=1.5]'
        + ' drives[exploration=40 replay=10 rest=0 play=30]'
        + ' training[off] knowledge[12 docs 340 chunks]',
    })
  })

  it('renders a legacy reading with its scheduler counters and omits empty segments', () => {
    const snapshot = nativeSnapshot({
      life: {
        isRunning: true,
        legacy: {
          isRunning: true,
          lifeState: 'sleeping',
          dominantNeed: 'fatigue',
          needs: { hunger: 12, fatigue: 80, boredom: 3, stress: 1, curiosity: 44 },
          totalHeartbeats: 137,
          totalEvents: 12,
          lastHeartbeat: '2026-09-23T11:59:00',
          lastActivity: '2026-09-23T11:58:00',
        },
      },
      knowledge: undefined,
    })
    const reading = renderLifeState(snapshot, BASE + 8_000, 400)
    expect(reading).toEqual({
      kind: 'inject',
      text: 'life-state age=8s source=legacy state=sleeping dominant=fatigue'
        + ' needs[hunger=12 fatigue=80 boredom=3 stress=1 curiosity=44]'
        + ' heartbeats=137 events=12 training[off]',
    })
  })

  it('renders the training state with modifiers and live progress', () => {
    const snapshot = nativeSnapshot({
      training: {
        isTraining: true,
        pauseRequested: true,
        stopRequested: false,
        publishing: false,
        checkpoints: [],
        progress: { fraction: 0.5, step: 500, loss: 1.25, elapsed: 10, eta: 10, epoch: 1, totalEpochs: 1, samplesPerSec: 100, totalSteps: 1000 },
      },
      availability: { runtime: 'ok', legacy: 'disabled', knowledge: 'disabled', trainingStream: 'streaming' },
      knowledge: undefined,
    })
    const reading = renderLifeState(snapshot, BASE + 1_000, 400)
    expect(reading).toEqual({
      kind: 'inject',
      text: 'life-state age=1s source=native tick=41 mode=wake'
        + ' needs[curiosity=42.5 fatigue=10 stress=1.5]'
        + ' drives[exploration=40 replay=10 rest=0 play=30]'
        + ' training[running pausing loss=1.3 step=500/1000]',
    })
  })

  it('reports a closed training stream', () => {
    const snapshot = nativeSnapshot({
      availability: { runtime: 'ok', legacy: 'disabled', knowledge: 'ok', trainingStream: 'closed' },
    })
    const reading = renderLifeState(snapshot, BASE, 400)
    if (reading.kind !== 'inject') throw new Error('expected an injectable reading')
    expect(reading.text).toContain('training[off stream-closed]')
  })

  it('omits an unreachable runtime without inventing a reading', () => {
    const snapshot = nativeSnapshot({
      availability: { runtime: 'down', legacy: 'down', knowledge: 'down', trainingStream: 'idle' },
      health: undefined,
      memory: undefined,
      life: undefined,
      knowledge: undefined,
      training: { isTraining: false, pauseRequested: false, stopRequested: false, publishing: false, checkpoints: [] },
    })
    expect(renderLifeState(snapshot, BASE + 1_000, 400)).toEqual({ kind: 'omit', reason: 'unreachable' })
  })

  it('omits a reading older than the staleness bound', () => {
    const reading = renderLifeState(nativeSnapshot(), BASE + STALE_AFTER_MS + 1_000, 400)
    expect(reading).toEqual({ kind: 'omit', reason: 'stale' })
  })

  it('truncates an overrun line and says so within the budget', () => {
    const reading = renderLifeState(nativeSnapshot(), BASE, 80)
    if (reading.kind !== 'inject') throw new Error('expected an injectable reading')
    expect(reading.text.length).toBe(80)
    expect(reading.text.endsWith('truncated=1')).toBe(true)
  })
})

describe('significantChange', () => {
  it('treats the first observation as significant', () => {
    expect(significantChange(undefined, nativeSnapshot())).toBe(true)
  })

  it('fires on a training flip but not on checkpoint roster churn', () => {
    const before = nativeSnapshot()
    const flipped = nativeSnapshot({
      training: { isTraining: true, pauseRequested: false, stopRequested: false, publishing: false, checkpoints: [{ filename: 'a.pt', step: 1, bytes: 1, modifiedUtc: '', savedAtUtc: '', numEpochs: 1 }] },
    })
    expect(significantChange(before, flipped)).toBe(true)
  })

  it('fires on a dominant-need change', () => {
    const before = nativeSnapshot({
      life: { isRunning: true, legacy: { isRunning: true, lifeState: 'idle', dominantNeed: 'fatigue', needs: { hunger: 12, fatigue: 80, boredom: 3, stress: 1, curiosity: 44 }, totalHeartbeats: 1, totalEvents: 1 } },
    })
    const after = nativeSnapshot({
      life: { isRunning: true, legacy: { isRunning: true, lifeState: 'idle', dominantNeed: 'curiosity', needs: { hunger: 12, fatigue: 80, boredom: 3, stress: 1, curiosity: 44 }, totalHeartbeats: 2, totalEvents: 1 } },
    })
    expect(significantChange(before, after)).toBe(true)
  })

  it('fires when one need drifts past the bound and stays quiet on small drift', () => {
    const before = nativeSnapshot()
    const quiet = nativeSnapshot({
      life: { isRunning: true, native: { tick: 42, mode: 'wake', needs: { curiosity: 50, fatigue: 10, stress: 1.5 }, drives: { exploration: 40, replay: 10, rest: 0, play: 30 } } },
    })
    const loud = nativeSnapshot({
      life: { isRunning: true, native: { tick: 43, mode: 'wake', needs: { curiosity: 42.5 + NEED_DRIFT + 0.5, fatigue: 10, stress: 1.5 }, drives: { exploration: 40, replay: 10, rest: 0, play: 30 } } },
    })
    expect(significantChange(before, quiet)).toBe(false)
    expect(significantChange(before, loud)).toBe(true)
  })
})