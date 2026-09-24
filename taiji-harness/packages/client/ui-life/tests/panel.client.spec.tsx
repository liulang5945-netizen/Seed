// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { LifeControlError, type ILife } from '@taiji/dsh-api-life-controller/client'
import type {
  LifeSnapshot,
  LifeSnapshotState,
} from '@taiji/dsh-api-life-controller/client'
import { RemoteError, type RemoteFailure } from '@taiji/dsh-typert-protocol'
import { LifePanel, type LifePanelProps } from '../src/client/LifePanel.tsx'
// Pulls the `LocaleNamespaceMap` augmentation into this program, so
// `LifePanelProps['t']` resolves to the framework-injected translate seat.
import type {} from '../src/client/index.ts'
import { en, type LifeLocaleKey } from '../src/client/locales.ts'

afterEach(cleanup)

const translate = (dict: typeof en): LifePanelProps['t'] => ((key: LifeLocaleKey, params?: Record<string, string>): string =>
  Object.entries(params ?? {}).reduce(
    (text, [name, value]) => text.replaceAll(`{${name}}`, value),
    dict[key],
  )) as LifePanelProps['t']

const t = translate(en)

/** The standard seat this fixture deliberately leaves unimplemented. */
const unusedStandardHook = (): never => { throw new Error('Life panel fixture does not provide global state') }
const standard = {
  usePanelInfo: unusedStandardHook,
  useWorkspaces: unusedStandardHook,
  useSessions: unusedStandardHook,
  useSessionStatus: unusedStandardHook,
  useSessionRetainInfo: unusedStandardHook,
  useResource: unusedStandardHook,
}

/** One complete native reading, every source answered. */
function nativeSnapshot(overrides: Partial<LifeSnapshot> = {}): LifeSnapshot {
  return {
    source: 'native',
    observedAt: '2026-09-23T08:00:00.000Z',
    fresh: true,
    pollIntervalMs: 5000,
    health: { state: 'ok', modelLoaded: true, modelName: 'test-model', seedActive: true, startupComplete: true },
    memory: { totalGb: 32, availableGb: 12.5, usedPct: 60.9 },
    life: {
      isRunning: true,
      native: { tick: 41, mode: 'wake', needs: { curiosity: 42.5, fatigue: 10 }, drives: { exploration: 40 } },
    },
    training: { isTraining: false, pauseRequested: false, stopRequested: false, publishing: false, checkpoints: [] },
    knowledge: { docCount: 12, chunkCount: 340, hasEmbeddings: true, embedDim: 768 },
    consolidation: {
      passes: 2,
      lastPassAt: 1_760_000_100,
      lastCorpus: 'data/consolidated/corpus-20260923T080000Z-pass-2.jsonl',
      projectedDigests: 3,
      running: false,
      spec: {
        reason: 'interaction journal holds 3 entries',
        datasets: ['data/consolidated/night-1.jsonl'],
        weaknesses: ['recency'],
      },
      lastReport: {
        reason: 'manual',
        specReason: 'interaction journal holds 3 entries',
        durationMs: 1_500,
        weaknesses: ['recency'],
        notes: ['corpus written'],
      },
      journal: { entries: 4, byKind: { interaction: 3, reflection: 1 }, sessions: 2, lastRecordedAt: 1_760_000_000 },
    },
    availability: { runtime: 'ok', legacy: 'disabled', knowledge: 'ok', trainingStream: 'idle' },
    unavailable: [],
    ...overrides,
  }
}

/** One legacy-scheduler reading: the gated surface answered. */
function legacySnapshot(overrides: Partial<LifeSnapshot> = {}): LifeSnapshot {
  return nativeSnapshot({
    source: 'legacy',
    availability: { runtime: 'ok', legacy: 'ok', knowledge: 'ok', trainingStream: 'idle' },
    life: {
      isRunning: true,
      legacy: {
        isRunning: true,
        lifeState: 'idle',
        dominantNeed: 'energy',
        needs: { energy: 35, curiosity: 80, fatigue: 20, stress: 5, social: 50 },
        totalHeartbeats: 128,
        totalEvents: 17,
        lastHeartbeat: '2026-09-23T07:59:30.000Z',
        lastActivity: '2026-09-23T07:55:00.000Z',
      },
    },
    ...overrides,
  })
}

/** An in-memory ILife whose verbs are observable fakes. */
function stubLife(snapshot: LifeSnapshot | undefined, state: LifeSnapshotState['state'] = 'ready'): {
  life: ILife
  emit: (next: LifeSnapshot) => void
} {
  let current: LifeSnapshotState = { snapshot, state, error: null }
  const listeners = new Set<() => void>()
  const accept = async (): Promise<{ message: string }> => ({ message: 'accepted' })
  const life: ILife = {
    getSnapshot: () => current,
    subscribe: listener => {
      listeners.add(listener)
      return () => { listeners.delete(listener) }
    },
    refresh: vi.fn(async () => {
      if (current.snapshot === undefined) throw new LifeControlError(new RemoteError('life/stream-failed', 'no snapshot', { reason: 'no snapshot' }))
      return current.snapshot
    }),
    trainStart: vi.fn(accept),
    trainPause: vi.fn(accept),
    trainResume: vi.fn(accept),
    trainStop: vi.fn(accept),
    trainReset: vi.fn(accept),
    consolidate: vi.fn(accept),
    lifeStart: vi.fn(accept),
    lifeStop: vi.fn(accept),
    lifeAction: vi.fn(accept),
  }
  return {
    life,
    emit: next => {
      current = { snapshot: next, state: 'ready', error: null }
      for (const listener of listeners) listener()
    },
  }
}

/** Render the page against a stub facade. */
function mountPanel(life: ILife): void {
  render(<LifePanel {...standard} t={t} life={life} />)
}

const UNAVAILABLE: RemoteFailure = new RemoteError(
  'life/unavailable',
  'legacy surface is not mounted',
  { source: 'life', reason: 'legacy surface is not mounted' },
)

describe('LifePanel', () => {
  it('renders a native reading across all six sections', () => {
    const { life } = stubLife(nativeSnapshot())
    mountPanel(life)

    expect(screen.getByText(en.sectionSource)).not.toBeNull()
    expect(screen.getByText(en.sectionLife)).not.toBeNull()
    expect(screen.getByText(en.sectionTraining)).not.toBeNull()
    expect(screen.getByText(en.sectionKnowledge)).not.toBeNull()
    expect(screen.getByText(en.sectionConsolidation)).not.toBeNull()
    expect(screen.getByText(en.sectionHost)).not.toBeNull()

    expect(screen.getByText(en.sourceNative)).not.toBeNull()
    expect(screen.getByText(en.freshBadge)).not.toBeNull()
    expect(screen.getByText('wake')).not.toBeNull()
    expect(screen.getByText('curiosity')).not.toBeNull()
    expect(screen.getByText('42.5')).not.toBeNull()
    expect(screen.getByText('exploration')).not.toBeNull()

    expect(screen.getByText(en.trainingIdle)).not.toBeNull()
    expect(screen.getByText(en.noProgress)).not.toBeNull()
    expect(screen.getByText(en.checkpointsEmpty)).not.toBeNull()

    expect(screen.getByText('test-model')).not.toBeNull()
    expect(screen.getByText(en.seedActive)).not.toBeNull()
  })

  it('renders the memory journal, the pass products, and runs a pass on click', async () => {
    const { life } = stubLife(nativeSnapshot())
    mountPanel(life)

    expect(screen.getByText('4')).not.toBeNull()
    expect(screen.getByText('interaction: 3 · reflection: 1')).not.toBeNull()
    expect(screen.getByText('2')).not.toBeNull()
    expect(screen.getByText('data/consolidated/corpus-20260923T080000Z-pass-2.jsonl')).not.toBeNull()
    // The gate reason appears in both the spec block and the report block.
    expect(screen.getAllByText('interaction journal holds 3 entries')).toHaveLength(2)
    expect(screen.getByText('data/consolidated/night-1.jsonl')).not.toBeNull()
    expect(screen.getByText('1.5 s')).not.toBeNull()
    expect(screen.getByText('recency')).not.toBeNull()
    expect(screen.getByText('corpus written')).not.toBeNull()
    expect(screen.queryByText(en.passRunning)).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: en.runConsolidate }))
    await waitFor(() => { expect(life.consolidate).toHaveBeenCalledTimes(1) })
  })

  it('reports an unserved consolidation surface and an unpassed readiness gate honestly', () => {
    const { consolidation: omitted, ...unServed } = nativeSnapshot()
    expect(omitted).toBeDefined()
    const missing = stubLife(unServed)
    mountPanel(missing.life)
    expect(screen.getByText(en.consolidationUnavailable)).not.toBeNull()
    expect(screen.queryByRole('button', { name: en.runConsolidate })).toBeNull()
    cleanup()

    const gated = stubLife(nativeSnapshot({
      consolidation: {
        passes: 0,
        lastPassAt: 0,
        lastCorpus: '',
        projectedDigests: 0,
        running: false,
        spec: null,
        lastReport: null,
        journal: { entries: 0, byKind: {}, sessions: 0, lastRecordedAt: 0 },
      },
    }))
    mountPanel(gated.life)
    expect(screen.getByText(en.specNotReady)).not.toBeNull()
    expect(screen.getByText(en.noReport)).not.toBeNull()
    expect(screen.getByText(en.noReadings)).not.toBeNull()
    expect(screen.getAllByText(en.notYet).length).toBeGreaterThanOrEqual(2)
  })

  it('renders the legacy organ with its scheduler facts and controls', () => {
    const { life } = stubLife(legacySnapshot())
    mountPanel(life)

    expect(screen.getByText(en.sourceLegacy)).not.toBeNull()
    // 'energy' names both the dominant need and its meter row.
    expect(screen.getAllByText('energy').length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('128')).not.toBeNull()
    expect(screen.getByText('17')).not.toBeNull()
    for (const name of [en.lifeStart, en.lifeStop, en.actionFeed, en.actionSleep, en.actionPlay]) {
      expect(screen.getByRole('button', { name })).not.toBeNull()
    }
  })

  it('marks a stale reading and lists the sources that did not answer', () => {
    const { life } = stubLife(nativeSnapshot({
      fresh: false,
      unavailable: ['legacy life surface: not mounted'],
    }))
    mountPanel(life)

    expect(screen.getByText(en.staleBadge)).not.toBeNull()
    expect(screen.getByText(en.unavailableTitle)).not.toBeNull()
    expect(screen.getByText('legacy life surface: not mounted')).not.toBeNull()
  })

  it('requires a confirming click before stopping training, then runs the verb', async () => {
    const { life } = stubLife(nativeSnapshot({
      training: {
        isTraining: true,
        pauseRequested: false,
        stopRequested: false,
        publishing: false,
        progress: { fraction: 0.25, step: 5, loss: 1.3, elapsed: 60, eta: 180, epoch: 1, totalEpochs: 4, samplesPerSec: 12.5, totalSteps: 20 },
        checkpoints: [
          { filename: 'ckpt-000005.pt', step: 5, bytes: 2048, modifiedUtc: '2026-09-23T08:01:00.000Z', savedAtUtc: '2026-09-23T08:01:00.000Z', numEpochs: 1 },
        ],
      },
    }))
    mountPanel(life)

    expect(screen.getByText(en.trainingRunning)).not.toBeNull()
    expect(screen.getByText('1.300')).not.toBeNull()
    expect(screen.getByText('ckpt-000005.pt')).not.toBeNull()

    const stop = screen.getByRole('button', { name: en.trainStop })
    expect(stop.hasAttribute('disabled')).toBe(false)
    fireEvent.click(stop)
    expect(screen.getByText(en.confirmStop)).not.toBeNull()
    expect(life.trainStop).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: en.trainStop }))
    await waitFor(() => { expect(life.trainStop).toHaveBeenCalledTimes(1) })
  })

  it('renders a control refusal as stable copy and clears it on the next success', async () => {
    const { life } = stubLife(legacySnapshot())
    vi.mocked(life.lifeStop).mockRejectedValue(new LifeControlError(UNAVAILABLE))
    mountPanel(life)

    fireEvent.click(screen.getByRole('button', { name: en.lifeStop }))
    const alert = await screen.findByRole('alert')
    expect(alert.textContent).toBe(en.errUnavailable.replace('{reason}', 'legacy surface is not mounted'))
    expect(alert.textContent).not.toContain('lifeStop')

    // An accepted action afterwards clears the refusal line.
    fireEvent.click(screen.getByRole('button', { name: en.lifeStart }))
    await waitFor(() => { expect(screen.queryByRole('alert')).toBeNull() })
  })

  it('shows the loading state first and the error state with a retry that re-reads', async () => {
    const { life } = stubLife(undefined, 'loading')
    mountPanel(life)
    expect(screen.getByText(en.loading)).not.toBeNull()

    // A terminal failure renders the retry offer, and retrying reads again.
    cleanup()
    const failing = stubLife(undefined, 'error')
    mountPanel(failing.life)
    expect(screen.getByText(en.errorTitle)).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: en.retry }))
    await waitFor(() => { expect(failing.life.refresh).toHaveBeenCalledTimes(1) })
  })
})
