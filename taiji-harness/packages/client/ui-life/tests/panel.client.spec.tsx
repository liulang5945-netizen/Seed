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
    training: {
      isTraining: false,
      pauseRequested: false,
      stopRequested: false,
      publishing: false,
      checkpoints: [],
      datasets: [
        { path: 'consolidated/night-1.jsonl', sizeBytes: 4096 },
        { path: 'simple_zh/dialogue_extended_clean.jsonl', sizeBytes: 108_327_171 },
      ],
    },
    knowledge: { docCount: 12, chunkCount: 340, hasEmbeddings: true, embedDim: 768 },
    consolidation: {
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
      },
      journal: { entries: 4, byKind: { interaction: 3, reflection: 1 }, sessions: 2, lastRecordedAt: 1_760_000_000 },
    },
    artifacts: { activeId: '', configuredId: '' },
    workbench: { status: 'ok', count: 16, source: 'seed_platform.workbench.CapabilitySnapshot', owner: 'Taiji native Workbench', revision: 6, error: '' },
    auth: { enabled: false, authenticated: true, tokenValid: false },
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

/**
 * An in-memory ILife whose verbs are observable fakes. The same fakes come
 * back under `mocks`: assertions read them as plain function properties,
 * because a method reference on `ILife` is an unbound-method violation.
 */
function stubLife(snapshot: LifeSnapshot | undefined, state: LifeSnapshotState['state'] = 'ready') {
  let current: LifeSnapshotState = { snapshot, state, error: null }
  const listeners = new Set<() => void>()
  const accept = async (): Promise<{ message: string }> => ({ message: 'accepted' })
  const refresh = vi.fn(async () => {
    if (current.snapshot === undefined) throw new LifeControlError(new RemoteError('life/stream-failed', 'no snapshot', { reason: 'no snapshot' }))
    return current.snapshot
  })
  const mocks = {
    refresh,
    trainStart: vi.fn(accept),
    trainResumeCheckpoint: vi.fn(accept),
    trainPause: vi.fn(accept),
    trainResume: vi.fn(accept),
    trainStop: vi.fn(accept),
    trainReset: vi.fn(accept),
    uploadDataset: vi.fn(async (_request: { name: string; data: string }) => ({ message: 'uploaded' })),
    consolidate: vi.fn(accept),
    activateCheckpoint: vi.fn(accept),
    lifeStart: vi.fn(accept),
    lifeStop: vi.fn(accept),
    lifeAction: vi.fn(accept),
  }
  const life: ILife = {
    getSnapshot: () => current,
    subscribe: (listener) => {
      listeners.add(listener)
      return () => { listeners.delete(listener) }
    },
    ...mocks,
  }
  return {
    life,
    mocks,
    emit: (next: LifeSnapshot) => {
      current = { snapshot: next, state: 'ready', error: null }
      for (const listener of listeners) listener()
    },
  }
}

/** Render the page against a stub facade. */
function mountPanel(life: ILife): void {
  render(<LifePanel {...standard} t={t} life={life} />)
}

/** The panel's one hidden dataset picker. */
function filePicker(): HTMLInputElement {
  const input = document.querySelector('input[type="file"]')
  if (!(input instanceof HTMLInputElement)) throw new Error('the panel must render one file picker')
  return input
}

/** One snapshot whose roster carries an extra (just uploaded) file. */
function withDataset(snapshot: LifeSnapshot, path: string, sizeBytes: number): LifeSnapshot {
  return {
    ...snapshot,
    training: { ...snapshot.training, datasets: [...(snapshot.training.datasets ?? []), { path, sizeBytes }] },
  }
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
    const { life, mocks } = stubLife(nativeSnapshot())
    mountPanel(life)

    expect(screen.getByText('4')).not.toBeNull()
    expect(screen.getByText('interaction: 3 · reflection: 1')).not.toBeNull()
    expect(screen.getByText('2')).not.toBeNull()
    expect(screen.getByText('consolidated/corpus-20260923T080000Z-pass-2.jsonl')).not.toBeNull()
    // The gate reason appears in both the spec block and the report block.
    expect(screen.getAllByText('interaction journal holds 3 entries')).toHaveLength(2)
    // The spec's dataset also sits in the training roster below.
    expect(screen.getAllByText('consolidated/night-1.jsonl').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('1.5 s')).not.toBeNull()
    expect(screen.getByText('recency')).not.toBeNull()
    expect(screen.getByText('corpus written')).not.toBeNull()
    expect(screen.queryByText(en.passRunning)).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: en.runConsolidate }))
    await waitFor(() => { expect(mocks.consolidate).toHaveBeenCalledTimes(1) })
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

  it('feeds the selection into trainStart, preselected from the data ring spec', async () => {
    const { life, mocks } = stubLife(nativeSnapshot())
    mountPanel(life)

    // The roster renders with runtime-owned sizes, and the spec's dataset arrives ticked.
    expect(screen.getByText('4.0 KB')).not.toBeNull()
    expect(screen.getByText('103.3 MB')).not.toBeNull()
    expect(screen.getByText(/1 selected/)).not.toBeNull()
    expect(screen.getByText(/names 1/)).not.toBeNull()
    const specDataset = screen.getByRole('checkbox', { name: /consolidated\/night-1\.jsonl/ })
    const otherDataset = screen.getByRole('checkbox', { name: /dialogue_extended_clean/ })
    expect((specDataset as HTMLInputElement).checked).toBe(true)
    expect((otherDataset as HTMLInputElement).checked).toBe(false)

    fireEvent.click(screen.getByRole('button', { name: en.trainStart }))
    await waitFor(() => {
      expect(mocks.trainStart).toHaveBeenLastCalledWith({ datasets: ['consolidated/night-1.jsonl'] })
    })

    // Unticking every dataset falls back to the runtime default corpus.
    fireEvent.click(specDataset)
    expect(screen.getByText(/0 selected/)).not.toBeNull()
    expect(screen.getByText(en.datasetsDefaultHint)).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: en.trainStart }))
    await waitFor(() => {
      expect(mocks.trainStart).toHaveBeenLastCalledWith({})
    })
  })

  it('reports an unserved roster honestly and still starts on the runtime default', async () => {
    const { life, mocks } = stubLife(nativeSnapshot({
      training: { isTraining: false, pauseRequested: false, stopRequested: false, publishing: false, checkpoints: [] },
    }))
    mountPanel(life)

    expect(screen.getByText(en.datasetsUnavailable)).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: en.trainStart }))
    await waitFor(() => {
      expect(mocks.trainStart).toHaveBeenLastCalledWith({})
    })
  })

  it('uploads a picked dataset as base64, shows the runtime message, and ticks the refreshed row once', async () => {
    const { life, mocks, emit } = stubLife(nativeSnapshot())
    mocks.uploadDataset.mockResolvedValue({ message: '数据集 `new-set.jsonl` 已成功上传并选中！' })
    mountPanel(life)

    const file = new File(['{"text":"hello"}\n'], 'new-set.jsonl', { type: 'application/jsonl' })
    fireEvent.change(filePicker(), { target: { files: [file] } })
    expect(screen.getByText('Selected new-set.jsonl (17 B)')).not.toBeNull()
    // Nothing is sent until the operator confirms the picked file.
    expect(mocks.uploadDataset).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: en.uploadSend }))
    await waitFor(() => { expect(mocks.uploadDataset).toHaveBeenCalledTimes(1) })
    const request = mocks.uploadDataset.mock.calls[0]![0]
    expect(request.name).toBe('new-set.jsonl')
    expect(atob(request.data)).toBe('{"text":"hello"}\n')
    // The runtime's own words land verbatim, and the roster is read again.
    expect(await screen.findByText('数据集 `new-set.jsonl` 已成功上传并选中！')).not.toBeNull()
    await waitFor(() => { expect(mocks.refresh).toHaveBeenCalledTimes(1) })
    expect(screen.queryByRole('button', { name: en.uploadSend })).toBeNull()

    // The refreshed roster carries the file; the upload ticks its row once.
    emit(withDataset(nativeSnapshot(), 'new-set.jsonl', 17))
    const row = await screen.findByRole('checkbox', { name: /new-set\.jsonl/ })
    await waitFor(() => { expect((row as HTMLInputElement).checked).toBe(true) })

    // An operator's own untick survives later roster identities.
    fireEvent.click(row)
    expect((row as HTMLInputElement).checked).toBe(false)
    emit(withDataset(nativeSnapshot(), 'new-set.jsonl', 17))
    const unticked = screen.getByRole('checkbox', { name: /new-set\.jsonl/ })
    await waitFor(() => { expect((unticked as HTMLInputElement).checked).toBe(false) })
  })

  it('refuses a picked file past the upload budget without reading it', () => {
    const { life, mocks } = stubLife(nativeSnapshot())
    mountPanel(life)

    const file = new File([new Uint8Array(1)], 'huge.jsonl')
    Object.defineProperty(file, 'size', { value: 200 * 1024 * 1024 + 1 })
    fireEvent.change(filePicker(), { target: { files: [file] } })

    expect(screen.getByRole('alert').textContent).toBe('The file exceeds 200 MB. Use a smaller dataset.')
    expect(screen.queryByRole('button', { name: en.uploadSend })).toBeNull()
    expect(mocks.uploadDataset).not.toHaveBeenCalled()
  })

  it('warns when the spec names a dataset the roster lacks', () => {
    const consolidation = nativeSnapshot().consolidation
    if (consolidation === undefined || consolidation.spec === null) throw new Error('fixture must carry a spec')
    const { life } = stubLife(nativeSnapshot({
      consolidation: {
        ...consolidation,
        spec: { ...consolidation.spec, datasets: ['consolidated/gone.jsonl', 'consolidated/night-1.jsonl'] },
      },
    }))
    mountPanel(life)

    expect(screen.getByText(/missing from the roster: consolidated\/gone\.jsonl/)).not.toBeNull()
    expect(screen.getByText(/names 2/)).not.toBeNull()
    // The missing entry cannot be ticked; the present one still preselects.
    expect(screen.getByText(/1 selected/)).not.toBeNull()
    expect(screen.queryByRole('checkbox', { name: /consolidated\/gone\.jsonl/ })).toBeNull()
  })

  it('resumes a training run from a checkpoint row with the selected datasets', async () => {
    const { life, mocks } = stubLife(nativeSnapshot({
      training: {
        isTraining: false,
        pauseRequested: false,
        stopRequested: false,
        publishing: false,
        datasets: [
          { path: 'consolidated/night-1.jsonl', sizeBytes: 4096 },
          { path: 'simple_zh/dialogue_extended_clean.jsonl', sizeBytes: 108_327_171 },
        ],
        checkpoints: [
          { filename: 'seed_beta.pt', step: 1_600_000, bytes: 2048, modifiedUtc: '2026-09-23T08:01:00.000Z', savedAtUtc: '2026-09-23T08:01:00.000Z', numEpochs: 1 },
        ],
        warnings: ['语料已变更：检查点原指纹 abc123，本次续训 def456'],
      },
    }))
    mountPanel(life)

    // The runtime's own drift notice renders verbatim beside its label.
    expect(screen.getByText(en.runWarning)).not.toBeNull()
    expect(screen.getByText('语料已变更：检查点原指纹 abc123，本次续训 def456')).not.toBeNull()

    // The spec preselects its dataset; resume sends the checkpoint plus that selection.
    fireEvent.click(screen.getByRole('button', { name: en.resumeFrom }))
    await waitFor(() => {
      expect(mocks.trainResumeCheckpoint).toHaveBeenLastCalledWith({
        checkpoint: 'seed_beta.pt',
        datasets: ['consolidated/night-1.jsonl'],
      })
    })

    // With nothing selected the request carries only the checkpoint name.
    fireEvent.click(screen.getByRole('checkbox', { name: /consolidated\/night-1\.jsonl/ }))
    fireEvent.click(screen.getByRole('button', { name: en.resumeFrom }))
    await waitFor(() => {
      expect(mocks.trainResumeCheckpoint).toHaveBeenLastCalledWith({ checkpoint: 'seed_beta.pt' })
    })
  })

  it('shows the publish state on the checkpoint rows and activates with confirmation', async () => {
    const { life, mocks } = stubLife(nativeSnapshot({
      artifacts: { activeId: 'seed_beta.pt', configuredId: 'other.pt' },
      training: {
        isTraining: false,
        pauseRequested: false,
        stopRequested: false,
        publishing: false,
        checkpoints: [
          { filename: 'seed_beta.pt', step: 10, bytes: 2048, modifiedUtc: '2026-09-23T08:00:00.000Z', savedAtUtc: '2026-09-23T08:00:00.000Z', numEpochs: 1 },
          { filename: 'other.pt', step: 20, bytes: 2048, modifiedUtc: '2026-09-23T09:00:00.000Z', savedAtUtc: '2026-09-23T09:00:00.000Z', numEpochs: 1 },
        ],
      },
    }))
    mountPanel(life)

    // The header states what answers and what the next start will use; each
    // name also appears on its checkpoint row, so both occurrences count.
    expect(screen.getByText(en.activeCheckpointLabel)).not.toBeNull()
    expect(screen.getAllByText('seed_beta.pt')).toHaveLength(2)
    expect(screen.getByText(en.configuredCheckpointLabel)).not.toBeNull()
    expect(screen.getAllByText('other.pt')).toHaveLength(2)
    // The active row carries the badge and cannot re-activate itself; the
    // other row offers an enabled activation. Rows follow checkpoint order.
    expect(screen.getByText(en.artifactsActiveBadge)).not.toBeNull()
    expect(screen.getByText(en.artifactsConfiguredBadge)).not.toBeNull()
    const activateButtons = screen.getAllByRole('button', { name: en.activateRow })
    expect((activateButtons[0] as HTMLButtonElement).disabled).toBe(true)
    const otherActivate = activateButtons[1]
    if (otherActivate === undefined) throw new Error('expected an activate button for other.pt')
    expect((otherActivate as HTMLButtonElement).disabled).toBe(false)
    fireEvent.click(otherActivate)
    expect(screen.getByText(en.confirmActivate)).not.toBeNull()
    expect(mocks.activateCheckpoint).not.toHaveBeenCalled()
    fireEvent.click(otherActivate)
    await waitFor(() => {
      expect(mocks.activateCheckpoint).toHaveBeenLastCalledWith({ checkpointId: 'other.pt' })
    })

    // Built-in switch is offered because a checkpoint is active, and is itself
    // a confirming verb: first click arms, second sends the empty name.
    const builtin = screen.getByRole('button', { name: en.activateBuiltin })
    fireEvent.click(builtin)
    expect(screen.getByText(en.confirmActivate)).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: en.activateBuiltin }))
    await waitFor(() => {
      expect(mocks.activateCheckpoint).toHaveBeenLastCalledWith({ checkpointId: '' })
    })
  })

  it('says honestly when the publish surface did not answer', () => {
    const { artifacts, ...rest } = nativeSnapshot()
    expect(artifacts).toBeDefined()
    const { life } = stubLife(rest)
    mountPanel(life)
    expect(screen.getByText(en.artifactsUnavailable)).not.toBeNull()
    expect(screen.queryByText(en.activeCheckpointLabel)).toBeNull()
  })

  it('states workbench capabilities and the three authentication outcomes', () => {
    mountPanel(stubLife(nativeSnapshot()).life)
    expect(screen.getByText(en.workbenchLabel)).not.toBeNull()
    expect(screen.getByText(/16 \(rev 6\)/)).not.toBeNull()
    expect(screen.getByText(/CapabilitySnapshot · Taiji native Workbench/)).not.toBeNull()
    expect(screen.getByText(en.authDisabled)).not.toBeNull()
    cleanup()

    mountPanel(stubLife(nativeSnapshot({ auth: { enabled: true, authenticated: true, tokenValid: true } })).life)
    expect(screen.getByText(en.authOk)).not.toBeNull()
    cleanup()

    mountPanel(stubLife(nativeSnapshot({ auth: { enabled: true, authenticated: false, tokenValid: false } })).life)
    expect(screen.getByText(en.authFailed)).not.toBeNull()
    cleanup()

    // A failed capability snapshot shows the runtime's own status and text, never a zero.
    mountPanel(stubLife(nativeSnapshot({
      workbench: { status: 'error', count: 0, source: '', owner: '', revision: 0, error: '快照构建失败' },
    })).life)
    expect(screen.getByText('error')).not.toBeNull()
    expect(screen.getByText('快照构建失败')).not.toBeNull()
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
    const { life, mocks } = stubLife(nativeSnapshot({
      training: {
        isTraining: true,
        pauseRequested: false,
        stopRequested: false,
        publishing: false,
        progress: {
          fraction: 0.25, step: 5, loss: 1.3, elapsed: 60, eta: 180,
          epoch: 1, totalEpochs: 4, samplesPerSec: 12.5, totalSteps: 20,
        },
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
    expect(mocks.trainStop).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: en.trainStop }))
    await waitFor(() => { expect(mocks.trainStop).toHaveBeenCalledTimes(1) })
  })

  it('renders a control refusal as stable copy and clears it on the next success', async () => {
    const { life, mocks } = stubLife(legacySnapshot())
    mocks.lifeStop.mockRejectedValue(new LifeControlError(UNAVAILABLE))
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
    await waitFor(() => { expect(failing.mocks.refresh).toHaveBeenCalledTimes(1) })
  })
})
