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
import { LifePanelIcon } from '../src/client/LifePanelIcon.tsx'
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
    knowledge: {
      docCount: 12,
      chunkCount: 340,
      hasEmbeddings: true,
      embedDim: 768,
      files: [
        { name: 'handbook.md', sizeBytes: 2048, status: 'indexed' },
        { name: 'loose.txt', status: 'pending' },
      ],
    },
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
        workbenchCapabilities: 16,
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
    deleteDataset: vi.fn(accept),
    deleteCheckpoint: vi.fn(accept),
    uploadKnowledge: vi.fn(async (_request: { name: string; data: string }) => ({ message: 'knowledge uploaded' })),
    deleteKnowledge: vi.fn(accept),
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

/** Every file picker the panel rendered, in document order. */
function filePickers(): HTMLInputElement[] {
  return [...document.querySelectorAll('input[type="file"]')].map((input) => {
    if (!(input instanceof HTMLInputElement)) throw new Error('expected a file input')
    return input
  })
}

/** The panel's training dataset picker, rendered before the knowledge one. */
function filePicker(): HTMLInputElement {
  const [first] = filePickers()
  if (first === undefined) throw new Error('the panel must render the dataset picker')
  return first
}

/** The knowledge section's picker, rendered after the training one. */
function knowledgePicker(): HTMLInputElement {
  const [, second] = filePickers()
  if (second === undefined) throw new Error('the panel must render the knowledge picker')
  return second
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

  it('keeps the consolidation sub-block titles and the action row direct children of the section', () => {
    // The sheet spaces this block with child selectors (`.section >`), so wrapping
    // any of these in another element silently returns it to the section's uniform
    // gap — the crowding that was reported. A later title is the case that carries
    // the extra room, so its position among the siblings is part of the contract.
    const { life } = stubLife(nativeSnapshot())
    mountPanel(life)

    const spec = screen.getByText(en.specTitle)
    const section = spec.parentElement
    expect(section?.tagName).toBe('DETAILS')
    expect(section?.id).toBe('life-consolidation')
    expect(spec.previousElementSibling).not.toBeNull()
    expect(screen.getByText(en.reportTitle).parentElement).toBe(section)
    expect(screen.getByRole('button', { name: en.runConsolidate }).parentElement?.parentElement).toBe(section)
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

  it('able to fold a dataset directory away and back', () => {
    const { life } = stubLife(nativeSnapshot())
    mountPanel(life)

    const toggle = screen.getByRole('button', { name: /consolidated/ })
    expect(toggle.getAttribute('aria-expanded')).toBe('true')
    fireEvent.click(toggle)
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    expect(screen.queryByRole('checkbox', { name: /consolidated\/night-1\.jsonl/ })).toBeNull()
    // The other directory's rows stay; folding one group never folds the rest.
    expect(screen.getByRole('checkbox', { name: /dialogue_extended_clean/ })).not.toBeNull()

    fireEvent.click(toggle)
    expect(screen.getByRole('checkbox', { name: /consolidated\/night-1\.jsonl/ })).not.toBeNull()
  })

  it('deletes the ticked datasets after a confirming click and prunes the selection', async () => {
    const { life, mocks } = stubLife(nativeSnapshot())
    mountPanel(life)

    fireEvent.click(screen.getByRole('checkbox', { name: /dialogue_extended_clean/ }))
    expect(screen.getByText(/2 selected/)).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: en.deletePickedDatasets }))
    // The armed button counts what would go, and nothing has left yet.
    const confirmCopy = t('confirmDeleteDatasets', { count: '2' })
    expect(screen.getByRole('button', { name: confirmCopy })).not.toBeNull()
    expect(mocks.deleteDataset).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: confirmCopy }))
    await waitFor(() => { expect(mocks.deleteDataset).toHaveBeenCalledTimes(2) })
    expect(mocks.deleteDataset).toHaveBeenCalledWith({ path: 'consolidated/night-1.jsonl' })
    expect(mocks.deleteDataset).toHaveBeenCalledWith({ path: 'simple_zh/dialogue_extended_clean.jsonl' })
    // The rows that went drop from the selection, the ack names the count, and
    // the roster is read again.
    await waitFor(() => { expect(screen.getByText(/0 selected/)).not.toBeNull() })
    expect(screen.getByText(t('deleteDone', { count: '2' }))).not.toBeNull()
    await waitFor(() => { expect(mocks.refresh).toHaveBeenCalled() })
  })

  it('deletes the ticked checkpoints after a confirming click and refuses the in-use rows', async () => {
    const { life, mocks } = stubLife(nativeSnapshot({
      artifacts: { activeId: 'seed_beta.pt', configuredId: '' },
      training: {
        isTraining: false,
        pauseRequested: false,
        stopRequested: false,
        publishing: false,
        checkpoints: [
          { filename: 'seed_beta.pt', step: 10, bytes: 2048, modifiedUtc: '2026-09-23T08:00:00.000Z', savedAtUtc: '2026-09-23T08:00:00.000Z', numEpochs: 1 },
          { filename: 'old_run.pt', step: 20, bytes: 4096, modifiedUtc: '2026-09-23T09:00:00.000Z', savedAtUtc: '2026-09-23T09:00:00.000Z', numEpochs: 1 },
        ],
      },
    }))
    mountPanel(life)

    // The answering checkpoint cannot even be ticked; an old one can.
    const activeBox = screen.getByRole('checkbox', { name: `${en.colSelect} seed_beta.pt` }) as HTMLInputElement
    expect(activeBox.disabled).toBe(true)
    fireEvent.click(screen.getByRole('checkbox', { name: `${en.colSelect} old_run.pt` }))
    fireEvent.click(screen.getByRole('button', { name: en.deletePickedCheckpoints }))
    const confirmCopy = t('confirmDeleteCheckpoints', { count: '1' })
    expect(mocks.deleteCheckpoint).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: confirmCopy }))
    await waitFor(() => { expect(mocks.deleteCheckpoint).toHaveBeenCalledTimes(1) })
    expect(mocks.deleteCheckpoint).toHaveBeenCalledWith({ filename: 'old_run.pt' })
    expect(screen.getByText(t('deleteDone', { count: '1' }))).not.toBeNull()
  })

  it('folds the checkpoint block away and back', () => {
    const { life } = stubLife(nativeSnapshot({
      training: {
        isTraining: false,
        pauseRequested: false,
        stopRequested: false,
        publishing: false,
        checkpoints: [
          { filename: 'old_run.pt', step: 20, bytes: 4096, modifiedUtc: '2026-09-23T09:00:00.000Z', savedAtUtc: '2026-09-23T09:00:00.000Z', numEpochs: 1 },
        ],
      },
    }))
    mountPanel(life)

    // Every row's controls carry the filename, so six rows never read as six
    // buttons with one name.
    const resumeOld = `${en.resumeFrom} old_run.pt`
    expect(screen.getByRole('button', { name: resumeOld })).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: en.checkpointsFold }))
    expect(screen.queryByRole('button', { name: resumeOld })).toBeNull()
    expect(screen.queryByText(en.activeCheckpointLabel)).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: t('checkpointsUnfold', { count: '1' }) }))
    expect(screen.getByRole('button', { name: resumeOld })).not.toBeNull()
    expect(screen.getByText(en.activeCheckpointLabel)).not.toBeNull()
  })

  it('uploads a knowledge document and deletes the ticked files after confirmation', async () => {
    const { life, mocks } = stubLife(nativeSnapshot())
    mountPanel(life)

    // The mounted files render with their index state; a pending file says so.
    expect(screen.getByText('handbook.md')).not.toBeNull()
    expect(screen.getByText(en.knowledgePending)).not.toBeNull()
    expect(screen.getByText('2.0 KB')).not.toBeNull()

    const file = new File(['# notes\n'], 'notes.md', { type: 'text/markdown' })
    fireEvent.change(knowledgePicker(), { target: { files: [file] } })
    expect(screen.getByText('Selected notes.md (8 B)')).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: en.uploadSend }))
    await waitFor(() => { expect(mocks.uploadKnowledge).toHaveBeenCalledTimes(1) })
    const request = mocks.uploadKnowledge.mock.calls[0]![0]
    expect(request.name).toBe('notes.md')
    expect(atob(request.data)).toBe('# notes\n')
    expect(await screen.findByText('knowledge uploaded')).not.toBeNull()

    fireEvent.click(screen.getByRole('checkbox', { name: 'loose.txt' }))
    fireEvent.click(screen.getByRole('button', { name: en.deletePickedKnowledge }))
    const confirmCopy = t('confirmDeleteKnowledge', { count: '1' })
    expect(mocks.deleteKnowledge).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: confirmCopy }))
    await waitFor(() => { expect(mocks.deleteKnowledge).toHaveBeenCalledTimes(1) })
    expect(mocks.deleteKnowledge).toHaveBeenCalledWith({ name: 'loose.txt' })
    expect(screen.getByText(t('deleteDone', { count: '1' }))).not.toBeNull()
  })

  it('says honestly when the knowledge file list did not answer', () => {
    const fixture = nativeSnapshot().knowledge
    if (fixture === undefined) throw new Error('fixture must carry a knowledge reading')
    const { files: omitted, ...withoutFiles } = fixture
    expect(omitted).toBeDefined()
    const { life } = stubLife(nativeSnapshot({ knowledge: withoutFiles }))
    mountPanel(life)

    expect(screen.getByText(en.knowledgeFilesUnavailable)).not.toBeNull()
    expect(screen.queryByRole('checkbox', { name: 'handbook.md' })).toBeNull()
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
    const resumeSeed = `${en.resumeFrom} seed_beta.pt`
    fireEvent.click(screen.getByRole('button', { name: resumeSeed }))
    await waitFor(() => {
      expect(mocks.trainResumeCheckpoint).toHaveBeenLastCalledWith({
        checkpoint: 'seed_beta.pt',
        datasets: ['consolidated/night-1.jsonl'],
      })
    })

    // With nothing selected the request carries only the checkpoint name.
    fireEvent.click(screen.getByRole('checkbox', { name: /consolidated\/night-1\.jsonl/ }))
    fireEvent.click(screen.getByRole('button', { name: resumeSeed }))
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
    const activateButtons = screen.getAllByRole('button', { name: new RegExp(`^${en.activateRow} `) })
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

  it('keys pause and resume off pauseRequested, the only paused reading the runtime carries', () => {
    // There is no separate `paused` flag in the training payload (runtime-client.ts:693-698),
    // so `pauseRequested` is what says "pausing or paused" and both verbs have to key off it.
    const running = stubLife(nativeSnapshot({
      training: { isTraining: true, pauseRequested: false, stopRequested: false, publishing: false, checkpoints: [] },
    }))
    mountPanel(running.life)
    expect(screen.getByRole('button', { name: en.trainPause }).hasAttribute('disabled')).toBe(false)
    expect(screen.getByRole('button', { name: en.trainResume }).hasAttribute('disabled')).toBe(true)
    cleanup()

    const pausing = stubLife(nativeSnapshot({
      training: { isTraining: true, pauseRequested: true, stopRequested: false, publishing: false, checkpoints: [] },
    }))
    mountPanel(pausing.life)
    expect(screen.getByText(en.trainingPauseRequested)).not.toBeNull()
    expect(screen.getByRole('button', { name: en.trainPause }).hasAttribute('disabled')).toBe(true)
    expect(screen.getByRole('button', { name: en.trainResume }).hasAttribute('disabled')).toBe(false)
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

  it('splits the training block into controls, datasets and checkpoints', () => {
    const { life } = stubLife(nativeSnapshot())
    mountPanel(life)

    // One wall of four lists became three sibling blocks, each with its own id
    // so a deep link can name any of them. Asserting the ids is what pins the
    // split: collapsing them back would keep every heading but lose the blocks.
    const controls = screen.getByText(en.sectionTraining).closest('details')
    const datasets = screen.getByText(en.datasetsTitle).closest('details')
    const checkpoints = screen.getByText(en.checkpointsTitle).closest('details')

    expect(controls?.id).toBe('life-training')
    expect(datasets?.id).toBe('life-training-data')
    expect(checkpoints?.id).toBe('life-checkpoints')
    expect(new Set([controls, datasets, checkpoints]).size).toBe(3)
  })
})

describe('LifePanelIcon', () => {
  it('renders the sidebar glyph at the size the sidebar asks for', () => {
    render(<LifePanelIcon {...standard} size={20} active={false} />)
    const glyph = document.querySelector('svg')
    expect(glyph).not.toBeNull()
    expect(glyph?.getAttribute('width')).toBe('20')
    expect(glyph?.getAttribute('height')).toBe('20')
    expect(glyph?.getAttribute('viewBox')).toBe('0 0 16 16')
  })
})

describe('LifePanel refusals from the failure shapes the Host sends', () => {
  it('names each typed refusal with its own copy', async () => {
    const { life, mocks } = stubLife(legacySnapshot())
    mocks.lifeStop
      .mockRejectedValueOnce(new LifeControlError(new RemoteError('life/runtime-error', 'boom', { status: 500, detail: 'engine exploded' })))
      .mockRejectedValueOnce(new LifeControlError(new RemoteError('life/runtime-unreachable', 'boom', { baseURL: 'http://127.0.0.1:8000', reason: 'socket closed' })))
      .mockRejectedValueOnce(new LifeControlError(new RemoteError('life/conflict', 'boom', { reason: 'already running' })))
      .mockRejectedValueOnce(new LifeControlError(new RemoteError('life/bad-request', 'boom', { field: 'action', reason: 'unknown verb' })))
      .mockRejectedValueOnce(new LifeControlError(new RemoteError('life/stream-failed', 'boom', { reason: 'stream lost' })))
      // A code the panel has never met arrives as a plain Host-shaped failure;
      // the structural match still routes it to the fallback copy.
      .mockRejectedValueOnce({ rpcError: { code: 'life/from-the-future', message: 'boom', details: {} } })
    mountPanel(life)

    const cases: [string, string][] = [
      [t('errRuntimeError', { status: '500', detail: 'engine exploded' }), 'runtime-error'],
      [t('errUnreachable', { reason: 'socket closed' }), 'runtime-unreachable'],
      [t('errConflict', { reason: 'already running' }), 'conflict'],
      [t('errBadRequest', { field: 'action', reason: 'unknown verb' }), 'bad-request'],
      [t('errStreamFailed'), 'stream-failed'],
      [t('errFallback', { code: 'life/from-the-future' }), 'default'],
    ]
    for (const [copy, code] of cases) {
      fireEvent.click(screen.getByRole('button', { name: en.lifeStop }))
      const alert = await screen.findByRole('alert')
      expect(alert.textContent).toBe(copy)
      if (code === 'default') {
        // The fallback names the code it did not know, so a new Host code is
        // never rendered as an empty refusal.
        expect(alert.textContent).toContain('life/from-the-future')
      }
    }
  })

  it('falls back to the stream-failure copy when a rejection carries no RPC shape at all', async () => {
    const { life, mocks } = stubLife(legacySnapshot())
    mocks.lifeStop
      .mockRejectedValueOnce('a bare string from somewhere')
      .mockRejectedValueOnce(new Error('plain host-side failure'))
    mountPanel(life)

    fireEvent.click(screen.getByRole('button', { name: en.lifeStop }))
    expect(await screen.findByRole('alert').then(alert => alert.textContent)).toBe(t('errStreamFailed'))

    fireEvent.click(screen.getByRole('button', { name: en.lifeStop }))
    await waitFor(() => { expect(screen.getByRole('alert').textContent).toBe(t('errStreamFailed')) })
  })

  it('runs the three scheduler actions and reports a failed one without losing the page', async () => {
    const { life, mocks } = stubLife(legacySnapshot())
    mocks.lifeAction
      .mockResolvedValueOnce({ message: 'fed' })
      .mockRejectedValueOnce(new LifeControlError(new RemoteError('life/conflict', 'boom', { reason: 'sleeping already' })))
      .mockResolvedValueOnce({ message: 'played' })
    mountPanel(life)

    fireEvent.click(screen.getByRole('button', { name: en.actionFeed }))
    await waitFor(() => { expect(mocks.lifeAction).toHaveBeenLastCalledWith({ action: 'feed' }) })
    fireEvent.click(screen.getByRole('button', { name: en.actionSleep }))
    await waitFor(() => { expect(mocks.lifeAction).toHaveBeenLastCalledWith({ action: 'sleep' }) })
    expect(await screen.findByRole('alert').then(alert => alert.textContent))
      .toBe(t('errConflict', { reason: 'sleeping already' }))
    fireEvent.click(screen.getByRole('button', { name: en.actionPlay }))
    await waitFor(() => { expect(mocks.lifeAction).toHaveBeenLastCalledWith({ action: 'play' }) })
    await waitFor(() => { expect(screen.queryByRole('alert')).toBeNull() })
  })
})

describe('LifePanel reader edge shapes', () => {
  it('surfaces a file-read failure as the stream-failure copy and sends nothing', async () => {
    const { life, mocks } = stubLife(nativeSnapshot())
    const spy = vi.spyOn(FileReader.prototype, 'readAsDataURL').mockImplementation(function (this: FileReader) {
      queueMicrotask(() => { this.onerror?.call(this, new ProgressEvent('error') as ProgressEvent<FileReader>) })
    })
    try {
      mountPanel(life)
      fireEvent.change(filePicker(), { target: { files: [new File(['x'], 'broken.jsonl')] } })
      fireEvent.click(screen.getByRole('button', { name: en.uploadSend }))

      expect(await screen.findByRole('alert').then(alert => alert.textContent)).toBe(t('errStreamFailed'))
      expect(mocks.uploadDataset).not.toHaveBeenCalled()
    } finally {
      spy.mockRestore()
    }
  })

  it('sends an empty payload when the data URL comes back malformed', async () => {
    const { life, mocks } = stubLife(nativeSnapshot())
    mocks.uploadDataset.mockResolvedValue({ message: 'ok' })
    const spy = vi.spyOn(FileReader.prototype, 'readAsDataURL').mockImplementation(function (this: FileReader) {
      // A non-string result (jsdom leaves it null here) degrades to an empty
      // payload; it may not crash the read.
      queueMicrotask(() => { this.onload?.call(this, new ProgressEvent('load') as ProgressEvent<FileReader>) })
    })
    try {
      mountPanel(life)
      fireEvent.change(filePicker(), { target: { files: [new File(['x'], 'odd.jsonl')] } })
      fireEvent.click(screen.getByRole('button', { name: en.uploadSend }))
      await waitFor(() => { expect(mocks.uploadDataset).toHaveBeenCalledTimes(1) })
      expect(mocks.uploadDataset.mock.calls[0]![0].data).toBe('')
    } finally {
      spy.mockRestore()
    }

    cleanup()
    const second = stubLife(nativeSnapshot())
    second.mocks.uploadDataset.mockResolvedValue({ message: 'ok' })
    const spy2 = vi.spyOn(FileReader.prototype, 'readAsDataURL').mockImplementation(function (this: FileReader) {
      // A string without the data-URL comma degrades the same way.
      Object.defineProperty(this, 'result', { value: 'no-comma-here' })
      queueMicrotask(() => { this.onload?.call(this, new ProgressEvent('load') as ProgressEvent<FileReader>) })
    })
    try {
      mountPanel(second.life)
      fireEvent.change(filePicker(), { target: { files: [new File(['x'], 'odd2.jsonl')] } })
      fireEvent.click(screen.getByRole('button', { name: en.uploadSend }))
      await waitFor(() => { expect(second.mocks.uploadDataset).toHaveBeenCalledTimes(1) })
      expect(second.mocks.uploadDataset.mock.calls[0]![0].data).toBe('')
    } finally {
      spy2.mockRestore()
    }
  })
})

describe('LifePanel edge readings', () => {
  it('renders an unparseable timestamp verbatim instead of a wrong date', () => {
    const { life } = stubLife(legacySnapshot({
      life: {
        isRunning: true,
        legacy: {
          isRunning: true,
          lifeState: 'idle',
          dominantNeed: 'energy',
          needs: { energy: 35, curiosity: 80, fatigue: 20, stress: 5, social: 50 },
          totalHeartbeats: 128,
          totalEvents: 17,
          lastHeartbeat: 'not-a-timestamp',
          lastActivity: '2026-09-23T07:55:00.000Z',
        },
      },
    }))
    mountPanel(life)
    expect(screen.getByText('not-a-timestamp')).not.toBeNull()
  })

  it('says there are no readings when an open need or drive map is empty', () => {
    const { life } = stubLife(nativeSnapshot({
      life: { isRunning: true, native: { tick: 41, mode: 'wake', needs: {}, drives: {} } },
    }))
    mountPanel(life)
    expect(screen.getAllByText(en.noReadings)).toHaveLength(2)
  })

  it('names the absent organ and the down runtime honestly', () => {
    const { life } = stubLife(nativeSnapshot({
      source: 'absent',
      availability: { runtime: 'down', legacy: 'disabled', knowledge: 'ok', trainingStream: 'idle' },
    }))
    mountPanel(life)
    expect(screen.getByText(en.sourceAbsent)).not.toBeNull()
    expect(screen.getByText(en.downBadge)).not.toBeNull()
  })

  it('shows the scheduler-stopped copy for a legacy organ that is not running', () => {
    const { life } = stubLife(legacySnapshot({
      life: {
        isRunning: true,
        legacy: {
          isRunning: false,
          lifeState: 'idle',
          dominantNeed: 'energy',
          needs: { energy: 35, curiosity: 80, fatigue: 20, stress: 5, social: 50 },
          totalHeartbeats: 128,
          totalEvents: 17,
          lastHeartbeat: '2026-09-23T07:59:30.000Z',
          lastActivity: '2026-09-23T07:55:00.000Z',
        },
      },
    }))
    mountPanel(life)
    expect(screen.getByText(en.schedulerStopped)).not.toBeNull()
  })

  it('says there is no life reading when the organ block is absent', () => {
    const snapshot = nativeSnapshot()
    const { life: omitted, ...rest } = snapshot
    expect(omitted).toBeDefined()
    const { life } = stubLife(rest)
    mountPanel(life)
    expect(screen.getByText(en.noReading)).not.toBeNull()
  })

  it('keeps one directory group for two files in the same directory', () => {
    const { life } = stubLife(nativeSnapshot({
      training: {
        isTraining: false,
        pauseRequested: false,
        stopRequested: false,
        publishing: false,
        checkpoints: [],
        datasets: [
          { path: 'consolidated/a.jsonl', sizeBytes: 10 },
          { path: 'consolidated/b.jsonl', sizeBytes: 20 },
        ],
      },
    }))
    mountPanel(life)
    expect(screen.getAllByRole('button', { name: /consolidated/ })).toHaveLength(1)
    expect(screen.getByRole('checkbox', { name: /consolidated\/a\.jsonl/ })).not.toBeNull()
    expect(screen.getByRole('checkbox', { name: /consolidated\/b\.jsonl/ })).not.toBeNull()
  })

  it('does not disturb a row an operator already ticked when an upload matches it by basename', async () => {
    const { life, mocks, emit } = stubLife(nativeSnapshot({
      training: {
        isTraining: false,
        pauseRequested: false,
        stopRequested: false,
        publishing: false,
        checkpoints: [],
        datasets: [{ path: 'other/new-set.jsonl', sizeBytes: 10 }],
      },
    }))
    mocks.uploadDataset.mockResolvedValue({ message: 'uploaded' })
    mountPanel(life)

    // The operator ticks the roster row whose basename the upload will carry.
    const existing = screen.getByRole('checkbox', { name: /other\/new-set\.jsonl/ })
    fireEvent.click(existing)
    expect((existing as HTMLInputElement).checked).toBe(true)

    const file = new File(['{"text":"hello"}\n'], 'new-set.jsonl', { type: 'application/jsonl' })
    fireEvent.change(filePicker(), { target: { files: [file] } })
    fireEvent.click(screen.getByRole('button', { name: en.uploadSend }))
    await waitFor(() => { expect(mocks.uploadDataset).toHaveBeenCalledTimes(1) })

    // The refreshed roster gains the uploaded root file, but the auto-select
    // matches the already-ticked row first and leaves the selection as-is.
    emit(withDataset(nativeSnapshot({
      training: {
        isTraining: false,
        pauseRequested: false,
        stopRequested: false,
        publishing: false,
        checkpoints: [],
        datasets: [{ path: 'other/new-set.jsonl', sizeBytes: 10 }],
      },
    }), 'new-set.jsonl', 17))
    await screen.findByRole('checkbox', { name: /^new-set\.jsonl/ })
    expect((screen.getByRole('checkbox', { name: /other\/new-set\.jsonl/ }) as HTMLInputElement).checked).toBe(true)
    expect((screen.getByRole('checkbox', { name: /^new-set\.jsonl/ }) as HTMLInputElement).checked).toBe(false)
  })

  it('names the streaming and closed training-stream states with their dots and warning tags', () => {
    const streaming = stubLife(nativeSnapshot({
      availability: { runtime: 'ok', legacy: 'disabled', knowledge: 'ok', trainingStream: 'streaming' },
      training: {
        isTraining: true,
        pauseRequested: false,
        stopRequested: true,
        publishing: true,
        checkpoints: [],
        progress: {
          fraction: 0.25, step: 5, loss: 1.3, elapsed: 60, eta: 180,
          epoch: 1, totalEpochs: 4, samplesPerSec: 12.5, totalSteps: 20,
        },
      },
    }))
    mountPanel(streaming.life)
    expect(screen.getByText(en.streamStreaming)).not.toBeNull()
    expect(screen.getByText(en.trainingStopRequested)).not.toBeNull()
    expect(screen.getByText(en.trainingPublishing)).not.toBeNull()
    cleanup()

    const closed = stubLife(nativeSnapshot({
      availability: { runtime: 'ok', legacy: 'disabled', knowledge: 'ok', trainingStream: 'closed' },
    }))
    mountPanel(closed.life)
    expect(screen.getByText(en.streamClosed)).not.toBeNull()
  })

  it('ignores a picker change that carries no file at all', () => {
    const { life } = stubLife(nativeSnapshot())
    mountPanel(life)
    fireEvent.change(filePicker(), { target: { files: [] } })
    fireEvent.change(knowledgePicker(), { target: { files: [] } })
    expect(screen.queryByText(/Selected /)).toBeNull()
  })

  it('keeps the upload ack when the follow-up roster read fails', async () => {
    const { life, mocks } = stubLife(nativeSnapshot())
    mocks.uploadDataset.mockResolvedValue({ message: 'uploaded' })
    mocks.refresh.mockRejectedValueOnce(new Error('refresh lost'))
    mountPanel(life)

    const file = new File(['{"text":"hello"}\n'], 'later.jsonl', { type: 'application/jsonl' })
    fireEvent.change(filePicker(), { target: { files: [file] } })
    fireEvent.click(screen.getByRole('button', { name: en.uploadSend }))
    expect(await screen.findByText('uploaded')).not.toBeNull()
    expect(mocks.uploadDataset).toHaveBeenCalledTimes(1)
  })

  it('keeps the ticked datasets and names the refusal when every delete fails, then prunes what went', async () => {
    const { life, mocks } = stubLife(nativeSnapshot())
    mountPanel(life)

    // The spec preselects night-1; ticking the other row arms a two-file delete.
    fireEvent.click(screen.getByRole('checkbox', { name: /dialogue_extended_clean/ }))
    mocks.deleteDataset.mockRejectedValue(new LifeControlError(new RemoteError('life/conflict', 'boom', { reason: 'busy' })))
    fireEvent.click(screen.getByRole('button', { name: en.deletePickedDatasets }))
    fireEvent.click(screen.getByRole('button', { name: t('confirmDeleteDatasets', { count: '2' }) }))
    const refusal = await screen.findByRole('alert')
    expect(refusal.textContent).toBe(t('errConflict', { reason: 'busy' }))
    expect((screen.getByRole('checkbox', { name: /consolidated\/night-1\.jsonl/ }) as HTMLInputElement).checked).toBe(true)
    expect(screen.queryByText(t('deleteDone', { count: '0' }))).toBeNull()

    // A later attempt that removes the first file but loses the roster read
    // still prunes what went and reports the count; the refresh failure is
    // swallowed after a committed delete.
    mocks.deleteDataset.mockReset()
    mocks.deleteDataset.mockResolvedValue({ message: 'deleted' })
    mocks.refresh.mockRejectedValueOnce(new Error('refresh lost'))
    fireEvent.click(screen.getByRole('button', { name: en.deletePickedDatasets }))
    fireEvent.click(screen.getByRole('button', { name: t('confirmDeleteDatasets', { count: '2' }) }))
    await waitFor(() => { expect(screen.getByText(t('deleteDone', { count: '2' }))).not.toBeNull() })
    await waitFor(() => { expect(screen.getByText(/0 selected/)).not.toBeNull() })
  })

  it('lets an operator untick a checkpoint, keeps rows when deletes fail, and reports what went', async () => {
    const { life, mocks } = stubLife(nativeSnapshot({
      artifacts: { activeId: '', configuredId: '' },
      training: {
        isTraining: false,
        pauseRequested: false,
        stopRequested: false,
        publishing: false,
        checkpoints: [
          { filename: 'old_run.pt', step: 20, bytes: 4096, modifiedUtc: '2026-09-23T09:00:00.000Z', savedAtUtc: '2026-09-23T09:00:00.000Z', numEpochs: 1 },
          { filename: 'older.pt', step: 30, bytes: 8192, modifiedUtc: '2026-09-23T10:00:00.000Z', savedAtUtc: '2026-09-23T10:00:00.000Z', numEpochs: 1 },
        ],
      },
    }))
    mountPanel(life)

    const first = screen.getByRole('checkbox', { name: `${en.colSelect} old_run.pt` }) as HTMLInputElement
    fireEvent.click(first)
    expect(first.checked).toBe(true)
    // Unticking drops it from the armed deletion.
    fireEvent.click(first)
    expect(first.checked).toBe(false)

    fireEvent.click(screen.getByRole('checkbox', { name: `${en.colSelect} old_run.pt` }))
    fireEvent.click(screen.getByRole('checkbox', { name: `${en.colSelect} older.pt` }))
    mocks.deleteCheckpoint.mockRejectedValue(new LifeControlError(new RemoteError('life/conflict', 'boom', { reason: 'in use' })))
    fireEvent.click(screen.getByRole('button', { name: en.deletePickedCheckpoints }))
    fireEvent.click(screen.getByRole('button', { name: t('confirmDeleteCheckpoints', { count: '2' }) }))
    await screen.findByRole('alert')
    expect((screen.getByRole('checkbox', { name: `${en.colSelect} old_run.pt` }) as HTMLInputElement).checked).toBe(true)
    expect(screen.queryByText(t('deleteDone', { count: '0' }))).toBeNull()

    mocks.deleteCheckpoint.mockReset()
    mocks.deleteCheckpoint.mockResolvedValue({ message: 'deleted' })
    mocks.refresh.mockRejectedValueOnce(new Error('refresh lost'))
    fireEvent.click(screen.getByRole('button', { name: en.deletePickedCheckpoints }))
    fireEvent.click(screen.getByRole('button', { name: t('confirmDeleteCheckpoints', { count: '2' }) }))
    await waitFor(() => { expect(screen.getByText(t('deleteDone', { count: '2' }))).not.toBeNull() })
  })

  it('runs the pause, resume and reset controls through their armed states', async () => {
    const { life, mocks } = stubLife(nativeSnapshot({
      training: {
        isTraining: true,
        pauseRequested: false,
        stopRequested: false,
        publishing: false,
        checkpoints: [],
      },
    }))
    mountPanel(life)

    fireEvent.click(screen.getByRole('button', { name: en.trainPause }))
    await waitFor(() => { expect(mocks.trainPause).toHaveBeenCalledTimes(1) })

    // Reset is a confirming verb: the first click only arms the line.
    fireEvent.click(screen.getByRole('button', { name: en.trainReset }))
    expect(screen.getByText(en.confirmReset)).not.toBeNull()
    expect(mocks.trainReset).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: en.trainReset }))
    await waitFor(() => { expect(mocks.trainReset).toHaveBeenCalledTimes(1) })
    cleanup()

    const pausing = stubLife(nativeSnapshot({
      training: {
        isTraining: true,
        pauseRequested: true,
        stopRequested: false,
        publishing: false,
        checkpoints: [],
      },
    }))
    mountPanel(pausing.life)
    fireEvent.click(screen.getByRole('button', { name: en.trainResume }))
    await waitFor(() => { expect(pausing.mocks.trainResume).toHaveBeenCalledTimes(1) })
  })

  it('picks and cancels a dataset through the visible buttons', () => {
    const { life } = stubLife(nativeSnapshot())
    mountPanel(life)

    // The Pick button drives the same input the tests fill directly; clicking
    // it must not throw in jsdom, where the chooser is a no-op.
    fireEvent.click(screen.getByRole('button', { name: en.uploadPick }))
    expect(screen.queryByRole('button', { name: en.uploadSend })).toBeNull()

    const file = new File(['{"text":"hello"}\n'], 'cancel-me.jsonl', { type: 'application/jsonl' })
    fireEvent.change(filePicker(), { target: { files: [file] } })
    expect(screen.getByRole('button', { name: en.uploadSend })).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: en.uploadCancel }))
    expect(screen.queryByRole('button', { name: en.uploadSend })).toBeNull()
  })

  it('says the roster is empty when the runtime lists no datasets', () => {
    const { life } = stubLife(nativeSnapshot({
      training: {
        isTraining: false,
        pauseRequested: false,
        stopRequested: false,
        publishing: false,
        checkpoints: [],
        datasets: [],
      },
    }))
    mountPanel(life)
    expect(screen.getByText(en.datasetsEmpty)).not.toBeNull()
  })

  it('refuses an oversized knowledge document without reading it', () => {
    const { life, mocks } = stubLife(nativeSnapshot())
    mountPanel(life)

    const file = new File([new Uint8Array(1)], 'huge.md')
    Object.defineProperty(file, 'size', { value: 200 * 1024 * 1024 + 1 })
    fireEvent.change(knowledgePicker(), { target: { files: [file] } })

    expect(screen.getByRole('alert').textContent).toBe('The file exceeds 200 MB. Use a smaller dataset.')
    expect(mocks.uploadKnowledge).not.toHaveBeenCalled()
  })

  it('says the knowledge surface is down instead of rendering an empty library', () => {
    const { life } = stubLife(nativeSnapshot({
      availability: { runtime: 'ok', legacy: 'disabled', knowledge: 'down', trainingStream: 'idle' },
    }))
    mountPanel(life)
    expect(screen.getByText(en.knowledgeUnavailable)).not.toBeNull()
    expect(screen.queryByRole('checkbox', { name: 'handbook.md' })).toBeNull()
  })

  it('shows a checkpoint whose savedAt is missing from the modified time', () => {
    const { life } = stubLife(nativeSnapshot({
      training: {
        isTraining: false,
        pauseRequested: false,
        stopRequested: false,
        publishing: false,
        checkpoints: [
          { filename: 'fresh.pt', step: 5, bytes: 2048, modifiedUtc: '2026-09-23T08:01:00.000Z', savedAtUtc: '', numEpochs: 1 },
        ],
      },
    }))
    mountPanel(life)
    // The row's timestamp cell falls back to the modified instant when no
    // savedAt is recorded; the row still names its file wherever it appears.
    expect(screen.getAllByText('fresh.pt').length).toBeGreaterThanOrEqual(1)
  })

  it('keeps the knowledge upload alive when the follow-up list read fails', async () => {
    const { life, mocks } = stubLife(nativeSnapshot())
    mocks.uploadKnowledge.mockResolvedValue({ message: 'knowledge uploaded' })
    mocks.refresh.mockRejectedValueOnce(new Error('refresh lost'))
    mountPanel(life)

    const file = new File(['# notes\n'], 'again.md', { type: 'text/markdown' })
    fireEvent.change(knowledgePicker(), { target: { files: [file] } })
    fireEvent.click(screen.getByRole('button', { name: en.uploadSend }))
    expect(await screen.findByText('knowledge uploaded')).not.toBeNull()
  })

  it('drives the knowledge upload through its pick and cancel buttons', () => {
    const { life } = stubLife(nativeSnapshot())
    mountPanel(life)

    fireEvent.click(screen.getByRole('button', { name: en.knowledgeUpload }))
    const file = new File(['# notes\n'], 'cancel-me.md', { type: 'text/markdown' })
    fireEvent.change(knowledgePicker(), { target: { files: [file] } })
    expect(screen.getByRole('button', { name: en.uploadSend })).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: en.uploadCancel }))
    expect(screen.queryByRole('button', { name: en.uploadSend })).toBeNull()
  })

  it('lets an operator untick a knowledge file, keeps rows when deletes fail, and reports what went', async () => {
    const { life, mocks } = stubLife(nativeSnapshot())
    mountPanel(life)

    const handbook = screen.getByRole('checkbox', { name: 'handbook.md' }) as HTMLInputElement
    fireEvent.click(handbook)
    expect(handbook.checked).toBe(true)
    fireEvent.click(handbook)
    expect(handbook.checked).toBe(false)

    fireEvent.click(screen.getByRole('checkbox', { name: 'handbook.md' }))
    fireEvent.click(screen.getByRole('checkbox', { name: 'loose.txt' }))
    mocks.deleteKnowledge.mockRejectedValue(new LifeControlError(new RemoteError('life/conflict', 'boom', { reason: 'busy' })))
    fireEvent.click(screen.getByRole('button', { name: en.deletePickedKnowledge }))
    fireEvent.click(screen.getByRole('button', { name: t('confirmDeleteKnowledge', { count: '2' }) }))
    await screen.findByRole('alert')
    expect((screen.getByRole('checkbox', { name: 'handbook.md' }) as HTMLInputElement).checked).toBe(true)
    expect(screen.queryByText(t('deleteDone', { count: '0' }))).toBeNull()

    mocks.deleteKnowledge.mockReset()
    mocks.deleteKnowledge.mockResolvedValue({ message: 'deleted' })
    mocks.refresh.mockRejectedValueOnce(new Error('refresh lost'))
    fireEvent.click(screen.getByRole('button', { name: en.deletePickedKnowledge }))
    fireEvent.click(screen.getByRole('button', { name: t('confirmDeleteKnowledge', { count: '2' }) }))
    await waitFor(() => { expect(screen.getByText(t('deleteDone', { count: '2' }))).not.toBeNull() })
  })

  it('renders an empty knowledge library, silent embeddings, and an absent embedding dimension', () => {
    const { life } = stubLife(nativeSnapshot({
      knowledge: {
        docCount: 0,
        chunkCount: 0,
        hasEmbeddings: false,
        embedDim: 0,
        files: [],
      },
    }))
    mountPanel(life)
    expect(screen.getByText(en.knowledgeEmpty)).not.toBeNull()
    expect(screen.getByText(en.embeddingsNo)).not.toBeNull()
    // hasEmbeddings false renders the idle dot beside the "yes" label.
    expect(screen.getByText(en.embeddingsYes)).not.toBeNull()
  })

  it('shows a running pass, an empty spec roster, and report lines without notes', () => {
    const { life } = stubLife(nativeSnapshot({
      consolidation: {
        passes: 0,
        lastPassAt: 0,
        lastCorpus: '',
        projectedDigests: 0,
        running: true,
        spec: { reason: 'waiting for interaction', datasets: [], weaknesses: [] },
        lastReport: {
          reason: 'manual',
          specReason: '',
          durationMs: 500,
          weaknesses: [],
          notes: [],
          workbenchCapabilities: 16,
        },
        journal: { entries: 0, byKind: {}, sessions: 0, lastRecordedAt: 0 },
      },
    }))
    mountPanel(life)
    expect(screen.getByText(en.passRunning)).not.toBeNull()
    expect(screen.getAllByText(en.notYet).length).toBeGreaterThanOrEqual(3)
    expect(screen.getByText(en.noWeaknesses)).not.toBeNull()
    expect(screen.getByText(en.noNotes)).not.toBeNull()
    // C6 ⑥：这一行把 pass 自己折算的工作台能力数带上面板，零也照实写零。
    expect(screen.getByText(en.passCapabilitiesLabel)).not.toBeNull()
    expect(screen.getAllByText('16').length).toBeGreaterThanOrEqual(1)
  })

  it('says the host projection is empty when no host surface answered', () => {
    const snapshot = nativeSnapshot()
    const { health, memory, workbench, auth, ...rest } = snapshot
    expect(health).toBeDefined()
    expect(memory).toBeDefined()
    expect(workbench).toBeDefined()
    expect(auth).toBeDefined()
    const { life } = stubLife(rest)
    mountPanel(life)
    expect(screen.getAllByText(en.noReading).length).toBeGreaterThanOrEqual(1)
  })

  it('renders a host reading without auth and with degraded health facts', () => {
    const snapshot = nativeSnapshot({
      health: { state: 'degraded', modelLoaded: false, modelName: '', seedActive: false, startupComplete: false },
    })
    const { auth, ...rest } = snapshot
    expect(auth).toBeDefined()
    const { life } = stubLife(rest)
    mountPanel(life)
    expect(screen.getByText(en.startupIncomplete)).not.toBeNull()
    expect(screen.getByText(en.modelNone)).not.toBeNull()
    expect(screen.getByText(en.seedInactive)).not.toBeNull()
    expect(screen.queryByText(en.authDisabled)).toBeNull()
  })
})
