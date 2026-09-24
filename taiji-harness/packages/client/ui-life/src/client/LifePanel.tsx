/**
 * The global Life page: what the Taiji local runtime reported last, and the
 * controls that act on it. Six sections top to bottom — where the reading
 * came from, the life organs, training with its checkpoint roster, the
 * knowledge base, memory with its consolidation products, and the host
 * projection. Every number comes from the controller's snapshot stream;
 * a control failure renders the Host's stable error code, never the raw RPC
 * text, and an accepted action refreshes through the stream instead of a
 * local state write.
 */

import { useCallback, useEffect, useSyncExternalStore, useState, type ReactNode } from 'react'
import type { ILife, LifeSnapshotState } from '@taiji/dsh-api-life-controller/client'
import type {
  LifeCheckpointView,
  LifeLegacyView,
  LifeNativeView,
  LifeProgressView,
  LifeSnapshot,
} from '@taiji/dsh-api-life-controller/client'
import type { RemoteFailure } from '@taiji/dsh-typert-protocol'
import { Button, StateDot, Tag, type StateDotState } from '@taiji/dsh-client-ui-primitives'
import type { InjectFace, PropsLocale, PropsRuntime } from '@taiji/dsh-client-ui-slots'
import type { LifeLocaleKey } from './locales.ts'
import css from './LifePanel.module.css'

/** Injected share of the panel: the Client Life facade. */
export interface LifePanelInjected {
  /** State and control verbs, backed by the snapshot stream. */
  readonly life: ILife
}

/** Full component props assembled by the main slot renderer. */
export type LifePanelProps =
  PropsRuntime<'main'>
  & PropsLocale<'life'>
  & InjectFace<LifePanelInjected>

/** A control the panel is waiting on, or one awaiting its confirming click. */
type PendingVerb = 'lifeStart' | 'lifeStop' | 'feed' | 'sleep' | 'play' | 'trainStart' | 'trainResumeCheckpoint' | 'trainPause' | 'trainResume' | 'trainStop' | 'trainReset' | 'consolidate' | 'activateCheckpoint'

/** The verbs a confirming second click protects. */
const CONFIRMED: ReadonlySet<PendingVerb> = new Set(['trainStop', 'trainReset', 'activateCheckpoint'])

/** Format one ISO instant for a fact row; an unparseable value passes through. */
function formatInstant(iso: string): string {
  const parsed = new Date(iso)
  return Number.isNaN(parsed.getTime()) ? iso : parsed.toLocaleString()
}

/** Format a byte count in the largest unit that stays readable. */
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/** Clamp a 0..100 meter to its track. */
function clampPct(value: number): number {
  return Math.min(100, Math.max(0, value))
}

/**
 * Read the Host refusal a `ctx.life` verb rejected with. The panel matches it
 * structurally rather than importing the controller's error class, because a
 * cross-plugin value import is what this client forbids; the two packages
 * collaborate through the `life` service instead.
 * @param error - whatever a control verb rejected with.
 * @returns the refusal, or undefined when the rejection carries none.
 */
function refusalOf(error: unknown): RemoteFailure | undefined {
  if (typeof error !== 'object' || error === null) return undefined
  const candidate = (error as { rpcError?: unknown }).rpcError
  return typeof candidate === 'object' && candidate !== null && 'code' in candidate && 'message' in candidate
    ? candidate as RemoteFailure
    : undefined
}

/** Resolve the panel copy for one Host failure. */
function errorText(rpc: RemoteFailure, t: LifePanelProps['t']): string {
  const details = (typeof rpc.details === 'object' && rpc.details !== null ? rpc.details : {}) as Record<string, unknown>
  switch (rpc.code) {
    case 'life/runtime-error':
      return t('errRuntimeError', { status: String(details.status ?? '') })
    case 'life/runtime-unreachable':
      return t('errUnreachable', { reason: String(details.reason ?? '') })
    case 'life/unavailable':
      return t('errUnavailable', { reason: String(details.reason ?? '') })
    case 'life/conflict':
      return t('errConflict', { reason: String(details.reason ?? '') })
    case 'life/bad-request':
      return t('errBadRequest', { field: String(details.field ?? ''), reason: String(details.reason ?? '') })
    case 'life/stream-failed':
      return t('errStreamFailed')
    default:
      return t('errFallback', { code: rpc.code })
  }
}

/**
 * Render the global Life page.
 * @param props - runtime share, the panel dictionaries, and the Life facade.
 * @returns the page element.
 */
export function LifePanel({ t, life }: LifePanelProps): ReactNode {
  // React calls both callbacks as bare functions, so the service's methods must
  // be bound here: an unbound `life.getSnapshot` would lose its receiver.
  const subscribe = useCallback((listener: () => void): (() => void) => life.subscribe(listener), [life])
  const readState = useCallback((): LifeSnapshotState => life.getSnapshot(), [life])
  const state = useSyncExternalStore(subscribe, readState)
  const [pending, setPending] = useState<PendingVerb | null>(null)
  const [confirming, setConfirming] = useState<PendingVerb | null>(null)
  const [failureText, setFailureText] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)

  /** Run one control verb, surfacing a refusal as the Host's stable error code. */
  const run = (verb: PendingVerb, invoke: () => Promise<unknown>): void => {
    if (CONFIRMED.has(verb) && confirming !== verb) {
      setConfirming(verb)
      return
    }
    setConfirming(null)
    setPending(verb)
    void invoke()
      .then(() => { setFailureText(null) })
      .catch((error: unknown) => {
        const refusal = refusalOf(error)
        setFailureText(refusal === undefined ? t('errStreamFailed') : errorText(refusal, t))
      })
      .finally(() => { setPending(null) })
  }

  if (state.state === 'loading') {
    return <div className={css.page}><p className={css.loading}>{t('loading')}</p></div>
  }

  if (state.state === 'error' || state.snapshot === undefined) {
    return (
      <div className={css.page}>
        <p className={css.errorLine}>{t('errorTitle')}</p>
        <Button
          disabled={refreshing}
          onClick={() => {
            setRefreshing(true)
            void life.refresh().catch(() => {}).finally(() => { setRefreshing(false) })
          }}
        >
          {t('retry')}
        </Button>
      </div>
    )
  }

  const snapshot = state.snapshot
  const shared = { t, snapshot, pending, run, life }

  return (
    <div className={css.page}>
      <header className={css.header}>
        <h2 className={css.title}>{t('title')}</h2>
        <p className={css.subtitle}>{t('subtitle')}</p>
      </header>

      <SourceSection t={t} snapshot={snapshot} />
      <LifeSection {...shared} />
      <TrainingSection {...shared} />
      <KnowledgeSection t={t} snapshot={snapshot} />
      <ConsolidationSection {...shared} />
      <HostSection t={t} snapshot={snapshot} />

      {failureText !== null && <p className={css.errorLine} role="alert">{failureText}</p>}
      {confirming !== null && (
        <p className={css.confirmLine}>
          {t(confirming === 'trainStop' ? 'confirmStop' : confirming === 'trainReset' ? 'confirmReset' : 'confirmActivate')}
        </p>
      )}
    </div>
  )
}

/** Shared props of the two control-carrying sections. */
interface SectionControlProps {
  readonly t: LifePanelProps['t']
  readonly snapshot: LifeSnapshot
  readonly pending: PendingVerb | null
  readonly run: (verb: PendingVerb, invoke: () => Promise<unknown>) => void
  readonly life: ILife
}

/**
 * Roster selection preselected from the data ring's spec: the spec's dataset
 * list applies itself whenever that list changes (including the first poll
 * that brings the roster in), while a manual choice survives later roster
 * refreshes — the poll replaces the array identity, never the user's ticks.
 * @param specDatasets - dataset paths the spec names, absent without a spec.
 * @param roster - the trainable roster, absent when the read did not answer.
 * @returns the current selection and the toggle for one roster entry.
 */
function useDatasetSelection(
  specDatasets: readonly string[] | undefined,
  roster: readonly { path: string }[] | undefined,
): { selected: ReadonlySet<string>; toggle: (path: string) => void } {
  const specKey = specDatasets?.join('\n') ?? null
  const [selected, setSelected] = useState<ReadonlySet<string>>(() => new Set())
  const [appliedKey, setAppliedKey] = useState<string | null>(null)
  useEffect(() => {
    if (specKey === null || roster === undefined || appliedKey === specKey) return
    const paths = new Set(roster.map(entry => entry.path))
    setSelected(new Set(specKey.split('\n').filter(path => path !== '' && paths.has(path))))
    setAppliedKey(specKey)
  }, [specKey, roster, appliedKey])
  /** Tick one roster entry without disturbing the rest of the selection. */
  const toggle = (path: string): void => {
    setSelected((current) => {
      const next = new Set(current)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }
  return { selected, toggle }
}

/** One labeled fact row. */
function Fact({ label, children }: { label: string; children: ReactNode }): ReactNode {
  return (
    <div className={css.fact}>
      <span className={css.factLabel}>{label}</span>
      <span className={css.factValue}>{children}</span>
    </div>
  )
}

/** One horizontal 0..100 meter. */
function Meter({ label, value }: { label: string; value: number }): ReactNode {
  return (
    <div className={css.meter}>
      <span className={css.meterLabel}>{label}</span>
      <span className={css.meterTrack}>
        <span className={css.meterFill} style={{ width: `${clampPct(value)}%` }} />
      </span>
      <span className={css.meterValue}>{value.toFixed(1)}</span>
    </div>
  )
}

/** Meters for one open need/drive map, insertion-ordered. */
function MeterMap({ t, values }: { t: LifePanelProps['t']; values: Readonly<Record<string, number>> }): ReactNode {
  const entries = Object.entries(values)
  if (entries.length === 0) return <p className={css.muted}>{t('noReadings')}</p>
  return (
    <div className={css.meters}>
      {entries.map(([name, value]) => <Meter key={name} label={name} value={value} />)}
    </div>
  )
}

/** The section-heading key for whichever organ answered. */
function sourceKeyOf(snapshot: LifeSnapshot): LifeLocaleKey {
  return snapshot.source === 'native' ? 'sourceNative' : snapshot.source === 'legacy' ? 'sourceLegacy' : 'sourceAbsent'
}

/** Section 1: which organ answered, when, and which sources did not. */
function SourceSection({ t, snapshot }: { t: LifePanelProps['t']; snapshot: LifeSnapshot }): ReactNode {
  const badge = snapshot.availability.runtime === 'down'
    ? { key: 'downBadge' as LifeLocaleKey, dot: 'error' as StateDotState }
    : snapshot.fresh
      ? { key: 'freshBadge' as LifeLocaleKey, dot: 'done' as StateDotState }
      : { key: 'staleBadge' as LifeLocaleKey, dot: 'warning' as StateDotState }
  return (
    <section className={css.section} aria-label={t('sectionSource')}>
      <h3 className={css.sectionTitle}>{t('sectionSource')}</h3>
      <div className={css.facts}>
        <Fact label={t('sourceLabel')}>
          {t(sourceKeyOf(snapshot))}
          <StateDot state={badge.dot} /> <span className={css.badgeText}>{t(badge.key)}</span>
        </Fact>
        <Fact label={t('observedAtLabel')}>{formatInstant(snapshot.observedAt)}</Fact>
        <Fact label={t('pollLabel')}>{t('secondsShort', { count: String(snapshot.pollIntervalMs / 1000) })}</Fact>
      </div>
      {snapshot.unavailable.length > 0 && (
        <div className={css.unavailable}>
          <p className={css.muted}>{t('unavailableTitle')}</p>
          <ul className={css.unavailableList}>
            {snapshot.unavailable.map(line => <li key={line}>{line}</li>)}
          </ul>
        </div>
      )}
    </section>
  )
}

/** The native organ block: mode, tick, and the open need and drive meters. */
function NativeOrgan({ t, native }: { t: LifePanelProps['t']; native: LifeNativeView }): ReactNode {
  return (
    <div className={css.organ}>
      <h4 className={css.organTitle}>{t('organNative')}</h4>
      <div className={css.facts}>
        <Fact label={t('modeLabel')}>{native.mode}</Fact>
        <Fact label={t('tickLabel')}>{native.tick}</Fact>
      </div>
      <p className={css.groupLabel}>{t('needsLabel')}</p>
      <MeterMap t={t} values={native.needs} />
      <p className={css.groupLabel}>{t('drivesLabel')}</p>
      <MeterMap t={t} values={native.drives} />
    </div>
  )
}

/** The legacy organ block: activity, dominant need, meters, and counters. */
function LegacyOrgan({ t, legacy }: { t: LifePanelProps['t']; legacy: LifeLegacyView }): ReactNode {
  return (
    <div className={css.organ}>
      <h4 className={css.organTitle}>{t('organLegacy')}</h4>
      <div className={css.facts}>
        <Fact label={t('lifeStateLabel')}>{legacy.isRunning ? legacy.lifeState : t('schedulerStopped')}</Fact>
        <Fact label={t('dominantLabel')}>{legacy.dominantNeed}</Fact>
        <Fact label={t('heartbeatsLabel')}>{legacy.totalHeartbeats}</Fact>
        <Fact label={t('eventsLabel')}>{legacy.totalEvents}</Fact>
        {legacy.lastHeartbeat !== undefined && <Fact label={t('lastHeartbeatLabel')}>{formatInstant(legacy.lastHeartbeat)}</Fact>}
        {legacy.lastActivity !== undefined && <Fact label={t('lastActivityLabel')}>{formatInstant(legacy.lastActivity)}</Fact>}
      </div>
      <p className={css.groupLabel}>{t('needsLabel')}</p>
      <MeterMap t={t} values={legacy.needs} />
    </div>
  )
}

/** Section 2: the organs' readings plus the legacy scheduler controls. */
function LifeSection({ t, snapshot, pending, run, life }: SectionControlProps): ReactNode {
  const lifeView = snapshot.life
  const busy = pending !== null
  return (
    <section className={css.section} aria-label={t('sectionLife')}>
      <h3 className={css.sectionTitle}>{t('sectionLife')}</h3>
      {lifeView === undefined
        ? <p className={css.muted}>{t('noReading')}</p>
        : (
          <>
            {lifeView.native !== undefined && <NativeOrgan t={t} native={lifeView.native} />}
            {lifeView.legacy !== undefined && <LegacyOrgan t={t} legacy={lifeView.legacy} />}
          </>
        )}
      <div className={css.actions} role="group" aria-label={t('organLegacy')}>
        <Button disabled={busy} onClick={() =>{  run('lifeStart', () => life.lifeStart()) }}>{t('lifeStart')}</Button>
        <Button disabled={busy} onClick={() =>{  run('lifeStop', () => life.lifeStop()) }}>{t('lifeStop')}</Button>
        <Button disabled={busy} onClick={() =>{  run('feed', () => life.lifeAction({ action: 'feed' })) }}>{t('actionFeed')}</Button>
        <Button disabled={busy} onClick={() =>{  run('sleep', () => life.lifeAction({ action: 'sleep' })) }}>{t('actionSleep')}</Button>
        <Button disabled={busy} onClick={() =>{  run('play', () => life.lifeAction({ action: 'play' })) }}>{t('actionPlay')}</Button>
      </div>
    </section>
  )
}

/** The checkpoint roster table, each row offering a resumed run and an activation. */
function Checkpoints({ t, checkpoints, artifacts, busy, onActivate, onResume }: {
  t: LifePanelProps['t']
  checkpoints: readonly LifeCheckpointView[]
  artifacts: LifeSnapshot['artifacts']
  busy: boolean
  onActivate: (filename: string) => void
  onResume: (filename: string) => void
}): ReactNode {
  if (checkpoints.length === 0) return <p className={css.muted}>{t('checkpointsEmpty')}</p>
  return (
    <table className={css.table}>
      <thead>
        <tr>
          <th scope="col">{t('colName')}</th>
          <th scope="col">{t('colStep')}</th>
          <th scope="col">{t('colSize')}</th>
          <th scope="col">{t('colSaved')}</th>
          <th scope="col">{t('colResume')}</th>
        </tr>
      </thead>
      <tbody>
        {checkpoints.map(cp => (
          <tr key={cp.filename}>
            <td>
              {cp.filename}
              {artifacts?.activeId === cp.filename && <Tag tone="solid">{t('artifactsActiveBadge')}</Tag>}
              {artifacts !== undefined && artifacts.configuredId === cp.filename && artifacts.configuredId !== artifacts.activeId && (
                <Tag tone="info">{t('artifactsConfiguredBadge')}</Tag>
              )}
            </td>
            <td>{cp.step}</td>
            <td>{formatBytes(cp.bytes)}</td>
            <td>{cp.savedAtUtc !== '' ? formatInstant(cp.savedAtUtc) : formatInstant(cp.modifiedUtc)}</td>
            <td>
              <div className={css.actions}>
                <Button disabled={busy} onClick={() => { onResume(cp.filename) }}>{t('resumeFrom')}</Button>
                <Button
                  disabled={busy || artifacts?.activeId === cp.filename}
                  onClick={() => { onActivate(cp.filename) }}
                >
                  {t('activateRow')}
                </Button>
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** One training progress metric. */
function ProgressMetric({ label, value }: { label: string; value: ReactNode }): ReactNode {
  return (
    <span className={css.progressMetric}>
      <span className={css.factLabel}>{label}</span> {value}
    </span>
  )
}

/** The progress block for the latest sample. */
function Progress({ t, progress }: { t: LifePanelProps['t']; progress: LifeProgressView }): ReactNode {
  return (
    <div className={css.progress}>
      <span className={css.progressTrack}>
        <span className={css.progressFill} style={{ width: `${clampPct(progress.fraction * 100)}%` }} />
      </span>
      <div className={css.progressMetrics}>
        <ProgressMetric label={t('lossLabel')} value={progress.loss.toFixed(3)} />
        <ProgressMetric label={t('stepLabel')} value={`${progress.step} / ${progress.totalSteps}`} />
        <ProgressMetric label={t('epochLabel')} value={`${progress.epoch} / ${progress.totalEpochs}`} />
        {progress.eta !== undefined && <ProgressMetric label={t('etaLabel')} value={t('secondsShort', { count: String(Math.round(progress.eta)) })} />}
        <ProgressMetric label={t('rateLabel')} value={t('rateUnit', { count: progress.samplesPerSec.toFixed(1) })} />
      </div>
    </div>
  )
}

/** Section 3: training state badges, progress, controls, and the roster. */
function TrainingSection({ t, snapshot, pending, run, life }: SectionControlProps): ReactNode {
  const training = snapshot.training
  const streamKey: LifeLocaleKey = snapshot.availability.trainingStream === 'streaming'
    ? 'streamStreaming'
    : snapshot.availability.trainingStream === 'closed'
      ? 'streamClosed'
      : 'streamIdle'
  const busy = pending !== null
  const active = training.isTraining
  const datasets = training.datasets
  const rosterPaths = datasets === undefined ? undefined : new Set(datasets.map(dataset => dataset.path))
  const specDatasets = snapshot.consolidation?.spec?.datasets
  const specMissing = rosterPaths === undefined || specDatasets === undefined
    ? []
    : specDatasets.filter(path => !rosterPaths.has(path))
  const { selected, toggle } = useDatasetSelection(specDatasets, datasets)

  return (
    <section className={css.section} aria-label={t('sectionTraining')}>
      <h3 className={css.sectionTitle}>{t('sectionTraining')}</h3>
      <div className={css.facts}>
        <Fact label={t('streamLabel')}>
          <StateDot state={snapshot.availability.trainingStream === 'streaming' ? 'ongoing' : snapshot.availability.trainingStream === 'closed' ? 'error' : 'idle'} />
          {t(streamKey)}
        </Fact>
      </div>
      <p className={css.trainingBadges}>
        <Tag tone={active ? 'solid' : 'neutral'}>{active ? t('trainingRunning') : t('trainingIdle')}</Tag>
        {training.pauseRequested && <Tag tone="warning">{t('trainingPauseRequested')}</Tag>}
        {training.stopRequested && <Tag tone="warning">{t('trainingStopRequested')}</Tag>}
        {training.publishing && <Tag tone="info">{t('trainingPublishing')}</Tag>}
      </p>
      {training.progress !== undefined ? <Progress t={t} progress={training.progress} /> : <p className={css.muted}>{t('noProgress')}</p>}
      {(training.warnings ?? []).map(message => (
        <p key={message} className={css.trainingBadges}>
          <Tag tone="warning">{t('runWarning')}</Tag> <span className={css.badgeText}>{message}</span>
        </p>
      ))}
      <h4 className={css.organTitle}>{t('datasetsTitle')}</h4>
      {datasets === undefined
        ? <p className={css.muted}>{t('datasetsUnavailable')}</p>
        : datasets.length === 0
          ? <p className={css.muted}>{t('datasetsEmpty')}</p>
          : (
            <>
              <ul className={css.datasetList}>
                {datasets.map(dataset => (
                  <li key={dataset.path} className={css.datasetRow}>
                    <label>
                      <input
                        type="checkbox"
                        checked={selected.has(dataset.path)}
                        disabled={busy || active}
                        onChange={() => { toggle(dataset.path) }}
                      />
                      <span className={css.datasetPath}>{dataset.path}</span>
                    </label>
                    <span className={css.datasetSize}>{formatBytes(dataset.sizeBytes)}</span>
                  </li>
                ))}
              </ul>
              <p className={css.muted}>
                {t('datasetsSelected', { count: String(selected.size) })}
                {specDatasets === undefined ? '' : ` · ${t('specNames', { count: String(specDatasets.length) })}`}
              </p>
              {specMissing.length > 0 && (
                <p className={css.muted}>{t('specMissing', { list: specMissing.join(', ') })}</p>
              )}
              {selected.size === 0 && <p className={css.muted}>{t('datasetsDefaultHint')}</p>}
            </>
          )}
      <div className={css.actions} role="group" aria-label={t('sectionTraining')}>
        <Button
          disabled={busy || active}
          onClick={() => { run('trainStart', () => life.trainStart(selected.size > 0 ? { datasets: [...selected] } : {})) }}
        >
          {t('trainStart')}
        </Button>
        <Button disabled={busy || !active || training.stopRequested} onClick={() =>{  run('trainPause', () => life.trainPause()) }}>{t('trainPause')}</Button>
        <Button disabled={busy || !active || training.stopRequested} onClick={() =>{  run('trainResume', () => life.trainResume()) }}>{t('trainResume')}</Button>
        <Button disabled={busy || !active} onClick={() =>{  run('trainStop', () => life.trainStop()) }}>{t('trainStop')}</Button>
        <Button disabled={busy || !active} onClick={() =>{  run('trainReset', () => life.trainReset()) }}>{t('trainReset')}</Button>
      </div>
      <h4 className={css.organTitle}>{t('checkpointsTitle')}</h4>
      {snapshot.artifacts === undefined
        ? <p className={css.muted}>{t('artifactsUnavailable')}</p>
        : (
          <div className={css.facts}>
            <Fact label={t('activeCheckpointLabel')}>
              {snapshot.artifacts.activeId === '' ? t('builtInModel') : snapshot.artifacts.activeId}
            </Fact>
            {snapshot.artifacts.configuredId !== snapshot.artifacts.activeId && (
              <Fact label={t('configuredCheckpointLabel')}>
                {snapshot.artifacts.configuredId === '' ? t('builtInModel') : snapshot.artifacts.configuredId}
              </Fact>
            )}
            <div className={css.actions}>
              <Button
                disabled={busy || snapshot.artifacts.activeId === ''}
                onClick={() => { run('activateCheckpoint', () => life.activateCheckpoint({ checkpointId: '' })) }}
              >
                {t('activateBuiltin')}
              </Button>
            </div>
          </div>
        )}
      <Checkpoints
        t={t}
        checkpoints={training.checkpoints}
        artifacts={snapshot.artifacts}
        busy={busy}
        onActivate={(filename) => { run('activateCheckpoint', () => life.activateCheckpoint({ checkpointId: filename })) }}
        onResume={(filename) => {
          run('trainResumeCheckpoint', () => life.trainResumeCheckpoint({
            checkpoint: filename,
            ...(selected.size > 0 ? { datasets: [...selected] } : {}),
          }))
        }}
      />
    </section>
  )
}

/** Section 4: the knowledge base size, or its unavailability. */
function KnowledgeSection({ t, snapshot }: { t: LifePanelProps['t']; snapshot: LifeSnapshot }): ReactNode {
  const knowledge = snapshot.availability.knowledge === 'ok' ? snapshot.knowledge : undefined
  return (
    <section className={css.section} aria-label={t('sectionKnowledge')}>
      <h3 className={css.sectionTitle}>{t('sectionKnowledge')}</h3>
      {knowledge === undefined
        ? <p className={css.muted}>{t('knowledgeUnavailable')}</p>
        : (
          <div className={css.facts}>
            <Fact label={t('docsLabel')}>{knowledge.docCount}</Fact>
            <Fact label={t('chunksLabel')}>{knowledge.chunkCount}</Fact>
            <Fact label={t('embedDimLabel')}>{knowledge.embedDim > 0 ? knowledge.embedDim : t('embeddingsNo')}</Fact>
            <Fact label={t('embeddingsYes')}><StateDot state={knowledge.hasEmbeddings ? 'done' : 'idle'} /></Fact>
          </div>
        )}
    </section>
  )
}

/** One line of a runtime-owned string list, rendered as its own row. */
function LineList({ title, lines, empty }: { title: string; lines: readonly string[]; empty: string }): ReactNode {
  return (
    <>
      <p className={css.groupLabel}>{title}</p>
      {lines.length === 0
        ? <p className={css.muted}>{empty}</p>
        : <ul className={css.unavailableList}>{lines.map(line => <li key={line}>{line}</li>)}</ul>}
    </>
  )
}

/** Section 5: the memory journal, the consolidation passes, and their products. */
function ConsolidationSection({ t, snapshot, pending, run, life }: SectionControlProps): ReactNode {
  const view = snapshot.consolidation
  const busy = pending !== null
  const kinds = view === undefined ? [] : Object.entries(view.journal.byKind)
  return (
    <section className={css.section} aria-label={t('sectionConsolidation')}>
      <h3 className={css.sectionTitle}>{t('sectionConsolidation')}</h3>
      {view === undefined
        ? <p className={css.muted}>{t('consolidationUnavailable')}</p>
        : (
          <>
            <div className={css.facts}>
              <Fact label={t('journalLabel')}>{view.journal.entries}</Fact>
              <Fact label={t('journalKindsLabel')}>
                {kinds.length === 0 ? t('noReadings') : kinds.map(([kind, count]) => `${kind}: ${String(count)}`).join(' · ')}
              </Fact>
              <Fact label={t('passesLabel')}>{view.passes}</Fact>
              <Fact label={t('lastPassLabel')}>
                {view.lastPassAt > 0 ? formatInstant(new Date(view.lastPassAt * 1000).toISOString()) : t('notYet')}
              </Fact>
              <Fact label={t('projectedDigestsLabel')}>{view.projectedDigests}</Fact>
              <Fact label={t('lastCorpusLabel')}>{view.lastCorpus !== '' ? view.lastCorpus : t('notYet')}</Fact>
            </div>
            {view.running && <p className={css.trainingBadges}><Tag tone="warning">{t('passRunning')}</Tag></p>}
            <h4 className={css.organTitle}>{t('specTitle')}</h4>
            {view.spec === null
              ? <p className={css.muted}>{t('specNotReady')}</p>
              : (
                <div className={css.facts}>
                  <Fact label={t('gateReasonLabel')}>{view.spec.reason}</Fact>
                  <Fact label={t('datasetsLabel')}>
                    {view.spec.datasets.length === 0 ? t('notYet') : view.spec.datasets.join(', ')}
                  </Fact>
                </div>
              )}
            <h4 className={css.organTitle}>{t('reportTitle')}</h4>
            {view.lastReport === null
              ? <p className={css.muted}>{t('noReport')}</p>
              : (
                <>
                  <div className={css.facts}>
                    <Fact label={t('triggerLabel')}>{view.lastReport.reason}</Fact>
                    <Fact label={t('gateReasonLabel')}>{view.lastReport.specReason !== '' ? view.lastReport.specReason : t('notYet')}</Fact>
                    <Fact label={t('durationLabel')}>{t('secondsShort', { count: (view.lastReport.durationMs / 1000).toFixed(1) })}</Fact>
                  </div>
                  <LineList title={t('weaknessesTitle')} lines={view.lastReport.weaknesses} empty={t('noWeaknesses')} />
                  <LineList title={t('notesTitle')} lines={view.lastReport.notes} empty={t('noNotes')} />
                </>
              )}
            <div className={css.actions} role="group" aria-label={t('sectionConsolidation')}>
              <Button disabled={busy} onClick={() => { run('consolidate', () => life.consolidate()) }}>{t('runConsolidate')}</Button>
            </div>
          </>
        )}
    </section>
  )
}

/** Section 6: the host projection — health, model, seed activity, memory, capabilities, auth. */
function HostSection({ t, snapshot }: { t: LifePanelProps['t']; snapshot: LifeSnapshot }): ReactNode {
  const { health, memory, workbench, auth } = snapshot
  if (health === undefined && memory === undefined && workbench === undefined && auth === undefined) {
    return <p className={css.muted}>{t('noReading')}</p>
  }
  const authText = auth === undefined
    ? null
    : !auth.enabled
      ? t('authDisabled')
      : auth.authenticated && auth.tokenValid
        ? t('authOk')
        : t('authFailed')
  return (
    <section className={css.section} aria-label={t('sectionHost')}>
      <h3 className={css.sectionTitle}>{t('sectionHost')}</h3>
      <div className={css.facts}>
        {health !== undefined && (
          <>
            <Fact label={t('healthLabel')}>
              {health.state}
              {!health.startupComplete && <Tag tone="warning">{t('startupIncomplete')}</Tag>}
            </Fact>
            <Fact label={t('modelLabel')}>{health.modelName !== '' ? health.modelName : t('modelNone')}</Fact>
            <Fact label={t('seedLabel')}>
              <StateDot state={health.seedActive ? 'done' : 'idle'} />
              {health.seedActive ? t('seedActive') : t('seedInactive')}
            </Fact>
          </>
        )}
        {workbench !== undefined && (
          <Fact label={t('workbenchLabel')}>
            {workbench.status === 'ok'
              ? t('workbenchCount', { count: String(workbench.count), revision: String(workbench.revision) })
              : <><Tag tone="warning">{workbench.status}</Tag> {workbench.error}</>}
            <span className={css.muted}> {workbench.source}{workbench.owner === '' ? '' : ` · ${workbench.owner}`}</span>
          </Fact>
        )}
        {auth !== undefined && authText !== null && (
          <Fact label={t('authLabel')}>
            <StateDot state={!auth.enabled ? 'idle' : auth.authenticated && auth.tokenValid ? 'done' : 'error'} />
            {authText}
          </Fact>
        )}
        {memory !== undefined && (
          <Fact label={t('memoryLabel')}>
            {t('memoryUsed', { pct: memory.usedPct.toFixed(0), available: memory.availableGb.toFixed(1), total: memory.totalGb.toFixed(1) })}
          </Fact>
        )}
      </div>
    </section>
  )
}
