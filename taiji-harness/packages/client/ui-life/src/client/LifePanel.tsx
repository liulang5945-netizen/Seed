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

import { useCallback, useEffect, useId, useRef, useSyncExternalStore, useState, type ChangeEvent, type ReactNode } from 'react'
import type { ILife, LifeSnapshotState } from '@taiji/dsh-api-life-controller/client'
import type {
  LifeCheckpointView,
  LifeDatasetView,
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
type PendingVerb = 'lifeStart' | 'lifeStop' | 'feed' | 'sleep' | 'play' | 'trainStart' | 'trainResumeCheckpoint' | 'trainPause' | 'trainResume' | 'trainStop' | 'trainReset' | 'consolidate' | 'activateCheckpoint' | 'uploadDataset' | 'deleteDataset' | 'deleteCheckpoint' | 'uploadKnowledge' | 'deleteKnowledge'

/** The verbs a confirming second click protects. */
const CONFIRMED: ReadonlySet<PendingVerb> = new Set(['trainStop', 'trainReset', 'activateCheckpoint', 'deleteDataset', 'deleteCheckpoint', 'deleteKnowledge'])

/** Suffixes the runtime's dataset roster trains on; the picker also filters by them. */
const DATASET_ACCEPT = '.jsonl,.ndjson,.json,.txt,.text,.md,.csv'

/** Document suffixes the runtime's parser reads; other names fall back to plain text. */
const KNOWLEDGE_ACCEPT = '.txt,.md,.json,.jsonl,.log,.yaml,.yml,.py,.js,.ts,.java,.c,.cpp,.h,.hpp,.sh,.bat,.ps1,.sql,.go,.rs,.swift,.csv,.pdf,.docx,.doc,.html,.htm,.epub,.xlsx,.xls,.pptx,.rtf,.xml,.png,.jpg,.jpeg,.webp,.bmp'

/** Bytes one upload may carry; the Host enforces the same budget before forwarding. */
const UPLOAD_MAX_BYTES = 200 * 1024 * 1024

/**
 * Read one picked file as canonical base64. The Client-to-Host channel is a
 * JSON string body, so the bytes travel encoded — the same shape attachment
 * admission uses for browser-picked files.
 */
async function readBase64(file: File): Promise<string> {
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => { reject(reader.error ?? new Error('the file could not be read')) }
    reader.onload = () => { resolve(typeof reader.result === 'string' ? reader.result : '') }
    reader.readAsDataURL(file)
  })
  const comma = dataUrl.indexOf(',')
  return comma === -1 ? '' : dataUrl.slice(comma + 1)
}

/** Basename of one roster path or picked name, for matching an upload against the roster. */
function basenameOf(path: string): string {
  return path.split(/[\\/]/u).pop() ?? path
}

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

/** Resolve the panel copy for one Host failure, from the failure's own typed details. */
function errorText(rpc: RemoteFailure, t: LifePanelProps['t']): string {
  switch (rpc.code) {
    case 'life/runtime-error':
      return t('errRuntimeError', { status: String(rpc.details.status), detail: rpc.details.detail })
    case 'life/runtime-unreachable':
      return t('errUnreachable', { reason: rpc.details.reason })
    case 'life/unavailable':
      return t('errUnavailable', { reason: rpc.details.reason })
    case 'life/conflict':
      return t('errConflict', { reason: rpc.details.reason })
    case 'life/bad-request':
      return t('errBadRequest', { field: rpc.details.field, reason: rpc.details.reason })
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
    // L1: two section outlines preview the cards that are on their way. The
    // outlines are real (empty, aria-hidden) nodes because a host can only
    // carry one ::before and one ::after — a nested pseudo would be dropped
    // silently and leave two empty frames. The copy stays the <p>'s only text
    // node, so the loading query still resolves to exactly one element.
    return (
      <div className={css.page}>
        <p className={css.loading} role="status">
          {t('loading')}
          <span className={css.loadingCard} aria-hidden="true" />
          <span className={css.loadingCardWide} aria-hidden="true" />
        </p>
      </div>
    )
  }

  if (state.state === 'error' || state.snapshot === undefined) {
    // L2: one state panel owns the centring, so the layout no longer depends
    // on the error line and the retry being adjacent siblings of the page.
    return (
      <div className={css.page}>
        <div className={css.statePanel}>
          <p className={css.errorLine} role="alert">{t('errorTitle')}</p>
          <Button
            disabled={refreshing}
            aria-busy={refreshing}
            onClick={() => {
              setRefreshing(true)
              void life.refresh().catch(() => {}).finally(() => { setRefreshing(false) })
            }}
          >
            {t('retry')}
          </Button>
        </div>
      </div>
    )
  }

  const snapshot = state.snapshot
  const shared = { t, snapshot, pending, confirming, run, life }

  return (
    <div className={css.page}>
      <div className={css.content}>
        <header className={css.header}>
          <h2 className={css.title}>{t('title')}</h2>
          <p className={css.subtitle}>{t('subtitle')}</p>
        </header>

        <SourceSection t={t} snapshot={snapshot} />
        <LifeSection {...shared} />
        <TrainingSection {...shared} />
        <KnowledgeSection {...shared} />
        <ConsolidationSection {...shared} />
        <HostSection t={t} snapshot={snapshot} />

        {failureText !== null && <p className={css.errorLine} role="alert">{failureText}</p>}
      </div>
    </div>
  )
}

/** Shared props of the control-carrying sections. */
interface SectionControlProps {
  readonly t: LifePanelProps['t']
  readonly snapshot: LifeSnapshot
  readonly pending: PendingVerb | null
  /** The verb awaiting its confirming second click, when one is armed. */
  readonly confirming: PendingVerb | null
  readonly run: (verb: PendingVerb, invoke: () => Promise<unknown>) => void
  readonly life: ILife
}

/**
 * Roster selection preselected from the data ring's spec: the spec's dataset
 * list applies itself whenever that list changes (including the first poll
 * that brings the roster in), while a manual choice survives later roster
 * refreshes — the poll replaces the array identity, never the user's ticks.
 * One upload also ticks its row once, when the refreshed roster first shows it.
 * @param specDatasets - dataset paths the spec names, absent without a spec.
 * @param roster - the trainable roster, absent when the read did not answer.
 * @param autoSelect - basename of the dataset an upload just landed, or null.
 * @returns the current selection and the toggle for one roster entry.
 */
function useDatasetSelection(
  specDatasets: readonly string[] | undefined,
  roster: readonly { path: string }[] | undefined,
  autoSelect: string | null,
): { selected: ReadonlySet<string>; toggle: (path: string) => void; remove: (paths: readonly string[]) => void } {
  const specKey = specDatasets?.join('\n') ?? null
  const [selected, setSelected] = useState<ReadonlySet<string>>(() => new Set())
  const [appliedKey, setAppliedKey] = useState<string | null>(null)
  const [appliedAuto, setAppliedAuto] = useState<string | null>(null)
  useEffect(() => {
    if (specKey === null || roster === undefined || appliedKey === specKey) return
    const paths = new Set(roster.map(entry => entry.path))
    setSelected(new Set(specKey.split('\n').filter(path => path !== '' && paths.has(path))))
    setAppliedKey(specKey)
  }, [specKey, roster, appliedKey])
  useEffect(() => {
    if (autoSelect === null || appliedAuto === autoSelect || roster === undefined) return
    const match = roster.find(entry => basenameOf(entry.path) === autoSelect)
    if (match === undefined) return
    setSelected(current => (current.has(match.path) ? current : new Set(current).add(match.path)))
    setAppliedAuto(autoSelect)
  }, [autoSelect, appliedAuto, roster])
  /** Tick one roster entry without disturbing the rest of the selection. */
  const toggle = (path: string): void => {
    setSelected((current) => {
      const next = new Set(current)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }
  /** Drop entries that no longer exist after a delete. */
  const remove = (paths: readonly string[]): void => {
    setSelected((current) => {
      const next = new Set(current)
      for (const path of paths) next.delete(path)
      return next
    })
  }
  return { selected, toggle, remove }
}

/**
 * Group the roster by its directory so one long flat list becomes a handful of
 * foldable blocks; root-level files form their own group under an empty name.
 * @param datasets - the roster the runtime listed.
 * @returns the groups in first-seen directory order.
 */
function groupDatasets(datasets: readonly LifeDatasetView[]): { dir: string; entries: LifeDatasetView[] }[] {
  const groups = new Map<string, LifeDatasetView[]>()
  for (const entry of datasets) {
    const cut = entry.path.lastIndexOf('/')
    const dir = cut === -1 ? '' : entry.path.slice(0, cut)
    const bucket = groups.get(dir)
    if (bucket === undefined) groups.set(dir, [entry])
    else bucket.push(entry)
  }
  return [...groups].map(([dir, entries]) => ({ dir, entries }))
}

/**
 * One labeled fact row, as a term/definition pair. The wrapping `<div>` keeps
 * label and value in one grid cell; HTML lets `<div>` group `<dt>`/`<dd>`
 * inside a `<dl>` without breaking the list.
 */
function Fact({ label, children }: { label: string; children: ReactNode }): ReactNode {
  return (
    <div className={css.fact}>
      <dt className={css.factLabel}>{label}</dt>
      <dd className={css.factValue}>{children}</dd>
    </div>
  )
}

/** One horizontal 0..100 meter, announced as a 0..100 quantity. */
function Meter({ label, value }: { label: string; value: number }): ReactNode {
  return (
    <div className={css.meter}>
      <span className={css.meterLabel}>{label}</span>
      <span
        className={css.meterTrack}
        role="meter"
        aria-label={label}
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={100}
      >
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
  const headingId = useId()
  const badge = snapshot.availability.runtime === 'down'
    ? { key: 'downBadge' as LifeLocaleKey, dot: 'error' as StateDotState }
    : snapshot.fresh
      ? { key: 'freshBadge' as LifeLocaleKey, dot: 'done' as StateDotState }
      : { key: 'staleBadge' as LifeLocaleKey, dot: 'warning' as StateDotState }
  return (
    // Folding lives on the native <details>: no JS state, and the summary's
    // accessible name is the heading's own copy — no second string to keep.
    <details className={css.section} id="life-source" aria-labelledby={headingId} open>
      <summary className={css.sectionSummary}>
        <h3 className={css.sectionTitle} id={headingId}>{t('sectionSource')}</h3>
      </summary>
      <dl className={css.facts}>
        <Fact label={t('sourceLabel')}>
          {t(sourceKeyOf(snapshot))}
          <StateDot state={badge.dot} /> <span className={css.badgeText}>{t(badge.key)}</span>
        </Fact>
        <Fact label={t('observedAtLabel')}>{formatInstant(snapshot.observedAt)}</Fact>
        <Fact label={t('pollLabel')}>{t('secondsShort', { count: String(snapshot.pollIntervalMs / 1000) })}</Fact>
      </dl>
      {snapshot.unavailable.length > 0 && (
        <div className={css.unavailable}>
          <p className={css.muted}>{t('unavailableTitle')}</p>
          <ul className={css.unavailableList}>
            {snapshot.unavailable.map(line => <li key={line}>{line}</li>)}
          </ul>
        </div>
      )}
    </details>
  )
}

/** The native organ block: mode, tick, and the open need and drive meters. */
function NativeOrgan({ t, native }: { t: LifePanelProps['t']; native: LifeNativeView }): ReactNode {
  return (
    <div className={css.organ}>
      <h4 className={css.organTitle}>{t('organNative')}</h4>
      <dl className={css.facts}>
        <Fact label={t('modeLabel')}>{native.mode}</Fact>
        <Fact label={t('tickLabel')}>{native.tick}</Fact>
      </dl>
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
      <dl className={css.facts}>
        <Fact label={t('lifeStateLabel')}>{legacy.isRunning ? legacy.lifeState : t('schedulerStopped')}</Fact>
        <Fact label={t('dominantLabel')}>{legacy.dominantNeed}</Fact>
        <Fact label={t('heartbeatsLabel')}>{legacy.totalHeartbeats}</Fact>
        <Fact label={t('eventsLabel')}>{legacy.totalEvents}</Fact>
        {legacy.lastHeartbeat !== undefined && <Fact label={t('lastHeartbeatLabel')}>{formatInstant(legacy.lastHeartbeat)}</Fact>}
        {legacy.lastActivity !== undefined && <Fact label={t('lastActivityLabel')}>{formatInstant(legacy.lastActivity)}</Fact>}
      </dl>
      <p className={css.groupLabel}>{t('needsLabel')}</p>
      <MeterMap t={t} values={legacy.needs} />
    </div>
  )
}

/** Section 2: the organs' readings plus the legacy scheduler controls. */
function LifeSection({ t, snapshot, pending, run, life }: SectionControlProps): ReactNode {
  const headingId = useId()
  const lifeView = snapshot.life
  const busy = pending !== null
  return (
    <details className={css.section} id="life-readings" aria-labelledby={headingId} open>
      <summary className={css.sectionSummary}>
        <h3 className={css.sectionTitle} id={headingId}>{t('sectionLife')}</h3>
      </summary>
      {lifeView === undefined
        ? <p className={css.muted}>{t('noReading')}</p>
        : (
          <>
            {lifeView.native !== undefined && <NativeOrgan t={t} native={lifeView.native} />}
            {lifeView.legacy !== undefined && <LegacyOrgan t={t} legacy={lifeView.legacy} />}
          </>
        )}
      {/* The five controls name themselves; a group label only repeated the
          heading the region already carries. */}
      <div className={css.actions}>
        <Button disabled={busy} aria-busy={pending === 'lifeStart'} onClick={() =>{  run('lifeStart', () => life.lifeStart()) }}>{t('lifeStart')}</Button>
        <Button className={css.dangerAction} disabled={busy} aria-busy={pending === 'lifeStop'} onClick={() =>{  run('lifeStop', () => life.lifeStop()) }}>{t('lifeStop')}</Button>
        <Button disabled={busy} aria-busy={pending === 'feed'} onClick={() =>{  run('feed', () => life.lifeAction({ action: 'feed' })) }}>{t('actionFeed')}</Button>
        <Button disabled={busy} aria-busy={pending === 'sleep'} onClick={() =>{  run('sleep', () => life.lifeAction({ action: 'sleep' })) }}>{t('actionSleep')}</Button>
        <Button disabled={busy} aria-busy={pending === 'play'} onClick={() =>{  run('play', () => life.lifeAction({ action: 'play' })) }}>{t('actionPlay')}</Button>
      </div>
    </details>
  )
}

/** The checkpoint roster table: selection, resume, and activation per row. */
function Checkpoints({ t, checkpoints, artifacts, busy, selected, onToggleSelect, onActivate, onResume, labelId }: {
  t: LifePanelProps['t']
  checkpoints: readonly LifeCheckpointView[]
  artifacts: LifeSnapshot['artifacts']
  busy: boolean
  selected: ReadonlySet<string>
  onToggleSelect: (filename: string) => void
  onActivate: (filename: string) => void
  onResume: (filename: string) => void
  /** Id of the heading that names this scrollable roster. */
  labelId: string
}): ReactNode {
  if (checkpoints.length === 0) return <p className={css.muted}>{t('checkpointsEmpty')}</p>
  return (
    // The wrapper owns the horizontal scroll so the table keeps display:table
    // and every cell role; it must be focusable or a keyboard-only operator
    // can never reach the columns past the fold.
    <div className={css.tableScroll} role="group" aria-labelledby={labelId} tabIndex={0}>
      <table className={css.table}>
        <thead>
          <tr>
            <th scope="col">{t('colSelect')}</th>
            <th scope="col">{t('colName')}</th>
            <th scope="col">{t('colStep')}</th>
            <th scope="col">{t('colSize')}</th>
            <th scope="col">{t('colSaved')}</th>
            <th scope="col">{t('colResume')}</th>
          </tr>
        </thead>
        <tbody>
          {checkpoints.map((cp) => {
            // The two checkpoints the runtime refuses to delete stay unticked by
            // construction: the answering model and the next start both live on
            // one of them.
            const locked = artifacts?.activeId === cp.filename || artifacts?.configuredId === cp.filename
            return (
              <tr key={cp.filename}>
                <td>
                  <input
                    type="checkbox"
                    checked={selected.has(cp.filename)}
                    disabled={busy || locked}
                    aria-label={`${t('colSelect')} ${cp.filename}`}
                    title={locked ? t('checkpointLocked') : undefined}
                    onChange={() => { onToggleSelect(cp.filename) }}
                  />
                </td>
                {/* The filename heads the row, so the row's checkbox and
                    controls read as "for seed_beta.pt" rather than dangling. */}
                <th scope="row">
                  {cp.filename}
                  {artifacts?.activeId === cp.filename && <Tag tone="solid">{t('artifactsActiveBadge')}</Tag>}
                  {artifacts !== undefined && artifacts.configuredId === cp.filename && artifacts.configuredId !== artifacts.activeId && (
                    <Tag tone="info">{t('artifactsConfiguredBadge')}</Tag>
                  )}
                </th>
                <td>{cp.step}</td>
                <td>{formatBytes(cp.bytes)}</td>
                <td>{cp.savedAtUtc !== '' ? formatInstant(cp.savedAtUtc) : formatInstant(cp.modifiedUtc)}</td>
                <td>
                  <div className={css.actions}>
                    {/* Six rows once offered six identically named buttons; the
                        filename suffix makes each one addressable. */}
                    <Button
                      disabled={busy}
                      aria-label={`${t('resumeFrom')} ${cp.filename}`}
                      onClick={() => { onResume(cp.filename) }}
                    >
                      {t('resumeFrom')}
                    </Button>
                    <Button
                      disabled={busy || artifacts?.activeId === cp.filename}
                      aria-label={`${t('activateRow')} ${cp.filename}`}
                      onClick={() => { onActivate(cp.filename) }}
                    >
                      {t('activateRow')}
                    </Button>
                  </div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
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
function Progress({ t, progress, labelId }: { t: LifePanelProps['t']; progress: LifeProgressView; labelId: string }): ReactNode {
  return (
    <div className={css.progress}>
      {/* Named by the section heading already on the page, so the bar reads as
          "Training 25%" without any new copy. */}
      <span
        className={css.progressTrack}
        role="progressbar"
        aria-labelledby={labelId}
        aria-valuenow={Math.round(progress.fraction * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
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

/** Section 3: training state badges, progress, controls, and the rosters. */
function TrainingSection({ t, snapshot, pending, confirming, run, life }: SectionControlProps): ReactNode {
  const headingId = useId()
  const runStateId = useId()
  const runControlId = useId()
  const datasetsId = useId()
  const checkpointsId = useId()
  const datasetImpactId = useId()
  const checkpointImpactId = useId()
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
  const [picked, setPicked] = useState<File | null>(null)
  const [uploadedName, setUploadedName] = useState<string | null>(null)
  const [uploadMessage, setUploadMessage] = useState<string | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [datasetAck, setDatasetAck] = useState<string | null>(null)
  const [foldedGroups, setFoldedGroups] = useState<ReadonlySet<string>>(() => new Set())
  const [checkpointsFolded, setCheckpointsFolded] = useState(false)
  const [checkpointSelection, setCheckpointSelection] = useState<ReadonlySet<string>>(() => new Set())
  const [checkpointAck, setCheckpointAck] = useState<string | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  const { selected, toggle, remove } = useDatasetSelection(specDatasets, datasets, uploadedName)

  /** Remember the picked file, refusing one the Host would only reject after the transfer. */
  const pickFile = (event: ChangeEvent<HTMLInputElement>): void => {
    const file = event.target.files?.[0] ?? null
    // Clearing the value lets the same file be picked again after a refusal.
    event.target.value = ''
    setUploadMessage(null)
    setUploadError(null)
    if (file === null) return
    if (file.size > UPLOAD_MAX_BYTES) {
      setPicked(null)
      setUploadError(t('uploadTooLarge', { limit: `${String(UPLOAD_MAX_BYTES / (1024 * 1024))} MB` }))
      return
    }
    setPicked(file)
  }

  /** Send the picked file, then read the roster again so its new row appears. */
  const uploadFile = (): void => {
    const file = picked
    if (file === null) return
    setUploadError(null)
    run('uploadDataset', async () => {
      const value = await life.uploadDataset({ name: file.name, data: await readBase64(file) })
      setUploadMessage(value.message)
      setPicked(null)
      setUploadedName(basenameOf(file.name))
      await life.refresh().catch(() => undefined)
    })
  }

  /** Fold or unfold one dataset directory block. */
  const toggleGroup = (dir: string): void => {
    setFoldedGroups((current) => {
      const next = new Set(current)
      if (next.has(dir)) next.delete(dir)
      else next.add(dir)
      return next
    })
  }

  /**
   * Delete every ticked dataset, one runtime call per file so a refusal names
   * the file it stopped on. The rows that really went drop from the selection
   * and the roster is read again; a partial delete keeps the rest ticked.
   */
  const deleteDatasets = async (): Promise<void> => {
    const targets = [...selected]
    let removed = 0
    setDatasetAck(null)
    try {
      for (const path of targets) {
        await life.deleteDataset({ path })
        removed += 1
      }
    } finally {
      if (removed > 0) {
        remove(targets.slice(0, removed))
        setDatasetAck(t('deleteDone', { count: String(removed) }))
        await life.refresh().catch(() => undefined)
      }
    }
  }

  /** Tick one checkpoint row for deletion. */
  const toggleCheckpoint = (filename: string): void => {
    setCheckpointSelection((current) => {
      const next = new Set(current)
      if (next.has(filename)) next.delete(filename)
      else next.add(filename)
      return next
    })
  }

  /** Delete every ticked checkpoint, one runtime call per name, pruning what went. */
  const deleteCheckpoints = async (): Promise<void> => {
    const targets = [...checkpointSelection]
    let removed = 0
    setCheckpointAck(null)
    try {
      for (const filename of targets) {
        await life.deleteCheckpoint({ filename })
        removed += 1
      }
    } finally {
      if (removed > 0) {
        setCheckpointSelection((current) => {
          const next = new Set(current)
          for (const filename of targets.slice(0, removed)) next.delete(filename)
          return next
        })
        setCheckpointAck(t('deleteDone', { count: String(removed) }))
        await life.refresh().catch(() => undefined)
      }
    }
  }

  return (
    <>
      <details className={css.section} id="life-training" aria-labelledby={headingId} open>
        <summary className={css.sectionSummary}>
          <h3 className={css.sectionTitle} id={headingId}>{t('sectionTraining')}</h3>
        </summary>
        {/* A · run state — the highest-frequency read: "what is it doing now".
          The heading names the group, so a screen reader reaches the badges and
          the bar as one unit instead of three unrelated widgets. */}
        <div className={css.organ} role="group" aria-labelledby={runStateId}>
          <h4 className={css.organTitle} id={runStateId}>{t('runStateTitle')}</h4>
          <dl className={css.facts}>
            <Fact label={t('streamLabel')}>
              <StateDot state={snapshot.availability.trainingStream === 'streaming' ? 'ongoing' : snapshot.availability.trainingStream === 'closed' ? 'error' : 'idle'} />
              {t(streamKey)}
            </Fact>
          </dl>
          <p className={css.trainingBadges}>
            <Tag tone={active ? 'solid' : 'neutral'}>{active ? t('trainingRunning') : t('trainingIdle')}</Tag>
            {training.pauseRequested && <Tag tone="warning">{t('trainingPauseRequested')}</Tag>}
            {training.stopRequested && <Tag tone="warning">{t('trainingStopRequested')}</Tag>}
            {training.publishing && <Tag tone="info">{t('trainingPublishing')}</Tag>}
          </p>
          {training.progress !== undefined
            ? <Progress t={t} progress={training.progress} labelId={headingId} />
            : <p className={css.muted}>{t('noProgress')}</p>}
          {(training.warnings ?? []).map(message => (
            <p key={message} className={css.trainingBadges}>
              <Tag tone="warning">{t('runWarning')}</Tag> <span className={css.badgeText}>{message}</span>
            </p>
          ))}
        </div>
        {/* C · run control — "what to do". Moved up next to run state so the
          training block holds both short groups and the roster below it
          becomes a block of its own. */}
        <div className={css.organ} role="group" aria-labelledby={runControlId}>
          <h4 className={css.organTitle} id={runControlId}>{t('runControlTitle')}</h4>
          <div className={css.actions}>
            <Button
              variant="primary"
              disabled={busy || active}
              aria-busy={pending === 'trainStart'}
              onClick={() => { run('trainStart', () => life.trainStart(selected.size > 0 ? { datasets: [...selected] } : {})) }}
            >
              {t('trainStart')}
            </Button>
            <Button disabled={busy || !active || training.stopRequested || training.pauseRequested} aria-busy={pending === 'trainPause'} onClick={() =>{  run('trainPause', () => life.trainPause()) }}>{t('trainPause')}</Button>
            <Button disabled={busy || !active || training.stopRequested || !training.pauseRequested} aria-busy={pending === 'trainResume'} onClick={() =>{  run('trainResume', () => life.trainResume()) }}>{t('trainResume')}</Button>
            {/* The destructive pair is pushed to the far end and set off by a
              divider, so "starts on the left, destroys on the right" becomes a
              position memory rather than something to re-read every time. */}
            <div className={css.actionCluster}>
              {confirming === 'trainStop' && <span className={css.confirmLine}>{t('confirmStop')}</span>}
              <Button className={css.dangerAction} disabled={busy || !active} aria-busy={pending === 'trainStop'} onClick={() =>{  run('trainStop', () => life.trainStop()) }}>{t('trainStop')}</Button>
              {confirming === 'trainReset' && <span className={css.confirmLine}>{t('confirmReset')}</span>}
              <Button className={css.dangerAction} disabled={busy || !active} aria-busy={pending === 'trainReset'} onClick={() =>{  run('trainReset', () => life.trainReset()) }}>{t('trainReset')}</Button>
            </div>
          </div>
        </div>
      </details>
      {/* The roster is a page of its own, so it no longer shares a block with
        the controls that act on it. */}
      <details className={css.section} id="life-training-data" aria-labelledby={datasetsId} open>
        <summary className={css.sectionSummary}>
          <h3 className={css.sectionTitle} id={datasetsId}>{t('datasetsTitle')}</h3>
        </summary>
        <div className={css.organ}>
          <div className={css.uploadBlock}>
            {/* The upload button already says "Upload training file"; a group label
              with the same words only doubled the announcement. */}
            <div className={css.actions}>
              <input
                ref={fileInput}
                className={css.fileInput}
                type="file"
                accept={DATASET_ACCEPT}
                onChange={pickFile}
              />
              <Button variant="primary" disabled={busy || active} aria-busy={pending === 'uploadDataset'} onClick={() => { fileInput.current?.click() }}>{t('uploadPick')}</Button>
              {picked !== null && <Button disabled={busy || active} aria-busy={pending === 'uploadDataset'} onClick={uploadFile}>{t('uploadSend')}</Button>}
              {picked !== null && (
                <Button disabled={busy || active} onClick={() => { setPicked(null) }}>{t('uploadCancel')}</Button>
              )}
            </div>
            <p className={css.muted}>{t('uploadHint', { limit: `${String(UPLOAD_MAX_BYTES / (1024 * 1024))} MB` })}</p>
          </div>
          {picked !== null && (
            <p className={css.muted}>{t('uploadReady', { name: picked.name, size: formatBytes(picked.size) })}</p>
          )}
          {uploadError !== null && <p className={css.errorLine} role="alert">{uploadError}</p>}
          {uploadMessage !== null && <p className={css.successLine}>{t('actionDone', { message: uploadMessage })}</p>}
          {datasetAck !== null && <p className={css.successLine}>{datasetAck}</p>}
          {datasets === undefined
            ? <p className={css.unavailableNote}>{t('datasetsUnavailable')}</p>
            : datasets.length === 0
              ? <p className={css.muted}>{t('datasetsEmpty')}</p>
              : (
                <>
                  <p className={css.muted}>{t('datasetsGroupHint')}</p>
                  {groupDatasets(datasets).map(group => (
                    <div key={group.dir} className={css.datasetGroup}>
                      <button
                        type="button"
                        className={css.groupToggle}
                        aria-expanded={!foldedGroups.has(group.dir)}
                        onClick={() => { toggleGroup(group.dir) }}
                      >
                        <span className={css.groupChevron}>{foldedGroups.has(group.dir) ? '▸' : '▾'}</span>
                        <span className={css.datasetPath}>{group.dir === '' ? t('datasetRootGroup') : group.dir}</span>
                        <span className={css.datasetSize}>{t('fileCount', { count: String(group.entries.length) })}</span>
                      </button>
                      {!foldedGroups.has(group.dir) && (
                        <ul className={css.datasetList}>
                          {group.entries.map(dataset => (
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
                      )}
                    </div>
                  ))}
                  {specDatasets !== undefined && (
                    <p className={css.muted}>{t('specNames', { count: String(specDatasets.length) })}</p>
                  )}
                  {specMissing.length > 0 && (
                    <p className={css.muted}>{t('specMissing', { list: specMissing.join(', ') })}</p>
                  )}
                  {selected.size === 0 && <p className={css.muted}>{t('datasetsDefaultHint')}</p>}
                </>
              )}
          {/* The count sits beside the delete it sizes: "how many I ticked" and
            "delete them" belong in one glance. */}
          <div className={css.actions} role="group" aria-labelledby={datasetsId}>
            {confirming === 'deleteDataset' && (
              <ul className={css.impactList} id={datasetImpactId}>
                {[...selected].map(path => <li key={path}>{path}</li>)}
              </ul>
            )}
            <span className={css.muted}>{t('datasetsSelected', { count: String(selected.size) })}</span>
            <Button
              className={css.dangerAction}
              disabled={busy || active || selected.size === 0}
              aria-busy={pending === 'deleteDataset'}
              aria-describedby={confirming === 'deleteDataset' ? datasetImpactId : undefined}
              onClick={() => { run('deleteDataset', deleteDatasets) }}
            >
              {confirming === 'deleteDataset'
                ? t('confirmDeleteDatasets', { count: String(selected.size) })
                : t('deletePickedDatasets')}
            </Button>
          </div>
        </div>
      </details>
      {/* The checkpoint roster is the last of the three: it is the widest thing
        on the page, so it keeps a full block to itself. */}
      <details className={css.section} id="life-checkpoints" aria-labelledby={checkpointsId} open>
        <summary className={css.sectionSummary}>
          <h3 className={css.sectionTitle} id={checkpointsId}>{t('checkpointsTitle')}</h3>
        </summary>
        <div className={css.organ}>
          <div className={css.subHeader}>
            <Button onClick={() => { setCheckpointsFolded(current => !current) }}>
              {checkpointsFolded ? t('checkpointsUnfold', { count: String(training.checkpoints.length) }) : t('checkpointsFold')}
            </Button>
          </div>
          {!checkpointsFolded && (
            <>
              {/* Activation arms from either a row or the built-in switch, so the
                confirming line is rendered once for the whole roster rather than
                once per control that could arm it. */}
              {confirming === 'activateCheckpoint' && <p className={css.confirmLine}>{t('confirmActivate')}</p>}
              {snapshot.artifacts === undefined
                ? <p className={css.unavailableNote}>{t('artifactsUnavailable')}</p>
                : (
                  <dl className={css.facts}>
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
                        aria-busy={pending === 'activateCheckpoint'}
                        onClick={() => { run('activateCheckpoint', () => life.activateCheckpoint({ checkpointId: '' })) }}
                      >
                        {t('activateBuiltin')}
                      </Button>
                    </div>
                  </dl>
                )}
              <Checkpoints
                t={t}
                checkpoints={training.checkpoints}
                artifacts={snapshot.artifacts}
                busy={busy}
                selected={checkpointSelection}
                labelId={checkpointsId}
                onToggleSelect={toggleCheckpoint}
                onActivate={(filename) => { run('activateCheckpoint', () => life.activateCheckpoint({ checkpointId: filename })) }}
                onResume={(filename) => {
                  run('trainResumeCheckpoint', () => life.trainResumeCheckpoint({
                    checkpoint: filename,
                    ...(selected.size > 0 ? { datasets: [...selected] } : {}),
                  }))
                }}
              />
              {training.checkpoints.length > 0 && (
                <div className={css.actions} role="group" aria-labelledby={checkpointsId}>
                  {confirming === 'deleteCheckpoint' && (
                    <ul className={css.impactList} id={checkpointImpactId}>
                      {[...checkpointSelection].map(filename => <li key={filename}>{filename}</li>)}
                    </ul>
                  )}
                  <Button
                    className={css.dangerAction}
                    disabled={busy || checkpointSelection.size === 0}
                    aria-busy={pending === 'deleteCheckpoint'}
                    aria-describedby={confirming === 'deleteCheckpoint' ? checkpointImpactId : undefined}
                    onClick={() => { run('deleteCheckpoint', deleteCheckpoints) }}
                  >
                    {confirming === 'deleteCheckpoint'
                      ? t('confirmDeleteCheckpoints', { count: String(checkpointSelection.size) })
                      : t('deletePickedCheckpoints')}
                  </Button>
                </div>
              )}
              {checkpointAck !== null && <p className={css.successLine}>{checkpointAck}</p>}
            </>
          )}
        </div>
      </details>
    </>
  )
}

/** Section 4: the knowledge base size, its mounted files, and their controls. */
function KnowledgeSection({ t, snapshot, pending, confirming, run, life }: SectionControlProps): ReactNode {
  const headingId = useId()
  const filesId = useId()
  const impactId = useId()
  const knowledge = snapshot.availability.knowledge === 'ok' ? snapshot.knowledge : undefined
  const busy = pending !== null
  const [picked, setPicked] = useState<File | null>(null)
  const [note, setNote] = useState<string | null>(null)
  const [pickError, setPickError] = useState<string | null>(null)
  const [ack, setAck] = useState<string | null>(null)
  const [selected, setSelected] = useState<ReadonlySet<string>>(() => new Set())
  const fileInput = useRef<HTMLInputElement>(null)

  /** Remember the picked document, refusing one past the budget before any transfer. */
  const pickFile = (event: ChangeEvent<HTMLInputElement>): void => {
    const file = event.target.files?.[0] ?? null
    event.target.value = ''
    setNote(null)
    setPickError(null)
    if (file === null) return
    if (file.size > UPLOAD_MAX_BYTES) {
      setPicked(null)
      setPickError(t('uploadTooLarge', { limit: `${String(UPLOAD_MAX_BYTES / (1024 * 1024))} MB` }))
      return
    }
    setPicked(file)
  }

  /** Send the picked document, then read the file list again so its row appears. */
  const uploadFile = (): void => {
    const file = picked
    if (file === null) return
    setPickError(null)
    run('uploadKnowledge', async () => {
      const value = await life.uploadKnowledge({ name: file.name, data: await readBase64(file) })
      setNote(value.message)
      setPicked(null)
      await life.refresh().catch(() => undefined)
    })
  }

  /** Tick one mounted file for deletion. */
  const toggleFile = (name: string): void => {
    setSelected((current) => {
      const next = new Set(current)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }

  /** Delete every ticked knowledge file, one runtime call per name. */
  const deleteFiles = async (): Promise<void> => {
    const targets = [...selected]
    let removed = 0
    setAck(null)
    try {
      for (const name of targets) {
        await life.deleteKnowledge({ name })
        removed += 1
      }
    } finally {
      if (removed > 0) {
        setSelected((current) => {
          const next = new Set(current)
          for (const name of targets.slice(0, removed)) next.delete(name)
          return next
        })
        setAck(t('deleteDone', { count: String(removed) }))
        await life.refresh().catch(() => undefined)
      }
    }
  }

  return (
    <details className={css.section} id="life-knowledge" aria-labelledby={headingId} open>
      <summary className={css.sectionSummary}>
        <h3 className={css.sectionTitle} id={headingId}>{t('sectionKnowledge')}</h3>
      </summary>
      {knowledge === undefined
        ? <p className={css.unavailableNote}>{t('knowledgeUnavailable')}</p>
        : (
          <>
            <dl className={css.facts}>
              <Fact label={t('docsLabel')}>{knowledge.docCount}</Fact>
              <Fact label={t('chunksLabel')}>{knowledge.chunkCount}</Fact>
              <Fact label={t('embedDimLabel')}>{knowledge.embedDim > 0 ? knowledge.embedDim : t('embeddingsNo')}</Fact>
              <Fact label={t('embeddingsYes')}><StateDot state={knowledge.hasEmbeddings ? 'done' : 'idle'} /></Fact>
            </dl>
            <div className={css.uploadBlock}>
              {/* The button and a group label would both have said
                  "Upload knowledge file". */}
              <div className={css.actions}>
                <input
                  ref={fileInput}
                  className={css.fileInput}
                  type="file"
                  accept={KNOWLEDGE_ACCEPT}
                  onChange={pickFile}
                />
                <Button variant="primary" disabled={busy} aria-busy={pending === 'uploadKnowledge'} onClick={() => { fileInput.current?.click() }}>{t('knowledgeUpload')}</Button>
                {picked !== null && <Button disabled={busy} aria-busy={pending === 'uploadKnowledge'} onClick={uploadFile}>{t('uploadSend')}</Button>}
                {picked !== null && (
                  <Button disabled={busy} onClick={() => { setPicked(null) }}>{t('uploadCancel')}</Button>
                )}
              </div>
              <p className={css.muted}>{t('knowledgeUploadHint', { limit: `${String(UPLOAD_MAX_BYTES / (1024 * 1024))} MB` })}</p>
            </div>
            {picked !== null && (
              <p className={css.muted}>{t('uploadReady', { name: picked.name, size: formatBytes(picked.size) })}</p>
            )}
            {pickError !== null && <p className={css.errorLine} role="alert">{pickError}</p>}
            {note !== null && <p className={css.successLine}>{t('actionDone', { message: note })}</p>}
            {ack !== null && <p className={css.successLine}>{ack}</p>}
            <h4 className={css.organTitle} id={filesId}>{t('knowledgeFilesTitle')}</h4>
            {knowledge.files === undefined
              ? <p className={css.unavailableNote}>{t('knowledgeFilesUnavailable')}</p>
              : knowledge.files.length === 0
                ? <p className={css.muted}>{t('knowledgeEmpty')}</p>
                : (
                  <ul className={css.datasetList}>
                    {knowledge.files.map(file => (
                      <li key={file.name} className={css.datasetRow}>
                        <label>
                          <input
                            type="checkbox"
                            checked={selected.has(file.name)}
                            disabled={busy}
                            onChange={() => { toggleFile(file.name) }}
                          />
                          <span className={css.datasetPath}>{file.name}</span>
                        </label>
                        {file.status !== 'indexed' && <Tag tone="warning">{t('knowledgePending')}</Tag>}
                        {file.sizeBytes !== undefined && <span className={css.datasetSize}>{formatBytes(file.sizeBytes)}</span>}
                      </li>
                    ))}
                  </ul>
                )}
            <div className={css.actions} role="group" aria-labelledby={filesId}>
              {confirming === 'deleteKnowledge' && (
                <ul className={css.impactList} id={impactId}>
                  {[...selected].map(name => <li key={name}>{name}</li>)}
                </ul>
              )}
              <Button
                className={css.dangerAction}
                disabled={busy || selected.size === 0}
                aria-busy={pending === 'deleteKnowledge'}
                aria-describedby={confirming === 'deleteKnowledge' ? impactId : undefined}
                onClick={() => { run('deleteKnowledge', deleteFiles) }}
              >
                {confirming === 'deleteKnowledge'
                  ? t('confirmDeleteKnowledge', { count: String(selected.size) })
                  : t('deletePickedKnowledge')}
              </Button>
            </div>
          </>
        )}
    </details>
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
  const headingId = useId()
  const view = snapshot.consolidation
  const busy = pending !== null
  const kinds = view === undefined ? [] : Object.entries(view.journal.byKind)
  return (
    <details className={css.section} id="life-consolidation" aria-labelledby={headingId} open>
      <summary className={css.sectionSummary}>
        <h3 className={css.sectionTitle} id={headingId}>{t('sectionConsolidation')}</h3>
      </summary>
      {view === undefined
        ? <p className={css.unavailableNote}>{t('consolidationUnavailable')}</p>
        : (
          <>
            <dl className={css.facts}>
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
            </dl>
            {view.running && <p className={css.trainingBadges}><Tag tone="warning">{t('passRunning')}</Tag></p>}
            <h4 className={css.organTitle}>{t('specTitle')}</h4>
            {view.spec === null
              ? <p className={css.muted}>{t('specNotReady')}</p>
              : (
                <dl className={css.facts}>
                  <Fact label={t('gateReasonLabel')}>{view.spec.reason}</Fact>
                  <Fact label={t('datasetsLabel')}>
                    {view.spec.datasets.length === 0 ? t('notYet') : view.spec.datasets.join(', ')}
                  </Fact>
                </dl>
              )}
            <h4 className={css.organTitle}>{t('reportTitle')}</h4>
            {view.lastReport === null
              ? <p className={css.muted}>{t('noReport')}</p>
              : (
                <>
                  <dl className={css.facts}>
                    <Fact label={t('triggerLabel')}>{view.lastReport.reason}</Fact>
                    <Fact label={t('gateReasonLabel')}>{view.lastReport.specReason !== '' ? view.lastReport.specReason : t('notYet')}</Fact>
                    <Fact label={t('durationLabel')}>{t('secondsShort', { count: (view.lastReport.durationMs / 1000).toFixed(1) })}</Fact>
                  </dl>
                  <LineList title={t('weaknessesTitle')} lines={view.lastReport.weaknesses} empty={t('noWeaknesses')} />
                  <LineList title={t('notesTitle')} lines={view.lastReport.notes} empty={t('noNotes')} />
                </>
              )}
            <div className={css.actions} role="group" aria-labelledby={headingId}>
              <Button
                variant="primary"
                disabled={busy}
                aria-busy={pending === 'consolidate'}
                onClick={() => { run('consolidate', () => life.consolidate()) }}
              >
                {t('runConsolidate')}
              </Button>
            </div>
          </>
        )}
    </details>
  )
}

/** Section 6: the host projection — health, model, seed activity, memory, capabilities, auth. */
function HostSection({ t, snapshot }: { t: LifePanelProps['t']; snapshot: LifeSnapshot }): ReactNode {
  const headingId = useId()
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
    <details className={css.section} id="life-host" aria-labelledby={headingId} open>
      <summary className={css.sectionSummary}>
        <h3 className={css.sectionTitle} id={headingId}>{t('sectionHost')}</h3>
      </summary>
      <dl className={css.facts}>
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
      </dl>
    </details>
  )
}
