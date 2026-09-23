/**
 * Model-visible rendering of one runtime snapshot. The format is one compact
 * single line: a reading rides a durable user message, so every character is
 * paid on every step it survives, and a panel-style multi-line report would
 * spend the context budget on decoration.
 *
 * @module @taiji/dsh-life-context/reading
 */

import type { LifeSnapshot } from '@taiji/dsh-api-life-controller/types'

/** A snapshot older than this is not injected: the model must not act on stale readings. */
export const STALE_AFTER_MS = 60_000

/** Need drift, in 0..100 points on any single dimension, that bypasses the refresh interval. */
export const NEED_DRIFT = 10

/** One rendering decision for a snapshot. */
export type LifeReading =
  | { readonly kind: 'inject'; readonly text: string }
  | { readonly kind: 'omit'; readonly reason: 'unreachable' | 'stale' }

/**
 * Render one snapshot as the single-line `life-state` reading.
 * @param snapshot - the observed runtime snapshot.
 * @param now - injection instant, for the reading's age.
 * @param maxChars - hard budget for the line; an overrun truncates and says so.
 * @returns the injectable line, or why the snapshot stays out of the context.
 */
export function renderLifeState(snapshot: LifeSnapshot, now: number, maxChars: number): LifeReading {
  if (snapshot.availability.runtime === 'down') return { kind: 'omit', reason: 'unreachable' }
  const observed = Date.parse(snapshot.observedAt)
  if (Number.isFinite(observed) && now - observed > STALE_AFTER_MS) return { kind: 'omit', reason: 'stale' }

  const segments: string[] = [ageSegment(snapshot, now)]
  const life = snapshot.life
  if (life?.native !== undefined) {
    segments.push('source=native', `tick=${format(life.native.tick)}`)
    if (life.native.mode !== '') segments.push(`mode=${life.native.mode}`)
    segments.push(mapSegment('needs', life.native.needs) ?? 'needs[]')
    const drives = mapSegment('drives', life.native.drives)
    if (drives !== undefined) segments.push(drives)
  } else if (life?.legacy !== undefined) {
    segments.push('source=legacy')
    if (life.legacy.lifeState !== '') segments.push(`state=${life.legacy.lifeState}`)
    if (life.legacy.dominantNeed !== '') segments.push(`dominant=${life.legacy.dominantNeed}`)
    segments.push(mapSegment('needs', life.legacy.needs) ?? 'needs[]')
    if (life.legacy.totalHeartbeats > 0) segments.push(`heartbeats=${format(life.legacy.totalHeartbeats)}`)
    if (life.legacy.totalEvents > 0) segments.push(`events=${format(life.legacy.totalEvents)}`)
  }
  segments.push(trainingSegment(snapshot))
  const knowledge = snapshot.knowledge
  if (knowledge !== undefined) {
    segments.push(`knowledge[${format(knowledge.docCount)} docs ${format(knowledge.chunkCount)} chunks]`)
  }

  const line = `life-state ${segments.join(' ')}`
  if (line.length <= maxChars) return { kind: 'inject', text: line }
  return { kind: 'inject', text: `${line.slice(0, Math.max(0, maxChars - ' truncated=1'.length))} truncated=1` }
}

/**
 * Whether one snapshot differs from the last injected one enough to inject
 * immediately, bypassing the refresh interval: a flipped training state, a
 * changed dominant need, or a single need drifting past {@link NEED_DRIFT}.
 * @param previous - the last injected snapshot, when there was one.
 * @param next - the snapshot just observed.
 * @returns whether the change is significant.
 */
export function significantChange(previous: LifeSnapshot | undefined, next: LifeSnapshot): boolean {
  if (previous === undefined) return true
  const a = previous.training
  const b = next.training
  if (a.isTraining !== b.isTraining || a.pauseRequested !== b.pauseRequested
    || a.stopRequested !== b.stopRequested) return true
  if ((previous.life?.legacy?.dominantNeed ?? '') !== (next.life?.legacy?.dominantNeed ?? '')) return true
  const needsA = previous.life?.native?.needs ?? previous.life?.legacy?.needs ?? {}
  const needsB = next.life?.native?.needs ?? next.life?.legacy?.needs ?? {}
  for (const key of new Set([...Object.keys(needsA), ...Object.keys(needsB)])) {
    if (Math.abs((needsB[key] ?? 0) - (needsA[key] ?? 0)) > NEED_DRIFT) return true
  }
  return false
}

function ageSegment(snapshot: LifeSnapshot, now: number): string {
  const observed = Date.parse(snapshot.observedAt)
  const ageSeconds = Number.isFinite(observed) ? Math.max(0, Math.round((now - observed) / 1000)) : 0
  return `age=${String(ageSeconds)}s`
}

function mapSegment(name: string, values: Readonly<Record<string, number>>): string | undefined {
  const entries = Object.entries(values)
  if (entries.length === 0) return undefined
  return `${name}[${entries.map(([key, value]) => `${key}=${format(value)}`).join(' ')}]`
}

function trainingSegment(snapshot: LifeSnapshot): string {
  const training = snapshot.training
  const parts = [training.isTraining ? 'running' : 'off']
  if (training.pauseRequested) parts.push('pausing')
  if (training.stopRequested) parts.push('stopping')
  if (snapshot.availability.trainingStream === 'closed') parts.push('stream-closed')
  if (training.publishing) parts.push('publishing')
  if (training.isTraining && training.progress !== undefined) {
    parts.push(`loss=${format(training.progress.loss)}`)
    parts.push(`step=${format(training.progress.step)}/${format(training.progress.totalSteps)}`)
  }
  return `training[${parts.join(' ')}]`
}

/** Render one number with at most one decimal place, deterministically. */
function format(value: number): string {
  return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(1)))
}