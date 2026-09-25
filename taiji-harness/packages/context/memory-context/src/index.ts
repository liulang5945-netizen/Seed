/**
 * Per-turn memory recall for the Taiji local runtime. On the first eligible
 * step of a turn the plugin reads the runtime's memory journal back over
 * `GET /api/memory/recall` and prepends the recalled entries to the step as
 * one durable user message. Recall is best-effort: a failure injects nothing,
 * logs once per process, and never delays or fails the step.
 *
 * @module @taiji/dsh-memory-context
 */

import type { Context } from '@taiji/cordis'
import z from '@taiji/schemastery'
import type { PreStepDecision } from '@taiji/dsh-agent'
import { createUserMessage } from '@taiji/dsh-llm'
import type { ContextFormed, UserMessage } from '@taiji/dsh-llm'
import type { Session } from '@taiji/dsh-session'
import { renderRecall } from './reading.ts'
import type { RecallEntry } from './reading.ts'

export { QUERY_ECHO_CHARS, RECALL_NOTE, TRUNCATED_MARKER, renderRecall } from './reading.ts'
export type { RecallEntry } from './reading.ts'

/** Cordis plugin name used by loader diagnostics and the injected message's source kind. */
export const name = 'memory-context'

declare module '@taiji/dsh-llm' {
  interface MessageSourceMap {
    /** Memory-recall attribution; readers preserve the content without this producer.
     * Its projection uses the kind to avoid repeated injection.
     * @persistenceAttribution
     */
    'memory-context': { kind: 'memory-context' } & ContextFormed
  }
}

/**
 * Required Host services. The plugin reads the turn from the event payload and
 * performs its own HTTP recall, so it declares no dependency.
 */
export const inject: string[] = []

/** Recall endpoint, journal budget, block budget, and per-recall timeout. Invalid values fail plugin load. */
export interface Config {
  /** Runtime base URL; trailing slashes are stripped. Default `http://127.0.0.1:8000`. */
  baseURL?: string
  /** Recall nothing when false. Default true. */
  enabled?: boolean
  /** Entries the runtime may return, best first. Default 5; `0` is invalid (a read-back that recalls nothing is the disabled case). */
  limit?: number
  /** Hard character budget for the whole rendered block. Default 600; below 80 fails plugin load. */
  maxChars?: number
  /** Milliseconds one recall may take before it is abandoned. Default 1500. */
  timeoutMs?: number
}

/** Schemastery validation for {@link Config}. */
export const Config: z<Config> = z.object({
  baseURL: z.string(),
  enabled: z.boolean(),
  limit: z.natural(),
  maxChars: z.natural(),
  timeoutMs: z.natural(),
})

/** Default runtime base URL. */
const DEFAULT_BASE_URL = 'http://127.0.0.1:8000'
/** Default entry budget. */
const DEFAULT_LIMIT = 5
/** Default block budget. */
const DEFAULT_MAX_CHARS = 600
/** Default per-recall timeout. */
const DEFAULT_TIMEOUT_MS = 1_500
/** Recall endpoint resolved against the configured base URL. */
const RECALL_PATH = '/api/memory/recall'
/** Characters of the query sent on the wire; the runtime matches its terms against the entry text. */
const QUERY_WIRE_CHARS = 200
/** Smallest block budget this plugin accepts, the same floor the sibling life-context reading uses. */
const MIN_MAX_CHARS = 80

/**
 * The visible text of one message: its text blocks joined by newlines. Other
 * blocks carry no text the runtime could match against.
 */
function visibleText(message: UserMessage): string {
  const parts: string[] = []
  for (const block of message.content) if (block.type === 'text') parts.push(block.text)
  return parts.join('\n')
}

/**
 * The turn's query: the visible text of the last message the user actually
 * sent. The step's messages are all user-role, but injected context also
 * carries a producer's own source kind, so the source kind is the test;
 * a step with no user-sourced message has nothing to recall for.
 */
function queryOf(messages: readonly UserMessage[]): string | undefined {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message === undefined || message.source.kind !== 'user') continue
    return visibleText(message)
  }
  return undefined
}

/** Whether one value is a plain object. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * Read the ranked entries out of a recall response, or report that the body is
 * not the documented one. Only the fields the block renders are read, and a
 * body that does not carry them is refused rather than rendered thin.
 */
function parseRecall(body: unknown): RecallEntry[] | undefined {
  if (!isRecord(body)) return undefined
  const raw: unknown = body['entries']
  if (!Array.isArray(raw)) return undefined
  const entries: RecallEntry[] = []
  for (const item of raw as unknown[]) {
    if (!isRecord(item)) return undefined
    const { kind, text, score } = item
    if (typeof kind !== 'string' || typeof text !== 'string' || typeof score !== 'number'
      || !Number.isFinite(score)) return undefined
    entries.push({ kind, text, score })
  }
  return entries
}

/**
 * One recall over real HTTP. Rejection means this turn injects nothing: the
 * caller treats every failure class alike, so no error escapes to the step.
 */
async function recall(baseURL: string, query: string, limit: number, timeoutMs: number): Promise<RecallEntry[]> {
  const params = new URLSearchParams({ query: query.slice(0, QUERY_WIRE_CHARS), limit: String(limit) })
  const response = await fetch(`${baseURL}${RECALL_PATH}?${params.toString()}`, {
    redirect: 'error',
    headers: { accept: 'application/json' },
    signal: AbortSignal.timeout(timeoutMs),
  })
  if (!response.ok) {
    throw new Error(`memory-context: memory recall rejected with HTTP ${String(response.status)}`)
  }
  const entries = parseRecall(await response.json())
  if (entries === undefined) {
    throw new Error('memory-context: memory recall response did not carry the documented entries')
  }
  return entries
}

/**
 * Recall the runtime's memory journal once per turn and prepend the ranked
 * entries to that turn's first eligible step for the lifetime of `ctx`.
 *
 * The listener is prepended in the `agent/pre-step` waterfall so an earlier
 * listener's replacement messages are still seen by {@link queryOf}, and it
 * awaits `next()` so the injected block rides the decision the rest of the
 * chain settled on.
 * @param ctx - plugin context; the listeners are disposed with it.
 * @param config - endpoint, entry budget, block budget, and timeout. Impossible values fail plugin load.
 * @throws when the base URL is empty or the entry budget, block budget, or timeout is not a positive safe integer.
 */
export function apply(ctx: Context, config: Config): void {
  if (config.enabled === false) return
  const baseURL = (config.baseURL ?? DEFAULT_BASE_URL).replace(/\/+$/u, '')
  const limit = config.limit ?? DEFAULT_LIMIT
  const maxChars = config.maxChars ?? DEFAULT_MAX_CHARS
  const timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS
  if (baseURL.length === 0) {
    throw new TypeError(`memory-context: baseURL must be a non-empty string, got ${JSON.stringify(config.baseURL)}`)
  }
  if (!Number.isSafeInteger(limit) || limit < 1) {
    throw new TypeError(`memory-context: limit must be a positive safe integer, got ${String(limit)}`)
  }
  if (!Number.isSafeInteger(maxChars) || maxChars < MIN_MAX_CHARS) {
    throw new TypeError(`memory-context: maxChars must be a safe integer of at least ${String(MIN_MAX_CHARS)}, got ${String(maxChars)}`)
  }
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1) {
    throw new TypeError(`memory-context: timeoutMs must be a positive safe integer, got ${String(timeoutMs)}`)
  }

  // Per-session memory of the last turn already recalled for, so no later step
  // of one turn recalls or injects a second time. Dropped on disposal.
  const served = new Map<Session, number>()
  // One warn per process: a runtime that is down must not log per turn, and
  // nothing is queued for retry.
  let failureLogged = false

  ctx.on('agent/pre-step', async ({ agent, messages, turn, signal }, next): Promise<PreStepDecision> => {
    const decision = await next()
    if (decision.kind === 'reject' || signal.aborted) return decision
    const session = agent.session
    if (served.get(session) === turn) return decision
    const query = queryOf(messages)
    if (query === undefined) return decision
    // Claim the turn before the request: the recall happens once whether it
    // succeeds, fails, or finds nothing.
    served.set(session, turn)
    let entries: readonly RecallEntry[]
    try {
      entries = await recall(baseURL, query, limit, timeoutMs)
    } catch (error: unknown) {
      if (!failureLogged) {
        failureLogged = true
        ctx.logger.warn(`memory-context: memory recall failed (${String(error)}); later failures stay silent`)
      }
      return decision
    }
    const block = renderRecall(entries, query, maxChars)
    if (block === undefined) return decision
    return {
      ...decision,
      messages: [
        createUserMessage({
          content: [{ type: 'text', text: block }],
          source: { kind: name, form: 'snapshot', sections: [{ name, text: block }] },
        }),
        ...decision.messages,
      ],
    }
  }, { prepend: true })

  ctx.on('session/disposed', (session) => { served.delete(session) })
}
