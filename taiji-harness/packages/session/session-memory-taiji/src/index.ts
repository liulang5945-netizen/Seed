/**
 * Reports each eligible agent turn's durable prompt/answer pair to the Taiji
 * local runtime's memory journal over `POST /api/memory/record`. The report is
 * a best-effort outbound side effect: it never delays or fails a turn, it
 * writes nothing to the model request, and the runtime's response is only
 * inspected for transport success.
 *
 * @module @taiji/dsh-session-memory-taiji
 */

import { createHash } from 'node:crypto'
import { basename } from 'node:path'
import type { Context } from '@taiji/cordis'
import z from '@taiji/schemastery'
import type {} from '@taiji/dsh-agent'
import type { Session } from '@taiji/dsh-session'

/** Cordis plugin name used by loader diagnostics. */
export const name = 'session-memory-taiji'

/**
 * Required Host services. The plugin reads the stopped turn from the event
 * payload and the session that payload carries, so it declares no dependency.
 */
export const inject: string[] = []

/** Report endpoint, sampling budget, and per-report timeout. Invalid values fail plugin load. */
export interface Config {
  /** Runtime base URL; trailing slashes are stripped. Default `http://127.0.0.1:8000`. */
  baseURL?: string
  /** Report no turn when false. Default true. */
  enabled?: boolean
  /** Character budget applied independently to the prompt and the answer. Default 2000. */
  maxTextChars?: number
  /** Milliseconds one report may take before it is abandoned. Default 5000. */
  timeoutMs?: number
}

/** Schemastery validation for {@link Config}. */
export const Config: z<Config> = z.object({
  baseURL: z.string(),
  enabled: z.boolean(),
  maxTextChars: z.natural(),
  timeoutMs: z.natural(),
})

/** One report body for `POST /api/memory/record`. */
interface MemoryRecord {
  readonly kind: 'interaction'
  readonly text: string
  readonly session_id: string
  readonly turn: number
  readonly tags: string[]
  readonly importance: number
  readonly source: 'taiji-harness'
  readonly metadata: Record<string, unknown>
}

/** One derived message in the session surface. */
type SurfaceMessage = ReturnType<Session['deriveMessages']>[number]

/** The durable facts this plugin reads from one stopped turn. */
interface TurnFacts {
  /** Visible text of the turn's last user-role message. */
  readonly prompt: string
  /** Visible text of the assistant-role messages after it. */
  readonly answer: string
  /** Names of the tools the turn's assistant messages actually called, de-duplicated. */
  readonly tools: readonly string[]
}

/** Default runtime base URL. */
const DEFAULT_BASE_URL = 'http://127.0.0.1:8000'
/** Default per-report timeout. */
const DEFAULT_TIMEOUT_MS = 5_000
/** Default per-side character budget. */
const DEFAULT_MAX_TEXT_CHARS = 2_000
/** Memory-record endpoint resolved against the configured base URL. */
const RECORD_PATH = '/api/memory/record'
/** Turns remembered per session, so a re-entrant stop never re-POSTs an old turn. */
const REMEMBERED_TURNS = 64

/** The visible text of one message: its text blocks joined by newlines. */
function visibleText(message: SurfaceMessage): string {
  const parts: string[] = []
  for (const block of message.content) if (block.type === 'text') parts.push(block.text)
  return parts.join('\n')
}

/**
 * Whether the session is an ordinary top-level one. Subagent sessions and
 * delegated children report nothing, the same eligibility the workspace
 * change recorder uses. A session without a working directory stays eligible:
 * the directory only supplies an optional tag here.
 */
function eligible(session: Session): boolean {
  const { origin, delegationDepth } = session.header
  return origin !== 'subagent' && (delegationDepth ?? 0) === 0
}

/**
 * Read the stopped turn's prompt, answer, and called tools from the session
 * surface. The last user-role message is the prompt; every assistant message
 * after it contributes its text blocks to the answer and its tool-call block
 * names to the tool list. A turn with neither prompt nor answer text is
 * unreportable.
 */
function readTurn(session: Session): TurnFacts | undefined {
  const messages = session.deriveMessages()
  const lastUserIndex = messages.findLastIndex(message => message.role === 'user')
  const userMessage = lastUserIndex < 0 ? undefined : messages[lastUserIndex]
  const prompt = userMessage === undefined ? '' : visibleText(userMessage)
  const answerParts: string[] = []
  const tools: string[] = []
  for (let index = lastUserIndex + 1; index < messages.length; index += 1) {
    const message = messages[index]
    if (message === undefined || message.role !== 'assistant') continue
    for (const block of message.content) {
      if (block.type === 'text') answerParts.push(block.text)
      else if (block.type === 'tool-call' && !tools.includes(block.name)) tools.push(block.name)
    }
  }
  const answer = answerParts.join('\n')
  if (prompt.length === 0 && answer.length === 0) return undefined
  return { prompt, answer, tools }
}

/** Clip text to the configured character budget. */
function clip(value: string, maxChars: number): string {
  return value.length > maxChars ? value.slice(0, maxChars) : value
}

/**
 * The importance heuristic over what this plugin observed: a durable base, a
 * tool-using turn, a turn whose answer is empty (a failed or proxyless turn),
 * and an aborted turn each raise it. The result is clamped to `0..1`.
 */
function importanceOf(facts: TurnFacts, aborted: boolean): number {
  const raw = 0.3
    + (facts.tools.length > 0 ? 0.25 : 0)
    + (facts.answer.length === 0 ? 0.2 : 0)
    + (aborted ? 0.25 : 0)
  return Math.min(1, Math.max(0, raw))
}

/**
 * The turn's tags: the routed provider and model when the session exposes a
 * request context, the workspace directory's basename when one is known, and
 * the number of tools called.
 */
function tagsOf(session: Session, facts: TurnFacts): string[] {
  const tags: string[] = []
  const context = session.requestContext()
  if (context !== undefined) tags.push(`provider:${context.provider}`, `model:${context.model}`)
  const { cwd } = session.header
  if (cwd !== undefined && cwd.length > 0) tags.push(`workspace:${basename(cwd)}`)
  tags.push(`tools:${String(facts.tools.length)}`)
  return tags
}

/** The turn's observed facts, keyed only by what the session can actually report. */
function metadataOf(session: Session, facts: TurnFacts, aborted: boolean): Record<string, unknown> {
  const metadata: Record<string, unknown> = {}
  const context = session.requestContext()
  if (context !== undefined) {
    metadata['provider'] = context.provider
    metadata['model'] = context.model
  }
  metadata['tools'] = [...facts.tools]
  metadata['toolCount'] = facts.tools.length
  metadata['answerChars'] = facts.answer.length
  metadata['promptChars'] = facts.prompt.length
  metadata['aborted'] = aborted
  return metadata
}

/** Compose the `问：/答：` dialogue record for one stopped turn. */
function buildRecord(session: Session, turn: number, facts: TurnFacts, aborted: boolean, maxTextChars: number): MemoryRecord {
  return {
    kind: 'interaction',
    text: `问：${clip(facts.prompt, maxTextChars)}\n答：${clip(facts.answer, maxTextChars)}`,
    session_id: String(session.id),
    turn,
    tags: tagsOf(session, facts),
    importance: importanceOf(facts, aborted),
    source: 'taiji-harness',
    metadata: metadataOf(session, facts, aborted),
  }
}

/** POST one record, resolving only on a successful response. */
async function postRecord(baseURL: string, record: MemoryRecord, timeoutMs: number): Promise<void> {
  const response = await fetch(`${baseURL}${RECORD_PATH}`, {
    method: 'POST',
    redirect: 'error',
    headers: { 'content-type': 'application/json', accept: 'application/json' },
    body: JSON.stringify(record),
    signal: AbortSignal.timeout(timeoutMs),
  })
  if (!response.ok) {
    throw new Error(`session-memory-taiji: memory record rejected with HTTP ${String(response.status)}`)
  }
  // Drain the small body so the connection is released.
  await response.arrayBuffer()
}

/**
 * Report every ordinary top-level agent turn's prompt/answer pair to the
 * runtime's memory journal for the lifetime of `ctx`.
 *
 * Session-title and compaction auxiliary calls are deliberately not reported:
 * they never reach `agent/turn-stopping`, which is this plugin's only seam.
 * Nothing here concerns training or the life organs; other packages own those.
 * @param ctx - plugin context; the listeners are disposed with it.
 * @param config - endpoint, budget, and timeout. Impossible values fail plugin load.
 * @throws when the budget or timeout is not a positive safe integer.
 */
export function apply(ctx: Context, config: Config): void {
  if (config.enabled === false) return
  const baseURL = (config.baseURL ?? DEFAULT_BASE_URL).replace(/\/+$/u, '')
  const maxTextChars = config.maxTextChars ?? DEFAULT_MAX_TEXT_CHARS
  const timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS
  if (baseURL.length === 0) {
    throw new TypeError(`session-memory-taiji: baseURL must be a non-empty string, got ${JSON.stringify(config.baseURL)}`)
  }
  if (!Number.isSafeInteger(maxTextChars) || maxTextChars < 1) {
    throw new TypeError(`session-memory-taiji: maxTextChars must be a safe integer of at least 1, got ${String(maxTextChars)}`)
  }
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1) {
    throw new TypeError(`session-memory-taiji: timeoutMs must be a safe integer of at least 1, got ${String(timeoutMs)}`)
  }

  // Per-session memory of the (turn → text digest) pair already reported, so a
  // re-entrant or duplicated `agent/turn-stopping` never re-POSTs one turn.
  // Bounded to the most recent turns; dropped when the session is disposed.
  const reported = new Map<Session, Map<number, string>>()
  // One debug line per process: a runtime that is down must not log per turn.
  let failureLogged = false

  ctx.on('agent/turn-stopping', ({ agent, turn, signal }) => {
    const { session } = agent
    if (!eligible(session)) return
    const facts = readTurn(session)
    if (facts === undefined) return
    const digest = createHash('sha256').update(`${facts.prompt}\u0000${facts.answer}`).digest('hex')
    let turns = reported.get(session)
    if (turns === undefined) {
      turns = new Map()
      reported.set(session, turns)
    }
    if (turns.get(turn) === digest) return
    turns.set(turn, digest)
    while (turns.size > REMEMBERED_TURNS) {
      const oldest = turns.keys().next().value
      if (oldest === undefined) break
      turns.delete(oldest)
    }

    const record = buildRecord(session, turn, facts, signal.aborted, maxTextChars)
    // Scheduled, never awaited: the turn must not wait on, or fail with, this report.
    void (async () => {
      try {
        await postRecord(baseURL, record, timeoutMs)
      } catch (error: unknown) {
        if (failureLogged) return
        failureLogged = true
        ctx.logger.debug(`session-memory-taiji: memory report failed (${String(error)}); later failures stay silent`)
      }
    })()
  })

  ctx.on('session/disposed', (session) => { reported.delete(session) })
}
