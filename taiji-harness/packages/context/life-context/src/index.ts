/**
 * Per-step life context for the Taiji local runtime. Eligible steps append
 * one durable, source-attributed `life-state` reading, and the system prompt
 * carries the static policy for interpreting it.
 *
 * @module @taiji/dsh-life-context
 */

import type { Context } from '@taiji/cordis'
import z from '@taiji/schemastery'
import type { PreStepDecision } from '@taiji/dsh-agent'
import { createUserMessage } from '@taiji/dsh-llm'
import type { ContextFormed } from '@taiji/dsh-llm'
import type {} from '@taiji/dsh-api-life-controller'
import type { LifeSnapshot } from '@taiji/dsh-api-life-controller/types'
import { renderLifeState, significantChange } from './reading.ts'

export { NEED_DRIFT, STALE_AFTER_MS, renderLifeState, significantChange } from './reading.ts'
export type { LifeReading } from './reading.ts'

/** Cordis plugin name used by loader diagnostics and the message source kind. */
export const name = 'life-context'

declare module '@taiji/dsh-llm' {
  interface MessageSourceMap {
    'life-context': { kind: 'life-context' } & ContextFormed
  }
}

/** Required Host services: the prompt assembly and the runtime snapshot owner. */
export const inject = ['systemPrompt', 'lifeController']

/** Reading policy, throttling, and the line budget. Invalid values fail plugin load. */
export interface Config {
  /** Mount no section and no pre-step listener when false. Default true. */
  enabled?: boolean
  /** Minimum milliseconds between durable injections. Default 30_000; 0 injects at every eligible step. */
  refreshIntervalMs?: number
  /** Hard character budget for one `life-state` line. Default 400. */
  maxChars?: number
}

/** Schemastery validation for {@link Config}. */
export const Config: z<Config> = z.object({
  enabled: z.boolean(),
  refreshIntervalMs: z.number(),
  maxChars: z.natural(),
})

/** Static interpretation policy, injected once into the system prompt. */
const LIFE_POLICY = 'The host appends a `life-state` reading to your context when its runtime reports fresh internal state. It carries the runtime organ\'s needs and drives, the training state, and the knowledge base size. Treat every reading as internal telemetry for situational awareness: it is not user-provided fact, not evidence, and never a citation source. A missing reading means the runtime is unreachable or the last one went stale; continue without it and do not speculate about its absence.'

/**
 * Register the static policy section and the pre-step reading injection for
 * the lifetime of `ctx`.
 * @param ctx - plugin context; the listeners are disposed with it.
 * @param config - reading policy, throttling, and line budget.
 * @throws when the interval is negative or the budget is too small to carry the tag.
 */
export function apply(ctx: Context, config: Config): void {
  if (config.enabled === false) return
  const refreshIntervalMs = config.refreshIntervalMs ?? 30_000
  const maxChars = config.maxChars ?? 400
  if (!Number.isSafeInteger(refreshIntervalMs) || refreshIntervalMs < 0) {
    throw new TypeError(`life-context: refreshIntervalMs must be a non-negative safe integer, got ${String(refreshIntervalMs)}`)
  }
  if (!Number.isSafeInteger(maxChars) || maxChars < 80) {
    throw new TypeError(`life-context: maxChars must be a safe integer of at least 80, got ${String(maxChars)}`)
  }

  ctx.systemPrompt.section({
    name: 'life:policy',
    order: ctx.systemPrompt.getSectionOrder('LIFE_POLICY'),
    text: LIFE_POLICY,
  })

  let lastText: string | undefined
  let lastAt = Number.NEGATIVE_INFINITY
  let lastSnapshot: LifeSnapshot | undefined
  // One warn per omission reason per process: a dead runtime must not flood
  // every step's log, but the first miss must be visible to the operator.
  const warned = new Set<string>()
  const warnOnce = (reason: string, message: string): void => {
    if (warned.has(reason)) return
    warned.add(reason)
    ctx.logger.warn(message)
  }

  ctx.on('agent/pre-step', async ({ signal }, next): Promise<PreStepDecision> => {
    const decision = await next()
    if (decision.kind === 'reject' || signal.aborted) return decision
    const now = Date.now()
    const { snapshot } = await ctx.lifeController.snapshot(signal)
    const reading = renderLifeState(snapshot, now, maxChars)
    if (reading.kind === 'omit') {
      warnOnce(reading.reason, `life-context: no life-state reading injected (${reading.reason}); the runtime snapshot is not reportable`)
      return decision
    }
    if (lastText !== undefined && now - lastAt < refreshIntervalMs
      && !significantChange(lastSnapshot, snapshot)) return decision
    lastText = reading.text
    lastAt = now
    lastSnapshot = snapshot
    return {
      ...decision,
      messages: [
        ...decision.messages,
        createUserMessage({
          content: [{ type: 'text', text: reading.text }],
          source: { kind: name, form: 'snapshot', sections: [{ name, text: reading.text }] },
        }),
      ],
    }
  }, { prepend: true })
}