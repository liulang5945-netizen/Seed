/**
 * Readiness of the Taiji runtime, read from its public `GET /api/health`
 * endpoint. The endpoint carries no credential and answers from process state
 * alone, so it is the only fact this adapter can obtain before a request.
 *
 * The verdict is deliberately conservative: only a runtime that reports its
 * own failure, or that answers with something other than a health verdict, is
 * treated as unable to serve. A runtime still loading is *not* — routing it
 * keeps the route visible and lets the request that follows record the real
 * outcome, which is what the combined failure is for.
 */

import { attributionHeaders } from '@taiji/dsh-llm'
import { HEALTH_PATH, HEALTH_PROBE_TIMEOUT_MS } from './defaults.ts'

/** Stripped trailing-slash root, so one configured spelling yields one request path. */
function root(baseURL: string): string {
  return baseURL.replace(/\/+$/u, '')
}

/** One readiness probe's provider-neutral outcome. */
export type TaijiReadiness =
  /** The runtime serves requests. */
  | 'ok'
  /** The runtime process is up but its model is still loading: not served yet, not failed. */
  | 'loading'
  /** The runtime is still downloading its payload: not served yet, not failed. */
  | 'downloading'
  /** The runtime reported its own startup failure. */
  | 'error'
  /** The probe could not reach the runtime, or it answered with no health verdict. */
  | 'unreachable'
  /** The runtime answered with a health payload this adapter does not recognize. */
  | 'unknown'

/**
 * Whether a route whose runtime reported `readiness` can serve a request.
 * `error` and `unreachable` are the runtime's own verdicts that it cannot;
 * `loading`, `downloading`, and `unknown` are absence of evidence, which keeps
 * the route registered so the request's own failure is what gets recorded.
 * @param readiness - one probe outcome.
 * @returns true when the route stays registered on the harness LLM service.
 */
export function isRoutable(readiness: TaijiReadiness): boolean {
  return readiness !== 'error' && readiness !== 'unreachable'
}

/**
 * Probe the runtime's readiness once.
 * @param baseURL - configured runtime root.
 * @returns the readiness verdict; transport failure and an unusable reply both
 *   report `unreachable`, because neither confirms a health verdict.
 */
export async function probeReadiness(baseURL: string): Promise<TaijiReadiness> {
  try {
    const response = await fetch(`${root(baseURL)}${HEALTH_PATH}`, {
      method: 'GET',
      redirect: 'error',
      signal: AbortSignal.timeout(HEALTH_PROBE_TIMEOUT_MS),
      headers: { ...attributionHeaders(), 'accept': 'application/json' },
    })
    if (!response.ok) return 'unreachable'
    const payload: unknown = await response.json()
    const status = typeof payload === 'object' && payload !== null
      ? (payload as { status?: unknown }).status
      : undefined
    if (status === 'ok' || status === 'loading' || status === 'downloading' || status === 'error') return status
    return 'unknown'
  } catch (_unreachableRuntime) {
    return 'unreachable'
  }
}
