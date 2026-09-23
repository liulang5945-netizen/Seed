/** Normalize Taiji runtime HTTP failures into provider-neutral `LlmError` facts. */

import { isContextWindowExceededError, isQuotaExceededError, LlmError } from '@taiji/dsh-llm'

/**
 * Read a human-readable reason from one error body without trusting its shape.
 * The runtime answers form and validation failures with FastAPI's `detail`;
 * anything else is left to the status-derived summary.
 */
async function errorDetail(response: Response): Promise<string> {
  const text = await response.text()
  let raw: unknown
  try {
    raw = JSON.parse(text)
  } catch (_nonJsonFailureBody) {
    // A non-JSON body still carries the status, which stays authoritative.
    return ''
  }
  if (typeof raw !== 'object' || raw === null) return ''
  const fields = raw as Record<string, unknown>
  if (typeof fields.detail === 'string') return fields.detail
  return typeof fields.message === 'string' ? fields.message : ''
}

/**
 * Classify one non-2xx chat response.
 * @param response - the failed response, its body still unread.
 * @returns the stable failure consumed by `LlmRuntime` and `llm-retry`.
 */
export async function httpFailure(response: Response): Promise<LlmError> {
  const status = response.status
  const detail = await errorDetail(response)
  const message = detail.length > 0 ? detail : `Taiji runtime request failed (${status})`
  let code: string
  if (status === 401 || status === 403) code = 'AUTH'
  else if (status === 402 || isQuotaExceededError(detail)) code = 'QUOTA'
  else if (status === 429) code = 'RATE_LIMIT'
  else if (isContextWindowExceededError(detail)) code = 'CONTEXT_WINDOW_EXCEEDED'
  else if (status === 400 || status === 413 || status === 422) code = 'INVALID_REQUEST'
  else if (status >= 500) code = 'SERVER'
  else code = `HTTP_${status}`
  return new LlmError(message, code, { status })
}