/** Read the Host-stored locale preference over the shared settings RPC. */

import { randomUUID } from 'node:crypto'

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * Read the saved UI language from the running Host's settings service.
 *
 * This reuses the Web application's existing `/api/settings/describe` RPC —
 * the same endpoint the browser client's locale feature reads through
 * `configForms` — so the shell picks its startup language from the one durable
 * preference, without a new Host protocol and without touching accounts or
 * providers.
 * @param origin - Authenticated Host origin (`scheme://host:port`).
 * @param cookie - Host browser-session cookie issued to this process.
 * @returns the stored `locale.preference`, or null when none is set.
 */
export async function readHostLocalePreference(origin: string, cookie: string): Promise<string | null> {
  const rpcId = randomUUID()
  const response = await fetch(new URL('/api/settings/describe', origin), {
    method: 'POST',
    redirect: 'error',
    headers: { 'content-type': 'application/json', cookie },
    body: JSON.stringify({ type: 'client-request', rpcId, method: 'settings/describe', payload: { args: {} } }),
  })
  if (!response.ok) throw new Error('desktop locale: settings request failed')
  const envelope: unknown = await response.json()
  if (!record(envelope) || envelope.type !== 'server-response' || envelope.rpcId !== rpcId
    || !record(envelope.result) || envelope.result.ok !== true || !record(envelope.result.value)
    || !Array.isArray(envelope.result.value.namespaces)) {
    throw new Error('desktop locale: settings RPC failed')
  }
  const locale: unknown = envelope.result.value.namespaces.find((item: unknown) => record(item) && item.ns === 'locale')
  if (!record(locale) || !record(locale.value)) return null
  const preference = locale.value.preference
  if (preference !== undefined && typeof preference !== 'string') {
    throw new Error('desktop locale: invalid locale preference')
  }
  return preference ?? null
}
