/** Fixed endpoints and catalog defaults for the Taiji local-runtime route. */
import type { TaijiCatalogModel } from './types.ts'

/** Selector label for the route and for its single advisory model entry. */
export const RUNTIME_DISPLAY_NAME = 'Taiji（本地运行时）'

/** The Taiji runtime's local HTTP root: this route's endpoint when configuration names none. */
export const DEFAULT_BASE_URL = 'http://127.0.0.1:8000'

/** The runtime's one-shot chat stream endpoint, appended to the configured root. */
export const CHAT_STREAM_PATH = '/api/chat/stream'

/** The runtime's public, credential-free readiness endpoint, appended to the configured root. */
export const HEALTH_PATH = '/api/health'

/**
 * Longest a readiness probe waits for the local runtime. The endpoint answers
 * from process state alone — it never waits for the model — so a localhost
 * round trip far below this bound is the whole expected cost, and the bound
 * exists only to keep a wedged helper from stalling plugin load.
 */
export const HEALTH_PROBE_TIMEOUT_MS = 2_000

/**
 * Cadence of the readiness re-probe, which is what keeps route membership
 * current after load: the runtime's readiness is its own process state and
 * changes without any Loader update, so a runtime that becomes reachable later
 * — its cold start outlasts a harness boot — is only found by asking again.
 */
export const DEFAULT_READINESS_POLL_MS = 5_000

/** Floor for the re-probe cadence: one probe per health timeout is the useful limit. */
export const MIN_READINESS_POLL_MS = 250

/** Advisory catalog for the runtime's single language organ; no wire model id exists to advertise. */
export const DEFAULT_MODELS: TaijiCatalogModel[] = [
  { id: 'taiji-local', name: RUNTIME_DISPLAY_NAME },
]
