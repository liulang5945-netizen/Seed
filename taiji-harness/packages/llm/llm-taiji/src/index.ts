/** Register the Taiji local runtime as a harness LLM provider route. */
import type {} from '@taiji/dsh-settings'
import type {} from '@taiji/cordis-plugin-loader'
import type { Context } from '@taiji/cordis'
import type { AdapterRegistrationHandle } from '@taiji/dsh-llm'
import { TaijiAdapter } from './adapter.ts'
import { Config, plainOptions, resolveAdapterOptions, resolveReadinessPollMs } from './config.ts'
import type { ResolvedTaijiOptions } from './config.ts'
import { RUNTIME_DISPLAY_NAME } from './defaults.ts'
import { isRoutable, probeReadiness } from './health.ts'

export { Config, plainOptions, resolveAdapterOptions, resolveReadinessPollMs } from './config.ts'
export type { Options, ResolvedTaijiOptions } from './config.ts'
export { TaijiAdapter } from './adapter.ts'
export type { TaijiAdapterOptions, TaijiCatalogModel, TaijiConnectionOptions } from './types.ts'
export { CHAT_STREAM_PATH, DEFAULT_BASE_URL, DEFAULT_MODELS, HEALTH_PATH, RUNTIME_DISPLAY_NAME } from './defaults.ts'
export { isRoutable, probeReadiness } from './health.ts'
export type { TaijiReadiness } from './health.ts'
export { buildChatRequest } from './chat.ts'
export type { TaijiChatRequest } from './chat.ts'

export const name = 'llm-taiji'
export const inject = ['llm']

const NS = 'llm-taiji'
const PROVIDER = 'taiji-local'

/**
 * Mount the Taiji runtime route.
 *
 * The route is registered through the harness LLM registry — the same
 * mechanism every adapter uses — and its membership is what makes the provider
 * routable. `GET /api/health` decides that membership: a runtime reporting its
 * own failure, or answering with no health verdict, is withdrawn from the
 * registry, while a runtime that is merely still loading stays registered so
 * the request it receives records the real outcome. Membership is re-probed
 * every `readinessPollMs` for as long as the plugin lives, so a runtime that
 * becomes reachable after load is routed without restarting the harness.
 * @param ctx - host context carrying the LLM service.
 * @param config - plugin config resolved once per operation.
 */
export async function apply(ctx: Context, config: Config): Promise<void> {
  ctx.inject(['settings'], (child) => { child.effect(() => child.settings.configure({ auto: false }, ctx.fiber)) })
  const options = (): ResolvedTaijiOptions => resolveAdapterOptions(plainOptions(config))
  // Fail loud at load: an unusable endpoint is a composition error, not a
  // request-time surprise.
  options()

  const adapter = new TaijiAdapter({ options })
  ctx.llm.registerConfigurableProviders([
    { provider: PROVIDER, displayName: RUNTIME_DISPLAY_NAME, settingsNs: ctx.fiber.entry?.options.id ?? NS, settingsPath: [] },
  ])

  let registration: AdapterRegistrationHandle | undefined
  const refreshRoute = async (): Promise<void> => {
    let connection: ResolvedTaijiOptions
    try {
      connection = options()
    } catch (error) {
      // A stored configuration the resolver refuses keeps the current route;
      // each request fails on its own resolve.
      ctx.logger.warn(error)
      return
    }
    const routable = isRoutable(await probeReadiness(connection.baseURL))
    if (registration === undefined) {
      if (!routable) return
      registration = ctx.llm.registerAdapter([PROVIDER], adapter)
      return
    }
    // The empty set is the withdrawn posture: the registration stays, so the
    // provider can return without observing a gap between two registrations.
    registration.replace(routable ? [PROVIDER] : [])
  }
  await refreshRoute()

  // One refresh at a time. Two concurrent probes would both find no
  // registration, and the second `registerAdapter` throws DUPLICATE_ADAPTER.
  let idle: Promise<void> = Promise.resolve()
  const scheduleRefresh = (): Promise<void> => {
    idle = idle.then(() => refreshRoute()).catch((error: unknown) => { ctx.logger.warn(error) })
    return idle
  }
  ctx.on('loader/volatile-update', () => { void scheduleRefresh() })

  // Readiness is the runtime's own process state: it changes with no Loader
  // update to announce it. A runtime whose cold start outlasts this plugin's
  // load is unreachable at the one probe membership used to depend on, so the
  // route is re-probed on a cadence for as long as the plugin lives.
  const pollMs = resolveReadinessPollMs(plainOptions(config))
  let timer: ReturnType<typeof setTimeout> | undefined
  let settleSleep: (() => void) | undefined
  let disposed = false
  ctx.effect(() => () => {
    disposed = true
    if (timer !== undefined) clearTimeout(timer)
    timer = undefined
    // Clearing the timer alone would leave the awaited sleep unsettled forever,
    // so the pending tick is ended here and the loop exits through its own check.
    settleSleep?.()
    settleSleep = undefined
  }, 'llm-taiji.readiness-poll')
  const poll = async (): Promise<void> => {
    while (!disposed) {
      await new Promise<void>((resolve) => {
        timer = setTimeout(resolve, pollMs)
        settleSleep = resolve
      })
      // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition -- Disposal can flip while the loop awaits the next tick.
      if (disposed) return
      await scheduleRefresh()
    }
  }
  void poll()
}
