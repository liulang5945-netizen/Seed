/** Readiness verdicts, and the route membership each one produces. */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@taiji/cordis'
import type {} from '@taiji/cordis-plugin-loader'
import { createVolatile, updateVolatile } from '@taiji/cosmokit'
import LlmRuntime from '@taiji/dsh-llm'
import type { AdapterRegistrationHandle } from '@taiji/dsh-llm'
import * as LlmTaiji from '../src/index.ts'
import { Config } from '../src/config.ts'
import type { Config as PluginConfig } from '../src/config.ts'
import { isRoutable, probeReadiness } from '../src/health.ts'
import type { TaijiReadiness } from '../src/health.ts'
import { closeMockRuntimes, closedRuntimeUrl, mockRuntime } from './mock-runtime.ts'

const PROVIDER = 'taiji-local'
const NS = 'llm-taiji'

afterEach(async () => {
  await closeMockRuntimes()
})

async function harness(baseURL: string, readinessPollMs?: number) {
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(LlmTaiji, { baseURL, ...(readinessPollMs === undefined ? {} : { readinessPollMs }) })
  return ctx
}

it('ends the readiness cadence on disposal, with no tick left suspended', async () => {
  const runtime = await mockRuntime()
  const ctx = await harness(runtime.url, 250)
  await vi.waitFor(() => { expect(runtime.health.length).toBeGreaterThan(0) })

  await ctx.fiber.dispose()
  const probesAtDispose = runtime.health.length
  // Two cadence intervals of silence is the contract: the armed timer is cleared
  // and the tick it held is settled, so the poll loop returns instead of hanging.
  await new Promise<void>((resolve) => { setTimeout(resolve, 700) })
  expect(runtime.health.length).toBe(probesAtDispose)
})

/**
 * Mount the route and hand back the schema-resolved config, which is the object
 * a Loader volatile update rewrites in place.
 */
async function harnessHoldingConfig(baseURL: string): Promise<{ ctx: Context; held: PluginConfig }> {
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  let held: PluginConfig | undefined
  await ctx.plugin({
    name: 'llm-taiji-config-probe',
    inject: LlmTaiji.inject,
    Config,
    apply(inner: Context, config: PluginConfig) {
      held = config
      return LlmTaiji.apply(inner, config)
    },
  }, { baseURL })
  if (held === undefined) throw new Error('the route plugin never applied')
  return { ctx, held }
}

const registered = (ctx: Context): string[] => ctx.llm.listProviders().map(provider => provider.id)

/** Poll-driven changes are observed, not awaited by hand: generous for a loaded worker. */
const WAIT = { timeout: 5_000, interval: 25 }

describe('Taiji readiness verdicts', () => {
  it('routes every verdict that is not a runtime-reported failure', () => {
    const verdicts: TaijiReadiness[] = ['ok', 'loading', 'downloading', 'error', 'unreachable', 'unknown']
    expect(verdicts.map(verdict => [verdict, isRoutable(verdict)])).toEqual([
      ['ok', true],
      ['loading', true],
      ['downloading', true],
      ['error', false],
      ['unreachable', false],
      ['unknown', true],
    ])
  })

  it('reads each verdict the health endpoint can report', async () => {
    const runtime = await mockRuntime()
    for (const status of ['ok', 'loading', 'downloading', 'error']) {
      runtime.healthStatus = status
      expect(await probeReadiness(runtime.url)).toBe(status)
    }
    // A verdict this adapter does not know is not evidence of failure.
    runtime.healthStatus = 'restarting'
    expect(await probeReadiness(runtime.url)).toBe('unknown')
    // Neither is a readiness body that is not an object at all.
    runtime.healthBody = 'null'
    expect(await probeReadiness(runtime.url)).toBe('unknown')
    delete runtime.healthBody
    // A non-health answer is not a verdict at all.
    runtime.healthHttpStatus = 503
    expect(await probeReadiness(runtime.url)).toBe('unreachable')
    expect(runtime.health).toEqual(Array.from({ length: 7 }, () => '/api/health'))
  })

  it('reports an unreachable runtime when nothing listens on the endpoint', async () => {
    expect(await probeReadiness(await closedRuntimeUrl())).toBe('unreachable')
  })
})

describe('Taiji route membership', () => {
  it.each([
    { reported: 'ok', routable: true },
    { reported: 'loading', routable: true },
    { reported: 'downloading', routable: true },
    { reported: 'restarting', routable: true },
    { reported: 'error', routable: false },
  ])('registers the route for a runtime reporting $reported (routable: $routable)', async ({ reported, routable }) => {
    const runtime = await mockRuntime()
    runtime.healthStatus = reported
    const ctx = await harness(runtime.url)

    expect(registered(ctx)).toEqual(routable ? [PROVIDER] : [])
    // The route's absence is never the provider's absence: configuration
    // surfaces still address it through the configurable-provider directory.
    expect(ctx.llm.listConfigurableProviders()).toEqual([
      { provider: PROVIDER, displayName: LlmTaiji.RUNTIME_DISPLAY_NAME, settingsNs: NS, settingsPath: [] },
    ])
    await ctx.fiber.dispose()
  })

  it('registers no route when nothing listens on the endpoint', async () => {
    const ctx = await harness(await closedRuntimeUrl())

    expect(registered(ctx)).toEqual([])
    await expect(ctx.llm.listModels(PROVIDER)).rejects.toMatchObject({ code: 'NO_ADAPTER' })
    await ctx.fiber.dispose()
  })

  it('withdraws and restores the route as the runtime comes and goes', async () => {
    const runtime = await mockRuntime()
    const ctx = await harness(runtime.url)
    expect(registered(ctx)).toEqual([PROVIDER])

    runtime.healthStatus = 'error'
    ctx.emit('loader/volatile-update', [])
    await vi.waitFor(() => { expect(registered(ctx)).toEqual([]) })

    runtime.healthStatus = 'ok'
    ctx.emit('loader/volatile-update', [])
    await vi.waitFor(() => { expect(registered(ctx)).toEqual([PROVIDER]) })
    await ctx.fiber.dispose()
  })

  it('registers the route when a runtime that was down at load comes up', async () => {
    const runtime = await mockRuntime()
    await runtime.stop()
    const ctx = await harness(runtime.url, 250)
    expect(registered(ctx)).toEqual([])

    // The runtime finishes its own startup on its own schedule; no Loader
    // update announces it, so the poll is the only thing that can notice.
    await runtime.restart()

    await vi.waitFor(() => { expect(registered(ctx)).toEqual([PROVIDER]) }, WAIT)
    await ctx.fiber.dispose()
  })

  it('withdraws and restores the route on readiness alone, with no Loader update', async () => {
    const runtime = await mockRuntime()
    const ctx = await harness(runtime.url, 250)
    expect(registered(ctx)).toEqual([PROVIDER])

    runtime.healthStatus = 'error'
    await vi.waitFor(() => { expect(registered(ctx)).toEqual([]) }, WAIT)

    runtime.healthStatus = 'ok'
    await vi.waitFor(() => { expect(registered(ctx)).toEqual([PROVIDER]) }, WAIT)
    await ctx.fiber.dispose()
  })

  it('stops probing once the plugin fiber is disposed', async () => {
    const runtime = await mockRuntime()
    const ctx = await harness(runtime.url, 250)
    await vi.waitFor(() => { expect(registered(ctx)).toEqual([PROVIDER]) }, WAIT)

    await ctx.fiber.dispose()
    const probes = runtime.health.length
    await new Promise(resolve => setTimeout(resolve, 600))
    expect(runtime.health).toHaveLength(probes)
  })

  it('keeps the current route when the stored configuration is refused', async () => {
    const runtime = await mockRuntime()
    const { ctx, held } = await harnessHoldingConfig(runtime.url)
    expect(registered(ctx)).toEqual([PROVIDER])
    const warn = vi.spyOn(ctx.logger, 'warn').mockImplementation(() => undefined)
    const probes = runtime.health.length

    // What a Loader volatile update leaves on the running reference: the value
    // changes, and no request has validated it yet.
    updateVolatile(held.baseURL, createVolatile('file:///not-an-http-root'))
    ctx.emit('loader/volatile-update', [])
    await vi.waitFor(() => { expect(warn).toHaveBeenCalled() })

    expect(registered(ctx)).toEqual([PROVIDER])
    expect(runtime.health).toHaveLength(probes)
    await ctx.fiber.dispose()
  })
})

it('logs a refresh the harness refused on a released registration and keeps the chain running', async () => {
  const runtime = await mockRuntime()
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  // The handle is taken from the registry at registration time: the refusal below
  // is the harness's own, not a stubbed exception on the path under test.
  const original = ctx.llm.registerAdapter.bind(ctx.llm)
  const handles: AdapterRegistrationHandle[] = []
  vi.spyOn(ctx.llm, 'registerAdapter').mockImplementation((providers, adapter) => {
    const handle = original(providers, adapter)
    handles.push(handle)
    return handle
  })
  await ctx.plugin(LlmTaiji, { baseURL: runtime.url, readinessPollMs: 60_000 })
  const warn = vi.spyOn(ctx.logger, 'warn').mockImplementation(() => undefined)

  // Something else released this plugin's route, so the next membership update is
  // refused by the registry: the refresh fails on its own and is reported.
  handles[0]!()
  const probes = runtime.health.length
  ctx.emit('loader/volatile-update', [])
  await vi.waitFor(() => { expect(warn).toHaveBeenCalledOnce() })
  expect((warn.mock.calls[0]![0] as Error).message).toContain('disposed adapter registration')

  // One refused refresh does not end the chain: the next update probes again,
  // is refused again on its own terms, and is reported again.
  ctx.emit('loader/volatile-update', [])
  await vi.waitFor(() => { expect(warn).toHaveBeenCalledTimes(2) })
  expect(runtime.health.length).toBe(probes + 2)
  await ctx.fiber.dispose()
})

it('arms no readiness cadence when the fiber unloads during its cold start', async () => {
  const runtime = await mockRuntime()
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  // Unload the entry while its first readiness probe is still in flight: the route
  // is never registered and the cadence never starts, so nothing keeps the loop.
  const fiber = ctx.plugin(LlmTaiji, { baseURL: runtime.url, readinessPollMs: 250 })

  await fiber.dispose()
  await vi.waitFor(() => { expect(registered(ctx)).toEqual([]) })

  const probes = runtime.health.length
  await new Promise<void>((resolve) => { setTimeout(resolve, 700) })
  expect(runtime.health.length).toBe(probes)
  await ctx.fiber.dispose()
})
