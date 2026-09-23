/** The plugin's registration facts: its page policy on the optional settings seam. */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@taiji/cordis'
import SettingsForms from '@taiji/dsh-settings'
import LlmRuntime from '@taiji/dsh-llm'
import * as LlmTaiji from '../src/index.ts'
import { closeMockRuntimes, mockRuntime } from './mock-runtime.ts'

afterEach(async () => {
  await closeMockRuntimes()
})

describe('Taiji plugin registration', () => {
  it('registers its page policy with the mounted settings service', async () => {
    const runtime = await mockRuntime()
    const configure = vi.fn((_presentation: { auto?: boolean }, _owner: unknown) => () => {})
    const settings = Object.assign(Object.create(SettingsForms.prototype), { configure })
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    ctx.provide('settings', settings)

    await ctx.plugin(LlmTaiji, { baseURL: runtime.url })

    // This route has no page of its own on the Models surface.
    expect(configure.mock.calls[0]?.[0]).toEqual({ auto: false })
    expect(ctx.llm.listProviders()).toEqual([{ id: 'taiji-local', name: LlmTaiji.RUNTIME_DISPLAY_NAME }])
    await ctx.fiber.dispose()
  })
})