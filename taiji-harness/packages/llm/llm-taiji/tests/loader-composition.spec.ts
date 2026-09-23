/** Booting the route from a real Loader composition, as a profile does. */
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { afterEach, describe, expect, it } from 'vitest'
import { Context } from '@taiji/cordis'
import Loader from '@taiji/cordis-plugin-loader'
import Include from '@taiji/cordis-plugin-include'
import LlmRuntime, { BlockAssembler, createUserMessage } from '@taiji/dsh-llm'
import * as LlmTaiji from '../src/index.ts'
import { closeMockRuntimes, DONE, finalFrame, mockRuntime } from './mock-runtime.ts'

let root: string | undefined
let context: Context | undefined

afterEach(async () => {
  await context?.fiber.dispose()
  context = undefined
  if (root !== undefined) await rm(root, { recursive: true, force: true })
  root = undefined
  await closeMockRuntimes()
})

/** Boot the shipped row composition over a loopback runtime. */
async function loadComposition(baseURL: string): Promise<Context> {
  root = await mkdtemp(join(tmpdir(), 'dsh-llm-taiji-composition-'))
  const configPath = join(root, 'cordis.yml')
  await writeFile(configPath, [
    '- id: llm',
    "  name: '@taiji/dsh-llm'",
    '- id: llm-taiji',
    "  name: '@taiji/dsh-llm-taiji'",
    '  config:',
    `    baseURL: ${JSON.stringify(baseURL)}`,
    '',
  ].join('\n'))

  const ctx = new Context()
  context = ctx
  ctx.baseUrl = `${pathToFileURL(root).href}/`
  await ctx.plugin(Loader)
  ctx.loader.builtins.include = Include
  const modules = new Map<string, unknown>([
    ['@taiji/dsh-llm', LlmRuntime],
    ['@taiji/dsh-llm-taiji', LlmTaiji],
  ])
  // The custom importer bypasses Node resolution; mirror the package manifests
  // a deployed cordis.yml has beside its declared dependencies.
  await Promise.all([...modules.keys()].map(async (packageName) => {
    const packageDir = join(root!, 'node_modules', ...packageName.split('/'))
    await mkdir(packageDir, { recursive: true })
    await writeFile(join(packageDir, 'package.json'), `${JSON.stringify({ name: packageName, version: '0.1.0-rc.8', type: 'module' })}\n`)
  }))
  ctx.loader.internal = {
    version: 'v2',
    import: async (specifier: string) => {
      if (!modules.has(specifier)) throw new Error(`unexpected Loader import: ${specifier}`)
      return modules.get(specifier)
    },
    loadCache: new Map(),
    register(): never { throw new Error('unexpected module hook registration') },
    getOrCreateModuleJob(): never { throw new Error('unexpected module job creation') },
    resolveSync(): never { throw new Error('unexpected synchronous module resolution') },
    load(): never { throw new Error('unexpected module load') },
  }
  await ctx.loader.create({ name: 'cordis:include', config: { path: pathToFileURL(configPath).href } })
  await ctx.loader.await()
  return ctx
}

describe('llm-taiji Loader composition', () => {
  it('boots the row, registers the route, and answers through the runtime', async () => {
    const runtime = await mockRuntime()
    runtime.script.push({ kind: 'frames', frames: [finalFrame('你好，世界'), DONE] })

    const ctx = await loadComposition(runtime.url)

    expect(ctx.llm.listProviders()).toEqual([{ id: 'taiji-local', name: LlmTaiji.RUNTIME_DISPLAY_NAME }])
    // The settings namespace is the Loader entry id, which is how the Models
    // page addresses this row's section.
    expect(ctx.llm.listConfigurableProviders()).toEqual([
      { provider: 'taiji-local', displayName: LlmTaiji.RUNTIME_DISPLAY_NAME, settingsNs: 'llm-taiji', settingsPath: [] },
    ])
    const assembler = new BlockAssembler()
    const messages = [createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: '你好' }] })]
    for await (const chunk of ctx.llm.stream({ provider: 'taiji-local', model: 'taiji-local', messages })) {
      assembler.push(chunk)
    }

    expect(assembler.blocks()).toEqual([{ type: 'text', text: '你好，世界' }])
    expect(assembler.finish).toEqual({ kind: 'stop' })
    expect(runtime.requests[0]?.body).toEqual({ prompt: '你好', history: [] })
  })
})
