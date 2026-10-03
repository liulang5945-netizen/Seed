/** Built dsh web + the `pnpm run dev:web --no-serve` watchers → browser HMR, with no page reload. */

import { existsSync, globSync, statSync } from 'node:fs'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { chromium } from 'playwright'
import { expect, it } from 'vitest'
import { Context } from '@taiji/cordis'
import type { Fiber } from '@taiji/cordis'
import LocalSubprocessRuntime from '@taiji/dsh-subprocess-local'
import type { SubprocessHandle, SubprocessSpawnSpec } from '@taiji/dsh-subprocess'
import { readClientBuildRecord } from '../../../scripts/client-build-environment.ts'
import { newEnglishPage, REPO_ROOT } from './support.ts'

const CLIENT_ARTIFACT_PATTERNS = [
  'apps/web/dist/**/*',
  'packages/*/*/lib/client.js',
  'packages/*/*/lib/client.js.map',
  'packages/*/*/lib/client.*.js',
  'packages/*/*/lib/client.*.js.map',
]

/** Return every artifact that `pnpm run dev:web` can rewrite. */
function clientArtifactPaths(): string[] {
  return globSync(CLIENT_ARTIFACT_PATTERNS, { cwd: REPO_ROOT })
    .map(path => join(REPO_ROOT, path))
    .filter(path => statSync(path).isFile())
    .sort()
}

function spawnSpec(argv: readonly string[], cwd: string, env?: Record<string, string>): SubprocessSpawnSpec {
  return {
    argv,
    cwd,
    stdio: { stdin: 'ignore', stdout: 'pipe', stderr: 'pipe' },
    graceMs: 5_000,
    ...env === undefined ? {} : { env },
  }
}

function waitForOutput(child: SubprocessHandle, pattern: RegExp, label: string): Promise<string> {
  return new Promise((resolveReady, reject) => {
    let output = ''
    let settled = false
    const cleanup = (): void => {
      clearTimeout(timer)
      child.stdout?.off('data', onData)
      child.stderr?.off('data', onData)
    }
    const resolveOnce = (value: string): void => {
      if (settled) return
      settled = true
      cleanup()
      resolveReady(value)
    }
    const rejectOnce = (error: Error): void => {
      if (settled) return
      settled = true
      cleanup()
      reject(error)
    }
    const onData = (chunk: Buffer): void => {
      output += chunk.toString()
      const match = pattern.exec(output)
      if (match === null) return
      resolveOnce(match[1] ?? match[0])
    }
    const timer = setTimeout(() => { rejectOnce(new Error(`${label} not ready:\n${output}`)) }, 60_000)
    child.stdout?.on('data', onData)
    child.stderr?.on('data', onData)
    void child.done.then((outcome) => {
      rejectOnce(new Error(`${label} exited before ready (${JSON.stringify(outcome)}):\n${output}`))
    }, (error: unknown) => {
      rejectOnce(new Error(`${label} failed before ready:\n${output}`, { cause: error }))
    })
  })
}

async function stopTree(child: SubprocessHandle): Promise<void> {
  child.terminate()
  const stopped = await child.waitForExit(AbortSignal.timeout(15_000))
  if (!stopped) throw new Error('managed process range did not stop after termination escalation')
  await child.done
}

it('hot-reloads a real client-plugin source edit without refreshing the page', async () => {
  const world = await mkdtemp(join(tmpdir(), 'dsh-web-hmr-world-'))
  const sourcePath = join(REPO_ROOT, 'packages/client/ui-conversation/src/client/locales.ts')
  const binPath = join(REPO_ROOT, 'apps/cli/lib/bin.js')
  if (!existsSync(binPath)) throw new Error('HMR browser test needs the built dsh bin; run pnpm run build first')
  const clientBuildEnvironment = readClientBuildRecord(REPO_ROOT).environment
  const originalClientArtifacts = await Promise.all(clientArtifactPaths()
    .map(async path => [path, await readFile(path)] as const))
  const originalClientArtifactPaths = new Set(originalClientArtifacts.map(([path]) => path))
  const originalSource = await readFile(sourcePath)
  const oldText = 'State at Its Utmost'
  const sourceNeedle = "'hero.headline': 'State at Its Utmost'"
  const newText = `HMR UPDATED ${'x'.repeat(80)}`
  // The owner-ruled brand closure made the hero headline English in
  // BOTH locale entries, so the needle appears twice; replacing only the first
  // would edit the zh entry while the en page renders the untouched one.
  const updatedSource = originalSource.toString().replaceAll(sourceNeedle, `'hero.headline': '${newText}'`)
  if (updatedSource === originalSource.toString()) throw new Error(`HMR source lacks ${JSON.stringify(sourceNeedle)}`)

  const subprocessCtx = new Context()
  let subprocessFiber: Fiber | undefined
  let watcher: SubprocessHandle | undefined
  let host: SubprocessHandle | undefined
  let browser: Awaited<ReturnType<typeof chromium.launch>> | undefined
  const failures: unknown[] = []
  try {
    subprocessFiber = await subprocessCtx.plugin(LocalSubprocessRuntime)
    // Watchers only: the built `dsh web` below is the server under test, and the
    // built tree is this lane's precondition rather than something to rebuild.
    // The watcher is started through the tsx entry that the `dev:web` alias runs, not
    // through the package manager: a `.cmd` shim cannot be spawned without a shell on
    // Windows, and going through `pnpm run` would also let its implicit install rewrite
    // the tracked workspace lockfile while the face is running.
    watcher = subprocessCtx.subprocess.spawn(spawnSpec(
      [process.execPath, join(REPO_ROOT, 'node_modules', 'tsx', 'dist', 'cli.mjs'), 'scripts/dev-web.ts', '--poll', '--skip-build', '--no-serve'],
      REPO_ROOT,
      { ...clientBuildEnvironment },
    ))
    await waitForOutput(watcher, /dev-web: watching/, 'tsx scripts/dev-web.ts --poll --skip-build --no-serve')
    host = subprocessCtx.subprocess.spawn(spawnSpec(
      [process.execPath, binPath, 'web', '--no-open', '--port', '0'],
      world,
      {
        DEEPSEEK_API_KEY: 'keyless-hmr-no-call',
        DSH_HOME: join(world, '.dsh'),
      },
    ))
    const baseUrl = await waitForOutput(host, /dsh web: (http:\/\/[^\s]+)/, 'built dsh web')
    browser = await chromium.launch()
    // The source needle is the English headline; a host-locale page would
    // render the zh entry (态之极境) and the wait below would time out.
    const page = await newEnglishPage(browser)
    const pageErrors: string[] = []
    page.on('pageerror', error => pageErrors.push(String(error)))
    await page.goto(baseUrl, { waitUntil: 'load' })
    await page.getByText(oldText, { exact: true }).waitFor({ timeout: 15_000 })
    const pageIdentity = await page.evaluate(() => {
      // In-page code: an import would not survive serialization, and the page
      // entropy source available in every context is getRandomValues.
      const identity = Array.from(crypto.getRandomValues(new Uint8Array(8)), byte => byte.toString(16).padStart(2, '0')).join('')
      Object.defineProperty(window, '__dshHmrPageIdentity', { value: identity })
      return identity
    })

    await writeFile(sourcePath, updatedSource)
    // The watcher rebuild runs alongside the rest of a face batch; 30s left no
    // headroom under that load (the ㊵-153 closure face red), so the hot-update
    // wait gets the same slow-runner treatment as the turn-tail barrier.
    await page.getByText(newText, { exact: true }).waitFor({ timeout: 60_000 })
    expect(await page.evaluate(() => (window as Window & { __dshHmrPageIdentity?: string }).__dshHmrPageIdentity))
      .toBe(pageIdentity)
    expect(pageErrors).toEqual([])
  } catch (error) {
    failures.push(error)
  } finally {
    await writeFile(sourcePath, originalSource).catch((error: unknown) => failures.push(error))
    if (watcher !== undefined) await stopTree(watcher).catch((error: unknown) => failures.push(error))
    if (host !== undefined) await stopTree(host).catch((error: unknown) => failures.push(error))
    await browser?.close().catch((error: unknown) => failures.push(error))
    await subprocessFiber?.dispose().catch((error: unknown) => failures.push(error))
    await Promise.all(clientArtifactPaths()
      .filter(path => !originalClientArtifactPaths.has(path))
      .map(async (path) => { await rm(path, { force: true }) }))
      .catch((error: unknown) => failures.push(error))
    await Promise.all(originalClientArtifacts.map(async ([path, content]) => {
      await writeFile(path, content)
    })).catch((error: unknown) => failures.push(error))
    try {
      readClientBuildRecord(REPO_ROOT)
    } catch (error) {
      failures.push(error)
    }
    await rm(world, { recursive: true, force: true }).catch((error: unknown) => failures.push(error))
  }
  if (failures.length > 0) throw new AggregateError(failures, 'HMR browser test or cleanup failed')
}, 180_000)
