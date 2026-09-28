/** Bare Vite must fail before it can present a bootless shell as a working GUI. */

import { fileURLToPath, pathToFileURL } from 'node:url'
import { join } from 'node:path'
import { existsSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { createServer } from 'node:net'
import { execa } from 'execa'
import { describe, expect, it } from 'vitest'

const WEB_ROOT = fileURLToPath(new URL('..', import.meta.url))

/** Reserve an available loopback port, then release it for the child invocation. */
async function freePort(): Promise<number> {
  const server = createServer()
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', resolve)
  })
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('port probe returned no address')
  await new Promise<void>((resolve, reject) => server.close((error) => {
    if (error === undefined) resolve()
    else reject(error)
  }))
  return address.port
}

describe('Web development entry', () => {
  it('rejects the package dev alias with the full-host correction', async () => {
    // The package manager is reached through corepack because the pnpm shim is not on
    // this machine's PATH; the config flag keeps the run from verifying (and so from
    // rewriting) the workspace lockfile as a side effect of invoking a script.
    const result = await execa('corepack', ['pnpm', 'run', 'dev'], {
      cwd: WEB_ROOT,
      reject: false,
      env: { ...process.env, npm_config_verify_deps_before_run: 'false' },
    })
    expect(result.exitCode).not.toBe(0)
    expect(result.stderr).toContain('apps/web is not a standalone application')
    expect(result.stderr).toContain('dsh web')
  })

  it('rejects the standalone Vite server with the full-host correction', async () => {
    const probeRoot = mkdtempSync(join(tmpdir(), 'dsh-vite-listen-probe-'))
    const marker = join(probeRoot, 'listen-called')
    const port = await freePort()
    try {
      const probeModule = fileURLToPath(new URL('./support/listen-probe.mjs', import.meta.url))
      const result = await execa(join(WEB_ROOT, 'node_modules/.bin/vite'), ['--host', '127.0.0.1', '--port', String(port)], {
        cwd: WEB_ROOT,
        reject: false,
        timeout: 10_000,
        env: {
          ...process.env,
          DSH_LISTEN_PROBE_MARKER: marker,
          NODE_OPTIONS: `${process.env.NODE_OPTIONS ?? ''} --import ${pathToFileURL(probeModule).href}`.trim(),
        },
      })
      expect(result.timedOut).toBe(false)
      expect(result.exitCode).not.toBe(0)
      expect(result.stderr).toContain('apps/web is not a standalone application')
      expect(result.stderr).toContain('dsh web')
      expect(result.stderr).toContain('window.__DSH_BOOT__')
      expect(existsSync(marker), 'Vite called Server.listen before rejecting standalone serve mode').toBe(false)
    } finally {
      rmSync(probeRoot, { recursive: true, force: true })
    }
  })
})
