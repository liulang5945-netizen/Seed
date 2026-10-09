/** D2 P1-③: the interpreter the offline backend installer is spawned with, and the log it writes to. */
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { BACKEND_MANIFEST_NAME, DesktopPythonBackendHost, primaryRuntimePython } from '../src/python-backend-host.ts'

const ROOT = join('runtime', 'primary-runtime')

describe('primaryRuntimePython', () => {
  it('addresses the win32 interpreter as the file itself, not as a directory holding python3', () => {
    expect(primaryRuntimePython(ROOT, 'win32')).toBe(join(ROOT, 'dependencies', 'python', 'python.exe'))
  })

  it('addresses the POSIX interpreter under bin/', () => {
    for (const platform of ['linux', 'darwin'] as const) {
      expect(primaryRuntimePython(ROOT, platform)).toBe(join(ROOT, 'dependencies', 'python', 'bin', 'python3'))
    }
  })
})

describe('DesktopPythonBackendHost', () => {
  let dir: string | undefined

  afterEach(() => {
    vi.unstubAllGlobals()
    if (dir !== undefined) rmSync(dir, { recursive: true, force: true })
    dir = undefined
  })

  it('records an installer that cannot be spawned in backend.log', async () => {
    // The port probe runs before the installer, so this spec owns its answer: a
    // runtime already serving on the port would take the adoption branch.
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false })))
    dir = mkdtempSync(join(tmpdir(), 'dsh-python-backend-'))
    const backendRoot = join(dir, 'backend')
    mkdirSync(backendRoot)
    writeFileSync(join(backendRoot, BACKEND_MANIFEST_NAME), '{"source":{"digest":"staged"}}', 'utf8')
    const host = new DesktopPythonBackendHost({
      backendRoot,
      pythonExec: join(dir, 'absent-interpreter'),
      userDataDir: dir,
    })

    await expect(host.start()).rejects.toThrow()

    const log = readFileSync(host.logFile(), 'utf8')
    expect(log).toContain(`running the offline installer with ${join(dir, 'absent-interpreter')}`)
    expect(log).toContain('installer spawn failed')
  })
})
