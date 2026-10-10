/**
 * The desktop shell is a GUI-subsystem process, so a console-subsystem child it spawns
 * without `windowsHide` gets its own visible console window on the desktop. Every child
 * launch in the shipped runtime path is asserted here: the offline backend installer,
 * the Python runtime, and the desktop host.
 */
import { EventEmitter } from 'node:events'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { BACKEND_MANIFEST_NAME, DesktopPythonBackendHost } from '../src/python-backend-host.ts'
import { DesktopHostProcess } from '../src/host-process.ts'

const { spawnMock } = vi.hoisted(() => ({ spawnMock: vi.fn() }))
vi.mock('node:child_process', async importOriginal => ({
  ...await importOriginal<typeof import('node:child_process')>(),
  spawn: spawnMock,
}))

interface SpawnCall {
  readonly command: string
  readonly args: string[]
  readonly options: Record<string, unknown>
}

const roots: string[] = []
const calls: SpawnCall[] = []

/** A `child_process.ChildProcess` stand-in, so a launch settles without starting a process. */
function fakeChild(): EventEmitter {
  return Object.assign(new EventEmitter(), {
    pid: 4242,
    stdout: new PassThrough(),
    stderr: new PassThrough(),
    stdin: new PassThrough(),
    connected: true,
    exitCode: null,
    kill: vi.fn(() => true),
    send: vi.fn((_message: unknown, callback?: (error: Error | null) => void) => { callback?.(null) }),
    disconnect: vi.fn(),
    unref: vi.fn(),
  })
}

function tempRoot(): string {
  const root = mkdtempSync(join(tmpdir(), 'dsh-console-visibility-'))
  roots.push(root)
  return root
}

afterEach(() => {
  spawnMock.mockReset()
  calls.splice(0, calls.length)
  vi.unstubAllGlobals()
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

describe('desktop child console visibility', () => {
  it('launches the offline installer and the Python runtime without a console window', async () => {
    spawnMock.mockImplementation((command: string, args: string[], options: Record<string, unknown>) => {
      const child = fakeChild()
      calls.push({ command, args, options })
      queueMicrotask(() => { child.emit('exit', 0, null) })
      return child
    })
    // The pre-install probe must report nothing worth adopting, or the install branch is
    // skipped; the health polls after the runtime spawns answer successfully so `start()`
    // settles instead of running to its deadline.
    let probes = 0
    vi.stubGlobal('fetch', vi.fn(async () => (++probes === 1
      ? { ok: false }
      : { ok: true, json: async () => ({ service: 'Taiji API' }) })))
    const root = tempRoot()
    const backendRoot = join(root, 'backend')
    mkdirSync(backendRoot)
    writeFileSync(join(backendRoot, BACKEND_MANIFEST_NAME), '{"source":{"digest":"staged"}}', 'utf8')
    const host = new DesktopPythonBackendHost({
      backendRoot,
      pythonExec: join(root, 'python'),
      userDataDir: root,
    })

    await host.start()

    expect(calls.map(call => call.args[0])).toEqual(['-I', '-m'])
    for (const call of calls) expect(call.options).toMatchObject({ windowsHide: true })
  })

  it('launches the desktop host without a console window', async () => {
    const root = tempRoot()
    spawnMock.mockImplementation((command: string, args: string[], options: Record<string, unknown>) => {
      const child = fakeChild()
      calls.push({ command, args, options })
      queueMicrotask(() => {
        child.emit('message', { type: 'ready', url: 'http://127.0.0.1:1/?token=fixture' })
      })
      return child
    })
    const host = new DesktopHostProcess(join(root, 'node'), root, root)

    await host.start()

    expect(calls).toHaveLength(1)
    expect(calls[0]?.options).toMatchObject({ windowsHide: true })
  })
})
