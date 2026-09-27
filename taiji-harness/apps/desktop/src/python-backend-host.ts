/**
 * DesktopBackendHost (D2 P1-③): own the Python backend's shipped lifecycle.
 *
 * The shipped backend tree (`resources.dsh/backend/`) carries the offline
 * wheelhouse, the installer, and a manifest.  This host turns that payload
 * into a running runtime:
 *
 * 1. `enabled()` — a structural existence gate: without a staged
 *    backend-manifest (no wheelhouse was generated) the host is off and the
 *    desktop behaves exactly as pre-D2.
 * 2. `start()` — idempotent offline install into a user-scoped venv
 *    (`install-backend.py` re-runs only when the manifest digest changes),
 *    then launches the Seed API from the venv and polls `/api/health` with
 *    capped exponential backoff.  If a healthy Taiji runtime already answers
 *    on the port (a shared development runtime), it is adopted as-is — no
 *    second process, no port fight.
 * 3. `stop()` — kills the child process tree on app quit.
 *
 * Deliberately Electron-free: everything path-shaped is injected, so the
 * lifecycle is unit-testable without an app shell.  The v1 port policy is
 * fixed-8000-with-adoption; multi-port rollover needs an `api/main.py` port
 * flag and stays a recorded P2 note (D2 design §3.3).
 */

import { spawn, type ChildProcess } from 'node:child_process'
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

export const BACKEND_MANIFEST_NAME = 'backend-manifest.json'
export const BACKEND_PORT = 8000
const HEALTH_TIMEOUT_MS = 120_000
const HEALTH_POLL_BASE_MS = 200
const HEALTH_POLL_MAX_MS = 5_000

export interface PythonBackendHostOptions {
  /** Shipped backend tree: backend-manifest.json, install-backend.py, wheelhouse/, code/. */
  backendRoot: string
  /** The primary-runtime interpreter (creates the venv and runs the installer). */
  pythonExec: string
  /** User-owned directory for the venv, endpoint file, and receipts. */
  userDataDir: string
  log?: (line: string) => void
}

export interface BackendEndpoint {
  endpoint: string
  adopted: boolean
  venv: string
}

export function backendManifestPath(backendRoot: string): string {
  return join(backendRoot, BACKEND_MANIFEST_NAME)
}

/** The primary-runtime interpreter inside a prepared payload.
 *
 * Mirrors `workspaceDependencyPaths` (packages/skill/tool-workspace-dependencies):
 * imported here would drag the package's source into the desktop tsc project,
 * so keep this one join in step with that helper when the layout moves.
 */
export function primaryRuntimePython(primaryRuntimeRoot: string): string {
  return join(primaryRuntimeRoot, 'dependencies', 'python', process.platform === 'win32' ? 'python.exe' : 'bin', 'python3')
}

export function isBackendShipped(backendRoot: string): boolean {
  return existsSync(backendManifestPath(backendRoot))
}

async function healthOk(endpoint: string, signal: AbortSignal): Promise<boolean> {
  try {
    const response = await fetch(`${endpoint}/api/health`, { signal })
    if (!response.ok) return false
    const body = (await response.json()) as { service?: string }
    return body.service === 'Taiji API'
  } catch {
    return false
  }
}

export class DesktopPythonBackendHost {
  private child: ChildProcess | undefined
  private readonly venv: string
  private readonly log: (line: string) => void

  constructor(private readonly options: PythonBackendHostOptions) {
    this.venv = join(options.userDataDir, 'backend-venv')
    this.log = options.log ?? (() => {})
  }

  enabled(): boolean {
    return isBackendShipped(this.options.backendRoot)
  }

  endpointFile(): string {
    return join(this.options.userDataDir, 'backend-endpoint.json')
  }

  async start(): Promise<BackendEndpoint> {
    const manifestPath = backendManifestPath(this.options.backendRoot)
    if (!existsSync(manifestPath)) {
      throw new Error('desktop backend: no backend manifest shipped; the channel is disabled')
    }
    const adoptedEndpoint = `http://127.0.0.1:${String(BACKEND_PORT)}`
    if (await healthOk(adoptedEndpoint, AbortSignal.timeout(2_000))) {
      this.log('desktop backend: adopting the runtime already serving on the port')
      this.writeEndpoint(adoptedEndpoint)
      return { endpoint: adoptedEndpoint, adopted: true, venv: this.venv }
    }

    await this.runInstaller(manifestPath)
    const codeRoot = join(this.options.backendRoot, 'code')
    this.child = spawn(this.venvPython(), ['-m', 'api.main'], {
      cwd: codeRoot,
      stdio: ['ignore', 'ignore', 'ignore'],
      detached: false,
    })
    this.child.once('exit', (code) => {
      this.log(`desktop backend: runtime exited (${String(code)})`)
    })

    let delay = HEALTH_POLL_BASE_MS
    const deadline = Date.now() + HEALTH_TIMEOUT_MS
    while (Date.now() < deadline) {
      if (await healthOk(adoptedEndpoint, AbortSignal.timeout(2_000))) {
        this.writeEndpoint(adoptedEndpoint)
        return { endpoint: adoptedEndpoint, adopted: false, venv: this.venv }
      }
      await new Promise(resolvePromise => setTimeout(resolvePromise, delay))
      delay = Math.min(delay * 2, HEALTH_POLL_MAX_MS)
    }
    throw new Error('desktop backend: the runtime did not become healthy in time')
  }

  stop(): void {
    this.child?.kill()
    this.child = undefined
  }

  private venvPython(): string {
    const windows = join(this.venv, 'Scripts', 'python.exe')
    return existsSync(windows) ? windows : join(this.venv, 'bin', 'python')
  }

  private writeEndpoint(endpoint: string): void {
    writeFileSync(
      this.endpointFile(),
      `${JSON.stringify({ endpoint, updated_at: Date.now() }, null, 2)}\n`,
      'utf8',
    )
  }

  private async runInstaller(manifestPath: string): Promise<void> {
    const receiptPath = join(this.options.userDataDir, 'backend-install-receipt.json')
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as { source?: { digest?: string } }
    if (existsSync(receiptPath)) {
      const receipt = JSON.parse(readFileSync(receiptPath, 'utf8')) as { snapshot?: string; venv?: string }
      if (receipt.snapshot === manifest.source?.digest && existsSync(this.venvPython())) {
        this.log('desktop backend: venv already satisfies the manifest')
        return
      }
    }
    this.log('desktop backend: running the offline installer')
    const installer = join(this.options.backendRoot, 'install-backend.py')
    await new Promise<void>((resolvePromise, rejectPromise) => {
      const child = spawn(this.options.pythonExec, [
        '-I', '-B', installer,
        '--wheelhouse', join(this.options.backendRoot, 'wheelhouse'),
        '--manifest', manifestPath,
        '--python', this.options.pythonExec,
        '--target', this.venv,
        '--receipt', receiptPath,
      ], { stdio: ['ignore', 'ignore', 'inherit'] })
      child.once('error', rejectPromise)
      child.once('exit', (code, signal) => {
        if (code === 0) resolvePromise()
        else rejectPromise(new Error(`desktop backend: installer failed (${String(code ?? signal)})`))
      })
    })
  }
}

export function readBackendEndpoint(userDataDir: string): string | undefined {
  const path = join(userDataDir, 'backend-endpoint.json')
  if (!existsSync(path)) return undefined
  try {
    return (JSON.parse(readFileSync(path, 'utf8')) as { endpoint?: string }).endpoint
  } catch {
    return undefined
  }
}
