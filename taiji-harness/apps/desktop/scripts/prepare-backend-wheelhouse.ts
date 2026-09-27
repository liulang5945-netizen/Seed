/**
 * Prepare the offline backend wheelhouse (D2 P1-①, owner-approved 2026-09-27).
 *
 * The backend (Python runtime: torch + the API server deps) must reach user
 * machines without any network fetch at install time, and it cannot ride the
 * primary-runtime `pythonPackages` payload (release size limits — G5 §1-D2).
 * This script materializes the wheelhouse the design defines
 * (TAIJI_D2_BACKEND_CHANNEL_DESIGN_20260927.md §2):
 *
 * 1. take the backend lock — either `pip freeze` of the running runtime's
 *    interpreter (the true pins of what actually works) or an explicit
 *    requirements file;
 * 2. download exact wheels in two passes: the torch family from the CPU-only
 *    index (the PyPI win wheel is the CUDA build), everything else from PyPI —
 *    both with `--no-deps` because a freeze list is already the full closure;
 * 3. fingerprint every wheel (sha256, size) into `backend-manifest.json`, the
 *    payload `install-backend.py` verifies before going offline.
 */

import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'

const WHEELHOUSE_MANIFEST_FORMAT = 'taiji-backend-wheelhouse-v1'
const DEFAULT_TORCH_INDEX = 'https://download.pytorch.org/whl/cpu'
const DEFAULT_PYPI_INDEX = 'https://pypi.org/simple'
const TORCH_FAMILY = /^(torch|pytorch-triton|pytorch-triton-windows)(==|$)/u

export interface WheelRecord {
  filename: string
  name: string
  version: string
  size: number
  sha256: string
}

export interface WheelhouseManifest {
  format: string
  generated_at: number
  source: { mode: 'freeze' | 'requirements'; python: string; digest: string; pins: number }
  target: { platform_tag: string; python_version_tag: string; index: string }
  wheels: WheelRecord[]
  totals: { count: number; bytes: number }
}

/** Split `pip freeze` output into the torch-family pins and everything else. */
export function splitTorchPins(pins: readonly string[]): { torch: string[]; rest: string[] } {
  const torch: string[] = []
  const rest: string[] = []
  for (const pin of pins) {
    const name = pin.replace(/\s*[[^\]]+\]$/u, '').split(/[=<>!~;]/u, 1)[0]?.trim() ?? ''
    if (TORCH_FAMILY.test(name)) torch.push(pin)
    else if (name) rest.push(pin)
  }
  return { torch, rest }
}

function run(command: string, args: readonly string[]): string {
  const result = spawnSync(command, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] })
  if (result.status !== 0) {
    const stderr = (result.stderr ?? '').split('\n').slice(-6).join('\n')
    throw new Error(`desktop backend: ${command} ${args.join(' ')} failed (${result.status}):\n${stderr}`)
  }
  return result.stdout ?? ''
}

/** Read the backend lock: exact pins of the interpreter that runs today. */
export function freezeBackendRequirements(pythonExec: string): string[] {
  const stdout = run(pythonExec, ['-m', 'pip', 'freeze', '--all'])
  return stdout
    .split(/\r?\n/u)
    .map(line => line.trim())
    .filter(line => line && !line.startsWith('#') && !line.startsWith('-'))
}

function sha256File(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex')
}

/** Parse one wheel filename into name/version (PEP 427 normalized enough for a manifest). */
export function parseWheelFilename(filename: string): { name: string; version: string } {
  const stem = filename.replace(/\.whl$/u, '')
  const firstDash = stem.indexOf('-')
  const secondDash = stem.indexOf('-', firstDash + 1)
  return {
    name: stem.slice(0, firstDash).replaceAll('_', '-').toLowerCase(),
    version: stem.slice(firstDash + 1, secondDash),
  }
}

export function buildManifest(
  wheelsDir: string,
  source: WheelhouseManifest['source'],
  target: WheelhouseManifest['target'],
): WheelhouseManifest {
  const wheels: WheelRecord[] = readdirSync(wheelsDir)
    .filter(file => file.endsWith('.whl'))
    .map((file) => {
      const path = join(wheelsDir, file)
      const { name, version } = parseWheelFilename(file)
      return { filename: file, name, version, size: statSync(path).size, sha256: sha256File(path) }
    })
    .sort((left, right) => left.filename.localeCompare(right.filename))
  return {
    format: WHEELHOUSE_MANIFEST_FORMAT,
    generated_at: Date.now(),
    source,
    target,
    wheels,
    totals: { count: wheels.length, bytes: wheels.reduce((sum, wheel) => sum + wheel.size, 0) },
  }
}

export interface PrepareOptions {
  out: string
  pythonExec: string
  requirements?: string
  torchIndex?: string
  pypiIndex?: string
  platformTag?: string
  pythonVersionTag?: string
  /** Repo root holding pyproject.toml; the Seed code wheel is built from here. */
  seedRoot?: string
  force?: boolean
}

export async function prepareBackendWheelhouse(options: PrepareOptions): Promise<WheelhouseManifest> {
  const outDir = resolve(options.out)
  const manifestPath = join(outDir, 'backend-manifest.json')
  const wheelsDir = join(outDir, 'wheelhouse')
  if (!options.force && existsSync(manifestPath)) {
    return JSON.parse(readFileSync(manifestPath, 'utf8')) as WheelhouseManifest
  }

  const pins =
    options.requirements === undefined
      ? freezeBackendRequirements(options.pythonExec)
      : readFileSync(options.requirements, 'utf8')
        .split(/\r?\n/u)
        .map(line => line.trim())
        .filter(line => line && !line.startsWith('#') && !line.startsWith('-'))
  if (pins.length === 0) throw new Error('desktop backend: the resolved requirements list is empty')
  const { torch, rest } = splitTorchPins(pins)

  const download = (pinsSubset: readonly string[], index: string): void => {
    const reqFile = join(outDir, `requirements-pass-${index.replace(/\W+/gu, '-')}.txt`)
    writeFileSync(reqFile, `${pinsSubset.join('\n')}\n`, 'utf8')
    run(options.pythonExec, [
      '-m', 'pip', 'download',
      '--only-binary=:all:',
      '--no-deps',
      '--index-url', index,
      '-r', reqFile,
      '-d', wheelsDir,
    ])
  }

  const { mkdirSync } = await import('node:fs')
  mkdirSync(wheelsDir, { recursive: true })
  if (torch.length > 0) download(torch, options.torchIndex ?? DEFAULT_TORCH_INDEX)
  if (rest.length > 0) download(rest, options.pypiIndex ?? DEFAULT_PYPI_INDEX)
  // NOTE: the Seed runtime SOURCE tree (api/seed_platform/taiji/neuroplex/instruments)
  // is NOT shipped as a wheel — `pip wheel` over the repo hangs on the data tree —
  // it is staged as plain source by prepare-dsh (backend:stage) and the venv runs
  // `python -m api.main` with that tree as its cwd (sys.path[0]).
  // The Seed code wheel itself: the freeze pins only third-party packages, but
  // `python -m api.main` runs from the venv, so the runtime source rides the
  // wheelhouse as its own wheel built from the repo root (pyproject packages
  // include seed*/taiji*/seed_platform*/neuroplex*/api*/instruments*).
  const seedRoot = options.seedRoot ?? resolve(outDir, '..', '..', '..', '..', '..')
  run(options.pythonExec, ['-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', '-w', wheelsDir, seedRoot])

  const manifest = buildManifest(
    wheelsDir,
    {
      mode: options.requirements === undefined ? 'freeze' : 'requirements',
      python: options.pythonExec,
      digest: createHash('sha256').update(pins.join('\n')).digest('hex'),
      pins: pins.length,
    },
    {
      platform_tag: options.platformTag ?? 'win_amd64',
      python_version_tag: options.pythonVersionTag ?? 'cp312',
      index: options.pypiIndex ?? DEFAULT_PYPI_INDEX,
    },
  )
  writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8')
  return manifest
}

const invokedDirectly = process.argv[1] !== undefined && import.meta.url === new URL(`file://${process.argv[1].replaceAll('\\', '/')}`).href
if (invokedDirectly) {
  const args = process.argv.slice(2)
  const flag = (name: string): string | undefined => {
    const index = args.indexOf(name)
    return index === -1 ? undefined : args[index + 1]
  }
  const pythonExec = flag('--python') ?? 'python'
  const manifest = await prepareBackendWheelhouse({
    out: flag('--out') ?? join(resolve('apps/desktop/.desktop-build'), 'backend'),
    pythonExec,
    ...(flag('--requirements') === undefined ? {} : { requirements: flag('--requirements') }),
    ...(flag('--torch-index') === undefined ? {} : { torchIndex: flag('--torch-index') }),
    ...(flag('--pypi-index') === undefined ? {} : { pypiIndex: flag('--pypi-index') }),
    ...(flag('--platform-tag') === undefined ? {} : { platformTag: flag('--platform-tag') }),
    ...(flag('--python-version-tag') === undefined ? {} : { pythonVersionTag: flag('--python-version-tag') }),
    force: args.includes('--force'),
  })
  console.log(`desktop backend: wheelhouse ready — ${manifest.totals.count} wheels, ${(manifest.totals.bytes / 1_048_576).toFixed(1)} MB`)
}
