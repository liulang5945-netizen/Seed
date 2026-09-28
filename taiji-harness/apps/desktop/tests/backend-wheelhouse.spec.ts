/** D2 P1-①: wheelhouse generator helpers — freeze split, wheel parsing, manifest build. */
import { mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { afterEach, describe, expect, it } from 'vitest'
import {
  buildManifest,
  parseWheelFilename,
  splitTorchPins,
} from '../scripts/prepare-backend-wheelhouse.ts'

const freeze = [
  'torch==2.9.1',
  'filelock==3.20.0',
  'numpy==2.3.4',
  'pytorch-triton==3.5.1',
]

describe('splitTorchPins', () => {
  it('routes the torch family to the CPU-index pass and drops junk', () => {
    const { torch, rest } = splitTorchPins(freeze)
    expect(torch).toEqual(['torch==2.9.1', 'pytorch-triton==3.5.1'])
    expect(rest).toEqual(['filelock==3.20.0', 'numpy==2.3.4'])
  })
})

describe('parseWheelFilename', () => {
  it('reads name and version from a PEP 427 wheel name', () => {
    expect(parseWheelFilename('numpy-2.3.4-cp312-cp312-win_amd64.whl')).toEqual({
      name: 'numpy',
      version: '2.3.4',
    })
    expect(parseWheelFilename('torch-2.9.1-cp312-none-win_amd64.whl')).toEqual({
      name: 'torch',
      version: '2.9.1',
    })
  })
})

describe('buildManifest', () => {
  const root = join(tmpdir(), `taiji-wheelhouse-spec-${String(process.pid)}`)
  const wheelsDir = join(root, 'wheelhouse')

  afterEach(() => {
    try { rmSync(root, { recursive: true, force: true }) } catch { /* temp dir may not exist */ }
  })

  it('fingerprints wheels with sha256 and totals the bytes', () => {
    mkdirSync(wheelsDir, { recursive: true })
    writeFileSync(join(wheelsDir, 'numpy-2.3.4-cp312-cp312-win_amd64.whl'), 'wheel-bytes-a')
    writeFileSync(join(wheelsDir, 'torch-2.9.1-cp312-none-win_amd64.whl'), 'wheel-bytes-b')
    const manifest = buildManifest(wheelsDir, { mode: 'freeze', python: 'python', digest: 'd'.repeat(64), pins: 2 }, { platform_tag: 'win_amd64', python_version_tag: 'cp312', index: 'https://pypi.org/simple' })
    expect(manifest.format).toBe('taiji-backend-wheelhouse-v1')
    expect(manifest.wheels.map(wheel => wheel.name)).toEqual(['numpy', 'torch'])
    expect(manifest.totals).toEqual({ count: 2, bytes: 26 })
    expect(manifest.wheels[0]!.sha256).toMatch(/^[a-f0-9]{64}$/u)
  })
})
