/** Published PDF and spreadsheet chunks retain their bundled license notices. */
import { spawnSync } from 'node:child_process'
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, dirname, join, resolve } from 'node:path'
import { createRequire } from 'node:module'
import { runInNewContext } from 'node:vm'
import { describe, expect, it } from 'vitest'

const packageRoot = resolve(import.meta.dirname, '..')
const bundlePath = join(packageRoot, 'lib/client.js')
const pdfChunkPath = join(packageRoot, 'lib/client.pdf.js')
const require = createRequire(import.meta.url)
const licenseNames = [
  'LICENSE',
  'cmaps/LICENSE',
  'standard_fonts/LICENSE_FOXIT',
  'standard_fonts/LICENSE_LIBERATION',
  'wasm/LICENSE_JBIG2',
  'wasm/LICENSE_OPENJPEG',
  'wasm/LICENSE_PDFJS_JBIG2',
  'wasm/LICENSE_PDFJS_OPENJPEG',
  'wasm/LICENSE_PDFJS_QCMS',
  'wasm/LICENSE_QCMS',
] as const

function run(command: string, args: string[], cwd: string, timeout: number): string {
  const result = spawnSync(command, args, { cwd, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, timeout })
  expect(result.error).toBeUndefined()
  expect(result.signal, result.stderr).toBeNull()
  expect(result.status, result.stderr).toBe(0)
  return result.stdout
}

function runPnpm(args: string[], cwd: string, timeout: number): string {
  const entrypoint = process.env.npm_execpath
  if (entrypoint !== undefined && entrypoint !== '') {
    return /\.[cm]?js$/iu.test(entrypoint)
      ? run(process.execPath, [entrypoint, ...args], cwd, timeout)
      : run(entrypoint, args, cwd, timeout)
  }
  if (process.platform !== 'win32') return run('pnpm', args, cwd, timeout)
  // Windows has no shell-free way to reach pnpm here: the `.CMD` shim spawns as EINVAL and the extensionless
  // corepack shim as ENOENT, and `pnpm exec vitest` leaves no `npm_execpath`. Resolve the workspace-installed
  // pnpm CLI through Node's own resolver and run it with this process's Node — no PATH lookup, no shell shim.
  const manifestPath = require.resolve('pnpm')
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as { bin?: string | Record<string, string> }
  /* v8 ignore next -- pnpm always publishes a bin entry; the guard turns a broken install into a loud error */
  const binEntry = (typeof manifest.bin === 'string' ? manifest.bin : manifest.bin?.pnpm) as string
  return run(process.execPath, [resolve(dirname(manifestPath), binEntry), ...args], cwd, timeout)
}

describe('published document preview licenses', () => {
  it.skipIf(!existsSync(bundlePath))('keeps bundled licenses in the packed lazy chunks', ({ task }) => {
    expect(existsSync(pdfChunkPath)).toBe(true)
    const output = mkdtempSync(join(tmpdir(), 'dsh-document-preview-pack-'))
    try {
      const packed = JSON.parse(runPnpm([
        'pack', '--json', '--pack-destination', output,
      ], packageRoot, task.timeout)) as { filename: string; files: { path: string }[] }
      expect(packed.files.map(file => file.path)).toContain('lib/client.js')
      expect(packed.files.map(file => file.path)).toContain('lib/client.pdf.js')
      expect(packed.files.some(file => file.path.endsWith('pdfjs-NOTICES.txt'))).toBe(false)

      const client = run('tar', ['-xOf', basename(packed.filename), 'package/lib/client.js'], output, task.timeout)
      const pdf = run('tar', ['-xOf', basename(packed.filename), 'package/lib/client.pdf.js'], output, task.timeout)
      expect([...client.matchAll(/require\.async\("(\.\/client[^"/]*\.js)"\)/gu)].map(match => match[1]))
        .toEqual(['./client.pdf.js', './client.excel.js'])
      expect(client).not.toMatch(/\brequire\("\.\/client[^"/]*\.js"\)/u)
      expect([...pdf.matchAll(/require\("(\.\/client[^"/]*\.js)"\)/gu)].map(match => match[1]))
        .toEqual([])
      expect(client).not.toContain('//! Bundled PDF.js license notices')
      expect(client).not.toContain('/pdfjs-dist/')
      expect(pdf).toContain('//! Bundled PDF.js license notices')
      const excel = run('tar', ['-xOf', basename(packed.filename), 'package/lib/client.excel.js'], output, task.timeout)
      expect(excel).not.toMatch(/\brequire\("\.\/client[^"/]*\.js"\)/u)
      expect(client).not.toContain('FortuneSheet')
      expect(excel).toContain('//! Bundled spreadsheet license notices')
      expect(excel).toContain('Copyright (c) 2022 Suzhou Ruilisi Technology Co., Ltd')
      expect(excel).toContain('Permission is hereby granted, free of charge')
      for (const dependency of ['xlsx', 'papaparse']) {
        const root = dirname(require.resolve(dependency === 'xlsx' ? dependency : `${dependency}/package.json`))
        const license = readFileSync(join(root, 'LICENSE'), 'utf8').trimEnd()
        expect(excel).toContain(license.split('\n').map(line => `// ${line}`).join('\n'))
      }
      let initialized = false
      runInNewContext(excel, { window: { __ModuleLoader__: { load: (registration: {
        factory: (resolve: (specifier: string) => unknown) => { ExcelBody: unknown }
      }) => {
        const loaded = registration.factory((specifier) => {
          if (specifier === '@taiji/dsh-client-ui-primitives') return {}
          if (specifier === 'react' || specifier === 'react/jsx-runtime' || specifier === 'react-dom') return require(specifier)
          throw new Error(`Unexpected browser dependency: ${specifier}`)
        })
        expect(typeof loaded.ExcelBody).toBe('function')
        initialized = true
      } } } })
      expect(initialized).toBe(true)
      const pdfRoot = dirname(require.resolve('pdfjs-dist/package.json'))
      for (const name of licenseNames) {
        const source = readFileSync(join(pdfRoot, name), 'utf8').trimEnd()
        const commented = [`// ${name}`, '// ', ...source.split('\n').map(line => `// ${line}`)].join('\n')
        expect(pdf, `${name} must be visible in package/lib/client.pdf.js`).toContain(commented)
      }
    } finally {
      rmSync(output, { recursive: true, force: true })
    }
  })
})
