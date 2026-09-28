import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { join } from 'node:path'
import { expect, it } from 'vitest'

const DIST_ROOT = fileURLToPath(new URL('../dist', import.meta.url))

it('ships install metadata with the built web application', async () => {
  const index = await readFile(join(DIST_ROOT, 'index.html'), 'utf8')
  expect(index).toContain('<link rel="manifest" href="./manifest.webmanifest" />')

  const manifest: unknown = JSON.parse(await readFile(join(DIST_ROOT, 'manifest.webmanifest'), 'utf8'))
  // No `id`: a browser resolves an explicit `id` against the start URL's origin,
  // so only an absent `id`, which defaults to the resolved `start_url`, gives
  // each mount its own identity. `public-mount.e2e.ts` reads the resolved form.
  expect(manifest).toEqual({
    name: 'Seed',
    short_name: 'Seed',
    start_url: './',
    scope: './',
    display: 'fullscreen',
    icons: [{
      src: 'favicon.svg',
      sizes: 'any',
      type: 'image/svg+xml',
      purpose: 'any',
    }],
  })
})

it('ships fixed-color favicons selected by document media queries', async () => {
  const index = await readFile(join(DIST_ROOT, 'index.html'), 'utf8')
  expect(index).toContain('<link rel="icon" type="image/svg+xml" href="./favicon-dark.svg" media="(prefers-color-scheme: dark)" />')
  expect(index).toContain('<link rel="icon" type="image/svg+xml" href="./favicon.svg" media="(prefers-color-scheme: light)" />')
  const light = await readFile(join(DIST_ROOT, 'favicon.svg'), 'utf8')
  const dark = await readFile(join(DIST_ROOT, 'favicon-dark.svg'), 'utf8')
  expect(light).not.toContain('<style>')
  // The mark paints light with its two colour layers and dark as one combined
  // silhouette, so what the pair must share is the traced contour, not byte equality.
  expect(light).toContain('fill="#124A38"')
  expect(light).toContain('fill="#AAD66A"')
  expect(dark).toContain('fill="#fff"')
  const lightShell = /d="([^"]{40})/.exec(light)?.[1]
  const darkShell = /d="([^"]{40})/.exec(dark)?.[1]
  if (lightShell === undefined || darkShell === undefined) throw new Error('favicon SVG carries no path data')
  expect(darkShell).toBe(lightShell)
})
