/** The Chinese config-catalog mirror renders from the same entries as the
 * English generator: fences must be byte-identical, anchors and headings
 * must match line for line, and only labels/prose may differ. */

import { describe, expect, it } from 'vitest'
import { type CatalogEntry, render } from './gen-config-catalog.ts'
import { extractFences, renderZh } from './gen-config-catalog-zh.ts'
import { githubSlug } from './verify-md-links.ts'

const ALPHA_PASTE = 'export interface AlphaConfig {\n  /** Model name. */\n  model?: string\n}'

const entries: CatalogEntry[] = [
  {
    pkg: '@t/dsh-alpha',
    dir: 'packages/t/alpha',
    entry: 'packages/t/alpha/src/index.ts',
    kind: 'config',
    inject: ['llm', 'sessions'],
    configTypeName: 'AlphaConfig',
    pastes: [{ text: ALPHA_PASTE, source: 'packages/t/alpha/src/index.ts:7' }],
    refs: [{ alias: 'Volatile', imported: 'Volatile', specifier: '@taiji/cordis' }],
    schemaKeys: null,
  },
  {
    pkg: '@t/dsh-epsilon',
    dir: 'packages/t/epsilon',
    entry: 'packages/t/epsilon/src/index.ts',
    kind: 'config',
    inject: [],
    configTypeName: 'EpsilonConfig',
    pastes: [{ text: 'export interface EpsilonConfig {\n  /** Nothing. */\n  off?: boolean\n}', source: 'packages/t/epsilon/src/config.ts:3' }],
    refs: [{ alias: 'AlphaConfig', imported: 'AlphaConfig', specifier: '@t/dsh-alpha' }],
    schemaKeys: null,
  },
  { pkg: '@t/dsh-beta', dir: 'packages/t/beta', entry: 'packages/t/beta/src/index.ts', kind: 'no-config', inject: ['loader'] },
  { pkg: '@t/dsh-gamma', dir: 'packages/t/gamma', entry: 'packages/t/gamma/src/index.ts', kind: 'seam', inject: [], className: 'GammaStore' },
  { pkg: '@t/dsh-delta', dir: 'packages/t/delta', entry: 'packages/t/delta/src/index.ts', kind: 'library', inject: [] },
]

describe('gen-config-catalog-zh', () => {
  it('pastes fences byte-identically to the English render', () => {
    const en = extractFences(render(entries))
    const zh = extractFences(renderZh(entries))
    expect(zh.length).toBeGreaterThan(0)
    expect(zh).toEqual(en)
  })

  it('mirrors anchors and package headings line for line', () => {
    const frame = (text: string): string[] => text
      .split('\n')
      .filter(line => line.startsWith('<a id=') || line.startsWith('## `'))
    expect(frame(renderZh(entries))).toEqual(frame(render(entries)))
  })

  it('translates the per-section labels and keeps pointer targets', () => {
    const zh = renderZh(entries)
    expect(zh).toContain('需要： `llm` · `sessions`')
    expect(zh).toContain('来源： [`packages/t/alpha/src/index.ts:7`](../packages/t/alpha/src/index.ts)')
    expect(zh).toContain('依赖： `Volatile` (`@taiji/cordis`)')
    expect(zh).toContain(`依赖： [\`AlphaConfig\`](#${githubSlug('@t/dsh-alpha')})`)
  })

  it('renders terse rows with translated requires and seam detail', () => {
    const zh = renderZh(entries)
    expect(zh).toContain('- `@t/dsh-beta` — 需要 `loader`（[`packages/t/beta/src/index.ts`](../packages/t/beta/src/index.ts)）')
    expect(zh).toContain('- `@t/dsh-gamma` — 抽象 `GammaStore`（[`packages/t/gamma/src/index.ts`](../packages/t/gamma/src/index.ts)）')
    expect(zh).toContain('- `@t/dsh-delta`（[`packages/t/delta/src/index.ts`](../packages/t/delta/src/index.ts)）')
  })

  it('carries the generated-file header, title, and language line', () => {
    const zh = renderZh(entries)
    expect(zh.startsWith('<!-- 本文件由 scripts/gen-config-catalog-zh.ts')).toBe(true)
    expect(zh).toContain('# 插件配置目录')
    expect(zh).toContain('[English](config-catalog.md) | 中文')
    expect(zh.endsWith('\n')).toBe(true)
  })
})
