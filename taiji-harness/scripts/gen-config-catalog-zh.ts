/**
 * Generate `docs/config-catalog.zh.md` — the Chinese mirror of the config
 * catalog — from the SAME collected entries as the English generator. Every
 * mechanical element (anchors, headings, verbatim fences, source pointers,
 * terse list rows) is rendered from source, so line-number or fence drift
 * between the two languages is structurally impossible; only the translated
 * frame (title, intros, labels) lives here and is reviewed as code. `--check`
 * verifies the committed artifact. After writing, re-record the pairing:
 * `verify-translation-pairing --write docs/config-catalog.md`.
 */

import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { FENCE, type CatalogEntry, type ConfigCatalogCopy, collectConfigCatalog, render, renderConfigEntryWith } from './gen-config-catalog.ts'

const root = resolve(import.meta.dirname, '..')
const OUT = 'docs/config-catalog.zh.md'

/** The translated frame. Intro paragraphs were reviewed as the hand-maintained
 * mirror's prose and are kept verbatim (with the `需要：` label and the
 * generated-file narrative updated); everything below them is mechanical. */
const TITLE = '# 插件配置目录'
const LANGUAGE_LINE = '[English](config-catalog.md) | 中文'
const INTRO_1 = '每个 `config:` 块均可由 `cordis.yml` 条目设置：针对每个可加载的 harness 包，原样列出其 `apply` 函数或服务构造函数接收的配置声明（包括 JSDoc），并附上所有引用类型——包内类型直接粘贴，其他类型则提供链接。粘贴的内容是插件声明的完整配置类型——运行时 schema 有意排除的字段是仅供运行时使用的 seam（其自身的 JSDoc 会如此说明），不能通过 `cordis.yml` 设置。这是以**部署**为轴的参考文档——插件作者所依据的连接方式请参阅各[子系统页面](subsystems/core.zh.md)中的生成 `cordis-surface` 区域，面向模型的工具 schema 请参阅[工具目录](tool-catalog.zh.md)，而 [subsystems/](subsystems/core.zh.md) 则记录了这些声明所引用的类型。'
const INTRO_2 = '本文件与英文版同为生成物：由 `scripts/gen-config-catalog-zh.ts` 从同一份源码收集结果渲染（更新先跑 `pnpm run gen-config-catalog` 再跑 `pnpm run gen-config-catalog-zh`），并通过 `pnpm run verify-config-catalog-zh`（`doc-sync` 的一部分）验证新鲜度——请勿手改。声明块使用 `ts config-catalog` 围栏（doc-typecheck 会跳过它，因为单独引用导入项的声明无法独立编译）。生成器还会将运行时 schemastery schema 与粘贴的声明进行交叉核对——每个经 schema 验证的键（包括嵌套键）都必须能在声明的配置类型中找到——因此，粘贴内容无法隐藏加载器接受的字段。'
const INTRO_3 = '`需要：` 行列出插件通过 `inject` 注入的服务键：其 `cordis.yml` 树还必须加载这些服务的提供者。范围限定为 harness 层级（`packages/`）；配置树还可能加载的 vendored cordis 插件（控制台日志记录器等）固定为上游源代码（参见 [vendoring policy](../vendor/README.md)），未收录于此目录。'
const NO_CONFIG_HEADING = '## 无配置的可加载插件'
const NO_CONFIG_INTRO = '这些插件通过 `cordis.yml` 中不含 `config:` 块的条目加载；它们未声明任何配置接口。'
const SEAM_HEADING = '## Seam 包（不可直接加载）'
const SEAM_INTRO = '抽象服务类——部署时应改为加载具体的实现包（参见[能力 seam](../.agents/notes/implemented/architecture/2026-06-13-capability-seams.zh.md)）。'
const LIBRARY_HEADING = '## 库包（无插件入口）'
const LIBRARY_INTRO = '由其他包作为库导入；`cordis.yml` 无法加载它们。'

/** `subsystems/<page>` references point at the zh page when one exists (the
 * pairing gate accepts either target; an existing zh page is the better
 * read). A missing zh side falls back to the English target, never a dead
 * link. */
function subsystemLink(page: string): string {
  const localized = `subsystems/${page.replace(/\.md$/, '.zh.md')}`
  return existsSync(resolve(root, 'docs', localized)) ? localized : `subsystems/${page}`
}

/** Chinese catalog copy: same mechanical renderers as the English generator,
 * with the translated labels and the localized subsystem target. */
const CATALOG_ZH: ConfigCatalogCopy = {
  requiresLabel: '需要： ', dependsLabel: '依赖： ', sourceLabel: '来源： ',
  subsystemHref: page => subsystemLink(page),
}

/** Render one terse list line (the no-config / seam / library sections). */
function renderTerse(entry: CatalogEntry, detail: string): string {
  const requires = entry.inject.length ? ` — 需要 ${entry.inject.map(k => `\`${k}\``).join(' · ')}` : ''
  return `- \`${entry.pkg}\`${detail}${requires}（[\`${entry.entry}\`](../${entry.entry})）`
}

/** Render the full Chinese catalog (pure, deterministic given sorted entries). */
export function renderZh(entries: CatalogEntry[]): string {
  const byName = new Map(entries.map(e => [e.pkg, e]))
  const lines: string[] = [
    '<!-- 本文件由 scripts/gen-config-catalog-zh.ts 从源码收集结果生成（与英文版同源），请勿手改。',
    '     更新：`pnpm run gen-config-catalog` → `pnpm run gen-config-catalog-zh` →',
    '     `pnpm run verify-translation-pairing --write docs/config-catalog.md` 重新记录配对。 -->',
    '',
    TITLE,
    '',
    LANGUAGE_LINE,
    '',
    INTRO_1,
    '',
    INTRO_2,
    '',
    INTRO_3,
    '',
  ]
  for (const entry of entries.filter(e => e.kind === 'config')) {
    lines.push(...renderConfigEntryWith(entry, byName, CATALOG_ZH))
  }
  lines.push(
    NO_CONFIG_HEADING,
    '',
    NO_CONFIG_INTRO,
    '',
    ...entries.filter(e => e.kind === 'no-config').map(e => renderTerse(e, '')),
    '',
    SEAM_HEADING,
    '',
    SEAM_INTRO,
    '',
    ...entries.filter(e => e.kind === 'seam').map(e => renderTerse(e, ` — 抽象 \`${e.className ?? ''}\``)),
    '',
    LIBRARY_HEADING,
    '',
    LIBRARY_INTRO,
    '',
    ...entries.filter(e => e.kind === 'library').map(e => renderTerse(e, '')),
    '',
  )
  return lines.join('\n')
}

/** Extract every `ts config-catalog` fence body, verbatim. */
export function extractFences(text: string): string[] {
  const open = '```' + FENCE
  const fences: string[] = []
  const lines = text.split('\n')
  for (let i = 0; i < lines.length; i += 1) {
    if (lines[i] !== open) continue
    const body: string[] = []
    i += 1
    while (i < lines.length && lines[i] !== '```') {
      body.push(lines[i] ?? '')
      i += 1
    }
    fences.push(body.join('\n'))
  }
  return fences
}

/** CLI entry: default writes the Chinese mirror, `--check` fails if the
 * committed copy is stale. Also asserts fence-for-fence parity with the
 * English render, so a future divergence between the two renderers is loud.
 * Guarded behind an entry-point check so importing this module for tests
 * neither rewrites the committed file nor calls process.exit. */
function main(): void {
  const entries = collectConfigCatalog(root)
  const content = renderZh(entries)
  const enFences = extractFences(render(entries))
  const zhFences = extractFences(content)
  if (enFences.length !== zhFences.length || enFences.some((f, i) => f !== zhFences[i])) {
    throw new Error('gen-config-catalog-zh: fence content diverges from the English render; align the two renderers before committing.')
  }
  if (process.argv.includes('--check')) {
    let committed: string | null = null
    try {
      committed = readFileSync(resolve(root, OUT), 'utf8')
    } catch {
      committed = null
    }
    if (committed === content) {
      console.log(`gen-config-catalog-zh: ${OUT} is up to date.`)
      process.exit(0)
    }
    console.error(`gen-config-catalog-zh: ${OUT} is stale. Run \`pnpm run gen-config-catalog-zh\`, re-record the pairing, and commit ${OUT}.`)
    process.exit(1)
  }
  writeFileSync(resolve(root, OUT), content)
  console.log(`gen-config-catalog-zh: wrote ${OUT}.`)
}

if (process.argv[1] && import.meta.filename === resolve(process.argv[1])) {
  main()
}
