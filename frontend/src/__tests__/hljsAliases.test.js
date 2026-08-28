/**
 * hljsAliases.js 再生成门禁。
 *
 * 为什么需要这个文件：src/composables/hljsAliases.js 是构建期固化的静态产物，
 * 升级 highlight.js 时它不会自动更新。一旦上游新增语言（例如 zig），用户写
 * ```zig 就会静默退化成无高亮纯文本——全部测试依然全绿。这是上一轮刚付过学费的
 * 「测试全绿但功能坏掉」失效模式，必须由门禁而不是记忆来兜住。
 *
 * 本文件从 scripts/gen-hljs-aliases.mjs 导入生成逻辑，不复制任何一行别名计算。
 * 若在此处重写一份「简化版」推导，就是上一轮「假测试」的原样重犯。
 *
 * 也因此本用例必须独立成文件：断言需要触碰完整语法清单，而 highlight.js 与
 * highlight.js/lib/core 是同一个单例对象（实测 sameObj=true），在 useMarkdown
 * 的测试文件里做这件事会污染按需加载的前提。
 */
import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import {
  buildAliasMap,
  renderModule,
  resolveHljsRoot,
  listGrammarFiles,
} from '../../scripts/gen-hljs-aliases.mjs'
import { HLJS_ALIASES } from '@/composables/hljsAliases.js'

const TARGET = path.join(process.cwd(), 'src', 'composables', 'hljsAliases.js')
const normalize = (text) => text.replace(/\r\n/g, '\n')

describe('hljsAliases 与 highlight.js 同步', () => {
  it('别名键集合与当前 highlight.js 完全一致', async () => {
    const live = await buildAliasMap()
    const liveKeys = Object.keys(live).sort()
    const fileKeys = Object.keys(HLJS_ALIASES).sort()

    const missing = liveKeys.filter((k) => !(k in HLJS_ALIASES))
    const extra = fileKeys.filter((k) => !(k in live))

    expect(
      { missing, extra },
      'highlight.js 别名已变化，执行 npm run gen:aliases 重新生成'
    ).toEqual({ missing: [], extra: [] })
  })

  it('别名指向的语法文件名逐项一致', async () => {
    const live = await buildAliasMap()
    const drift = Object.keys(live)
      .filter((k) => HLJS_ALIASES[k] !== live[k])
      .map((k) => `${k}: 文件=${HLJS_ALIASES[k]} 实际=${live[k]}`)

    expect(drift, '别名归属发生漂移，执行 npm run gen:aliases').toEqual([])
  })

  it('磁盘文件与生成器输出逐字节一致（等价于 --check）', async () => {
    const expected = renderModule(await buildAliasMap())
    const actual = normalize(readFileSync(TARGET, 'utf8'))

    expect(actual, 'hljsAliases.js 已过期或被手改，执行 npm run gen:aliases').toBe(expected)
  })

  it('每个别名的目标都是真实存在的语法文件', async () => {
    const grammars = new Set(await listGrammarFiles(resolveHljsRoot()))
    const dangling = Object.entries(HLJS_ALIASES)
      .filter(([, name]) => !grammars.has(name))
      .map(([alias, name]) => `${alias} -> ${name}`)

    expect(dangling, '别名指向了不存在的语法，按需加载会 404').toEqual([])
  })

  it('别名冲突按上游注册顺序取后者', async () => {
    const live = await buildAliasMap()
    // ls 被 lasso 与 livescript 同时声明，ml 被 ocaml 与 sml 同时声明。
    // 上游 registerAliases 直接覆盖，注册在后者胜；此处锁定实测结论。
    expect(live.ls).toBe('livescript')
    expect(live.ml).toBe('sml')
    expect(HLJS_ALIASES.ls).toBe('livescript')
    expect(HLJS_ALIASES.ml).toBe('sml')
  })

  it('常用别名可用', () => {
    expect(HLJS_ALIASES.py).toBe('python')
    expect(HLJS_ALIASES.js).toBe('javascript')
    expect(HLJS_ALIASES.ts).toBe('typescript')
    expect(HLJS_ALIASES.sh).toBe('bash')
    expect(HLJS_ALIASES.yml).toBe('yaml')
    expect(HLJS_ALIASES['c++']).toBe('cpp')
  })
})
