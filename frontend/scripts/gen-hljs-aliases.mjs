import { createRequire } from 'node:module';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { readdir } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

/**
 * highlight.js 别名表生成器。
 *
 * 为什么需要它：useMarkdown.js 按需加载语法，而 highlight.js 的别名只在语法
 * 注册之后才存在，且对未注册语言调用 hljs.highlight 会直接抛错。所以必须在
 * 构建期把「别名 -> 语法文件名」固化下来。
 *
 * 本文件同时被 CLI 和 src/__tests__/hljsAliases.test.js 使用——生成逻辑只允许
 * 存在一份，测试不得复制一份「简化版」自测自答。
 */

const require = createRequire(import.meta.url);

const IDENT = /^[A-Za-z_$][A-Za-z0-9_$]*$/;

export function resolveHljsRoot() {
  let dir = path.dirname(require.resolve('highlight.js/lib/core'));
  for (let i = 0; i < 5; i += 1) {
    if (existsSync(path.join(dir, 'es', 'languages'))) return dir;
    dir = path.dirname(dir);
  }
  throw new Error('无法定位 highlight.js 包根目录');
}

/**
 * 复刻上游的语法注册顺序。别名存在真实冲突（ls: lasso/livescript、
 * ml: ocaml/sml），上游 lib/index.js 按注册顺序「后者胜」，而该顺序并非
 * 文件名字典序，因此必须从 lib/index.js 解析而不能自行排序。
 */
export function readRegistrationOrder(root) {
  const src = readFileSync(path.join(root, 'lib', 'index.js'), 'utf8');
  return [...src.matchAll(/registerLanguage\('([^']+)'/g)].map((m) => m[1]);
}

export async function listGrammarFiles(root) {
  const dir = path.join(root, 'es', 'languages');
  // es/languages 下有 384 个文件：192 个真实语法 + 192 个 `<name>.js.js` 兼容 shim
  const files = (await readdir(dir)).filter((f) => f.endsWith('.js') && !f.endsWith('.js.js'));
  return files.map((f) => f.slice(0, -3));
}

export async function buildAliasMap() {
  const root = resolveHljsRoot();
  const order = readRegistrationOrder(root);
  const present = new Set(await listGrammarFiles(root));

  const missing = order.filter((n) => !present.has(n));
  const unregistered = [...present].filter((n) => !order.includes(n));
  if (missing.length || unregistered.length) {
    throw new Error(
      `highlight.js 语法清单不自洽：lib/index.js 独有 [${missing}]，es/languages 独有 [${unregistered}]`
    );
  }

  // newInstance() 拿一个隔离实例：语法定义工厂需要 hljs 的正则辅助函数，
  // 但绝不能污染 useMarkdown.js 正在使用的那个单例（两者是同一对象）。
  const probe = (await import('highlight.js/lib/core')).default.newInstance();
  const aliases = {};

  for (const name of order) {
    const url = pathToFileURL(path.join(root, 'es', 'languages', `${name}.js`)).href;
    const mod = await import(url);
    const define = mod.default || mod;
    if (typeof define !== 'function') throw new Error(`语法模块不是工厂函数: ${name}`);
    for (const raw of define(probe).aliases || []) {
      const key = String(raw).toLowerCase();
      if (key === name) continue;
      aliases[key] = name;
    }
  }

  return aliases;
}

export function renderModule(aliases) {
  const lines = Object.keys(aliases)
    .sort()
    .map((key) => {
      const left = IDENT.test(key) ? key : JSON.stringify(key);
      return `  ${left}: '${aliases[key]}',`;
    });
  return [
    '// 由 scripts/gen-hljs-aliases.mjs 生成，请勿手改。',
    '// 升级 highlight.js 后执行 npm run gen:aliases 重新生成。',
    'export const HLJS_ALIASES = {',
    ...lines,
    '}',
    '',
  ].join('\n');
}

export const TARGET = path.join(process.cwd(), 'src', 'composables', 'hljsAliases.js');

function normalize(text) {
  return text.replace(/\r\n/g, '\n');
}

async function main() {
  const check = process.argv.includes('--check');
  const expected = renderModule(await buildAliasMap());
  const count = expected.split('\n').length - 5;

  if (check) {
    const actual = existsSync(TARGET) ? normalize(readFileSync(TARGET, 'utf8')) : '';
    if (actual !== expected) {
      console.error('hljsAliases.js 已过期，请执行 npm run gen:aliases');
      process.exit(1);
    }
    console.log(`hljsAliases.js 与 highlight.js 一致（${count} 条别名）`);
    return;
  }

  writeFileSync(TARGET, expected, 'utf8');
  console.log(`已写入 ${TARGET}（${count} 条别名）`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
