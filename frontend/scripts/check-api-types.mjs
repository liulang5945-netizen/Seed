/**
 * 漂移门：`src/api/schema.d.ts` 必须逐字节等于「从冻结快照重新生成」的结果。
 *
 * 与 `check:aliases`（gen-hljs-aliases.mjs --check）同一约定：生成物提交入库，
 * 但必须有门证明它与真源一致。真源是 `tests/snapshots/openapi_baseline.json`，
 * 由后端 CI 守护；这里只负责前端这一侧的生成物不会悄悄过期。
 */

import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const snapshot = path.resolve(frontendRoot, '..', 'tests', 'snapshots', 'openapi_baseline.json')
const committed = path.join(frontendRoot, 'src', 'api', 'schema.d.ts')
const cli = path.join(frontendRoot, 'node_modules', 'openapi-typescript', 'bin', 'cli.js')

if (!fs.existsSync(snapshot)) {
  console.error(`[api-types] FAIL: 找不到冻结快照 ${snapshot}`)
  process.exit(1)
}
if (!fs.existsSync(cli)) {
  console.error('[api-types] FAIL: 缺少 openapi-typescript，请先 npm ci')
  process.exit(1)
}

const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'seed-api-types-'))
const fresh = path.join(scratch, 'schema.d.ts')

try {
  execFileSync(process.execPath, [cli, snapshot, '-o', fresh], { stdio: 'pipe' })
  const expected = fs.readFileSync(fresh, 'utf8')
  const actual = fs.readFileSync(committed, 'utf8')
  if (expected !== actual) {
    console.error('[api-types] FAIL: src/api/schema.d.ts 与冻结快照不一致')
    console.error('  刷新方式：npm run gen:api-types，并把结果一并提交')
    process.exit(1)
  }
  console.log(`[api-types] PASS: schema.d.ts 与 openapi_baseline.json 逐字节一致（${actual.length} 字节）`)
} finally {
  try {
    fs.rmSync(scratch, { recursive: true, force: true })
  } catch (cause) {
    console.warn(`[api-types] 临时目录清理失败（非致命）：${cause.message}`)
  }
}
