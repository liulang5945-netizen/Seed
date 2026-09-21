/**
 * 让 electron-builder 面对一个全新输出目录。
 *
 * ## 为什么是「重命名」而不是「删除」
 *
 * 实测（2026-09-21，本机）：
 * - `release/` 已存在时，构建会稳定死住 —— 日志停在 `• copying unpacked Electron`，
 *   `release/win-unpacked` 拷到 32/75 个文件后零字节增长（两采样 18 秒判定 FROZEN）。
 * - 改为每次先清空 `release/` 后能跑通（34–39 秒），**但清空本身也会死住**：
 *   同一条 `shutil.rmtree` 有一次 0.7 秒完成、另一次 3 分 56 秒零进展（FROZEN）。
 *
 * 也就是说「递归删除 ~500 MB 的 Electron 二进制树」在这个环境里不可靠。而 rename 是
 * 纯元数据操作，与目录体积无关，瞬时完成；重命名之后 electron-builder 拿到的同样是
 * 一个不存在的 `release/`，因此两个坑一起绕开。
 *
 * 与 `scripts/release.py` 的 `clean_outputs()`、`seed.spec` 里「旧 Tree 条目 / 陈旧 TOC
 * 会被增量复用」那条注释是同一个教训：**陈旧增量产物不得复用**。
 *
 * ## 残留
 *
 * 上一个 `release.stale.*` 会在下次运行时被「尽力」删除一次（失败就留着，绝不因此挂住
 * 构建），并在结束时列出，便于人工清理。
 */

import fs from 'node:fs'
import path from 'node:path'

const projectDir = process.cwd()
const releaseDir = path.join(projectDir, 'release')

/** 尽力删除，失败即放弃 —— 删除绝不能成为构建的关键路径。 */
function bestEffortRemove(target) {
  try {
    fs.rmSync(target, { recursive: true, force: true })
    return true
  } catch {
    return false
  }
}

function staleDirs() {
  return fs
    .readdirSync(projectDir)
    .filter((name) => name.startsWith('release.stale.'))
    .sort()
}

for (const name of staleDirs()) {
  bestEffortRemove(path.join(projectDir, name))
}

if (fs.existsSync(releaseDir)) {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  const stale = path.join(projectDir, `release.stale.${stamp}`)
  fs.renameSync(releaseDir, stale)
  console.log(`[clean-release] release/ 已重命名为 ${path.basename(stale)}（瞬时，不删内容）`)
} else {
  console.log('[clean-release] release/ 不存在，无需处理')
}

const leftovers = staleDirs()
if (leftovers.length > 0) {
  console.log(`[clean-release] 注意：以下陈旧目录未能删除，可在方便时手工清理（约 500 MB/个）:`)
  for (const name of leftovers) console.log(`  ${name}/`)
}
