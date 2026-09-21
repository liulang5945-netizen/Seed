/**
 * 清空 electron-builder 的输出目录。
 *
 * 为什么必须有这一步（实测 2026-09-21）：不清 `release/` 直接重跑，构建会稳定死住 ——
 * 日志停在 `• copying unpacked Electron`，`release/win-unpacked` 拷到 32/75 个文件后
 * 18 秒零字节增长（两采样判定 FROZEN），等 5 分钟无进展；CPU 近零、无残留进程占锁。
 * 清空同一目录后，同一条 `npm run dist` 39 秒跑完。
 *
 * 与 `scripts/release.py` 的 `clean_outputs()`、以及 `desktop/seed.spec` 里
 * 「旧 Tree 条目 / 陈旧 TOC 会被增量复用」那条注释是同一个教训：
 * **陈旧增量产物不得复用**。PyQt6 侧早就有这一步，Electron 侧必须有对等物。
 */

import fs from 'node:fs'

const targets = ['release']

for (const target of targets) {
  fs.rmSync(target, { recursive: true, force: true })
}

console.log(`[clean-release] 已清空: ${targets.join(', ')}`)
