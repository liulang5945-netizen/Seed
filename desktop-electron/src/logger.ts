/**
 * 日志 —— 与 desktop/main.py 的 logging 配置对齐。
 *
 * main.py 的注释点出一个关键约束：打包（windowed）模式没有 stderr，必须同时写文件，
 * 否则 frozen 启动卡死类问题根本无从排查。Electron 打包后同理，因此文件 sink 是主通道。
 */

import * as fs from 'node:fs'
import * as path from 'node:path'
import { LOG_DIR } from './config'

export type LogLevel = 'debug' | 'info' | 'warn' | 'error'

let sink: fs.WriteStream | null = null
let sinkReady = false

function ensureSink(): fs.WriteStream | null {
  if (sinkReady) return sink
  sinkReady = true
  try {
    fs.mkdirSync(LOG_DIR, { recursive: true })
    sink = fs.createWriteStream(path.join(LOG_DIR, 'desktop_main.log'), { flags: 'a' })
    sink.on('error', () => {
      // 写日志失败不能反过来打断客户端；断开 sink 后退化为仅控制台。
      sink = null
    })
  } catch (cause) {
    sink = null
    writeConsole(`[SeedDesktop] 日志文件不可用，退化为仅控制台输出：${String(cause)}`)
  }
  return sink
}

function stamp(): string {
  // 与 PyQt6 侧 desktop_main.log 保持同构前缀，双轨期可直接并排比对
  return new Date().toISOString().replace('T', ' ').slice(0, 23)
}

function describe(detail: unknown): string {
  if (detail === undefined) return ''
  if (detail instanceof Error) return ` :: ${detail.stack ?? detail.message}`
  if (typeof detail === 'string') return ` :: ${detail}`
  try {
    return ` :: ${JSON.stringify(detail)}`
  } catch {
    return ` :: ${String(detail)}`
  }
}

function writeConsole(line: string): void {
  try {
    process.stderr.write(line + '\n')
  } catch {
    // 打包（windowed）模式下 stdout/stderr 可能为 null，属预期
  }
}

export function log(level: LogLevel, message: string, detail?: unknown): void {
  const line = `${stamp()} - SeedDesktop - ${level.toUpperCase()} - ${message}${describe(detail)}`
  ensureSink()?.write(line + '\n')
  // main.py 保留 debug 级不进控制台；这里同样只放行 info 以上
  if (level !== 'debug') writeConsole(line)
}

export const logger = {
  debug: (message: string, detail?: unknown): void => log('debug', message, detail),
  info: (message: string, detail?: unknown): void => log('info', message, detail),
  warn: (message: string, detail?: unknown): void => log('warn', message, detail),
  error: (message: string, detail?: unknown): void => log('error', message, detail),
}

/** 子进程输出重定向用的追加句柄。
 *
 * main.py 明确不用 PIPE：无人消费时 Windows 下约 4KB 即写满，子进程（uvicorn 启动
 * 日志很快超限）会阻塞挂起，看门狗误判为死亡并反复重启。Node 侧同样必须给文件 fd，
 * 绝不能用 'pipe'。
 */
export function openChildLog(name: string): number {
  fs.mkdirSync(LOG_DIR, { recursive: true })
  return fs.openSync(path.join(LOG_DIR, name), 'a')
}
