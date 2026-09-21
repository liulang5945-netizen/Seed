/**
 * 后端进程管理 —— desktop/main.py: BackendManager 的移植。
 *
 * 两处与 PyQt6 侧**有意不同**，都写在下面各自的注释里：
 *  1. Job Object 不可用（Node 无法直接调 Win32），孤儿回收改用「显式持有者记录」，
 *     判定精度不降反升（见 owner record 一节）。
 *  2. main.py 用 _RestartWorker(QThread) 把阻塞式健康探测挪出 GUI 主线程；Node 是
 *     异步的，直接 await 即可，那个线程类整体消失。
 */

import { spawn, type ChildProcess } from 'node:child_process'
import * as fs from 'node:fs'
import * as net from 'node:net'
import * as path from 'node:path'
import {
  BACKEND_HOST,
  FROZEN,
  HEALTH_PATH,
  LOG_DIR,
  ROOT_DIR,
  SWITCH_MODEL_PATH,
} from './config'
import { logger, openChildLog } from './logger'

const OWNER_RECORD_FILE = path.join(LOG_DIR, 'seed_backend.owner.json')

interface OwnerRecord {
  /** 持有该后端的 Electron 主进程 PID —— 它是否存活决定该后端是不是孤儿。 */
  electronPid: number
  backendPid: number
  port: number
  recordedAt: string
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms))

/** PID 是否存活。EPERM 表示存在但无权限，仍算存活。 */
function isPidAlive(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return false
  try {
    process.kill(pid, 0)
    return true
  } catch (cause) {
    return (cause as NodeJS.ErrnoException).code === 'EPERM'
  }
}

/** 端口是否空闲（能 bind 上）。用于确认孤儿被回收后端口真的释放了。 */
function isPortFree(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const probe = net.createServer()
    probe.once('error', () => resolve(false))
    probe.once('listening', () => {
      probe.close(() => resolve(true))
    })
    probe.listen(port, '127.0.0.1')
  })
}

/** 打包模式下定位 SeedBackend.exe（desktop/seed.spec 的第二个入口产物）。 */
function resolveFrozenBackend(): string | null {
  const candidates = [
    path.join(ROOT_DIR, 'SeedBackend.exe'),
    path.join(process.resourcesPath || ROOT_DIR, 'SeedBackend.exe'),
    path.join(ROOT_DIR, 'resources', 'SeedBackend.exe'),
  ]
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate
  }
  return null
}

/** 开发模式下的 Python 解释器。
 *
 * PyQt6 侧用的是「启动 main.py 的那一个解释器」（sys.executable）；Electron 没有这个
 * 语义，因此改为显式环境变量。默认取 PATH 上的 python，本机实际应设为 Python312。
 */
function resolvePython(): string {
  return process.env.SEED_PYTHON ?? 'python'
}

export class BackendManager {
  private process: ChildProcess | null = null
  private running = false
  private logFd: number | null = null

  readonly port = Number(process.env.SEED_PORT ?? 8000)
  readonly host = BACKEND_HOST

  isRunning(): boolean {
    return this.running && this.process !== null && this.process.exitCode === null && !this.process.killed
  }

  async start(): Promise<void> {
    if (this.isRunning()) return
    this.running = false // 复位陈旧标志（进程死亡后看门狗重启路径）
    await this.reapOrphanedBackend()

    if (FROZEN) {
      await this.startFrozen()
      return
    }

    const command = resolvePython()
    const args = [
      '-m',
      'uvicorn',
      'api.app:app',
      '--host',
      this.host,
      '--port',
      String(this.port),
      '--log-level',
      'info',
    ]

    try {
      this.logFd = openChildLog('desktop_backend.log')
      // stdio 必须是文件 fd —— 绝不能用 'pipe'（见 logger.openChildLog 的说明）
      const child = spawn(command, args, {
        cwd: ROOT_DIR,
        stdio: ['ignore', this.logFd, this.logFd],
        // windowsHide 在 Windows 上落到 CREATE_NO_WINDOW，等价 main.py 的 creationflags
        windowsHide: true,
      })
      this.process = child
      this.running = true
      logger.info(`Backend started on port ${this.port} (PID: ${child.pid})`)
      this.writeOwnerRecord(child.pid)

      await this.waitForReady()
      await this.maybeActivateSeed()
    } catch (cause) {
      logger.error('Failed to start backend', cause)
    }
  }

  private async startFrozen(): Promise<void> {
    const worker = resolveFrozenBackend()
    if (worker === null) {
      logger.error(`SeedBackend.exe not found under ${ROOT_DIR}`)
      return
    }

    try {
      this.logFd = openChildLog('desktop_backend.log')
      const child = spawn(worker, [this.host, String(this.port)], {
        cwd: ROOT_DIR,
        stdio: ['ignore', this.logFd, this.logFd],
        windowsHide: true,
      })
      this.process = child
      this.running = true
      logger.info(`Backend worker started on port ${this.port} (PID: ${child.pid})`)
      this.writeOwnerRecord(child.pid)

      // 打包冷启动较慢，给足预算（同 main.py: timeout=120）
      await this.waitForReady(120_000)
      await this.maybeActivateSeed()
    } catch (cause) {
      logger.error('Failed to start backend worker', cause)
    }
  }

  private async maybeActivateSeed(): Promise<void> {
    if (process.env.SEED_RUNTIME !== '1') return
    try {
      const response = await fetch(`http://127.0.0.1:${this.port}${SWITCH_MODEL_PATH}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ checkpoint_id: '' }),
        signal: AbortSignal.timeout(120_000),
      })
      logger.info(`Seed native runtime activated (HTTP ${response.status})`)
    } catch (cause) {
      logger.warn('Seed runtime activation failed', cause)
    }
  }

  private async waitForReady(timeoutMs = 30_000): Promise<boolean> {
    const deadline = Date.now() + timeoutMs
    while (Date.now() < deadline) {
      try {
        const response = await fetch(`http://127.0.0.1:${this.port}${HEALTH_PATH}`, {
          signal: AbortSignal.timeout(2_000),
        })
        if (response.ok) {
          logger.info('Backend is ready')
          return true
        }
      } catch {
        // 未就绪：继续轮询
      }
      await sleep(500)
    }
    logger.warn('Backend startup timeout')
    return false
  }

  /**
   * 回收上一轮遗留的孤儿后端。
   *
   * PyQt6 侧靠内核 Job Object 覆盖「主进程怎么死都回收」，并用 CreateToolhelp32Snapshot
   * 取「端口持有者的父进程是否还在」来判孤儿。Node 两者都拿不到，因此换一条更强的路：
   * **启动子进程时把 (electronPid, backendPid) 显式记到 logs/seed_backend.owner.json**。
   * 于是判定变成三态，且不需要 netstat / tasklist / 任何 Win32 调用：
   *   - 记录里 electronPid 仍存活 ⇒ 那是另一个正在运行的客户端实例，**不碰**；
   *   - electronPid 已死、backendPid 存活 ⇒ 确凿的本产品孤儿，**回收**；
   *   - 两者皆死 / 无记录 ⇒ 无可回收对象，留给后端自己 bind 失败并报错。
   *
   * 已知缺口（双轨期独有）：PyQt6 侧不会写这份记录，所以它留下的 SeedBackend.exe 孤儿
   * Electron 认不出来。反向路径由 PyQt6 自己那套 image+ppid 规则兜底，故任一侧启动都会
   * 自愈；若要彻底闭合，需让两侧共用同一份 owner record（已记入简报待办）。
   */
  private async reapOrphanedBackend(): Promise<void> {
    const record = this.readOwnerRecord()
    if (record === null) return

    if (isPidAlive(record.electronPid)) {
      logger.info(
        `Port ${this.port} is held by a live client instance (Electron PID ${record.electronPid}); leaving it alone`,
      )
      return
    }
    if (!isPidAlive(record.backendPid)) {
      this.clearOwnerRecord()
      return
    }

    logger.warn(`Reaping orphaned backend on port ${this.port} (PID: ${record.backendPid})`)
    try {
      process.kill(record.backendPid)
    } catch (cause) {
      logger.warn(`Failed to reap orphaned backend PID ${record.backendPid}`, cause)
      return
    }
    for (let attempt = 0; attempt < 20; attempt += 1) {
      if (await isPortFree(this.port)) {
        this.clearOwnerRecord()
        return
      }
      await sleep(200)
    }
    logger.warn(`Port ${this.port} still held after reaping PID ${record.backendPid}`)
  }

  private readOwnerRecord(): OwnerRecord | null {
    try {
      if (!fs.existsSync(OWNER_RECORD_FILE)) return null
      const parsed: unknown = JSON.parse(fs.readFileSync(OWNER_RECORD_FILE, 'utf8'))
      if (parsed === null || typeof parsed !== 'object') return null
      const candidate = parsed as Partial<OwnerRecord>
      if (typeof candidate.electronPid !== 'number' || typeof candidate.backendPid !== 'number') {
        return null
      }
      return {
        electronPid: candidate.electronPid,
        backendPid: candidate.backendPid,
        port: typeof candidate.port === 'number' ? candidate.port : this.port,
        recordedAt: typeof candidate.recordedAt === 'string' ? candidate.recordedAt : '',
      }
    } catch (cause) {
      logger.warn('Failed to read backend owner record', cause)
      return null
    }
  }

  private writeOwnerRecord(backendPid: number | undefined): void {
    if (backendPid === undefined) return
    try {
      fs.mkdirSync(LOG_DIR, { recursive: true })
      const record: OwnerRecord = {
        electronPid: process.pid,
        backendPid,
        port: this.port,
        recordedAt: new Date().toISOString(),
      }
      fs.writeFileSync(OWNER_RECORD_FILE, `${JSON.stringify(record, null, 2)}\n`, 'utf8')
    } catch (cause) {
      logger.warn('Failed to write backend owner record', cause)
    }
  }

  private clearOwnerRecord(): void {
    try {
      fs.rmSync(OWNER_RECORD_FILE, { force: true })
    } catch (cause) {
      logger.warn('Failed to clear backend owner record', cause)
    }
  }

  stop(): void {
    if (this.process !== null) {
      try {
        this.process.kill()
        logger.info('Backend stopped')
      } catch (cause) {
        logger.debug('BackendManager.stop failed', cause)
      }
      this.process = null
      this.running = false
    }
    this.clearOwnerRecord()
    this.closeLog()
  }

  private closeLog(): void {
    const fd = this.logFd
    this.logFd = null
    if (fd === null) return
    try {
      fs.closeSync(fd)
    } catch (cause) {
      logger.warn('Failed to close backend log handle', cause)
    }
  }
}

export { isPidAlive, sleep }
