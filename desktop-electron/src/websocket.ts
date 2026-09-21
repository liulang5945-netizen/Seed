/**
 * WebSocket 核心服务器管理（端口 8765）—— desktop/main.py: WebSocketManager 的移植。
 *
 * 为什么不能省：`frontend/src/composables/useWebSocket.js` 连的就是 8765，`App.vue` 在
 * mount 时发起连接并把生命事件路由到 `runtimeStore.handleLifeEvent`。历史发布记录里也有
 * 一条已验证行为——「强杀 Seed.exe → SeedBackend.exe alive=False，8000/8765 全部释放」。
 * 所以这是出货契约的一部分，不是可选装饰。
 *
 * 与 PyQt6 侧的差异：main.py 的 frozen 分支用**进程内守护线程**跑（Python 可以直接 import
 * 该模块），Electron 无法在 Node 进程里跑 Python，因此改为独立子进程 `SeedWs.exe`。
 * 该入口目前在 desktop/seed.spec 里**尚不存在**（只有 Seed.exe / SeedBackend.exe 两个入口），
 * 属本次未闭合项：frozen 分支会明确报错并记日志，不会静默降级。
 */

import { spawn, type ChildProcess } from 'node:child_process'
import * as fs from 'node:fs'
import * as net from 'node:net'
import * as path from 'node:path'
import { FROZEN, ROOT_DIR, WS_PORT } from './config'
import { logger, openChildLog } from './logger'
import { sleep } from './backend'

/** 打包模式下定位 WS 工作进程（seed.spec 待补的第三个入口）。 */
function resolveFrozenWsWorker(): string | null {
  const candidates = [
    path.join(ROOT_DIR, 'SeedWs.exe'),
    path.join(process.resourcesPath || ROOT_DIR, 'SeedWs.exe'),
    path.join(ROOT_DIR, 'resources', 'SeedWs.exe'),
  ]
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate
  }
  return null
}

function resolvePython(): string {
  return process.env.SEED_PYTHON ?? 'python'
}

/** 探测端口是否已有监听（等价于 main.py 里 connect_ex == 0 那一步）。 */
function canConnect(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = net.createConnection({ port, host: '127.0.0.1' })
    const settle = (value: boolean): void => {
      socket.removeAllListeners()
      socket.destroy()
      resolve(value)
    }
    socket.setTimeout(1_000)
    socket.once('connect', () => settle(true))
    socket.once('timeout', () => settle(false))
    socket.once('error', () => settle(false))
  })
}

export class WebSocketManager {
  private process: ChildProcess | null = null
  private running = false
  private logFd: number | null = null

  readonly port = WS_PORT

  isRunning(): boolean {
    return this.running && this.process !== null && this.process.exitCode === null && !this.process.killed
  }

  async start(): Promise<void> {
    if (this.isRunning()) return
    this.running = false // 复位陈旧标志（进程死亡后看门狗重启路径）

    const command = FROZEN ? resolveFrozenWsWorker() : resolvePython()
    if (command === null) {
      logger.error(
        'SeedWs.exe not found：打包模式下 WebSocket 服务器缺少入口。' +
          '需在 desktop/seed.spec 增加第三个入口（seed_ws.py -> SeedWs.exe）后再启用 Electron 出货路径。',
      )
      return
    }
    // 打包侧 desktop/seed_ws.py 接受 [port]；dev 侧走的是模块入口
    // `python -m neuroplex.core.websocket_server`，其 __main__ 固定调用
    // start_server() 默认端口 8765（与 main.py 的 WS_PORT 同源），因此**不要**在
    // dev 下设置 SEED_WS_PORT，否则两侧会不一致。
    const args = FROZEN ? [String(this.port)] : ['-m', 'neuroplex.core.websocket_server']

    try {
      this.logFd = openChildLog('desktop_ws.log')
      const child = spawn(command, args, {
        cwd: ROOT_DIR,
        stdio: ['ignore', this.logFd, this.logFd],
        windowsHide: true,
      })
      this.process = child
      this.running = true
      logger.info(`WebSocket server started on port ${this.port} (PID: ${child.pid})`)

      await this.waitForReady(FROZEN ? 60_000 : 15_000)
    } catch (cause) {
      logger.error('Failed to start WebSocket server', cause)
    }
  }

  /**
   * 等待就绪。
   *
   * main.py 额外用 Toolhelp32 校验「监听者正是自己的子进程」。Node 取不到该信息，改为
   * 「端口可连 + 子进程仍存活」两条件合取：若 8765 被别人占用，我们的子进程会因 bind
   * 失败而退出，第二条即刻为假——main.py 里那条 poll() 判断覆盖的是同一组失效模式。
   */
  private async waitForReady(timeoutMs: number): Promise<boolean> {
    const deadline = Date.now() + timeoutMs
    while (Date.now() < deadline) {
      if (this.process !== null && this.process.exitCode !== null) return false
      if (await canConnect(this.port)) {
        logger.info(`WebSocket server ready on port ${this.port}`)
        return true
      }
      await sleep(500)
    }
    logger.warn('WebSocket server startup timeout')
    return false
  }

  stop(): void {
    if (this.process !== null) {
      try {
        this.process.kill()
        logger.info('WebSocket server stopped')
      } catch (cause) {
        logger.debug('WebSocketManager.stop failed', cause)
      }
      this.process = null
      this.running = false
    }
    this.closeLog()
  }

  private closeLog(): void {
    const fd = this.logFd
    this.logFd = null
    if (fd === null) return
    try {
      fs.closeSync(fd)
    } catch (cause) {
      logger.warn('Failed to close ws log handle', cause)
    }
  }
}
