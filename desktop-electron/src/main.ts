/**
 * Seed 桌面客户端（Electron 壳）—— desktop/main.py: main() 的移植。
 *
 * 双轨期声明：desktop/（PyQt6）仍是出货路径，本目录是并行实现，不替代、不改动它。
 *
 * 整体上消失了的三块（都是 Qt/PyInstaller 特有）：
 *  1. main.py 46–113 行的 frozen Qt DLL 战场疤痕（QTWEBENGINEPROCESS_PATH、
 *     _prepare_frozen_qt_dll_path、六个全局 DLL 句柄、--disable-gpu --single-process）
 *     —— Electron 自带 Chromium，无此类加载顺序问题；
 *  2. _EdgeResizeFilter（约 50 行）：Qt 需要应用级事件过滤器才能让无边框窗口可缩放，
 *     Electron 的 frame:false 窗口由系统原生处理边缘缩放；
 *  3. _apply_window_shape（QRegion 圆角遮罩）：改由透明窗口 + 前端 CSS 圆角承担。
 *
 * 反过来新增的两块（Node 拿不到 Win32）：
 *  1. 拖拽：Electron 无 startSystemMove，改用注入 `-webkit-app-region: drag`（主进程侧
 *     注入 CSS，前端零改动），并因此需要对 toggle-maximize 去抖，见 TOGGLE_DEBOUNCE_MS；
 *  2. 桥接自检 verifyBridge()：contextBridge 能否如预期承载「构造器 + 回调内传函数」
 *     是本移植唯一无法离线证明的接口，故加显式自检，失败时留明确日志而不是静默失效。
 */

import * as fs from 'node:fs'
import * as path from 'node:path'
import {
  BrowserWindow,
  Menu,
  Tray,
  app,
  ipcMain,
  nativeImage,
  type IpcMainEvent,
  type Tray as TrayType,
} from 'electron'
import { BackendManager } from './backend'
import {
  APP_USER_MODEL_ID,
  DISABLE_GPU,
  FROZEN,
  ROOT_DIR,
  USER_AGENT,
  buildFrontendUrl,
} from './config'
import { logger } from './logger'
import { type DesktopSettings, type WindowGeometry, loadSettings, saveSettings } from './settings'
import { WebSocketManager } from './websocket'

/** main.py: seed.spec 打包时 cwd 固定到 exe 目录；Electron 侧同理，保证 data/、checkpoints/ 可预期。 */
if (FROZEN) {
  try {
    process.chdir(ROOT_DIR)
  } catch (cause) {
    logger.debug('chdir to install root failed', cause)
  }
}

/** main.py: _set_windows_app_identity() —— 必须在任何窗口创建之前调用。 */
app.setAppUserModelId(APP_USER_MODEL_ID)
if (DISABLE_GPU) {
  // 实测（2026-09-21，所有者机器）：只给 --disable-gpu 不够 —— Chromium 仍会起一个
  // 软件合成的 GPU 进程，而在受限 VM 上它会被自身沙箱挡住，直接
  //   FATAL: GPU process isn't usable. Goodbye.
  // 四件套（no-sandbox + disable-gpu + disable-gpu-compositing + in-process-gpu）
  // 是多次验证过能跑通的组合。--no-sandbox 属安全降级，但本应用只加载
  // 127.0.0.1 的本地内容，且 PyQt6 出货版的 --single-process 本就是同级取舍。
  // 正常机器上用 SEED_DISABLE_GPU=0 可恢复完整沙箱 + GPU。
  app.commandLine.appendSwitch('no-sandbox')
  app.commandLine.appendSwitch('disable-gpu')
  app.commandLine.appendSwitch('disable-gpu-compositing')
  app.commandLine.appendSwitch('in-process-gpu')
  app.commandLine.appendSwitch('disable-webgl')
}

const backend = new BackendManager()
const websocket = new WebSocketManager()

let mainWindow: BrowserWindow | null = null
let tray: TrayType | null = null
/**
 * 真退出标志：quit() 与任何 app.quit() 来源（before-quit 统一置位）都会置 true，
 * 窗口 close 拦截据此放行。没有它，close→preventDefault→hide 会把 app.quit()
 * 的窗口关闭步骤拦死 ⇒ 托盘「退出」表现为「点了没反应」（2026-09-21 所有者实测
 * 症状，SEED_TRAY_SMOKE=1 红跑 exit 5 复现、修复后 exit 0 闭合）。
 */
let quitting = false
let settings: DesktopSettings = {}
let frontendLoaded = false
let statusTimer: NodeJS.Timeout | null = null
let restarting = false

/**
 * 双击最大化去抖。
 *
 * 注入 app-region: drag 之后，Windows 会把标题栏双击当作「最大化/还原」原生处理，
 * 而前端 AppTitlebar.vue 的 @dblclick 也会走桥调用 toggleMaximize()——两者叠加会互相
 * 抵消，表现为「双击没反应」。这里对 toggle-maximize 做 300ms 去抖，让一次物理双击只
 * 生效一次。**这是移植引入的新逻辑，PyQt6 侧没有对应物。**
 */
const TOGGLE_DEBOUNCE_MS = 300
let lastToggleAt = 0

/**
 * 无边框窗口的拖拽区域。
 *
 * 主进程侧注入，因此不需要改 AppTitlebar.vue（本次迁移硬约束是零 .vue 改动）。
 * 按钮必须显式 no-drag，否则点不动。
 */
const DRAG_REGION_CSS = `
.app-titlebar { -webkit-app-region: drag; }
.app-titlebar button,
.titlebar-btn,
.titlebar-window-btn { -webkit-app-region: no-drag; }
`

/**
 * 无边框窗口不应出现滚动条。
 *
 * 所有者实测截图：占位屏（以及部分页面）撑出 body 滚动条，在 `transparent: true` 的
 * 透明窗口上表现为右侧一条竖向长条，视觉上像「边上多了根柱子」。根因是页面自身
 * `body` 有默认 margin 且内容超高。这里从壳侧统一压掉，不需要改任何 .vue。
 */
const NO_SCROLLBAR_CSS = `
html, body { margin: 0; padding: 0; overflow: hidden; height: 100%; }
`

/**
 * main.py: _find_brand_icon() —— 覆盖 frozen / dev 两种布局。
 *
 * 实测订正：frozen 首次跑挂在这里，日志是 `Brand icon not found; tray disabled`。
 * 原因是候选路径沿用了 Electron 的 `process.resourcesPath` 概念，而本项目 frozen 布局是
 * PyInstaller 的 onedir —— datas 落在 `_internal/` 下：
 *   dist/Seed/_internal/frontend/dist/seed-taiji-network.png
 *   dist/Seed/_internal/icon.ico
 * 两种布局都保留探测（electron-builder 打包时资源会在 resources/），并记录候选数，
 * 使再次落空时是可诊断的告警而不是静默禁用托盘。
 */
let brandIconCache: string | null | undefined

function findBrandIcon(): string | null {
  // 结果缓存：本函数被 createWindow（setIcon）与 createTray 各调一次，且日志里的
  // `Brand icon not found` 会随之重复。实测打包版一次启动打出 3 条同样的 WARN，
  // 属于纯噪声。缓存后每条启动只告警一次。
  if (brandIconCache !== undefined) return brandIconCache

  const relativeCandidates = FROZEN
    ? [
        path.join('frontend', 'dist', 'seed-taiji-network.png'),
        path.join('frontend', 'dist', 'favicon.ico'),
        'icon.ico',
      ]
    : [
        path.join('frontend', 'public', 'seed-taiji-network.png'),
        path.join('frontend', 'dist', 'seed-taiji-network.png'),
        path.join('frontend', 'public', 'favicon.ico'),
        'icon.ico',
      ]

  const roots = FROZEN
    ? [path.join(ROOT_DIR, '_internal'), process.resourcesPath, ROOT_DIR].filter(
        (root): root is string => typeof root === 'string' && root.length > 0,
      )
    : [ROOT_DIR]

  const candidates = roots.flatMap((root) =>
    relativeCandidates.map((relative) => path.join(root, relative)),
  )
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      brandIconCache = candidate
      return candidate
    }
  }
  logger.warn(`Brand icon not found; 窗口与托盘图标将缺省。已探测 ${candidates.length} 个路径`)
  brandIconCache = null
  return null
}

/** main.py: SeedWindow._sync_window_state() —— 把最大化状态写回 DOM 供 CSS/按钮联动。 */
function syncWindowState(): void {
  if (mainWindow === null || mainWindow.isDestroyed() || !frontendLoaded) return
  const flag = mainWindow.isMaximized() ? 'true' : 'false'
  mainWindow.webContents
    .executeJavaScript(`document.documentElement.setAttribute('data-maximized','${flag}')`)
    .catch(() => undefined)
}

/**
 * 桥接自检：本移植唯一无法离线证明的接口。
 * 失败只记 error，不阻断启动——前端在没有桥时会自动隐藏三个窗口按钮（其既有降级路径）。
 */
function verifyBridge(): void {
  if (mainWindow === null || mainWindow.isDestroyed()) return
  mainWindow.webContents
    .executeJavaScript(
      `JSON.stringify({
         transport: !!window.qt && !!window.qt.webChannelTransport,
         ctor: typeof window.QWebChannel === 'function'
       })`,
    )
    .then((raw: string) => {
      const parsed = JSON.parse(raw) as { transport: boolean; ctor: boolean }
      if (parsed.transport && parsed.ctor) {
        logger.info('Window bridge self-check passed (qt.webChannelTransport + QWebChannel present)')
      } else {
        logger.error(
          `Window bridge self-check FAILED: webChannelTransport=${parsed.transport} QWebChannel=${parsed.ctor}。` +
            '标题栏的最小化/最大化/关闭按钮将不可用（前端会静默隐藏它们），需检查 preload 的 contextBridge 暴露方式。',
        )
      }
    })
    .catch((cause: unknown) => logger.warn('Window bridge self-check threw', cause))
}

/** main.py: _WindowBridge._run_js 的等价物：托盘「生命状态」切到 /life。 */
function showLifeStatus(): void {
  showWindow()
  mainWindow?.webContents.executeJavaScript("location.hash='/life'").catch(() => undefined)
}

function showWindow(): void {
  if (mainWindow === null || mainWindow.isDestroyed()) return
  if (mainWindow.isMinimized()) mainWindow.restore()
  mainWindow.show()
  mainWindow.focus()
}

function currentGeometry(): WindowGeometry | null {
  if (mainWindow === null || mainWindow.isDestroyed()) return null
  const bounds = mainWindow.getBounds()
  return { x: bounds.x, y: bounds.y, width: bounds.width, height: bounds.height }
}

/** main.py: closeEvent 的几何持久化部分。 */
function persistGeometry(): void {
  const geometry = currentGeometry()
  if (geometry === null) return
  settings = { ...settings, geometry }
  saveSettings(settings)
}

function createWindow(): BrowserWindow {
  const geometry = settings.geometry
  const window = new BrowserWindow({
    x: geometry?.x,
    y: geometry?.y,
    width: geometry?.width ?? 1280,
    height: geometry?.height ?? 800,
    minWidth: 1024,
    minHeight: 700,
    title: 'Seed - AI 生命体',
    // main.py: FramelessWindowHint + WA_TranslucentBackground，标题栏由前端 AppTitlebar.vue 绘制
    frame: false,
    transparent: true,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      // main.py 关掉了 WebGL（QWebEngine 在受限桌面的 renderer 卡死规避）。
      // Electron 的 GPU 路径正是本次迁移的收益之一，故默认保留；用 SEED_DISABLE_GPU=1
      // 可复现 QWebEngine 的纯 CPU 基线，便于双轨对照。
      webgl: !DISABLE_GPU,
    },
  })

  if (!geometry) window.center()
  const icon = findBrandIcon()
  if (icon !== null) window.setIcon(icon)

  window.once('ready-to-show', () => {
    window.show()
  })

  window.on('maximize', syncWindowState)
  window.on('unmaximize', syncWindowState)
  window.on('resize', syncWindowState)
  window.on('enter-full-screen', syncWindowState)
  window.on('leave-full-screen', syncWindowState)

  // main.py: closeEvent —— 关闭即最小化到托盘；无托盘或真退出（quitting）时放行
  window.on('close', (event) => {
    persistGeometry()
    if (tray !== null && !quitting) {
      event.preventDefault()
      window.hide()
    }
  })

  window.webContents.on('did-finish-load', () => {
    void window.webContents.insertCSS(DRAG_REGION_CSS)
    void window.webContents.insertCSS(NO_SCROLLBAR_CSS)
    if (frontendLoaded) return
    frontendLoaded = true
    logger.info('Frontend loaded successfully')
    verifyBridge()
    void window.webContents.executeJavaScript(`
      window.onerror = function(msg, url, line, col, error) {
        console.error('JS Error:', msg, 'at', url, ':', line);
        return false;
      };
      setTimeout(() => {
        const app = document.getElementById('app');
        if (app && app.children.length === 0) {
          document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;background:#0d1117;color:#e2e8f0;font-family:sans-serif;"><div style="text-align:center;"><h1 style="font-size:48px;margin-bottom:16px;">Seed</h1><p style="color:#94a3b8;margin-top:8px;">界面加载中，请稍候...</p></div></div>';
        }
      }, 3000);
    `)
    syncWindowState()
  })

  window.webContents.on('did-fail-load', (_event, errorCode, errorDescription, validatedURL, isMainFrame) => {
    if (!isMainFrame) return
    logger.warn(`Page load failed: ${validatedURL} (${errorCode} ${errorDescription})`)
    frontendLoaded = false
    const html = `<html><body style="background:#0d1117;color:#e2e8f0;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;"><div style="text-align:center;"><h2>加载失败</h2><p style="color:#94a3b8;">无法加载页面: ${validatedURL}</p><p style="color:#64748b;font-size:12px;margin-top:16px;">请检查后端服务是否运行</p></div></body></html>`
    void window.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(html)}`)
    setTimeout(() => void loadFrontend(), 5_000)
  })

  return window
}

/** main.py: SeedWindow._load_frontend() —— 后端未就绪就 1s 后重试。 */
async function loadFrontend(): Promise<void> {
  if (mainWindow === null || mainWindow.isDestroyed() || frontendLoaded) return
  if (!backend.isRunning()) {
    setTimeout(() => void loadFrontend(), 1_000)
    return
  }
  const url = buildFrontendUrl(backend.port)
  logger.info(`Loading frontend: ${url}`)
  try {
    await mainWindow.loadURL(url, { userAgent: USER_AGENT })
  } catch (cause) {
    logger.debug('loadURL rejected (did-fail-load 会接管)', cause)
  }
}

/** main.py: SeedWindow._create_tray() */
function createTray(): void {
  const icon = findBrandIcon()
  if (icon === null) {
    logger.warn('Brand icon not found; tray disabled')
    return
  }
  tray = new Tray(nativeImage.createFromPath(icon))
  // main.py: 关闭窗口后不弹系统气泡，恢复方式由 tooltip 承载
  tray.setToolTip('Seed — 双击图标恢复窗口')
  tray.setContextMenu(
    Menu.buildFromTemplate([
      { label: '显示窗口', click: showWindow },
      { type: 'separator' },
      { label: '生命状态', click: showLifeStatus },
      { type: 'separator' },
      { label: '退出', click: quit },
    ]),
  )
  // main.py: ActivationReason.Trigger == 单击
  tray.on('click', showWindow)
}

function quit(): void {
  quitting = true
  backend.stop()
  websocket.stop()
  if (statusTimer !== null) {
    clearInterval(statusTimer)
    statusTimer = null
  }
  app.quit()
}

/** main.py: _check_backend + _RestartWorker —— Node 是异步的，线程类整体消失。 */
async function checkBackend(): Promise<void> {
  if (restarting) return

  // 入口缺失是永久性状况（打包产物里就没有这个文件），重试只会每 10 秒刷一遍同样的
  // ERROR。实测于 release/win-unpacked：壳-only 的 Electron 包会无限刷错误日志。
  // 首次已明确报过错，这里停手，并停掉定时器本身。
  const backendMissing = backend.isArtifactMissing()
  const wsMissing = websocket.isArtifactMissing()
  if (backendMissing && wsMissing) {
    if (statusTimer !== null) {
      clearInterval(statusTimer)
      statusTimer = null
    }
    logger.error('打包产物缺少 Python 侧载荷（后端与 WebSocket 入口均缺失），已停止看门狗重试。')
    return
  }

  const needBackend = !backend.isRunning() && !backendMissing
  const needWs = !websocket.isRunning() && !wsMissing
  if (!needBackend && !needWs) return
  restarting = true
  try {
    if (needBackend) {
      logger.warn('Backend stopped, restarting...')
      await backend.start()
    }
    if (needWs) {
      logger.warn('WebSocket server stopped, restarting...')
      await websocket.start()
    }
  } catch (cause) {
    logger.error('Watchdog restart failed', cause)
  } finally {
    restarting = false
  }
}

function registerIpc(): void {
  ipcMain.on('seed:window', (event: IpcMainEvent, command: unknown) => {
    if (mainWindow === null || mainWindow.isDestroyed()) return
    switch (command) {
      case 'minimize':
        mainWindow.minimize()
        break
      case 'toggle-maximize':
        if (Date.now() - lastToggleAt < TOGGLE_DEBOUNCE_MS) break
        lastToggleAt = Date.now()
        if (mainWindow.isMaximized()) mainWindow.unmaximize()
        else mainWindow.maximize()
        break
      case 'close':
        mainWindow.close()
        break
      case 'start-drag':
        // 拖拽由注入的 -webkit-app-region: drag 原生承担（Electron 无 startSystemMove）。
        break
      default:
        logger.debug('Unknown window command from renderer', command)
        break
    }
    event.returnValue = undefined
  })

  ipcMain.on('seed:window:is-maximized', (event: IpcMainEvent) => {
    event.returnValue = mainWindow !== null && !mainWindow.isDestroyed() && mainWindow.isMaximized()
  })
}

async function bootstrap(): Promise<void> {
  settings = loadSettings()

  // main.py: 先起后端与 WS，再建窗（start() 内含健康探测）
  await backend.start()
  await websocket.start()

  mainWindow = createWindow()
  createTray()
  registerIpc()

  statusTimer = setInterval(() => void checkBackend(), 10_000)
  await loadFrontend()
}

app.on('window-all-closed', () => {
  // 有托盘时不因窗口全关而退出（与 PyQt6 的关闭到托盘一致）；无托盘才退出
  if (tray === null) quit()
})

app.on('before-quit', () => {
  quitting = true
  backend.stop()
  websocket.stop()
})

void app.whenReady().then(bootstrap).catch((cause: unknown) => {
  logger.error('Bootstrap failed', cause)
  app.exit(1)
})

// ---- 托盘退出链路 smoke（SEED_TRAY_SMOKE=1 启用；验证用接缝，与 config.ts 的
// SEED_FORCE_FROZEN 同风格，出货路径不受影响）。----
// 步骤①：窗口 close —— 预期被拦截为 hide、进程存活（关闭到托盘行为）；
// 步骤②：调用 quit() —— 预期进程在 10s 内真退出（exit 0）。
// 红绿判据：close 拦截若无 quitting 放行，quit() 会被窗口 close→preventDefault
// 拦死 ⇒ 进程不退出，10s 后 exit 5 —— 即「点退出没反应」的可复现红。
if (process.env.SEED_TRAY_SMOKE === '1') {
  void app.whenReady().then(() => {
    setTimeout(() => {
      if (mainWindow === null) {
        logger.error('SMOKE-FAIL: window missing')
        app.exit(3)
        return
      }
      mainWindow.close()
      setTimeout(() => {
        if (mainWindow !== null && !mainWindow.isDestroyed() && mainWindow.isVisible()) {
          logger.error('SMOKE-FAIL: window still visible after close（hide 拦截失效）')
          app.exit(4)
          return
        }
        logger.info('SMOKE-STEP1-OK: close→hide, process alive')
        setTimeout(() => {
          logger.info('SMOKE-STEP2: invoking quit()')
          quit()
          setTimeout(() => {
            logger.error('SMOKE-FAIL: process alive 10s after quit() —— 「点退出没反应」复现')
            app.exit(5)
          }, 10_000)
        }, 1_000)
      }, 1_000)
    }, 15_000)
  })
}
