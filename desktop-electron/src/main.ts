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
  app.commandLine.appendSwitch('disable-gpu')
  app.commandLine.appendSwitch('disable-webgl')
}

const backend = new BackendManager()
const websocket = new WebSocketManager()

let mainWindow: BrowserWindow | null = null
let tray: TrayType | null = null
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

/** main.py: _find_brand_icon() —— 覆盖 frozen / dev 两种布局，找不到再退到根 icon.ico。 */
function findBrandIcon(): string | null {
  const candidates = FROZEN
    ? [
        path.join(process.resourcesPath || ROOT_DIR, 'frontend', 'dist', 'seed-taiji-network.png'),
        path.join(ROOT_DIR, 'frontend', 'dist', 'seed-taiji-network.png'),
        path.join(ROOT_DIR, 'icon.ico'),
      ]
    : [
        path.join(ROOT_DIR, 'frontend', 'public', 'seed-taiji-network.png'),
        path.join(ROOT_DIR, 'frontend', 'dist', 'seed-taiji-network.png'),
        path.join(ROOT_DIR, 'frontend', 'public', 'favicon.ico'),
        path.join(ROOT_DIR, 'icon.ico'),
      ]
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate
  }
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

  // main.py: closeEvent —— 关闭即最小化到托盘；无托盘时才真正退出
  window.on('close', (event) => {
    persistGeometry()
    if (tray !== null) {
      event.preventDefault()
      window.hide()
    }
  })

  window.webContents.on('did-finish-load', () => {
    void window.webContents.insertCSS(DRAG_REGION_CSS)
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
  const needBackend = !backend.isRunning()
  const needWs = !websocket.isRunning()
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
  backend.stop()
  websocket.stop()
})

void app.whenReady().then(bootstrap).catch((cause: unknown) => {
  logger.error('Bootstrap failed', cause)
  app.exit(1)
})
