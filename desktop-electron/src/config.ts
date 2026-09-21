/**
 * 端口 / 路径 / 环境契约 —— 与 desktop/main.py 的对应项逐一对照移植。
 *
 * 双轨期纪律：这里的每个常量都必须能在 PyQt6 侧找到同源项（注释标出）。
 * 任一端口、路径、环境变量在两侧不一致，就是双轨期最危险的一类漂移。
 */

import * as path from 'node:path'
import { app } from 'electron'

/**
 * main.py: FROZEN = getattr(sys, "frozen", False)
 *
 * `SEED_FORCE_FROZEN=1` 是本移植新增的**验证用接缝**：SeedBackend.exe / SeedWs.exe 只有
 * 在 packaged 状态下才会被走到，而 electron-builder 打一次包成本很高。该开关允许直接在
 * 源码树上跑 frozen 分支、对真实 PyInstaller 产物做端到端验证。默认关闭，出货路径不受影响。
 */
export const FROZEN = app.isPackaged || process.env.SEED_FORCE_FROZEN === '1'

/**
 * main.py: ROOT_DIR = exe 目录（frozen）/ 仓库根（开发）。
 * 开发态 __dirname 为 <repo>/desktop-electron/dist ⇒ 上溯两级 = <repo>。
 *
 * `SEED_ROOT_DIR` 与 SEED_FORCE_FROZEN 配套：源码树上跑 frozen 分支时，exe 是
 * node_modules 里的 electron.exe，`path.dirname(app.getPath('exe'))` 指向错误位置，
 * 必须显式指定产物根目录（即 PyInstaller 的 dist/Seed/）。
 */
export const ROOT_DIR =
  process.env.SEED_ROOT_DIR ??
  (FROZEN ? path.dirname(app.getPath('exe')) : path.resolve(__dirname, '..', '..'))

/** main.py: SETTINGS_FILE = ROOT_DIR / "desktop" / "settings.json" —— 刻意复用同一份文件。 */
export const SETTINGS_FILE = path.join(ROOT_DIR, 'desktop', 'settings.json')

/** main.py: LOG_DIR = ROOT_DIR / "logs" */
export const LOG_DIR = path.join(ROOT_DIR, 'logs')

/** main.py: BACKEND_PORT / WS_PORT / HEALTH_PATH / SWITCH_MODEL_PATH */
export const BACKEND_PORT = Number(process.env.SEED_PORT ?? 8000)
export const WS_PORT = Number(process.env.SEED_WS_PORT ?? 8765)
export const HEALTH_PATH = '/api/health'
export const SWITCH_MODEL_PATH = '/api/runtime/activate'

/** main.py: self.host = os.environ.get("SEED_HOST", "127.0.0.1") */
export const BACKEND_HOST = process.env.SEED_HOST ?? '127.0.0.1'

/** main.py: profile.setHttpUserAgent("SeedDesktop/1.6.0") */
export const USER_AGENT = 'SeedDesktop/1.6.0'

/** main.py: APP_USER_MODEL_ID —— 决定 Windows 通知/任务栏归属，必须在建窗前设置。 */
export const APP_USER_MODEL_ID = 'Seed.Desktop.Shell'

/** main.py: ORPHAN_BACKEND_IMAGE —— 只存在于本产品包内的映像名，用于孤儿判定。 */
export const ORPHAN_BACKEND_IMAGE = 'SeedBackend.exe'

/**
 * main.py 在导入 PyQt6 之前设 `QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu --single-process`
 * 以规避受限桌面下的 renderer 卡死。Electron 的 Chromium 无同类冻结问题，默认不开；
 * 仍保留显式开关，便于在同样受限的机器上复现该降级。
 */
export const DISABLE_GPU = process.env.SEED_DISABLE_GPU === '1'

/** main.py: build_frontend_url() */
export function buildFrontendUrl(port: number): string {
  return `http://127.0.0.1:${port}/#/?taiji_client=desktop`
}
