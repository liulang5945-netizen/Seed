/**
 * 窗口设置持久化 —— main.py: load_settings() / save_settings()。
 *
 * 刻意复用同一个 `desktop/settings.json`：双轨期在 PyQt6 客户端里调好的窗口位置，
 * 换 Electron 启动应当原位恢复；反之亦然。写失败只记日志，不影响退出流程（同 main.py）。
 */

import * as fs from 'node:fs'
import * as path from 'node:path'
import { SETTINGS_FILE } from './config'
import { logger } from './logger'

export interface WindowGeometry {
  x: number
  y: number
  width: number
  height: number
}

export interface DesktopSettings {
  geometry?: WindowGeometry
}

export function loadSettings(): DesktopSettings {
  try {
    if (!fs.existsSync(SETTINGS_FILE)) return {}
    const parsed: unknown = JSON.parse(fs.readFileSync(SETTINGS_FILE, 'utf8'))
    if (parsed !== null && typeof parsed === 'object') return parsed as DesktopSettings
    return {}
  } catch (cause) {
    logger.warn('Failed to load settings, using defaults', cause)
    return {}
  }
}

export function saveSettings(settings: DesktopSettings): void {
  try {
    fs.mkdirSync(path.dirname(SETTINGS_FILE), { recursive: true })
    fs.writeFileSync(SETTINGS_FILE, `${JSON.stringify(settings, null, 2)}\n`, 'utf8')
  } catch (cause) {
    logger.warn('Failed to save settings', cause)
  }
}
