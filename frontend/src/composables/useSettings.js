/**
 * 设置持久化 composable
 * 从 useApi.js 中提取的设置保存/加载逻辑
 */
import { nativeApi } from './nativeApi.js'

let _settingsSaveTimer = null

export function useSettings() {
  const saveSettingsToServer = async (settings) => {
    try {
      await nativeApi.settingsSave(settings)
    } catch (e) { console.warn('[Settings] 保存失败:', e.message) }
  }

  const debouncedSaveSettings = (settings) => {
    if (_settingsSaveTimer) clearTimeout(_settingsSaveTimer)
    _settingsSaveTimer = setTimeout(() => saveSettingsToServer(settings), 2000)
  }

  const loadSettingsFromServer = async () => {
    try {
      return await nativeApi.settingsGet()
    } catch (e) {
      console.warn('[Settings] 服务端加载失败:', e.message)
    }
    return null
  }

  return {
    saveSettingsToServer,
    debouncedSaveSettings,
    loadSettingsFromServer,
  }
}
