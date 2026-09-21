/**
 * 工作区 Dock 状态（全局壳层）。
 *
 * Dock 挂在 App.vue 的 .app-body（AppSidebar + router-wrapper 之后的第三个面板），
 * 任何页面都能侧拉出 IDE 工作区。档位与宽度持久化，编辑器内部状态
 * （文件树/打开标签/光标）留在 WorkspaceDock 组件内——单实例常驻无需提升。
 */
import { defineStore } from 'pinia'

function readMode() {
  const saved = localStorage.getItem('taiji_dock_mode')
  return saved === 'split' || saved === 'full' ? saved : 'collapsed'
}

function readWidth() {
  const saved = parseInt(localStorage.getItem('taiji_dock_width') || '', 10)
  return Number.isFinite(saved) ? Math.min(960, Math.max(480, saved)) : 560
}

export const useWorkspaceStore = defineStore('workspace', {
  state: () => ({
    /** collapsed=收起（不占宽） / split=分栏 / full=全宽（router-wrapper 让位） */
    dockMode: readMode(),
    /** 分栏档宽度（px），拖拽调整范围 480–960 */
    dockWidth: readWidth(),
  }),
  getters: {
    isOpen: (state) => state.dockMode !== 'collapsed',
    isFull: (state) => state.dockMode === 'full',
  },
  actions: {
    open(mode = 'split') {
      this.dockMode = mode === 'full' ? 'full' : 'split'
      this.persistMode()
    },
    toggle(mode = 'split') {
      this.dockMode = this.dockMode === 'collapsed' ? (mode === 'full' ? 'full' : 'split') : 'collapsed'
      this.persistMode()
    },
    collapse() {
      this.dockMode = 'collapsed'
      this.persistMode()
    },
    setMode(mode) {
      if (mode === 'collapsed' || mode === 'split' || mode === 'full') {
        this.dockMode = mode
        this.persistMode()
      }
    },
    setWidth(width) {
      this.dockWidth = Math.min(960, Math.max(480, width))
      localStorage.setItem('taiji_dock_width', String(this.dockWidth))
    },
    persistMode() {
      localStorage.setItem('taiji_dock_mode', this.dockMode)
    },
  },
})
