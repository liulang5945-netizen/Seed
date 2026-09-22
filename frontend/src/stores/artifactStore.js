/**
 * 工件面板（Artifacts）状态 —— Claude 式侧拉工件视图。
 *
 * 面板内容来自对话工件：助手消息里的代码块经「面板打开」按钮注入 currentArtifact，
 * 面板以 预览/代码 两个视图展示（预览仅 HTML 可渲染，其余类型显示代码）。
 * 档位与宽度持久化沿用原 Dock 机制；IDE 完整功能在独立的 /workspace 路由页。
 */
import { defineStore } from 'pinia'

function readMode() {
  const saved = localStorage.getItem('taiji_dock_mode')
  return saved === 'split' || saved === 'full' ? saved : 'collapsed'
}

function readWidth() {
  const saved = parseInt(localStorage.getItem('taiji_dock_width') || '', 10)
  return Number.isFinite(saved) ? Math.min(960, Math.max(420, saved)) : 560
}

export const useArtifactStore = defineStore('artifact', {
  state: () => ({
    /** collapsed=收起 / split=分栏 / full=全宽（router 区让位） */
    dockMode: readMode(),
    /** 分栏档宽度（px），拖拽调整范围 420–960 */
    dockWidth: readWidth(),
    /** 当前工件：对话驱动的 { title, language, content, kind }，null=无工件 */
    currentArtifact: null,
  }),
  getters: {
    isOpen: (state) => state.dockMode !== 'collapsed' && state.currentArtifact !== null,
    isFull: (state) => state.dockMode === 'full',
  },
  actions: {
    /** 打开工件（对话驱动入口）；首次打开默认分栏 */
    openArtifact(artifact) {
      this.currentArtifact = artifact
      if (this.dockMode === 'collapsed') {
        this.dockMode = 'split'
        this.persistMode()
      }
    },
    closeArtifact() {
      this.currentArtifact = null
      this.dockMode = 'collapsed'
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
      this.dockWidth = Math.min(960, Math.max(420, width))
      localStorage.setItem('taiji_dock_width', String(this.dockWidth))
    },
    persistMode() {
      localStorage.setItem('taiji_dock_mode', this.dockMode)
    },
  },
})
