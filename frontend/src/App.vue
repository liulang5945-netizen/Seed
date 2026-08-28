<template>
  <n-config-provider :theme="naiveTheme" :theme-overrides="themeOverrides">
    <n-loading-bar-provider>
      <n-dialog-provider>
        <n-notification-provider>
          <n-message-provider>
            <div class="app-wrapper" @dragenter="onDragEnter" @dragleave="onDragLeave" @dragover="onDragOver" @drop="onDrop">
              <ToastManager ref="toastRef" />
              <ConfirmDialog ref="confirmRef" />
              <RuntimeExceptionCenter />

              <!-- === Titlebar（自绘，与下方共享同一背景宿主） === -->
              <AppTitlebar :collapsed="sidebarCollapsed" @toggle-sidebar="toggleSidebar" />

              <div class="app-body" :class="{ 'sidebar-collapsed': sidebarCollapsed }">
                <!-- === Sidebar === -->
                <AppSidebar 
                  :width="sidebarWidth" 
                  :is-resizing="isResizing"
                  @resize-start="onSidebarResizeStart" 
                />

                <!-- === Router View === -->
                <div class="router-wrapper">
                  <RouteErrorView v-if="routeError" :message="routeError" />
                  <router-view v-else v-slot="{ Component }">
                    <transition name="route" mode="out-in">
                      <keep-alive
                        :include="['ChatView', 'TrainingView', 'WorkspaceView', 'KBView', 'AgentConfigView', 'LifeStatusView']"
                        :max="6"
                      >
                        <component :is="Component" />
                      </keep-alive>
                    </transition>
                  </router-view>
                </div>
              </div>

              <div
                v-if="dragOver"
                class="global-drag-overlay"
                @click="clearDragState"
                @dragenter.stop
                @dragover.prevent
                @dragleave="onDragLeave"
                @drop.prevent.stop="onDrop"
              >
                <div class="drag-overlay-panel">
                  <UploadCloud :size="36" />
                  <p>{{ appStore.t('drop_release') }}</p>
                </div>
              </div>
            </div>
          </n-message-provider>
        </n-notification-provider>
      </n-dialog-provider>
    </n-loading-bar-provider>
  </n-config-provider>
</template>

<script setup>
import { ref, computed, onErrorCaptured, onMounted, onUnmounted, provide } from 'vue'
import { darkTheme, lightTheme } from 'naive-ui'
import ToastManager from './components/ToastManager.vue'
import ConfirmDialog from './components/ConfirmDialog.vue'
import RuntimeExceptionCenter from './components/RuntimeExceptionCenter.vue'
import AppSidebar from './components/AppSidebar.vue'
import AppTitlebar from './components/AppTitlebar.vue'
import RouteErrorView from './components/RouteErrorView.vue'
import { UploadCloud } from 'lucide-vue-next'
import { useAppStore } from './stores/appStore.js'
import { useChatStore } from './stores/chatStore.js'
import { useRuntimeStore } from './stores/runtimeStore.js'
import { useApi } from './composables/useApi.js'
import { useWebSocket } from './composables/useWebSocket.js'
import { API_BASE, authFetch } from './composables/apiClient.js'
import { loadCheckpoints, trainAbortController } from './composables/useTraining.js'
import router from './router'

const appStore = useAppStore()
const chatStore = useChatStore()
const runtimeStore = useRuntimeStore()
const routeError = ref('')

// WebSocket 实时通道（8765）：连接由 useWebSocket 在 mount 时自动发起，
// 失败时其内部有 6 次重连上限并优雅降级，不影响主路径（HTTP 轮询照常工作）。
const { on: onWsMessage } = useWebSocket()

// Naive UI 主题
const naiveTheme = computed(() => {
  return appStore.resolvedTheme === 'light' ? lightTheme : darkTheme
})

const themeOverrides = computed(() => {
  // 5 套主题的主色映射（与 themes.css 保持一致）
  const themePrimaryColors = {
    classic: '#0065fd',
    dark: '#4d8cff',
    teal: '#0d9488',
    violet: '#7c3aed',
    warm: '#ea580c',
  }
  // 自定义 accent 优先，否则用当前主题的主色
  const dt = appStore.resolvedDataTheme
  const primary = appStore.currentAccent || themePrimaryColors[dt] || '#0065fd'
  return {
    common: {
      primaryColor: primary,
      primaryColorHover: primary + 'cc',
      primaryColorPressed: primary + 'aa',
      primaryColorSuppl: primary + '88',
      borderRadius: '12px',
      borderRadiusSmall: '8px',
      fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif',
    },
    Button: {
      borderRadiusMedium: '12px',
      borderRadiusSmall: '8px',
    },
    Input: {
      borderRadius: '12px',
    },
    Card: {
      borderRadius: '20px',
    },
    Dialog: {
      borderRadius: '20px',
    },
    Notification: {
      borderRadius: '12px',
    },
  }
})

// Toast & Confirm
const toastRef = ref(null)
const confirmRef = ref(null)
const toast = (msg, type = 'info') => { if (toastRef.value) toastRef.value.showToast(msg, type) }
const $confirm = (options) => confirmRef.value ? confirmRef.value.show(options) : Promise.resolve(false)
provide('toast', toast)
provide('$confirm', $confirm)

// API connection
const { startHealthCheck, stopHealthCheck } = useApi()

// 侧边栏宽度调整
const sidebarWidth = ref(parseInt(localStorage.getItem('taiji_sidebar_width') || '248'))
const isResizing = ref(false)

// 侧边栏收起：由自绘标题栏的第一个按钮驱动，状态与宽度同样持久化
const sidebarCollapsed = ref(localStorage.getItem('taiji_sidebar_collapsed') === '1')

function toggleSidebar() {
  sidebarCollapsed.value = !sidebarCollapsed.value
  localStorage.setItem('taiji_sidebar_collapsed', sidebarCollapsed.value ? '1' : '0')
}

function onSidebarResizeStart(event) {
  event.preventDefault()
  isResizing.value = true
  const startX = event.clientX
  const startWidth = sidebarWidth.value
  
  function onMouseMove(e) {
    const newWidth = Math.min(400, Math.max(200, startWidth + (e.clientX - startX)))
    sidebarWidth.value = newWidth
    document.documentElement.style.setProperty('--sidebar-width', newWidth + 'px')
  }
  
  function onMouseUp() {
    isResizing.value = false
    localStorage.setItem('taiji_sidebar_width', sidebarWidth.value.toString())
    document.removeEventListener('mousemove', onMouseMove)
    document.removeEventListener('mouseup', onMouseUp)
  }
  
  document.addEventListener('mousemove', onMouseMove)
  document.addEventListener('mouseup', onMouseUp)
}

// 恢复侧边栏宽度
onMounted(() => {
  const savedWidth = localStorage.getItem('taiji_sidebar_width')
  if (savedWidth) {
    document.documentElement.style.setProperty('--sidebar-width', savedWidth + 'px')
  }
})

function onRouteError(error) {
  routeError.value = error?.message || '页面加载失败'
}

router.onError((error) => {
  onRouteError(error)
})

router.beforeEach(() => {
  routeError.value = ''
  clearDragState()
})

router.afterEach(() => {
  routeError.value = ''
})

// return false 会阻止错误继续冒泡到 window，若不主动打日志就完全静默，
// 渲染期异常（如非法标签名）将只表现为白屏而无任何可诊断线索。
onErrorCaptured((error, instance, info) => {
  console.error('[App] 捕获渲染错误:', info, error)
  onRouteError(error)
  return false
})

// Drag
const dragOver = ref(false)
let dragCounter = 0
let dragResetTimer = null
function isFileDrag(event) {
  return Array.from(event.dataTransfer?.types || []).includes('Files')
}
function clearDragState() {
  dragCounter = 0
  dragOver.value = false
  if (dragResetTimer) {
    clearTimeout(dragResetTimer)
    dragResetTimer = null
  }
}
function scheduleDragReset() {
  if (dragResetTimer) clearTimeout(dragResetTimer)
  dragResetTimer = setTimeout(clearDragState, 1200)
}
const onDragEnter = (event) => {
  if (!isFileDrag(event)) return
  event.preventDefault()
  dragCounter++
  dragOver.value = true
  scheduleDragReset()
}
const onDragOver = (event) => {
  if (!isFileDrag(event)) return
  event.preventDefault()
  dragOver.value = true
  scheduleDragReset()
}
const onDragLeave = (event) => {
  if (!isFileDrag(event)) return
  dragCounter--
  if (dragCounter <= 0) clearDragState()
}
const onDrop = (event) => {
  if (isFileDrag(event)) event.preventDefault()
  clearDragState()
}

// Lifecycle
onMounted(async () => {
  window.addEventListener('blur', clearDragState)
  window.addEventListener('keyup', onGlobalKeyup)
  try {
    const r = await authFetch(`${API_BASE}/api/settings`);
    if (r.ok) {
      const saved = await r.json();
      if (saved && typeof saved === 'object') {
        for (const [key, value] of Object.entries(saved)) {
          const storageKey = `taiji_${key}`;
          if (value !== undefined && value !== null) {
            localStorage.setItem(storageKey, typeof value === 'string' ? value : JSON.stringify(value));
          }
        }
        appStore.restoreUISettings(saved);
      }
    }
  } catch (e) { /* 静默处理 */ }

  await chatStore.loadSessions()
  if (chatStore.sessions.length === 0) {
    chatStore.createNewSession()
  }

  startHealthCheck()
  loadCheckpoints()

  // 生命事件实时推送 → 更新 runtimeStore 的需求面板
  onWsMessage(({ event_type, data }) => {
    if (event_type) runtimeStore.handleLifeEvent({ event_type, data })
  })
})

onUnmounted(() => {
  window.removeEventListener('blur', clearDragState)
  window.removeEventListener('keyup', onGlobalKeyup)
  clearDragState()
  stopHealthCheck()
  if (trainAbortController) trainAbortController.abort()
})

function onGlobalKeyup(event) {
  if (event.key === 'Escape') clearDragState()
}
</script>

<style>
@import './assets/styles/index.css';

/* ===== 路由切换过渡：仅淡入，无离场动画 =====
   刻意不给 .route-leave-active 任何 transition：mode="out-in" 下 leave 会同步
   结束，delayedLeave 竞态窗口为零。快速连点切页时 enter 即使被打断，元素也只是
   丢掉 class 回落到 opacity:1，不可能卡在透明态导致白屏。 */
.route-enter-active {
  transition: opacity 0.22s cubic-bezier(0.22, 0.61, 0.36, 1), transform 0.22s cubic-bezier(0.22, 0.61, 0.36, 1);
  will-change: opacity, transform;
}
.route-enter-from {
  opacity: 0;
  transform: translateY(8px);
}
@media (prefers-reduced-motion: reduce) {
  .route-enter-active {
    transition: opacity 0.1s linear;
    transform: none;
  }
}

.drag-overlay-panel {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  color: var(--text);
}

.drag-overlay-panel p {
  margin: 0;
  font-size: 18px;
  font-weight: 650;
  color: var(--text);
}

/* 深色主题覆盖 */
.theme-dark .drag-overlay-panel {
  color: #e0e0e0;
}

.theme-dark .drag-overlay-panel p {
  color: #e0e0e0;
}
</style>
