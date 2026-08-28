<template>
  <div class="app-titlebar" @mousedown="onBarMouseDown" @dblclick="onBarDoubleClick">
    <button
      class="titlebar-btn"
      type="button"
      :title="collapsed ? '展开侧边栏' : '收起侧边栏'"
      :aria-label="collapsed ? '展开侧边栏' : '收起侧边栏'"
      @click="$emit('toggle-sidebar')"
    >
      <PanelLeftClose v-if="!collapsed" :size="17" aria-hidden="true" />
      <PanelLeft v-else :size="17" aria-hidden="true" />
    </button>

    <div class="titlebar-drag" />

    <div v-if="hasBridge" class="titlebar-window-controls">
      <button
        class="titlebar-window-btn"
        type="button"
        title="最小化"
        aria-label="最小化"
        @click="minimize"
      >
        <Minus :size="15" aria-hidden="true" />
      </button>
      <button
        class="titlebar-window-btn"
        type="button"
        :title="maximized ? '向下还原' : '最大化'"
        :aria-label="maximized ? '向下还原' : '最大化'"
        @click="toggleMaximize"
      >
        <Copy v-if="maximized" :size="13" aria-hidden="true" />
        <Square v-else :size="13" aria-hidden="true" />
      </button>
      <button
        class="titlebar-window-btn is-close"
        type="button"
        title="关闭"
        aria-label="关闭"
        @click="closeWindow"
      >
        <X :size="16" aria-hidden="true" />
      </button>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref } from 'vue'
import { Copy, Minus, PanelLeft, PanelLeftClose, Square, X } from 'lucide-vue-next'

defineProps({
  collapsed: { type: Boolean, default: false },
})

defineEmits(['toggle-sidebar'])

const bridge = ref(null)
const hasBridge = ref(false)
const maximized = ref(false)

let observer = null

function syncMaximized() {
  maximized.value = document.documentElement.getAttribute('data-maximized') === 'true'
}

onMounted(() => {
  // 窗口控制桥由 desktop/main.py 的 _WindowBridge 提供，客户端库以
  // QWebEngineScript 在 DocumentCreation 阶段注入。浏览器里调试时两者都不存在，
  // 此时 hasBridge 为 false，三个窗口按钮自动隐藏，其余布局照常工作。
  const transport = window.qt?.webChannelTransport
  if (transport && typeof window.QWebChannel === 'function') {
    new window.QWebChannel(transport, (channel) => {
      const obj = channel.objects?.seedWindow
      if (!obj) return
      bridge.value = obj
      hasBridge.value = true
      syncMaximized()
    })
  }

  // Python 侧通过 data-maximized 回写真实窗口状态（resizeEvent / changeEvent 驱动），
  // 这里只观察该属性，不做轮询。
  syncMaximized()
  observer = new MutationObserver(syncMaximized)
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-maximized'],
  })
})

onUnmounted(() => {
  observer?.disconnect()
  observer = null
})

function isInteractive(target) {
  return !!(target instanceof Element && target.closest('button, input, a'))
}

function onBarMouseDown(event) {
  if (event.button !== 0 || isInteractive(event.target)) return
  bridge.value?.startDrag()
}

function onBarDoubleClick(event) {
  if (isInteractive(event.target)) return
  toggleMaximize()
}

function minimize() {
  bridge.value?.minimize()
}

function toggleMaximize() {
  bridge.value?.toggleMaximize()
}

function closeWindow() {
  bridge.value?.close()
}
</script>
