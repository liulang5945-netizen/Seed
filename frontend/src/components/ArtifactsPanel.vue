<template>
  <section
    v-show="store.isOpen"
    class="artifacts-panel"
    :class="{ 'panel-full': store.isFull }"
    aria-label="工件面板"
  >
    <!-- 左缘拖拽调宽（仅分栏档） -->
    <div
      v-if="!store.isFull"
      class="panel-resize-handle"
      :class="{ active: dockResizing }"
      @mousedown="startDockResize"
    ></div>

    <!-- 头部：工件名 + 预览/代码 tab + 档位 -->
    <header class="ap-head">
      <span class="ap-title" :title="store.currentArtifact?.title">{{ store.currentArtifact?.title || '工件' }}</span>
      <div class="ap-tabs" role="tablist">
        <button
          class="ap-tab" :class="{ active: view === 'preview' }"
          role="tab" :aria-selected="view === 'preview'"
          @click="view = 'preview'"
        >预览</button>
        <button
          class="ap-tab" :class="{ active: view === 'code' }"
          role="tab" :aria-selected="view === 'code'"
          @click="view = 'code'"
        >代码</button>
      </div>
      <div class="ap-actions">
        <button
          class="ap-btn" :class="{ active: !store.isFull }"
          title="分栏视图" @click="store.setMode('split')"
        ><Columns2 :size="14" /></button>
        <button
          class="ap-btn" :class="{ active: store.isFull }"
          title="全宽视图" @click="store.setMode('full')"
        ><Maximize2 :size="14" /></button>
        <button class="ap-btn" title="收起面板" @click="store.closeArtifact()">
          <PanelRightClose :size="14" />
        </button>
      </div>
    </header>

    <!-- 内容区：单一视图，随 tab 切换 -->
    <div class="ap-body">
      <!-- 预览：仅 HTML 工件可渲染 -->
      <iframe
        v-if="view === 'preview' && isHtml"
        class="ap-preview"
        sandbox="allow-scripts"
        :srcdoc="store.currentArtifact.content"
        title="工件预览"
      ></iframe>
      <div v-else-if="view === 'preview'" class="ap-empty">
        <EyeOff :size="26" />
        <p>{{ store.currentArtifact?.language?.toUpperCase() || '该类型' }} 工件暂不支持预览，切到「代码」查看源码</p>
      </div>
      <!-- 代码：高亮只读 -->
      <div v-else class="ap-code markdown-body" v-html="codeHtml"></div>
    </div>
  </section>
</template>

<script setup>
/**
 * 工件面板（Claude Artifacts 式侧拉视图）。
 *
 * 与原「Dock 塞整个 IDE 三栏」不同——这里只做一件事：展示对话驱动的单个工件，
 * 预览（HTML iframe）/代码（高亮只读）两个视图切换，宽度自适应视口。
 * IDE 完整功能（文件树/终端/多文件）在独立的 /workspace 路由页，职责不重复。
 */
import { computed, ref, watch } from 'vue'
import { Columns2, Maximize2, PanelRightClose, EyeOff } from 'lucide-vue-next'
import { useArtifactStore } from '../stores/artifactStore.js'
import { useMarkdown } from '../composables/useMarkdown.js'

const store = useArtifactStore()
const { renderMarkdown } = useMarkdown()

const view = ref('preview')
const dockResizing = ref(false)

const isHtml = computed(() => {
  const lang = (store.currentArtifact?.language || '').toLowerCase()
  return lang === 'html' || lang === 'htm' || lang === 'svg'
})

// 代码视图：把工件内容包一层 fence 交给统一渲染管线（语法动态加载 + 高亮）
const codeHtml = computed(() => {
  const artifact = store.currentArtifact
  if (!artifact) return ''
  const lang = artifact.language || 'text'
  // 内容里若本身含 ``` 会破坏 fence——用四反引号围栏兜底
  const fence = artifact.content.includes('```') ? '````' : '```'
  return renderMarkdown(`${fence}${lang}\n${artifact.content}\n${fence}`)
})

// 切换工件时回到预览视图（主流行为：新工件默认预览）
watch(() => store.currentArtifact, () => { view.value = 'preview' })

// 面板左缘拖拽调宽（仅分栏档；向左拖=加宽）
function startDockResize(e) {
  e.preventDefault()
  dockResizing.value = true
  const startX = e.clientX, startW = store.dockWidth
  const onMove = (ev) => { store.setWidth(startW - (ev.clientX - startX)) }
  const onUp = () => {
    dockResizing.value = false
    document.removeEventListener('mousemove', onMove)
    document.removeEventListener('mouseup', onUp)
  }
  document.addEventListener('mousemove', onMove)
  document.addEventListener('mouseup', onUp)
}
</script>

<style scoped>
.artifacts-panel {
  position: relative;
  display: flex;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
  flex: none;
  /* 宽度自适应视口：窄屏不挤爆对话列（修「挤压变形」——原 560px 定宽在小窗口必然溢出）。
     拖拽调宽后由内联 style 覆盖（store.dockWidth）。 */
  width: clamp(420px, 44vw, 760px);
  margin: 8px 8px 8px 0;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg, 16px);
  box-shadow: var(--shadow-sm, 0 1px 2px rgba(0, 0, 0, 0.06));
  overflow: hidden;
  animation: ap-in 0.18s cubic-bezier(0.22, 0.61, 0.36, 1);
}
.artifacts-panel.panel-full {
  flex: 1;
  width: auto !important;
}
@keyframes ap-in {
  from { opacity: 0; transform: translateX(12px); }
  to { opacity: 1; transform: none; }
}
@media (prefers-reduced-motion: reduce) {
  .artifacts-panel { animation: none; }
}

.panel-resize-handle {
  position: absolute;
  left: -3px;
  top: 0;
  bottom: 0;
  width: 6px;
  z-index: 6;
  cursor: col-resize;
  background: transparent;
  transition: background 120ms ease;
}
.panel-resize-handle:hover,
.panel-resize-handle.active { background: color-mix(in srgb, var(--primary) 35%, transparent); }

/* ── 头部 ── */
.ap-head {
  flex: none;
  height: 44px;
  padding: 0 10px 0 14px;
  display: flex;
  align-items: center;
  gap: 10px;
  border-bottom: 1px solid var(--border);
}
.ap-title {
  min-width: 0;
  flex: 1;
  font-size: 0.82rem;
  font-weight: 650;
  color: var(--foreground);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ap-tabs {
  display: flex;
  gap: 2px;
  background: var(--muted);
  border-radius: 9px;
  padding: 2px;
  flex: none;
}
.ap-tab {
  height: 26px;
  padding: 0 12px;
  border: 0;
  border-radius: 7px;
  background: transparent;
  color: var(--muted-foreground);
  font-size: 0.74rem;
  font-weight: 550;
  cursor: pointer;
  transition: background 120ms ease, color 120ms ease;
}
.ap-tab.active {
  background: var(--card);
  color: var(--foreground);
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
}
.ap-tab:focus-visible { outline: 2px solid var(--ring); outline-offset: 1px; }
.ap-actions { display: flex; align-items: center; gap: 2px; flex: none; }
.ap-btn {
  width: 30px;
  height: 30px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: var(--muted-foreground);
  cursor: pointer;
  transition: background 120ms ease, color 120ms ease;
}
.ap-btn:hover { background: var(--muted); color: var(--foreground); }
.ap-btn.active { color: var(--primary); background: color-mix(in srgb, var(--primary) 10%, transparent); }
.ap-btn:focus-visible { outline: 2px solid var(--ring); outline-offset: 1px; }

/* ── 内容区 ── */
.ap-body {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.ap-preview {
  flex: 1;
  width: 100%;
  border: 0;
  background: #fff;
}
.ap-empty {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: var(--muted-foreground);
  font-size: 0.8rem;
  padding: 20px;
  text-align: center;
}
.ap-code {
  flex: 1;
  overflow: auto;
  padding: 12px 14px;
  font-size: 0.82rem;
}

/* 窄视口：面板升级为更大占比，保证代码可读（对话列仍保留最小可读宽度） */
@media (max-width: 1100px) {
  .artifacts-panel { width: clamp(420px, 56vw, 760px); }
}
</style>
