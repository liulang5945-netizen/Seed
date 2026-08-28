<template>
  <div class="agent-page">
    <div class="page-container">

      <RuntimeEvidenceStrip context="agent" compact />

      <!-- 1. 基本信息卡片 -->
      <section class="cfg-card">
        <div class="cfg-card-head">
          <div class="cfg-title">
            <h2>基本信息</h2>
            <span class="sub">查看 Taiji 原生能力执行平面</span>
          </div>
          <button class="btn btn-ghost sm" :disabled="runtimeStore.toolsLoading" @click="refreshAgentRuntime">
            <RefreshCw :size="14" :class="{ spin: runtimeStore.toolsLoading }" /> 刷新
          </button>
        </div>
        <div class="form-grid">
          <div class="form-field">
            <label class="form-label" for="agent-name">运行时名称<span class="req">*</span></label>
            <input id="agent-name" class="input" type="text" value="Taiji 原生运行时" readonly>
          </div>
          <div class="form-field">
            <label class="form-label" for="agent-domain">能力域<span class="req">*</span></label>
            <select id="agent-domain" class="select" disabled>
              <option value="language">语言</option>
              <option value="reasoning" selected>推理</option>
              <option value="code">代码</option>
              <option value="knowledge">知识</option>
              <option value="multi">多域</option>
            </select>
          </div>
          <div class="form-field full">
            <label class="form-label" for="agent-desc">描述</label>
            <textarea id="agent-desc" class="textarea" rows="3" readonly>能力由原生 registry 声明并按权限执行；语言 provider 作为可替换的语言器官接入。</textarea>
          </div>
        </div>
      </section>

      <!-- 2. 状态概览 -->
      <section class="overview-grid">
        <div class="ov-card" :class="runtimeStore.connectionClass">
          <span class="ov-ic">
            <svg class="ic-svg" viewBox="0 0 24 24"><path d="M5 12.55a11 11 0 0 1 14.08 0M1.42 9a16 16 0 0 1 21.16 0M8.53 16.11a6 6 0 0 1 6.95 0M12 20h.01"/></svg>
          </span>
          <div class="ov-text">
            <small class="ov-label">连接状态</small>
            <strong class="ov-value">
              <span class="ov-dot"></span>
              {{ runtimeStore.connectionStatus }}
            </strong>
          </div>
        </div>
        <div class="ov-card">
          <span class="ov-ic">
            <svg class="ic-svg" viewBox="0 0 24 24"><path d="M14.7 6.3a4 4 0 0 0-5.6 5.6L3 18v3h3l6.1-6.1a4 4 0 0 0 5.6-5.6l-2.1 2.1-2-2 2.1-2.1Z"/></svg>
          </span>
          <div class="ov-text">
            <small class="ov-label">工具数量</small>
            <strong class="ov-value">{{ runtimeStore.tools.length }}<span class="ov-unit"> 个可用</span></strong>
          </div>
        </div>
        <div v-if="runtimeStore.memoryAvailableGb" class="ov-card">
          <span class="ov-ic">
            <svg class="ic-svg" viewBox="0 0 24 24"><rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 10v4M11 10v4M15 10v4"/></svg>
          </span>
          <div class="ov-text">
            <small class="ov-label">可用内存</small>
            <strong class="ov-value">{{ runtimeStore.memoryAvailableGb.toFixed(1) }}<span class="ov-unit"> GB</span></strong>
          </div>
        </div>
      </section>

      <!-- 原生能力 registry -->
      <section id="ac-panel-tools" class="tab-panel active" aria-label="原生能力">
        <div class="filter-row">
          <div class="search-field">
            <Search :size="15" />
            <input v-model="toolQuery" placeholder="搜索工具名或描述..." />
          </div>
          <select v-model="toolSource" class="select">
            <option value="">全部来源</option>
            <option v-for="s in toolSources" :key="s" :value="s">{{ s }}</option>
          </select>
        </div>
        <div class="tool-grid">
          <div v-for="tool in filteredTools" :key="tool.name" class="tool-card" :class="{ off: tool.enabled === false }">
            <div class="tool-head">
              <span class="tool-ic">
                <svg class="ic-svg" viewBox="0 0 24 24"><path d="M14.7 6.3a4 4 0 0 0-5.6 5.6L3 18v3h3l6.1-6.1a4 4 0 0 0 5.6-5.6l-2.1 2.1-2-2 2.1-2.1Z"/></svg>
              </span>
              <button class="toggle" :class="{ on: tool.enabled !== false }" :aria-pressed="tool.enabled !== false" aria-label="启用工具"></button>
            </div>
            <div class="tool-name">{{ tool.name }}</div>
            <div class="tool-desc">{{ tool.description || '暂无描述' }}</div>
          </div>
        </div>
        <div class="panel-foot">
          <span class="foot-hint">共 {{ filteredTools.length }} 个工具</span>
        </div>
        <div v-if="!filteredTools.length && !runtimeStore.toolsLoading" class="empty-msg">无匹配工具</div>
      </section>

    </div>
  </div>
</template>

<script setup>
defineOptions({ name: 'AgentConfigView' })
import { computed, inject, onActivated, ref } from 'vue'
import { RefreshCw, Search } from 'lucide-vue-next'
import { useRuntimeStore } from '../stores/runtimeStore.js'
import RuntimeEvidenceStrip from '../components/RuntimeEvidenceStrip.vue'

const runtimeStore = useRuntimeStore()
const toast = inject('toast', () => {})

const toolQuery = ref('')
const toolSource = ref('')

const toolSources = computed(() => {
  const sources = new Set(runtimeStore.tools.map(t => t.source).filter(Boolean))
  return [...sources].sort()
})

const filteredTools = computed(() => {
  let list = runtimeStore.tools
  if (toolSource.value) list = list.filter(t => t.source === toolSource.value)
  if (toolQuery.value) {
    const q = toolQuery.value.toLowerCase()
    list = list.filter(t => t.name?.toLowerCase().includes(q) || t.description?.toLowerCase().includes(q))
  }
  return list
})

// R5: sourceLabel/categoryLabel/permissionLabel 未被模板引用，已移除（需要时从 git 历史恢复）。

async function refreshAgentRuntime({ silent = false } = {}) {
  await runtimeStore.refreshTools()
  // 自动刷新（进入页面）不打扰；仅手动点击「刷新」时提示
  if (!silent) toast('已刷新', 'success')
}

// keep-alive 缓存后重新进入页面时也刷新运行时状态（首次挂载同样触发，静默）
onActivated(() => refreshAgentRuntime({ silent: true }))
</script>

<style scoped>
/* ═══ 原生能力页 · 豆包设计 token 体系 ═══ */
.agent-page {
  height: 100%;
  overflow-y: auto;
  background: var(--background);
  color: var(--foreground);
}

/* 内容容器 */
.page-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 28px;
  display: flex;
  flex-direction: column;
  gap: 22px;
}

/* 按钮 */
.btn {
  height: 36px;
  padding: 0 15px;
  border-radius: 999px;
  border: 1px solid transparent;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  font-size: 0.86rem;
  font-weight: 500;
  cursor: pointer;
  white-space: nowrap;
  transition: background 150ms ease, border-color 150ms ease, transform 120ms ease, color 150ms ease;
}
.btn:active { transform: translateY(1px); }
.btn.sm { height: 30px; padding: 0 12px; font-size: 0.8rem; }
.btn:disabled { opacity: 0.55; cursor: not-allowed; }
.btn-primary { background: var(--primary); color: var(--primary-foreground); }
.btn-primary:hover:not(:disabled) { background: color-mix(in srgb, var(--primary) 90%, var(--foreground)); }
.btn-ghost { background: var(--muted); color: var(--foreground); }
.btn-ghost:hover:not(:disabled) { background: color-mix(in srgb, var(--muted) 80%, var(--foreground) 12%); }
.btn-outline { background: var(--background); color: var(--foreground); border-color: var(--border); }
.btn-outline:hover:not(:disabled) { background: var(--muted); }

/* 配置卡片 */
.cfg-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 18px;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.cfg-card:hover { border-color: color-mix(in srgb, var(--primary) 25%, var(--border)); }
.cfg-card-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.cfg-title { display: flex; flex-direction: column; gap: 3px; }
.cfg-card-head h2 { margin: 0; font-size: 1rem; font-weight: 600; color: var(--foreground); }
.cfg-card-head .sub { font-size: 0.76rem; color: var(--muted-foreground); }

/* 表单 */
.form-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.form-field { display: flex; flex-direction: column; gap: 6px; }
.form-field.full { grid-column: 1 / -1; }
.form-label {
  font-size: 0.8rem;
  font-weight: 500;
  color: var(--foreground);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.form-label .req { color: var(--destructive); margin-left: 2px; }
.form-label .hint { font-size: 0.7rem; font-weight: 400; color: var(--muted-foreground); }

.input, .textarea, .select {
  width: 100%;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--background);
  color: var(--foreground);
  padding: 9px 12px;
  font-size: 0.86rem;
  outline: none;
  transition: border-color 160ms ease, box-shadow 160ms ease;
}
.input::placeholder, .textarea::placeholder { color: var(--muted-foreground); }
.input:focus, .textarea:focus, .select:focus {
  border-color: var(--ring);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--ring) 18%, transparent);
}
.textarea { resize: vertical; min-height: 76px; line-height: 1.5; }
.select {
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%237f8d9f' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 10px center;
  padding-right: 32px;
}

/* 概览网格 */
.overview-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(168px, 1fr));
  gap: 12px;
}
.ov-card {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 16px;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--card);
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.ov-card:hover { border-color: color-mix(in srgb, var(--primary) 25%, var(--border)); }
.ov-card.connected { border-color: var(--success); }
.ov-ic {
  width: 36px; height: 36px; border-radius: 10px;
  display: grid; place-items: center; flex: none;
  background: color-mix(in srgb, var(--primary) 14%, transparent);
  color: var(--primary);
}
.ov-card.connected .ov-ic { background: var(--success-light); color: var(--success); }
.ov-ic .ic-svg { width: 18px; height: 18px; stroke: currentColor; fill: none; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }
.ov-text { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.ov-label { font-size: 0.72rem; color: var(--muted-foreground); }
.ov-value { font-size: 0.92rem; font-weight: 600; color: var(--foreground); display: inline-flex; align-items: center; gap: 6px; }
.ov-unit { font-size: 0.74rem; font-weight: 400; color: var(--muted-foreground); }
.ov-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted-foreground); flex: none; }
.ov-card.connected .ov-dot { background: var(--success); }

/* 标签页 */
/* 不画 border-bottom：全应用唯一外围边框归 .router-wrapper（见 styles/shell.css） */
.tabs { display: flex; align-items: center; gap: 4px; padding: 0 4px; }
.tab {
  appearance: none;
  background: transparent;
  border: 0;
  padding: 10px 14px;
  font-size: 0.88rem;
  color: var(--muted-foreground);
  cursor: pointer;
  position: relative;
  transition: color 150ms ease;
}
.tab:hover { color: var(--foreground); }
.tab:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
  border-radius: var(--radius-sm, 6px);
}
.tab.active { color: var(--foreground); font-weight: 600; }
.tab.active::after {
  content: '';
  position: absolute;
  left: 10px; right: 10px; bottom: 0;
  height: 2px; border-radius: 2px; background: var(--primary);
}

/* 面板常驻 DOM + 零动画：切换是 0ms 的显隐，保留滚动位置与输入内容。
   本视图面板是 flex 布局，显隐需用 display: none / flex（不能用 block） */
.tab-panel { display: none; flex-direction: column; gap: 16px; }
.tab-panel.active { display: flex; }

/* 筛选 */
.filter-row { display: flex; gap: 8px; align-items: center; }
.search-field {
  flex: 1;
  min-width: 0;
  height: 38px;
  padding: 0 12px;
  border-radius: 10px;
  background: var(--muted);
  border: 1px solid transparent;
  display: flex;
  align-items: center;
  gap: 9px;
  color: var(--muted-foreground);
  transition: border-color 160ms ease, background 160ms ease;
}
.search-field:focus-within { border-color: var(--ring); background: var(--background); }
.search-field input {
  flex: 1; min-width: 0; border: 0; outline: none; background: transparent;
  font-size: 0.86rem; color: var(--foreground);
}
.search-field input::placeholder { color: var(--muted-foreground); }
.filter-row .select { width: 140px; height: 38px; flex: none; }

/* 工具网格 */
.tool-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}
.tool-card {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px;
  background: var(--card);
  display: flex;
  flex-direction: column;
  gap: 10px;
  transition: border-color 160ms ease, background 160ms ease, transform 160ms ease;
}
.tool-card:hover { border-color: color-mix(in srgb, var(--primary) 35%, var(--border)); transform: translateY(-1px); }
.tool-card.off { opacity: 0.65; }
.tool-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.tool-ic {
  width: 34px; height: 34px; border-radius: 9px;
  display: grid; place-items: center;
  background: color-mix(in srgb, var(--primary) 14%, transparent);
  color: var(--primary);
  flex: none;
}
.tool-card.off .tool-ic { background: var(--muted); color: var(--muted-foreground); }
.ic-svg { width: 18px; height: 18px; stroke: currentColor; fill: none; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }
.tool-name { font-size: 0.84rem; font-weight: 600; color: var(--foreground); word-break: break-word; }
.tool-desc { font-size: 0.74rem; color: var(--muted-foreground); line-height: 1.45; min-height: 2.2em; }

/* 开关 toggle */
.toggle {
  width: 38px; height: 22px; border-radius: 999px;
  background: var(--muted); border: 1px solid var(--border);
  position: relative; cursor: pointer;
  transition: background 160ms ease, border-color 160ms ease;
  flex: none; padding: 0;
}
.toggle::after {
  content: ''; position: absolute; top: 2px; left: 2px;
  width: 16px; height: 16px; border-radius: 50%;
  background: var(--background);
  box-shadow: 0 1px 2px rgba(0,0,0,0.18);
  transition: transform 180ms cubic-bezier(.4,0,.2,1);
}
.toggle.on { background: var(--primary); border-color: var(--primary); }
.toggle.on::after { transform: translateX(16px); background: var(--primary-foreground); box-shadow: none; }

/* 状态 chip */
.status-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 3px 10px; border-radius: 999px;
  font-size: 0.74rem; font-weight: 500;
  white-space: nowrap;
}
.status-chip::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.status-running { background: color-mix(in srgb, var(--chart-2) 18%, transparent); color: var(--chart-2); }
.status-stopped { background: color-mix(in srgb, var(--muted-foreground) 16%, transparent); color: var(--muted-foreground); }

/* 底部 */
.panel-foot { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding-top: 4px; }
.foot-hint { font-size: 0.76rem; color: var(--muted-foreground); }

.empty-msg {
  padding: 36px; text-align: center;
  color: var(--muted-foreground); font-size: 0.84rem;
  border: 1px dashed color-mix(in srgb, var(--muted-foreground) 30%, transparent);
  border-radius: 12px;
}

.spin { animation: ag-spin 0.9s linear infinite; }
@keyframes ag-spin { to { transform: rotate(360deg); } }

@media (max-width: 880px) {
  .page-container { padding: 16px; }
  .overview-grid { grid-template-columns: 1fr; }
  .tool-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .form-grid { grid-template-columns: 1fr; }
  .filter-row { flex-wrap: wrap; }
  .filter-row .select { width: 100%; }
}
</style>
