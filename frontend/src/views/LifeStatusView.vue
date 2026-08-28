<template>
  <div class="life-status-view">
    <!-- ═══ 顶栏 ═══ -->
    <header class="topbar">
      <div class="topbar-left">
        <span class="topbar-title">生命状态</span>
        <span class="topbar-sub">{{ runtimeStore.health.isTaiji ? '实时查看 Taiji 原生状态通路' : '实时查看原生运行时状态' }}</span>
      </div>
      <span class="topbar-spacer"></span>
      <n-tag
        :type="runtimeStore.connectionClass === 'connected' ? 'success' : 'error'"
        size="small"
        round
      >
        {{ runtimeStore.connectionStatus }}
      </n-tag>
      <button class="btn btn-outline" @click="exportReport">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
        导出报告
      </button>
    </header>

    <RuntimeEvidenceStrip context="life" />

    <!-- ═══ 滚动内容区 ═══ -->
    <div class="scroll-area">

      <!-- ═══ 生命活动操作栏（兼容状态通路，原生动作器接入后启用） ═══ -->
      <section class="action-bar">
        <span class="action-bar-title">生命活动</span>
        <div class="action-buttons">
          <button class="btn btn-life" style="--life-color: var(--chart-2);" :disabled="actionLoading" @click="feedTaiji">
            <span class="life-btn-emoji">🍚</span>喂养
          </button>
          <button class="btn btn-life" style="--life-color: var(--chart-1);" :disabled="actionLoading" @click="sleepTaiji">
            <span class="life-btn-emoji">💤</span>睡眠
          </button>
          <button class="btn btn-life" style="--life-color: var(--chart-3);" :disabled="actionLoading" @click="playTaiji">
            <span class="life-btn-emoji">🎮</span>玩耍
          </button>
          <button class="btn btn-life" style="--life-color: var(--chart-4);" :disabled="actionLoading" @click="evolveTaiji">
            <span class="life-btn-emoji">🧬</span>进化
          </button>
        </div>
        <span v-if="currentActivity" class="action-current">{{ currentActivity }}</span>
        <span v-if="actionResult" class="action-result">{{ actionResult }}</span>
      </section>

      <LifeNativeStatusPanel
        v-if="runtimeStore.health.isTaiji"
        :connection-status="runtimeStore.connectionStatus"
        :model-name="runtimeStore.health.modelName"
        :language-provider-state="runtimeStore.health.languageProvider?.state"
        :language-provider-backend="runtimeStore.health.languageProvider?.backend_id"
        :tool-count="runtimeStore.tools.length"
        :workbench-detail="runtimeStore.statusEvidence.workbench.detail"
        :has-needs-data="hasNeedsData"
        :health-message="runtimeStore.health.message"
      />

      <template v-else>

      <!-- ═══ Seed 原生运行时数据来源说明（诚实呈现：needs 未接入，无假数据） ═══ -->
      <div v-if="runtimeStore.health.isSeed" class="seed-datasource-note">
        <span class="dsn-badge">DATA SOURCE</span>
        <div>
          <strong>当前运行时：Seed 原生（{{ runtimeStore.health.modelName || 'seed' }}）</strong>
          <p>下方「需求五维 / 生命表达 / 需求明细」需要运行时的 needs 上报通道。当前原生运行时尚未提供该通道，因此这些面板显示「暂无数据」——不是模型输出，也不是估算值。内存与连接状态为系统实测。</p>
        </div>
      </div>

      <!-- ═══ KPI 卡片行（全部来自原生状态快照与运行时实测，无估算值） ═══ -->
      <div class="kpi-grid">
        <!-- 卡1：累计交互 -->
        <div class="kpi-card" style="--kpi-color: var(--chart-1);">
          <div class="kpi-label">
            <svg class="kpi-icon" viewBox="0 0 24 24"><path d="M8 10h8M8 14h5"/><path d="M21 12a8 8 0 0 1-11.6 7.2L4 21l1.8-5.4A8 8 0 1 1 21 12Z"/></svg>
            累计交互
          </div>
          <div class="kpi-value">{{ life.total_interactions != null ? Number(life.total_interactions).toLocaleString() : '暂无数据' }}</div>
          <div class="kpi-trend trend-stable"><span class="kpi-src">来自原生状态快照</span></div>
        </div>

        <!-- 卡2：运行时长 -->
        <div class="kpi-card" style="--kpi-color: var(--chart-2);">
          <div class="kpi-label">
            <svg class="kpi-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><polyline points="12 7 12 12 15 14"/></svg>
            运行时长
          </div>
          <div class="kpi-value">{{ life.uptime_seconds != null ? fmtTime(life.uptime_seconds) : '暂无数据' }}</div>
          <div class="kpi-trend trend-stable"><span class="kpi-src">调度器 uptime</span></div>
        </div>

        <!-- 卡3：生命调度器 -->
        <div class="kpi-card" style="--kpi-color: var(--chart-3);">
          <div class="kpi-label">
            <svg class="kpi-icon" viewBox="0 0 24 24"><path d="M3 12h3l1.5-6 3 12 1.5-6h8"/></svg>
            生命调度器
          </div>
          <div class="kpi-value">{{ life.is_running != null ? (life.is_running ? '运行中' : '未启动') : '暂无数据' }}</div>
          <div class="kpi-trend trend-stable"><span class="kpi-src">life_scheduler 状态</span></div>
        </div>

        <!-- 卡4：内存余量 -->
        <div class="kpi-card" style="--kpi-color: var(--chart-4);">
          <div class="kpi-label">
            <svg class="kpi-icon" viewBox="0 0 24 24"><rect x="7" y="5" width="4" height="14" rx="1"/><rect x="13" y="7" width="4" height="10" rx="1"/></svg>
            内存余量
          </div>
          <div class="kpi-value">{{ runtimeStore.memoryAvailablePct != null ? Math.round(runtimeStore.memoryAvailablePct) + '%' : '暂无数据' }}</div>
          <div class="kpi-trend trend-stable"><span class="kpi-src">系统可用内存</span></div>
        </div>
      </div>

      <LifeNeedsDashboard
        :needs="life.needs || {}"
        :alive="!!life.is_running"
        :has-needs-data="hasNeedsData"
        :life-expressions="runtimeStore.lifeExpressions"
        :need-rows="needRows"
        :activity-log="activityLog"
      />
      </template>
    </div>
  </div>
</template>

<script setup>
defineOptions({ name: 'LifeStatusView' })
import { ref, computed, onActivated, onDeactivated, onUnmounted, inject } from 'vue'
import { useRuntimeStore } from '@/stores/runtimeStore.js'
import { fmtTime } from '@/composables/useTraining.js'
import LifeNativeStatusPanel from '@/components/LifeNativeStatusPanel.vue'
import LifeNeedsDashboard from '@/components/LifeNeedsDashboard.vue'
import RuntimeEvidenceStrip from '@/components/RuntimeEvidenceStrip.vue'

const runtimeStore = useRuntimeStore()
const toast = inject('toast', () => {})

const activityLog = ref([])
const currentActivity = ref('')
const actionResult = ref('')
const actionLoading = ref(false)

// 从 runtimeStore 获取生命数据（来源 /api/runtime/status 的原生状态快照）
const life = computed(() => runtimeStore.life || {})
const hasNeedsData = computed(() => Object.keys(life.value?.needs || {}).length > 0)

// 需求明细：只展示后端真实上报的五个 needs 维度
const NEED_META = [
  { key: 'hunger', label: '饥饿 · 知识摄取' },
  { key: 'fatigue', label: '疲劳 · 睡眠需求' },
  { key: 'curiosity', label: '好奇 · 探索驱动' },
  { key: 'stress', label: '压力 · 错误负担' },
  { key: 'boredom', label: '无聊 · 活动需求' },
]
const needRows = computed(() => NEED_META.map((meta) => {
  const raw = life.value?.needs?.[meta.key]
  const value = typeof raw === 'number' && Number.isFinite(raw)
    ? Math.max(0, Math.min(100, raw))
    : null
  const state = value == null ? 'none' : value > 70 ? 'alert' : value >= 40 ? 'watch' : 'calm'
  return { ...meta, value, state }
}))

// 添加事件记录（仅记录真实发生的操作与后端回执）
function addLog(type, emoji, message) {
  activityLog.value.unshift({
    time: new Date().toLocaleTimeString(),
    type,
    emoji,
    message,
  })
  if (activityLog.value.length > 50) {
    activityLog.value.pop()
  }
}

// 原生动作器尚未提供 feed/sleep/play/evolve 的正式能力契约；
// 客户端保留按钮布局，但不伪造状态变化或调用历史接口。
const NATIVE_RUNTIME_TIP = 'Taiji 原生动作器尚未接入生命活动能力'
async function callLifeAction(action) {
  currentActivity.value = ''
  actionResult.value = NATIVE_RUNTIME_TIP
  toast(NATIVE_RUNTIME_TIP, 'info')
  addLog(action, 'ℹ️', NATIVE_RUNTIME_TIP)
}

function feedTaiji() {
  currentActivity.value = '🍚 喂养中...'
  callLifeAction('feed')
}
function sleepTaiji() {
  currentActivity.value = '💤 睡眠中...'
  callLifeAction('sleep')
}
function playTaiji() {
  currentActivity.value = '🎮 玩耍中...'
  callLifeAction('play')
}
function evolveTaiji() {
  currentActivity.value = '🧬 进化查询中...'
  callLifeAction('evolve')
}

// 导出报告：聚合当前页面真实数据为 JSON 快照并 Blob 下载
function exportReport() {
  const snapshot = {
    exported_at: new Date().toISOString(),
    source: 'LifeStatusView 快照（仅含真实数据）',
    runtime: {
      connection: runtimeStore.connectionStatus,
      health_state: runtimeStore.health.state,
      is_taiji: !!runtimeStore.health.isTaiji,
    },
    life: JSON.parse(JSON.stringify(life.value || {})),
    memory: {
      available_pct: runtimeStore.memoryAvailablePct,
      available_gb: runtimeStore.memoryAvailableGb,
      level: runtimeStore.memoryLevel,
    },
    life_expressions: runtimeStore.lifeExpressions.map((e) => ({
      type: e.type,
      text: e.text,
      priority: e.priority,
    })),
    activity_log: [...activityLog.value],
  }
  const blob = new Blob([JSON.stringify(snapshot, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `life-status-${new Date().toISOString().replace(/[:.]/g, '-')}.json`
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(url)
  addLog('export', '📄', '已导出生命状态快照（JSON）')
  toast('✅ 生命状态快照已导出', 'success')
}

let refreshInterval = null

// App 级健康检查每 15 秒已刷新同一负载（/api/runtime/status），
// 本页只做低频刷新，避免重复轮询。
function startPolling() {
  stopPolling() // 先清后启，避免重复启动
  runtimeStore.refreshAll().catch(() => {})
  refreshInterval = setInterval(() => {
    runtimeStore.refreshAll().catch(() => {})
  }, 60000)
}
function stopPolling() {
  if (refreshInterval) {
    clearInterval(refreshInterval)
    refreshInterval = null
  }
}

// keep-alive 缓存后离开页面时停止轮询，回来时恢复（首次挂载同样触发 onActivated）
onActivated(() => startPolling())
onDeactivated(() => stopPolling())

// 兜底：组件真正卸载时也清理定时器
onUnmounted(() => stopPolling())
</script>

<style scoped>
/* ═══ 视图容器 ═══ */
.life-status-view {
  --chart-1: var(--primary);
  --chart-2: var(--success, #10b981);
  --chart-3: var(--warning, #f59e0b);
  --chart-4: var(--destructive, #ef4444);

  height: 100%;
  display: flex;
  flex-direction: column;
  background: var(--background);
  color: var(--foreground);
  font-family: var(--font-sans);
  overflow: hidden;
}

/* ═══ 顶栏 ═══ */
/* 不画 border-bottom：外围边框由 .router-wrapper 独占（见 styles/shell.css） */
.topbar {
  height: 52px;
  flex: none;
  padding: 0 18px;
  display: flex;
  align-items: center;
  gap: 12px;
}
.topbar-left {
  display: flex;
  flex-direction: column;
  justify-content: center;
  line-height: 1.15;
}
.topbar-title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--foreground);
}
.topbar-sub {
  margin-top: 2px;
  font-size: 0.72rem;
  color: var(--muted-foreground);
}
.topbar-spacer {
  flex: 1;
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
  transition: background 150ms ease, border-color 150ms ease, transform 120ms ease, color 150ms ease;
}
.btn:active { transform: translateY(1px); }
.btn:focus-visible {
  outline: 2px solid var(--ring);
  outline-offset: 2px;
}
.btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}
.btn-outline {
  background: var(--background);
  color: var(--foreground);
  border-color: var(--border);
}
.btn-outline:hover {
  background: var(--muted);
}

/* ═══ 生命活动操作栏 ═══ */
.action-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  padding: 14px 18px;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--card);
}
.action-bar-title {
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--foreground);
}
.action-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}
.btn-life {
  background: color-mix(in srgb, var(--life-color, var(--primary)) 10%, var(--background));
  border-color: color-mix(in srgb, var(--life-color, var(--primary)) 34%, var(--border));
  color: var(--foreground);
}
.btn-life:hover:not(:disabled) {
  background: color-mix(in srgb, var(--life-color, var(--primary)) 18%, var(--background));
  border-color: color-mix(in srgb, var(--life-color, var(--primary)) 55%, var(--border));
}
.life-btn-emoji {
  font-size: 0.95rem;
  line-height: 1;
}
.action-current {
  font-size: 0.78rem;
  color: var(--muted-foreground);
}
.action-result {
  flex-basis: 100%;
  font-size: 0.78rem;
  color: var(--muted-foreground);
}

/* ═══ 滚动内容区 ═══ */
.scroll-area {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 28px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* ═══ Seed 数据来源说明卡 ═══ */
.seed-datasource-note {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px dashed var(--border);
  background: var(--card);
}
.dsn-badge {
  flex: none;
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  padding: 3px 8px;
  border-radius: 6px;
  margin-top: 2px;
  background: var(--muted);
  color: var(--muted-foreground);
  font-family: var(--font-mono, monospace);
}
.seed-datasource-note strong {
  display: block;
  font-size: 0.86rem;
  color: var(--foreground);
  margin-bottom: 4px;
}
.seed-datasource-note p {
  margin: 0;
  font-size: 0.78rem;
  line-height: 1.6;
  color: var(--muted-foreground);
}

/* ═══ KPI 卡片行 ═══ */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
}
.kpi-card {
  position: relative;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 18px 18px 16px 22px;
  overflow: hidden;
  transition: border-color 160ms ease, transform 160ms ease;
}
.kpi-card:hover {
  border-color: color-mix(in srgb, var(--primary) 30%, var(--border));
  transform: translateY(-2px);
}
.kpi-card::before {
  content: "";
  position: absolute;
  left: 0;
  top: 10px;
  bottom: 10px;
  width: 4px;
  background: var(--kpi-color, var(--chart-1));
  border-radius: 0 4px 4px 0;
}
.kpi-label {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 0.8rem;
  color: var(--muted-foreground);
  margin-bottom: 10px;
}
.kpi-icon {
  width: 16px;
  height: 16px;
  flex: none;
  color: var(--kpi-color, var(--chart-1));
  stroke: currentColor;
  fill: none;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.kpi-value {
  font-size: 2rem;
  font-weight: 700;
  color: var(--foreground);
  letter-spacing: -0.02em;
  line-height: 1.1;
  font-variant-numeric: tabular-nums;
}
.kpi-trend {
  margin-top: 8px;
  font-size: 0.76rem;
  font-weight: 600;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.kpi-src {
  font-weight: 400;
  color: var(--muted-foreground);
}
.trend-stable { color: var(--muted-foreground); }

/* ═══ 响应式 ═══ */
@media (max-width: 1180px) {
  .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
  .kpi-grid { grid-template-columns: 1fr; }
  .scroll-area { padding: 18px; gap: 16px; }
  .topbar { padding: 0 14px; }
}
</style>
