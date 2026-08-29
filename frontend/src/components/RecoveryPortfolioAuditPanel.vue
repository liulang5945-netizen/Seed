<!--
Recovery portfolio 客户端审计回放视图（plans/active/roadmap/04_EXECUTION_PLAN.md §2）。

只读：唯一数据路径是 refreshRecoveryPortfolioContext + refreshRecoveryPortfolio（两个 GET）；
本组件不调用 maintain/register/select/execute/preview，也不展示 parameters / evidence /
可直接复用的执行输入。绑定键来自服务端只读 context 投影（parent loop / snapshot / revision），
不来自输入框、固定 loop id 或「最近一次」猜测。

stale / revision mismatch 时保留最后一个已验证快照并标记过期；切换 parent loop 或卸载时
清除关联状态，避免跨循环串读。
-->
<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useWorkbenchProjection } from '../composables/useWorkbenchProjection.js'

const workbench = useWorkbenchProjection()

// —— 本地只读状态（唯一权威永远在服务端投影里） ——
const binding = ref({ parentLoopId: '', revision: null })
const lastValid = ref(null)
const stale = ref(false)
const loadError = ref(null)
const loading = ref(true)
let requestSeq = 0
let disposed = false

const context = computed(() => workbench.recoveryContext.value)
const hasPortfolio = computed(() => context.value?.has_portfolio === true)

const snapshot = computed(() => lastValid.value)
const errorCode = computed(() => loadError.value?.code || '')

const STATUS_LABELS = {
  active: '活动',
  selected: '已选中',
  completed: '已完成',
  failed: '失败',
  expired: '已过期',
  evicted: '已逐出',
}

function shortDigest(value) {
  if (!value) return '—'
  return value.length > 12 ? `${value.slice(0, 6)}…${value.slice(-6)}` : value
}

async function fetchPortfolio() {
  if (!binding.value.parentLoopId) return
  const seq = ++requestSeq
  loading.value = true
  try {
    const payload = await workbench.refreshRecoveryPortfolio(
      binding.value.parentLoopId,
      binding.value.revision,
    )
    if (disposed || seq !== requestSeq) return
    if (!payload || !payload.revision || !payload.branches) {
      throw new Error('portfolio_unavailable')
    }
    lastValid.value = payload
    stale.value = false
    loadError.value = null
    binding.value.revision = payload.revision
  } catch (cause) {
    if (disposed || seq !== requestSeq) return
    const code = cause?.message || 'portfolio_unavailable'
    if (code === 'portfolio_revision_stale') {
      // 服务端 revision 已推移：保留最后一个已验证快照，仅标记过期
      stale.value = true
      loadError.value = null
    } else {
      stale.value = false
      loadError.value = { code, message: String(cause?.message || cause) }
    }
  } finally {
    if (!disposed && seq === requestSeq) loading.value = false
  }
}

async function rebind() {
  if (!workbench.snapshotId.value) return
  const seq = ++requestSeq
  // 切换绑定前先清空上一循环的关联状态，避免跨循环串读（§2.2）
  lastValid.value = null
  stale.value = false
  loadError.value = null
  loading.value = true
  try {
    const ctx = await workbench.refreshRecoveryPortfolioContext()
    if (disposed || seq !== requestSeq) return
    if (ctx?.has_portfolio !== true) {
      // 无 portfolio / 已卸载链路：清空关联状态（§2.2 空态与切循环清除）
      binding.value = { parentLoopId: '', revision: null }
      loading.value = false
      return
    }
    binding.value = {
      parentLoopId: ctx.parent_loop_id,
      revision: ctx.revision,
    }
    loading.value = false
    await fetchPortfolio()
  } catch (cause) {
    if (disposed || seq !== requestSeq) return
    stale.value = false
    loadError.value = { code: cause?.message || 'portfolio_unavailable', message: String(cause?.message || cause) }
    loading.value = false
  }
}

// 事件投影每 2s 轮询一次；portfolio 出现/被替换都从这里被感知，
// 避免给审计视图再加一套独立轮询。
watch(
  () => [workbench.snapshotId.value, workbench.events.value.length],
  () => {
    if (disposed) return
    if (!context.value) {
      rebind()
      return
    }
    if (context.value.has_portfolio && context.value.revision !== binding.value.revision) {
      fetchPortfolio()
    }
  },
)

// parent loop 变化 → 重新绑定（rebind 会在取回新 context 后重取 portfolio）
watch(
  () => context.value?.parent_loop_id || '',
  (next, previous) => {
    if (next === previous) return
    rebind()
  },
)

onBeforeUnmount(() => {
  disposed = true
  lastValid.value = null
  stale.value = false
  loadError.value = null
})

// 首次挂载即绑定
rebind()
</script>

<template>
  <div class="recovery-audit" data-audit-panel>
    <div class="prop-group-title">恢复组合审计</div>

    <!-- 无 portfolio：结构化空态 -->
    <div v-if="!hasPortfolio && !loadError" class="prop-hint">
      {{ loading ? '读取 recovery portfolio 上下文…' : '无 recovery portfolio（等待 Taiji 恢复链路）' }}
    </div>

    <!-- 无法取到任何已验证快照的结构化错误 -->
    <div v-else-if="!snapshot && loadError" class="audit-error" data-audit-error>
      <span class="audit-error-code">{{ errorCode }}</span>
      <span class="audit-error-message">{{ loadError.message }}</span>
    </div>

    <template v-else-if="snapshot">
      <!-- stale：保留最后一个已验证快照，仅标记过期 -->
      <div v-if="stale" class="audit-stale" data-audit-stale>
        快照已过期（服务端 revision 已推移，等待下一次验证）
      </div>
      <div v-else-if="loadError" class="audit-error" data-audit-error>
        <span class="audit-error-code">{{ errorCode }}</span>
      </div>

      <!-- §2.2-1 快照元数据与新鲜度 -->
      <div class="prop-row"><span class="prop-label">快照</span><span class="prop-value audit-mono" :title="snapshot.snapshot_id">{{ shortDigest(snapshot.snapshot_id) }}</span></div>
      <div class="prop-row"><span class="prop-label">revision</span><span class="prop-value audit-mono">#{{ snapshot.revision }}</span></div>
      <div class="prop-row"><span class="prop-label">当前 tick</span><span class="prop-value">{{ snapshot.current_tick }}</span></div>
      <div class="prop-row"><span class="prop-label">容量</span><span class="prop-value">{{ snapshot.counts ? Object.values(snapshot.counts).reduce((a, b) => a + b, 0) : 0 }} / {{ snapshot.max_branches }}</span></div>
      <div class="prop-row"><span class="prop-label">TTL</span><span class="prop-value">{{ snapshot.branch_ttl_ticks }} ticks</span></div>
      <div class="prop-row"><span class="prop-label">末次维护</span><span class="prop-value">tick {{ snapshot.last_maintenance_tick }}</span></div>
      <div class="prop-row"><span class="prop-label">已选分支</span><span class="prop-value audit-mono">{{ shortDigest(snapshot.selected_branch_id) }}</span></div>

      <!-- §2.2-2 分支生命周期与 lineage -->
      <div v-if="snapshot.branches && snapshot.branches.length" class="prop-group-title">分支（{{ snapshot.branches.length }}）</div>
      <div v-for="branch in snapshot.branches || []" :key="branch.branch_id" class="audit-branch" :data-status="branch.status">
        <div class="audit-branch-head">
          <span class="audit-status" :class="`status-${branch.status}`">{{ STATUS_LABELS[branch.status] || branch.status }}</span>
          <span class="prop-value audit-mono" :title="branch.branch_id">{{ shortDigest(branch.branch_id) }}</span>
        </div>
        <div class="prop-row"><span class="prop-label">loop</span><span class="prop-value audit-mono">{{ shortDigest(branch.loop_id) }}</span></div>
        <div class="prop-row"><span class="prop-label">能力</span><span class="prop-value">{{ branch.capability_id }}</span></div>
        <div class="prop-row"><span class="prop-label">证据</span><span class="prop-value audit-mono">{{ shortDigest(branch.source_evidence_id) }}</span></div>
        <div class="prop-row"><span class="prop-label">后继态</span><span class="prop-value audit-mono">{{ shortDigest(branch.source_after_state_digest) }}</span></div>
        <div class="prop-row"><span class="prop-label">预算</span><span class="prop-value">{{ branch.budget_units }} / {{ branch.budget_limit }}</span></div>
        <div class="prop-row"><span class="prop-label">步骤</span><span class="prop-value">{{ branch.completed_steps }}</span></div>
        <div class="prop-row"><span class="prop-label">frontier</span><span class="prop-value">{{ (branch.frontier_affordance_ids || []).length }} 个</span></div>
        <div class="prop-row"><span class="prop-label">存活</span><span class="prop-value">tick {{ branch.created_tick }} → {{ branch.expires_at_tick }}</span></div>
        <div v-if="branch.terminal_reason" class="prop-row"><span class="prop-label">终止</span><span class="prop-value">{{ branch.terminal_reason }}</span></div>
      </div>

      <!-- §2.2-3 eviction tombstone：原因与次序可审计，无可执行细节 -->
      <div v-if="snapshot.evicted_branches && snapshot.evicted_branches.length" class="prop-group-title">逐出墓碑（{{ snapshot.evicted_branches.length }}）</div>
      <div v-for="tomb in snapshot.evicted_branches || []" :key="tomb.branch_id" class="audit-tombstone">
        <span class="audit-status status-evicted">{{ STATUS_LABELS.evicted }}</span>
        <span class="audit-mono" :title="tomb.branch_id">{{ shortDigest(tomb.branch_id) }}</span>
        <span class="audit-mono" :title="tomb.source_after_state_digest" style="margin-left: 6px; color: var(--text-dim, #888)">{{ shortDigest(tomb.source_after_state_digest) }}</span>
        <div class="prop-row"><span class="prop-label">原因</span><span class="prop-value">{{ tomb.reason }}</span></div>
        <div class="prop-row"><span class="prop-label">逐出 tick</span><span class="prop-value">{{ tomb.evicted_tick }}</span></div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.recovery-audit {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.audit-mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
}
.audit-status {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 8px;
  font-size: 11px;
  line-height: 16px;
}
.status-active { background: rgba(64, 158, 255, 0.15); color: #409eff; }
.status-selected { background: rgba(103, 194, 58, 0.15); color: #67c23a; }
.status-completed { background: rgba(144, 147, 153, 0.15); color: #909399; }
.status-failed { background: rgba(245, 108, 108, 0.15); color: #f56c6c; }
.status-expired { background: rgba(230, 162, 60, 0.15); color: #e6a23c; }
.status-evicted { background: rgba(245, 108, 108, 0.12); color: #f56c6c; }
.audit-branch, .audit-tombstone {
  border-left: 2px solid rgba(128, 128, 128, 0.25);
  padding-left: 8px;
  margin: 4px 0;
}
.audit-branch-head {
  display: flex;
  align-items: center;
  gap: 6px;
}
.audit-stale, .audit-error {
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
}
.audit-stale {
  background: rgba(230, 162, 60, 0.12);
  color: #e6a23c;
}
.audit-error {
  background: rgba(245, 108, 108, 0.1);
  color: #f56c6c;
  display: flex;
  gap: 6px;
  align-items: baseline;
}
.audit-error-code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;
  opacity: 0.8;
}
</style>