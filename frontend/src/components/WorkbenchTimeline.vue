<template>
  <section class="workbench-timeline" aria-label="Taiji 工作台执行轨迹">
    <!-- 头部 -->
    <div class="wt-head">
      <span class="wt-eyebrow">TAIJI WORKBENCH</span>
      <span class="wt-title">工作台轨迹</span>
      <button class="wt-close" type="button" title="关闭工作台轨迹" @click="emit('reset')">
        <X :size="13" />
      </button>
    </div>

    <div v-if="error" class="wt-error" role="alert">{{ error }}</div>

    <!-- append-only 时间线：阶段逐节追加，已出现的节点不消失 -->
    <ol class="wt-track">
      <!-- 节点 1：目标证据 -->
      <li v-if="interpretation" class="wt-node">
        <span class="wt-dot ready"></span>
        <div class="wt-body">
          <div class="wt-node-head">
            <strong>目标证据</strong>
            <span class="wt-meta">{{ interpretationStatus }}<template v-if="interpretation.goal?.goal_id"> · {{ interpretation.goal.goal_id }}</template></span>
          </div>
          <p class="wt-goal">{{ goalDescription }}</p>
        </div>
      </li>

      <!-- 节点 2：语义步骤 -->
      <li v-if="semanticSteps.length" class="wt-node">
        <span class="wt-dot ready"></span>
        <div class="wt-body">
          <div class="wt-node-head"><strong>语义步骤</strong><span class="wt-meta">{{ semanticSteps.length }} 步候选</span></div>
          <div class="wt-steps">
            <div v-for="(step, index) in semanticSteps" :key="step.step_id || index" class="wt-step">
              <span class="wt-step-index">{{ index + 1 }}</span>
              <span>{{ step.description || `语义步骤 ${index + 1}` }}</span>
            </div>
          </div>
        </div>
      </li>

      <!-- 节点 2'：无语义 provider 的边界说明 -->
      <li v-else-if="interpretation && !plan" class="wt-node">
        <span class="wt-dot waiting"></span>
        <div class="wt-body">
          <div class="wt-node-head"><strong>等待语义器官</strong></div>
          <p class="wt-note">Taiji 已接收任务，但当前没有可用的语义 provider 来生成可审查的工作台步骤。语言器官只负责表达，不会替代任务理解或工具选择。</p>
        </div>
      </li>

      <!-- 节点 3：执行计划 -->
      <li v-if="plan" class="wt-node">
        <span class="wt-dot" :class="plan?.status === 'rejected' ? 'danger' : 'ready'"></span>
        <div class="wt-body">
          <div class="wt-node-head">
            <strong>执行计划</strong>
            <span class="wt-meta" :class="statusClass">{{ planStatus }}</span>
          </div>
          <div v-if="plan.planning?.steps?.length" class="wt-steps">
            <div v-for="(step, index) in plan.planning.steps" :key="step.step_id || index" class="wt-step">
              <span class="wt-step-index">{{ index + 1 }}</span>
              <div class="wt-step-copy">
                <strong>{{ step.step_id || `语义步骤 ${index + 1}` }}</strong>
                <span>{{ step.grounding?.[0]?.action_kind || '待 Taiji grounding' }}</span>
              </div>
            </div>
          </div>

          <!-- 审批（有则追加） -->
          <div v-if="approvalRequirements.length" class="wt-approvals">
            <p class="wt-sub">需要你的确认</p>
            <div v-for="requirement in approvalRequirements" :key="requirement.request_id" class="wt-approval-row">
              <div class="wt-approval-copy">
                <strong>{{ requirement.capability_id }}</strong>
                <span>{{ previewLabel(requirement.preview) }}</span>
              </div>
              <button
                class="wt-action secondary"
                type="button"
                :disabled="busy || isApproved(requirement.request_id)"
                @click="emit('approve', requirement.request_id)"
              >
                {{ isApproved(requirement.request_id) ? '已确认' : '确认' }}
              </button>
            </div>
          </div>

          <div class="wt-actions">
            <button
              v-if="canExecute"
              class="wt-action primary"
              type="button"
              :disabled="busy"
              @click="emit('execute')"
            >
              {{ busy ? '处理中…' : '执行计划' }}
            </button>
          </div>
        </div>
      </li>

      <!-- 节点 4：执行结果 -->
      <li v-if="execution" class="wt-node">
        <span class="wt-dot" :class="executionClass"></span>
        <div class="wt-body">
          <div class="wt-node-head"><strong>{{ executionTitle }}</strong></div>
          <p class="wt-note">{{ executionSummary }}</p>
        </div>
      </li>

      <!-- 末端进行中节点 -->
      <li v-if="busy && !execution" class="wt-node wt-node-active">
        <span class="wt-dot pulse"></span>
        <div class="wt-body">
          <div class="wt-node-head"><strong>Taiji 处理中</strong></div>
          <p class="wt-note">正在推进当前阶段…</p>
        </div>
      </li>
    </ol>
  </section>
</template>

<script setup>
/**
 * dsh 式工作台轨迹：append-only 垂直时间线（替代原悬浮卡片 WorkbenchTaskCard）。
 *
 * 与消息流同列（ChatView 的 chat-stage 内），阶段随执行推进逐节追加——
 * 目标证据 → 语义步骤 → 执行计划（含审批/执行交互）→ 结果。左侧连线 +
 * 状态点语言与 ChatMessageList 的 workbench-trace 行统一。
 * Props/emits 与原卡片完全一致，ChatView 零逻辑改动替换。
 */
import { computed } from 'vue'
import { X } from 'lucide-vue-next'

defineOptions({ name: 'WorkbenchTimeline' })

const props = defineProps({
  interpretation: { type: Object, default: null },
  plan: { type: Object, default: null },
  approval: { type: Object, default: null },
  approvalTokens: { type: Object, default: () => ({}) },
  execution: { type: Object, default: null },
  busy: { type: Boolean, default: false },
  error: { type: String, default: '' },
})

const emit = defineEmits(['approve', 'execute', 'reset'])

const goalDescription = computed(() =>
  props.interpretation?.goal?.description
  || props.interpretation?.interpretation?.goal_description
  || 'Taiji 正在建立当前任务目标。'
)
const interpretationStatus = computed(() => ({
  resolved: '已解析',
  candidate: '候选',
  ambiguous: '有歧义',
}[props.interpretation?.interpretation?.status] || '已接收'))
const approvalRequirements = computed(() => props.plan?.approval_requirements || [])
const semanticSteps = computed(() => props.interpretation?.decomposition?.steps || [])
const canExecute = computed(() => Boolean(
  props.plan?.plan_id
  && approvalRequirements.value.every((item) => Boolean(props.approvalTokens[item.request_id]))
  && !props.execution
))
const planStatus = computed(() => ({
  needs_approval: '等待确认',
  planned: '可执行',
  rejected: '已拒绝',
}[props.plan?.status] || '计划已更新'))
const statusClass = computed(() => props.plan?.status === 'rejected' ? 'danger' : 'waiting')
const executionClass = computed(() => props.execution?.status === 'completed' ? 'success' : 'danger')
const executionTitle = computed(() => props.execution?.status === 'completed' ? '工作台执行完成' : '工作台执行未完成')
const executionSummary = computed(() => {
  if (props.execution?.status === 'completed') return '结果已由 Taiji 记录，可在工作台查看执行轨迹。'
  return props.execution?.reason_code || props.execution?.error_code || 'Taiji 已停止本次执行。'
})

function isApproved(requestId) {
  return Boolean(props.approvalTokens[requestId])
    || props.approval?.request_id === requestId
}

function previewLabel(preview) {
  if (!preview || typeof preview !== 'object') return 'Taiji 将在执行前再次校验当前状态'
  return preview.summary || preview.description || 'Taiji 将在执行前再次校验当前状态'
}
</script>

<style scoped>
.workbench-timeline {
  border: 1px solid color-mix(in srgb, var(--primary) 22%, var(--border));
  border-radius: 14px;
  background: color-mix(in srgb, var(--primary) 4%, var(--card));
  padding: 13px 16px 15px;
  overflow: hidden;
}

/* ── 头部 ── */
.wt-head {
  display: flex;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 11px;
}
.wt-eyebrow {
  color: var(--primary);
  font: 600 0.6rem/1 var(--font-mono, monospace);
  letter-spacing: 0.11em;
}
.wt-title { font-size: 0.8rem; font-weight: 650; color: var(--foreground); }
.wt-close {
  margin-left: auto;
  align-self: center;
  width: 24px; height: 24px;
  display: inline-flex; align-items: center; justify-content: center;
  border: 0; border-radius: 7px;
  color: var(--muted-foreground); background: transparent;
  cursor: pointer;
}
.wt-close:hover { color: var(--foreground); background: var(--muted); }
.wt-error {
  margin-bottom: 10px;
  padding: 7px 10px;
  border-radius: 9px;
  color: var(--destructive);
  background: color-mix(in srgb, var(--destructive) 9%, transparent);
  font-size: 0.74rem;
}

/* ── append-only 时间线轨道 ── */
.wt-track {
  list-style: none;
  margin: 0;
  padding: 0;
  position: relative;
}
/* 左侧连线：贯穿所有节点 */
.wt-track::before {
  content: '';
  position: absolute;
  left: 4px;
  top: 8px;
  bottom: 8px;
  width: 1.5px;
  background: color-mix(in srgb, var(--primary) 24%, var(--border));
}
.wt-node {
  position: relative;
  display: flex;
  gap: 11px;
  padding: 0 0 14px 0;
}
.wt-node:last-child { padding-bottom: 0; }

/* 状态点（覆盖在连线之上） */
.wt-dot {
  position: relative;
  z-index: 1;
  width: 9px; height: 9px;
  margin-top: 4px;
  border-radius: 50%;
  flex: none;
  background: var(--muted-foreground);
  box-shadow: 0 0 0 3px var(--card);
}
.wt-dot.ready { background: var(--success, var(--chart-2)); box-shadow: 0 0 0 3px var(--card), 0 0 8px color-mix(in srgb, var(--success, var(--chart-2)) 55%, transparent); }
.wt-dot.waiting { background: var(--warning, var(--chart-3)); box-shadow: 0 0 0 3px var(--card); }
.wt-dot.danger { background: var(--destructive); box-shadow: 0 0 0 3px var(--card); }
.wt-dot.pulse { background: var(--primary); animation: wt-pulse 1.3s ease-in-out infinite; }
@keyframes wt-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.35; }
}
@media (prefers-reduced-motion: reduce) {
  .wt-dot.pulse { animation: none; opacity: 0.65; }
}

/* 节点体 */
.wt-body { min-width: 0; flex: 1; }
.wt-node-head {
  display: flex;
  align-items: baseline;
  gap: 8px;
  flex-wrap: wrap;
}
.wt-node-head strong { font-size: 0.78rem; color: var(--foreground); }
.wt-meta {
  color: var(--muted-foreground);
  font: 0.66rem/1.4 var(--font-mono, monospace);
}
.wt-meta.waiting { color: var(--warning, var(--chart-3)); }
.wt-meta.danger { color: var(--destructive); }
.wt-goal { margin: 5px 0 0; color: var(--foreground); font-size: 0.84rem; line-height: 1.5; }
.wt-note { margin: 5px 0 0; color: var(--muted-foreground); font-size: 0.74rem; line-height: 1.5; }

/* 步骤列表 */
.wt-steps { display: grid; gap: 6px; margin-top: 8px; }
.wt-step {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--foreground);
  font-size: 0.74rem;
}
.wt-step-index {
  width: 18px; height: 18px;
  display: grid; place-items: center;
  border-radius: 50%;
  color: var(--primary);
  background: color-mix(in srgb, var(--primary) 13%, transparent);
  font: 700 0.62rem/1 var(--font-mono, monospace);
  flex: none;
}
.wt-step-copy { display: flex; flex-direction: column; gap: 1px; min-width: 0; }
.wt-step-copy strong { overflow: hidden; color: var(--foreground); font-size: 0.72rem; text-overflow: ellipsis; white-space: nowrap; }
.wt-step-copy span { color: var(--muted-foreground); font: 0.65rem/1.2 var(--font-mono, monospace); }

/* 审批 */
.wt-approvals { margin-top: 11px; }
.wt-sub { margin: 0 0 6px; color: var(--muted-foreground); font-size: 0.68rem; font-weight: 650; }
.wt-approval-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 10px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: color-mix(in srgb, var(--card) 70%, transparent);
}
.wt-approval-row + .wt-approval-row { margin-top: 6px; }
.wt-approval-copy { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.wt-approval-copy strong { color: var(--foreground); font: 600 0.7rem/1.2 var(--font-mono, monospace); }
.wt-approval-copy span { overflow: hidden; color: var(--muted-foreground); font-size: 0.68rem; text-overflow: ellipsis; white-space: nowrap; }

/* 动作 */
.wt-actions { display: flex; justify-content: flex-end; margin-top: 10px; }
.wt-actions:empty { display: none; }
.wt-action {
  min-height: 28px;
  padding: 0 12px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.72rem;
  font-weight: 650;
  font-family: inherit;
}
.wt-action.primary { border: 1px solid var(--primary); color: var(--primary-foreground); background: var(--primary); }
.wt-action.secondary { border: 1px solid var(--border); color: var(--foreground); background: var(--muted); }
.wt-action:hover:not(:disabled) { filter: brightness(1.05); }
.wt-action:disabled { opacity: 0.48; cursor: not-allowed; }
</style>
