<template>
  <aside class="workbench-task-card" aria-label="Taiji 工作台任务">
    <div class="card-head">
      <div class="card-title-wrap">
        <span class="card-mark" aria-hidden="true"><span></span></span>
        <div>
          <p class="eyebrow">TAIJI WORKBENCH</p>
          <h2>工作台任务</h2>
        </div>
      </div>
      <button class="quiet-button" type="button" title="关闭工作台任务" @click="emit('reset')">关闭</button>
    </div>

    <div v-if="error" class="task-error" role="alert">{{ error }}</div>

    <section v-if="interpretation" class="task-intake">
      <div class="section-label"><span class="status-dot ready"></span>Taiji 目标证据</div>
      <p class="goal-copy">{{ goalDescription }}</p>
      <div class="evidence-meta">
        <span>{{ interpretationStatus }}</span>
        <span v-if="interpretation.goal?.goal_id">{{ interpretation.goal.goal_id }}</span>
      </div>
    </section>

    <section v-if="interpretation && !plan" class="boundary-note">
      <span class="status-dot waiting"></span>
      <div>
        <strong>{{ semanticSteps.length ? '等待 Taiji grounding' : '等待语义器官' }}</strong>
        <p v-if="semanticSteps.length">语义 provider 已提交候选步骤，Taiji 将根据实时 Workbench 证据决定是否形成计划；此处仍未产生工具调用或执行副作用。</p>
        <p v-else>Taiji 已接收任务，但当前没有可用的语义 provider 来生成可审查的工作台步骤。语言器官只负责表达，不会替代任务理解或工具选择。</p>
      </div>
    </section>

    <section v-if="semanticSteps.length" class="semantic-preview">
      <div class="section-label"><span class="status-dot ready"></span>语义步骤证据</div>
      <div v-for="(step, index) in semanticSteps" :key="step.step_id || index" class="semantic-step">
        <span class="step-index">{{ index + 1 }}</span>
        <span>{{ step.description || `语义步骤 ${index + 1}` }}</span>
      </div>
    </section>

    <section v-if="plan" class="task-plan">
      <div class="section-label"><span class="status-dot ready"></span>Taiji 执行计划</div>
      <div v-if="plan.planning?.steps?.length" class="step-list">
        <div v-for="(step, index) in plan.planning.steps" :key="step.step_id || index" class="step-row">
          <span class="step-index">{{ index + 1 }}</span>
          <div class="step-copy">
            <strong>{{ step.step_id || `语义步骤 ${index + 1}` }}</strong>
            <span>{{ step.grounding?.[0]?.action_kind || '待 Taiji grounding' }}</span>
          </div>
        </div>
      </div>

      <div v-if="approvalRequirements.length" class="approval-list">
        <p class="sub-label">需要你的确认</p>
        <div v-for="requirement in approvalRequirements" :key="requirement.request_id" class="approval-row">
          <div>
            <strong>{{ requirement.capability_id }}</strong>
            <span>{{ previewLabel(requirement.preview) }}</span>
          </div>
          <button
            class="action-button secondary"
            type="button"
            :disabled="busy || isApproved(requirement.request_id)"
            @click="emit('approve', requirement.request_id)"
          >
            {{ isApproved(requirement.request_id) ? '已确认' : '确认' }}
          </button>
        </div>
      </div>

      <div class="plan-actions">
        <span class="plan-status" :class="statusClass">{{ planStatus }}</span>
        <button
          v-if="canExecute"
          class="action-button primary"
          type="button"
          :disabled="busy"
          @click="emit('execute')"
        >
          {{ busy ? '处理中…' : '执行计划' }}
        </button>
      </div>
    </section>

    <section v-if="execution" class="execution-result" :class="executionClass">
      <span class="status-dot" :class="executionClass"></span>
      <div>
        <strong>{{ executionTitle }}</strong>
        <p>{{ executionSummary }}</p>
      </div>
    </section>

    <div v-if="busy && !execution" class="progress-line" aria-label="Taiji 正在处理"></div>
  </aside>
</template>

<script setup>
import { computed } from 'vue'

defineOptions({ name: 'WorkbenchTaskCard' })

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
.workbench-task-card { position: relative; overflow: hidden; padding: 16px 18px; border: 1px solid color-mix(in srgb, var(--primary) 26%, var(--border)); border-radius: 18px; background: linear-gradient(135deg, color-mix(in srgb, var(--primary) 7%, var(--card)), var(--card)); box-shadow: 0 8px 26px color-mix(in srgb, var(--primary) 8%, transparent); }
.card-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.card-title-wrap { display: flex; align-items: center; gap: 10px; }
.card-mark { width: 30px; height: 30px; display: grid; place-items: center; border: 1px solid color-mix(in srgb, var(--primary) 30%, var(--border)); border-radius: 10px; background: color-mix(in srgb, var(--primary) 12%, transparent); }
.card-mark::before, .card-mark::after, .card-mark span { content: ''; width: 13px; height: 1px; position: absolute; background: var(--primary); transform-origin: center; }
.card-mark { position: relative; }
.card-mark::before { transform: rotate(60deg); }
.card-mark::after { transform: rotate(-60deg); }
.card-mark span { transform: rotate(0deg); }
.eyebrow { margin: 0 0 2px; color: var(--primary); font: 600 0.62rem/1 var(--font-mono); letter-spacing: 0.11em; }
.card-head h2 { margin: 0; font-size: 0.94rem; font-weight: 700; }
.quiet-button { border: 0; padding: 5px 8px; border-radius: 7px; color: var(--muted-foreground); background: transparent; cursor: pointer; font-size: 0.74rem; }
.quiet-button:hover { color: var(--foreground); background: var(--muted); }
.task-intake, .task-plan { margin-top: 15px; padding-top: 13px; border-top: 1px solid color-mix(in srgb, var(--border) 78%, transparent); }
.section-label, .sub-label { color: var(--muted-foreground); font-size: 0.7rem; font-weight: 650; }
.section-label { display: flex; align-items: center; gap: 7px; }
.goal-copy { margin: 8px 0 0; color: var(--foreground); font-size: 0.88rem; line-height: 1.5; }
.evidence-meta { display: flex; gap: 9px; margin-top: 7px; color: var(--muted-foreground); font: 0.68rem/1.4 var(--font-mono); }
.boundary-note, .execution-result { display: flex; align-items: flex-start; gap: 9px; margin-top: 13px; padding: 10px 11px; border-radius: 11px; background: color-mix(in srgb, var(--muted) 68%, transparent); }
.boundary-note strong, .execution-result strong { display: block; font-size: 0.76rem; }
.boundary-note p, .execution-result p { margin: 3px 0 0; color: var(--muted-foreground); font-size: 0.72rem; line-height: 1.5; }
.semantic-preview { margin-top: 13px; padding-top: 13px; border-top: 1px solid color-mix(in srgb, var(--border) 78%, transparent); }
.semantic-step { display: flex; align-items: center; gap: 8px; margin-top: 8px; color: var(--foreground); font-size: 0.74rem; }
.status-dot { width: 7px; height: 7px; margin-top: 5px; border-radius: 50%; flex: none; background: var(--muted-foreground); }
.status-dot.ready, .status-dot.success { background: var(--success, var(--chart-2)); box-shadow: 0 0 8px color-mix(in srgb, var(--success, var(--chart-2)) 55%, transparent); }
.status-dot.waiting { background: var(--warning, var(--chart-3)); }
.status-dot.danger { background: var(--destructive); }
.step-list { display: grid; gap: 7px; margin-top: 10px; }
.step-row { display: flex; align-items: center; gap: 8px; }
.step-index { width: 19px; height: 19px; display: grid; place-items: center; border-radius: 50%; color: var(--primary); background: color-mix(in srgb, var(--primary) 13%, transparent); font: 700 0.65rem/1 var(--font-mono); }
.step-copy { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.step-copy strong { overflow: hidden; color: var(--foreground); font-size: 0.74rem; text-overflow: ellipsis; white-space: nowrap; }
.step-copy span { color: var(--muted-foreground); font: 0.67rem/1.2 var(--font-mono); }
.approval-list { margin-top: 13px; }
.sub-label { margin: 0 0 7px; }
.approval-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 9px 10px; border: 1px solid var(--border); border-radius: 11px; background: color-mix(in srgb, var(--card) 70%, transparent); }
.approval-row > div { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.approval-row strong { color: var(--foreground); font: 600 0.72rem/1.2 var(--font-mono); }
.approval-row span { overflow: hidden; color: var(--muted-foreground); font-size: 0.7rem; text-overflow: ellipsis; white-space: nowrap; }
.plan-actions { display: flex; align-items: center; justify-content: flex-end; gap: 10px; margin-top: 13px; }
.plan-status { margin-right: auto; font-size: 0.7rem; }
.plan-status.waiting { color: var(--warning, var(--chart-3)); }
.plan-status.danger { color: var(--destructive); }
.action-button { min-height: 29px; padding: 0 11px; border-radius: 8px; cursor: pointer; font-size: 0.72rem; font-weight: 650; }
.action-button.primary { border: 1px solid var(--primary); color: var(--primary-foreground); background: var(--primary); }
.action-button.secondary { border: 1px solid var(--border); color: var(--foreground); background: var(--muted); }
.action-button:hover:not(:disabled) { filter: brightness(1.05); transform: translateY(-1px); }
.action-button:disabled { opacity: 0.48; cursor: not-allowed; }
.task-error { margin-top: 11px; padding: 8px 10px; border-radius: 9px; color: var(--destructive); background: color-mix(in srgb, var(--destructive) 9%, transparent); font-size: 0.74rem; }
.progress-line { height: 2px; margin: 13px -18px -16px; background: linear-gradient(90deg, transparent, var(--primary), transparent); animation: workbench-progress 1.4s ease-in-out infinite; }
@keyframes workbench-progress { 0% { opacity: 0.25; transform: translateX(-35%); } 50% { opacity: 1; } 100% { opacity: 0.25; transform: translateX(35%); } }
@media (prefers-reduced-motion: reduce) { .progress-line { animation: none; opacity: 0.65; } .action-button:hover:not(:disabled) { transform: none; } }
</style>
