<template>
  <main class="chat-workbench">
    <!-- 顶栏：三栏布局，左侧 spacer，中间标题，右侧 vitals -->
    <header class="topbar">
      <div class="topbar-spacer"></div>
      <div class="topbar-center">
        <div class="topbar-title">{{ chatStore.currentSessionName || '新对话' }}</div>
      </div>
      <div class="topbar-spacer"></div>
      <div class="vitals" aria-label="生命体征">
        <span v-for="v in vitalChips" :key="v.label" class="vital-chip" :title="v.label">
          <span class="vdot" :class="v.dot"></span>
          <span class="vlabel">{{ v.label }}</span>
          <span class="vvalue">{{ v.value }}</span>
        </span>
      </div>
    </header>

    <div ref="messagesArea" class="scroll-area chat-scroll">
      <div class="chat-stage">
        <ChatMessageList
          :messages="chatStore.messages"
          :displayed-messages="displayedMessages"
          :quick-hints="quickHints"
          :runtime-notice="runtimeNotice"
          :language-provider-notice="runtimeStore.languageProviderNotice"
          :connection-class="runtimeStore.connectionClass"
          :connection-status="runtimeStore.connectionStatus"
          :is-taiji="runtimeStore.health.isTaiji"
          :runtime-state="runtimeStore.health.state"
          :show-example="showExample"
          :has-more-messages="hasMoreMessages"
          :message-limit="messageLimit"
          :is-loading="chatStore.isLoading"
          :is-receiving="chatStore.isReceiving"
          @select-hint="chatStore.chatInput = $event"
          @toggle-example="showExample = !showExample"
          @show-more="showMore"
          @copy="copyMsg"
          @like="likeMsg"
          @regenerate="chatStore.regenerateMessage"
        />
        <WorkbenchTaskCard
          v-if="workbenchMode || workbench.interpretation.value || workbench.plan.value || workbench.execution.value"
          :interpretation="workbench.interpretation.value"
          :plan="workbench.plan.value"
          :approval="workbench.approval.value"
          :approval-tokens="approvalTokens"
          :execution="workbench.execution.value"
          :busy="workbench.busy.value"
          :error="workbench.error.value"
          @approve="approveWorkbenchRequest"
          @execute="executeWorkbenchPlan"
          @reset="resetWorkbench"
        />
      </div>

      <ChatComposer
        ref="composerRef"
        :model-value="chatStore.chatInput"
        :can-send="canSend"
        :is-receiving="chatStore.isReceiving"
        :uploading="uploading"
        :quick-hints="quickHints"
        :prompt-templates="promptTemplates"
        :workbench-mode="workbenchMode"
        @update:model-value="chatStore.chatInput = $event"
        @send="handleSend"
        @stop="chatStore.stopGeneration()"
        @files-picked="onFilesPicked"
        @insert-template="insertTemplate"
        @apply-quick-hint="applyQuickHint"
        @toggle-workbench="toggleWorkbench"
      />
    </div>
  </main>
</template>

<script setup>
import { computed, inject, nextTick, onMounted, ref, watch } from 'vue'
import { Brain, Bug, GitBranch, LineChart, ScrollText, SlidersHorizontal } from 'lucide-vue-next'
import ChatComposer from './ChatComposer.vue'
import ChatMessageList from './ChatMessageList.vue'
import WorkbenchTaskCard from './WorkbenchTaskCard.vue'
import { nativeApi } from '@/composables/nativeApi.js'
import { useChatStore } from '@/stores/chatStore.js'
import { useNaturalLanguageWorkbench } from '@/composables/useNaturalLanguageWorkbench.js'
import { useRuntimeStore } from '@/stores/runtimeStore.js'

defineOptions({ name: 'ChatView' })

const chatStore = useChatStore()
const runtimeStore = useRuntimeStore()
const workbench = useNaturalLanguageWorkbench()
const toast = inject('toast', () => {})

const messagesArea = ref(null)
const composerRef = ref(null)
const showExample = ref(false)
const uploading = ref(false)
const workbenchMode = ref(false)
const approvalTokens = ref({})

const vitalChips = computed(() => {
  const isTaiji = runtimeStore.health.isTaiji
  return [
    { dot: 'c1', label: '运行时', value: isTaiji ? 'Taiji Native' : 'Seed runtime' },
    { dot: 'c2', label: '输入', value: isTaiji ? 'raw bytes' : '对照流' },
    { dot: 'c3', label: '学习', value: isTaiji ? '局部可塑性' : '状态不可用' },
  ]
})
const runtimeNotice = computed(() => runtimeStore.runtimeNotice)
const quickHints = [
  { icon: Brain, text: '解释 Taiji 的原始字节输入' },
  { icon: Bug, text: '查看一次状态推进与输出' },
  { icon: SlidersHorizontal, text: '生成 Taiji 原生训练配置' },
  { icon: LineChart, text: '分析局部预测误差' },
  { icon: GitBranch, text: '解释情景场如何巩固记忆' },
  { icon: ScrollText, text: '查看 action → outcome 闭环' },
]
const canSend = computed(() =>
  !!chatStore.chatInput.trim() && !chatStore.isLoading && runtimeStore.health.state === 'connected' && runtimeStore.health.modelLoaded
)

function scrollToBottom() {
  nextTick(() => {
    if (messagesArea.value) messagesArea.value.scrollTop = messagesArea.value.scrollHeight
  })
}
watch(() => chatStore.messages.length, scrollToBottom)
watch(() => chatStore.isReceiving, scrollToBottom)

const messagePageSize = 50
const messageLimit = ref(messagePageSize)
const displayedMessages = computed(() => {
  const msgs = chatStore.messages
  return msgs.length > messageLimit.value ? msgs.slice(-messageLimit.value) : msgs
})
const hasMoreMessages = computed(() => chatStore.messages.length > messageLimit.value)
function showMore() { messageLimit.value += messagePageSize }

function handleSend() {
  if (!canSend.value) {
    if (chatStore.isLoading) {
      toast('正在生成回复，请稍候或点击「中断执行」', 'info')
    } else if (runtimeStore.health.state !== 'connected') {
      toast(`运行时未就绪（${runtimeStore.connectionStatus}），请等待连接恢复`, 'warning')
    } else if (!runtimeStore.health.modelLoaded) {
      toast('Taiji 原生运行时尚未就绪，暂时无法发送', 'warning')
    } else if (!chatStore.chatInput.trim()) {
      composerRef.value?.focus()
    }
    return
  }
  if (workbenchMode.value) {
    void startWorkbenchTask()
  } else {
    chatStore.sendMessage()
  }
  scrollToBottom()
}

async function startWorkbenchTask() {
  const prompt = chatStore.chatInput.trim()
  if (!prompt || workbench.busy.value) return
  const history = []
  let pendingUser = ''
  for (const msg of chatStore.messages) {
    if (msg.role === 'user') pendingUser = msg.content || ''
    else if (msg.role === 'assistant' && pendingUser) {
      history.push([pendingUser, msg.content || ''])
      pendingUser = ''
    }
  }
  chatStore.appendRuntimeMessage('user', prompt)
  chatStore.chatInput = ''
  approvalTokens.value = {}
  try {
    const interpretation = await workbench.interpretTask({ prompt, history })
    if (interpretation?.provider_evidence && interpretation?.decomposition?.steps?.length) {
      const capabilities = await nativeApi.workbenchCapabilities()
      await workbench.planTask({
        prompt,
        semantic_evidence: interpretation.provider_evidence,
        snapshot_id: capabilities.snapshot_id,
        loop_id: `chat-workbench:${interpretation.interpretation?.interpretation_id || prompt}`,
        max_steps: interpretation.decomposition.steps.length,
        max_budget_units: 1,
        novelty: 0,
        resource_budget: 1,
      })
    }
  } catch {
    toast(workbench.error.value || 'Taiji 工作台无法接收当前任务', 'error')
  }
}

async function approveWorkbenchRequest(requestId) {
  try {
    const result = await workbench.approveRequest(requestId)
    if (result?.approval_token) {
      approvalTokens.value = { ...approvalTokens.value, [requestId]: result.approval_token }
    }
  } catch {
    toast(workbench.error.value || '工作台审批失败', 'error')
  }
}

async function executeWorkbenchPlan() {
  try {
    await workbench.executePlan(approvalTokens.value)
  } catch {
    toast(workbench.error.value || '工作台执行失败', 'error')
  }
}

function toggleWorkbench() {
  workbenchMode.value = !workbenchMode.value
}

function resetWorkbench() {
  workbenchMode.value = false
  approvalTokens.value = {}
  workbench.reset()
}

async function copyMsg(content) {
  try {
    await navigator.clipboard.writeText(content)
    toast('已复制', 'success')
  } catch {
    toast('复制失败', 'error')
  }
}
function likeMsg() { toast('已点赞', 'success') }

const promptTemplates = {
  code: '请帮我解释以下代码：\n```\n\n```',
  summarize: '请帮我总结以下内容：\n',
  translate: '请将以下内容翻译成中文：\n',
}

function insertTemplate(text) {
  chatStore.chatInput = chatStore.chatInput.trim() ? `${chatStore.chatInput}\n${text}` : text
}
function applyQuickHint(text) { chatStore.chatInput = text }

// 附件上传由父视图持有 native API 副作用，composer 只负责传递用户选中的文件。
async function onFilesPicked(files) {
  if (!files.length) return
  uploading.value = true
  try {
    for (const file of files) {
      const formData = new FormData()
      formData.append('file', file)
      const res = await nativeApi.chatUpload(formData)
      const data = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(data.detail || `上传失败 (${res.status})`)
      const block = `【附件：${file.name}】\n${data.parsed_text || ''}`.trimEnd()
      chatStore.chatInput = chatStore.chatInput ? `${chatStore.chatInput}\n${block}` : block
    }
    toast(`已附加 ${files.length} 个附件`, 'success')
  } catch (err) {
    toast(`附件上传失败：${err.message}`, 'error')
  } finally {
    uploading.value = false
  }
}

onMounted(scrollToBottom)
</script>

<style scoped>
/* ===== 主布局 ===== */
.chat-workbench { display: flex; flex-direction: column; height: 100%; min-height: 0; background: var(--background); color: var(--foreground); }

/* 不画 border-bottom：外围边框由 .router-wrapper 独占（见 styles/shell.css） */
.topbar { height: 52px; flex: none; padding: 0 18px; display: flex; align-items: center; gap: 12px; background: transparent; position: relative; z-index: 5; }
.topbar-spacer { flex: 1; }
.topbar-center { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); text-align: center; pointer-events: none; }
.topbar-title { font-size: 0.92rem; font-weight: 600; color: var(--foreground); }

/* 生命体征芯片（顶栏右侧） */
.vitals { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }
.vital-chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px 4px 8px; border-radius: 999px; background: var(--muted); border: 1px solid var(--border); font-size: 0.72rem; line-height: 1; color: var(--foreground); white-space: nowrap; }
.vital-chip .vdot { width: 7px; height: 7px; border-radius: 50%; flex: none; box-shadow: 0 0 6px 0 currentColor; }
.vital-chip .vdot.c1 { background: var(--chart-1); color: var(--chart-1); }
.vital-chip .vdot.c2 { background: var(--chart-2); color: var(--chart-2); }
.vital-chip .vdot.c3 { background: var(--chart-3); color: var(--chart-3); }
.vital-chip .vdot.c-danger { background: var(--destructive); color: var(--destructive); }
.vital-chip .vlabel { color: var(--muted-foreground); }
.vital-chip .vvalue { font-weight: 600; font-variant-numeric: tabular-nums; }

/* ===== 滚动区 ===== */
.scroll-area { flex: 1; min-height: 0; overflow-y: auto; scroll-behavior: smooth; }
.scroll-area.chat-scroll { display: flex; flex-direction: column; position: relative; }
.chat-stage { flex: 1 0 auto; max-width: 780px; width: 100%; margin: 0 auto; padding: 44px 28px 16px; display: flex; flex-direction: column; gap: 30px; }

@media (max-width: 1080px) { .vitals { display: none; } }
@media (max-width: 880px) {
  .topbar-center { display: none; }
  .chat-stage { padding: 32px 18px 12px; }
}
</style>
