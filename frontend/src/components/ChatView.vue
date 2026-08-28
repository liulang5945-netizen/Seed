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

    <RuntimeEvidenceStrip context="chat" compact />

    <!-- 对话滚动区 -->
    <div ref="messagesArea" class="scroll-area chat-scroll">
      <div class="chat-stage">
        <!-- 欢迎区 + 建议词 + 示例对话（无消息时显示） -->
        <template v-if="chatStore.messages.length === 0">
          <section class="chat-welcome">
            <div class="welcome-logo" aria-hidden="true">
              <TaijiLogo :size="72" :thinking="runtimeStore.health.state === 'connected'" />
            </div>
            <h1>有什么我能帮你的吗？</h1>
            <div class="welcome-sub">
              <span class="ok-dot"></span>
              {{ runtimeStore.connectionStatus }} · {{ runtimeStore.health.isTaiji ? 'Taiji Native 语言通路' : 'Seed 运行时' }}
            </div>
            <div v-if="runtimeStore.languageProviderNotice" class="provider-notice">
              <span class="runtime-notice-dot warning"></span>
              <span>{{ runtimeStore.languageProviderNotice.message }}</span>
            </div>
          </section>

          <!-- 建议词云 -->
          <div class="suggestions" role="list">
            <button v-for="hint in quickHints" :key="hint.text" class="suggestion" type="button" @click="chatStore.chatInput = hint.text">
              <component :is="hint.icon" :size="16" class="sicon" />
              <span>{{ hint.text }}</span>
            </button>
          </div>

          <!-- 示例对话：默认收起，点击展开第一段示例 -->
          <div class="chat-thread chat-thread-example">
            <button class="thread-divider example-toggle" type="button" :aria-expanded="showExample" @click="showExample = !showExample">
              <span class="example-toggle-text">{{ showExample ? '收起示例对话' : '查看示例对话' }}</span>
              <ChevronDown class="example-chevron" :class="{ open: showExample }" :size="14" />
            </button>

            <template v-if="showExample">
              <!-- 示例用户提问 -->
              <div class="msg msg-user">
                <span class="av av-user" aria-label="用户">
                  <User :size="16" />
                </span>
                <div class="msg-body">
                  <span class="msg-name">你</span>
                  <div class="bubble">
                    <p>Taiji 如何把原始字节推进成一次输出？</p>
                  </div>
                </div>
              </div>

              <!-- 示例 AI 回复（静态演示，不提供操作按钮） -->
              <div class="msg msg-ai">
                <TaijiLogo class="av av-ai" :size="32" :thinking="false" aria-label="Seed" />
                <div class="msg-body">
                  <span class="msg-name">Seed</span>
                  <div class="bubble">
                    <p><span class="lead">Taiji 的一次状态推进</span>输入先进入原生运行时，形成一帧带来源的输入证据；运行时推进持续状态与已接入的关联机制，最后交给语言器官形成可读表达。</p>
                    <ol class="msg-steps">
                      <li><strong>输入</strong>：把用户消息包装为带来源、时间和置信度的原生输入帧。</li>
                      <li><strong>状态</strong>：由当前 Taiji runtime 推进状态，并保留可审计的运行结果。</li>
                      <li><strong>表达</strong>：语言器官把表达计划形成可读文本；如果未形成，则明确显示原始输出。</li>
                    </ol>
                    <p>这里的智能不是某个固定器官名称或固定规模的宣传，而是来自状态、学习、环境反馈和表达边界协同后的可验证闭环。</p>
                  </div>
                </div>
              </div>
            </template>
          </div>
        </template>

        <!-- 消息线程（有消息时显示） -->
        <template v-else>
          <div class="chat-thread">
            <div v-if="runtimeNotice" class="runtime-notice" :class="runtimeStore.connectionClass">
              <span class="runtime-notice-dot"></span>
              <div>
                <strong>{{ runtimeNotice.title }}</strong>
                <p>{{ runtimeNotice.message }}</p>
              </div>
            </div>
            <div v-if="runtimeStore.languageProviderNotice" class="runtime-notice warning">
              <span class="runtime-notice-dot warning"></span>
              <div>
                <strong>{{ runtimeStore.languageProviderNotice.title }}</strong>
                <p>{{ runtimeStore.languageProviderNotice.message }}</p>
              </div>
            </div>

            <div v-if="hasMoreMessages" class="load-more-row">
              <button class="load-more-btn" @click="showMore">
                显示更多消息（{{ messageLimit }}/{{ chatStore.messages.length }}）
              </button>
            </div>

            <article
v-for="msg in displayedMessages" :key="msg.id"
              v-memo="[msg.id, msg.content, msg.role, msg.unreadable]"
              :class="['msg', msg.role === 'user' ? 'msg-user' : 'msg-ai']">
              <TaijiLogo v-if="msg.role === 'assistant'" class="av av-ai" :size="32" :thinking="false" aria-label="Seed" />
              <span v-else class="av av-user" aria-label="用户">
                <User :size="16" />
              </span>
              <div class="msg-body">
                <span class="msg-name">{{ msg.role === 'user' ? '你' : 'Seed' }}</span>
                <!-- 不可读的原始字节输出：以「原始输出」卡片呈现，不伪装成正常回复 -->
                <div v-if="msg.role === 'assistant' && isRawOutput(msg)" class="bubble raw-output">
                  <div class="raw-head">
                    <span class="raw-badge">RAW</span>
                    <span class="raw-title">语言器官原始输出</span>
                  </div>
                  <p class="raw-desc">当前语言表层未能形成可读文本，以下为调试输出：</p>
                  <pre class="raw-pre">{{ msg.content }}</pre>
                </div>
                <div v-else-if="msg.role === 'user'" class="bubble"><div class="text-content">{{ msg.content }}</div></div>
                <div v-else class="bubble markdown-body" v-html="renderMarkdown(msg.content)" />
                <div v-if="msg.role === 'assistant' && msg.content" class="msg-actions">
                  <button class="msg-action-btn" title="复制" @click="copyMsg(msg.content)"><Copy :size="14" /></button>
                  <button class="msg-action-btn" title="赞" @click="likeMsg(msg.id)"><ThumbsUp :size="14" /></button>
                  <button class="msg-action-btn" title="重新生成" @click="chatStore.regenerateMessage(msg.id)"><RotateCcw :size="14" /></button>
                </div>
              </div>
            </article>

            <article v-if="chatStore.isLoading" class="msg msg-ai thinking-row">
              <TaijiLogo class="av av-ai breathing" :size="32" :thinking="true" aria-label="Seed" />
              <div class="msg-body">
                <span class="msg-name">{{ chatStore.isReceiving ? 'Seed · 正在回应' : 'Seed · 正在启动' }}</span>
                <div v-if="!chatStore.isReceiving" class="bubble loading-bubble">
                  <span class="thinking-animation"><span class="think-dot"></span><span class="think-dot"></span><span class="think-dot"></span></span>
                </div>
              </div>
            </article>
          </div>
        </template>
      </div>

      <!-- 底部输入区：圆角胶囊形 + 工具芯片 -->
      <div class="composer-wrap">
        <div v-if="chatStore.isReceiving" class="stop-container">
          <button class="stop-btn" @click="chatStore.stopGeneration()">
            <Square :size="13" fill="currentColor" /> 中断执行
          </button>
        </div>

        <div class="composer">
          <textarea
ref="inputRef" v-model="chatStore.chatInput"
            :placeholder="inputPlaceholder"
            rows="1" @keydown="onKeydown" />
          <div class="tools">
            <!-- 快捷提问面板（由"快速"chip 展开，复用首屏建议词） -->
            <div v-if="showQuickPanel" class="quick-panel" role="menu" aria-label="快捷提问">
              <button v-for="hint in quickHints" :key="hint.text" class="quick-item" type="button" role="menuitem" @click="applyQuickHint(hint.text)">
                <component :is="hint.icon" :size="14" class="sicon" />
                <span>{{ hint.text }}</span>
              </button>
            </div>
            <button class="composer-chip round" type="button" title="添加附件" :disabled="uploading" @click="onChipAdd">
              <Plus :size="16" />
            </button>
            <button class="composer-chip" type="button" title="快速" :class="{ open: showQuickPanel }" :aria-expanded="showQuickPanel" @click="showQuickPanel = !showQuickPanel">
              <Zap :size="16" />
              <span class="chip-label">快速</span>
            </button>
            <button class="composer-chip" type="button" title="代码" @click="insertTemplate(promptTemplates.code)">
              <Code :size="16" />
              <span class="chip-label">代码</span>
            </button>
            <button class="composer-chip" type="button" title="总结" @click="insertTemplate(promptTemplates.summarize)">
              <AlignLeft :size="16" />
              <span class="chip-label">总结</span>
            </button>
            <button class="composer-chip" type="button" title="翻译" @click="insertTemplate(promptTemplates.translate)">
              <Languages :size="16" />
              <span class="chip-label">翻译</span>
            </button>
            <span class="spacer"></span>
            <!-- 不用 disabled：保持可点击，点击后由 handleSend 解释未就绪原因 -->
            <button class="send" type="button" :class="{ unavailable: !canSend }" :title="canSend ? '发送' : '运行时就绪后可发送'" @click="handleSend">
              <Send :size="16" />
            </button>
          </div>
          <input ref="fileInput" class="file-input-hidden" type="file" multiple tabindex="-1" aria-hidden="true" @change="onFilePicked">
        </div>

        <div class="composer-foot">
          <span class="kbd">Enter</span> 发送
          <span aria-hidden="true">·</span>
          <span class="kbd">Shift</span>+<span class="kbd">Enter</span> 换行
          <span aria-hidden="true">·</span>
          Seed基于 Taiji 原生状态与局部可塑性生成，请核对关键信息
        </div>
      </div>
    </div>
  </main>
</template>

<script setup>
defineOptions({ name: 'ChatView' })
import { ref, computed, watch, nextTick, onMounted, inject } from 'vue'
import { User, RotateCcw, Copy, Square, Send, Code, ThumbsUp, Brain, Bug, SlidersHorizontal, LineChart, GitBranch, ScrollText, Plus, Zap, AlignLeft, Languages, ChevronDown } from 'lucide-vue-next'
import TaijiLogo from './TaijiLogo.vue'
import RuntimeEvidenceStrip from './RuntimeEvidenceStrip.vue'
import { useChatStore } from '@/stores/chatStore.js'
import { useRuntimeStore } from '@/stores/runtimeStore.js'
import { useMarkdown } from '@/composables/useMarkdown.js'
import { API_BASE, authFetch } from '@/composables/apiClient.js'

const chatStore = useChatStore()
const runtimeStore = useRuntimeStore()
const { renderMarkdown } = useMarkdown()
const toast = inject('toast', () => {})

const messagesArea = ref(null)
const inputRef = ref(null)
const fileInput = ref(null)
const showExample = ref(false)  // 示例对话默认收起
const showQuickPanel = ref(false)  // 快捷提问面板（"快速"chip）
const uploading = ref(false)  // 附件上传中

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

// 历史消息（后端无 readable 标注）用同一启发式判定：替换字符/控制字符占比 >= 2%
function looksUnreadable(text) {
  if (!text) return false
  let bad = 0
  for (const ch of text) {
    const code = ch.charCodeAt(0)
    if (ch === '�' || (code < 32 && !'\n\r\t'.includes(ch))) bad++
  }
  return bad / text.length >= 0.02
}
function isRawOutput(msg) {
  return msg.unreadable === true || looksUnreadable(msg.content || '')
}

function scrollToBottom() { nextTick(() => { if (messagesArea.value) messagesArea.value.scrollTop = messagesArea.value.scrollHeight }) }
watch(() => chatStore.messages.length, scrollToBottom)
watch(() => chatStore.isReceiving, scrollToBottom)

const isRecording = ref(false)
const messagePageSize = 50
const messageLimit = ref(messagePageSize)
const displayedMessages = computed(() => {
  const msgs = chatStore.messages
  return msgs.length > messageLimit.value ? msgs.slice(-messageLimit.value) : msgs
})
const hasMoreMessages = computed(() => chatStore.messages.length > messageLimit.value)
function showMore() { messageLimit.value += messagePageSize }
const inputPlaceholder = computed(() => {
  if (isRecording.value) return '正在录音...'
  return '输入任务、问题或文件说明'
})

function handleSend() {
  if (!canSend.value) {
    // 明确告知为什么不能发送，而不是静默无响应
    if (chatStore.isLoading) {
      toast('正在生成回复，请稍候或点击「中断执行」', 'info')
    } else if (runtimeStore.health.state !== 'connected') {
      toast(`运行时未就绪（${runtimeStore.connectionStatus}），请等待连接恢复`, 'warning')
    } else if (!runtimeStore.health.modelLoaded) {
      toast('Taiji 原生运行时尚未就绪，暂时无法发送', 'warning')
    } else if (!chatStore.chatInput.trim()) {
      inputRef.value?.focus()
    }
    return
  }
  chatStore.sendMessage()
  scrollToBottom()
}
function onKeydown(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() } }

// R5: 语音/图片/视频/文件上传与拍照的 WIP 处理函数已移除（无对应模板入口，
// 属未接入的死代码；需要时从 git 历史恢复）。

async function copyMsg(content) { try { await navigator.clipboard.writeText(content); toast('已复制', 'success') } catch { toast('复制失败', 'error') } }
function likeMsg() { toast('已点赞', 'success') }

// ===== composer chip 真实行为 =====
// 提示词模板：代码问答 / 总结 / 翻译（纯前端可用能力）。
// 知识库与多模态能力尚未进入当前 native capability snapshot，入口不在聊天页伪造。
const promptTemplates = {
  code: '请帮我解释以下代码：\n```\n\n```',
  summarize: '请帮我总结以下内容：\n',
  translate: '请将以下内容翻译成中文：\n',
}

function insertTemplate(text) {
  chatStore.chatInput = chatStore.chatInput.trim() ? `${chatStore.chatInput}\n${text}` : text
  showQuickPanel.value = false
  nextTick(() => inputRef.value?.focus())
}

function applyQuickHint(text) {
  chatStore.chatInput = text
  showQuickPanel.value = false
  nextTick(() => inputRef.value?.focus())
}

// "添加"：打开文件选择，经 /api/chat/upload 解析后将内容注入输入框作为上下文。
function onChipAdd() { fileInput.value?.click() }

async function onFilePicked(e) {
  const files = Array.from(e.target.files || [])
  e.target.value = ''
  if (!files.length) return
  uploading.value = true
  try {
    for (const file of files) {
      const formData = new FormData()
      formData.append('file', file)
      const res = await authFetch(`${API_BASE}/api/chat/upload`, { method: 'POST', body: formData })
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
.chat-workbench {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
  background: var(--background);
  color: var(--foreground);
}

/* ===== 顶栏：三栏布局 + 居中标题 ===== */
/* 不画 border-bottom：外围边框由 .router-wrapper 独占（见 styles/shell.css） */
.topbar {
  height: 52px;
  flex: none;
  padding: 0 18px;
  display: flex;
  align-items: center;
  gap: 12px;
  background: transparent;
  position: relative;
  z-index: 5;
}
.topbar-spacer { flex: 1; }
.topbar-center {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  pointer-events: none;
}
.topbar-title { font-size: 0.92rem; font-weight: 600; color: var(--foreground); }

/* 生命体征芯片（顶栏右侧） */
.vitals { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }
.vital-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px 4px 8px;
  border-radius: 999px;
  background: var(--muted);
  border: 1px solid var(--border);
  font-size: 0.72rem;
  line-height: 1;
  color: var(--foreground);
  white-space: nowrap;
}
.vital-chip .vdot {
  width: 7px; height: 7px; border-radius: 50%; flex: none;
  box-shadow: 0 0 6px 0 currentColor;
}
.vital-chip .vdot.c1 { background: var(--chart-1); color: var(--chart-1); }
.vital-chip .vdot.c2 { background: var(--chart-2); color: var(--chart-2); }
.vital-chip .vdot.c3 { background: var(--chart-3); color: var(--chart-3); }
.vital-chip .vdot.c-danger { background: var(--destructive); color: var(--destructive); }
.vital-chip .vlabel { color: var(--muted-foreground); }
.vital-chip .vvalue { font-weight: 600; font-variant-numeric: tabular-nums; }

/* ===== 滚动区 ===== */
.scroll-area {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  scroll-behavior: smooth;
}
.scroll-area.chat-scroll {
  display: flex;
  flex-direction: column;
  position: relative;
}

/* 对话主舞台
   flex-shrink 必须为 0：滚动容器是 flex 列，若允许 stage 收缩到小于内容高度，
   内容会以 overflow:visible 溢出绘制，但父级 scrollHeight 仍按 stage 盒子计算，
   导致滚动条永不出现（内容被 sticky 输入栏遮住且无法滚动）。 */
.chat-stage {
  flex: 1 0 auto;
  max-width: 780px;
  width: 100%;
  margin: 0 auto;
  padding: 44px 28px 16px;
  display: flex;
  flex-direction: column;
  gap: 30px;
}

/* ===== 欢迎区 ===== */
.chat-welcome {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 18px;
  text-align: center;
}
.welcome-logo {
  width: 72px; height: 72px;
  position: relative;
  flex: none;
  filter: drop-shadow(0 6px 18px color-mix(in srgb, var(--foreground) 18%, transparent));
}
.welcome-logo img {
  width: 100%; height: 100%;
  object-fit: contain;
  display: block;
  border-radius: 50%;
}
.welcome-logo::before {
  content: "";
  position: absolute;
  inset: -8px;
  border-radius: 50%;
  border: 1px dashed color-mix(in srgb, var(--foreground) 22%, transparent);
}
@keyframes taiji-spin {
  to { transform: rotate(360deg); }
}
.chat-welcome h1 {
  margin: 0;
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.2;
  color: var(--foreground);
}
.welcome-sub {
  font-size: 0.86rem;
  color: var(--muted-foreground);
  display: inline-flex;
  align-items: center;
  gap: 7px;
}
.welcome-sub .ok-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--chart-2);
  flex: none;
}

/* 建议词云 */
.suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: center;
  max-width: 640px;
  margin: 0 auto;
}
.suggestion {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--card);
  color: var(--foreground);
  font-size: 0.88rem;
  line-height: 1;
  cursor: pointer;
  transition: border-color .16s ease, background .16s ease, color .16s ease, transform .12s ease;
}
.suggestion:hover {
  border-color: color-mix(in srgb, var(--primary) 48%, var(--border));
  background: color-mix(in srgb, var(--accent) 45%, var(--card));
  color: var(--primary);
}
.suggestion:active { transform: translateY(1px); }
.suggestion .sicon { width: 16px; height: 16px; flex: none; color: var(--primary); }

/* ===== 消息线程 ===== */
.chat-thread {
  display: flex;
  flex-direction: column;
  gap: 20px;
  padding-top: 4px;
}

/* 示例对话分割线 */
.thread-divider {
  display: flex;
  align-items: center;
  gap: 12px;
  color: var(--muted-foreground);
  font-size: 0.74rem;
}
.thread-divider::before,
.thread-divider::after {
  content: "";
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* 示例对话折叠入口（复用 thread-divider 分割线形态） */
.example-toggle {
  width: 100%;
  border: 0;
  background: transparent;
  padding: 0;
  cursor: pointer;
  font: inherit;
}
.example-toggle:hover .example-toggle-text { color: var(--foreground); }
.example-toggle .example-chevron {
  color: var(--muted-foreground);
  transition: transform .18s ease;
  flex: none;
}
.example-toggle .example-chevron.open { transform: rotate(180deg); }

/* 消息行 */
.msg {
  display: flex;
  gap: 12px;
  max-width: 100%;
  align-items: flex-start;
  content-visibility: auto;
  /* 屏外消息的高度估值：过小会让 scrollHeight 失真、滚动条跳动 */
  contain-intrinsic-size: auto 140px;
}
.msg-user { flex-direction: row-reverse; }

/* 头像 */
.av {
  width: 32px; height: 32px; border-radius: 50%;
  flex: none;
  display: grid;
  place-items: center;
  font-size: 0.8rem;
  font-weight: 600;
  margin-top: 2px;
}
.av-ai {
  background: transparent;
  object-fit: contain;
  padding: 0;
  border: 1px solid var(--border);
}
.av-ai.breathing { opacity: 0.82; }
.av-user {
  background: var(--accent);
  color: var(--accent-foreground);
  border: 0;
}
.av-user :deep(svg) { width: 16px; height: 16px; color: var(--accent-foreground); }

/* 消息体 */
.msg-body { min-width: 0; max-width: 78%; display: flex; flex-direction: column; gap: 5px; }
.msg-user .msg-body { align-items: flex-end; }
.msg-name {
  font-size: 0.74rem;
  color: var(--muted-foreground);
  padding: 0 4px;
}

/* 气泡 */
.bubble {
  padding: 12px 16px;
  border-radius: 18px;
  font-size: 0.92rem;
  line-height: 1.62;
  max-width: 100%;
}
.msg-user .bubble {
  background: var(--primary);
  color: var(--primary-foreground);
  border-bottom-right-radius: 6px;
}
.msg-ai .bubble {
  background: var(--card);
  border: 1px solid var(--border);
  color: var(--foreground);
  border-bottom-left-radius: 6px;
}
.bubble p { margin: 0; }
.bubble p + p { margin-top: 8px; }
.bubble .lead { font-weight: 600; }
.text-content { white-space: pre-wrap; word-break: break-word; }

/* 步骤列表 */
.msg-steps {
  margin: 8px 0 0;
  padding: 0;
  list-style: none;
  counter-reset: step;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.msg-steps li {
  position: relative;
  padding-left: 26px;
  counter-increment: step;
  font-size: 0.86rem;
  line-height: 1.55;
}
.msg-steps li::before {
  content: counter(step);
  position: absolute;
  left: 0; top: 1px;
  width: 18px; height: 18px;
  border-radius: 50%;
  background: color-mix(in srgb, var(--primary) 14%, transparent);
  color: var(--primary);
  font-size: 0.7rem;
  font-weight: 700;
  display: grid;
  place-items: center;
}
.msg-steps code, .bubble code {
  font-family: var(--font-mono);
  font-size: 0.82em;
  background: color-mix(in srgb, var(--primary) 12%, transparent);
  color: var(--primary);
  padding: 1px 6px;
  border-radius: 6px;
}

/* 消息操作 */
.msg-actions {
  display: flex;
  gap: 4px;
  margin-top: 4px;
}
.msg-action-btn {
  width: 28px; height: 28px; display: grid; place-items: center;
  border: 0; border-radius: 8px; color: var(--muted-foreground);
  background: transparent; cursor: pointer;
  transition: background .14s ease, color .14s ease;
}
.msg-action-btn:hover { background: var(--muted); color: var(--foreground); }

/* 思考动画 */
.thinking-row .bubble { width: fit-content; }
.thinking-animation { display: flex; gap: 5px; align-items: center; padding: 4px 0; }
.think-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--primary); animation: dotBounce 1.2s ease-in-out infinite; }
.think-dot:nth-child(2) { animation-delay: 0.15s; }
.think-dot:nth-child(3) { animation-delay: 0.3s; }

/* 运行状态通知 */
.runtime-notice {
  display: flex; align-items: flex-start; gap: 10px; padding: 14px 16px;
  border-radius: 16px; background: var(--card); border: 1px solid var(--border);
  box-shadow: var(--shadow-sm);
}
.runtime-notice-dot { width: 8px; height: 8px; margin-top: 5px; border-radius: 50%; background: var(--muted-foreground); flex-shrink: 0; }
.runtime-notice-dot.warning { background: var(--warning); }
.runtime-notice.loading .runtime-notice-dot,
.runtime-notice.connecting .runtime-notice-dot { background: var(--warning); }
.runtime-notice.error .runtime-notice-dot { background: var(--destructive); }
.runtime-notice.connected .runtime-notice-dot { background: var(--success); }
.runtime-notice strong { display: block; color: var(--foreground); font-size: 13px; font-weight: 650; }
.runtime-notice p { margin: 3px 0 0; color: var(--muted-foreground); font-size: 12px; line-height: 1.5; }
.provider-notice { display: inline-flex; align-items: center; gap: 6px; margin-top: 8px; color: var(--warning); font-size: 12px; }
.provider-notice .runtime-notice-dot { margin-top: 0; }

/* 加载更多 */
.load-more-row { text-align: center; padding: 4px 0; }
.load-more-btn {
  background: var(--muted); border: 1px solid var(--border);
  border-radius: 999px; color: var(--muted-foreground);
  font-size: 12px; padding: 6px 16px; cursor: pointer;
  transition: var(--transition-fast);
}
.load-more-btn:hover { background: var(--accent); color: var(--accent-foreground); }

/* ===== 输入区（sticky 吸底） ===== */
.composer-wrap {
  position: sticky;
  bottom: 0;
  flex: none;
  z-index: 2;
  max-width: 780px;
  width: 100%;
  margin: 0 auto;
  padding: 12px 28px 20px;
  background: linear-gradient(to top,
    var(--background) 68%,
    color-mix(in srgb, var(--background) 40%, transparent));
}
.composer-wrap .composer {
  box-shadow: 0 6px 24px color-mix(in srgb, var(--chart-4) 10%, transparent);
}

.stop-container { display: flex; justify-content: center; margin-bottom: 8px; }
.stop-btn {
  display: inline-flex; align-items: center; gap: 5px; height: 32px; padding: 0 16px;
  border: 1px solid var(--destructive); border-radius: 999px; color: var(--destructive);
  background: var(--danger-light); cursor: pointer; font-size: 12px; transition: var(--transition-fast);
}
.stop-btn:hover { background: color-mix(in srgb, var(--destructive) 18%, transparent); transform: scale(1.02); }

/* 胶囊形输入框 */
.composer {
  border: 1px solid var(--border);
  border-radius: 22px;
  background: var(--card);
  padding: 14px 16px 10px;
  transition: border-color .16s ease, box-shadow .16s ease;
}
.composer:focus-within {
  border-color: var(--primary);
  box-shadow: 0 0 0 3px var(--primary-light), 0 6px 24px color-mix(in srgb, var(--primary) 10%, transparent);
}
.composer textarea {
  width: 100%;
  border: 0; outline: none; background: transparent; color: var(--foreground);
  font-family: var(--font-sans); font-size: 14px; line-height: 1.6; resize: none;
  padding: 4px 2px 8px; min-height: 24px; max-height: 150px;
  display: block;
}
.composer textarea::placeholder { color: var(--muted-foreground); }

/* 工具行 */
.tools {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  padding-top: 4px;
  position: relative;
}

/* 快捷提问面板（"快速"chip 展开） */
.quick-panel {
  position: absolute;
  bottom: calc(100% + 8px);
  left: 0;
  z-index: 20;
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 300px;
  max-width: min(440px, 100%);
  padding: 6px;
  border-radius: 14px;
  border: 1px solid var(--border);
  background: var(--card);
  box-shadow: 0 8px 24px color-mix(in srgb, var(--foreground) 14%, transparent);
  animation: quick-panel-in .16s ease;
}
@keyframes quick-panel-in {
  from { opacity: 0; transform: translateY(4px); }
}
.quick-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border: 0;
  border-radius: 10px;
  background: transparent;
  text-align: left;
  color: var(--foreground);
  font-size: 0.82rem;
  line-height: 1.4;
  cursor: pointer;
  transition: background .14s ease, color .14s ease;
}
.quick-item:hover { background: var(--muted); color: var(--primary); }
.quick-item .sicon { flex: none; color: var(--primary); }

.file-input-hidden { display: none; }
.composer-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 7px 11px;
  border: 0; border-radius: 999px;
  background: transparent;
  color: color-mix(in srgb, var(--foreground) 78%, var(--muted-foreground));
  font-size: 13px; cursor: pointer;
  transition: background .14s ease, color .14s ease;
}
.composer-chip:hover { background: var(--muted); color: var(--foreground); }
.composer-chip.open {
  color: var(--primary);
  background: color-mix(in srgb, var(--primary) 12%, transparent);
}
.composer-chip.round {
  width: 32px; height: 32px;
  padding: 0;
  justify-content: center;
}
.composer-chip.active { color: var(--destructive); background: var(--danger-light); }
.composer-chip:disabled { opacity: 0.4; cursor: not-allowed; }
.composer-chip:disabled:hover { background: transparent; color: color-mix(in srgb, var(--foreground) 78%, var(--muted-foreground)); }
.chip-label { font-size: 13px; }
.composer-chip :deep(svg) { width: 16px; height: 16px; }

.spacer { flex: 1; }

.send {
  width: 36px; height: 36px; display: inline-flex; align-items: center; justify-content: center;
  border: 0; border-radius: 999px; background: var(--primary); color: var(--primary-foreground);
  cursor: pointer; transition: var(--transition-fast); flex: none;
}
.send:hover:not(:disabled) { background: var(--primary-hover); }
.send:disabled { opacity: 0.4; cursor: not-allowed; }
.send.unavailable { opacity: 0.45; cursor: not-allowed; }
.send.unavailable:hover { background: var(--primary); }

/* ===== 语言表层异常时的调试输出卡片 ===== */
.bubble.raw-output {
  background: var(--muted);
  border: 1px dashed color-mix(in srgb, var(--warning, #f59e0b) 45%, var(--border));
  max-width: 100%;
}
.raw-head {
  display: flex; align-items: center; gap: 7px; margin-bottom: 6px;
}
.raw-badge {
  font-size: 0.64rem; font-weight: 700; letter-spacing: 0.08em;
  padding: 2px 7px; border-radius: 6px;
  background: color-mix(in srgb, var(--warning, #f59e0b) 16%, transparent);
  color: var(--warning, #b45309);
  font-family: var(--font-mono);
}
.raw-title {
  font-size: 0.78rem; font-weight: 600; color: var(--foreground);
}
.raw-desc {
  margin: 0 0 8px; font-size: 0.76rem; line-height: 1.5; color: var(--muted-foreground);
}
.raw-pre {
  margin: 0; padding: 10px 12px; border-radius: 10px;
  background: color-mix(in srgb, var(--foreground) 6%, transparent);
  font-family: var(--font-mono); font-size: 0.74rem; line-height: 1.55;
  color: var(--muted-foreground);
  white-space: pre-wrap; word-break: break-all;
  max-height: 220px; overflow-y: auto;
}

/* 输入区底部提示 */
.composer-foot {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  margin-top: 9px;
  font-size: 0.72rem;
  color: var(--muted-foreground);
}
.composer-foot .kbd {
  display: inline-flex;
  align-items: center;
  padding: 1px 6px;
  border-radius: 6px;
  background: color-mix(in srgb, var(--muted-foreground) 14%, transparent);
  color: var(--muted-foreground);
  font-size: 0.7rem;
  font-weight: 600;
}

/* ===== Markdown 渲染 ===== */
.markdown-body { color: inherit; }

.markdown-body :deep(.code-block-wrapper) {
  margin: 10px 0;
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  background: var(--muted);
}
.markdown-body :deep(.code-header) {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  font-size: 0.74rem;
  color: var(--muted-foreground);
}
.markdown-body :deep(.code-lang) {
  font-family: var(--font-mono);
  font-weight: 600;
  color: var(--foreground);
}
.markdown-body :deep(.code-copy-btn) {
  margin-left: auto;
  border: 0;
  background: transparent;
  color: var(--muted-foreground);
  font-size: 0.72rem;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 7px;
  border-radius: 7px;
  transition: background .14s ease, color .14s ease;
}
.markdown-body :deep(.code-copy-btn:hover) { background: var(--card); color: var(--foreground); }
.markdown-body :deep(pre) {
  margin: 0;
  padding: 12px 14px;
  overflow-x: auto;
  font-family: var(--font-mono);
  font-size: 0.8rem;
  line-height: 1.6;
  background: transparent;
  border: 0;
}
.markdown-body :deep(code) { font-family: var(--font-mono); font-size: 0.85em; background: color-mix(in srgb, var(--primary) 12%, transparent); color: var(--primary); padding: 1px 6px; border-radius: 6px; }
.markdown-body :deep(pre code) { background: transparent; padding: 0; color: var(--foreground); font-size: inherit; }
.markdown-body :deep(p) { margin: 0 0 8px; }
.markdown-body :deep(p:last-child) { margin-bottom: 0; }
.markdown-body :deep(ul), .markdown-body :deep(ol) { padding-left: 20px; margin: 6px 0; }
.markdown-body :deep(li) { margin: 3px 0; }
.markdown-body :deep(blockquote) { border-left: 3px solid var(--primary-light); padding-left: 12px; color: var(--secondary-foreground); margin: 8px 0; }
.markdown-body :deep(img) { max-width: 100%; height: auto; border-radius: 6px; margin: 8px 0; display: block; border: 1px solid var(--border); }
.markdown-body :deep(a) { color: var(--primary); text-decoration: none; }
.markdown-body :deep(a:hover) { text-decoration: underline; }
.markdown-body :deep(table) { border-collapse: collapse; width: 100%; margin: 8px 0; }
.markdown-body :deep(th), .markdown-body :deep(td) { border: 1px solid var(--border); padding: 6px 10px; font-size: 13px; }
.markdown-body :deep(th) { background: var(--muted); font-weight: 600; }
.markdown-body :deep(hr) { border: none; border-top: 1px solid var(--border); margin: 12px 0; }

/* ===== 动画关键帧 ===== */
@keyframes taijiBreathe {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.75; transform: scale(0.97); }
}
@keyframes dotBounce {
  0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
  40% { transform: translateY(-4px); opacity: 1; }
}

@media (prefers-reduced-motion: reduce) {
  .think-dot { animation: none !important; }
}

/* ===== 响应式 ===== */
@media (max-width: 1080px) {
  .vitals { display: none; }
}
@media (max-width: 880px) {
  .topbar-center { display: none; }
  .chat-stage { padding: 32px 18px 12px; }
  .composer-wrap { padding: 10px 18px 16px; }
  .chat-welcome h1 { font-size: 1.6rem; }
  .msg-body { max-width: 88%; }
  .chip-label { display: none; }
}
</style>
