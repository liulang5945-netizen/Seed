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

    <!-- 对话滚动区 -->
    <div class="scroll-area chat-scroll" ref="messagesArea">
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
              {{ runtimeStore.connectionStatus }} · {{ runtimeStore.health.isTaiji ? 'Taiji Native 原始字节通路' : 'Seed 运行时' }}
            </div>
          </section>

          <!-- 建议词云 -->
          <div class="suggestions" role="list">
            <button class="suggestion" v-for="hint in quickHints" :key="hint.text" @click="chatStore.chatInput = hint.text" type="button">
              <component :is="hint.icon" :size="16" class="sicon" />
              <span>{{ hint.text }}</span>
            </button>
          </div>

          <!-- 示例对话分割线 -->
          <div class="chat-thread">
            <div class="thread-divider">示例对话</div>

            <!-- 示例用户提问 1 -->
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

            <!-- 示例 AI 回复 1 -->
            <div class="msg msg-ai">
              <TaijiLogo class="av av-ai" :size="32" :thinking="false" aria-label="Seed" />
              <div class="msg-body">
                <span class="msg-name">Seed</span>
                <div class="bubble">
                  <p><span class="lead">Taiji 的一次状态推进</span>从原始字节开始，不经过 tokenizer 或学习式 embedding。输入先进入 <code>ByteSensor</code>，再经过预测织体与 <code>EpisodicField</code>，最后由 <code>ByteMotor</code> 产生字节或动作。</p>
                  <ol class="msg-steps">
                    <li><strong>感觉</strong>：固定的 257 个感受器接收当前 byte 和 episode 边界。</li>
                    <li><strong>预测与记忆</strong>：递归预测织体推进状态，情景场保存 episode 关联。</li>
                    <li><strong>动作</strong>：稀疏 motor group 选择下一字节或动作，结果回到下一次感觉。</li>
                  </ol>
                  <p>这里的智能来自持续状态、预测误差和环境反馈的闭环，而不是一次性计算 token logits。</p>
                </div>
                <div class="msg-actions">
                  <button type="button" class="msg-action-btn" title="复制"><Copy :size="14" /></button>
                  <button type="button" class="msg-action-btn" title="赞"><ThumbsUp :size="14" /></button>
                  <button type="button" class="msg-action-btn" title="重新生成"><RotateCcw :size="14" /></button>
                </div>
              </div>
            </div>

            <!-- 示例用户提问 2 -->
            <div class="msg msg-user">
              <span class="av av-user" aria-label="用户">
                <User :size="16" />
              </span>
              <div class="msg-body">
                <span class="msg-name">你</span>
                <div class="bubble">
                  <p>Taiji 的局部可塑性没有产生有效学习，应该怎么排查？</p>
                </div>
              </div>
            </div>

            <!-- 示例 AI 回复 2 -->
            <div class="msg msg-ai">
              <TaijiLogo class="av av-ai" :size="32" :thinking="false" aria-label="Seed" />
              <div class="msg-body">
                <span class="msg-name">Seed</span>
                <div class="bubble">
                  <p>Taiji 不依赖 Transformer 的 BPTT 或全局梯度。先按一次真实状态推进检查局部误差与突触更新：</p>
                  <ol class="msg-steps">
                    <li>确认输入仍是原始 byte 流，并在 episode 边界处正确 reset。</li>
                    <li>检查预测误差、递归误差和 action→outcome 误差是否有非零信号。</li>
                    <li>确认局部 plasticity 只更新已有突触，且 checkpoint 保存了持续状态。</li>
                  </ol>
                  <p>可以先记录一个短 episode 的 observe→learn→generate 链路：</p>
                  <div class="msg-code">
                    <div class="code-head">
                      <Code :size="14" />
                      <span class="lang">taiji-native.json</span>
                      <button class="copy" type="button"><Copy :size="13" />复制</button>
                    </div>
                    <pre><span class="k">input</span>: <span class="s">raw_bytes</span>
  <span class="k">sensor</span>: <span class="s">ByteSensor(257)</span>
  <span class="k">learning</span>: <span class="s">local_plasticity</span>
  <span class="k">output</span>: <span class="s">ByteMotor</span></pre>
                  </div>
                </div>
                <div class="msg-actions">
                  <button type="button" class="msg-action-btn" title="复制"><Copy :size="14" /></button>
                  <button type="button" class="msg-action-btn" title="赞"><ThumbsUp :size="14" /></button>
                  <button type="button" class="msg-action-btn" title="重新生成"><RotateCcw :size="14" /></button>
                </div>
              </div>
            </div>
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

            <div v-if="hasMoreMessages" class="load-more-row">
              <button class="load-more-btn" @click="showMore">
                显示更多消息（{{ messageLimit }}/{{ chatStore.messages.length }}）
              </button>
            </div>

            <article v-for="msg in displayedMessages" :key="msg.id"
              :class="['msg', msg.role === 'user' ? 'msg-user' : 'msg-ai']"
              v-memo="[msg.id, msg.content, msg.role]">
              <TaijiLogo v-if="msg.role === 'assistant'" class="av av-ai" :size="32" :thinking="false" aria-label="Seed" />
              <span v-else class="av av-user" aria-label="用户">
                <User :size="16" />
              </span>
              <div class="msg-body">
                <span class="msg-name">{{ msg.role === 'user' ? '你' : 'Seed' }}</span>
                <div class="bubble">
                  <div v-if="msg.role === 'user'" class="text-content">{{ msg.content }}</div>
                  <div v-else class="markdown-body" v-html="renderMarkdown(msg.content)" />
                </div>
                <div v-if="msg.role === 'assistant' && msg.content" class="msg-actions">
                  <button class="msg-action-btn" @click="copyMsg(msg.content)" title="复制"><Copy :size="14" /></button>
                  <button class="msg-action-btn" @click="likeMsg(msg.id)" title="赞"><ThumbsUp :size="14" /></button>
                  <button class="msg-action-btn" @click="chatStore.regenerateMessage(msg.id)" title="重新生成"><RotateCcw :size="14" /></button>
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
        <div class="stop-container" v-if="chatStore.isReceiving">
          <button class="stop-btn" @click="chatStore.stopGeneration()">
            <Square :size="13" fill="currentColor" /> 中断执行
          </button>
        </div>

        <div class="composer">
          <textarea ref="inputRef" v-model="chatStore.chatInput"
            :placeholder="inputPlaceholder"
            rows="1" @keydown="onKeydown" />
          <div class="tools">
            <button class="composer-chip round" type="button" title="添加" @click="onChipAdd">
              <Plus :size="16" />
            </button>
            <button class="composer-chip" type="button" title="快速">
              <Zap :size="16" />
              <span class="chip-label">快速</span>
            </button>
            <button class="composer-chip" type="button" title="知识库">
              <BookOpen :size="16" />
              <span class="chip-label">知识库</span>
            </button>
            <button class="composer-chip" type="button" title="图像生成">
              <ImageIcon :size="16" />
              <span class="chip-label">图像生成</span>
            </button>
            <button class="composer-chip" type="button" title="代码">
              <Code :size="16" />
              <span class="chip-label">代码</span>
            </button>
            <button class="composer-chip" type="button" title="更多" @click="onChipMore">
              <MoreHorizontal :size="16" />
              <span class="chip-label">更多</span>
            </button>
            <span class="spacer"></span>
            <button class="send" type="button" :class="{ unavailable: !canSend }" :disabled="!canSend" @click="handleSend" title="发送">
              <Send :size="16" />
            </button>
          </div>
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
import { Activity, User, Bot, RotateCcw, Copy, Square, Send, Lightbulb, Code, BookOpen, Mic, Image as ImageIcon, Video, FileText, Camera, ThumbsUp, Brain, Bug, SlidersHorizontal, LineChart, GitBranch, ScrollText, Plus, Zap, MoreHorizontal } from 'lucide-vue-next'
import TaijiLogo from './TaijiLogo.vue'
import { useChatStore } from '@/stores/chatStore.js'
import { useAppStore } from '@/stores/appStore.js'
import { useRuntimeStore } from '@/stores/runtimeStore.js'
import { useMarkdown } from '@/composables/useMarkdown.js'
import { authFetch } from '@/composables/apiClient.js'

const chatStore = useChatStore()
const appStore = useAppStore()
const runtimeStore = useRuntimeStore()
const { renderMarkdown } = useMarkdown()
const toast = inject('toast', () => {})
const t = (key) => appStore.t(key)

const messagesArea = ref(null)
const inputRef = ref(null)
const engineModel = ref('agent')  // 统一使用 ReAct 引擎

const vitalChips = computed(() => {
  const isTaiji = runtimeStore.health.isTaiji
  return [
    { dot: 'c1', label: '运行时', value: isTaiji ? 'Taiji Native' : 'Cortex' },
    { dot: 'c2', label: '输入', value: isTaiji ? 'raw bytes' : '对照流' },
    { dot: 'c3', label: '学习', value: isTaiji ? '局部可塑性' : '冻结对照' },
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
  if (!canSend.value) return
  chatStore.sendMessage(engineModel.value)
  scrollToBottom()
}
function onKeydown(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() } }

// 语音输入
async function toggleVoice() {
  if (isRecording.value) {
    isRecording.value = false
    // TODO: 停止录音并发送音频
    toast('语音功能开发中', 'info')
  } else {
    isRecording.value = true
    // TODO: 开始录音
    toast('语音功能开发中', 'info')
  }
}

// 图片上传
async function onImageSelect(e) {
  const file = e.target.files[0]
  if (!file) return
  const formData = new FormData()
  formData.append('file', file)
  try {
    const resp = await authFetch('/api/taiji/upload', { method: 'POST', body: formData })
    if (resp.ok) {
      const data = await resp.json()
      chatStore.chatInput += `[图片: ${data.filename}] `
      toast('图片已上传', 'success')
    } else {
      toast('图片上传失败', 'error')
    }
  } catch (e) {
    console.warn('[ChatView] image upload failed:', e.message)
    toast('图片上传失败', 'error')
  }
  e.target.value = ''
}

// 视频上传
async function onVideoSelect(e) {
  const file = e.target.files[0]
  if (!file) return
  const formData = new FormData()
  formData.append('file', file)
  try {
    const resp = await authFetch('/api/taiji/upload', { method: 'POST', body: formData })
    if (resp.ok) {
      const data = await resp.json()
      chatStore.chatInput += `[视频: ${data.filename}] `
      toast('视频已上传', 'success')
    } else {
      toast('视频上传失败', 'error')
    }
  } catch (e) {
    console.warn('[ChatView] video upload failed:', e.message)
    toast('视频上传失败', 'error')
  }
  e.target.value = ''
}

// 文件上传
async function onFileSelect(e) {
  const files = e.target.files
  if (!files.length) return
  for (const file of files) {
    const formData = new FormData()
    formData.append('file', file)
    try {
      const resp = await authFetch('/api/taiji/upload', { method: 'POST', body: formData })
      if (resp.ok) {
        const data = await resp.json()
        chatStore.chatInput += `[文件: ${data.filename}] `
      }
    } catch (e) { console.warn('[ChatView] file upload failed:', e.message) }
  }
  toast(`已上传 ${files.length} 个文件`, 'success')
  e.target.value = ''
}

// 拍照/录像
async function toggleCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    // TODO: 实现拍照/录像逻辑
    toast('摄像头功能开发中', 'info')
    stream.getTracks().forEach(t => t.stop())
  } catch (e) {
    console.warn('[ChatView] camera access denied:', e.message)
    toast('无法访问摄像头', 'error')
  }
}

async function copyMsg(content) { try { await navigator.clipboard.writeText(content); toast('已复制', 'success') } catch { toast('复制失败', 'error') } }
function likeMsg() { toast('已点赞', 'success') }

// composer chip 占位功能
function onChipAdd() { toast('添加功能开发中', 'info') }
function onChipMore() { toast('更多功能开发中', 'info') }

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
.topbar {
  height: 52px;
  flex: none;
  padding: 0 18px;
  display: flex;
  align-items: center;
  gap: 12px;
  border-bottom: 1px solid var(--border);
  background: var(--background);
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

/* 对话主舞台 */
.chat-stage {
  flex: 1;
  min-height: 0;
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

/* 消息行 */
.msg {
  display: flex;
  gap: 12px;
  max-width: 100%;
  align-items: flex-start;
  content-visibility: auto;
  contain-intrinsic-size: auto 80px;
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

/* 代码块 */
.msg-code {
  margin: 10px 0 0;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid var(--border);
  background: var(--muted);
}
.msg-code .code-head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  font-size: 0.74rem;
  color: var(--muted-foreground);
}
.msg-code .code-head .lang {
  font-family: var(--font-mono);
  font-weight: 600;
  color: var(--foreground);
}
.msg-code .code-head .copy {
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
.msg-code .code-head .copy:hover { background: var(--background); color: var(--foreground); }
.msg-code pre {
  margin: 0;
  padding: 12px 14px;
  overflow-x: auto;
  font-family: var(--font-mono);
  font-size: 0.8rem;
  line-height: 1.6;
  color: var(--foreground);
}
.msg-code .k { color: var(--chart-2); }
.msg-code .n { color: var(--chart-4); }
.msg-code .c { color: var(--muted-foreground); }

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
.runtime-notice.loading .runtime-notice-dot,
.runtime-notice.connecting .runtime-notice-dot { background: var(--warning); }
.runtime-notice.error .runtime-notice-dot { background: var(--destructive); }
.runtime-notice.connected .runtime-notice-dot { background: var(--success); }
.runtime-notice strong { display: block; color: var(--foreground); font-size: 13px; font-weight: 650; }
.runtime-notice p { margin: 3px 0 0; color: var(--muted-foreground); font-size: 12px; line-height: 1.5; }

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
}
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
.send.unavailable { opacity: 0.4; }

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
