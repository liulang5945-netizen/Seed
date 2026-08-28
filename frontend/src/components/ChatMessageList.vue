<template>
  <template v-if="messages.length === 0">
    <section class="chat-welcome">
      <div class="welcome-logo" aria-hidden="true">
        <TaijiLogo :size="72" :thinking="runtimeState === 'connected'" />
      </div>
      <h1>有什么我能帮你的吗？</h1>
      <div class="welcome-sub">
        <span class="ok-dot"></span>
        {{ connectionStatus }} · {{ isTaiji ? 'Taiji Native 语言通路' : 'Seed 运行时' }}
      </div>
      <div v-if="languageProviderNotice" class="provider-notice">
        <span class="runtime-notice-dot warning"></span>
        <span>{{ languageProviderNotice.message }}</span>
      </div>
    </section>

    <div class="suggestions" role="list">
      <button
        v-for="hint in quickHints"
        :key="hint.text"
        class="suggestion"
        type="button"
        @click="emit('select-hint', hint.text)"
      >
        <component :is="hint.icon" :size="16" class="sicon" />
        <span>{{ hint.text }}</span>
      </button>
    </div>

    <div class="chat-thread chat-thread-example">
      <button class="thread-divider example-toggle" type="button" :aria-expanded="showExample" @click="emit('toggle-example')">
        <span class="example-toggle-text">{{ showExample ? '收起示例对话' : '查看示例对话' }}</span>
        <ChevronDown class="example-chevron" :class="{ open: showExample }" :size="14" />
      </button>

      <template v-if="showExample">
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

  <template v-else>
    <div class="chat-thread">
      <div v-if="runtimeNotice" class="runtime-notice" :class="connectionClass">
        <span class="runtime-notice-dot"></span>
        <div>
          <strong>{{ runtimeNotice.title }}</strong>
          <p>{{ runtimeNotice.message }}</p>
        </div>
      </div>
      <div v-if="languageProviderNotice" class="runtime-notice warning">
        <span class="runtime-notice-dot warning"></span>
        <div>
          <strong>{{ languageProviderNotice.title }}</strong>
          <p>{{ languageProviderNotice.message }}</p>
        </div>
      </div>

      <div v-if="hasMoreMessages" class="load-more-row">
        <button class="load-more-btn" @click="emit('show-more')">
          显示更多消息（{{ messageLimit }}/{{ messages.length }}）
        </button>
      </div>

      <article
        v-for="msg in displayedMessages"
        :key="msg.id"
        v-memo="[msg.id, msg.content, msg.role, msg.unreadable]"
        :class="['msg', msg.role === 'user' ? 'msg-user' : 'msg-ai']"
      >
        <TaijiLogo v-if="msg.role === 'assistant'" class="av av-ai" :size="32" :thinking="false" aria-label="Seed" />
        <span v-else class="av av-user" aria-label="用户">
          <User :size="16" />
        </span>
        <div class="msg-body">
          <span class="msg-name">{{ msg.role === 'user' ? '你' : 'Seed' }}</span>
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
            <button class="msg-action-btn" title="复制" @click="emit('copy', msg.content)"><Copy :size="14" /></button>
            <button class="msg-action-btn" title="赞" @click="emit('like', msg.id)"><ThumbsUp :size="14" /></button>
            <button class="msg-action-btn" title="重新生成" @click="emit('regenerate', msg.id)"><RotateCcw :size="14" /></button>
          </div>
        </div>
      </article>

      <article v-if="isLoading" class="msg msg-ai thinking-row">
        <TaijiLogo class="av av-ai breathing" :size="32" :thinking="true" aria-label="Seed" />
        <div class="msg-body">
          <span class="msg-name">{{ isReceiving ? 'Seed · 正在回应' : 'Seed · 正在启动' }}</span>
          <div v-if="!isReceiving" class="bubble loading-bubble">
            <span class="thinking-animation"><span class="think-dot"></span><span class="think-dot"></span><span class="think-dot"></span></span>
          </div>
        </div>
      </article>
    </div>
  </template>
</template>

<script setup>
import { ChevronDown, Copy, RotateCcw, ThumbsUp, User } from 'lucide-vue-next'
import TaijiLogo from './TaijiLogo.vue'
import { useMarkdown } from '../composables/useMarkdown.js'

defineOptions({ name: 'ChatMessageList' })

defineProps({
  messages: { type: Array, default: () => [] },
  displayedMessages: { type: Array, default: () => [] },
  quickHints: { type: Array, default: () => [] },
  runtimeNotice: { type: Object, default: null },
  languageProviderNotice: { type: Object, default: null },
  connectionClass: { type: String, default: '' },
  connectionStatus: { type: String, default: '' },
  isTaiji: { type: Boolean, default: false },
  runtimeState: { type: String, default: '' },
  showExample: { type: Boolean, default: false },
  hasMoreMessages: { type: Boolean, default: false },
  messageLimit: { type: Number, default: 0 },
  isLoading: { type: Boolean, default: false },
  isReceiving: { type: Boolean, default: false },
})

const emit = defineEmits(['select-hint', 'toggle-example', 'show-more', 'copy', 'like', 'regenerate'])
const { renderMarkdown } = useMarkdown()

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

</script>

<style scoped>
.chat-welcome { display: flex; flex-direction: column; align-items: center; gap: 18px; text-align: center; }
.welcome-logo { width: 72px; height: 72px; position: relative; flex: none; filter: drop-shadow(0 6px 18px color-mix(in srgb, var(--foreground) 18%, transparent)); }
.welcome-logo img { width: 100%; height: 100%; object-fit: contain; display: block; border-radius: 50%; }
.welcome-logo::before { content: ""; position: absolute; inset: -8px; border-radius: 50%; border: 1px dashed color-mix(in srgb, var(--foreground) 22%, transparent); }
.chat-welcome h1 { margin: 0; font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.2; color: var(--foreground); }
.welcome-sub { font-size: 0.86rem; color: var(--muted-foreground); display: inline-flex; align-items: center; gap: 7px; }
.welcome-sub .ok-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--chart-2); flex: none; }
.suggestions { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; max-width: 640px; margin: 0 auto; }
.suggestion { display: inline-flex; align-items: center; gap: 8px; padding: 10px 16px; border-radius: 999px; border: 1px solid var(--border); background: var(--card); color: var(--foreground); font-size: 0.88rem; line-height: 1; cursor: pointer; transition: border-color .16s ease, background .16s ease, color .16s ease, transform .12s ease; }
.suggestion:hover { border-color: color-mix(in srgb, var(--primary) 48%, var(--border)); background: color-mix(in srgb, var(--accent) 45%, var(--card)); color: var(--primary); }
.suggestion:active { transform: translateY(1px); }
.suggestion .sicon { width: 16px; height: 16px; flex: none; color: var(--primary); }
.chat-thread { display: flex; flex-direction: column; gap: 20px; padding-top: 4px; }
.thread-divider { display: flex; align-items: center; gap: 12px; color: var(--muted-foreground); font-size: 0.74rem; }
.thread-divider::before, .thread-divider::after { content: ""; flex: 1; height: 1px; background: var(--border); }
.example-toggle { width: 100%; border: 0; background: transparent; padding: 0; cursor: pointer; font: inherit; }
.example-toggle:hover .example-toggle-text { color: var(--foreground); }
.example-toggle .example-chevron { color: var(--muted-foreground); transition: transform .18s ease; flex: none; }
.example-toggle .example-chevron.open { transform: rotate(180deg); }
.msg { display: flex; gap: 12px; max-width: 100%; align-items: flex-start; content-visibility: auto; contain-intrinsic-size: auto 140px; }
.msg-user { flex-direction: row-reverse; }
.av { width: 32px; height: 32px; border-radius: 50%; flex: none; display: grid; place-items: center; font-size: 0.8rem; font-weight: 600; margin-top: 2px; }
.av-ai { background: transparent; object-fit: contain; padding: 0; border: 1px solid var(--border); }
.av-ai.breathing { opacity: 0.82; }
.av-user { background: var(--accent); color: var(--accent-foreground); border: 0; }
.av-user :deep(svg) { width: 16px; height: 16px; color: var(--accent-foreground); }
.msg-body { min-width: 0; max-width: 78%; display: flex; flex-direction: column; gap: 5px; }
.msg-user .msg-body { align-items: flex-end; }
.msg-name { font-size: 0.74rem; color: var(--muted-foreground); padding: 0 4px; }
.bubble { padding: 12px 16px; border-radius: 18px; font-size: 0.92rem; line-height: 1.62; max-width: 100%; }
.msg-user .bubble { background: var(--primary); color: var(--primary-foreground); border-bottom-right-radius: 6px; }
.msg-ai .bubble { background: var(--card); border: 1px solid var(--border); color: var(--foreground); border-bottom-left-radius: 6px; }
.bubble p { margin: 0; }
.bubble p + p { margin-top: 8px; }
.bubble .lead { font-weight: 600; }
.text-content { white-space: pre-wrap; word-break: break-word; }
.msg-steps { margin: 8px 0 0; padding: 0; list-style: none; counter-reset: step; display: flex; flex-direction: column; gap: 6px; }
.msg-steps li { position: relative; padding-left: 26px; counter-increment: step; font-size: 0.86rem; line-height: 1.55; }
.msg-steps li::before { content: counter(step); position: absolute; left: 0; top: 1px; width: 18px; height: 18px; border-radius: 50%; background: color-mix(in srgb, var(--primary) 14%, transparent); color: var(--primary); font-size: 0.7rem; font-weight: 700; display: grid; place-items: center; }
.msg-steps code, .bubble code { font-family: var(--font-mono); font-size: 0.82em; background: color-mix(in srgb, var(--primary) 12%, transparent); color: var(--primary); padding: 1px 6px; border-radius: 6px; }
.msg-actions { display: flex; gap: 4px; margin-top: 4px; }
.msg-action-btn { width: 28px; height: 28px; display: grid; place-items: center; border: 0; border-radius: 8px; color: var(--muted-foreground); background: transparent; cursor: pointer; transition: background .14s ease, color .14s ease; }
.msg-action-btn:hover { background: var(--muted); color: var(--foreground); }
.thinking-row .bubble { width: fit-content; }
.thinking-animation { display: flex; gap: 5px; align-items: center; padding: 4px 0; }
.think-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--primary); animation: dotBounce 1.2s ease-in-out infinite; }
.think-dot:nth-child(2) { animation-delay: 0.15s; }
.think-dot:nth-child(3) { animation-delay: 0.3s; }
.runtime-notice { display: flex; align-items: flex-start; gap: 10px; padding: 14px 16px; border-radius: 16px; background: var(--card); border: 1px solid var(--border); box-shadow: var(--shadow-sm); }
.runtime-notice-dot { width: 8px; height: 8px; margin-top: 5px; border-radius: 50%; background: var(--muted-foreground); flex-shrink: 0; }
.runtime-notice-dot.warning { background: var(--warning); }
.runtime-notice.loading .runtime-notice-dot, .runtime-notice.connecting .runtime-notice-dot { background: var(--warning); }
.runtime-notice.error .runtime-notice-dot { background: var(--destructive); }
.runtime-notice.connected .runtime-notice-dot { background: var(--success); }
.runtime-notice strong { display: block; color: var(--foreground); font-size: 13px; font-weight: 650; }
.runtime-notice p { margin: 3px 0 0; color: var(--muted-foreground); font-size: 12px; line-height: 1.5; }
.provider-notice { display: inline-flex; align-items: center; gap: 6px; margin-top: 8px; color: var(--warning); font-size: 12px; }
.provider-notice .runtime-notice-dot { margin-top: 0; }
.load-more-row { text-align: center; padding: 4px 0; }
.load-more-btn { background: var(--muted); border: 1px solid var(--border); border-radius: 999px; color: var(--muted-foreground); font-size: 12px; padding: 6px 16px; cursor: pointer; transition: var(--transition-fast); }
.load-more-btn:hover { background: var(--accent); color: var(--accent-foreground); }
.bubble.raw-output { background: var(--muted); border: 1px dashed color-mix(in srgb, var(--warning, #f59e0b) 45%, var(--border)); max-width: 100%; }
.raw-head { display: flex; align-items: center; gap: 7px; margin-bottom: 6px; }
.raw-badge { font-size: 0.64rem; font-weight: 700; letter-spacing: 0.08em; padding: 2px 7px; border-radius: 6px; background: color-mix(in srgb, var(--warning, #f59e0b) 16%, transparent); color: var(--warning, #b45309); font-family: var(--font-mono); }
.raw-title { font-size: 0.78rem; font-weight: 600; color: var(--foreground); }
.raw-desc { margin: 0 0 8px; font-size: 0.76rem; line-height: 1.5; color: var(--muted-foreground); }
.raw-pre { margin: 0; padding: 10px 12px; border-radius: 10px; background: color-mix(in srgb, var(--foreground) 6%, transparent); font-family: var(--font-mono); font-size: 0.74rem; line-height: 1.55; color: var(--muted-foreground); white-space: pre-wrap; word-break: break-all; max-height: 220px; overflow-y: auto; }
.markdown-body { color: inherit; }
.markdown-body :deep(.code-block-wrapper) { margin: 10px 0; border: 1px solid var(--border); border-radius: 12px; overflow: hidden; background: var(--muted); }
.markdown-body :deep(.code-header) { display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-bottom: 1px solid var(--border); font-size: 0.74rem; color: var(--muted-foreground); }
.markdown-body :deep(.code-lang) { font-family: var(--font-mono); font-weight: 600; color: var(--foreground); }
.markdown-body :deep(.code-copy-btn) { margin-left: auto; border: 0; background: transparent; color: var(--muted-foreground); font-size: 0.72rem; cursor: pointer; display: inline-flex; align-items: center; gap: 4px; padding: 3px 7px; border-radius: 7px; transition: background .14s ease, color .14s ease; }
.markdown-body :deep(.code-copy-btn:hover) { background: var(--card); color: var(--foreground); }
.markdown-body :deep(pre) { margin: 0; padding: 12px 14px; overflow-x: auto; font-family: var(--font-mono); font-size: 0.8rem; line-height: 1.6; background: transparent; border: 0; }
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
@keyframes dotBounce { 0%, 80%, 100% { transform: translateY(0); opacity: 0.4; } 40% { transform: translateY(-4px); opacity: 1; } }
@media (prefers-reduced-motion: reduce) { .think-dot { animation: none !important; } }
@media (max-width: 880px) { .chat-welcome h1 { font-size: 1.6rem; } .msg-body { max-width: 88%; } }
</style>
