<template>
  <div class="composer-wrap">
    <div v-if="isReceiving" class="stop-container">
      <button class="stop-btn" @click="emit('stop')">
        <Square :size="13" fill="currentColor" /> 中断执行
      </button>
    </div>

    <div class="composer">
      <textarea
        ref="inputRef"
        :value="modelValue"
        :placeholder="inputPlaceholder"
        rows="1"
        @input="emit('update:modelValue', $event.target.value)"
        @keydown="onKeydown"
      />
      <div class="tools">
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
        <button class="composer-chip" type="button" title="工作台" :class="{ active: workbenchMode }" :aria-pressed="workbenchMode" @click="emit('toggle-workbench')">
          <PanelsTopLeft :size="16" />
          <span class="chip-label">工作台</span>
        </button>
        <span class="spacer"></span>
        <button class="send" type="button" :class="{ unavailable: !canSend }" :title="canSend ? '发送' : '运行时就绪后可发送'" @click="emit('send')">
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
</template>

<script setup>
import { ref } from 'vue'
import { AlignLeft, Code, Languages, PanelsTopLeft, Plus, Send, Square, Zap } from 'lucide-vue-next'

defineOptions({ name: 'ChatComposer' })

defineProps({
  modelValue: { type: String, default: '' },
  inputPlaceholder: { type: String, default: '输入任务、问题或文件说明' },
  canSend: { type: Boolean, default: false },
  isReceiving: { type: Boolean, default: false },
  uploading: { type: Boolean, default: false },
  quickHints: { type: Array, default: () => [] },
  promptTemplates: { type: Object, default: () => ({}) },
  workbenchMode: { type: Boolean, default: false },
})

const emit = defineEmits(['update:modelValue', 'send', 'stop', 'files-picked', 'insert-template', 'apply-quick-hint', 'toggle-workbench'])
const inputRef = ref(null)
const fileInput = ref(null)
const showQuickPanel = ref(false)

function onKeydown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    emit('send')
  }
}

function insertTemplate(text) {
  showQuickPanel.value = false
  emit('insert-template', text)
  inputRef.value?.focus()
}

function applyQuickHint(text) {
  showQuickPanel.value = false
  emit('apply-quick-hint', text)
  inputRef.value?.focus()
}

function onChipAdd() {
  fileInput.value?.click()
}

function onFilePicked(event) {
  const files = Array.from(event.target.files || [])
  event.target.value = ''
  if (files.length) emit('files-picked', files)
}

function focus() {
  inputRef.value?.focus()
}

defineExpose({ focus })
</script>

<style scoped>
.composer-wrap { position: sticky; bottom: 0; flex: none; z-index: 2; max-width: 780px; width: 100%; margin: 0 auto; padding: 12px 28px 20px; background: linear-gradient(to top, var(--background) 68%, color-mix(in srgb, var(--background) 40%, transparent)); }
.composer-wrap .composer { box-shadow: 0 6px 24px color-mix(in srgb, var(--chart-4) 10%, transparent); }
.stop-container { display: flex; justify-content: center; margin-bottom: 8px; }
.stop-btn { display: inline-flex; align-items: center; gap: 5px; height: 32px; padding: 0 16px; border: 1px solid var(--destructive); border-radius: 999px; color: var(--destructive); background: var(--danger-light); cursor: pointer; font-size: 12px; transition: var(--transition-fast); }
.stop-btn:hover { background: color-mix(in srgb, var(--destructive) 18%, transparent); transform: scale(1.02); }
.composer { border: 1px solid var(--border); border-radius: 22px; background: var(--card); padding: 14px 16px 10px; transition: border-color .16s ease, box-shadow .16s ease; }
.composer:focus-within { border-color: var(--primary); box-shadow: 0 0 0 3px var(--primary-light), 0 6px 24px color-mix(in srgb, var(--primary) 10%, transparent); }
.composer textarea { width: 100%; border: 0; outline: none; background: transparent; color: var(--foreground); font-family: var(--font-sans); font-size: 14px; line-height: 1.6; resize: none; padding: 4px 2px 8px; min-height: 24px; max-height: 150px; display: block; }
.composer textarea::placeholder { color: var(--muted-foreground); }
.tools { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; padding-top: 4px; position: relative; }
.quick-panel { position: absolute; bottom: calc(100% + 8px); left: 0; z-index: 20; display: flex; flex-direction: column; gap: 2px; min-width: 300px; max-width: min(440px, 100%); padding: 6px; border-radius: 14px; border: 1px solid var(--border); background: var(--card); box-shadow: 0 8px 24px color-mix(in srgb, var(--foreground) 14%, transparent); animation: quick-panel-in .16s ease; }
@keyframes quick-panel-in { from { opacity: 0; transform: translateY(4px); } }
.quick-item { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border: 0; border-radius: 10px; background: transparent; text-align: left; color: var(--foreground); font-size: 0.82rem; line-height: 1.4; cursor: pointer; transition: background .14s ease, color .14s ease; }
.quick-item:hover { background: var(--muted); color: var(--primary); }
.quick-item .sicon { flex: none; color: var(--primary); }
.file-input-hidden { display: none; }
.composer-chip { display: inline-flex; align-items: center; gap: 6px; padding: 7px 11px; border: 0; border-radius: 999px; background: transparent; color: color-mix(in srgb, var(--foreground) 78%, var(--muted-foreground)); font-size: 13px; cursor: pointer; transition: background .14s ease, color .14s ease; }
.composer-chip:hover { background: var(--muted); color: var(--foreground); }
.composer-chip.open { color: var(--primary); background: color-mix(in srgb, var(--primary) 12%, transparent); }
.composer-chip.round { width: 32px; height: 32px; padding: 0; justify-content: center; }
.composer-chip.active { color: var(--destructive); background: var(--danger-light); }
.composer-chip:disabled { opacity: 0.4; cursor: not-allowed; }
.composer-chip:disabled:hover { background: transparent; color: color-mix(in srgb, var(--foreground) 78%, var(--muted-foreground)); }
.chip-label { font-size: 13px; }
.composer-chip :deep(svg) { width: 16px; height: 16px; }
.spacer { flex: 1; }
.send { width: 36px; height: 36px; display: inline-flex; align-items: center; justify-content: center; border: 0; border-radius: 999px; background: var(--primary); color: var(--primary-foreground); cursor: pointer; transition: var(--transition-fast); flex: none; }
.send:hover:not(:disabled) { background: var(--primary-hover); }
.send:disabled { opacity: 0.4; cursor: not-allowed; }
.send.unavailable { opacity: 0.45; cursor: not-allowed; }
.send.unavailable:hover { background: var(--primary); }
.composer-foot { display: flex; align-items: center; justify-content: center; gap: 6px; margin-top: 9px; font-size: 0.72rem; color: var(--muted-foreground); }
.composer-foot .kbd { display: inline-flex; align-items: center; padding: 1px 6px; border-radius: 6px; background: color-mix(in srgb, var(--muted-foreground) 14%, transparent); color: var(--muted-foreground); font-size: 0.7rem; font-weight: 600; }
@media (max-width: 880px) { .composer-wrap { padding: 10px 18px 16px; } .chip-label { display: none; } }
</style>
