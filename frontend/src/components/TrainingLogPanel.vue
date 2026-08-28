<template>
  <section
    id="tk-panel-logs"
    class="tab-panel"
    :class="{ active }"
    role="tabpanel"
    aria-labelledby="tk-tab-logs"
  >
    <div v-if="trainLog" class="log-panel">
      <div class="log-head">
        <div class="log-dots">
          <i class="dot-danger"></i>
          <i class="dot-warning"></i>
          <i class="dot-primary"></i>
        </div>
        <span class="log-title">training.log — {{ modelLabel }}</span>
        <span class="log-spacer"></span>
        <span class="log-tag">实时 · tail -f</span>
        <n-button size="tiny" quaternary round class="log-clear-btn" @click="emit('clear')">
          <template #icon><Trash2 :size="14" /></template>清空
        </n-button>
      </div>
      <pre ref="trainLogRef" class="log-body">{{ trainLog }}</pre>
    </div>
    <div v-else class="tk-card">
      <div class="log-empty">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M8 10h8M8 14h5"/></svg>
        <p>暂无训练日志</p>
        <span class="card-sub">训练开始后将实时输出日志</span>
      </div>
    </div>
  </section>
</template>

<script setup>
import { Trash2 } from 'lucide-vue-next'
import { trainLogRef } from '../composables/useTraining.js'

defineProps({
  active: { type: Boolean, default: false },
  trainLog: { type: String, default: '' },
  modelLabel: { type: String, default: 'Seed模型' },
})

const emit = defineEmits(['clear'])
</script>

<style scoped>
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.tk-card { background: var(--card, var(--bg-card)); border: 1px solid var(--border); border-radius: calc(var(--radius) * 0.7); padding: 22px; }
.card-sub { color: var(--muted-foreground, var(--text-muted)); font-size: 0.78rem; }
.log-panel { background: var(--card, var(--bg-card)); border: 1px solid var(--border); border-radius: calc(var(--radius) * 0.7); overflow: hidden; }
.log-head { display: flex; align-items: center; gap: 12px; padding: 12px 16px; border-bottom: 1px solid var(--border); background: color-mix(in srgb, var(--muted) 45%, var(--card, var(--bg-card))); }
.log-dots { display: flex; gap: 6px; }
.log-dots i { width: 11px; height: 11px; border-radius: 50%; display: block; }
.dot-danger { background: var(--destructive); }
.dot-warning { background: var(--warning, #eab308); }
.dot-primary { background: var(--chart-2); }
.log-title { font-size: 0.82rem; font-weight: 600; font-family: var(--font-mono); color: var(--foreground, var(--text)); }
.log-spacer { flex: 1; }
.log-tag { font: 600 0.7rem var(--font-mono); color: var(--muted-foreground, var(--text-muted)); padding: 3px 9px; border: 1px solid var(--border); border-radius: 999px; }
.log-clear-btn { margin-left: 8px; }
.log-body { font-family: var(--font-mono); font-size: 0.8rem; line-height: 1.75; padding: 16px 20px; overflow-x: auto; background: color-mix(in srgb, var(--background, var(--bg)) 86%, var(--card, var(--bg-card))); color: var(--foreground, var(--text)); white-space: pre-wrap; word-break: break-all; max-height: 400px; overflow-y: auto; margin: 0; }
.log-empty { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; padding: 60px 20px; text-align: center; }
.log-empty svg { width: 48px; height: 48px; color: var(--muted-foreground); opacity: 0.5; }
.log-empty p { margin: 0; font-size: 0.9rem; font-weight: 500; color: var(--foreground, var(--text)); }
</style>
