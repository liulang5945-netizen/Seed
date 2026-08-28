<template>
  <section class="runtime-evidence" :class="{ compact }" aria-label="运行时状态证据">
    <div class="evidence-head">
      <div>
        <span class="evidence-eyebrow">STATUS EVIDENCE</span>
        <strong>状态依据</strong>
      </div>
      <span class="evidence-refresh">{{ store.statusEvidence.freshness.label }}</span>
    </div>

    <div class="evidence-grid">
      <article v-for="item in rows" :key="item.label" class="evidence-item">
        <div class="evidence-item-head">
          <span class="evidence-label">{{ item.label }}</span>
          <span class="evidence-availability" :class="availabilityClass(item.availability)">
            {{ item.availability }}
          </span>
        </div>
        <strong class="evidence-owner">{{ item.owner }}</strong>
        <span class="evidence-meta">{{ item.freshness.label }} · {{ item.source }}</span>
        <span v-if="item.detail" class="evidence-detail">{{ item.detail }}</span>
      </article>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import { useRuntimeStore } from '../stores/runtimeStore.js'

const props = defineProps({
  context: { type: String, default: 'all' },
  compact: { type: Boolean, default: false },
})

const store = useRuntimeStore()
const contextRows = {
  chat: ['runtime', 'provider'],
  life: ['runtime', 'homeostasis'],
  agent: ['runtime', 'workbench'],
  training: ['runtime', 'training'],
  settings: ['runtime', 'provider', 'workbench'],
  knowledge: ['runtime', 'knowledge'],
  all: ['runtime', 'provider', 'workbench', 'homeostasis', 'training'],
}

const rows = computed(() => (contextRows[props.context] || contextRows.all)
  .map(key => store.statusEvidence[key])
  .filter(Boolean))

function availabilityClass(value) {
  const text = String(value || '')
  if (/可用|已接入|已上报|训练中/.test(text)) return 'is-ready'
  if (/回退|空闲|未上报|未接入|未知|未连接/.test(text)) return 'is-muted'
  return 'is-warning'
}
</script>

<style scoped>
.runtime-evidence {
  margin: 0 0 18px;
  padding: 14px 16px;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: color-mix(in srgb, var(--card) 92%, var(--primary) 8%);
}
.runtime-evidence.compact { padding: 11px 13px; }
.evidence-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 11px; }
.evidence-head > div { display: flex; align-items: baseline; gap: 8px; }
.evidence-eyebrow { color: var(--muted-foreground); font-size: 10px; letter-spacing: .12em; }
.evidence-head strong { color: var(--foreground); font-size: .82rem; }
.evidence-refresh { color: var(--muted-foreground); font-size: .72rem; }
.evidence-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; }
.evidence-item { min-width: 0; padding: 10px; border: 1px solid color-mix(in srgb, var(--border) 80%, transparent); border-radius: 10px; background: var(--background); }
.evidence-item-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.evidence-label { color: var(--muted-foreground); font-size: .72rem; }
.evidence-availability { border-radius: 999px; padding: 2px 6px; font-size: .68rem; white-space: nowrap; }
.evidence-availability.is-ready { color: var(--success, #16803b); background: color-mix(in srgb, var(--success, #16803b) 12%, transparent); }
.evidence-availability.is-muted { color: var(--muted-foreground); background: var(--muted); }
.evidence-availability.is-warning { color: var(--warning, #a15c00); background: color-mix(in srgb, var(--warning, #a15c00) 12%, transparent); }
.evidence-owner, .evidence-meta, .evidence-detail { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.evidence-owner { margin-top: 7px; color: var(--foreground); font-size: .77rem; font-weight: 600; }
.evidence-meta, .evidence-detail { margin-top: 4px; color: var(--muted-foreground); font-size: .68rem; }
@media (max-width: 720px) { .evidence-grid { grid-template-columns: 1fr 1fr; } }
@media (max-width: 460px) { .evidence-grid { grid-template-columns: 1fr; } }
</style>
