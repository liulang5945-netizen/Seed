<template>
  <section class="api-metrics" :class="{ compact }" aria-label="原生接口观测">
    <div class="metrics-head">
      <div>
        <span class="metrics-eyebrow">TRACE / SLO</span>
        <strong>原生接口观测</strong>
      </div>
      <span class="metrics-refresh">不含请求内容</span>
    </div>

    <div v-if="rows.length" class="metrics-grid">
      <article v-for="item in rows" :key="item.path" class="metric-item">
        <div class="metric-path" :title="item.path">{{ item.path }}</div>
        <div class="metric-stats">
          <span>请求 {{ item.requests }}</span>
          <span class="metric-success">成功 {{ item.successes }}</span>
          <span v-if="item.failures" class="metric-failure">失败 {{ item.failures }}</span>
        </div>
        <div class="metric-meta">
          <span>平均 {{ item.average_latency_ms }} ms</span>
          <span :class="statusClass(item.last_status)">{{ statusLabel(item.last_status) }}</span>
        </div>
      </article>
    </div>
    <p v-else class="metrics-empty">暂无 nativeApi 请求记录</p>
  </section>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { nativeApiMetrics } from '../composables/nativeApi.js'

const props = defineProps({
  compact: { type: Boolean, default: false },
  limit: { type: Number, default: 6 },
  refreshMs: { type: Number, default: 2000 },
})

const revision = ref(0)
let refreshTimer = null

const rows = computed(() => {
  revision.value
  return Object.entries(nativeApiMetrics.snapshot())
    .map(([path, value]) => ({ path, ...value }))
    .sort((a, b) => b.last_observed_at - a.last_observed_at || b.requests - a.requests)
    .slice(0, Math.max(1, props.limit))
})

function statusLabel(status) {
  const code = Number(status || 0)
  return code ? `HTTP ${code}` : '网络失败'
}

function statusClass(status) {
  return Number(status || 0) >= 200 && Number(status || 0) < 400
    ? 'is-ready'
    : 'is-warning'
}

onMounted(() => {
  refreshTimer = window.setInterval(() => {
    revision.value += 1
  }, Math.max(250, props.refreshMs))
})

onUnmounted(() => {
  if (refreshTimer !== null) window.clearInterval(refreshTimer)
  refreshTimer = null
})
</script>

<style scoped>
.api-metrics {
  margin: 10px 0 18px;
  padding: 12px 14px;
  border: 1px solid color-mix(in srgb, var(--border) 88%, var(--primary) 12%);
  border-radius: 12px;
  background: color-mix(in srgb, var(--card) 96%, var(--primary) 4%);
}
.api-metrics.compact { margin-bottom: 14px; padding: 10px 12px; }
.metrics-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 9px; }
.metrics-head > div { display: flex; align-items: baseline; gap: 8px; }
.metrics-eyebrow { color: var(--muted-foreground); font-size: 10px; letter-spacing: .1em; }
.metrics-head strong { color: var(--foreground); font-size: .78rem; }
.metrics-refresh { color: var(--muted-foreground); font-size: .68rem; }
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 7px; }
.metric-item { min-width: 0; padding: 8px 9px; border: 1px solid color-mix(in srgb, var(--border) 78%, transparent); border-radius: 9px; background: var(--background); }
.metric-path { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--foreground); font-family: var(--font-mono); font-size: .68rem; }
.metric-stats, .metric-meta { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 5px; color: var(--muted-foreground); font-size: .66rem; }
.metric-success, .metric-meta .is-ready { color: var(--success, #16803b); }
.metric-failure, .metric-meta .is-warning { color: var(--warning, #a15c00); }
.metrics-empty { margin: 0; color: var(--muted-foreground); font-size: .7rem; }
</style>
