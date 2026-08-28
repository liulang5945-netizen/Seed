<template>
  <div class="life-needs-dashboard">
    <div class="chart-grid">
      <div class="panel">
        <div class="panel-head">
          <span class="panel-title">
            <svg class="pt-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 3v4M12 17v4M3 12h4M17 12h4"/><circle cx="12" cy="12" r="3" fill="currentColor" stroke="none" opacity="0.3"/></svg>
            需求五维雷达
          </span>
          <span class="panel-sub">实时需求值</span>
        </div>
        <div class="chart-wrap">
          <NeedsPentagram :needs="needs" :alive="alive" />
        </div>
        <p v-if="!hasNeedsData" class="panel-empty">暂无需求数据——生命调度器尚未上报 needs。</p>
      </div>

      <div class="panel">
        <div class="panel-head">
          <span class="panel-title">
            <svg class="pt-icon" viewBox="0 0 24 24"><path d="M12 21a9 9 0 1 0-9-9c0 1.6.4 3.1 1.2 4.4L3 21l4.6-1.2A9 9 0 0 0 12 21Z"/></svg>
            生命表达
          </span>
          <span class="panel-sub">由需求状态推导</span>
        </div>
        <div v-if="lifeExpressions.length" class="expr-list">
          <div v-for="(expr, i) in lifeExpressions" :key="i" class="expr-item" :class="'expr-' + expr.priority">
            <span class="expr-emoji">{{ expr.emoji }}</span>
            <span class="expr-text">{{ expr.text }}</span>
          </div>
        </div>
        <p v-else class="panel-empty">当前没有主动表达——各项需求都在平稳区间。</p>
      </div>
    </div>

    <div class="bottom-grid">
      <div class="panel">
        <div class="panel-head">
          <span class="panel-title">
            <svg class="pt-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="2"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1"/></svg>
            需求明细
          </span>
          <span class="panel-sub">共 {{ needRows.length }} 项</span>
        </div>
        <table class="neuron-table">
          <thead>
            <tr><th>需求</th><th>当前值</th><th>强度</th><th>状态</th></tr>
          </thead>
          <tbody>
            <tr v-for="row in needRows" :key="row.key">
              <td><span class="n-domain">{{ row.label }}</span></td>
              <td><span class="n-id">{{ row.value != null ? Math.round(row.value) : '暂无数据' }}</span></td>
              <td>
                <div class="n-activity">
                  <div class="n-progress"><div class="n-progress-bar" :style="{ width: (row.value != null ? row.value : 0) + '%' }"></div></div>
                  <span class="n-progress-text">{{ row.value != null ? Math.round(row.value) + '%' : '--' }}</span>
                </div>
              </td>
              <td>
                <span v-if="row.value == null" class="chip chip-dormant">暂无数据</span>
                <span v-else-if="row.state === 'alert'" class="chip chip-alert">偏高</span>
                <span v-else-if="row.state === 'watch'" class="chip chip-learning">关注</span>
                <span v-else class="chip chip-active">平稳</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="panel">
        <div class="panel-head">
          <span class="panel-title">
            <svg class="pt-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
            生命事件流
          </span>
          <span class="panel-sub">本页操作记录</span>
        </div>
        <div class="event-list">
          <template v-if="activityLog.length">
            <div v-for="(log, i) in activityLog" :key="i" class="event-item" :class="'ev-' + log.type">
              <div class="event-dot"><span class="ev-emoji">{{ log.emoji }}</span></div>
              <div class="event-body">
                <div class="event-text">{{ log.message }}</div>
                <div class="event-meta">{{ log.time }}</div>
              </div>
            </div>
          </template>
          <p v-else class="event-empty">暂无生命事件。点击上方「喂养 / 睡眠 / 玩耍 / 进化」触发一次生命活动，或等待调度器自动运行。</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import NeedsPentagram from './NeedsPentagram.vue'

defineOptions({ name: 'LifeNeedsDashboard' })

defineProps({
  needs: { type: Object, default: () => ({}) },
  alive: { type: Boolean, default: false },
  hasNeedsData: { type: Boolean, default: false },
  lifeExpressions: { type: Array, default: () => [] },
  needRows: { type: Array, default: () => [] },
  activityLog: { type: Array, default: () => [] },
})
</script>

<style scoped>
.life-needs-dashboard { display: flex; flex-direction: column; gap: 16px; }
.chart-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }
.panel { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 18px 20px; display: flex; flex-direction: column; }
.panel-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
.panel-title { font-size: 0.92rem; font-weight: 600; color: var(--foreground); display: flex; align-items: center; gap: 8px; }
.pt-icon { width: 17px; height: 17px; flex: none; color: var(--primary); stroke: currentColor; fill: none; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }
.panel-sub { font-size: 0.74rem; color: var(--muted-foreground); }
.panel-empty { margin: auto 0; padding: 24px 8px; color: var(--muted-foreground); font-size: 0.8rem; line-height: 1.6; text-align: center; }
.chart-wrap { flex: 1; display: grid; place-items: center; min-height: 260px; }
.chart-wrap > * { width: 100%; max-width: 320px; }
.expr-list { display: flex; flex-direction: column; gap: 10px; padding: 6px 0; }
.expr-item { display: flex; align-items: flex-start; gap: 10px; padding: 10px 12px; border: 1px solid var(--border); border-radius: 10px; background: color-mix(in srgb, var(--accent) 20%, transparent); }
.expr-item.expr-high { border-color: color-mix(in srgb, var(--destructive) 40%, var(--border)); }
.expr-emoji { font-size: 1.05rem; line-height: 1.4; }
.expr-text { font-size: 0.82rem; color: var(--foreground); line-height: 1.5; }
.bottom-grid { display: grid; grid-template-columns: 1.7fr minmax(0, 1fr); gap: 16px; align-items: start; }
.neuron-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
.neuron-table thead th { text-align: left; font-weight: 600; color: var(--muted-foreground); font-size: 0.76rem; padding: 10px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; }
.neuron-table tbody td { padding: 11px 12px; border-bottom: 1px solid var(--border); color: var(--foreground); vertical-align: middle; white-space: nowrap; }
.neuron-table tbody tr:last-child td { border-bottom: 0; }
.neuron-table tbody tr:hover { background: color-mix(in srgb, var(--accent) 35%, transparent); }
.n-id { font-family: var(--font-mono); font-size: 0.8rem; color: var(--foreground); }
.n-domain { color: var(--muted-foreground); }
.n-activity { display: flex; align-items: center; gap: 9px; min-width: 130px; }
.n-progress { flex: 1; height: 6px; border-radius: 999px; background: var(--muted); overflow: hidden; }
.n-progress-bar { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--chart-1), var(--chart-2)); transition: width 300ms ease; }
.n-progress-text { font-size: 0.76rem; color: var(--muted-foreground); font-variant-numeric: tabular-nums; width: 36px; text-align: right; }
.chip { display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 999px; font-size: 0.74rem; font-weight: 500; line-height: 1.5; }
.chip::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex: none; }
.chip-active { color: var(--chart-2); background: color-mix(in srgb, var(--chart-2) 14%, transparent); }
.chip-dormant { color: var(--muted-foreground); background: color-mix(in srgb, var(--muted-foreground) 14%, transparent); }
.chip-learning { color: var(--chart-3); background: color-mix(in srgb, var(--chart-3) 14%, transparent); }
.chip-alert { color: var(--destructive); background: color-mix(in srgb, var(--destructive) 14%, transparent); }
.event-list { display: flex; flex-direction: column; gap: 0; margin-top: 2px; }
.event-item { display: flex; gap: 12px; padding: 10px 6px; border-bottom: 1px dashed var(--border); transition: background 140ms ease; }
.event-item:hover { background: color-mix(in srgb, var(--accent) 25%, transparent); }
.event-item:last-child { border-bottom: 0; }
.event-dot { width: 30px; height: 30px; border-radius: 10px; flex: none; display: grid; place-items: center; background: color-mix(in srgb, var(--chart-1) 14%, transparent); color: var(--chart-1); }
.ev-emoji { font-size: 1rem; line-height: 1; }
.event-body { flex: 1; min-width: 0; }
.event-text { font-size: 0.82rem; color: var(--foreground); line-height: 1.45; }
.event-meta { font-size: 0.72rem; color: var(--muted-foreground); margin-top: 4px; }
.event-empty { margin: 0; padding: 20px 6px; color: var(--muted-foreground); font-size: 0.8rem; line-height: 1.7; }
.ev-feed .event-dot { background: color-mix(in srgb, var(--chart-2) 14%, transparent); color: var(--chart-2); }
.ev-sleep .event-dot { background: color-mix(in srgb, var(--chart-1) 14%, transparent); color: var(--chart-1); }
.ev-play .event-dot { background: color-mix(in srgb, var(--chart-3) 14%, transparent); color: var(--chart-3); }
.ev-evolve .event-dot { background: color-mix(in srgb, var(--chart-4) 14%, transparent); color: var(--chart-4); }
.ev-export .event-dot { background: color-mix(in srgb, var(--primary) 14%, transparent); color: var(--primary); }
@media (max-width: 1180px) { .chart-grid, .bottom-grid { grid-template-columns: 1fr; } }
</style>
