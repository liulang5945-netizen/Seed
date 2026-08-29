<template>
  <section
    id="tk-panel-overview"
    class="tab-panel"
    :class="{ active }"
    role="tabpanel"
    aria-labelledby="tk-tab-overview"
  >
    <div v-if="trainState === 'running' || trainState === 'paused'" class="tk-card">
      <div class="progress-hero">
        <div>
          <div class="card-sub progress-title">Seed 原生 byte-stream 训练</div>
          <div class="pct">{{ trainProgress }}<span class="unit">%</span></div>
        </div>
        <div class="right-meta">
          <span v-if="trainState === 'running'" class="status-chip-run">训练中</span>
          <span v-else class="status-chip-paused">已暂停</span>
        </div>
      </div>
      <div class="progress-bar">
        <div class="fill" :class="{ paused: trainState === 'paused' }" :style="{ width: trainProgress + '%' }"></div>
      </div>
      <div class="progress-meta">
        <span class="pm-item"><svg class="pm-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>当前 epoch <b>{{ trainMetrics.epoch }} / {{ trainMetrics.total_epochs }}</b></span>
        <span class="pm-item"><svg class="pm-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>step <b>{{ trainMetrics.total_steps > 0 ? Math.round(trainMetrics.epoch * trainMetrics.total_steps / trainMetrics.total_epochs) : '--' }} / {{ trainMetrics.total_steps || '--' }}</b></span>
        <span class="pm-item"><svg class="pm-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l1.5-5 3 10 1.5-5h8"/></svg>剩余约 <b>{{ fmtTime(trainMetrics.eta) }}</b></span>
        <span class="pm-item"><svg class="pm-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6h16v12H4z"/><path d="M4 10h16M8 14h4"/></svg>吞吐 <b>{{ throughput }}</b></span>
      </div>
    </div>

    <div v-else class="tk-card">
      <div class="progress-hero">
        <div>
          <div class="card-sub progress-title">Seed 原生 byte-stream 训练</div>
          <div class="pct">0<span class="unit">%</span></div>
        </div>
        <div class="right-meta"><span class="status-chip-idle">待开始</span></div>
      </div>
      <div class="progress-bar"><div class="fill" style="width:0%"></div></div>
      <div class="progress-meta">
        <span class="pm-item"><svg class="pm-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>进度 <b>待开始</b></span>
        <span class="pm-item"><svg class="pm-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>step <b>0 / --</b></span>
        <span class="pm-item"><svg class="pm-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l1.5-5 3 10 1.5-5h8"/></svg>剩余约 <b>--</b></span>
        <span class="pm-item"><svg class="pm-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6h16v12H4z"/><path d="M4 10h16M8 14h4"/></svg>吞吐 <b>--</b></span>
      </div>
    </div>

    <div class="charts-grid">
      <div class="tk-card chart-card">
        <div class="chart-head">
          <h4>Loss 曲线</h4>
          <span class="legend"><i></i>训练损失</span>
        </div>
        <canvas v-if="trainLoss.length >= 2" ref="lossCanvasRef" class="loss-canvas" width="600" height="170"></canvas>
        <div v-else class="chart-empty">暂无数据，训练开始后自动绘制</div>
      </div>
    </div>

    <div class="metrics-row">
      <div class="metric-card">
        <div class="m-label"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l5-6 4 4 5-8 4 5"/></svg>Train Loss</div>
        <div class="m-value">{{ trainMetrics.current_loss != null ? trainMetrics.current_loss.toFixed(2) : '--' }}</div>
        <span class="m-trend flat">{{ trainState === 'running' ? '实时上报' : '待训练开始' }}</span>
      </div>
      <div class="metric-card">
        <div class="m-label"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6h16v12H4z"/><path d="M4 10h16M8 14h4"/></svg>吞吐</div>
        <div class="m-value">{{ throughputValue }}</div>
        <span class="m-trend flat">symbols/s</span>
      </div>
      <div class="metric-card">
        <div class="m-label"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>剩余时间</div>
        <div class="m-value">{{ fmtTime(trainMetrics.eta) }}</div>
        <span class="m-trend flat">ETA</span>
      </div>
    </div>

    <div v-if="pendingCheckpoints.length > 0" class="tk-card checkpoint-card">
      <div class="card-head">
        <h3><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 4h11l3 3v13H5z"/><path d="M8 4v5h7M8 14h8"/></svg>检查点列表</h3>
        <span class="card-sub">{{ pendingCheckpoints.length }} 个可用</span>
      </div>
      <div class="ckpt-list">
        <div v-for="ckpt in pendingCheckpoints" :key="ckpt.filename" class="ckpt-item">
          <span class="ckpt-ic"><PackageIcon :size="16" /></span>
          <div class="ckpt-body">
            <div class="ckpt-name">{{ ckpt.filename }}</div>
            <div class="ckpt-meta">
              <span class="ckpt-tag">Epoch <b>{{ ckpt.epoch }}</b></span>
              <span class="ckpt-tag">Step <b>{{ ckpt.step }}</b></span>
              <span class="ckpt-tag">Loss <b>{{ ckpt.loss?.toFixed(4) || '--' }}</b></span>
            </div>
          </div>
          <n-button size="small" type="info" quaternary round @click="emit('resume')">
            <template #icon><RefreshCw :size="14" /></template>恢复
          </n-button>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, nextTick, onMounted, watch } from 'vue'
import { Package as PackageIcon, RefreshCw } from 'lucide-vue-next'
import { drawLossChart, lossCanvasRef } from '../composables/useTraining.js'

const props = defineProps({
  active: { type: Boolean, default: false },
  trainState: { type: String, default: 'idle' },
  trainProgress: { type: Number, default: 0 },
  trainMetrics: { type: Object, required: true },
  trainLoss: { type: Array, default: () => [] },
  pendingCheckpoints: { type: Array, default: () => [] },
  fmtTime: { type: Function, required: true },
})

const emit = defineEmits(['resume'])

const formatRate = (value) => {
  const rate = Number(value)
  if (!Number.isFinite(rate) || rate < 0.005) return '--'
  return (rate < 0.1 ? rate.toFixed(2) : rate.toFixed(1)) + '/s'
}

const throughput = computed(() => formatRate(props.trainMetrics.samples_per_sec))
const throughputValue = computed(() => formatRate(props.trainMetrics.samples_per_sec).replace('/s', ''))

watch(() => props.trainLoss.length, () => {
  nextTick(() => drawLossChart())
})

// 面板常驻但用 display 切换，隐藏期间 canvas 测量为 0×0；重新激活后必须重绘，
// 否则曲线永远停留在空白状态。
watch(() => props.active, (isActive) => {
  if (isActive) nextTick(() => drawLossChart())
})

onMounted(() => nextTick(() => drawLossChart()))
</script>

<style scoped>
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.tk-card {
  background: var(--card, var(--bg-card));
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) * 0.7);
  padding: 22px;
  transition: border-color 0.16s ease;
}
.tk-card:hover { border-color: color-mix(in srgb, var(--primary) 20%, var(--border)); }
.tk-card .card-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
.tk-card h3 { margin: 0; font-size: 1rem; font-weight: 600; color: var(--foreground, var(--text)); display: flex; align-items: center; gap: 8px; }
.tk-card .card-sub { color: var(--muted-foreground, var(--text-muted)); font-size: 0.78rem; }
.progress-hero { display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; flex-wrap: wrap; }
.progress-title { margin-bottom: 6px; }
.progress-hero .pct { font-size: 2.7rem; font-weight: 700; line-height: 1; color: var(--foreground, var(--text)); font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
.progress-hero .pct .unit { font-size: 1.1rem; color: var(--muted-foreground, var(--text-muted)); font-weight: 500; margin-left: 2px; }
.right-meta { display: flex; align-items: center; gap: 18px; }
.status-chip-run, .status-chip-paused, .status-chip-idle { display: inline-flex; align-items: center; gap: 7px; padding: 5px 12px; border-radius: 999px; font: 600 0.76rem var(--font-sans, inherit); }
.status-chip-run { background: color-mix(in srgb, var(--chart-1) 16%, transparent); color: var(--chart-1); }
.status-chip-paused { background: color-mix(in srgb, var(--warning) 16%, transparent); color: var(--warning); }
.status-chip-idle { background: color-mix(in srgb, var(--muted-foreground, var(--text-muted)) 16%, transparent); color: var(--muted-foreground, var(--text-muted)); }
.status-chip-run::before, .status-chip-paused::before, .status-chip-idle::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.status-chip-run::before { animation: tkPulse 1.4s ease-in-out infinite; }
@keyframes tkPulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
.progress-bar { height: 12px; border-radius: 999px; background: var(--muted, var(--bg-muted)); overflow: hidden; margin-top: 16px; position: relative; }
.progress-bar .fill { height: 100%; background: linear-gradient(90deg, var(--chart-1), var(--chart-2)); border-radius: 999px; transition: width 0.4s ease; }
.progress-bar .fill.paused { background: linear-gradient(90deg, var(--warning), #f97316); }
.progress-meta { display: flex; gap: 30px; color: var(--muted-foreground, var(--text-muted)); font-size: 0.82rem; margin-top: 14px; flex-wrap: wrap; }
.progress-meta .pm-item { display: flex; align-items: center; gap: 6px; }
.progress-meta b { color: var(--foreground, var(--text)); font-weight: 600; font-variant-numeric: tabular-nums; }
.progress-meta .pm-icon { width: 15px; height: 15px; opacity: 0.7; }
.charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin: 18px 0; }
.chart-card .chart-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.chart-card h4 { margin: 0; font-size: 0.92rem; font-weight: 600; color: var(--foreground, var(--text)); }
.chart-card .legend { display: inline-flex; align-items: center; gap: 6px; font-size: 0.74rem; color: var(--muted-foreground, var(--text-muted)); }
.chart-card .legend i { width: 11px; height: 11px; border-radius: 3px; display: inline-block; background: var(--chart-2); }
.chart-empty { display: flex; align-items: center; justify-content: center; height: 170px; color: var(--muted-foreground, var(--text-muted)); font-size: 0.82rem; background: color-mix(in srgb, var(--muted) 40%, transparent); border-radius: calc(var(--radius) * 0.5); }
.loss-canvas { width: 100%; height: auto; max-height: 200px; display: block; }
.metrics-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 18px; }
.metric-card { background: var(--card, var(--bg-card)); border: 1px solid var(--border); border-radius: calc(var(--radius) * 0.6); padding: 16px 18px; transition: border-color 0.16s ease, transform 0.16s ease; }
.metric-card:hover { border-color: color-mix(in srgb, var(--primary) 35%, var(--border)); transform: translateY(-2px); }
.metric-card .m-label { font-size: 0.76rem; color: var(--muted-foreground, var(--text-muted)); display: flex; align-items: center; gap: 7px; }
.metric-card .m-label .ic { width: 15px; height: 15px; color: var(--chart-2); }
.metric-card .m-value { font-size: 1.7rem; font-weight: 700; margin-top: 8px; color: var(--foreground, var(--text)); font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }
.metric-card .m-trend { font-size: 0.74rem; margin-top: 5px; display: inline-flex; align-items: center; gap: 4px; }
.m-trend.flat { color: var(--muted-foreground, var(--text-muted)); }
.checkpoint-card { margin-top: 18px; }
.ckpt-list { display: flex; flex-direction: column; gap: 8px; }
.ckpt-item { display: flex; align-items: center; gap: 12px; padding: 12px 14px; border: 1px solid var(--border); border-left: 3px solid var(--primary); border-radius: calc(var(--radius) * 0.5); background: var(--card, var(--bg-card)); transition: border-color 0.16s ease, background 0.16s ease, transform 0.16s ease; }
.ckpt-item:hover { border-color: color-mix(in srgb, var(--primary) 30%, var(--border)); border-left-color: var(--primary); background: color-mix(in srgb, var(--primary) 5%, transparent); transform: translateX(2px); }
.ckpt-ic { width: 36px; height: 36px; flex-shrink: 0; display: grid; place-items: center; border-radius: 10px; background: var(--primary-light, color-mix(in srgb, var(--primary) 12%, transparent)); color: var(--primary); }
.ckpt-body { flex: 1; min-width: 0; }
.ckpt-name { font-size: 0.86rem; font-weight: 600; color: var(--foreground, var(--text)); font-family: var(--font-mono); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ckpt-meta { font-size: 0.76rem; color: var(--muted-foreground, var(--text-muted)); margin-top: 5px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.ckpt-tag { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 6px; background: var(--muted, var(--bg-muted)); }
.ckpt-meta b { color: var(--foreground, var(--text)); font-weight: 600; font-variant-numeric: tabular-nums; }
@media (max-width: 1180px) { .charts-grid { grid-template-columns: 1fr; } .metrics-row { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 720px) { .metrics-row { grid-template-columns: 1fr; } .progress-hero { flex-direction: column; align-items: flex-start; } }
@media (prefers-reduced-motion: reduce) { .status-chip-run::before { animation: none; } .metric-card:hover, .ckpt-item:hover { transform: none; } }
</style>
