<template>
  <div class="needs-pentagram" role="img" :aria-label="ariaLabel" :data-tier="totalTier">
    <svg viewBox="0 0 320 320" xmlns="http://www.w3.org/2000/svg" class="pentagram-svg">
      <polygon v-for="level in 5" :key="'guide-' + level" :points="gridPolygon(level * 0.2)" class="pentagram-guide" />
      <line v-for="(pt, i) in outerVertices" :key="'axis-' + i" :x1="cx" :y1="cy" :x2="pt.x" :y2="pt.y" class="pentagram-axis" :class="{ critical: needLevels[i] === 'alert' }" />
      <g class="pentagram-body" :class="{ breathing: alive }">
        <path :d="dataPath" class="pentagram-fill" />
        <path :d="dataPath" class="pentagram-stroke" fill="none" />
      </g>
      <text v-for="(pt, i) in labelPositions" :key="'label-' + i" :x="pt.x" :y="pt.y" :text-anchor="pt.anchor" :dominant-baseline="pt.baseline" class="pentagram-label">{{ needLabels[needKeys[i]] }} <tspan class="pentagram-value">{{ Math.round(needs[needKeys[i]] || 0) }}</tspan></text>
    </svg>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  needs: { type: Object, default: () => ({ hunger: 0, fatigue: 0, boredom: 0, stress: 0, curiosity: 0 }) },
  alive: { type: Boolean, default: false },
})

const cx = 160; const cy = 158; const outerR = 105; const innerR = 14
const needKeys = ['hunger', 'fatigue', 'boredom', 'stress', 'curiosity']
const needLabels = { hunger: '饿', fatigue: '累', boredom: '闷', stress: '压', curiosity: '奇' }
const WATCH_LEVEL = 40
const ALERT_LEVEL = 70

const ariaLabel = computed(() =>
  needKeys.map(k => needLabels[k] + ':' + Math.round(props.needs[k] || 0)).join('，')
)

const needLevels = computed(() =>
  needKeys.map(k => {
    const val = props.needs[k] || 0
    return val > ALERT_LEVEL ? 'alert' : val >= WATCH_LEVEL ? 'watch' : 'calm'
  })
)

const needsTotal = computed(() =>
  needKeys.reduce((sum, k) => sum + Math.max(0, Math.min(100, props.needs[k] || 0)), 0)
)

const totalTier = computed(() => {
  const total = needsTotal.value
  if (total > ALERT_LEVEL * needKeys.length) return 'alert'
  if (total >= WATCH_LEVEL * needKeys.length) return 'watch'
  return 'calm'
})

const angleFor = (i) => -Math.PI / 2 + i * (2 * Math.PI / 5)
const vertexAt = (r, i) => ({ x: cx + r * Math.cos(angleFor(i)), y: cy + r * Math.sin(angleFor(i)) })

const dataVertices = computed(() =>
  needKeys.map((k, i) => {
    const val = Math.max(0, Math.min(100, props.needs[k] || 0))
    return vertexAt(innerR + (val / 100) * (outerR - innerR), i)
  })
)
const outerVertices = computed(() => needKeys.map((_, i) => vertexAt(outerR, i)))
const dataPath = computed(() =>
  'M' + dataVertices.value.map(pt => pt.x.toFixed(1) + ' ' + pt.y.toFixed(1)).join('L') + 'Z'
)
const gridPolygon = (fraction) => {
  const r = innerR + fraction * (outerR - innerR)
  return needKeys.map((_, i) => vertexAt(r, i)).map(pt => pt.x.toFixed(1) + ',' + pt.y.toFixed(1)).join(' ')
}
const labelPositions = computed(() =>
  needKeys.map((_, i) => {
    const a = angleFor(i); const dx = Math.cos(a); const dy = Math.sin(a); const r = outerR + 24
    let ox = 0, oy = 0
    if (Math.abs(dy) > 0.8) oy = dy > 0 ? 4 : -2
    else if (Math.abs(dx) > 0.8) ox = dx > 0 ? 4 : -4
    return {
      x: cx + r * dx + ox, y: cy + r * dy + oy,
      anchor: dx > 0.3 ? 'start' : dx < -0.3 ? 'end' : 'middle',
      baseline: dy > 0.5 ? 'hanging' : dy < -0.5 ? 'auto' : 'middle',
    }
  })
)
</script>

<style scoped>
.needs-pentagram { display: flex; align-items: center; justify-content: center; user-select: none; --tier: var(--needs-tier-calm); }
.needs-pentagram[data-tier='calm'] { --tier: var(--needs-tier-calm); }
.needs-pentagram[data-tier='watch'] { --tier: var(--needs-tier-watch); }
.needs-pentagram[data-tier='alert'] { --tier: var(--needs-tier-alert); }
.pentagram-svg { width: 100%; max-width: 320px; height: auto; aspect-ratio: 1; display: block; }
.pentagram-guide { fill: none; stroke: color-mix(in srgb, var(--muted-foreground) 16%, transparent); stroke-width: 1; }
.pentagram-guide:last-of-type { stroke: color-mix(in srgb, var(--muted-foreground) 32%, transparent); stroke-width: 1.2; }
.pentagram-axis {
  stroke: color-mix(in srgb, var(--muted-foreground) 14%, transparent);
  stroke-width: 0.75;
  transition: stroke 0.4s var(--ease), stroke-width 0.4s var(--ease);
}
.pentagram-axis.critical { stroke: color-mix(in srgb, var(--tier) 55%, transparent); stroke-width: 1.6; }
.pentagram-body { transform-origin: 160px 158px; transition: transform 0.5s var(--ease); }
.pentagram-body.breathing { animation: pentagram-breathe 6s ease-in-out infinite; }
@keyframes pentagram-breathe { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.025); } }
.pentagram-fill { fill: color-mix(in srgb, var(--tier) 20%, transparent); transition: d 0.6s var(--ease), fill 0.4s var(--ease); }
.pentagram-stroke {
  stroke: var(--tier);
  stroke-width: 1.8;
  stroke-linejoin: round;
  transition: d 0.6s var(--ease), stroke 0.4s var(--ease);
}
.pentagram-label { fill: var(--muted-foreground, var(--text-muted)); font-size: 12px; font-weight: 500; font-family: var(--font); }
.pentagram-value { fill: var(--tier); font-weight: 600; transition: fill 0.4s var(--ease); }
@media (prefers-reduced-motion: reduce) {
  .pentagram-body.breathing { animation: none; }
  .pentagram-fill, .pentagram-stroke, .pentagram-axis, .pentagram-value { transition: none; }
}
</style>
