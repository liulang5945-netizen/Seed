<template>
  <div class="needs-pentagram" role="img" :aria-label="ariaLabel">
    <svg viewBox="0 0 320 320" xmlns="http://www.w3.org/2000/svg" class="pentagram-svg">
      <defs>
        <radialGradient :id="areaGradientId" gradientUnits="userSpaceOnUse" :cx="cx" :cy="cy" :r="outerR">
          <stop class="area-core" offset="0%" />
          <stop class="area-edge" offset="100%" />
        </radialGradient>
      </defs>
      <polygon v-for="level in 5" :key="'guide-' + level" :points="gridPolygon(level * 0.2)" class="pentagram-guide" />
      <line v-for="(pt, i) in outerVertices" :key="'axis-' + i" :x1="cx" :y1="cy" :x2="pt.x" :y2="pt.y" class="pentagram-axis" />
      <g class="pentagram-body" :class="{ breathing: alive }">
        <path :d="dataPath" class="pentagram-fill" :fill="`url(#${areaGradientId})`" />
        <path :d="dataPath" class="pentagram-stroke" fill="none" />
      </g>
      <circle v-for="(pt, i) in dataVertices" :key="'dot-' + i" :cx="pt.x" :cy="pt.y" :r="(needs[needKeys[i]] || 0) > 70 ? 5.5 : 4" class="pentagram-dot" :class="{ critical: (needs[needKeys[i]] || 0) > 70 }" />
      <text v-for="(pt, i) in labelPositions" :key="'label-' + i" :x="pt.x" :y="pt.y" :text-anchor="pt.anchor" :dominant-baseline="pt.baseline" class="pentagram-label" :class="{ critical: (needs[needKeys[i]] || 0) > 70 }">{{ needLabels[needKeys[i]] }} {{ Math.round(needs[needKeys[i]] || 0) }}</text>
    </svg>
  </div>
</template>

<script setup>
import { computed, useId } from 'vue'

const props = defineProps({
  needs: { type: Object, default: () => ({ hunger: 0, fatigue: 0, boredom: 0, stress: 0, curiosity: 0 }) },
  alive: { type: Boolean, default: false },
})

const cx = 160; const cy = 158; const outerR = 105; const innerR = 14
const areaGradientId = 'pentagram-area-' + useId()
const needKeys = ['hunger', 'fatigue', 'boredom', 'stress', 'curiosity']
const needLabels = { hunger: '饿', fatigue: '累', boredom: '闷', stress: '压', curiosity: '奇' }

const ariaLabel = computed(() =>
  needKeys.map(k => needLabels[k] + ':' + Math.round(props.needs[k] || 0)).join('，')
)

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
.needs-pentagram { display: flex; align-items: center; justify-content: center; user-select: none; }
.pentagram-svg { width: 100%; max-width: 320px; height: auto; aspect-ratio: 1; display: block; }
.area-core { stop-color: var(--chart-1); stop-opacity: 0.42; }
.area-edge { stop-color: var(--chart-3); stop-opacity: 0.14; }
.pentagram-guide { fill: none; stroke: color-mix(in srgb, var(--chart-2) 22%, var(--border)); stroke-width: 1; }
.pentagram-guide:last-of-type { stroke: color-mix(in srgb, var(--chart-2) 40%, var(--border)); stroke-width: 1.4; }
.pentagram-axis { stroke: color-mix(in srgb, var(--chart-2) 16%, var(--border)); stroke-width: 0.75; }
.pentagram-body { transform-origin: 160px 158px; transition: transform 0.5s var(--ease); }
.pentagram-body.breathing { animation: pentagram-breathe 6s ease-in-out infinite; }
@keyframes pentagram-breathe { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.025); } }
.pentagram-fill { transition: d 0.6s var(--ease); }
.pentagram-stroke { stroke: var(--chart-2); stroke-width: 2; stroke-linejoin: round; transition: d 0.6s var(--ease); }
.pentagram-dot {
  fill: var(--card, var(--bg-card));
  stroke: var(--chart-2);
  stroke-width: 2;
  transition: fill 0.4s var(--ease), stroke 0.4s var(--ease), r 0.4s var(--ease);
}
.pentagram-dot.critical { fill: var(--danger); stroke: var(--danger); animation: critical-pulse 1.8s ease-in-out infinite; }
@keyframes critical-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
.pentagram-label { fill: var(--muted-foreground, var(--text-muted)); font-size: 12px; font-weight: 500; font-family: var(--font); }
.pentagram-label.critical { fill: var(--danger); font-weight: 600; }
@media (prefers-reduced-motion: reduce) {
  .pentagram-body.breathing { animation: none; }
  .pentagram-dot.critical { animation: none; }
  .pentagram-fill, .pentagram-stroke { transition: none; }
}
</style>