<template>
  <SettingsPanelSection title="Taiji 运行设置">

    <div class="setting-row setting-row--first">
      <div class="setting-left">
        <span class="setting-label">局部激活阈值</span>
        <p class="setting-desc">控制单步局部状态更新的激活门槛，不代表全局同步或相位共振</p>
      </div>
      <div class="setting-right">
        <div class="range-wrap">
          <input
            :value="activationThreshold"
            type="range"
            min="0"
            max="1"
            step="0.01"
            aria-label="局部激活阈值"
            :disabled="saving"
            @change="emitNumber('activation-threshold-change', $event)"
          />
          <span class="range-value">{{ Number(activationThreshold).toFixed(2) }}</span>
        </div>
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-left">
        <span class="setting-label">响应超时</span>
        <p class="setting-desc">一次状态推进等待后端返回的最长时间（毫秒）</p>
      </div>
      <div class="setting-right">
        <input
          :value="responseTimeoutMs"
          type="number"
          min="10"
          max="10000"
          aria-label="响应超时"
          :disabled="saving"
          @change="emitNumber('response-timeout-change', $event)"
        />
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-left">
        <span class="setting-label">自动巩固</span>
        <p class="setting-desc">高频突触自动强化并写入持久记忆</p>
      </div>
      <div class="setting-right">
        <label class="toggle" aria-label="自动巩固开关">
          <input
            :checked="autoConsolidation"
            type="checkbox"
            :disabled="saving"
            @change="emit('auto-consolidation-change', $event.target.checked)"
          />
          <span class="track"><span class="thumb"></span></span>
        </label>
      </div>
    </div>

    <div class="setting-row setting-row--last">
      <div class="setting-left">
        <span class="setting-label">睡眠模式</span>
        <p class="setting-desc">非活跃时段暂停后台状态推进以节省算力，唤醒后恢复 Taiji 持续状态</p>
      </div>
      <div class="setting-right">
        <label class="toggle" aria-label="睡眠模式开关">
          <input
            :checked="sleepMode"
            type="checkbox"
            :disabled="saving"
            @change="emit('sleep-mode-change', $event.target.checked)"
          />
          <span class="track"><span class="thumb"></span></span>
        </label>
      </div>
    </div>
  </SettingsPanelSection>
</template>

<script setup>
import SettingsPanelSection from './SettingsPanelSection.vue'

defineOptions({ name: 'SettingsTaijiPanel' })

defineProps({
  activationThreshold: { type: Number, default: 0.72 },
  responseTimeoutMs: { type: Number, default: 100 },
  autoConsolidation: { type: Boolean, default: true },
  sleepMode: { type: Boolean, default: false },
  saving: { type: Boolean, default: false },
})

const emit = defineEmits([
  'activation-threshold-change',
  'response-timeout-change',
  'auto-consolidation-change',
  'sleep-mode-change',
])

function emitNumber(eventName, event) {
  emit(eventName, Number(event.target.value))
}
</script>

<style scoped>
.settings-section { display: flex; flex-direction: column; }
.settings-section h2 { font-size: 1.15rem; font-weight: 650; margin: 0 0 18px; color: var(--foreground); letter-spacing: -0.01em; }
.setting-row { display: flex; align-items: center; justify-content: space-between; padding: 14px 0; border-bottom: 1px solid var(--border); gap: 16px; }
.setting-row--first { padding-top: 0; }
.setting-row--last { border-bottom: none; padding-bottom: 0; }
.setting-left { min-width: 0; flex: 1; }
.setting-right { flex-shrink: 0; display: flex; align-items: center; gap: 8px; }
.setting-label { font-size: 0.88rem; font-weight: 500; color: var(--foreground); line-height: 1.3; }
.setting-desc { font-size: 0.76rem; color: var(--muted-foreground); margin: 3px 0 0; line-height: 1.4; }
.setting-right input[type="number"] { width: 90px; text-align: center; font-variant-numeric: tabular-nums; background: var(--background); color: var(--foreground); border: 1px solid var(--border); border-radius: calc(var(--radius) * 0.45); padding: 7px 12px; font-size: 0.86rem; font-family: var(--font-sans); outline: none; transition: border-color 160ms ease, box-shadow 160ms ease; }
.setting-right input[type="number"]:focus { border-color: var(--ring); box-shadow: 0 0 0 3px color-mix(in srgb, var(--ring) 16%, transparent); }
.range-wrap { display: flex; align-items: center; gap: 10px; }
.range-value { font-size: 0.82rem; font-weight: 600; font-variant-numeric: tabular-nums; color: var(--primary); min-width: 42px; text-align: right; }
.setting-right input[type="range"] { -webkit-appearance: none; appearance: none; width: 140px; height: 6px; background: var(--border); border-radius: 999px; border: 0; padding: 0; outline: none; cursor: pointer; }
.setting-right input[type="range"]:focus { box-shadow: none; }
.setting-right input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%; background: var(--primary); cursor: pointer; border: 2px solid var(--background); box-shadow: 0 1px 4px color-mix(in srgb, var(--foreground) 18%, transparent); transition: transform 120ms ease; }
.setting-right input[type="range"]::-webkit-slider-thumb:hover { transform: scale(1.15); }
.setting-right input[type="range"]::-moz-range-thumb { width: 18px; height: 18px; border-radius: 50%; background: var(--primary); cursor: pointer; border: 2px solid var(--background); box-shadow: 0 1px 4px color-mix(in srgb, var(--foreground) 18%, transparent); }
.toggle { position: relative; display: inline-block; width: 42px; height: 24px; flex-shrink: 0; cursor: pointer; }
.toggle input { opacity: 0; width: 0; height: 0; position: absolute; border: 0; padding: 0; }
.toggle .track { position: absolute; inset: 0; background: var(--border); border-radius: 999px; cursor: pointer; transition: background 200ms ease; }
.toggle .thumb { position: absolute; top: 3px; left: 3px; width: 18px; height: 18px; background: var(--background); border-radius: 50%; transition: transform 200ms cubic-bezier(0.34, 1.56, 0.64, 1); box-shadow: 0 1px 3px color-mix(in srgb, var(--foreground) 16%, transparent); }
.toggle input:checked + .track { background: var(--primary); }
.toggle input:checked + .track .thumb { transform: translateX(18px); }
.toggle input:focus-visible + .track { box-shadow: 0 0 0 3px color-mix(in srgb, var(--ring) 20%, transparent); }
</style>
