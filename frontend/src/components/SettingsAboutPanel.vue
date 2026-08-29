<template>
  <SettingsPanelSection title="关于">

    <div class="setting-row setting-row--first">
      <div class="about-meta">
        <div class="meta-line">
          <span class="product-name">Seed神经元系统</span>
          <span class="meta-tag">v{{ appVersion }}</span>
        </div>
        <div class="meta-line meta-line--muted">
          原生基底
          <span class="meta-tag">seed-native-v1</span>
        </div>
        <div class="meta-line meta-line--muted">
          状态通路
          <span class="meta-tag">Taiji runtime → native capabilities</span>
        </div>
      </div>
    </div>

    <div class="setting-row setting-row--last">
      <div class="setting-left">
        <span class="setting-label">开源许可</span>
        <p class="setting-desc">查看本系统使用的第三方组件许可协议</p>
      </div>
      <div class="setting-right">
        <button class="btn-sm btn-ghost" aria-label="查看许可" @click="emit('show-license')">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/><path d="M12 18v-6"/><path d="M9 15h6"/></svg>
          查看许可
        </button>
      </div>
    </div>
  </SettingsPanelSection>
</template>

<script setup>
import SettingsPanelSection from './SettingsPanelSection.vue'

defineOptions({ name: 'SettingsAboutPanel' })

defineProps({
  appVersion: { type: String, default: '1.0.0' },
})

const emit = defineEmits(['show-license'])
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
.about-meta { display: flex; flex-direction: column; gap: 6px; padding: 6px 0; }
.meta-line { display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: var(--foreground); }
.product-name { font-weight: 600; }
.meta-line--muted { color: var(--muted-foreground); }
.meta-tag { display: inline-flex; align-items: center; min-height: 20px; padding: 0 8px; border-radius: 999px; background: var(--muted); color: var(--muted-foreground); font-size: 0.72rem; font-variant-numeric: tabular-nums; }
.btn-sm { height: 32px; padding: 0 13px; font-size: 0.8rem; border-radius: 999px; display: inline-flex; align-items: center; gap: 6px; border: 1px solid transparent; cursor: pointer; font-weight: 500; transition: background 150ms ease, border-color 150ms ease, transform 120ms ease; }
.btn-sm:active { transform: translateY(1px); }
.btn-sm:focus-visible { outline: 2px solid var(--ring); outline-offset: 2px; }
.btn-sm.btn-ghost { background: var(--muted); color: var(--foreground); }
.btn-sm.btn-ghost:hover { background: color-mix(in srgb, var(--muted) 80%, var(--foreground) 14%); }
</style>
