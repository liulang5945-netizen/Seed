<template>
  <section class="settings-section">
    <h2>通用设置</h2>

    <div class="setting-row setting-row--col setting-row--first">
      <span class="setting-label">外观主题</span>
      <div class="theme-previews">
        <div
          v-for="theme in themes"
          :key="theme.id"
          class="theme-preview-card"
          :class="{ active: currentTheme === theme.id }"
          :title="theme.desc"
          @click="emit('theme-change', theme.id)"
        >
          <span class="theme-swatch" :style="{ background: theme.gradient }"></span>
          <span class="theme-name">{{ theme.name }}</span>
        </div>
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-left">
        <span class="setting-label">默认语言</span>
        <p class="setting-desc">界面与交互的显示语言</p>
      </div>
      <div class="setting-right">
        <select :value="uiLanguage" aria-label="默认语言" :disabled="saving" @change="onChange('ui-language-change', $event.target.value)">
          <option value="zh-CN">简体中文</option>
          <option value="zh-TW">繁體中文</option>
          <option value="en">English</option>
          <option value="ja">日本語</option>
          <option value="ko">한국어</option>
        </select>
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-left">
        <span class="setting-label">时区</span>
        <p class="setting-desc">用于定时任务、日志时间戳等</p>
      </div>
      <div class="setting-right">
        <select :value="timezone" aria-label="时区" :disabled="saving" @change="onChange('timezone-change', $event.target.value)">
          <option value="Asia/Shanghai">Asia/Shanghai (UTC+8)</option>
          <option value="Asia/Tokyo">Asia/Tokyo (UTC+9)</option>
          <option value="Asia/Seoul">Asia/Seoul (UTC+9)</option>
          <option value="Asia/Singapore">Asia/Singapore (UTC+8)</option>
          <option value="America/Los_Angeles">America/Los_Angeles (UTC-8)</option>
          <option value="America/New_York">America/New_York (UTC-5)</option>
          <option value="Europe/London">Europe/London (UTC+0)</option>
          <option value="Europe/Berlin">Europe/Berlin (UTC+1)</option>
        </select>
      </div>
    </div>

    <div class="setting-row setting-row--last">
      <div class="setting-left">
        <span class="setting-label">界面密度</span>
        <p class="setting-desc">调整元素间距与信息密度</p>
      </div>
      <div class="setting-right">
        <div class="radio-group" role="radiogroup" aria-label="界面密度">
          <label v-for="density in densities" :key="density.value" class="radio-chip">
            <input :checked="uiDensity === density.value" type="radio" name="density" :value="density.value" :disabled="saving" @change="onChange('ui-density-change', density.value)">
            <span class="rc-label">{{ density.label }}</span>
          </label>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
defineOptions({ name: 'SettingsGeneralPanel' })

defineProps({
  themes: { type: Array, default: () => [] },
  currentTheme: { type: String, default: '' },
  uiLanguage: { type: String, default: '' },
  timezone: { type: String, default: '' },
  uiDensity: { type: String, default: '' },
  saving: { type: Boolean, default: false },
})

const emit = defineEmits(['theme-change', 'ui-language-change', 'timezone-change', 'ui-density-change'])
const densities = [
  { value: 'compact', label: '紧凑' },
  { value: 'default', label: '默认' },
  { value: 'comfortable', label: '宽松' },
]

function onChange(eventName, value) {
  emit(eventName, value)
}
</script>

<style scoped>
.settings-section { display: flex; flex-direction: column; }
.settings-section h2 { font-size: 1.15rem; font-weight: 650; margin: 0 0 18px; color: var(--foreground); letter-spacing: -0.01em; }
.setting-row { display: flex; align-items: center; justify-content: space-between; padding: 14px 0; border-bottom: 1px solid var(--border); gap: 16px; }
.setting-row--first { padding-top: 0; }
.setting-row--last { border-bottom: none; padding-bottom: 0; }
.setting-row--col { flex-direction: column; align-items: stretch; gap: 12px; }
.setting-left { min-width: 0; flex: 1; }
.setting-right { flex-shrink: 0; display: flex; align-items: center; gap: 8px; }
.setting-label { font-size: 0.88rem; font-weight: 500; color: var(--foreground); line-height: 1.3; }
.setting-desc { font-size: 0.76rem; color: var(--muted-foreground); margin: 3px 0 0; line-height: 1.4; }
.setting-right select { background: var(--background); color: var(--foreground); border: 1px solid var(--border); border-radius: calc(var(--radius) * 0.45); padding: 7px 32px 7px 12px; font-size: 0.86rem; font-family: var(--font-sans); outline: none; transition: border-color 160ms ease, box-shadow 160ms ease; appearance: none; -webkit-appearance: none; color: var(--muted-foreground); background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%237f8d9f' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 10px center; background-size: 15px; cursor: pointer; }
.setting-right select:focus { border-color: var(--ring); box-shadow: 0 0 0 3px color-mix(in srgb, var(--ring) 16%, transparent); }
.radio-group { display: flex; gap: 4px; background: var(--muted); border-radius: calc(var(--radius) * 0.5); padding: 3px; }
.radio-chip { position: relative; cursor: pointer; }
.radio-chip input { position: absolute; opacity: 0; width: 0; height: 0; }
.radio-chip .rc-label { display: block; padding: 5px 13px; border-radius: calc(var(--radius) * 0.4); font-size: 0.8rem; font-weight: 450; color: var(--muted-foreground); transition: background 150ms ease, color 150ms ease, font-weight 150ms ease; user-select: none; white-space: nowrap; }
.radio-chip input:checked + .rc-label { background: var(--background); color: var(--foreground); font-weight: 600; box-shadow: 0 1px 2px color-mix(in srgb, var(--foreground) 8%, transparent); }
.radio-chip:hover .rc-label { color: var(--foreground); }
.theme-previews { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
.theme-preview-card { border: 2px solid var(--border); border-radius: 12px; padding: 8px 8px 10px; cursor: pointer; background: var(--background); text-align: center; transition: border-color 160ms ease, box-shadow 160ms ease, transform 140ms ease; }
.theme-preview-card:hover { border-color: color-mix(in srgb, var(--primary) 35%, var(--border)); transform: translateY(-1px); }
.theme-preview-card.active { border-color: var(--primary); box-shadow: 0 0 0 3px color-mix(in srgb, var(--ring) 18%, transparent); }
.theme-swatch { width: 100%; height: 36px; border-radius: 8px; display: block; }
.theme-name { font-size: 0.72rem; margin-top: 7px; color: var(--muted-foreground); font-weight: 450; transition: color 150ms ease; }
.theme-preview-card.active .theme-name { color: var(--foreground); font-weight: 600; }
@media (max-width: 640px) { .theme-previews { grid-template-columns: repeat(2, 1fr); } }
</style>
