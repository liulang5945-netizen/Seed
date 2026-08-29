<template>
  <SettingsPanelSection title="运行环境" class="settings-runtime-panel">

    <div class="setting-row">
      <div class="setting-left">
        <span class="setting-label">允许未认证终端访问</span>
        <p class="setting-desc">认证未启用时，允许工作台终端直接连接。开启会降低本地安全性，仅建议在受信任的本机环境使用。</p>
      </div>
      <div class="setting-right">
        <label class="toggle" aria-label="允许未认证终端访问开关">
          <input
            type="checkbox"
            :checked="modelValue"
            :disabled="saving"
            @change="onToggle"
          />
          <span class="track"><span class="thumb"></span></span>
        </label>
      </div>
    </div>

    <div class="setting-row setting-row--last">
      <div class="setting-left">
        <span class="setting-label">当前状态</span>
        <p class="setting-desc">{{ runtimeStatusText }}</p>
      </div>
    </div>
  </SettingsPanelSection>
</template>

<script setup>
import SettingsPanelSection from './SettingsPanelSection.vue'

defineProps({
  modelValue: { type: Boolean, default: false },
  saving: { type: Boolean, default: false },
  runtimeStatusText: { type: String, default: '检测中…' },
});

const emit = defineEmits(['update:modelValue', 'change']);

const onToggle = (event) => {
  emit('update:modelValue', event.target.checked);
  emit('change');
};
</script>

<style scoped>
.settings-section {
  display: flex;
  flex-direction: column;
}
.settings-section h2 {
  font-size: 1.15rem;
  font-weight: 650;
  margin: 0 0 18px;
  color: var(--foreground);
  letter-spacing: -0.01em;
}
.setting-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 0;
  border-bottom: 1px solid var(--border);
  gap: 16px;
}
.setting-row--last {
  border-bottom: none;
  padding-bottom: 0;
}
.setting-left {
  min-width: 0;
  flex: 1;
}
.setting-right {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}
.setting-label {
  font-size: 0.88rem;
  font-weight: 500;
  color: var(--foreground);
  line-height: 1.3;
}
.setting-desc {
  font-size: 0.76rem;
  color: var(--muted-foreground);
  margin: 3px 0 0;
  line-height: 1.4;
}
.toggle {
  position: relative;
  display: inline-block;
  width: 42px;
  height: 24px;
  flex-shrink: 0;
  cursor: pointer;
}
.toggle input {
  opacity: 0;
  width: 0;
  height: 0;
  position: absolute;
  border: 0;
  padding: 0;
}
.toggle .track {
  position: absolute;
  inset: 0;
  background: var(--border);
  border-radius: 999px;
  cursor: pointer;
  transition: background 200ms ease;
}
.toggle .thumb {
  position: absolute;
  top: 3px;
  left: 3px;
  width: 18px;
  height: 18px;
  background: var(--background);
  border-radius: 50%;
  transition: transform 200ms cubic-bezier(0.34, 1.56, 0.64, 1);
  box-shadow: 0 1px 3px color-mix(in srgb, var(--foreground) 16%, transparent);
}
.toggle input:checked + .track {
  background: var(--primary);
}
.toggle input:checked + .track .thumb {
  transform: translateX(18px);
}
.toggle input:focus-visible + .track {
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--ring) 20%, transparent);
}
@media (max-width: 500px) {
  .setting-row {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
  .setting-right {
    width: 100%;
    justify-content: flex-start;
  }
}
</style>
