<template>
  <section class="settings-section">
    <h2>数据与隐私</h2>

    <div class="setting-row setting-row--first">
      <div class="setting-left">
        <span class="setting-label">对话保留</span>
        <p class="setting-desc">历史对话的自动保留时长</p>
      </div>
      <div class="setting-right">
        <select :value="chatRetentionDays" aria-label="对话保留" :disabled="saving" @change="emit('retention-change', $event.target.value)">
          <option value="30">30 天</option>
          <option value="90">90 天</option>
          <option value="180">180 天</option>
          <option value="365">365 天</option>
          <option value="forever">永久保留</option>
        </select>
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-left">
        <span class="setting-label">自动清理</span>
        <p class="setting-desc">过期对话与缓存文件到达保留期限后自动移除</p>
      </div>
      <div class="setting-right">
        <label class="toggle" aria-label="自动清理开关">
          <input
            :checked="chatAutoCleanup"
            type="checkbox"
            :disabled="saving"
            @change="emit('auto-cleanup-change', $event.target.checked)"
          />
          <span class="track"><span class="thumb"></span></span>
        </label>
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-left">
        <span class="setting-label">导出数据</span>
        <p class="setting-desc">导出所有对话记录、配置快照与 Taiji 状态</p>
      </div>
      <div class="setting-right">
        <button class="btn-sm btn-outline" :disabled="exporting" @click="emit('export-data')">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/></svg>
          {{ exporting ? '导出中…' : '导出' }}
        </button>
      </div>
    </div>

    <div class="setting-row setting-row--last">
      <div class="danger-zone">
        <h3>
          <svg class="dz-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z"/><path d="M12 9v4M12 17h.01"/></svg>
          危险操作
        </h3>
        <p>重置将清空所有本地对话会话记录；不会删除模型权重、检查点、Taiji 状态与配置项。此操作不可撤销，建议先导出数据再进行重置。</p>
        <button class="btn-destructive" :disabled="resetting" @click="emit('reset-seed')">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M9.5 9.5h5v6h-5z"/><path d="M14.5 9.5 13 7h-2l-1.5 2.5"/></svg>
          {{ resetting ? '重置中…' : '重置Seed' }}
        </button>
      </div>
    </div>
  </section>
</template>

<script setup>
defineOptions({ name: 'SettingsPrivacyPanel' })

defineProps({
  chatRetentionDays: { type: String, default: '90' },
  chatAutoCleanup: { type: Boolean, default: true },
  saving: { type: Boolean, default: false },
  exporting: { type: Boolean, default: false },
  resetting: { type: Boolean, default: false },
})

const emit = defineEmits(['retention-change', 'auto-cleanup-change', 'export-data', 'reset-seed'])
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
.setting-right select { background: var(--background); color: var(--foreground); border: 1px solid var(--border); border-radius: calc(var(--radius) * 0.45); padding: 7px 32px 7px 12px; font-size: 0.86rem; font-family: var(--font-sans); outline: none; transition: border-color 160ms ease, box-shadow 160ms ease; appearance: none; -webkit-appearance: none; color: var(--muted-foreground); background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%237f8d9f' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 10px center; background-size: 15px; cursor: pointer; }
.setting-right select:focus { border-color: var(--ring); box-shadow: 0 0 0 3px color-mix(in srgb, var(--ring) 16%, transparent); }
.toggle { position: relative; display: inline-block; width: 42px; height: 24px; flex-shrink: 0; cursor: pointer; }
.toggle input { opacity: 0; width: 0; height: 0; position: absolute; border: 0; padding: 0; }
.toggle .track { position: absolute; inset: 0; background: var(--border); border-radius: 999px; cursor: pointer; transition: background 200ms ease; }
.toggle .thumb { position: absolute; top: 3px; left: 3px; width: 18px; height: 18px; background: var(--background); border-radius: 50%; transition: transform 200ms cubic-bezier(0.34, 1.56, 0.64, 1); box-shadow: 0 1px 3px color-mix(in srgb, var(--foreground) 16%, transparent); }
.toggle input:checked + .track { background: var(--primary); }
.toggle input:checked + .track .thumb { transform: translateX(18px); }
.toggle input:focus-visible + .track { box-shadow: 0 0 0 3px color-mix(in srgb, var(--ring) 20%, transparent); }
.btn-sm { height: 32px; padding: 0 13px; font-size: 0.8rem; border-radius: 999px; display: inline-flex; align-items: center; gap: 6px; border: 1px solid transparent; cursor: pointer; font-weight: 500; transition: background 150ms ease, border-color 150ms ease, transform 120ms ease; }
.btn-sm:active { transform: translateY(1px); }
.btn-sm:focus-visible { outline: 2px solid var(--ring); outline-offset: 2px; }
.btn-sm.btn-outline { background: var(--background); color: var(--foreground); border-color: var(--border); }
.btn-sm.btn-outline:hover { background: var(--muted); }
.danger-zone { width: 100%; border: 1px solid var(--destructive); border-radius: 13px; padding: 20px; margin-top: 8px; background: color-mix(in srgb, var(--destructive) 4%, transparent); }
.danger-zone h3 { color: var(--destructive); margin: 0 0 6px; font-size: 0.94rem; font-weight: 650; display: flex; align-items: center; gap: 7px; }
.danger-zone .dz-icon { width: 18px; height: 18px; flex: none; }
.danger-zone p { font-size: 0.78rem; color: var(--muted-foreground); margin: 0 0 14px; line-height: 1.5; }
.btn-destructive { display: inline-flex; align-items: center; gap: 7px; height: 36px; padding: 0 16px; border: 1px solid transparent; border-radius: 999px; background: var(--destructive); color: var(--destructive-foreground); font-size: 0.85rem; font-weight: 500; cursor: pointer; transition: background 150ms ease, transform 120ms ease; }
.btn-destructive:hover { background: color-mix(in srgb, var(--destructive) 88%, var(--foreground) 12%); }
.btn-destructive:active { transform: translateY(1px); }
.btn-destructive:focus-visible { outline: 2px solid var(--ring); outline-offset: 2px; }
</style>
