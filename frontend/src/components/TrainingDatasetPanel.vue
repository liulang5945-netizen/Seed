<template>
  <section
    id="tk-panel-dataset"
    class="tab-panel"
    :class="{ active }"
    role="tabpanel"
    aria-labelledby="tk-tab-dataset"
  >
    <div class="tk-card upload-card">
      <div class="card-head">
        <h3><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M5 12h14"/></svg>数据上传</h3>
      </div>
      <FileUploadQueue
        native-dataset-upload
        :accept="accept"
        :icon="BarChart2"
        title="训练数据上传"
        :upload-icon="Download"
        :drop-text="t('train_upload')"
        :accept-hint="t('train_support')"
        success-text="数据集上传成功"
        @all-uploaded="emit('refresh')"
      />
    </div>

    <div class="ds-wrap">
      <div class="dataset-toolbar">
        <h3>训练数据集</h3>
        <span class="card-sub">· {{ trainFiles.length }} 个文件</span>
        <span class="toolbar-spacer"></span>
        <n-checkbox v-if="trainFiles.length" :checked="allSelected" size="small" @update:checked="emit('toggle-select-all')">全选</n-checkbox>
        <n-button v-if="selectedDatasets.length > 0" size="small" type="error" round @click="emit('delete-selected')">
          <template #icon><Trash2 :size="14" /></template>删除选中 ({{ selectedDatasets.length }})
        </n-button>
        <n-button size="small" round @click="emit('refresh')">
          <template #icon><RefreshCw :size="14" /></template>刷新
        </n-button>
      </div>
      <table v-if="trainFiles.length" class="ds-table">
        <thead>
          <tr>
            <th style="width:40px"></th>
            <th>数据集名称</th>
            <th style="width:120px">大小</th>
            <th style="width:120px">状态</th>
            <th style="width:160px">操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="filename in trainFiles" :key="filename" :class="{ selected: selectedDatasets.includes(filename) }">
            <td><n-checkbox :checked="selectedDatasets.includes(filename)" @update:checked="emit('toggle-dataset', filename)" /></td>
            <td>
              <span class="ds-name">
                <span class="ds-ic"><PackageIcon :size="14" /></span>
                {{ filename }}
              </span>
            </td>
            <td><span class="ds-num">{{ formatSize(fileSizes[filename]) }}</span></td>
            <td><span class="sc sc-ok">已就绪</span></td>
            <td>
              <div class="ds-act">
                <button @click="emit('preview', filename)"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>查看</button>
                <button class="danger" @click="emit('delete', filename)"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 7h14M9 7V5h6v2M7 7l1 12h8l1-12"/></svg>移除</button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
      <n-empty v-else :description="t('train_no_data')" style="padding:40px 0" />

      <div v-if="trainPreview" class="ds-preview">
        <div class="ds-preview-head">
          {{ t('dataset_preview') }} ({{ trainPreview.count || 0 }} {{ t('samples') }})
          <span v-if="trainPreview.report && trainPreview.report.truncated" class="preview-note">· 已采样前 {{ trainPreview.count }} 条</span>
          <span v-if="trainPreview.native_trainable === false" class="preview-warn">· 不符合原生 text 合同</span>
        </div>
        <div v-for="(sample, index) in (trainPreview.samples || [])" :key="index" class="preview-sample">
          <div class="preview-label">{{ t('document') }} #{{ index + 1 }}</div>
          <div class="preview-text">{{ sample.text }}</div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { BarChart2, Download, Package as PackageIcon, RefreshCw, Trash2 } from 'lucide-vue-next'
import FileUploadQueue from './FileUploadQueue.vue'

defineProps({
  active: { type: Boolean, default: false },
  trainFiles: { type: Array, default: () => [] },
  fileSizes: { type: Object, default: () => ({}) },
  selectedDatasets: { type: Array, default: () => [] },
  trainPreview: { type: Object, default: null },
  allSelected: { type: Boolean, default: false },
  t: { type: Function, required: true },
  accept: {
    type: String,
    default: '.jsonl,.json,.txt,.csv,.md,.pdf,.docx,.doc,.xlsx,.xls,.pptx,.html,.htm,.epub,.rtf,.xml,.log,.py,.js,.ts,.css,.java,.c,.cpp,.sh,.sql,.png,.jpg,.jpeg,.bmp,.gif,.webp,.tiff,.tif',
  },
})

const emit = defineEmits([
  'refresh',
  'toggle-select-all',
  'delete-selected',
  'toggle-dataset',
  'preview',
  'delete',
])

function formatSize(bytes) {
  if (bytes == null || !Number.isFinite(bytes)) return '--'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = bytes
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${unit === 0 ? value : value.toFixed(1)} ${units[unit]}`
}
</script>

<style scoped>
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.tk-card {
  background: var(--card, var(--bg-card));
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) * 0.7);
  padding: 22px;
}
.upload-card { margin-bottom: 18px; }
.tk-card .card-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
.tk-card h3 { margin: 0; font-size: 1rem; font-weight: 600; color: var(--foreground, var(--text)); display: flex; align-items: center; gap: 8px; }
.tk-card .card-sub, .card-sub { color: var(--muted-foreground, var(--text-muted)); font-size: 0.78rem; }
.ic { width: 17px; height: 17px; color: var(--primary); }
.ds-wrap { background: var(--card, var(--bg-card)); border: 1px solid var(--border); border-radius: calc(var(--radius) * 0.7); overflow: hidden; }
.dataset-toolbar { display: flex; align-items: center; gap: 10px; padding: 14px 18px; border-bottom: 1px solid var(--border); }
.dataset-toolbar h3 { margin: 0; font-size: 1rem; font-weight: 600; color: var(--foreground, var(--text)); }
.toolbar-spacer { flex: 1; }
.ds-table { width: 100%; border-collapse: collapse; }
.ds-table th { text-align: left; font: 600 0.72rem/1 var(--font-mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted-foreground, var(--text-muted)); padding: 12px 18px; border-bottom: 1px solid var(--border); background: color-mix(in srgb, var(--muted) 35%, transparent); }
.ds-table td { padding: 14px 18px; border-bottom: 1px solid color-mix(in srgb, var(--border) 55%, transparent); font-size: 0.86rem; color: var(--foreground, var(--text)); }
.ds-table tr:last-child td { border-bottom: 0; }
.ds-table tbody tr { transition: background 0.12s; }
.ds-table tbody tr:hover { background: color-mix(in srgb, var(--accent, var(--primary-light)) 14%, transparent); }
.ds-table tbody tr.selected { background: color-mix(in srgb, var(--primary) 8%, transparent); }
.ds-name { display: flex; align-items: center; gap: 11px; font-weight: 500; }
.ds-ic { width: 30px; height: 30px; border-radius: 8px; display: grid; place-items: center; color: var(--primary-foreground, #fff); background: linear-gradient(135deg, var(--chart-1), var(--chart-2)); flex: none; font-size: 0.9rem; }
.ds-ic :deep(svg) { width: 15px; height: 15px; }
.ds-num { font-variant-numeric: tabular-nums; font-weight: 600; color: var(--foreground, var(--text)); }
.sc { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; font: 500 0.74rem var(--font-sans, inherit); }
.sc::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.sc-ok { background: color-mix(in srgb, var(--chart-2) 16%, transparent); color: var(--chart-2); }
.ds-act { display: inline-flex; align-items: center; gap: 4px; }
.ds-act button { border: 0; background: transparent; color: var(--muted-foreground, var(--text-muted)); cursor: pointer; padding: 5px 9px; border-radius: 7px; font-size: 0.78rem; display: inline-flex; align-items: center; gap: 5px; transition: background 0.14s ease, color 0.14s ease; }
.ds-act button .ic { width: 14px; height: 14px; color: currentColor; }
.ds-act button:hover { background: var(--muted, var(--bg-muted)); color: var(--foreground, var(--text)); }
.ds-act button.danger:hover { color: var(--destructive, var(--danger)); }
.ds-preview { padding: 18px; border-top: 1px solid var(--border); background: color-mix(in srgb, var(--muted) 25%, transparent); }
.ds-preview-head { font-size: 0.84rem; font-weight: 600; color: var(--foreground, var(--text)); margin-bottom: 10px; }
.preview-note { font-weight: 500; color: var(--muted-foreground, var(--text-secondary)); }
.preview-warn { font-weight: 600; color: var(--destructive, var(--danger)); }
.preview-sample { padding: 12px 14px; background: var(--card, var(--bg-card)); border: 1px solid var(--border); border-radius: calc(var(--radius) * 0.5); margin-bottom: 8px; }
.preview-label { font-size: 0.74rem; color: var(--primary); font-weight: 600; margin-bottom: 4px; }
.preview-text { font-size: 0.84rem; color: var(--muted-foreground, var(--text-secondary)); word-break: break-all; line-height: 1.6; margin-bottom: 8px; }
.preview-text:last-child { margin-bottom: 0; }
@media (max-width: 720px) {
  .dataset-toolbar { flex-wrap: wrap; }
  .toolbar-spacer { display: none; }
  .ds-table { display: block; overflow-x: auto; }
}
</style>
