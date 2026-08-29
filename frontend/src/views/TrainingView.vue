<template>
  <section class="dedicated-view training-view">
    <!-- 顶栏标题 -->
    <div class="view-header">
      <div class="header-left">
        <h2>微调训练</h2>
        <span class="header-sub">
          <span>Seed 原生 · 参数预算驱动</span>
        </span>
      </div>
      <div class="header-actions">
        <n-button v-if="trainState === 'running' || trainState === 'paused'" type="error" round @click="stopTraining(toast)">
          <template #icon><StopCircle :size="14" /></template>停止训练
        </n-button>
        <n-button v-if="trainState === 'running'" type="primary" round @click="saveCheckpoint">
          <template #icon><PackageIcon :size="14" /></template>保存检查点
        </n-button>
        <n-button v-if="trainState === 'idle' || trainState === 'completed'" type="primary" round @click="startTaijiTraining(toast)">
          <template #icon><Zap :size="14" /></template>开始Seed微调
        </n-button>
      </div>
    </div>

    <RuntimeEvidenceStrip context="training" />

    <div class="view-body">
      <!-- 标签页 -->
      <div class="tabs" role="tablist" aria-label="训练管理" @keydown="onTablistKeydown">
        <button id="tk-tab-overview" class="tab" :class="{ active: isActive('overview') }" data-tab-id="overview" role="tab" :aria-selected="isActive('overview')" :tabindex="isActive('overview') ? 0 : -1" aria-controls="tk-panel-overview" @click="selectTab('overview')">训练概览</button>
        <button id="tk-tab-hyperparams" class="tab" :class="{ active: isActive('hyperparams') }" data-tab-id="hyperparams" role="tab" :aria-selected="isActive('hyperparams')" :tabindex="isActive('hyperparams') ? 0 : -1" aria-controls="tk-panel-hyperparams" @click="selectTab('hyperparams')">超参数</button>
        <button id="tk-tab-dataset" class="tab" :class="{ active: isActive('dataset') }" data-tab-id="dataset" role="tab" :aria-selected="isActive('dataset')" :tabindex="isActive('dataset') ? 0 : -1" aria-controls="tk-panel-dataset" @click="selectTab('dataset')">数据集</button>
        <button id="tk-tab-logs" class="tab" :class="{ active: isActive('logs') }" data-tab-id="logs" role="tab" :aria-selected="isActive('logs')" :tabindex="isActive('logs') ? 0 : -1" aria-controls="tk-panel-logs" @click="selectTab('logs')">日志</button>
      </div>

      <!-- ═══ Tab 1 · 训练概览 ═══ -->
      <TrainingOverviewPanel
        :active="isActive('overview')"
        :train-state="trainState"
        :train-progress="trainProgress"
        :train-metrics="trainMetrics"
        :train-loss="trainLoss"
        :pending-checkpoints="pendingCheckpoints"
        :fmt-time="fmtTime"
        @resume="resumeFromCheckpoint(toast, $confirm)"
      />

      <!-- ═══ Tab 2 · 超参数 ═══ -->
      <section id="tk-panel-hyperparams" class="tab-panel" :class="{ active: isActive('hyperparams') }" role="tabpanel" aria-labelledby="tk-tab-hyperparams">
        <div class="tk-card">
          <div class="card-head">
            <h3><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" style="width:17px;height:17px;color:var(--primary)"><circle cx="6" cy="7" r="2.2"/><circle cx="6" cy="17" r="2.2"/><circle cx="18" cy="12" r="2.2"/><path d="M8 7.5 12 5M8 16.5 12 19M8 16l7-4M8 8l7 3"/></svg>训练超参数</h3>
            <span class="card-sub">Seed 原生 · 参数预算驱动</span>
          </div>
          <div class="hp-grid">
            <div class="hp-field">
              <label class="hp-label">参数预算</label>
              <n-input-number v-model:value="taijiTrainParams.parameter_budget" :min="93367" :step="10000" />
              <span class="hp-desc">由预算自动规划区域、突触和情景记忆容量</span>
            </div>
            <div class="hp-field">
              <label class="hp-label">最大训练字节数</label>
              <n-input-number v-model:value="taijiTrainParams.max_symbols" :min="1" :step="10000" />
              <span class="hp-desc">限制本次 raw-byte 在线学习的输入规模</span>
            </div>
            <div class="hp-field">
              <label class="hp-label">训练设备</label>
              <n-select v-model:value="taijiTrainParams.device" :options="deviceOptions" />
              <span class="hp-desc">auto 会在可用时选择 CUDA，否则使用 CPU</span>
            </div>
            <div class="hp-field">
              <label class="hp-label">随机种子 (seed)</label>
              <n-input-number v-model:value="taijiTrainParams.seed" :min="0" :step="1" />
              <span class="hp-desc">控制权重初始化与数据打乱的可复现性</span>
            </div>
          </div>
          <div class="hp-actions">
            <n-button type="primary" round @click="applyAndRestart">应用并重启训练</n-button>
            <n-button round @click="resetDefaults">恢复默认值</n-button>
            <span style="flex:1"></span>
            <span class="card-sub" style="align-self:center">检查点按固定周期自动落盘</span>
          </div>
        </div>
      </section>

      <!-- ═══ Tab 3 · 数据集 ═══ -->
      <TrainingDatasetPanel
        :active="isActive('dataset')"
        :train-files="trainFiles"
        :file-sizes="trainFileSizes"
        :selected-datasets="selectedDatasets"
        :train-preview="trainPreview"
        :all-selected="isAllSelected()"
        :t="t"
        @refresh="loadTrainDatasets"
        @toggle-select-all="toggleSelectAll"
        @delete-selected="deleteSelectedDatasets(toast)"
        @toggle-dataset="toggleDataset"
        @preview="previewDataset"
        @delete="deleteTrainFile"
      />

      <!-- ═══ Tab 4 · 日志 ═══ -->
      <TrainingLogPanel
        :active="isActive('logs')"
        :train-log="trainLog"
        :model-label="taijiModelInfo.size || 'Seed模型'"
        @clear="clearTrainLog"
      />

      <!-- 训练控制（悬浮底栏样式，便于随时操作） -->
      <TrainingControlBar
        :train-state="trainState"
        :t="t"
        @pause="pauseTraining(toast)"
        @resume="resumeTraining(toast)"
        @stop="stopTraining(toast)"
      />

      <!-- 硬件诊断 -->
      <n-alert v-if="trainDevice.message" type="info" class="hw-alert" style="margin-top:18px">
        <template #icon><Monitor :size="16" /></template>
        {{ trainDevice.message }}
      </n-alert>
    </div>
  </section>
</template>

<script setup>
import { Monitor, Zap, Package as PackageIcon, StopCircle } from 'lucide-vue-next';

import { inject, onActivated } from 'vue';
import TrainingOverviewPanel from '../components/TrainingOverviewPanel.vue';
import TrainingDatasetPanel from '../components/TrainingDatasetPanel.vue';
import TrainingLogPanel from '../components/TrainingLogPanel.vue';
import TrainingControlBar from '../components/TrainingControlBar.vue';
import { useApi } from '../composables/useApi.js';
import { useTabs } from '../composables/useTabs.js';
import RuntimeEvidenceStrip from '../components/RuntimeEvidenceStrip.vue';
import {
  trainState, trainLog, trainLoss, trainFiles, trainFileSizes,
  selectedDatasets, trainPreview,
  trainProgress,
  pendingCheckpoints, trainMetrics, trainDevice,
  fmtTime,
  clearTrainLog,
  loadTrainDatasets, previewDataset, deleteTrainFile, deleteSelectedDatasets,
  toggleSelectAll, toggleDataset, isAllSelected,
  pauseTraining, resumeTraining, stopTraining,
  loadCheckpoints, resumeFromCheckpoint,
  taijiModelInfo, taijiTrainParams,
  startTaijiTraining, detectTaijiModel,
} from '../composables/useTraining.js';

defineOptions({ name: 'TrainingView' });

const toast = inject('toast');
const $confirm = inject('$confirm');
const { t } = useApi();

// 标签页状态收敛到 useTabs：DOM 常驻、切换 0ms、状态同步到 ?tab=
const { isActive, selectTab, onTablistKeydown } = useTabs(['overview', 'hyperparams', 'dataset', 'logs']);

// 训练设备选项（与后端 /api/train/native 的 device 字段对齐）
const deviceOptions = [
  { label: 'auto（自动选择）', value: 'auto' },
  { label: 'cpu', value: 'cpu' },
  { label: 'cuda', value: 'cuda' },
];

// 保存检查点（占位，实际由后端处理）
const saveCheckpoint = () => {
  toast?.success?.('检查点保存请求已发送');
};

// 应用并重启训练
const applyAndRestart = () => {
  if (trainState.value === 'running' || trainState.value === 'paused') {
    stopTraining(toast);
    setTimeout(() => startTaijiTraining(toast), 800);
  } else {
    startTaijiTraining(toast);
  }
};

// 恢复默认值（与后端 /api/train/native 实际参数对齐）
const resetDefaults = () => {
  Object.assign(taijiTrainParams, {
    parameter_budget: 300000, max_symbols: 200000, device: 'auto', seed: 20260822,
  });
  toast?.success?.('已恢复默认超参数');
};

// keep-alive 缓存后 setup 顶层代码不会再次执行；
// onActivated 在首次挂载与每次重新激活时都会触发，统一由它刷新。
onActivated(() => {
  detectTaijiModel();
  loadTrainDatasets();
  loadCheckpoints();
});
</script>

<style scoped>
/* ═══ 训练视图 · 画布设计语言对齐 ═══ */
.training-view {
  --chart-1: var(--chart-1, #557fff);
  --chart-2: var(--chart-2, #0065fd);
  --chart-3: var(--chart-3, #0057da);
  --chart-4: var(--chart-4, #0043ad);
  --chart-5: var(--chart-5, #002e7d);
  --font-mono: "JetBrains Mono", "Fira Code", "Consolas", "Courier New", monospace;
  --destructive: var(--danger, #ef4444);
  --warning: var(--warning, #eab308);
}

/* ===== 顶栏标题 ===== */
.view-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 18px;
}
.header-left { display: flex; flex-direction: column; gap: 3px; }
.header-left h2 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--foreground, var(--text));
  letter-spacing: -0.01em;
}
.header-sub {
  font-size: 0.78rem;
  color: var(--muted-foreground, var(--text-muted));
}
.header-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: none;
}

/* ===== 标签页（匹配画布） ===== */
/* 不画 border-bottom：全应用唯一外围边框归 .router-wrapper（见 styles/shell.css） */
.tabs {
  display: flex;
  gap: 2px;
  margin-bottom: 22px;
}
.tab {
  appearance: none;
  border: 0;
  background: transparent;
  color: var(--muted-foreground, var(--text-muted));
  padding: 11px 18px;
  font: 600 0.88rem/1 var(--font-sans, inherit);
  cursor: pointer;
  position: relative;
  transition: color 0.15s;
}
.tab:hover { color: var(--foreground, var(--text)); }
.tab:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
  border-radius: var(--radius-sm, 6px);
}
.tab.active { color: var(--primary); }
.tab.active::after {
  content: "";
  position: absolute;
  left: 14px;
  right: 14px;
  bottom: 0;
  height: 2px;
  background: var(--primary);
  border-radius: 2px;
}

/* ===== Tab Panel ===== */
/* 常驻 DOM + 零动画：切换是 0ms 的显隐，不是重建后淡入 */
.tab-panel { display: none; }
.tab-panel.active { display: block; }

/* ===== 卡片基础（匹配画布 tk-card） ===== */
.tk-card {
  background: var(--card, var(--bg-card));
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) * 0.7);
  padding: 22px;
  transition: border-color 0.16s ease;
}
.tk-card:hover { border-color: color-mix(in srgb, var(--primary) 20%, var(--border)); }
.tk-card .card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}
.tk-card h3 {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--foreground, var(--text));
  display: flex;
  align-items: center;
  gap: 8px;
}
.tk-card .card-sub {
  color: var(--muted-foreground, var(--text-muted));
  font-size: 0.78rem;
}

/* ===== 超参数网格（Seed 原生 4 字段） ===== */
.hp-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px 32px;
}
.hp-field {
  display: flex;
  flex-direction: column;
  gap: 7px;
}
.hp-field .hp-label {
  font-size: 0.84rem;
  font-weight: 600;
  color: var(--foreground, var(--text));
  display: flex;
  align-items: center;
  gap: 7px;
}
.hp-field .hp-label .ic {
  width: 15px;
  height: 15px;
  color: var(--muted-foreground, var(--text-muted));
}
.hp-field .hp-desc {
  font-size: 0.74rem;
  color: var(--muted-foreground, var(--text-muted));
  line-height: 1.4;
}

/* Naive UI 输入框适配 */
.hp-field :deep(.n-input-number) {
  width: 100%;
}
.hp-field :deep(.n-input) {
  border-radius: calc(var(--radius) * 0.42);
}
.hp-field :deep(.n-input .n-input__border),
.hp-field :deep(.n-input .n-input__state-border) {
  border-radius: calc(var(--radius) * 0.42);
}
.hp-field :deep(.n-input .n-input__input),
.hp-field :deep(.n-input .n-input__textarea-el) {
  font-family: var(--font-mono);
  font-size: 0.88rem;
}

/* 超参数操作栏 */
.hp-actions {
  display: flex;
  gap: 10px;
  margin-top: 26px;
  padding-top: 20px;
  border-top: 1px solid var(--border);
  align-items: center;
}

/* ===== 硬件诊断 ===== */
.hw-alert {
  border-radius: calc(var(--radius) * 0.6);
}

/* ===== 响应式（匹配画布断点） ===== */
@media (max-width: 1180px) {
  .hp-grid {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 720px) {
  .view-header {
    flex-direction: column;
    align-items: flex-start;
  }
  .header-actions {
    width: 100%;
    justify-content: flex-end;
  }
}

</style>
