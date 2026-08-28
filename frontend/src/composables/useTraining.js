
/**
 * 训练相关状态和逻辑
 * 从 App.vue 拆出，减轻主文件臃肿
 *
 * 仅支持 Seed 原生模式 — raw-byte Taiji 训练，走 /api/train/native。
 * 旧训练流后端已不存在，相关死路径已移除。
 */
import { ref, reactive, nextTick } from 'vue';
import { API_BASE, authFetch } from './apiClient.js';

// ===== 导出状态 =====

export const trainState = ref('idle');            // idle | running | paused | completed
export const trainLog = ref('');
export const trainLoss = ref([]);
export const trainFiles = ref([]);
export const selectedDatasets = ref([]);
export const trainPreview = ref(null);
export let trainAbortController = null;
export let trainReader = null;
export const trainProgress = ref(0);
export const pendingCheckpoints = ref([]);

export const trainMetrics = reactive({
  elapsed: 0, eta: null, lr: null, epoch: 0, total_epochs: 0,
  grad_norm: null, samples_per_sec: 0, total_steps: 0, current_loss: null,
});

export const trainDevice = reactive({
  device_type: '', device_name: '', gpu_name: null,
  gpu_memory_gb: null, ram_gb: null, message: '',
});

export const lossCanvasRef = ref(null);
export const trainLogRef = ref(null);

// ===== Seed模式状态 =====

export const isTaijiModel = ref(false);           // 当前是否为Seed模型
export const taijiModelInfo = reactive({           // Seed模型信息
  size: '', parameters: {}, config: {},
  available_sizes: [], checkpoints: {},
});
export const taijiTrainParams = reactive({         // Seed专属训练参数（与 /api/train/native 请求体对齐）
  parameter_budget: 300000, max_symbols: 200000, device: 'auto', seed: 20260822,
});

// ===== 辅助函数 =====

export const fmtTime = (s) => {
  if (s == null || !isFinite(s)) return '--';
  if (s < 0) s = 0;
  if (s < 60) return `${Math.round(s)}秒`;
  if (s < 3600) return `${Math.floor(s / 60)}分${Math.round(s % 60)}秒`;
  return `${Math.floor(s / 3600)}时${Math.floor((s % 3600) / 60)}分`;
};

export const autoScrollTrainLog = () => {
  nextTick(() => {
    if (trainLogRef.value) {
      trainLogRef.value.scrollTop = trainLogRef.value.scrollHeight;
    }
  });
};

export const clearTrainLog = () => { trainLog.value = ''; };

// ===== 数据集管理 =====

export const isAllSelected = () => {
  return trainFiles.value.length > 0 && selectedDatasets.value.length === trainFiles.value.length;
};

export function toggleSelectAll() {
  if (isAllSelected()) {
    selectedDatasets.value = [];
  } else {
    selectedDatasets.value = [...trainFiles.value];
  }
}

export function toggleDataset(filename) {
  const idx = selectedDatasets.value.indexOf(filename);
  if (idx >= 0) selectedDatasets.value.splice(idx, 1);
  else selectedDatasets.value.push(filename);
}

export async function loadTrainDatasets() {
  try {
    const res = await authFetch(`${API_BASE}/api/train/files`);
    if (res.ok) {
      const data = await res.json();
      trainFiles.value = data.files || [];
    }
  } catch (e) { /* silent */ }
}

export async function previewDataset(filename) {
  try {
    const res = await authFetch(`${API_BASE}/api/train/preview/${encodeURIComponent(filename)}`);
    if (res.ok) trainPreview.value = await res.json();
  } catch (e) { /* console.warn(e) */ }
}

export async function deleteTrainFile(filename) {
  try {
    await authFetch(`${API_BASE}/api/train/file/${encodeURIComponent(filename)}`, { method: 'DELETE' });
    const idx = selectedDatasets.value.indexOf(filename);
    if (idx >= 0) selectedDatasets.value.splice(idx, 1);
    loadTrainDatasets();
  } catch (e) { /* silent */ }
}

export async function deleteSelectedDatasets(_toast) {
  if (selectedDatasets.value.length === 0) return;
  if (!_toast) return;
  let successCount = 0;
  for (const filename of [...selectedDatasets.value]) {
    try {
      await authFetch(`${API_BASE}/api/train/file/${encodeURIComponent(filename)}`, { method: 'DELETE' });
      const idx = selectedDatasets.value.indexOf(filename);
      if (idx >= 0) selectedDatasets.value.splice(idx, 1);
      successCount++;
    } catch (e) {
      if (_toast) _toast(`❌ 删除 ${filename} 失败: ${e.message}`, 'error');
    }
  }
  await loadTrainDatasets();
  if (_toast) _toast(`✅ 已删除 ${successCount} 个文件`, 'success');
}

// ===== 训练流程 =====

export async function pauseTraining(toast) {
  try {
    await authFetch(`${API_BASE}/api/train/pause`, { method: 'POST' });
    trainState.value = 'paused';
    toast('⏸ 训练已暂停', 'info');
  } catch (e) { toast(`❌ 暂停失败: ${e.message}`, 'error'); }
}

export async function resumeTraining(toast) {
  try {
    await authFetch(`${API_BASE}/api/train/resume`, { method: 'POST' });
    trainState.value = 'running';
    toast('▶ 训练已恢复', 'info');
  } catch (e) { toast(`❌ 恢复失败: ${e.message}`, 'error'); }
}

export async function stopTraining(toast) {
  if (trainAbortController) {
    try { trainAbortController.abort(); } catch (e) { }
    trainAbortController = null;
  }
  if (trainReader) {
    try { trainReader.cancel(); } catch (e) { }
    trainReader = null;
  }
  try { await authFetch(`${API_BASE}/api/train/stop`, { method: 'POST' }); } catch (e) { }
  trainState.value = 'idle';
  trainProgress.value = 0;
  toast('⏹ 训练已停止', 'info');
}

// ===== 检查点 =====

export async function loadCheckpoints() {
  try {
    const res = await authFetch(`${API_BASE}/api/train/checkpoints`);
    if (res.ok) {
      const data = await res.json();
      pendingCheckpoints.value = data.checkpoints || [];
    }
  } catch (e) { /* silent */ }
}

export async function resumeFromCheckpoint(toast, $confirm) {
  if (pendingCheckpoints.value.length === 0) { toast('⚠ 没有找到检查点', 'warning'); return; }

  const latestCkpt = pendingCheckpoints.value[0];
  const datasetMsg = selectedDatasets.value.length > 0
    ? `所选数据集: ${selectedDatasets.value.join(', ')}`
    : '将使用检查点中保存的数据集路径（如文件已被删除，请先上传并选择数据集）';
  const ok = await $confirm({
    title: '🔄 恢复训练',
    message: `将从检查点 "${latestCkpt.filename}" (Epoch ${latestCkpt.epoch}, Step ${latestCkpt.step}, Loss=${latestCkpt.loss?.toFixed(4) || '?'}) 继续训练\n\n${datasetMsg}`,
    confirmText: '确认恢复',
  });
  if (!ok) return;

  trainState.value = 'running';
  trainLog.value = '';
  trainLoss.value = [];
  trainProgress.value = 0;
  trainAbortController = new AbortController();
  Object.assign(trainMetrics, { elapsed: 0, eta: null, lr: null, epoch: latestCkpt.epoch, total_epochs: latestCkpt.num_epochs || 1, grad_norm: null, samples_per_sec: 0, total_steps: 0, current_loss: null });
  Object.assign(trainDevice, { device_type: '', device_name: '', gpu_name: null, gpu_memory_gb: null, ram_gb: null, message: '' });

  const body = { checkpoint: latestCkpt.filename };
  if (selectedDatasets.value.length > 0) {
    body.datasets = [...selectedDatasets.value];
  }

  try {
    const res = await authFetch(`${API_BASE}/api/train/resume_checkpoint`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: trainAbortController.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    trainReader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let resumeCompleted = false;

    while (true) {
      const { done, value } = await trainReader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const payload = line.slice(6);
          if (payload === '[DONE]') {
            if (!resumeCompleted && trainState.value === 'running') trainState.value = 'completed';
            break;
          }
          try {
            const evt = JSON.parse(payload);
            if (evt.type === 'progress') {
              resumeCompleted = true;
              trainProgress.value = Math.round((evt.fraction || 0) * 100);
              if (evt.memory_status) trainLog.value += `🧠 ${evt.memory_status}\n`;
              else trainLog.value += `${evt.desc}\n`;
              if (evt.loss != null) trainLoss.value.push({ step: evt.step || 0, loss: evt.loss });
              if (evt.elapsed != null) trainMetrics.elapsed = evt.elapsed;
              if (evt.eta != null) trainMetrics.eta = evt.eta;
              if (evt.lr != null) trainMetrics.lr = evt.lr;
              if (evt.epoch != null) trainMetrics.epoch = evt.epoch;
              if (evt.total_epochs != null) trainMetrics.total_epochs = evt.total_epochs;
              if (evt.grad_norm != null) trainMetrics.grad_norm = evt.grad_norm;
              if (evt.samples_per_sec != null) trainMetrics.samples_per_sec = evt.samples_per_sec;
              if (evt.total_steps != null) trainMetrics.total_steps = evt.total_steps;
              if (evt.loss != null) trainMetrics.current_loss = evt.loss;
              autoScrollTrainLog();
            } else if (evt.type === 'hardware_diag') {
              trainDevice.device_type = evt.device_type || '';
              trainDevice.device_name = evt.device_name || '';
              trainDevice.gpu_name = evt.gpu_name || null;
              trainDevice.gpu_memory_gb = evt.gpu_memory_gb || null;
              trainDevice.ram_gb = evt.ram_gb || null;
              trainDevice.message = evt.message || '';
              trainLog.value += `${evt.message}\n`;
              autoScrollTrainLog();
            } else if (evt.type === 'warning') {
              trainLog.value += `⚠️ ${evt.message}\n`;
              autoScrollTrainLog();
            } else if (evt.type === 'error') {
              trainLog.value += `❌ ${evt.message}\n`;
              trainState.value = 'idle';
              autoScrollTrainLog();
              toast(`❌ 恢复训练失败: ${evt.message}`, 'error');
            } else if (evt.type === 'completed') {
              resumeCompleted = true;
              trainLog.value += `✅ ${evt.message}\n`;
              trainState.value = 'completed';
              trainProgress.value = 100;
              autoScrollTrainLog();
            } else if (evt.type === 'stopped') {
              trainLog.value += `⏹ ${evt.message}\n`;
              trainState.value = 'idle';
              autoScrollTrainLog();
            }
          } catch (e) { console.debug('[useTraining] parse error:', e.message) } /* skip */ }
        }
      }
  } catch (err) {
    if (err.name !== 'AbortError') {
      trainLog.value += `❌ ${err.message}\n`;
      trainState.value = 'idle';
      autoScrollTrainLog();
      toast(`❌ ${err.message}`, 'error');
    } else {
      trainState.value = 'idle';
    }
  }
}

// ===== Taiji 原生运行时检测与微调 =====

/**
 * 检测当前加载的运行时是否为 Seed 原生 Taiji。
 * 兼容运行时不再被当作 Seed 训练主体。
 * 应在训练页面 onMounted 时调用。
 */
export async function detectTaijiModel() {
  try {
    const res = await authFetch(`${API_BASE}/api/runtime/status`);
    if (res.ok) {
      const data = await res.json();
      if (data.health?.is_seed) {
        isTaijiModel.value = true;
        Object.assign(taijiModelInfo, {
          size: 'Seed Native',
          parameters: { active: data.health?.model_name || '' },
          config: {},
          available_sizes: [],
          checkpoints: {},
        });
        return true;
      }
    }
  } catch (e) { /* not a taiji model */ }
  isTaijiModel.value = false;
  return false;
}

/**
 * 启动 Seed 原生 Taiji byte-stream 训练。
 */
export async function startTaijiTraining(toast) {
  trainState.value = 'running';
  trainLog.value = '';
  trainLoss.value = [];
  trainProgress.value = 0;
  trainAbortController = new AbortController();
  Object.assign(trainMetrics, {
    elapsed: 0, eta: null, lr: null, epoch: 1,
    total_epochs: 1,
    grad_norm: null, samples_per_sec: 0, total_steps: 0, current_loss: null,
  });
  Object.assign(trainDevice, {
    device_type: '', device_name: '', gpu_name: null,
    gpu_memory_gb: null, ram_gb: null, message: '',
  });

  try {
    trainLog.value += '🧠 启动 Taiji 原生训练（raw bytes + local plasticity）...\n';
    autoScrollTrainLog();

    const res = await authFetch(`${API_BASE}/api/train/native`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        datasets: selectedDatasets.value.length ? [...selectedDatasets.value] : null,
        parameter_budget: taijiTrainParams.parameter_budget,
        max_symbols: taijiTrainParams.max_symbols || null,
        device: taijiTrainParams.device || 'auto',
        // seed 被清空（null/非整数）时后端会返回 422，发送前兜底默认值
        seed: Number.isInteger(taijiTrainParams.seed) ? taijiTrainParams.seed : 20260822,
      }),
      signal: trainAbortController.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    trainReader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await trainReader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6);
        if (payload === '[DONE]') continue;
        let evt;
        try {
          evt = JSON.parse(payload);
        } catch (parseError) {
          console.debug('[useTraining] native SSE parse error:', parseError.message);
          continue;
        }
        if (evt.type === 'hardware_diag') {
          Object.assign(trainDevice, {
            device_type: evt.device_type || '', device_name: evt.device_name || '',
            gpu_name: evt.gpu_name || null, gpu_memory_gb: evt.gpu_memory_gb || null,
            ram_gb: evt.ram_gb || null, message: evt.message || '',
          });
        } else if (evt.type === 'progress') {
          trainProgress.value = Math.round((evt.fraction || 0) * 100);
          trainLog.value += `${evt.desc || ''}\n`;
          if (evt.loss != null) {
            trainLoss.value.push({ step: evt.step || 0, loss: evt.loss });
            trainMetrics.current_loss = evt.loss;
          }
          Object.assign(trainMetrics, {
            elapsed: evt.elapsed ?? trainMetrics.elapsed,
            eta: evt.eta ?? trainMetrics.eta,
            epoch: evt.epoch ?? trainMetrics.epoch,
            total_epochs: evt.total_epochs ?? trainMetrics.total_epochs,
            samples_per_sec: evt.samples_per_sec ?? trainMetrics.samples_per_sec,
            total_steps: evt.total_steps ?? trainMetrics.total_steps,
          });
        } else if (evt.type === 'completed') {
          trainState.value = 'completed';
          trainProgress.value = 100;
          trainLog.value += `✅ ${evt.message || '原生训练完成'}\n`;
        } else if (evt.type === 'error') {
          throw new Error(evt.message || '原生训练失败');
        }
        autoScrollTrainLog();
      }
    }
    if (trainState.value === 'running') trainState.value = 'completed';
    toast('✅ Taiji 原生训练完成', 'success');
    await detectTaijiModel();
  } catch (err) {
    if (err.name !== 'AbortError') {
      trainLog.value += `❌ ${err.message}\n`;
      trainState.value = 'idle';
      autoScrollTrainLog();
      toast(`❌ ${err.message}`, 'error');
    } else {
      trainState.value = 'idle';
    }
  }
}

// ===== Loss 曲线绘图 =====

export function drawLossChart() {
  const canvas = lossCanvasRef.value;
  if (!canvas || trainLoss.value.length < 2) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width, h = rect.height;
  const pad = { top: 12, right: 16, bottom: 28, left: 48 };
  const pw = w - pad.left - pad.right;
  const ph = h - pad.top - pad.bottom;
  const data = trainLoss.value;
  const losses = data.map(d => d.loss);
  const minL = Math.min(...losses), maxL = Math.max(...losses);
  const range = maxL - minL || 1;

  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(148,163,184,0.12)';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (ph / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
  }

  ctx.fillStyle = '#94a3b8';
  ctx.font = '10px ' + getComputedStyle(document.documentElement).fontFamily;
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const val = maxL - (range / 4) * i;
    const y = pad.top + (ph / 4) * i + 3;
    ctx.fillText(val.toFixed(3), pad.left - 6, y);
  }

  ctx.textAlign = 'center';
  const steps = data.map(d => d.step);
  ctx.fillText('Step ' + steps[0], pad.left, h - 4);
  ctx.fillText('Step ' + steps[steps.length - 1], w - pad.right, h - 4);

  const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ph);
  grad.addColorStop(0, 'rgba(99,102,241,0.28)');
  grad.addColorStop(1, 'rgba(99,102,241,0.02)');

  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = pad.left + (i / (data.length - 1)) * pw;
    const y = pad.top + ((maxL - data[i].loss) / range) * ph;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.lineTo(pad.left + pw, pad.top + ph);
  ctx.lineTo(pad.left, pad.top + ph);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = pad.left + (i / (data.length - 1)) * pw;
    const y = pad.top + ((maxL - data[i].loss) / range) * ph;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = '#1a1a1a';
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.stroke();

  for (let i = 0; i < data.length; i++) {
    const x = pad.left + (i / (data.length - 1)) * pw;
    const y = pad.top + ((maxL - data[i].loss) / range) * ph;
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = '#1a1a1a';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}
