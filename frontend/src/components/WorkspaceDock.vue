<template>
  <section
    v-show="store.isOpen"
    class="workspace-dock"
    :class="{ 'dock-full': store.isFull }"
    :style="store.isFull ? null : { width: store.dockWidth + 'px' }"
    aria-label="IDE 工作区"
  >
    <!-- 左缘拖拽调宽（仅分栏档） -->
    <div
      v-if="!store.isFull"
      class="dock-resize-handle"
      :class="{ active: dockResizing }"
      @mousedown="startDockResize"
    ></div>

    <!-- 紧凑工具条 -->
    <header class="dock-topbar">
      <div class="dock-heading">
        <span
          class="workbench-projection"
          :class="{ ready: workbenchReady }"
          :title="workbenchStatusText"
        >
          <span class="workbench-projection-dot"></span>
        </span>
        <span class="dock-title">工作区</span>
        <span class="dock-path" :title="workspacePath">{{ workspacePath || 'Seed脚本与配置编辑' }}</span>
      </div>
      <div class="dock-actions">
        <button class="dock-btn" title="打开文件夹" @click="openPathDialog">
          <FolderOpen :size="14" />
        </button>
        <button class="dock-btn" title="快速打开文件 (Ctrl+P)" @click="openQuickOpen">
          <Search :size="14" />
        </button>
        <button class="dock-btn" :class="{ active: showTerminal }" title="终端 (Ctrl+`)" @click="showTerminal = !showTerminal">
          <Terminal :size="14" />
        </button>
        <button class="dock-btn" :disabled="running" title="运行当前文件" @click="handleRun">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        </button>
        <button class="dock-btn" title="保存 (Ctrl+S)" @click="handleSave">
          <Save :size="14" />
        </button>
        <span class="dock-sep"></span>
        <button
          class="dock-btn"
          :class="{ active: !store.isFull }"
          title="分栏视图"
          @click="store.setMode('split')"
        >
          <Columns2 :size="14" />
        </button>
        <button
          class="dock-btn"
          :class="{ active: store.isFull }"
          title="全宽视图"
          @click="store.setMode('full')"
        >
          <Maximize2 :size="14" />
        </button>
        <button class="dock-btn" title="收起工作区" @click="store.collapse()">
          <PanelRightClose :size="14" />
        </button>
      </div>
    </header>

    <!-- 主体：首次展开后常驻挂载（v-show），Monaco 状态天然保持 -->
    <div v-if="bodyMounted" class="dock-body">
      <div class="ide-layout" :style="{ gridTemplateColumns: sidebarWidth + 'px minmax(0, 1fr) 232px' }">
        <!-- 左栏：文件树 -->
        <WorkspaceFileTree
          :file-tree="fileTree"
          :flat-list="flatList"
          :expanded-dirs="expandedDirs"
          :get-file-icon="getFileIcon"
          @new-file="handleNewFile"
          @refresh="loadTree"
          @select-node="handleTreeClick"
          @context-node="showContextMenu"
          @resize="startResize"
        />

        <!-- 中栏：编辑器 + 终端 -->
        <WorkspaceEditorPane
          ref="editorPane"
          :show-terminal="showTerminal"
          :terminal-height="terminalHeight"
          :approval-handler="approveWorkbenchMutation"
          @saved="onFileSaved"
          @save-error="onSaveError"
          @resize-terminal="startTerminalResize"
        />

        <!-- 右栏：属性与检查器 -->
        <div class="panel panel-right">
          <div class="panel-header">
            <Clock :size="13" />
            属性
          </div>
          <div class="panel-body">
            <div v-if="currentFile" class="prop-group">
              <div class="prop-group-title">文件信息</div>
              <div class="prop-row"><span class="prop-label">文件名</span><span class="prop-value">{{ currentFile.name }}</span></div>
              <div class="prop-row"><span class="prop-label">路径</span><span class="prop-value prop-truncate" :title="currentFile.path">{{ currentFile.path }}</span></div>
              <div class="prop-row"><span class="prop-label">大小</span><span class="prop-value">{{ formatFileSize(currentFile.content?.length || 0) }}</span></div>
              <div class="prop-row"><span class="prop-label">行数</span><span class="prop-value">{{ countLines(currentFile.content) }}</span></div>
              <div class="prop-row"><span class="prop-label">类型</span><span class="prop-value">{{ currentFile.language?.toUpperCase() || '—' }}</span></div>
            </div>
            <div v-else class="prop-empty">未打开文件</div>
            <div class="prop-group">
              <div class="prop-group-title">工作区统计</div>
              <div class="prop-row"><span class="prop-label">文件数</span><span class="prop-value">{{ workspaceStats.files }}</span></div>
              <div class="prop-row"><span class="prop-label">目录数</span><span class="prop-value">{{ workspaceStats.dirs }}</span></div>
            </div>
            <div class="prop-group workbench-contract">
              <div class="prop-group-title">Taiji 工作台</div>
              <div class="prop-row"><span class="prop-label">能力快照</span><span class="prop-value">{{ workbench.snapshotId.value ? '已验证' : '未读取' }}</span></div>
              <div v-if="workbench.latestOutcome.value" class="prop-row">
                <span class="prop-label">最近结果</span>
                <span class="prop-value">{{ workbench.latestOutcome.value.status }}</span>
              </div>
              <div v-if="workbench.error.value" class="prop-error">{{ workbench.error.value }}</div>
              <div v-else class="prop-hint">只读文件与目录由 native capability 提供</div>
              <RecoveryPortfolioAuditPanel />
            </div>
          </div>
        </div>
      </div>

      <!-- 底部状态栏 -->
      <div class="status-bar">
        <span v-if="monacoEditor?.activeTab" class="status-item"><Crosshair :size="12" />Ln {{ monacoEditor.cursorLine }}, Col {{ monacoEditor.cursorCol }}</span>
        <span v-else class="status-item"><Crosshair :size="12" />—</span>
        <span class="status-item"><FileText :size="12" />UTF-8</span>
        <span class="status-item"><FileCode :size="12" />{{ monacoEditor?.language?.toUpperCase() || '—' }}</span>
        <span v-if="monacoEditor?.isDirty" class="status-item status-dirty"><span class="dirty-dot"></span>未保存</span>
        <span class="status-spacer"></span>
        <span class="status-item"><Activity :size="12" class="status-sync-icon" />神经元同步中</span>
      </div>
    </div>
    <div v-else class="dock-empty">
      <FolderOpen :size="28" />
      <p>正在展开工作区…</p>
    </div>

    <!-- 对话框：新建/重命名 -->
    <div v-if="inputDialog.visible" class="dlg-overlay" @click.self="cancelInputDialog">
      <div class="dlg-box">
        <h3>{{ inputDialog.title }}</h3>
        <input v-model="inputDialog.value" class="dlg-input" :placeholder="inputDialog.placeholder" @keydown.enter="confirmInputDialog" @keydown.escape="cancelInputDialog" />
        <div class="dlg-actions">
          <button class="dlg-btn primary" @click="confirmInputDialog">确认</button>
          <button class="dlg-btn" @click="cancelInputDialog">取消</button>
        </div>
      </div>
    </div>
    <WorkspacePathDialog
      v-model:visible="showPathDialog"
      v-model:path="newPathInput"
      :quick-paths="quickPaths"
      :picking="picking"
      :error="pathDialogError"
      @browse="browseFolder"
      @apply="applyNewPath"
    />

    <!-- 快速打开（Ctrl+P） -->
    <div v-if="quickOpen.visible" class="dlg-overlay" @click.self="quickOpen.visible = false">
      <div class="quickopen-box">
        <div class="quickopen-input-row">
          <Search :size="15" class="icon-sm" />
          <input
            ref="quickOpenInput" v-model="quickOpen.query" class="quickopen-input"
            placeholder="输入文件名，回车打开（Esc 关闭）"
            @keydown.enter.prevent="openQuickOpenSelection"
            @keydown.escape.prevent="quickOpen.visible = false"
            @keydown.arrow-down.prevent="quickOpenIndex = Math.min(quickOpenMatches.length - 1, quickOpenIndex + 1)"
            @keydown.arrow-up.prevent="quickOpenIndex = Math.max(0, quickOpenIndex - 1)" />
        </div>
        <div class="quickopen-list">
          <div
            v-for="(node, i) in quickOpenMatches" :key="node.path"
            class="quickopen-item" :class="{ active: i === quickOpenIndex }"
            @mouseenter="quickOpenIndex = i" @click="openQuickOpenFile(node)">
            <component :is="node.type === 'directory' ? Folder : getFileIcon(node.name)" :size="14" class="tree-icon" />
            <span class="quickopen-name">{{ node.name }}</span>
            <span class="quickopen-path">{{ node.path }}</span>
          </div>
          <p v-if="!quickOpenMatches.length" class="quickopen-empty">没有匹配的文件</p>
        </div>
      </div>
    </div>

    <!-- 右键菜单 -->
    <div v-if="contextMenu.visible" class="ctx-menu" :style="{ left: contextMenu.x + 'px', top: contextMenu.y + 'px' }" @click="contextMenu.visible = false">
      <div class="ctx-item" @click="openInEditor"><Edit3 :size="13" /> 打开</div>
      <template v-if="contextMenu.node?.type === 'file'">
        <div class="ctx-sep"></div>
        <div class="ctx-item" @click="renameItem"><Edit2 :size="13" /> 重命名</div>
        <div class="ctx-sep"></div>
        <div class="ctx-item danger" @click="deleteItem"><Trash2 :size="13" /> 删除</div>
      </template>
    </div>
  </section>
</template>

<script setup>
/**
 * 全局工作区 Dock（dsh 式侧栏工作区）。
 *
 * 由 WorkspaceView（原 /workspace 独立路由页）整体迁移而来，三个行为变化：
 * 1. 常驻 App 壳层（.app-body 第三个面板），v-show 显隐——Monaco/文件树状态跨页面保持；
 * 2. 档位 collapsed/split/full 由 workspaceStore 驱动并持久化，full 时 router-wrapper 让位；
 * 3. 生命周期从 keep-alive 的 onActivated/onDeactivated 改为「档位切换驱动」：
 *    首次展开挂载 body（规避 display:none 下 Monaco 初始化）、可见时 workbench.start()。
 */
import { ref, reactive, computed, watch, onMounted, onUnmounted, inject, nextTick } from 'vue';
import { nativeApi } from '../composables/nativeApi.js';
import { useWorkbenchProjection } from '../composables/useWorkbenchProjection.js';
import { useWorkspaceStore } from '../stores/workspaceStore.js';
import { Terminal, FolderOpen, Folder, FileCode, FileText, Image as ImageIcon, Database, Edit3, Edit2, Trash2, Crosshair, Activity, Search, Clock, Save, Columns2, Maximize2, PanelRightClose } from 'lucide-vue-next';
import WorkspacePathDialog from './WorkspacePathDialog.vue';
import WorkspaceFileTree from './WorkspaceFileTree.vue';
import WorkspaceEditorPane from './WorkspaceEditorPane.vue';
import RecoveryPortfolioAuditPanel from './RecoveryPortfolioAuditPanel.vue';

const toast = inject('toast');
const $confirm = inject('$confirm');
const workbench = useWorkbenchProjection();
const store = useWorkspaceStore();

const workspacePath = ref('');
const fileTree = ref([]);
const flatList = ref([]);
const expandedDirs = reactive(new Set());
const showTerminal = ref(false);
const running = ref(false);
const sidebarWidth = ref(200);
const terminalHeight = ref(240);
const editorPane = ref(null);
const monacoEditor = computed(() => editorPane.value?.monacoEditor || null);
const contextMenu = ref({ visible: false, x: 0, y: 0, node: null });
const inputDialog = ref({ visible: false, title: '', value: '', placeholder: '', resolve: null });
const showPathDialog = ref(false);
const newPathInput = ref('');
const pathDialogError = ref('');
const quickPaths = ref([]);
const picking = ref(false);
// 快速打开面板（Ctrl+P）
const quickOpen = reactive({ visible: false, query: '' });
const quickOpenInput = ref(null);
const quickOpenIndex = ref(0);
const projectedEventSequence = ref(0);
// 首次展开后才挂载 body（Monaco 在 display:none 下初始化会拿到零尺寸）
const bodyMounted = ref(false);
const dockResizing = ref(false);

const workbenchReady = computed(() => (
  workbench.isEnabled('workspace.list') && workbench.isEnabled('workspace.read')
));
const workbenchStatusText = computed(() => {
  if (workbench.error.value) return workbench.error.value;
  if (!workbench.snapshotId.value) return '正在读取 Taiji native workbench capability snapshot';
  return `native capability snapshot ${workbench.snapshotId.value.slice(0, 12)}`;
});
const quickOpenMatches = computed(() => {
  const q = quickOpen.query.trim().toLowerCase();
  const all = allFiles.value;
  if (!q) return all.slice(0, 50);
  // 简易模糊匹配：子序列或路径包含，按命中位置排序
  const scored = [];
  for (const node of all) {
    const name = node.name.toLowerCase();
    const path = node.path.toLowerCase();
    let score = -1;
    const idx = name.indexOf(q);
    if (idx >= 0) score = idx === 0 ? 0 : 10;
    else if (path.includes(q)) score = 50;
    else {
      let pi = 0;
      for (let ci = 0; ci < name.length && pi < q.length; ci++) {
        if (name[ci] === q[pi]) pi++;
      }
      if (pi === q.length) score = 100;
    }
    if (score >= 0) scored.push({ node, score });
  }
  scored.sort((a, b) => a.score - b.score || a.node.path.localeCompare(b.node.path));
  return scored.slice(0, 50).map(s => s.node);
});
// 工作区全量文件清单（不受目录展开状态影响）
const allFiles = computed(() => {
  const out = [];
  const walk = (nodes) => {
    for (const n of nodes) {
      if (n.type === 'file') out.push(n);
      else if (n.children) walk(n.children);
    }
  };
  walk(fileTree.value);
  return out;
});

// 当前激活文件（从 MonacoEditor 读取）
const currentFile = computed(() => {
  if (!monacoEditor.value?.openTabs || !monacoEditor.value?.activeTab) return null;
  return monacoEditor.value.openTabs.find(t => t.path === monacoEditor.value.activeTab) || null;
});

// 工作区统计：基于已加载的 fileTree 递归计算，零新增请求
const workspaceStats = computed(() => {
  let files = 0, dirs = 0;
  const walk = (nodes) => {
    for (const n of nodes) {
      if (n.type === 'directory') { dirs += 1; walk(n.children || []); }
      else files += 1;
    }
  };
  walk(fileTree.value);
  return { files, dirs };
});

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function countLines(content) {
  if (!content) return 0;
  return content.split('\n').length;
}

function confirmInputDialog() {
  const val = inputDialog.value.value.trim();
  inputDialog.value.visible = false;
  if (inputDialog.value.resolve) inputDialog.value.resolve(val || null);
}
function cancelInputDialog() {
  inputDialog.value.visible = false;
  if (inputDialog.value.resolve) inputDialog.value.resolve(null);
}

async function openPathDialog() {
  showPathDialog.value = true;
  pathDialogError.value = '';
  if (quickPaths.value.length) return;
  try {
    const data = await nativeApi.systemQuickPaths();
    quickPaths.value = data.paths || [];
  } catch (e) { /* 快速路径仅为便捷入口，加载失败不打断对话框 */ }
}

// 系统级目录选择：后端弹原生对话框（PowerShell BrowseForFolder）
async function browseFolder() {
  if (picking.value) return;
  picking.value = true;
  pathDialogError.value = '';
  try {
    const r = await nativeApi.systemSelectFolder();
    const data = await r.json().catch(() => ({}));
    if (r.ok && data.status === 'ok' && data.path) {
      newPathInput.value = data.path;
      applyNewPath(); // 选中即切换，少一次点击
    } else if (data.status === 'cancel') {
      // 用户取消，静默
    } else {
      pathDialogError.value = data.detail || '目录选择失败';
    }
  } catch (e) {
    pathDialogError.value = '无法打开系统目录选择框: ' + e.message;
  } finally {
    picking.value = false;
  }
}

// ===== 快速打开（Ctrl+P）=====
function openQuickOpen() {
  if (!store.isOpen) store.open('split');
  quickOpen.visible = true;
  quickOpen.query = '';
  quickOpenIndex.value = 0;
  nextTick(() => quickOpenInput.value?.focus());
}
function openQuickOpenFile(node) {
  if (node.type !== 'file') return;
  quickOpen.visible = false;
  if (monacoEditor.value) monacoEditor.value.openFile(node.path);
}
function openQuickOpenSelection() {
  const node = quickOpenMatches.value[quickOpenIndex.value];
  if (node) openQuickOpenFile(node);
}

// ===== Dock 快捷键：Ctrl+P 快速打开 / Ctrl+` 终端（常驻，收起时先展开） =====
function onGlobalKeydown(e) {
  if (e.ctrlKey && !e.altKey && !e.shiftKey) {
    if (e.key.toLowerCase() === 'p') { e.preventDefault(); openQuickOpen(); }
    else if (e.key === '`' || e.code === 'Backquote') {
      e.preventDefault();
      if (!store.isOpen) store.open('split');
      nextTick(() => { showTerminal.value = !showTerminal.value; });
    }
  }
  if (e.key === 'Escape' && quickOpen.visible) quickOpen.visible = false;
}

function showInputDialog(title, placeholder, initialValue = '') {
  return new Promise((resolve) => {
    inputDialog.value = { visible: true, title, value: initialValue, placeholder, resolve };
  });
}

async function handleNewFile() {
  const name = await showInputDialog('新建文件', '输入文件名，如 main.py');
  if (!name) return;
  try {
    const result = await executeNativeMutation({
      kind: 'workspace.create',
      parameters: { path: name, content: '' },
      expectedOutcome: `create ${name}`,
    });
    if (!result) return;
    toast(`已创建 ${name}`, 'success');
    await loadTree();
    await monacoEditor.value?.openFile(result.path || name);
  } catch (e) { toast('创建文件失败: ' + e.message, 'error'); }
}

async function handleSave() {
  if (!monacoEditor.value?.activeTab) { toast('没有可保存的活动文件', 'info'); return; }
  // 失败详情由 save-error 事件统一 toast（覆盖工具条保存 / Ctrl+S / 编辑器工具栏三条入口）
  await monacoEditor.value.saveFile();
}

function onFileSaved(path) { toast(`已保存 ${path}`, 'success'); }
function onSaveError(detail) { toast(detail || '保存失败', 'error'); }

async function handleRun() {
  const tab = currentFile.value;
  if (!tab) { toast('没有可运行的活动文件', 'info'); return; }
  const assessment = tab.languageAssessment || {};
  const candidates = [
    assessment.runner_id,
    ...(assessment.toolchain_commands || []),
  ].filter(Boolean);
  const available = new Set(assessment.available_toolchains || []);
  const runner = candidates.find(command => available.has(command));
  if (!runner) {
    toast('当前语言没有可用的原生 runner', 'info');
    return;
  }
  running.value = true;
  try {
    const result = await executeNativeMutation({
      kind: 'terminal.run',
      parameters: {
        argv: [runner, tab.path],
        cwd: '.',
        timeout_seconds: 30,
        env: {},
        env_allowlist: [],
        output_limit: 65536,
        expected_artifacts: [],
        execution_kind: 'command',
      },
      expectedOutcome: `run ${tab.path} with ${runner}`,
    });
    if (!result) return;
    if (result.success) {
      const output = [result.stdout, result.stderr].filter(Boolean).join('\n').trim();
      toast(output ? `运行成功：${output}`.slice(0, 200) : '运行成功', 'success');
    } else {
      toast(result.stderr || '运行失败', 'error');
    }
  } catch (e) { toast('运行失败: ' + e.message, 'error'); }
  finally { running.value = false; }
}

async function applyNewPath() {
  const path = newPathInput.value.trim();
  if (!path) { pathDialogError.value = '请输入路径'; return; }
  pathDialogError.value = '';
  try {
    const data = await workbench.setWorkspaceRoot(path);
    if (data.status === 'ok') {
      workspacePath.value = data.path;
      showPathDialog.value = false;
      newPathInput.value = '';
      expandedDirs.clear();
      loadTree();
      toast('路径已切换', 'success');
    } else {
      pathDialogError.value = data.detail || '路径设置失败';
    }
  } catch (e) { pathDialogError.value = e.message; }
}

async function loadWorkspacePath() {
  try {
    await workbench.ensureCapabilities();
    workspacePath.value = workbench.workspaceRoot.value || '';
  } catch (e) { toast('加载工作路径失败: ' + e.message, 'error') }
}

async function loadTree() {
  try {
    await workbench.ensureCapabilities();
    fileTree.value = mapWorkbenchEntries(await workbench.listDirectory('.'));
    expandedDirs.clear();
    recomputeFlatList();
  } catch (e) { toast('加载文件树失败: ' + e.message, 'error') }
}

function mapWorkbenchEntries(entries) {
  return entries.map(entry => ({
    ...entry,
    children: entry.type === 'directory' ? null : undefined,
  }));
}

function recomputeFlatList() {
  flatList.value = flattenTree(fileTree.value, 0);
}

function flattenTree(nodes, depth) {
  let result = [];
  for (const node of nodes) {
    node.depth = depth;
    result.push(node);
    if (node.type === 'directory' && expandedDirs.has(node.path) && node.children) {
      result = result.concat(flattenTree(node.children, depth + 1));
    }
  }
  return result;
}

async function handleTreeClick(node) {
  if (node.type === 'directory') {
    if (expandedDirs.has(node.path)) expandedDirs.delete(node.path);
    else {
      try {
        if (!Array.isArray(node.children)) {
          node.children = mapWorkbenchEntries(await workbench.listDirectory(node.path));
        }
        expandedDirs.add(node.path);
      } catch (e) {
        toast('加载目录失败: ' + e.message, 'error');
        return;
      }
    }
    recomputeFlatList();
  } else {
    if (monacoEditor.value) monacoEditor.value.openFile(node.path);
  }
}

function getFileIcon(name) {
  const ext = name.split('.').pop()?.toLowerCase();
  const map = { py: FileCode, js: FileCode, ts: FileCode, json: FileCode, html: FileCode, css: FileCode, vue: FileCode, md: FileText, txt: FileText, sh: Terminal, sql: Database, png: ImageIcon, jpg: ImageIcon };
  return map[ext] || FileText;
}

function showContextMenu(e, node) { contextMenu.value = { visible: true, x: e.clientX, y: e.clientY, node }; }
function openInEditor() { if (contextMenu.value.node?.type === 'file' && monacoEditor.value) monacoEditor.value.openFile(contextMenu.value.node.path); }

// 重命名后把已打开标签的路径/名称同步到新路径（支持目录前缀替换）
function syncTabsAfterRename(oldPath, newPath) {
  const editor = monacoEditor.value;
  if (!editor?.openTabs) return;
  const sep = oldPath.includes('\\') ? '\\' : '/';
  for (const tab of editor.openTabs) {
    let mapped = null;
    if (tab.path === oldPath) mapped = newPath;
    else if (tab.path.startsWith(oldPath + sep)) mapped = newPath + tab.path.slice(oldPath.length);
    if (!mapped) continue;
    if (editor.activeTab === tab.path) editor.activeTab = mapped;
    tab.path = mapped;
    tab.name = mapped.split(/[\\/]/).pop();
  }
}

async function renameItem() {
  const node = contextMenu.value.node;
  if (!node || node.type !== 'file') return;
  const newName = await showInputDialog('重命名', '输入新名称', node.name);
  if (!newName || newName === node.name) return;
  const sepIdx = Math.max(node.path.lastIndexOf('/'), node.path.lastIndexOf('\\'));
  const parentPrefix = sepIdx >= 0 ? node.path.slice(0, sepIdx + 1) : '';
  const newPath = parentPrefix + newName;
  try {
    const file = await workbench.readFile(node.path);
    const result = await executeNativeMutation({
      kind: 'workspace.rename',
      parameters: { path: node.path, new_path: newPath, before_digest: file.digest },
      expectedOutcome: `rename ${node.path} to ${newPath}`,
    });
    if (result) {
      syncTabsAfterRename(node.path, result.new_path || newPath);
      toast(`已重命名为 ${newName}`, 'success');
      await loadTree();
    }
  } catch (e) { toast('重命名失败: ' + e.message, 'error'); }
}
async function deleteItem() {
  const node = contextMenu.value.node;
  if (!node || node.type !== 'file') return;
  try {
    const file = await workbench.readFile(node.path);
    const result = await executeNativeMutation({
      kind: 'workspace.delete',
      parameters: { path: node.path, before_digest: file.digest },
      expectedOutcome: `delete ${node.path}`,
    });
    if (result) {
      toast(`已删除 ${node.name}`, 'success');
      await loadTree();
    }
  } catch (e) { toast('删除失败: ' + e.message, 'error') }
}

async function approveWorkbenchMutation({ kind, parameters, preview }) {
  const mutation = preview?.preview?.mutation || {};
  const target = parameters.path || mutation.path || parameters.argv?.join(' ') || kind;
  const operation = { 'workspace.apply_patch': '保存文件', 'workspace.create': '新建文件', 'workspace.rename': '重命名文件', 'workspace.delete': '删除文件', 'terminal.run': '运行命令' }[kind] || kind;
  if (typeof $confirm === 'function') {
    return $confirm({
      title: `确认${operation}`,
      message: `${target}\n该操作将通过 Taiji 原生工作台执行。`,
      type: kind === 'workspace.delete' ? 'danger' : 'warning',
    });
  }
  return window.confirm(`${operation}: ${target}`);
}

async function executeNativeMutation({ kind, parameters, expectedOutcome }) {
  if (!workbench.isEnabled(kind)) throw new Error(`原生工作台未提供 ${kind}`);
  const intentId = `ui:${kind}:${Date.now()}`;
  const preview = await workbench.previewIntent({ intentId, kind, parameters, expectedOutcome });
  if (!preview.approval?.approval_token || !preview.preview) {
    throw new Error(preview.policy?.reason_code || '原生工作台未发出审批预览');
  }
  if (!await approveWorkbenchMutation({ kind, parameters, preview })) return null;
  const result = await workbench.executeIntent({
    intentId,
    kind,
    parameters,
    expectedOutcome,
    approvalToken: preview.approval.approval_token,
  });
  if (!result.outcome?.success) throw new Error(result.outcome?.error || '原生工作台执行失败');
  return result.outcome.result || {};
}

let resizing = false; // eslint-disable-line no-unused-vars -- 仅供未来拖拽阈值判断读取
function startResize(e) {
  resizing = true;
  const startX = e.clientX, startW = sidebarWidth.value;
  const onMove = (ev) => { sidebarWidth.value = Math.max(120, Math.min(400, startW + ev.clientX - startX)); };
  const onUp = () => { resizing = false; document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}

function startTerminalResize(e) {
  const startY = e.clientY, startH = terminalHeight.value;
  const onMove = (ev) => { terminalHeight.value = Math.max(80, Math.min(500, startH - (ev.clientY - startY))); };
  const onUp = () => { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}

// Dock 左缘拖拽调宽（仅分栏档；方向与侧栏相反：向左拖=加宽）
function startDockResize(e) {
  e.preventDefault();
  dockResizing.value = true;
  const startX = e.clientX, startW = store.dockWidth;
  const onMove = (ev) => { store.setWidth(startW - (ev.clientX - startX)); };
  const onUp = () => {
    dockResizing.value = false;
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
  };
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}

function closeCtx() { contextMenu.value.visible = false; }

// ===== 档位切换驱动生命周期（替代原 keep-alive 的 onActivated/onDeactivated） =====
function beginOpen() {
  // 展开：首次挂载 body + 启动 projection + 加载树（幂等）
  bodyMounted.value = true;
  workbench.start();
  loadWorkspacePath();
  loadTree();
  nextTick(() => monacoEditor.value?.layout?.());
}
function beginClose() {
  // 收起：停 projection、收终端与浮层（与原 onDeactivated 语义一致）
  workbench.stop();
  showTerminal.value = false;
  quickOpen.visible = false;
}
watch(() => store.dockMode, (mode, prev) => {
  const isOpen = mode !== 'collapsed';
  // immediate 首帧 prev 为 undefined：若初始即展开（如 localStorage 恢复 split/full），做展开初始化
  if (prev === undefined) {
    if (isOpen) beginOpen();
    return;
  }
  const wasOpen = prev !== 'collapsed';
  if (isOpen && !wasOpen) beginOpen();
  else if (!isOpen && wasOpen) beginClose();
}, { immediate: true });

// workbench 事件投影：editor.open 成功 → 编辑器打开对应文件
watch(workbench.events, (nextEvents) => {
  for (const event of nextEvents) {
    if (event.sequence <= projectedEventSequence.value) continue;
    projectedEventSequence.value = event.sequence;
    const outcome = event.phase === 'outcome' ? event.payload?.outcome : null;
    const path = outcome?.capability_id === 'editor.open' && outcome.success
      ? outcome.result?.path
      : '';
    if (path && typeof monacoEditor.value?.openFile === 'function') {
      monacoEditor.value.openFile(path);
    }
  }
}, { deep: true });

onMounted(() => {
  document.addEventListener('click', closeCtx);
  document.addEventListener('keydown', onGlobalKeydown);
});
onUnmounted(() => {
  workbench.stop();
  document.removeEventListener('click', closeCtx);
  document.removeEventListener('keydown', onGlobalKeydown);
});
</script>

<style scoped>
/* ===== Dock 容器：与 .router-wrapper 同一边框圆角语言（见 styles/shell.css） ===== */
.workspace-dock {
  position: relative;
  display: flex;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
  flex: none;
  margin: 8px 8px 8px 0;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg, 16px);
  box-shadow: var(--shadow-sm, 0 1px 2px rgba(0, 0, 0, 0.06));
  overflow: hidden;
  container: workspace-dock / inline-size;
  animation: dock-in 0.18s cubic-bezier(0.22, 0.61, 0.36, 1);
}
.workspace-dock.dock-full {
  flex: 1;
  width: auto !important;
}
@keyframes dock-in {
  from { opacity: 0; transform: translateX(12px); }
  to { opacity: 1; transform: none; }
}
@media (prefers-reduced-motion: reduce) {
  .workspace-dock { animation: none; }
}

/* 左缘拖拽手柄 */
.dock-resize-handle {
  position: absolute;
  left: -3px;
  top: 0;
  bottom: 0;
  width: 6px;
  z-index: 6;
  cursor: col-resize;
  background: transparent;
  transition: background 120ms ease;
}
.dock-resize-handle:hover,
.dock-resize-handle.active { background: color-mix(in srgb, var(--primary) 35%, transparent); }

/* ── 紧凑工具条 ── */
.dock-topbar {
  flex: none;
  height: 44px;
  padding: 0 10px 0 14px;
  display: flex;
  align-items: center;
  gap: 10px;
  border-bottom: 1px solid var(--border);
  background: var(--card);
}
.dock-heading {
  min-width: 0;
  flex: 1;
  display: flex;
  align-items: center;
  gap: 8px;
}
.dock-title {
  font-size: 0.82rem;
  font-weight: 650;
  color: var(--foreground);
  flex: none;
}
.dock-path {
  font-size: 0.72rem;
  color: var(--muted-foreground);
  font-family: var(--font-mono, ui-monospace, monospace);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.dock-actions {
  display: flex;
  align-items: center;
  gap: 2px;
  flex: none;
}
.dock-btn {
  width: 30px;
  height: 30px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: var(--muted-foreground);
  cursor: pointer;
  transition: background 120ms ease, color 120ms ease;
}
.dock-btn:hover { background: var(--muted); color: var(--foreground); }
.dock-btn.active { color: var(--primary); background: color-mix(in srgb, var(--primary) 10%, transparent); }
.dock-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.dock-btn:focus-visible { outline: 2px solid var(--ring); outline-offset: 1px; }
.dock-sep {
  width: 1px;
  height: 16px;
  background: var(--border);
  margin: 0 4px;
  flex: none;
}
.workbench-projection {
  display: inline-flex;
  align-items: center;
  color: var(--muted-foreground);
  flex: none;
}
.workbench-projection.ready { color: var(--chart-2); }
.workbench-projection-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: currentColor;
}

/* ── 主体 ── */
.dock-body {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}
.dock-empty {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: var(--muted-foreground);
  font-size: 0.82rem;
}

/* ── 三栏 IDE 布局（承自 WorkspaceView） ── */
.ide-layout {
  flex: 1;
  min-height: 0;
  display: grid;
}
.panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
  position: relative;
}
.panel-right { border-left: 1px solid var(--border); }
.panel-header {
  padding: 10px 12px;
  font-size: 0.7rem;
  font-weight: 600;
  color: var(--muted-foreground);
  letter-spacing: 0.04em;
  text-transform: uppercase;
  border-bottom: 1px solid var(--border);
  flex: none;
  display: flex;
  align-items: center;
  gap: 6px;
}
.panel-body {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

/* ── 底部状态栏 ── */
.status-bar {
  height: 26px;
  flex: none;
  border-top: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 12px;
  gap: 14px;
  font-size: 0.72rem;
  color: var(--muted-foreground);
  background: var(--card);
}
.status-item {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  white-space: nowrap;
}
.status-item svg { flex-shrink: 0; color: var(--muted-foreground); }
.status-sync-icon { color: var(--chart-2) !important; }
.dirty-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: currentColor;
  flex-shrink: 0;
}
.status-spacer { flex: 1; }
.status-dirty { color: var(--chart-4); }

/* ── 右栏属性面板 ── */
.prop-group { margin-bottom: 12px; }
.prop-group-title {
  font-size: 0.68rem;
  font-weight: 600;
  color: var(--muted-foreground);
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-bottom: 5px;
  padding: 0 4px;
}
.prop-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 4px 8px;
  border-radius: 6px;
  font-size: 0.78rem;
}
.prop-row:hover { background: var(--muted); }
.prop-label { color: var(--muted-foreground); }
.prop-value { font-weight: 500; font-variant-numeric: tabular-nums; }
.prop-error {
  padding: 6px 8px;
  border-radius: 6px;
  color: var(--chart-4);
  background: color-mix(in srgb, var(--chart-4) 10%, transparent);
  font-size: 0.74rem;
  line-height: 1.4;
}
.prop-hint {
  padding: 4px 8px;
  color: var(--muted-foreground);
  font-size: 0.7rem;
  line-height: 1.4;
}
.prop-truncate {
  max-width: 110px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  text-align: right;
}
.prop-empty {
  text-align: center;
  padding: 20px 12px;
  color: var(--muted-foreground);
  font-size: 0.78rem;
}
.icon-sm { width: 15px; height: 15px; flex: none; }

/* ── 对话框 ── */
.dlg-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10000;
  backdrop-filter: blur(4px);
  -webkit-backdrop-filter: blur(4px);
}
.dlg-box {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius, 12px);
  padding: 24px;
  min-width: 380px;
  max-width: 90vw;
  box-shadow: 0 16px 48px rgba(0, 0, 0, 0.4);
}
.dlg-box h3 {
  margin: 0 0 12px;
  font-size: 15px;
  color: var(--foreground);
}
.dlg-input {
  width: 100%;
  padding: 9px 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--background);
  color: var(--foreground);
  font-family: inherit;
  font-size: 13px;
  outline: none;
  margin-bottom: 12px;
  transition: border-color 0.2s;
  box-sizing: border-box;
}
.dlg-input:focus { border-color: var(--primary); }
.dlg-actions { display: flex; gap: 8px; justify-content: flex-end; }
.dlg-btn {
  padding: 7px 16px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: transparent;
  color: var(--muted-foreground);
  font-size: 13px;
  cursor: pointer;
  font-family: inherit;
  transition: background 150ms ease, color 150ms ease;
}
.dlg-btn:hover { background: var(--muted); color: var(--foreground); }
.dlg-btn.primary {
  border: 0;
  background: var(--primary);
  color: var(--primary-foreground);
}
.dlg-btn.primary:hover {
  background: color-mix(in srgb, var(--primary) 90%, var(--foreground));
}

/* ── 快速打开（Ctrl+P） ── */
.quickopen-box {
  width: min(560px, 90vw);
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius, 12px);
  box-shadow: 0 16px 48px rgba(0, 0, 0, 0.35);
  overflow: hidden;
  align-self: flex-start;
  margin-top: 12vh;
}
.quickopen-input-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 14px;
  border-bottom: 1px solid var(--border);
}
.quickopen-input {
  flex: 1;
  border: 0;
  outline: none;
  background: transparent;
  color: var(--foreground);
  font-family: inherit;
  font-size: 14px;
}
.quickopen-input::placeholder { color: var(--muted-foreground); }
.quickopen-list {
  max-height: 320px;
  overflow-y: auto;
  padding: 6px;
}
.quickopen-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.84rem;
  color: var(--foreground);
}
.quickopen-item.active { background: var(--muted); }
.quickopen-item.active .quickopen-name { color: var(--primary); }
.quickopen-name { flex-shrink: 0; font-weight: 500; }
.quickopen-path {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  text-align: right;
  font-size: 0.72rem;
  color: var(--muted-foreground);
  font-family: var(--font-mono, ui-monospace, monospace);
}
.quickopen-empty {
  margin: 0;
  padding: 18px;
  text-align: center;
  color: var(--muted-foreground);
  font-size: 0.82rem;
}

/* ── 右键菜单 ── */
.ctx-menu {
  position: fixed;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius, 12px);
  padding: 4px;
  z-index: 9999;
  min-width: 140px;
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
}
.ctx-item {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 6px 12px;
  font-size: 12px;
  color: var(--foreground);
  cursor: pointer;
  border-radius: 8px;
  transition: background 0.1s;
}
.ctx-item:hover { background: var(--muted); }
.ctx-item.danger:hover {
  background: rgba(239, 68, 68, 0.1);
  color: var(--destructive, #ef4444);
}
.ctx-sep {
  height: 1px;
  background: var(--border);
  margin: 3px 8px;
}

/* ── 响应式：按 dock 实际宽度收缩栏位（容器查询，窄窗不再误用 viewport 断点） ── */
@container workspace-dock (max-width: 860px) {
  .ide-layout { grid-template-columns: minmax(160px, 200px) minmax(0, 1fr) !important; }
  .panel-right { display: none; }
}
@container workspace-dock (max-width: 620px) {
  .ide-layout { grid-template-columns: minmax(0, 1fr) !important; }
  .dock-path { display: none; }
}
</style>
