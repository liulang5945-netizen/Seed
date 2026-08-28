<template>
  <div class="workspace-view">
    <!-- 顶栏 -->
    <header class="topbar">
      <div>
        <div class="topbar-title">IDE 工作区</div>
        <div class="topbar-sub topbar-path" :title="workspacePath">{{ workspacePath || 'Seed脚本与配置编辑' }}</div>
      </div>
      <div class="topbar-spacer"></div>
      <span
        class="workbench-projection"
        :class="{ ready: workbenchReady }"
        :title="workbenchStatusText"
      >
        <span class="workbench-projection-dot"></span>
        Taiji 工作台 · {{ workbenchReady ? '已接入' : '未接入' }}
      </span>
      <button class="btn btn-outline" @click="openPathDialog">
        <FolderOpen :size="15" />
        打开文件夹
      </button>
      <button class="btn btn-outline" title="快速打开文件 (Ctrl+P)" @click="openQuickOpen">
        <Search :size="15" />
        搜索文件
      </button>
      <button class="btn btn-outline" :class="{ active: showTerminal }" @click="showTerminal = !showTerminal">
        <Terminal :size="15" />
        {{ showTerminal ? '收起终端' : '终端' }}
      </button>
      <button class="btn btn-primary" :disabled="running" @click="handleRun">
        <svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        {{ running ? '运行中…' : '运行' }}
      </button>
      <button class="btn btn-outline" @click="handleSave">保存</button>
    </header>

    <!-- 主体工作区 -->
    <div class="workspace-body">
      <div class="ide-layout" :style="{ gridTemplateColumns: sidebarWidth + 'px minmax(0, 1fr) 260px' }">
        <!-- 左栏：文件树 -->
        <div class="panel panel-left">
          <div class="panel-header">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
            <span class="panel-header-spacer"></span>
            <button class="icon-btn" title="新建文件" @click="handleNewFile">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            </button>
            <button class="icon-btn" title="刷新文件树" @click="loadTree">
              <RefreshCw :size="13" />
            </button>
          </div>
          <div class="panel-body">
            <div v-if="!fileTree.length" class="tree-empty">
              <p class="tree-empty-text">当前工作区还没有文件。<br>用顶栏「打开文件夹」切换目录，或新建第一个文件开始工作。</p>
            </div>
            <template v-for="node in flatList" :key="node.path">
              <div
                class="tree-item"
                :class="{ 'tree-folder': node.type === 'directory' }"
                :style="{ paddingLeft: (node.depth * 18 + 8) + 'px' }"
                @click="handleTreeClick(node)"
                @contextmenu.prevent="showContextMenu($event, node)"
              >
                <component :is="node.type === 'directory' ? (expandedDirs.has(node.path) ? FolderOpen : Folder) : getFileIcon(node.name)" :size="14" class="tree-icon" />
                <span class="tree-label">{{ node.name }}</span>
              </div>
            </template>
          </div>
          <div class="resize-col" @mousedown="startResize"></div>
        </div>

        <!-- 中栏：编辑器 + 终端 -->
        <div class="panel panel-center">
          <div class="editor-area">
            <MonacoEditor
              ref="monacoEditor"
              class="monaco-container"
              :approval-handler="approveWorkbenchMutation"
              @saved="onFileSaved"
              @save-error="onSaveError"
            />
            <!-- 终端 -->
            <Transition name="term-slide">
              <div v-if="showTerminal" class="ide-terminal" :style="{ height: terminalHeight + 'px' }">
                <div class="resize-row" @mousedown="startTerminalResize"></div>
                <WebTerminal ref="webTerminal" />
              </div>
            </Transition>
          </div>
        </div>

        <!-- 右栏：属性与检查器 -->
        <div class="panel panel-right">
          <div class="panel-header">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>
            属性
          </div>
          <div class="panel-body">
            <div v-if="currentFile" class="prop-group">
              <div class="prop-group-title">文件信息</div>
              <div class="prop-row"><span class="prop-label">文件名</span><span class="prop-value">{{ currentFile.name }}</span></div>
              <div class="prop-row"><span class="prop-label">路径</span><span class="prop-value prop-truncate" :title="currentFile.path">{{ currentFile.path }}</span></div>
              <div class="prop-row"><span class="prop-label">大小</span><span class="prop-value">{{ formatFileSize(currentFile.content?.length || 0) }}</span></div>
              <div class="prop-row"><span class="prop-label">编码</span><span class="prop-value">UTF-8</span></div>
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

    <!-- 对话框 -->
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
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, onMounted, onUnmounted, onActivated, onDeactivated, inject, nextTick } from 'vue';
import { nativeApi } from '../composables/nativeApi.js';
import { useWorkbenchProjection } from '../composables/useWorkbenchProjection.js';
import { Terminal, FolderOpen, Folder, FileCode, FileText, Image as ImageIcon, Database, Edit3, Edit2, Trash2, Crosshair, Activity, Search, RefreshCw } from 'lucide-vue-next';
import MonacoEditor from '../components/MonacoEditor.vue';
import WebTerminal from '../components/WebTerminal.vue';
import WorkspacePathDialog from '../components/WorkspacePathDialog.vue';

defineOptions({ name: 'WorkspaceView' });

const toast = inject('toast');
const $confirm = inject('$confirm');
const workbench = useWorkbenchProjection();

const workspacePath = ref('');
const fileTree = ref([]);
const flatList = ref([]);
const expandedDirs = reactive(new Set());
const showTerminal = ref(false);
const running = ref(false);
const sidebarWidth = ref(220);
const terminalHeight = ref(280);
const monacoEditor = ref(null);
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
      // 子序列匹配（如 "apj" 匹配 "app.json"）
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

// ===== IDE 快捷键：Ctrl+P 快速打开 / Ctrl+` 终端 =====
function onGlobalKeydown(e) {
  if (e.ctrlKey && !e.altKey && !e.shiftKey) {
    if (e.key.toLowerCase() === 'p') { e.preventDefault(); openQuickOpen(); }
    else if (e.key === '`' || e.code === 'Backquote') { e.preventDefault(); showTerminal.value = !showTerminal.value; }
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
  // 失败详情由 save-error 事件统一 toast（覆盖顶栏保存 / Ctrl+S / 编辑器工具栏三条入口）
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
  // 分隔符沿用原路径风格（后端 relpath 在 Windows 上为 \）
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
  // 新名称沿用原目录：取父路径前缀（兼容 / 与 \ 分隔符）
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

function closeCtx() { contextMenu.value.visible = false; }
onMounted(() => {
  document.addEventListener('click', closeCtx);
  loadWorkspacePath();
});
// keep-alive 组件首次挂载与重新激活都会触发 onActivated，
// loadTree 统一由它负责，避免首次挂载双发请求。
// IDE 快捷键（Ctrl+P / Ctrl+`）仅在 IDE 激活期间生效。
onActivated(() => {
  workbench.start();
  loadTree();
  document.addEventListener('keydown', onGlobalKeydown);
});

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
// 离开页面（被 keep-alive 缓存）时收起终端并卸载快捷键，
// v-if 卸载会触发 WebTerminal 的 onBeforeUnmount 清理（WS / xterm / ResizeObserver）。
onDeactivated(() => {
  workbench.stop();
  showTerminal.value = false;
  quickOpen.visible = false;
  document.removeEventListener('keydown', onGlobalKeydown);
});
onUnmounted(() => {
  workbench.stop();
  document.removeEventListener('click', closeCtx);
  document.removeEventListener('keydown', onGlobalKeydown);
});
</script>

<style scoped>
.workspace-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  width: 100%;
  overflow: hidden;
  background: var(--background);
  color: var(--foreground);
}

/* ── 顶栏 ── */
/* 不画 border-bottom：外围边框由 .router-wrapper 独占（见 styles/shell.css） */
.topbar {
  height: 52px;
  flex: none;
  padding: 0 18px;
  display: flex;
  align-items: center;
  gap: 12px;
}
.topbar-title { font-size: 0.92rem; font-weight: 600; }
.topbar-sub { font-size: 0.72rem; color: var(--muted-foreground); margin-top: 1px; }
.topbar-path {
  max-width: 320px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.topbar-spacer { flex: 1; }
.workbench-projection {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--muted-foreground);
  font-size: 0.74rem;
  white-space: nowrap;
}
.workbench-projection.ready { color: var(--chart-2); }
.workbench-projection-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: currentColor;
}

/* ── 按钮 ── */
.btn {
  height: 36px;
  padding: 0 15px;
  border-radius: 999px;
  border: 1px solid transparent;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  font-size: 0.86rem;
  font-weight: 500;
  transition: background 150ms ease, border-color 150ms ease, transform 120ms ease, color 150ms ease;
  cursor: pointer;
  font-family: inherit;
}
.btn:active { transform: translateY(1px); }
.btn:focus-visible { outline: 2px solid var(--ring); outline-offset: 2px; }
.btn-primary { background: var(--primary); color: var(--primary-foreground); }
.btn-primary:hover { background: color-mix(in srgb, var(--primary) 90%, var(--foreground)); }
.btn-outline { background: var(--background); color: var(--foreground); border-color: var(--border); }
.btn-outline:hover { background: var(--muted); }
.btn-outline.active { border-color: var(--primary); color: var(--primary); }
.btn:disabled { opacity: 0.6; cursor: not-allowed; }
.btn-sm { height: 28px; padding: 0 12px; font-size: 0.78rem; margin-top: 10px; }
.icon-sm { width: 15px; height: 15px; flex: none; }

/* ── 主体工作区 ── */
.workspace-body {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

/* ── 三栏 IDE 布局 ── */
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
.panel-left { border-right: 1px solid var(--border); }
.panel-center { border-right: 1px solid var(--border); }
.panel-right { border-left: 1px solid var(--border); }

.panel-header {
  padding: 12px 14px;
  font-size: 0.74rem;
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
.panel-header-spacer { flex: 1; }
.icon-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 3px;
  border: 0;
  border-radius: 5px;
  background: transparent;
  color: var(--muted-foreground);
  cursor: pointer;
  transition: background 120ms ease, color 120ms ease;
}
.icon-btn:hover { background: var(--muted); color: var(--foreground); }
.icon-btn:focus-visible { outline: 2px solid var(--ring); outline-offset: 1px; }

/* ── 文件树 ── */
.tree-empty {
  text-align: center;
  padding: 24px 12px;
  color: var(--muted-foreground);
  font-size: 0.8rem;
}
.tree-empty-text {
  margin: 0 0 4px;
  line-height: 1.7;
}
.tree-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 8px;
  border-radius: 6px;
  font-size: 0.84rem;
  cursor: pointer;
  transition: background 120ms ease;
  color: var(--foreground);
}
.tree-item:hover { background: var(--muted); }
.tree-folder { font-weight: 600; }
.tree-icon {
  width: 16px;
  height: 16px;
  flex: none;
  color: var(--muted-foreground);
}
.tree-label {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.resize-col {
  position: absolute;
  top: 0;
  right: -2px;
  width: 4px;
  height: 100%;
  cursor: col-resize;
  z-index: 10;
  transition: background 0.15s;
}
.resize-col:hover { background: var(--primary); }

/* ── 编辑器区域 ── */
.editor-area {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.monaco-container {
  flex: 1;
  min-height: 0;
  width: 100%;
}
.editor-area :deep(.monaco-wrapper) {
  height: 100% !important;
  min-height: 0 !important;
}
.editor-area :deep(.monaco-editor-container) {
  flex: 1 !important;
  min-height: 0 !important;
  height: auto !important;
}

/* ── 终端 ── */
.ide-terminal {
  flex-shrink: 0;
  border-top: 1px solid var(--border);
  background: var(--card);
  position: relative;
}
.resize-row {
  position: absolute;
  top: -3px;
  left: 0;
  right: 0;
  height: 6px;
  cursor: row-resize;
  z-index: 10;
  transition: background 0.15s;
}
.resize-row:hover { background: var(--primary); }

/* 终端过渡动画 */
.term-slide-enter-active,
.term-slide-leave-active {
  transition: transform 0.2s ease;
}
.term-slide-enter-from,
.term-slide-leave-to {
  transform: translateY(100%);
}

/* ── 底部状态栏 ── */
.status-bar {
  height: 28px;
  flex: none;
  border-top: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 14px;
  gap: 16px;
  font-size: 0.74rem;
  color: var(--muted-foreground);
  background: var(--card);
}
/* 状态项：图标 + 文本对齐（统一 12px 图标，与主流 IDE 状态栏一致） */
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
.prop-group { margin-bottom: 14px; }
.prop-group-title {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--muted-foreground);
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-bottom: 6px;
  padding: 0 4px;
}
.prop-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 5px 8px;
  border-radius: 6px;
  font-size: 0.82rem;
}
.prop-row:hover { background: var(--muted); }
.prop-label { color: var(--muted-foreground); }
.prop-value { font-weight: 500; font-variant-numeric: tabular-nums; }
.prop-error {
  padding: 6px 8px;
  border-radius: 6px;
  color: var(--chart-4);
  background: color-mix(in srgb, var(--chart-4) 10%, transparent);
  font-size: 0.76rem;
  line-height: 1.4;
}
.prop-hint {
  padding: 4px 8px;
  color: var(--muted-foreground);
  font-size: 0.72rem;
  line-height: 1.4;
}
.prop-truncate {
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  text-align: right;
}
.prop-empty {
  text-align: center;
  padding: 20px 12px;
  color: var(--muted-foreground);
  font-size: 0.8rem;
}

/* ── 快捷按钮 ── */
/* .quick-btn 随右栏「快捷操作」分组一并移除：终端开关统一由顶栏按钮承担 */

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
  border-radius: var(--radius);
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
  border-radius: calc(var(--radius) * 0.6);
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
  border-radius: calc(var(--radius) * 0.6);
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
.dlg-error {
  color: var(--destructive, #ef4444);
  font-size: 12px;
  margin: 8px 0 0;
}

/* ── 快速打开（Ctrl+P） ── */
.quickopen-box {
  width: min(560px, 90vw);
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
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
  font-family: var(--font-mono, monospace);
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
  border-radius: var(--radius);
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

/* 响应式 */
@media (max-width: 880px) {
  .ide-layout { grid-template-columns: 1fr !important; }
  .panel-left, .panel-right { display: none; }
}
</style>
