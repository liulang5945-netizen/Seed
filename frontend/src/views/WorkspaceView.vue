<template>
  <div class="workspace-view">
    <!-- 顶栏 -->
    <header class="topbar">
      <div>
        <div class="topbar-title">IDE 工作区</div>
        <div class="topbar-sub topbar-path" :title="workspacePath">{{ workspacePath || 'Seed脚本与配置编辑' }}</div>
      </div>
      <div class="topbar-spacer"></div>
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
            项目文件
            <span class="panel-header-spacer"></span>
            <button class="icon-btn" title="新建文件" @click="handleNewFile">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            </button>
            <button class="icon-btn" title="新建文件夹" @click="handleNewFolder">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><line x1="12" y1="11" x2="12" y2="16"/><line x1="9.5" y1="13.5" x2="14.5" y2="13.5"/></svg>
            </button>
            <button class="icon-btn" title="刷新文件树" @click="loadTree">
              <RefreshCw :size="13" />
            </button>
          </div>
          <div class="panel-body">
            <div v-if="!fileTree.length" class="tree-empty">
              <p class="tree-empty-text">当前工作区还没有文件。<br>切换到已有目录，或新建第一个文件开始工作。</p>
              <button class="btn btn-outline btn-sm" @click="openPathDialog">切换目录</button>
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
            <MonacoEditor ref="monacoEditor" class="monaco-container" @saved="onFileSaved" @save-error="onSaveError" />
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
            <div class="prop-group">
              <div class="prop-group-title">快捷操作</div>
              <button class="quick-btn" @click="showTerminal = !showTerminal"><Terminal :size="15" class="icon-sm" />{{ showTerminal ? '收起终端' : '打开终端' }}</button>
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
    <div v-if="showPathDialog" class="dlg-overlay" @click.self="showPathDialog = false">
      <div class="dlg-box">
        <h3>打开项目文件夹</h3>
        <button class="btn btn-primary browse-btn" :disabled="picking" @click="browseFolder">
          <FolderOpen :size="15" class="icon-sm" />
          {{ picking ? '正在等待选择…' : '浏览系统目录…' }}
        </button>
        <div v-if="quickPaths.length" class="quick-paths">
          <button v-for="qp in quickPaths" :key="qp.path" class="qp-btn" @click="newPathInput = qp.path">
            <FolderOpen :size="11" /> {{ qp.label }}
          </button>
        </div>
        <input v-model="newPathInput" class="dlg-input" placeholder="或输入完整路径" @keydown.enter="applyNewPath" />
        <div class="dlg-actions">
          <button class="dlg-btn primary" @click="applyNewPath">切换</button>
          <button class="dlg-btn" @click="showPathDialog = false">取消</button>
        </div>
        <p v-if="pathDialogError" class="dlg-error">{{ pathDialogError }}</p>
      </div>
    </div>

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
      <div class="ctx-item" @click="revealItem"><FolderOpen :size="13" /> 在资源管理器中显示</div>
      <div class="ctx-sep"></div>
      <div class="ctx-item" @click="renameItem"><Edit2 :size="13" /> 重命名</div>
      <div class="ctx-sep"></div>
      <div class="ctx-item danger" @click="deleteItem"><Trash2 :size="13" /> 删除</div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted, onActivated, onDeactivated, inject, nextTick } from 'vue';
import { API_BASE, authFetch } from '../composables/apiClient.js';
import { Terminal, FolderOpen, Folder, FileCode, FileText, Image as ImageIcon, Database, Edit3, Edit2, Trash2, Crosshair, Activity, Search, RefreshCw } from 'lucide-vue-next';
import MonacoEditor from '../components/MonacoEditor.vue';
import WebTerminal from '../components/WebTerminal.vue';

defineOptions({ name: 'WorkspaceView' });

const toast = inject('toast');
const $confirm = inject('$confirm');

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
    const r = await authFetch(`${API_BASE}/api/workspace/quick_paths`);
    if (r.ok) { const d = await r.json(); quickPaths.value = d.paths || []; }
  } catch (e) { /* 快速路径仅为便捷入口，加载失败不打断对话框 */ }
}

// 系统级目录选择：后端弹原生对话框（PowerShell BrowseForFolder）
async function browseFolder() {
  if (picking.value) return;
  picking.value = true;
  pathDialogError.value = '';
  try {
    const r = await authFetch(`${API_BASE}/api/workspace/pick_folder`, { method: 'POST' });
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

async function handleNewFolder() {
  const name = await showInputDialog('新建文件夹', '输入文件夹名，如 utils', '');
  if (!name) return;
  try {
    const r = await authFetch(`${API_BASE}/api/workspace/mkdir`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    const data = await r.json().catch(() => ({}));
    if (r.ok && data.status === 'ok') { toast(`已创建文件夹 ${name}`, 'success'); loadTree(); }
    else toast(data.detail || '创建文件夹失败', 'error');
  } catch (e) { toast('创建文件夹失败: ' + e.message, 'error'); }
}

async function revealItem() {
  const node = contextMenu.value.node;
  if (!node) return;
  try {
    const r = await authFetch(`${API_BASE}/api/workspace/reveal`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: node.path }),
    });
    if (!r.ok) { const d = await r.json().catch(() => ({})); toast(d.detail || '打开资源管理器失败', 'error'); }
  } catch (e) { toast('打开资源管理器失败: ' + e.message, 'error'); }
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
    const r = await authFetch(`${API_BASE}/api/workspace/file`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, content: '' }),
    });
    const data = await r.json().catch(() => ({}));
    if (r.ok && data.status === 'ok') { toast(`已创建 ${name}`, 'success'); loadTree(); }
    else toast(data.detail || data.message || '创建文件失败', 'error');
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
  running.value = true;
  try {
    const r = await authFetch(`${API_BASE}/api/workspace/run`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code: tab.content || '' }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) { toast(data.detail || `运行失败（HTTP ${r.status}）`, 'error'); return; }
    if (data.success) toast(data.output ? `运行成功：${data.output}`.slice(0, 200) : '运行成功', 'success');
    else toast(data.error || '运行失败', 'error');
  } catch (e) { toast('运行失败: ' + e.message, 'error'); }
  finally { running.value = false; }
}

async function applyNewPath() {
  const path = newPathInput.value.trim();
  if (!path) { pathDialogError.value = '请输入路径'; return; }
  pathDialogError.value = '';
  try {
    const r = await authFetch(`${API_BASE}/api/workspace/path`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
    const data = await r.json();
    if (r.ok && data.status === 'ok') {
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
    const r = await authFetch(`${API_BASE}/api/workspace/path`);
    if (r.ok) { const d = await r.json(); workspacePath.value = d.path || ''; }
  } catch (e) { toast('加载工作路径失败: ' + e.message, 'error') }
}

async function loadTree() {
  try {
    const r = await authFetch(`${API_BASE}/api/workspace/tree`);
    if (r.ok) { const d = await r.json(); fileTree.value = d.tree || []; recomputeFlatList(); }
  } catch (e) { toast('加载文件树失败: ' + e.message, 'error') }
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

function handleTreeClick(node) {
  if (node.type === 'directory') {
    if (expandedDirs.has(node.path)) expandedDirs.delete(node.path);
    else expandedDirs.add(node.path);
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
  if (!node) return;
  const newName = await showInputDialog('重命名', '输入新名称', node.name);
  if (!newName || newName === node.name) return;
  // 新名称沿用原目录：取父路径前缀（兼容 / 与 \ 分隔符）
  const sepIdx = Math.max(node.path.lastIndexOf('/'), node.path.lastIndexOf('\\'));
  const parentPrefix = sepIdx >= 0 ? node.path.slice(0, sepIdx + 1) : '';
  const newPath = parentPrefix + newName;
  try {
    const r = await authFetch(`${API_BASE}/api/workspace/rename`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ old_name: node.path, new_name: newPath }),
    });
    const data = await r.json().catch(() => ({}));
    if (r.ok && data.status === 'ok') {
      syncTabsAfterRename(node.path, data.path || newPath);
      toast(`已重命名为 ${newName}`, 'success');
      loadTree();
    } else toast(data.detail || data.message || '重命名失败', 'error');
  } catch (e) { toast('重命名失败: ' + e.message, 'error'); }
}
async function deleteItem() {
  const node = contextMenu.value.node;
  if (!node) return;
  const ok = await $confirm({ title: '删除确认', message: `确定删除 ${node.name}？`, type: 'danger' });
  if (!ok) return;
  try { const r = await authFetch(`${API_BASE}/api/workspace/delete/${node.path}`, { method: 'DELETE' }); if (r.ok) loadTree(); } catch (e) { toast('删除失败: ' + e.message, 'error') }
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
  loadTree();
  document.addEventListener('keydown', onGlobalKeydown);
});
// 离开页面（被 keep-alive 缓存）时收起终端并卸载快捷键，
// v-if 卸载会触发 WebTerminal 的 onBeforeUnmount 清理（WS / xterm / ResizeObserver）。
onDeactivated(() => {
  showTerminal.value = false;
  quickOpen.visible = false;
  document.removeEventListener('keydown', onGlobalKeydown);
});
onUnmounted(() => {
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
.topbar {
  height: 52px;
  flex: none;
  padding: 0 18px;
  display: flex;
  align-items: center;
  gap: 12px;
  border-bottom: 1px solid var(--border);
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
.quick-btn {
  display: flex;
  align-items: center;
  gap: 7px;
  width: 100%;
  padding: 7px 10px;
  border-radius: 6px;
  border: 0;
  background: transparent;
  color: var(--foreground);
  font-size: 0.82rem;
  cursor: pointer;
  text-align: left;
  transition: background 120ms ease;
  font-family: inherit;
}
.quick-btn:hover { background: var(--muted); }
.quick-btn .icon-sm {
  width: 15px;
  height: 15px;
  color: var(--muted-foreground);
}

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

.quick-paths {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-bottom: 12px;
}
.qp-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  font-size: 12px;
  color: var(--muted-foreground);
  background: var(--muted);
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) * 0.5);
  cursor: pointer;
  transition: background 150ms ease, color 150ms ease, border-color 150ms ease;
  font-family: inherit;
}
.qp-btn:hover {
  background: color-mix(in srgb, var(--primary) 12%, var(--background));
  color: var(--primary);
  border-color: color-mix(in srgb, var(--primary) 30%, var(--border));
}

/* ── 浏览系统目录按钮 ── */
.browse-btn {
  width: 100%;
  justify-content: center;
  margin-bottom: 12px;
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
