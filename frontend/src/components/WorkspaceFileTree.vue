<template>
  <div class="panel panel-left">
    <div class="panel-header">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
      <span class="panel-header-spacer"></span>
      <button class="icon-btn" title="新建文件" @click="emit('new-file')">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
      </button>
      <button class="icon-btn" title="刷新文件树" @click="emit('refresh')">
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
          @click="emit('select-node', node)"
          @contextmenu.prevent="emit('context-node', $event, node)"
        >
          <component :is="node.type === 'directory' ? (expandedDirs.has(node.path) ? FolderOpen : Folder) : getFileIcon(node.name)" :size="14" class="tree-icon" />
          <span class="tree-label">{{ node.name }}</span>
        </div>
      </template>
    </div>
    <div class="resize-col" @mousedown="emit('resize', $event)"></div>
  </div>
</template>

<script setup>
import { FolderOpen, Folder, RefreshCw } from 'lucide-vue-next'

defineOptions({ name: 'WorkspaceFileTree' })

defineProps({
  fileTree: { type: Array, default: () => [] },
  flatList: { type: Array, default: () => [] },
  expandedDirs: { type: Object, default: () => new Set() },
  getFileIcon: { type: Function, required: true },
})

const emit = defineEmits(['new-file', 'refresh', 'select-node', 'context-node', 'resize'])
</script>

<style scoped>
.panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
  position: relative;
}
.panel-left { border-right: 1px solid var(--border); }
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
}
.icon-btn:hover { background: var(--muted); color: var(--foreground); }
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
@container workspace-view (max-width: 700px) {
  .panel-left { display: none; }
}
@media (max-width: 880px) {
  .panel-left { display: none; }
}
</style>
