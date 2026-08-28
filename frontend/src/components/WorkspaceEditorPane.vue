<template>
  <div class="panel panel-center">
    <div class="editor-area">
      <MonacoEditor
        ref="monacoEditor"
        class="monaco-container"
        :approval-handler="approvalHandler"
        @saved="emit('saved', $event)"
        @save-error="emit('save-error', $event)"
      />
      <Transition name="term-slide">
        <div v-if="showTerminal" class="ide-terminal" :style="{ height: `${terminalHeight}px` }">
          <div class="resize-row" @mousedown="emit('resize-terminal', $event)"></div>
          <WebTerminal ref="webTerminal" />
        </div>
      </Transition>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import MonacoEditor from './MonacoEditor.vue'
import WebTerminal from './WebTerminal.vue'

defineOptions({ name: 'WorkspaceEditorPane' })

defineProps({
  showTerminal: { type: Boolean, default: false },
  terminalHeight: { type: Number, default: 280 },
  approvalHandler: { type: Function, required: true },
})

const monacoEditor = ref(null)
const webTerminal = ref(null)
const emit = defineEmits(['saved', 'save-error', 'resize-terminal'])

defineExpose({ monacoEditor, webTerminal })
</script>

<style scoped>
.panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
  position: relative;
}
.panel-center { border-right: 1px solid var(--border); }
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
.term-slide-enter-active,
.term-slide-leave-active {
  transition: transform 0.2s ease;
}
.term-slide-enter-from,
.term-slide-leave-to {
  transform: translateY(100%);
}
</style>
