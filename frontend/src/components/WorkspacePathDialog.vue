<template>
  <div v-if="visible" class="dlg-overlay" @click.self="emit('update:visible', false)">
    <div class="dlg-box">
      <h3>打开项目文件夹</h3>
      <button class="btn btn-primary browse-btn" :disabled="picking" @click="emit('browse')">
        <FolderOpen :size="15" class="icon-sm" />
        {{ picking ? '正在等待选择…' : '浏览系统目录…' }}
      </button>
      <div v-if="quickPaths.length" class="quick-paths">
        <button v-for="quickPath in quickPaths" :key="quickPath.path" class="qp-btn" @click="emit('update:path', quickPath.path)">
          <FolderOpen :size="11" /> {{ quickPath.label }}
        </button>
      </div>
      <input
        :value="path"
        class="dlg-input"
        placeholder="或输入完整路径"
        @input="emit('update:path', $event.target.value)"
        @keydown.enter="emit('apply')"
      />
      <div class="dlg-actions">
        <button class="dlg-btn primary" @click="emit('apply')">切换</button>
        <button class="dlg-btn" @click="emit('update:visible', false)">取消</button>
      </div>
      <p v-if="error" class="dlg-error">{{ error }}</p>
    </div>
  </div>
</template>

<script setup>
import { FolderOpen } from 'lucide-vue-next'

defineOptions({ name: 'WorkspacePathDialog' })

defineProps({
  visible: { type: Boolean, default: false },
  path: { type: String, default: '' },
  quickPaths: { type: Array, default: () => [] },
  picking: { type: Boolean, default: false },
  error: { type: String, default: '' },
})

const emit = defineEmits(['update:visible', 'update:path', 'browse', 'apply'])
</script>

<style scoped>
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
  cursor: pointer;
}
.btn-primary {
  color: var(--primary-foreground);
  background: var(--primary);
}
.browse-btn {
  width: 100%;
  justify-content: center;
  margin-bottom: 12px;
}
</style>
