<template>
  <section
    v-if="entries.length"
    class="client-extension-slot"
    :data-extension-slot="slotName"
    :aria-label="ariaLabel"
  >
    <article
      v-for="manifest in entries"
      :key="entryKey(manifest)"
      class="client-extension-mount"
      :data-plugin-id="manifest.plugin_id"
      :data-extension-state="stateFor(manifest)"
    >
      <slot
        name="extension"
        :manifest="manifest"
        :state="stateFor(manifest)"
        :snapshot-id="snapshotId"
      >
        <div class="client-extension-fallback">
          <div class="client-extension-mark" aria-hidden="true">◌</div>
          <div class="client-extension-copy">
            <strong>{{ labelFor(manifest) }}</strong>
            <span>{{ versionFor(manifest) }}</span>
          </div>
          <span class="client-extension-state">{{ stateLabel(stateFor(manifest)) }}</span>
          <p v-if="descriptionFor(manifest)">{{ descriptionFor(manifest) }}</p>
        </div>
      </slot>
    </article>
  </section>
</template>

<script setup>
import { computed, inject } from 'vue'

const props = defineProps({
  slotName: { type: String, required: true },
  ariaLabel: { type: String, default: '客户端扩展' },
})

const clientExtensions = inject('clientExtensions', null)

const entries = computed(() => {
  const result = clientExtensions?.slotManifests?.(props.slotName)
  return Array.isArray(result) ? result : []
})

const snapshotId = computed(() => clientExtensions?.snapshot?.value?.snapshot_id || '')

const lifecycleState = computed(() => {
  const records = Array.isArray(clientExtensions?.lifecycle?.value)
    ? clientExtensions.lifecycle.value
    : []
  const states = new Map()
  for (const record of records) {
    if (record?.plugin_id && record?.state) states.set(record.plugin_id, record.state)
  }
  return states
})

function entryKey(manifest) {
  return `${manifest?.plugin_id || 'extension'}:${manifest?.plugin_digest || manifest?.version || 'unknown'}`
}

function metadataValue(manifest, key) {
  const value = manifest?.metadata?.[key]
  return typeof value === 'string' && value.trim() ? value.trim() : ''
}

function labelFor(manifest) {
  return metadataValue(manifest, 'label') || metadataValue(manifest, 'title') || manifest?.plugin_id || '客户端扩展'
}

function versionFor(manifest) {
  const version = manifest?.plugin_version || manifest?.version
  return version ? `v${version}` : '声明式扩展'
}

function descriptionFor(manifest) {
  return metadataValue(manifest, 'description')
}

function stateFor(manifest) {
  return lifecycleState.value.get(manifest?.plugin_id) || 'active'
}

function stateLabel(state) {
  return {
    active: '已挂载',
    dependency_lost: '依赖中断',
    dependency_recovered: '已恢复',
    failed: '挂载失败',
    quarantined: '已隔离',
    draining: '正在卸载',
    retired: '已卸载',
    rolled_back: '已回滚',
  }[state] || '已准备'
}
</script>

<style scoped>
.client-extension-slot {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px 12px;
}

.client-extension-mount {
  min-width: 0;
}

.client-extension-fallback {
  display: flex;
  align-items: center;
  gap: 9px;
  min-height: 34px;
  padding: 8px 10px;
  border: 1px solid color-mix(in srgb, var(--border) 76%, transparent);
  border-radius: 10px;
  background: color-mix(in srgb, var(--card) 86%, var(--bg-muted));
}

.client-extension-mark {
  display: grid;
  width: 22px;
  height: 22px;
  flex: 0 0 22px;
  place-items: center;
  border-radius: 7px;
  color: var(--primary);
  background: color-mix(in srgb, var(--primary) 12%, transparent);
  font-size: 17px;
  line-height: 1;
}

.client-extension-copy {
  display: flex;
  min-width: 0;
  flex: 1;
  align-items: baseline;
  gap: 7px;
}

.client-extension-copy strong {
  overflow: hidden;
  color: var(--text);
  font-size: 0.78rem;
  font-weight: 650;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.client-extension-copy span,
.client-extension-state {
  color: var(--text-muted);
  font-size: 0.68rem;
  white-space: nowrap;
}

.client-extension-state {
  padding: 3px 7px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--primary) 9%, transparent);
}

.client-extension-fallback p {
  display: none;
}
</style>
