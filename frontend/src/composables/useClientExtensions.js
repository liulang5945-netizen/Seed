import { computed, readonly, ref } from 'vue'
import { nativeApi } from './nativeApi.js'

const snapshot = ref(null)
const policy = ref(null)
const lifecycle = ref([])
const dependencyHealth = ref({})
const error = ref('')
const loading = ref(false)

function activeManifests() {
  return Array.isArray(snapshot.value?.manifests) ? snapshot.value.manifests : []
}

const slots = computed(() => {
  const result = new Map()
  for (const manifest of activeManifests()) {
    for (const slot of manifest.slots || []) {
      const entries = result.get(slot) || []
      entries.push(manifest)
      result.set(slot, entries)
    }
  }
  return result
})

function slotManifests(slot) {
  return slots.value.get(String(slot)) || []
}

async function refresh() {
  loading.value = true
  try {
    const payload = await nativeApi.clientExtensions()
    snapshot.value = payload.snapshot || null
    policy.value = payload.policy || null
    lifecycle.value = Array.isArray(payload.lifecycle) ? payload.lifecycle : []
    dependencyHealth.value = payload.dependency_health || {}
    error.value = ''
    return payload
  } catch (cause) {
    error.value = cause?.message || '客户端扩展状态不可用'
    throw cause
  } finally {
    loading.value = false
  }
}

async function prepare(manifests, options = {}) {
  const payload = await nativeApi.clientExtensionsPrepare({
    capability_snapshot_id: options.capabilitySnapshotId || undefined,
    manifests: Array.isArray(manifests) ? manifests : [],
    dependency_health: options.dependencyHealth,
    states: options.states,
  })
  return payload
}

async function commit(preparedId) {
  const payload = await nativeApi.clientExtensionsCommit({ prepared_id: preparedId })
  await refresh()
  return payload
}

async function reportDependency(service, healthy) {
  const payload = await nativeApi.clientExtensionsDependency({ service, healthy })
  await refresh()
  return payload
}

async function rollback(snapshotId) {
  const payload = await nativeApi.clientExtensionsRollback({ snapshot_id: snapshotId })
  await refresh()
  return payload
}

async function beginCall(pluginId) {
  return nativeApi.clientExtensionsBeginCall(pluginId)
}

async function endCall(pluginId) {
  return nativeApi.clientExtensionsEndCall(pluginId)
}

async function retire(pluginId) {
  const payload = await nativeApi.clientExtensionsRetire(pluginId)
  await refresh()
  return payload
}

async function quarantine(pluginId, reason) {
  const payload = await nativeApi.clientExtensionsQuarantine(pluginId, reason)
  await refresh()
  return payload
}

export function useClientExtensions() {
  return {
    snapshot: readonly(snapshot),
    policy: readonly(policy),
    lifecycle: readonly(lifecycle),
    dependencyHealth: readonly(dependencyHealth),
    error: readonly(error),
    loading: readonly(loading),
    activeManifests: computed(activeManifests),
    slots,
    slotManifests,
    refresh,
    prepare,
    commit,
    reportDependency,
    rollback,
    beginCall,
    endCall,
    retire,
    quarantine,
  }
}
