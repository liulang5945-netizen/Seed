/**
 * Native Workbench capability and audit projection.
 *
 * The projection is intentionally read-only.  It is the client-side view of
 * Seed's capability snapshot and event stream; it is not a second tool
 * registry and it does not execute shell commands or file writes.
 */
import { computed, readonly, ref } from 'vue'
import { API_BASE, authFetch } from './apiClient.js'

const capabilities = ref(null)
const events = ref([])
const error = ref('')
const loading = ref(false)
let consumerCount = 0
let eventTimer = null

function capabilityMap() {
  return new Map((capabilities.value?.capabilities || []).map(item => [item.capability_id, item]))
}

function isEnabled(capabilityId) {
  return capabilityMap().get(capabilityId)?.enabled === true
}

async function readJson(path, options) {
  const url = `${API_BASE}${path}`
  const response = options === undefined
    ? await authFetch(url)
    : await authFetch(url, options)
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = typeof payload.detail === 'string'
      ? payload.detail
      : payload.detail?.error || payload.message
    throw new Error(detail || `Workbench 请求失败（HTTP ${response.status}）`)
  }
  return payload
}

async function refreshCapabilities() {
  const payload = await readJson('/api/workbench/capabilities')
  capabilities.value = payload
  error.value = ''
  return payload
}

async function refreshEvents() {
  const payload = await readJson('/api/workbench/events')
  events.value = Array.isArray(payload.events) ? payload.events : []
  error.value = ''
  return events.value
}

async function refresh() {
  loading.value = true
  try {
    await Promise.all([refreshCapabilities(), refreshEvents()])
    return { capabilities: capabilities.value, events: events.value }
  } catch (cause) {
    error.value = cause?.message || 'Workbench 投影不可用'
    throw cause
  } finally {
    loading.value = false
  }
}

async function ensureCapabilities() {
  if (capabilities.value) return capabilities.value
  return refreshCapabilities()
}

async function listDirectory(path = '.') {
  await ensureCapabilities()
  if (!isEnabled('workspace.list')) {
    throw new Error('workspace.list 未接入')
  }
  const query = encodeURIComponent(path || '.')
  const payload = await readJson(`/api/workbench/files?path=${query}`)
  error.value = ''
  return Array.isArray(payload.entries) ? payload.entries : []
}

async function readFile(path) {
  await ensureCapabilities()
  if (!isEnabled('workspace.read')) {
    throw new Error('workspace.read 未接入')
  }
  const query = encodeURIComponent(path)
  const payload = await readJson(`/api/workbench/file?path=${query}`)
  error.value = ''
  return payload
}

function start() {
  consumerCount += 1
  if (eventTimer) return
  refresh().catch(() => {})
  eventTimer = setInterval(() => {
    refreshEvents().catch(() => {})
  }, 2000)
}

function stop() {
  consumerCount = Math.max(0, consumerCount - 1)
  if (consumerCount !== 0 || !eventTimer) return
  clearInterval(eventTimer)
  eventTimer = null
}

const snapshotId = computed(() => capabilities.value?.snapshot_id || '')
const workspaceRoot = computed(() => capabilities.value?.workspace_root || '')
const latestOutcome = computed(() => {
  const outcomeEvents = events.value.filter(event => event.phase === 'outcome')
  return outcomeEvents.length ? outcomeEvents[outcomeEvents.length - 1].payload?.outcome || null : null
})

export function useWorkbenchProjection() {
  return {
    capabilities: readonly(capabilities),
    events: readonly(events),
    error: readonly(error),
    loading: readonly(loading),
    snapshotId,
    workspaceRoot,
    latestOutcome,
    isEnabled,
    refresh,
    refreshCapabilities,
    refreshEvents,
    ensureCapabilities,
    listDirectory,
    readFile,
    start,
    stop,
  }
}
