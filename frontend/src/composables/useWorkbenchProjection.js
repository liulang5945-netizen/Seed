/**
 * Native Workbench capability and audit projection.
 *
 * The projection is the client-side view of Seed's capability snapshot and
 * event stream.  It is not a second tool registry and it never executes shell
 * commands or file writes; the editor language override is the one explicit,
 * reversible UI command exposed here.
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

async function resolveProgrammingLanguage(path, lspLanguageId = '') {
  await ensureCapabilities()
  const query = `path=${encodeURIComponent(path)}${lspLanguageId
    ? `&lsp_language_id=${encodeURIComponent(lspLanguageId)}`
    : ''}`
  const payload = await readJson(`/api/workbench/programming-language?${query}`)
  error.value = ''
  return payload
}

async function setEditorLanguage({
  path,
  programmingLanguageId,
  editorLanguageId,
  userOverride = true,
  clearOverride = false,
}) {
  await ensureCapabilities()
  const payload = await readJson('/api/workbench/execute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      intent_id: `ui:editor.set_language:${Date.now()}`,
      kind: 'editor.set_language',
      parameters: {
        path,
        programming_language_id: programmingLanguageId,
        editor_language_id: editorLanguageId,
        user_override: userOverride,
        clear_override: clearOverride,
      },
      snapshot_id: snapshotId.value,
      confidence: 1,
      tick: 0,
    }),
  })
  error.value = ''
  return payload.outcome?.result || payload
}

async function previewIntent({
  intentId,
  kind,
  parameters = {},
  expectedOutcome = '',
  confidence = 1,
  tick = 0,
}) {
  await ensureCapabilities()
  const payload = await readJson('/api/workbench/preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      intent_id: intentId,
      kind,
      parameters,
      expected_outcome: expectedOutcome,
      confidence,
      tick,
      snapshot_id: snapshotId.value,
    }),
  })
  error.value = ''
  return payload
}

async function executeIntent({
  intentId,
  kind,
  parameters = {},
  expectedOutcome = '',
  confidence = 1,
  tick = 0,
  approvalToken = '',
}) {
  await ensureCapabilities()
  const payload = await readJson('/api/workbench/execute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      intent_id: intentId,
      kind,
      parameters,
      expected_outcome: expectedOutcome,
      confidence,
      tick,
      approval_token: approvalToken,
      snapshot_id: snapshotId.value,
    }),
  })
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
const programmingLanguages = computed(() => capabilities.value?.programming_languages || [])
const programmingLanguageRegistryRevision = computed(
  () => capabilities.value?.programming_language_registry_revision || ''
)
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
    programmingLanguages,
    programmingLanguageRegistryRevision,
    latestOutcome,
    isEnabled,
    refresh,
    refreshCapabilities,
    refreshEvents,
    ensureCapabilities,
    listDirectory,
    readFile,
    resolveProgrammingLanguage,
    setEditorLanguage,
    previewIntent,
    executeIntent,
    start,
    stop,
  }
}
