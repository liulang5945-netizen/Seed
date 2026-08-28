/**
 * Seed native API facade.
 *
 * Product code should depend on these named operations instead of rebuilding
 * `/api/...` URLs and JSON request options in each view.  The endpoint paths
 * remain explicit here so the OpenAPI contract checker has one client-side
 * source to audit; request payloads stay plain objects and are serialized only
 * at this boundary.
 */
import { API_BASE, authFetch } from './apiClient.js'

export const nativeApiPaths = Object.freeze({
  runtime: Object.freeze({
    bootstrap: '/api/runtime/bootstrap',
    status: '/api/runtime/status',
  }),
  workbench: Object.freeze({
    capabilities: '/api/workbench/capabilities',
    events: '/api/workbench/events',
    files: '/api/workbench/files',
    workspace: '/api/workbench/workspace',
    file: '/api/workbench/file',
    programmingLanguage: '/api/workbench/programming-language',
    preview: '/api/workbench/preview',
    execute: '/api/workbench/execute',
    loopPreflight: '/api/workbench/loop/preflight',
    loopExecute: '/api/workbench/loop/execute',
  }),
  system: Object.freeze({
    quickPaths: '/api/system/quick_paths',
    selectFolder: '/api/system/select_folder',
    selectFile: '/api/system/select_file',
    validatePath: '/api/system/validate_path',
    openFolder: '/api/system/open_folder',
  }),
})

function urlFor(path, query = {}) {
  const search = new URLSearchParams()
  for (const [name, value] of Object.entries(query)) {
    if (value !== undefined && value !== null && value !== '') search.set(name, String(value))
  }
  const queryString = search.toString()
  return `${API_BASE}${path}${queryString ? `?${queryString}` : ''}`
}

function jsonOptions(payload, method = 'POST') {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }
}

async function readJson(path, options) {
  const response = options === undefined
    ? await authFetch(path)
    : await authFetch(path, options)
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = typeof payload.detail === 'string'
      ? payload.detail
      : payload.detail?.error || payload.message
    throw new Error(detail || `Native API 请求失败（HTTP ${response.status}）`)
  }
  return payload
}

function request(path, options) {
  return options === undefined
    ? authFetch(`${API_BASE}${path}`)
    : authFetch(`${API_BASE}${path}`, options)
}

/**
 * @typedef {Object} NativeApiFacade
 * @property {(options?: RequestInit) => Promise<Response>} runtimeBootstrap
 * @property {(options?: RequestInit) => Promise<Response>} runtimeStatus
 * @property {() => Promise<Object>} workbenchCapabilities
 * @property {() => Promise<Object>} workbenchEvents
 * @property {(path?: string) => Promise<Object>} workbenchFiles
 * @property {(payload: Object) => Promise<Object>} setWorkbenchWorkspace
 * @property {(path: string) => Promise<Object>} workbenchFile
 * @property {(path: string, lspLanguageId?: string) => Promise<Object>} programmingLanguage
 * @property {(payload: Object) => Promise<Object>} workbenchPreview
 * @property {(payload: Object) => Promise<Object>} workbenchExecute
 * @property {(payload: Object) => Promise<Object>} loopPreflight
 * @property {(payload: Object) => Promise<Object>} loopExecute
 * @property {() => Promise<Object>} systemQuickPaths
 * @property {(title?: string) => Promise<Response>} systemSelectFolder
 * @property {() => Promise<Response>} systemSelectFile
 * @property {(payload: Object) => Promise<Object>} systemValidatePath
 * @property {(payload: Object) => Promise<Object>} systemOpenFolder
 */

/** @type {NativeApiFacade} */
export const nativeApi = Object.freeze({
  runtimeBootstrap: (options) => request(nativeApiPaths.runtime.bootstrap, options),
  runtimeStatus: (options) => request(nativeApiPaths.runtime.status, options),

  workbenchCapabilities: () => readJson(`${API_BASE}${nativeApiPaths.workbench.capabilities}`),
  workbenchEvents: () => readJson(`${API_BASE}${nativeApiPaths.workbench.events}`),
  workbenchFiles: (path = '.') => readJson(urlFor(nativeApiPaths.workbench.files, { path: path || '.' })),
  setWorkbenchWorkspace: (payload) => readJson(
    `${API_BASE}${nativeApiPaths.workbench.workspace}`,
    jsonOptions(payload),
  ),
  workbenchFile: (path) => readJson(urlFor(nativeApiPaths.workbench.file, { path })),
  programmingLanguage: (path, lspLanguageId = '') => readJson(
    urlFor(nativeApiPaths.workbench.programmingLanguage, { path, lsp_language_id: lspLanguageId }),
  ),
  workbenchPreview: (payload) => readJson(
    `${API_BASE}${nativeApiPaths.workbench.preview}`,
    jsonOptions(payload),
  ),
  workbenchExecute: (payload) => readJson(
    `${API_BASE}${nativeApiPaths.workbench.execute}`,
    jsonOptions(payload),
  ),
  loopPreflight: (payload) => readJson(
    `${API_BASE}${nativeApiPaths.workbench.loopPreflight}`,
    jsonOptions(payload),
  ),
  loopExecute: (payload) => readJson(
    `${API_BASE}${nativeApiPaths.workbench.loopExecute}`,
    jsonOptions(payload),
  ),

  systemQuickPaths: () => readJson(`${API_BASE}${nativeApiPaths.system.quickPaths}`),
  systemSelectFolder: (title = '') => request(
    urlFor(nativeApiPaths.system.selectFolder, { title }),
  ),
  systemSelectFile: () => request(nativeApiPaths.system.selectFile),
  systemValidatePath: (payload) => readJson(
    `${API_BASE}${nativeApiPaths.system.validatePath}`,
    jsonOptions(payload),
  ),
  systemOpenFolder: (payload) => readJson(
    `${API_BASE}${nativeApiPaths.system.openFolder}`,
    jsonOptions(payload),
  ),
})
