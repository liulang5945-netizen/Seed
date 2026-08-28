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
  auth: Object.freeze({
    login: '/api/auth/login',
    changePassword: '/api/auth/change_password',
    status: '/api/auth/status',
    enable: '/api/auth/enable',
    disable: '/api/auth/disable',
    refresh: '/api/auth/refresh',
  }),
  settings: Object.freeze({
    all: '/api/settings',
    runtime: '/api/settings/runtime',
  }),
  chat: Object.freeze({
    sessions: '/api/chat/sessions',
    history: '/api/chat/history/{session_id}',
    stream: '/api/chat/stream',
    upload: '/api/chat/upload',
  }),
  training: Object.freeze({
    uploadDataset: '/api/train/upload_dataset',
    files: '/api/train/files',
    file: '/api/train/file/{filename}',
    preview: '/api/train/preview/{filename}',
    pause: '/api/train/pause',
    resume: '/api/train/resume',
    stop: '/api/train/stop',
    checkpoints: '/api/train/checkpoints',
    resumeCheckpoint: '/api/train/resume_checkpoint',
    native: '/api/train/native',
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
    health: '/api/health',
    version: '/api/system/version',
    reset: '/api/system/reset',
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

function resourcePath(template, value, parameter) {
  return template.replace(`{${parameter}}`, encodeURIComponent(String(value)))
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
 * @property {(payload: Object) => Promise<Object>} authLogin
 * @property {(payload: Object) => Promise<Object>} authChangePassword
 * @property {(payload: Object) => Promise<Object>} authEnable
 * @property {() => Promise<Object>} authDisable
 * @property {() => Promise<Object>} authStatus
 * @property {() => Promise<Object>} authRefresh
 * @property {() => Promise<Object>} settingsGet
 * @property {(payload: Object) => Promise<Object>} settingsSave
 * @property {() => Promise<Object>} runtimeSettingsGet
 * @property {(payload: Object) => Promise<Object>} runtimeSettingsSave
 * @property {() => Promise<Array>} chatSessions
 * @property {(sessionId: string) => Promise<Object>} chatHistory
 * @property {(sessionId: string, payload: Object) => Promise<Object>} saveChatHistory
 * @property {(sessionId: string) => Promise<Object>} deleteChatHistory
 * @property {(payload: Object, options?: RequestInit) => Promise<Response>} chatStream
 * @property {(formData: FormData, options?: RequestInit) => Promise<Response>} chatUpload
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
 * @property {() => Promise<Object>} trainingFiles
 * @property {(filename: string) => Promise<Object>} trainingPreview
 * @property {(filename: string) => Promise<Object>} deleteTrainingFile
 * @property {() => Promise<Object>} pauseTraining
 * @property {() => Promise<Object>} resumeTraining
 * @property {() => Promise<Object>} stopTraining
 * @property {() => Promise<Object>} trainingCheckpoints
 * @property {(payload: Object, options?: RequestInit) => Promise<Response>} resumeTrainingCheckpoint
 * @property {(payload: Object, options?: RequestInit) => Promise<Response>} nativeTraining
 * @property {(formData: FormData, options?: RequestInit) => Promise<Response>} uploadTrainingDataset
 * @property {() => Promise<Object>} systemQuickPaths
 * @property {(title?: string) => Promise<Response>} systemSelectFolder
 * @property {() => Promise<Response>} systemSelectFile
 * @property {(payload: Object) => Promise<Object>} systemValidatePath
 * @property {(payload: Object) => Promise<Object>} systemOpenFolder
 * @property {() => Promise<Object>} systemHealth
 * @property {() => Promise<Object>} systemVersion
 * @property {(payload: Object) => Promise<Object>} systemReset
 */

/** @type {NativeApiFacade} */
export const nativeApi = Object.freeze({
  runtimeBootstrap: (options) => request(nativeApiPaths.runtime.bootstrap, options),
  runtimeStatus: (options) => request(nativeApiPaths.runtime.status, options),

  authLogin: (payload) => readJson(
    `${API_BASE}${nativeApiPaths.auth.login}`,
    jsonOptions(payload),
  ),
  authChangePassword: (payload) => readJson(
    `${API_BASE}${nativeApiPaths.auth.changePassword}`,
    jsonOptions(payload),
  ),
  authEnable: (payload) => readJson(
    `${API_BASE}${nativeApiPaths.auth.enable}`,
    jsonOptions(payload),
  ),
  authDisable: () => readJson(
    `${API_BASE}${nativeApiPaths.auth.disable}`,
    jsonOptions({}),
  ),
  authStatus: () => readJson(`${API_BASE}${nativeApiPaths.auth.status}`),
  authRefresh: () => readJson(
    `${API_BASE}${nativeApiPaths.auth.refresh}`,
    jsonOptions({}),
  ),

  settingsGet: () => readJson(`${API_BASE}${nativeApiPaths.settings.all}`),
  settingsSave: (payload) => readJson(
    `${API_BASE}${nativeApiPaths.settings.all}`,
    jsonOptions(payload),
  ),
  runtimeSettingsGet: () => readJson(`${API_BASE}${nativeApiPaths.settings.runtime}`),
  runtimeSettingsSave: (payload) => readJson(
    `${API_BASE}${nativeApiPaths.settings.runtime}`,
    jsonOptions(payload),
  ),

  chatSessions: () => readJson(`${API_BASE}${nativeApiPaths.chat.sessions}`),
  chatHistory: (sessionId) => readJson(
    `${API_BASE}${resourcePath(nativeApiPaths.chat.history, sessionId, 'session_id')}`,
  ),
  saveChatHistory: (sessionId, payload) => readJson(
    `${API_BASE}${resourcePath(nativeApiPaths.chat.history, sessionId, 'session_id')}`,
    jsonOptions(payload),
  ),
  deleteChatHistory: (sessionId) => readJson(
    `${API_BASE}${resourcePath(nativeApiPaths.chat.history, sessionId, 'session_id')}`,
    { method: 'DELETE' },
  ),
  chatStream: (payload, options = {}) => request(nativeApiPaths.chat.stream, {
    ...jsonOptions(payload),
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    retries: 0,
  }),
  chatUpload: (formData, options = {}) => request(nativeApiPaths.chat.upload, {
    method: 'POST',
    ...options,
    body: formData,
  }),

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

  trainingFiles: () => readJson(`${API_BASE}${nativeApiPaths.training.files}`),
  trainingPreview: (filename) => readJson(
    `${API_BASE}${resourcePath(nativeApiPaths.training.preview, filename, 'filename')}`,
  ),
  deleteTrainingFile: (filename) => readJson(
    `${API_BASE}${resourcePath(nativeApiPaths.training.file, filename, 'filename')}`,
    { method: 'DELETE' },
  ),
  pauseTraining: () => readJson(`${API_BASE}${nativeApiPaths.training.pause}`, jsonOptions({})),
  resumeTraining: () => readJson(`${API_BASE}${nativeApiPaths.training.resume}`, jsonOptions({})),
  stopTraining: () => readJson(`${API_BASE}${nativeApiPaths.training.stop}`, jsonOptions({})),
  trainingCheckpoints: () => readJson(`${API_BASE}${nativeApiPaths.training.checkpoints}`),
  resumeTrainingCheckpoint: (payload, options = {}) => request(nativeApiPaths.training.resumeCheckpoint, {
    ...jsonOptions(payload),
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
  }),
  nativeTraining: (payload, options = {}) => request(nativeApiPaths.training.native, {
    ...jsonOptions(payload),
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
  }),
  uploadTrainingDataset: (formData, options = {}) => request(nativeApiPaths.training.uploadDataset, {
    method: 'POST',
    ...options,
    body: formData,
  }),

  systemQuickPaths: () => readJson(`${API_BASE}${nativeApiPaths.system.quickPaths}`),
  systemHealth: () => readJson(`${API_BASE}${nativeApiPaths.system.health}`),
  systemVersion: () => readJson(`${API_BASE}${nativeApiPaths.system.version}`),
  systemReset: (payload) => readJson(
    `${API_BASE}${nativeApiPaths.system.reset}`,
    jsonOptions(payload),
  ),
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
