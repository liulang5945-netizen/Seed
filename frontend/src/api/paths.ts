/**
 * Seed 原生 API 路径表 —— 前端唯一真源。
 *
 * 为什么上移到 `src/api/` 而不是留在 facade 里：`Object.freeze` 不保留字符串
 * 字面量类型（每个属性都会被拓宽为 `string`），只有 TS 文件里的 `const` 类型
 * 参数才能让每个端点保持字面量类型，从而被 `contract.ts` 编译期钉在 OpenAPI
 * 冻结快照上。JS 文件里做不到这件事——这是本次迁移唯一需要挪动位置的原因。
 *
 * 运行时语义与迁移前逐字节一致：仍逐组 `Object.freeze`，仍以同名导出。
 */

const frozenGroup = <const T extends Record<string, string>>(entries: T): Readonly<T> =>
  Object.freeze(entries)

export const nativeApiPaths = Object.freeze({
  runtime: frozenGroup({
    bootstrap: '/api/runtime/bootstrap',
    status: '/api/runtime/status',
  }),
  auth: frozenGroup({
    login: '/api/auth/login',
    changePassword: '/api/auth/change_password',
    status: '/api/auth/status',
    enable: '/api/auth/enable',
    disable: '/api/auth/disable',
    refresh: '/api/auth/refresh',
  }),
  settings: frozenGroup({
    all: '/api/settings',
    runtime: '/api/settings/runtime',
  }),
  chat: frozenGroup({
    sessions: '/api/chat/sessions',
    history: '/api/chat/history/{session_id}',
    stream: '/api/chat/stream',
    workbenchStream: '/api/chat/workbench/stream',
    workbenchInterpret: '/api/chat/workbench/interpret',
    workbenchNaturalLanguagePlan: '/api/chat/workbench/natural-language/plan',
    workbenchNaturalLanguageApprove: '/api/chat/workbench/natural-language/approve',
    workbenchNaturalLanguageExecute: '/api/chat/workbench/natural-language/execute',
    upload: '/api/chat/upload',
  }),
  training: frozenGroup({
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
  workbench: frozenGroup({
    capabilities: '/api/workbench/capabilities',
    events: '/api/workbench/events',
    files: '/api/workbench/files',
    workspace: '/api/workbench/workspace',
    file: '/api/workbench/file',
    programmingLanguage: '/api/workbench/programming-language',
    preview: '/api/workbench/preview',
    execute: '/api/workbench/execute',
    taijiAdmit: '/api/workbench/taiji/admit',
    taijiExecute: '/api/workbench/taiji/execute',
    taijiProject: '/api/workbench/taiji/project',
    taijiReproject: '/api/workbench/taiji/reproject',
    taijiRecoveryPortfolio: '/api/workbench/taiji/recovery-branch/portfolio',
    taijiRecoveryPortfolioContext: '/api/workbench/taiji/recovery-branch/context',
    loopPreflight: '/api/workbench/loop/preflight',
    loopExecute: '/api/workbench/loop/execute',
  }),
  clientExtensions: frozenGroup({
    status: '/api/client-extensions',
    prepare: '/api/client-extensions/prepare',
    commit: '/api/client-extensions/commit',
    dependency: '/api/client-extensions/dependency',
    rollback: '/api/client-extensions/rollback',
    beginCall: '/api/client-extensions/{plugin_id}/call/begin',
    endCall: '/api/client-extensions/{plugin_id}/call/end',
    retire: '/api/client-extensions/{plugin_id}/retire',
    quarantine: '/api/client-extensions/{plugin_id}/quarantine',
  }),
  system: frozenGroup({
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
