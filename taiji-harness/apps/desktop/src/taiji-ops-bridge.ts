/**
 * Typed desktop/client bridge for Taiji N36/N37/N38 ops telemetry & live dashboard.
 */

export interface TaijiOpsOverview {
  schema: string
  scope: string
  boundaries: Record<string, unknown>
  mainline_closed: string[]
  fsm_states: string[]
  fsm_transitions: Record<string, string[]>
  n38_pretraining: N38TrainingStatus
}

export interface N38TrainingStatus {
  schema?: string
  status: string
  preset?: string
  params?: number
  pid?: number
  opt_step?: number
  tokens_seen?: number
  tokens_per_sec?: number
  elapsed_sec?: number
  max_hours?: number
  progress_pct?: number
  train_loss?: number
  val_loss?: number
  val_ppl?: number
  best_val_loss?: number
  lr?: number
  sample_generation?: string
  updated_utc?: string
}

export interface TaijiChangeClassification {
  change_kind: string
  tier: string
  route: string
  allowed_in_shadow: boolean
  reopen_research_required: boolean
  impacted_stages: string[]
}

export interface NativeGrowthPolicy {
  auto_grow_enabled: boolean
  max_params: number
  max_layers: number
  max_group_units: number
  max_fast_heads: number
  max_retention_drop: number
  min_gain: number
}

export type NativeGrowthPolicyPatch = Partial<NativeGrowthPolicy>

export interface NativeBackboneOpsStatus {
  schema: string
  architecture: string
  non_transformer: boolean
  status: Record<string, unknown>
  preflight: Record<string, unknown> | null
  growth_policy: NativeGrowthPolicy
  baseline_transformer: N38TrainingStatus & { role?: string }
}

const DEFAULT_BACKEND_URL = 'http://127.0.0.1:8000'

export function getTaijiOpsDashboardUrl(baseUrl = DEFAULT_BACKEND_URL): string {
  return `${baseUrl.replace(/\/+$/, '')}/taiji-ops`
}

export async function fetchTaijiOpsOverview(
  baseUrl = DEFAULT_BACKEND_URL,
  signal?: AbortSignal,
): Promise<TaijiOpsOverview> {
  const init: RequestInit | undefined = signal ? { signal } : undefined
  const res = await fetch(`${baseUrl.replace(/\/+$/, '')}/api/taiji/ops/overview`, init)
  if (!res.ok) {
    throw new Error(`fetchTaijiOpsOverview failed: HTTP ${res.status}`)
  }
  return (await res.json()) as TaijiOpsOverview
}

export async function fetchN38TrainingStatus(
  baseUrl = DEFAULT_BACKEND_URL,
  signal?: AbortSignal,
): Promise<{ status: N38TrainingStatus; recent_metrics: Record<string, unknown>[] }> {
  const init: RequestInit | undefined = signal ? { signal } : undefined
  const res = await fetch(`${baseUrl.replace(/\/+$/, '')}/api/taiji/ops/n38-training`, init)
  if (!res.ok) {
    throw new Error(`fetchN38TrainingStatus failed: HTTP ${res.status}`)
  }
  return (await res.json()) as {
    status: N38TrainingStatus
    recent_metrics: Record<string, unknown>[]
  }
}

export async function classifyTaijiChange(
  kind: string,
  baseUrl = DEFAULT_BACKEND_URL,
  signal?: AbortSignal,
): Promise<TaijiChangeClassification> {
  const init: RequestInit = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind }),
    ...(signal ? { signal } : {}),
  }
  const res = await fetch(`${baseUrl.replace(/\/+$/, '')}/api/taiji/ops/classify-change`, init)
  if (!res.ok) {
    throw new Error(`classifyTaijiChange failed: HTTP ${res.status}`)
  }
  return (await res.json()) as TaijiChangeClassification
}

export async function fetchNativeBackboneStatus(
  baseUrl = DEFAULT_BACKEND_URL,
  signal?: AbortSignal,
): Promise<NativeBackboneOpsStatus> {
  const init: RequestInit | undefined = signal ? { signal } : undefined
  const res = await fetch(`${baseUrl.replace(/\/+$/, '')}/api/taiji/ops/native-backbone`, init)
  if (!res.ok) {
    throw new Error(`fetchNativeBackboneStatus failed: HTTP ${res.status}`)
  }
  return (await res.json()) as NativeBackboneOpsStatus
}

export async function updateNativeGrowthPolicy(
  patch: NativeGrowthPolicyPatch,
  baseUrl = DEFAULT_BACKEND_URL,
  signal?: AbortSignal,
): Promise<{ status: string; growth_policy: NativeGrowthPolicy }> {
  const init: RequestInit = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
    ...(signal ? { signal } : {}),
  }
  const res = await fetch(`${baseUrl.replace(/\/+$/, '')}/api/taiji/ops/growth-policy`, init)
  if (!res.ok) {
    throw new Error(`updateNativeGrowthPolicy failed: HTTP ${res.status}`)
  }
  return (await res.json()) as { status: string; growth_policy: NativeGrowthPolicy }
}
