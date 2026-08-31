import { ref } from 'vue'
import { nativeApi } from './nativeApi.js'

const CLIENT_FORBIDDEN_FIELDS = new Set([
  'parameter_bindings',
  'patch',
  'before_digest',
  'expected_after_digest',
  'action_intent',
  'intent',
])

function assertTaijiOwnedPayload(payload) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new TypeError('Taiji Workbench 请求必须是对象')
  }
  const leaked = Object.keys(payload).filter((key) => CLIENT_FORBIDDEN_FIELDS.has(key))
  if (leaked.length) {
    throw new Error(`Taiji Workbench 请求不得注入 ${leaked.join(', ')}`)
  }
}

/**
 * Product-facing transport for Taiji-owned natural-language Workbench tasks.
 *
 * The client stores only Taiji's returned plan/approval/outcome. It never
 * derives a patch, digest, ActionIntent, or final capability binding locally.
 */
export function useNaturalLanguageWorkbench() {
  const interpretation = ref(null)
  const plan = ref(null)
  const approval = ref(null)
  const execution = ref(null)
  const busy = ref(false)
  const error = ref('')

  async function run(operation) {
    busy.value = true
    error.value = ''
    try {
      return await operation()
    } catch (cause) {
      error.value = cause?.message || 'Taiji Workbench 请求失败'
      throw cause
    } finally {
      busy.value = false
    }
  }

  function planTask(payload) {
    assertTaijiOwnedPayload(payload)
    return run(async () => {
      const result = await nativeApi.chatWorkbenchPlanNaturalLanguage(payload)
      plan.value = result
      approval.value = null
      execution.value = null
      return result
    })
  }

  function interpretTask(payload) {
    assertTaijiOwnedPayload(payload)
    return run(async () => {
      const result = await nativeApi.chatWorkbenchInterpret(payload)
      interpretation.value = result
      plan.value = null
      approval.value = null
      execution.value = null
      return result
    })
  }

  function approveRequest(requestId) {
    if (!plan.value?.plan_id) throw new Error('当前没有可审批的 Taiji Workbench 计划')
    if (!requestId) throw new Error('审批必须绑定 request_id')
    return run(async () => {
      const result = await nativeApi.chatWorkbenchApproveNaturalLanguage({
        plan_id: plan.value.plan_id,
        request_id: requestId,
      })
      approval.value = result
      return result
    })
  }

  function executePlan(approvalTokens = {}) {
    if (!plan.value?.plan_id) throw new Error('当前没有可执行的 Taiji Workbench 计划')
    return run(async () => {
      const result = await nativeApi.chatWorkbenchExecuteNaturalLanguage({
        plan_id: plan.value.plan_id,
        approval_tokens: approvalTokens,
      })
      execution.value = result
      return result
    })
  }

  function reset() {
    interpretation.value = null
    plan.value = null
    approval.value = null
    execution.value = null
    error.value = ''
  }

  return {
    interpretation,
    plan,
    approval,
    execution,
    busy,
    error,
    interpretTask,
    planTask,
    approveRequest,
    executePlan,
    reset,
  }
}
