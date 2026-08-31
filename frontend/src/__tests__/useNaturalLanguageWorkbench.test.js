import { beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  planNaturalLanguage: vi.fn(),
  approveNaturalLanguage: vi.fn(),
  executeNaturalLanguage: vi.fn(),
  interpretNaturalLanguage: vi.fn(),
}))

const { planNaturalLanguage, approveNaturalLanguage, executeNaturalLanguage, interpretNaturalLanguage } = mocks

vi.mock('../composables/nativeApi.js', () => ({
  nativeApi: {
    chatWorkbenchPlanNaturalLanguage: mocks.planNaturalLanguage,
    chatWorkbenchApproveNaturalLanguage: mocks.approveNaturalLanguage,
    chatWorkbenchExecuteNaturalLanguage: mocks.executeNaturalLanguage,
    chatWorkbenchInterpret: mocks.interpretNaturalLanguage,
  },
}))

import { useNaturalLanguageWorkbench } from '../composables/useNaturalLanguageWorkbench.js'

beforeEach(() => {
  planNaturalLanguage.mockReset()
  approveNaturalLanguage.mockReset()
  executeNaturalLanguage.mockReset()
  interpretNaturalLanguage.mockReset()
})

describe('useNaturalLanguageWorkbench', () => {
  it('keeps plan, approval and execution state on the Taiji protocol', async () => {
    planNaturalLanguage.mockResolvedValueOnce({ plan_id: 'plan-1', status: 'needs_approval' })
    approveNaturalLanguage.mockResolvedValueOnce({ approval_token: 'token-1' })
    executeNaturalLanguage.mockResolvedValueOnce({ status: 'completed' })
    const task = useNaturalLanguageWorkbench()

    await task.planTask({
      prompt: '把文件中的 Seed 改成 Taiji',
      semantic_evidence: { semantic_steps: [] },
      snapshot_id: 'snapshot-1',
      loop_id: 'loop-1',
    })
    await task.approveRequest('request-1')
    await task.executePlan({ 'request-1': 'token-1' })

    expect(planNaturalLanguage).toHaveBeenCalledWith(expect.objectContaining({ loop_id: 'loop-1' }))
    expect(approveNaturalLanguage).toHaveBeenCalledWith({ plan_id: 'plan-1', request_id: 'request-1' })
    expect(executeNaturalLanguage).toHaveBeenCalledWith({
      plan_id: 'plan-1',
      approval_tokens: { 'request-1': 'token-1' },
    })
    expect(task.execution.value).toEqual({ status: 'completed' })
    expect(task.busy.value).toBe(false)
  })

  it('rejects client-side patch, digest and intent injection', async () => {
    const task = useNaturalLanguageWorkbench()

    expect(() => task.planTask({
      prompt: 'edit',
      semantic_evidence: {},
      snapshot_id: 'snapshot-1',
      loop_id: 'loop-1',
      patch: {},
    })).toThrow('不得注入 patch')
    expect(planNaturalLanguage).not.toHaveBeenCalled()
  })

  it('keeps goal intake separate from plan and execution ownership', async () => {
    interpretNaturalLanguage.mockResolvedValueOnce({
      interpretation: { status: 'candidate' },
      goal: { description: '查看工作区文件' },
      execution: { status: 'not_planned', side_effects: false },
    })
    const task = useNaturalLanguageWorkbench()
    await task.interpretTask({ prompt: '查看 README.md', history: [] })

    expect(interpretNaturalLanguage).toHaveBeenCalledWith({ prompt: '查看 README.md', history: [] })
    expect(task.interpretation.value.goal.description).toBe('查看工作区文件')
    expect(task.plan.value).toBe(null)
    expect(task.execution.value).toBe(null)
  })
})
