import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { ref } from 'vue'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import RecoveryPortfolioAuditPanel from '../components/RecoveryPortfolioAuditPanel.vue'

// 隔离 composable：面板的全部行为都从 projection 的只读结果派生
const projection = {
  snapshotId: ref('snap-a'),
  events: ref([]),
  error: ref(''),
  recoveryContext: ref(null),
  recoveryPortfolio: ref(null),
  refreshRecoveryPortfolioContext: vi.fn(),
  refreshRecoveryPortfolio: vi.fn(),
}

vi.mock('../composables/useWorkbenchProjection.js', () => ({
  useWorkbenchProjection: () => projection,
}))

const portfolioSnapshot = () => ({
  format: 'taiji-recovery-portfolio-v1',
  version: 1,
  status: 'portfolio_snapshot',
  parent_loop_id: 'parent-a',
  snapshot_id: 'snap-a',
  revision: 3,
  current_tick: 5,
  max_branches: 2,
  branch_ttl_ticks: 200,
  last_maintenance_tick: 1,
  selected_branch_id: 'recovery-branch:sel',
  counts: { active: 1, selected: 1, completed: 1, failed: 1, expired: 0, evicted: 1 },
  liveness_due_branch_ids: [],
  branches: [
    {
      branch_id: 'recovery-branch:selected',
      loop_id: 'loop:selected',
      parent_loop_id: 'parent-a',
      capability_id: 'workspace.read',
      source_evidence_id: 'evidence:selected',
      source_after_state_digest: 'digest:selected',
      status: 'selected',
      budget_limit: 1.0,
      budget_units: 0.5,
      completed_steps: 2,
      frontier_affordance_ids: ['affordance:next'],
      created_tick: 1,
      last_touched_tick: 3,
      expires_at_tick: 201,
      terminal_reason: '',
    },
  ],
  evicted_branches: [
    {
      branch_id: 'branch:evicted',
      loop_id: 'loop:evicted',
      source_evidence_id: 'evidence:evicted',
      source_after_state_digest: 'digest:evicted',
      status: 'evicted',
      evicted_tick: 40,
      reason: 'capacity_exhausted',
    },
  ],
})

beforeEach(() => {
  projection.recoveryContext.value = null
  projection.recoveryPortfolio.value = null
  projection.error.value = ''
  projection.events.value = []
  projection.refreshRecoveryPortfolioContext.mockReset()
  projection.refreshRecoveryPortfolio.mockReset()
  // context 永远返回「当前」上下文，与服务端一致
  projection.refreshRecoveryPortfolioContext.mockImplementation(
    () => Promise.resolve(projection.recoveryContext.value),
  )
})

const mountPanel = () => mount(RecoveryPortfolioAuditPanel)

describe('RecoveryPortfolioAuditPanel', () => {
  it('无 portfolio 上下文时显示空态，且不读取 portfolio', async () => {
    projection.recoveryContext.value = { status: 'portfolio_context', has_portfolio: false }
    projection.refreshRecoveryPortfolioContext.mockResolvedValue(projection.recoveryContext.value)

    const wrapper = mountPanel()
    await flushPromises()

    expect(wrapper.text()).toContain('无 recovery portfolio')
    expect(projection.refreshRecoveryPortfolio).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('渲染五种生命周期/预算/lineage/墓碑，绝不渲染 parameters 与 evidence', async () => {
    const ctx = {
      status: 'portfolio_context',
      has_portfolio: true,
      parent_loop_id: 'parent-a',
      snapshot_id: 'snap-a',
      revision: 3,
      selected_branch_id: '',
    }
    projection.recoveryContext.value = ctx
    projection.refreshRecoveryPortfolioContext.mockResolvedValue(ctx)
    projection.refreshRecoveryPortfolio.mockResolvedValue(portfolioSnapshot())

    const wrapper = mountPanel()
    await flushPromises()
    await flushPromises()

    expect(wrapper.text()).toContain('已选中')
    expect(wrapper.text()).toContain('capacity_exhausted')
    expect(wrapper.text()).toContain('#3')
    expect(wrapper.text()).toContain('5 / 2')
    expect(wrapper.text()).toContain('证据')
    // 只读合同：不展示任何可复用执行输入
    expect(wrapper.text()).not.toContain('parameters')
    expect(wrapper.text()).not.toContain('evidence":')
    expect(wrapper.text()).not.toContain('frontier_affordance_ids":')

    expect(projection.refreshRecoveryPortfolio).toHaveBeenCalledWith('parent-a', 3)
    wrapper.unmount()
  })

  it('revision 推移（stale）时保留最后一个已验证快照并标记过期', async () => {
    const ctx = {
      status: 'portfolio_context',
      has_portfolio: true,
      parent_loop_id: 'parent-a',
      snapshot_id: 'snap-a',
      revision: 3,
      selected_branch_id: '',
    }
    projection.recoveryContext.value = ctx
    projection.refreshRecoveryPortfolioContext.mockResolvedValue(ctx)
    projection.refreshRecoveryPortfolio.mockResolvedValueOnce(portfolioSnapshot())

    const wrapper = mountPanel()
    await flushPromises()
    await flushPromises()
    expect(wrapper.text()).toContain('已选中')
    expect(wrapper.find('[data-audit-stale]').exists()).toBe(false)

    // 服务端已把 revision 推移到 4，下一次事件脉冲触发带旧期望值的重取 → 被拒为 stale
    projection.refreshRecoveryPortfolio.mockRejectedValueOnce(
      new Error('portfolio_revision_stale'),
    )
    // 兜底：任何额外重取同样被拒为 stale，绝不返回 undefined 干扰断言
    projection.refreshRecoveryPortfolio.mockRejectedValue(
      new Error('portfolio_revision_stale'),
    )
    projection.recoveryContext.value = { ...ctx, revision: 4 }
    projection.events.value = [{ sequence: 1, phase: 'outcome' }]
    await flushPromises()

    // 旧快照仍在，且显示过期标记
    expect(wrapper.text()).toContain('已选中')
    expect(wrapper.find('[data-audit-stale]').exists()).toBe(true)
    expect(wrapper.find('[data-audit-error]').exists()).toBe(false)
    wrapper.unmount()
  })

  it('parent loop 切换时清除上一循环状态，绝不跨循环串读', async () => {
    const ctxA = {
      status: 'portfolio_context', has_portfolio: true,
      parent_loop_id: 'parent-a', snapshot_id: 'snap-a', revision: 1,
      selected_branch_id: '',
    }
    projection.recoveryContext.value = ctxA
    projection.refreshRecoveryPortfolioContext.mockResolvedValue(ctxA)
    const snapshotA = portfolioSnapshot()
    snapshotA.branches[0].branch_id = 'recovery-branch:from-a'
    projection.refreshRecoveryPortfolio.mockResolvedValue(snapshotA)

    const wrapper = mountPanel()
    await flushPromises()
    expect(wrapper.text()).toContain('from-a')

    // 切换 parent loop：旧绑定必须被丢弃
    projection.refreshRecoveryPortfolioContext.mockResolvedValue({
      ...ctxA,
      parent_loop_id: 'parent-b',
    })
    projection.refreshRecoveryPortfolio.mockResolvedValue({
      ...portfolioSnapshot(),
      parent_loop_id: 'parent-b',
      revision: 1,
      branches: [{ ...portfolioSnapshot().branches[0], branch_id: 'recovery-branch:from-b' }],
      evicted_branches: [],
    })
    projection.recoveryContext.value = { ...ctxA, parent_loop_id: 'parent-b' }
    await flushPromises()
    await flushPromises()

    expect(wrapper.text()).toContain('from-b')
    expect(wrapper.text()).not.toContain('from-a')
    wrapper.unmount()
  })

  it('只读：整个生命周期只调用 context/portfolio 两个只读方法', async () => {
    const ctx = {
      status: 'portfolio_context', has_portfolio: true,
      parent_loop_id: 'parent-a', snapshot_id: 'snap-a', revision: 3,
      selected_branch_id: '',
    }
    projection.recoveryContext.value = ctx
    projection.refreshRecoveryPortfolioContext.mockResolvedValue(ctx)
    projection.refreshRecoveryPortfolio.mockResolvedValue(portfolioSnapshot())

    const wrapper = mountPanel()
    await flushPromises()
    await flushPromises()

    expect(projection.refreshRecoveryPortfolioContext).toHaveBeenCalled()
    expect(projection.refreshRecoveryPortfolio).toHaveBeenCalledWith('parent-a', 3)
    // 只读合同（静态）：面板源码不得引用任何 mutation 投影方法
    const panelPath = path.join(process.cwd(), 'src/components/RecoveryPortfolioAuditPanel.vue')
    const source = await readFile(panelPath, 'utf-8')
    for (const forbidden of ['preflightLoop', 'executeLoop', 'previewIntent', 'executeIntent']) {
      expect(source).not.toContain(forbidden)
    }
    wrapper.unmount()
  })
})