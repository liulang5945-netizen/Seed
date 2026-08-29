import { describe, it, expect, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import RuntimeEvidenceStrip from '../components/RuntimeEvidenceStrip.vue'
import { useRuntimeStore } from '../stores/runtimeStore.js'

describe('RuntimeEvidenceStrip', () => {
  beforeEach(() => setActivePinia(createPinia()))

  it('可见呈现来源、归属、刷新时效和可用性', () => {
    const store = useRuntimeStore()
    const observedAt = Math.floor(Date.now() / 1000)
    store.applyRuntimeStatus({
      timestamp: observedAt,
      health: { state: 'connected', model_loaded: true, is_taiji: true },
      tools: {
        status: 'ok',
        snapshot_id: 'workbench-snapshot-1234',
        revision: 2,
        owner: 'Taiji native Workbench',
        observed_at: observedAt,
        tools: [],
      },
      life: { needs: {} },
      training: { is_training: false },
    })

    const wrapper = mount(RuntimeEvidenceStrip, { props: { context: 'agent' } })

    expect(wrapper.text()).toContain('状态依据')
    expect(wrapper.text()).toContain('Taiji runtime')
    expect(wrapper.text()).toContain('/api/runtime/status.health')
    expect(wrapper.text()).toContain('已加载（0 项）')
    expect(wrapper.text()).toContain('刚刚')
  })

  it('折叠模式默认不占用内容区，展开后才显示审计明细', async () => {
    const store = useRuntimeStore()
    store.applyRuntimeStatus({
      timestamp: Math.floor(Date.now() / 1000),
      health: { state: 'connected', model_loaded: true, is_taiji: true },
      life: { needs: {} },
    })
    const wrapper = mount(RuntimeEvidenceStrip, { props: { context: 'life', collapsible: true } })

    expect(wrapper.find('button.evidence-toggle').exists()).toBe(true)
    expect(wrapper.find('.evidence-grid').exists()).toBe(false)
    expect(wrapper.find('button.evidence-toggle').attributes('aria-expanded')).toBe('false')

    await wrapper.find('button.evidence-toggle').trigger('click')
    expect(wrapper.find('.evidence-grid').exists()).toBe(true)
    expect(wrapper.find('button.evidence-toggle').attributes('aria-expanded')).toBe('true')
  })
})
