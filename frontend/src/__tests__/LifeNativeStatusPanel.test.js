import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import LifeNativeStatusPanel from '../components/LifeNativeStatusPanel.vue'

describe('LifeNativeStatusPanel', () => {
  it('renders runtime evidence from explicit native status props', () => {
    const wrapper = mount(LifeNativeStatusPanel, {
      props: {
        connectionStatus: '已连接',
        modelName: 'Seed native',
        languageProviderState: 'ready',
        languageProviderBackend: 'qwen-local',
        toolCount: 6,
        workbenchDetail: 'native capability snapshot',
        hasNeedsData: true,
        healthMessage: 'runtime ready',
      },
    })

    expect(wrapper.find('.native-taiji-status').exists()).toBe(true)
    expect(wrapper.text()).toContain('Seed native')
    expect(wrapper.text()).toContain('qwen-local')
    expect(wrapper.text()).toContain('6 项能力')
    expect(wrapper.text()).toContain('已上报')
    expect(wrapper.text()).toContain('runtime ready')
  })

  it('marks missing needs as not reported instead of inventing values', () => {
    const wrapper = mount(LifeNativeStatusPanel, { props: { hasNeedsData: false } })

    expect(wrapper.text()).toContain('未上报')
    expect(wrapper.text()).toContain('未提供 provider artifact')
    expect(wrapper.text()).not.toContain('神经元数量：')
  })
})
