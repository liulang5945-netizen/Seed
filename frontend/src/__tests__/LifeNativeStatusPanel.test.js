import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import LifeNativeStatusPanel from '../components/LifeNativeStatusPanel.vue'

const NEED_ROWS = [
  { key: 'hunger', label: '饥饿 · 知识摄取', value: null, state: 'none' },
  { key: 'fatigue', label: '疲劳 · 睡眠需求', value: 84.75, state: 'alert' },
  { key: 'curiosity', label: '好奇 · 探索驱动', value: 42.5, state: 'watch' },
  { key: 'stress', label: '压力 · 错误负担', value: 3.25, state: 'calm' },
  { key: 'boredom', label: '无聊 · 活动需求', value: null, state: 'none' },
]

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
        needRows: NEED_ROWS,
        healthMessage: 'runtime ready',
      },
    })

    expect(wrapper.find('.native-taiji-status').exists()).toBe(true)
    expect(wrapper.text()).toContain('Seed native')
    expect(wrapper.text()).toContain('qwen-local')
    expect(wrapper.text()).toContain('6 项能力')
    expect(wrapper.text()).toContain('3 维已上报')
    expect(wrapper.text()).toContain('runtime ready')
  })

  it('renders one bar per measured dimension and skips unmeasured ones', () => {
    const wrapper = mount(LifeNativeStatusPanel, { props: { needRows: NEED_ROWS } })

    const rows = wrapper.findAll('.native-needs-list li')
    expect(rows).toHaveLength(3)
    expect(rows[0].text()).toContain('84.8')
    expect(rows[0].classes()).toContain('state-alert')
    expect(rows[0].find('.nn-fill').attributes('style')).toContain('width: 84.75%')

    // Dimensions the runtime never measured are named, never filled with a value.
    expect(wrapper.find('.native-needs-foot').text()).toContain('饥饿')
    expect(wrapper.find('.native-needs-foot').text()).toContain('无聊')
  })

  it('marks missing needs as not reported instead of inventing values', () => {
    const wrapper = mount(LifeNativeStatusPanel, { props: { needRows: [] } })

    expect(wrapper.text()).toContain('未上报')
    expect(wrapper.text()).toContain('未提供 provider artifact')
    expect(wrapper.find('.native-needs').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('神经元数量：')
  })

  it('never renders a bar when every dimension is unmeasured', () => {
    const wrapper = mount(LifeNativeStatusPanel, {
      props: { needRows: NEED_ROWS.map((row) => ({ ...row, value: null, state: 'none' })) },
    })

    expect(wrapper.text()).toContain('未上报')
    expect(wrapper.findAll('.native-needs-list li')).toHaveLength(0)
  })
})
