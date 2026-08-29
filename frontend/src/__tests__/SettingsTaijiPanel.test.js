import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import SettingsTaijiPanel from '../components/SettingsTaijiPanel.vue'

describe('SettingsTaijiPanel', () => {
  it('renders Taiji values and forwards normalized change events', async () => {
    const wrapper = mount(SettingsTaijiPanel, {
      props: {
        activationThreshold: 0.72,
        responseTimeoutMs: 100,
        autoConsolidation: true,
        sleepMode: false,
      },
    })

    expect(wrapper.text()).toContain('Taiji 运行设置')
    expect(wrapper.find('[aria-label="局部激活阈值"]').element.value).toBe('0.72')

    await wrapper.find('[aria-label="局部激活阈值"]').setValue('0.85')
    await wrapper.find('[aria-label="响应超时"]').setValue('250')
    await wrapper.find('[aria-label="自动巩固开关"] input').setValue(false)
    await wrapper.find('[aria-label="睡眠模式开关"] input').setValue(true)

    expect(wrapper.emitted('activation-threshold-change')).toEqual([[0.85]])
    expect(wrapper.emitted('response-timeout-change')).toEqual([[250]])
    expect(wrapper.emitted('auto-consolidation-change')).toEqual([[false]])
    expect(wrapper.emitted('sleep-mode-change')).toEqual([[true]])
  })

  it('disables every mutable control while saving', () => {
    const wrapper = mount(SettingsTaijiPanel, { props: { saving: true } })

    expect(wrapper.findAll('input:disabled')).toHaveLength(4)
  })
})
