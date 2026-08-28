import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import SettingsRuntimePanel from '../components/SettingsRuntimePanel.vue'

describe('SettingsRuntimePanel', () => {
  it('renders runtime status and emits the updated toggle value', async () => {
    const wrapper = mount(SettingsRuntimePanel, {
      props: {
        modelValue: false,
        runtimeStatusText: 'Seed 原生运行时激活中',
      },
    })

    expect(wrapper.text()).toContain('Seed 原生运行时激活中')
    const toggle = wrapper.find('input[type="checkbox"]')
    await toggle.setValue(true)

    expect(wrapper.emitted('update:modelValue')).toEqual([[true]])
    expect(wrapper.emitted('change')).toHaveLength(1)
  })
})
