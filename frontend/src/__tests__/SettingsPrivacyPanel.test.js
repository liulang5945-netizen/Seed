import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import SettingsPrivacyPanel from '../components/SettingsPrivacyPanel.vue'

describe('SettingsPrivacyPanel', () => {
  it('renders values and forwards privacy actions', async () => {
    const wrapper = mount(SettingsPrivacyPanel, {
      props: { chatRetentionDays: '90', chatAutoCleanup: true },
    })

    expect(wrapper.text()).toContain('数据与隐私')
    await wrapper.find('[aria-label="对话保留"]').setValue('180')
    await wrapper.find('[aria-label="自动清理开关"] input').setValue(false)
    await wrapper.find('button.btn-outline').trigger('click')
    await wrapper.find('button.btn-destructive').trigger('click')

    expect(wrapper.emitted('retention-change')).toEqual([['180']])
    expect(wrapper.emitted('auto-cleanup-change')).toEqual([[false]])
    expect(wrapper.emitted('export-data')).toHaveLength(1)
    expect(wrapper.emitted('reset-seed')).toHaveLength(1)
  })

  it('disables controls according to saving, exporting and resetting states', () => {
    const wrapper = mount(SettingsPrivacyPanel, {
      props: { saving: true, exporting: true, resetting: true },
    })

    expect(wrapper.find('select').element.disabled).toBe(true)
    expect(wrapper.find('[aria-label="自动清理开关"] input').element.disabled).toBe(true)
    expect(wrapper.find('button.btn-outline').element.disabled).toBe(true)
    expect(wrapper.find('button.btn-destructive').element.disabled).toBe(true)
  })
})
