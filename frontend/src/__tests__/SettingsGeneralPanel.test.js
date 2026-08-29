import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import SettingsGeneralPanel from '../components/SettingsGeneralPanel.vue'

describe('SettingsGeneralPanel', () => {
  it('renders explicit settings values and forwards every change boundary', async () => {
    const wrapper = mount(SettingsGeneralPanel, {
      props: {
        themes: [{ id: 'classic', name: 'Classic', desc: '经典', gradient: '#fff' }],
        currentTheme: 'classic',
        uiLanguage: 'zh-CN',
        timezone: 'Asia/Shanghai',
        uiDensity: 'default',
      },
    })

    expect(wrapper.text()).toContain('通用设置')
    expect(wrapper.find('.theme-preview-card.active').exists()).toBe(true)

    await wrapper.find('[aria-label="默认语言"]').setValue('en')
    await wrapper.find('[aria-label="时区"]').setValue('Asia/Tokyo')
    await wrapper.find('input[value="compact"]').setValue(true)
    await wrapper.find('.theme-preview-card').trigger('click')

    expect(wrapper.emitted('ui-language-change')).toEqual([['en']])
    expect(wrapper.emitted('timezone-change')).toEqual([['Asia/Tokyo']])
    expect(wrapper.emitted('ui-density-change')).toEqual([['compact']])
    expect(wrapper.emitted('theme-change')).toEqual([['classic']])
  })

  it('disables all mutable controls while saving', () => {
    const wrapper = mount(SettingsGeneralPanel, {
      props: {
        saving: true,
        themes: [{ id: 'classic', name: 'Classic', gradient: '#fff' }],
      },
    })

    expect(wrapper.findAll('select:disabled')).toHaveLength(2)
    expect(wrapper.findAll('input:disabled')).toHaveLength(3)
  })
})
