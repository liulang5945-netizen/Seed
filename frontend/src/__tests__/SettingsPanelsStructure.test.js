import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import SettingsGeneralPanel from '../components/SettingsGeneralPanel.vue'
import SettingsTaijiPanel from '../components/SettingsTaijiPanel.vue'
import SettingsPrivacyPanel from '../components/SettingsPrivacyPanel.vue'
import SettingsAboutPanel from '../components/SettingsAboutPanel.vue'
import SettingsRuntimePanel from '../components/SettingsRuntimePanel.vue'

describe('Settings panels structure contract', () => {
  it.each([
    [SettingsGeneralPanel, '通用设置', { themes: [] }],
    [SettingsTaijiPanel, 'Taiji 运行设置', {}],
    [SettingsPrivacyPanel, '数据与隐私', {}],
    [SettingsAboutPanel, '关于', {}],
    [SettingsRuntimePanel, '运行环境', {}],
  ])('所有设置面板通过共享 section 渲染标题：%s', (component, title, props) => {
    const wrapper = mount(component, { props })

    expect(wrapper.find('section.settings-section > h2').text()).toBe(title)
    expect(wrapper.find('section.settings-section > h2').element.parentElement.classList.contains('settings-section')).toBe(true)
  })
})
