import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import SettingsAboutPanel from '../components/SettingsAboutPanel.vue'

describe('SettingsAboutPanel', () => {
  it('renders version and native route metadata', () => {
    const wrapper = mount(SettingsAboutPanel, { props: { appVersion: '2.4.1' } })

    expect(wrapper.text()).toContain('v2.4.1')
    expect(wrapper.text()).toContain('seed-native-v1')
    expect(wrapper.text()).toContain('Taiji runtime → native capabilities')
  })

  it('forwards the license entry action without owning modal state', async () => {
    const wrapper = mount(SettingsAboutPanel)

    await wrapper.find('button[aria-label="查看许可"]').trigger('click')

    expect(wrapper.emitted('show-license')).toHaveLength(1)
  })
})
