import { describe, it, expect, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import AgentConfigView from '../views/AgentConfigView.vue'

describe('AgentConfigView（Taiji 原生能力）', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('只呈现原生能力 registry，不呈现旧 Agent/MCP 管理入口', () => {
    const wrapper = mount(AgentConfigView, {
      global: {
        provide: { toast: () => {} },
      },
    })

    const text = wrapper.text()
    expect(wrapper.find('#agent-name').element.value).toBe('Taiji 原生运行时')
    expect(text).toContain('原生能力')
    expect(text).not.toContain('MCP')
    expect(text).not.toContain('市场')
    expect(text).not.toContain('Agent')
    expect(wrapper.find('.tabs').exists()).toBe(false)
  })
})
