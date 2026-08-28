import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import TrainingLogPanel from '../components/TrainingLogPanel.vue'

describe('TrainingLogPanel', () => {
  it('renders the live log and forwards the clear action', async () => {
    const wrapper = mount(TrainingLogPanel, {
      props: {
        active: true,
        trainLog: 'epoch=1 loss=0.42',
        modelLabel: 'Seed Taiji',
      },
      global: {
        stubs: {
          NButton: { template: '<button @click="$emit(\'click\')"><slot /></button>' },
        },
      },
    })

    expect(wrapper.find('section.tab-panel').classes()).toContain('active')
    expect(wrapper.text()).toContain('training.log — Seed Taiji')
    expect(wrapper.find('.log-body').text()).toContain('epoch=1 loss=0.42')

    await wrapper.find('.log-clear-btn').trigger('click')
    expect(wrapper.emitted('clear')).toHaveLength(1)
  })

  it('keeps the empty state when no log has arrived', () => {
    const wrapper = mount(TrainingLogPanel, {
      props: { active: false, trainLog: '' },
    })

    expect(wrapper.find('.log-empty').exists()).toBe(true)
    expect(wrapper.text()).toContain('训练开始后将实时输出日志')
    expect(wrapper.find('.log-body').exists()).toBe(false)
  })
})
