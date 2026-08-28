import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import TrainingControlBar from '../components/TrainingControlBar.vue'

const t = (key) => ({
  pause_training: '暂停训练',
  resume_training: '恢复训练',
  stop_training: '停止训练',
}[key] || key)

const mountControlBar = (trainState) => mount(TrainingControlBar, {
  props: { trainState, t },
  global: {
    stubs: {
      NButton: { template: '<button @click="$emit(\'click\')"><slot /></button>' },
    },
  },
})

describe('TrainingControlBar', () => {
  it('shows pause and stop actions while running', async () => {
    const wrapper = mountControlBar('running')

    expect(wrapper.find('.train-ctrl-bar').exists()).toBe(true)
    expect(wrapper.text()).toContain('暂停训练')
    expect(wrapper.text()).toContain('停止训练')
    expect(wrapper.text()).not.toContain('恢复训练')

    const buttons = wrapper.findAll('button')
    await buttons[0].trigger('click')
    await buttons[1].trigger('click')
    expect(wrapper.emitted('pause')).toHaveLength(1)
    expect(wrapper.emitted('stop')).toHaveLength(1)
  })

  it('switches to resume while paused and hides itself when idle', async () => {
    const wrapper = mountControlBar('paused')

    expect(wrapper.text()).toContain('恢复训练')
    expect(wrapper.text()).not.toContain('暂停训练')
    await wrapper.findAll('button')[0].trigger('click')
    expect(wrapper.emitted('resume')).toHaveLength(1)

    await wrapper.setProps({ trainState: 'idle' })
    expect(wrapper.find('.train-ctrl-bar').exists()).toBe(false)
  })
})
