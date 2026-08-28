import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import TrainingOverviewPanel from '../components/TrainingOverviewPanel.vue'

const metrics = {
  epoch: 1,
  total_epochs: 2,
  total_steps: 8,
  eta: 12,
  samples_per_sec: 0.5,
  current_loss: 0.25,
}

describe('TrainingOverviewPanel', () => {
  it('renders the native idle overview without legacy training fields', () => {
    const wrapper = mount(TrainingOverviewPanel, {
      props: { active: true, trainMetrics: metrics, fmtTime: (value) => `${value}s` },
    })

    expect(wrapper.find('section.tab-panel').classes()).toContain('active')
    expect(wrapper.text()).toContain('Seed 原生 byte-stream 训练')
    expect(wrapper.text()).toContain('Train Loss')
    expect(wrapper.text()).not.toContain('学习率')
  })

  it('forwards checkpoint recovery without owning training side effects', async () => {
    const wrapper = mount(TrainingOverviewPanel, {
      props: {
        active: true,
        trainState: 'paused',
        trainProgress: 35,
        trainMetrics: metrics,
        pendingCheckpoints: [{ filename: 'seed.pt', epoch: 1, step: 4, loss: 0.25 }],
        fmtTime: (value) => `${value}s`,
      },
      global: { stubs: { NButton: { template: '<button @click="$emit(\'click\')"><slot /></button>' } } },
    })

    expect(wrapper.text()).toContain('seed.pt')
    await wrapper.find('.ckpt-item button').trigger('click')
    expect(wrapper.emitted('resume')).toHaveLength(1)
  })
})
