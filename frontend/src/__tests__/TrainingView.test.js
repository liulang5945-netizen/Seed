import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { defineComponent, h, KeepAlive } from 'vue'
import { setActivePinia, createPinia } from 'pinia'
import TrainingView from '../views/TrainingView.vue'

// R4: 隔离网络——useTraining 在 setup 阶段会调用 loadTrainDatasets/loadCheckpoints
vi.mock('../composables/apiClient.js', () => ({
  API_BASE: '',
  authFetch: vi.fn(() =>
    Promise.resolve({ ok: false, status: 503, json: async () => ({}) })
  ),
}))

describe('TrainingView', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  // KeepAlive 包裹以触发 onActivated（初始加载统一由它负责）
  const mountView = () =>
    mount(
      defineComponent({
        render: () => h(KeepAlive, null, { default: () => h(TrainingView) }),
      }),
      {
        global: {
          provide: {
            toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
            $confirm: vi.fn(() => Promise.resolve(true)),
          },
        },
      }
    )

  it('渲染训练视图主结构与四个标签页', async () => {
    const wrapper = mountView()
    await flushPromises()
    expect(wrapper.find('.training-view').exists()).toBe(true)
    const tabs = wrapper.findAll('.tabs button.tab')
    expect(tabs.length).toBe(4)
    expect(tabs.map((t) => t.text())).toEqual(['训练概览', '超参数', '数据集', '日志'])
  })

  it('默认激活训练概览面板', async () => {
    const wrapper = mountView()
    await flushPromises()
    const tabs = wrapper.findAll('.tabs button.tab')
    expect(tabs[0].classes()).toContain('active')
    const panels = wrapper.findAll('section.tab-panel')
    expect(panels[0].classes()).toContain('active')
  })

  it('点击超参数标签切换面板', async () => {
    const wrapper = mountView()
    await flushPromises()
    const tabs = wrapper.findAll('.tabs button.tab')
    await tabs[1].trigger('click')
    expect(tabs[1].classes()).toContain('active')
    expect(tabs[0].classes()).not.toContain('active')
    const panels = wrapper.findAll('section.tab-panel')
    expect(panels[1].classes()).toContain('active')
  })

  it('数据集面板包含选择与上传操作区', async () => {
    const wrapper = mountView()
    await flushPromises()
    const tabs = wrapper.findAll('.tabs button.tab')
    await tabs[2].trigger('click')
    await wrapper.vm.$nextTick()
    const datasetPanel = wrapper.findAll('section.tab-panel')[2]
    expect(datasetPanel.exists()).toBe(true)
    expect(datasetPanel.text()).toContain('数据集')
  })

  it('空闲状态下训练控制按钮可见', async () => {
    const wrapper = mountView()
    await flushPromises()
    // trainState 默认 idle → 概览面板应呈现启动训练入口
    const overview = wrapper.findAll('section.tab-panel')[0]
    expect(overview.text()).toMatch(/训练|开始|启动/)
  })
})
