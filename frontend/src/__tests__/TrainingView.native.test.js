import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { defineComponent, h, KeepAlive } from 'vue'
import { setActivePinia, createPinia } from 'pinia'
import TrainingView from '../views/TrainingView.vue'
import { isTaijiModel } from '../composables/useTraining.js'

// 模拟 /api/runtime/status 返回 health.is_seed=true（Seed 原生运行时），
// 其余训练相关接口返回空/失败，隔离网络。
vi.mock('../composables/apiClient.js', () => ({
  API_BASE: '',
  authFetch: vi.fn((url) => {
    if (url.includes('/api/runtime/status')) {
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({ health: { is_seed: true, model_name: 'seed-beta' } }),
      })
    }
    return Promise.resolve({ ok: false, status: 503, json: async () => ({}) })
  }),
}))

describe('TrainingView（Seed 原生运行时）', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    // 模块级状态在测试间共享，显式复位避免串扰
    isTaijiModel.value = false
  })

  // KeepAlive 包裹以触发 onActivated（detectTaijiModel 等初始加载统一由它负责）
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

  it('挂载时 detectTaijiModel 将 isTaijiModel 置为 true', async () => {
    const wrapper = mountView()
    await flushPromises()
    expect(isTaijiModel.value).toBe(true)
    wrapper.unmount()
  })

  it('超参区渲染原生字段（参数预算/最大训练字节数/随机种子），不渲染学习率等 Legacy 字段', async () => {
    const wrapper = mountView()
    await flushPromises()
    const hpPanel = wrapper.findAll('section.tab-panel')[1]
    expect(hpPanel.exists()).toBe(true)
    const text = hpPanel.text()
    // Seed 原生字段存在
    expect(text).toContain('参数预算')
    expect(text).toContain('最大训练字节数')
    expect(text).toContain('训练设备')
    expect(text).toContain('随机种子')
    // Legacy Transformer/LoRA 字段不再渲染
    expect(text).not.toContain('学习率')
    expect(text).not.toContain('批大小')
    expect(text).not.toContain('Epoch 数')
    expect(text).not.toContain('预热步数')
    expect(text).not.toContain('权重衰减')
    expect(text).not.toContain('梯度累积')
    expect(text).not.toContain('最大序列长度')
    expect(text).not.toContain('LoRA rank')
    // 副标题固定为原生说明
    expect(text).toContain('Seed 原生 · 参数预算驱动')
    // 检查点说明为只读文案，不再暴露 save_steps/keep_checkpoints
    expect(text).toContain('检查点按固定周期自动落盘')
    expect(text).not.toContain('保存步频')
    wrapper.unmount()
  })

  it('训练设备为下拉选择而非自由文本输入', async () => {
    const wrapper = mountView()
    await flushPromises()
    const hpPanel = wrapper.findAll('section.tab-panel')[1]
    // n-select 渲染为 .n-select 容器（设备字段已从自由文本输入改为下拉）
    expect(hpPanel.find('.n-select').exists()).toBe(true)
    wrapper.unmount()
  })

  it('概览页不再渲染静态学习率曲线与虚假指标卡', async () => {
    const wrapper = mountView()
    await flushPromises()
    const overview = wrapper.findAll('section.tab-panel')[0]
    expect(overview.find('svg.chart-svg').exists()).toBe(false)
    const text = overview.text()
    expect(text).not.toContain('Eval Loss')
    expect(text).not.toContain('Accuracy')
    expect(text).not.toContain('cosine decay')
    // 真实指标仍保留
    expect(text).toContain('Train Loss')
    expect(text).toContain('吞吐')
    expect(text).toContain('剩余时间')
    wrapper.unmount()
  })

  it('训练页面不再呈现旧模型发布或量化入口', async () => {
    const wrapper = mountView()
    await flushPromises()
    const text = wrapper.text()
    expect(text).not.toContain('GGUF')
    expect(text).not.toContain('发布模型')
    expect(text).not.toContain('LoRA')
    wrapper.unmount()
  })
})
