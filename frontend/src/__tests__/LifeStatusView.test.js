import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { defineComponent, h, KeepAlive } from 'vue'
import { setActivePinia, createPinia } from 'pinia'
import LifeStatusView from '../views/LifeStatusView.vue'
import { useRuntimeStore } from '../stores/runtimeStore.js'
import { fmtTime } from '../composables/useTraining.js'
import { authFetch } from '../composables/apiClient.js'

// 隔离网络——所有请求经 mock 路由，不发真实请求
vi.mock('../composables/apiClient.js', () => ({
  API_BASE: '',
  authFetch: vi.fn(),
}))

const jsonResponse = (data, ok = true, status = 200) => ({
  ok,
  status,
  json: async () => data,
})

const lifePostCalls = () =>
  authFetch.mock.calls.filter(
    ([u, o]) => u.startsWith('/api/life/') && (o?.method || 'GET').toUpperCase() === 'POST'
  )

beforeEach(() => {
  setActivePinia(createPinia())
  authFetch.mockReset()
  authFetch.mockImplementation((url, options = {}) => {
    const method = (options.method || 'GET').toUpperCase()
    if (url.startsWith('/api/life/') && method === 'POST') {
      return Promise.resolve(jsonResponse({ success: true, message: '模拟完成' }))
    }
    // /api/runtime/status 等轮询请求统一返回空负载
    return Promise.resolve(jsonResponse({}))
  })
})

// KeepAlive 包裹以触发 onActivated（轮询与首次刷新统一由它负责），
// 与 TrainingView/WorkspaceView 测试保持一致
const mountView = ({ toast = vi.fn() } = {}) =>
  mount(
    defineComponent({
      render: () => h(KeepAlive, null, { default: () => h(LifeStatusView) }),
    }),
    {
      global: {
        provide: { toast },
      },
    }
  )

describe('LifeStatusView', () => {
  it('不出现任何硬编码假数据，缺失数据标注"暂无数据"', async () => {
    const wrapper = mountView()
    await flushPromises()
    const text = wrapper.text()
    // R5 前的假 KPI / 假事件流特征值一律不得出现
    expect(text).not.toContain('1247')
    expect(text).not.toContain('0.873')
    expect(text).not.toContain('0.72')
    expect(text).not.toContain('N-0842')
    expect(text).not.toContain('共振峰值检测')
    // 内存余量无真实来源时明确标注
    expect(text).toContain('暂无数据')
  })

  it('KPI 与需求明细展示 runtimeStore 中的真实生命数据', async () => {
    const store = useRuntimeStore()
    store.life = {
      is_running: true,
      total_interactions: 42,
      uptime_seconds: 3661,
      needs: { hunger: 66, fatigue: 12, curiosity: 88, stress: 5, boredom: 20 },
    }
    const wrapper = mountView()
    await flushPromises()
    const text = wrapper.text()
    expect(text).toContain('42')
    expect(text).toContain(fmtTime(3661))
    expect(text).toContain('运行中')
    // 需求明细基于真实 needs 渲染
    expect(text).toContain('88')
    expect(text).toContain('66')
    expect(text).toContain('偏高') // curiosity=88 > 70
  })

  it('事件流无数据时显示空态文案而非硬编码演示事件', async () => {
    const wrapper = mountView()
    await flushPromises()
    expect(wrapper.text()).toContain('暂无生命事件')
    expect(wrapper.findAll('.event-item').length).toBe(0)
  })

  it('四个生命活动按钮存在但不会调用历史生命接口', async () => {
    const wrapper = mountView()
    await flushPromises()
    const actions = ['喂养', '睡眠', '玩耍', '进化']
    for (const label of actions) {
      const btn = wrapper.findAll('button').find((b) => b.text().includes(label))
      expect(btn, `按钮「${label}」应存在`).toBeTruthy()
      await btn.trigger('click')
      await flushPromises()
    }
    expect(lifePostCalls().length).toBe(0)
    expect(wrapper.text()).toContain('Taiji 原生动作器尚未接入生命活动能力')
  })

  it('操作未接入时事件流记录诚实的能力边界', async () => {
    const wrapper = mountView()
    await flushPromises()
    const btn = wrapper.findAll('button').find((b) => b.text().includes('喂养'))
    await btn.trigger('click')
    await flushPromises()
    expect(wrapper.text()).not.toContain('暂无生命事件')
    expect(wrapper.find('.event-item').text()).toContain('Taiji 原生动作器尚未接入生命活动能力')
  })

  it('原生运行时点击操作按钮给出提示且不发起 /api/life 请求', async () => {
    const store = useRuntimeStore()
    store.health = { ...store.health, isTaiji: true }
    const toastFn = vi.fn()
    const wrapper = mountView({ toast: toastFn })
    await flushPromises()
    const before = lifePostCalls().length
    const btn = wrapper.findAll('button').find((b) => b.text().includes('喂养'))
    await btn.trigger('click')
    await flushPromises()
    expect(lifePostCalls().length).toBe(before)
    expect(toastFn).toHaveBeenCalledWith(
      expect.stringContaining('原生动作器尚未接入'),
      'info'
    )
  })

  it('原生运行时把实测 needs 渲染成驱动条，未测维度不编造数值', async () => {
    const store = useRuntimeStore()
    store.health = { ...store.health, isTaiji: true }
    // 后端 _native_life_section 只上报 Taiji 实测的三维（0-100 换算后）
    store.life = {
      status: 'seed',
      is_running: true,
      needs: { curiosity: 42.5, fatigue: 84.75, stress: 3.25 },
      total_interactions: 0,
      uptime_seconds: 0,
    }
    const wrapper = mountView()
    await flushPromises()

    expect(wrapper.text()).toContain('3 维已上报')
    const rows = wrapper.findAll('.native-needs-list li')
    expect(rows).toHaveLength(3)
    expect(wrapper.find('.native-needs-foot').text()).toContain('饥饿')
    expect(wrapper.text()).not.toContain('50.0')
  })

  it('原生运行时未上报 needs 时不渲染驱动条', async () => {
    const store = useRuntimeStore()
    store.health = { ...store.health, isTaiji: true }
    store.life = { status: 'seed', is_running: false, needs: {} }
    const wrapper = mountView()
    await flushPromises()

    expect(wrapper.text()).toContain('未上报')
    expect(wrapper.find('.native-needs').exists()).toBe(false)
  })

  it('导出报告生成真实数据快照并触发下载', async () => {
    const store = useRuntimeStore()
    store.life = { is_running: true, total_interactions: 7, uptime_seconds: 60, needs: { hunger: 10 } }

    const createObjectURL = vi.fn(() => 'blob:mock-url')
    const revokeObjectURL = vi.fn()
    URL.createObjectURL = createObjectURL
    URL.revokeObjectURL = revokeObjectURL
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(() => {})

    try {
      const wrapper = mountView()
      await flushPromises()
      const btn = wrapper.findAll('button').find((b) => b.text().includes('导出报告'))
      expect(btn).toBeTruthy()
      await btn.trigger('click')
      await flushPromises()

      expect(createObjectURL).toHaveBeenCalledTimes(1)
      const blob = createObjectURL.mock.calls[0][0]
      expect(blob).toBeInstanceOf(Blob)
      expect(blob.type).toBe('application/json')
      if (typeof blob.text === 'function') {
        const snapshot = JSON.parse(await blob.text())
        expect(snapshot.life.total_interactions).toBe(7)
        expect(snapshot.activity_log).toBeInstanceOf(Array)
      }
      expect(clickSpy).toHaveBeenCalled()
      expect(revokeObjectURL).toHaveBeenCalledWith('blob:mock-url')
    } finally {
      clickSpy.mockRestore()
    }
  })
})
