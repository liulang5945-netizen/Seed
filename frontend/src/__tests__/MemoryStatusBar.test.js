import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import MemoryStatusBar from '../components/MemoryStatusBar.vue'
import { useRuntimeStore } from '../stores/runtimeStore.js'

// F04: MemoryStatusBar 组件测试——内存环渲染、百分比换算与告警事件
describe('MemoryStatusBar', () => {
  let store

  beforeEach(() => {
    setActivePinia(createPinia())
    store = useRuntimeStore()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  const mockMemory = (over) => {
    store.refreshMemory = vi.fn().mockResolvedValue({
      status: 'ok',
      level: 1,
      available_pct: 0.62,
      available_gb: 9.9,
      level_emoji: '🟡',
      level_desc: '中等',
      trend: 'stable',
      ...over,
    })
  }

  it('加载中显示 level-loading 与 0%', () => {
    store.refreshMemory = vi.fn().mockReturnValue(new Promise(() => {})) // 永不 resolve
    const wrapper = mount(MemoryStatusBar)
    expect(wrapper.find('.memory-ring').classes()).toContain('level-loading')
    expect(wrapper.find('.ring-text').text()).toBe('0%')
    wrapper.unmount()
  })

  it('按可用比例换算已用百分比并设置级别类', async () => {
    mockMemory()
    const wrapper = mount(MemoryStatusBar)
    await flushPromises()

    // used = round((1 - 0.62) * 100) = 38
    expect(wrapper.find('.ring-text').text()).toBe('38%')
    expect(wrapper.find('.memory-ring').classes()).toContain('level-1')
    expect(wrapper.attributes('title')).toContain('中等')
    wrapper.unmount()
  })

  it('level >= 3 时触发 memory-warning 事件', async () => {
    mockMemory({ level: 3, available_pct: 0.08, level_desc: '告急', level_emoji: '🔴' })
    const wrapper = mount(MemoryStatusBar)
    await flushPromises()

    const emitted = wrapper.emitted('memory-warning')
    expect(emitted).toBeTruthy()
    expect(emitted[0][0].level).toBe(3)
    wrapper.unmount()
  })

  it('level < 3 时不触发告警', async () => {
    mockMemory({ level: 2 })
    const wrapper = mount(MemoryStatusBar)
    await flushPromises()

    expect(wrapper.emitted('memory-warning')).toBeUndefined()
    wrapper.unmount()
  })

  it('refreshMemory 返回 null 时保持加载态不崩溃', async () => {
    store.refreshMemory = vi.fn().mockResolvedValue(null)
    const wrapper = mount(MemoryStatusBar)
    await flushPromises()

    expect(wrapper.find('.memory-ring').classes()).toContain('level-loading')
    wrapper.unmount()
  })
})
