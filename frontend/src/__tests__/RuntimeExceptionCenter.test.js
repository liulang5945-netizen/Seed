import { describe, it, expect, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import RuntimeExceptionCenter from '../components/RuntimeExceptionCenter.vue'
import { useRuntimeStore } from '../stores/runtimeStore.js'

// F04: RuntimeExceptionCenter 组件测试——异常卡片渲染、过滤与关闭
describe('RuntimeExceptionCenter', () => {
  let store

  beforeEach(() => {
    setActivePinia(createPinia())
    store = useRuntimeStore()
  })

  const mountCenter = () =>
    mount(RuntimeExceptionCenter, { global: { plugins: [] } })

  it('无异常时不渲染任何内容', () => {
    const wrapper = mountCenter()
    expect(wrapper.find('.exception-center').exists()).toBe(false)
  })

  it('渲染异常卡片（标题与消息）', async () => {
    store.addException('error', '运行时断开', { message: '连接丢失' })
    const wrapper = mountCenter()
    await wrapper.vm.$nextTick()

    const card = wrapper.find('.exception-card')
    expect(card.exists()).toBe(true)
    expect(card.classes()).toContain('exception-danger')
    expect(wrapper.find('.exception-title').text()).toBe('运行时断开')
    expect(wrapper.find('.exception-message').text()).toContain('连接丢失')
  })

  it('最多显示 2 条异常', async () => {
    store.addException('error', '异常 A', { message: 'a' })
    store.addException('warning', '异常 B', { message: 'b' })
    store.addException('info', '异常 C', { message: 'c' })
    const wrapper = mountCenter()
    await wrapper.vm.$nextTick()

    expect(wrapper.findAll('.exception-card').length).toBe(2)
  })

  it('过滤终端相关异常（终端不可用不展示）', async () => {
    store.addException('warning', '终端不可用', { message: 'xterm 未就绪' })
    store.addException('error', '运行时断开', { message: '连接丢失' })
    const wrapper = mountCenter()
    await wrapper.vm.$nextTick()

    const titles = wrapper.findAll('.exception-title').map((n) => n.text())
    expect(titles).not.toContain('终端不可用')
    expect(titles).toContain('运行时断开')
  })

  it('点击关闭按钮移除对应异常', async () => {
    store.addException('error', '运行时断开', { message: '连接丢失' })
    const wrapper = mountCenter()
    await wrapper.vm.$nextTick()

    await wrapper.find('.exception-dismiss').trigger('click')
    await wrapper.vm.$nextTick()

    expect(store.exceptions.length).toBe(0)
    expect(wrapper.find('.exception-center').exists()).toBe(false)
  })
})
