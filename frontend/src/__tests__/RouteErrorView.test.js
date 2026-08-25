import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { createRouter, createMemoryHistory } from 'vue-router'
import RouteErrorView from '../components/RouteErrorView.vue'

// F07: RouteErrorView 组件测试——错误消息渲染、按钮交互、默认文案
describe('RouteErrorView', () => {
  const mockRouter = (path = '/') =>
    createRouter({
      history: createMemoryHistory(),
      routes: [{ path, component: { render: () => null } }],
    })

  it('渲染错误面板主结构', () => {
    const router = mockRouter()
    const wrapper = mount(RouteErrorView, { global: { plugins: [router] } })
    expect(wrapper.find('.route-error-view').exists()).toBe(true)
    expect(wrapper.find('.route-error-panel').exists()).toBe(true)
    expect(wrapper.find('h2').text()).toBe('页面加载失败')
  })

  it('无 message prop 时显示默认文案', () => {
    const router = mockRouter()
    const wrapper = mount(RouteErrorView, { global: { plugins: [router] } })
    expect(wrapper.find('.route-error-panel p').text()).toContain('当前页面模块没有正确加载')
  })

  it('传入自定义 message 时覆盖默认文案', () => {
    const router = mockRouter()
    const wrapper = mount(RouteErrorView, {
      props: { message: '模块加载超时' },
      global: { plugins: [router] },
    })
    expect(wrapper.find('.route-error-panel p').text()).toBe('模块加载超时')
  })

  it('渲染"回到对话"和"刷新"两个按钮', () => {
    const router = mockRouter()
    const wrapper = mount(RouteErrorView, { global: { plugins: [router] } })
    const buttons = wrapper.findAll('button')
    expect(buttons.length).toBe(2)
    expect(buttons[0].text()).toContain('回到对话')
    expect(buttons[1].text()).toContain('刷新')
  })

  it('点击"回到对话"触发路由跳转到 /', async () => {
    const router = mockRouter()
    const pushSpy = vi.spyOn(router, 'push')
    const wrapper = mount(RouteErrorView, { global: { plugins: [router] } })
    await wrapper.find('.primary-btn').trigger('click')
    expect(pushSpy).toHaveBeenCalledWith('/')
  })
})
