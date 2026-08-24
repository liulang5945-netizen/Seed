import { describe, it, expect, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import ConfirmDialog from '../components/ConfirmDialog.vue'

// F04: ConfirmDialog 组件测试——显示/确认/取消/ESC 关闭与 Promise 语义
describe('ConfirmDialog', () => {
  let wrapper

  beforeEach(() => {
    wrapper = mount(ConfirmDialog)
  })

  it('初始状态不渲染对话框', () => {
    expect(wrapper.find('.confirm-overlay').exists()).toBe(false)
  })

  it('show() 渲染标题、消息与自定义按钮文案', async () => {
    wrapper.vm.show({
      title: '删除会话',
      message: '此操作不可恢复',
      type: 'danger',
      confirmText: '删除',
      cancelText: '保留',
    })
    await wrapper.vm.$nextTick()

    expect(wrapper.find('.confirm-overlay').exists()).toBe(true)
    expect(wrapper.find('.confirm-title').text()).toBe('删除会话')
    expect(wrapper.find('.confirm-message').text()).toBe('此操作不可恢复')
    const buttons = wrapper.findAll('.confirm-btn')
    expect(buttons[0].text()).toBe('保留')
    expect(buttons[1].text()).toBe('删除')
    expect(buttons[1].classes()).toContain('danger')
  })

  it('点击确认按钮 → Promise 解析为 true 并关闭', async () => {
    const promise = wrapper.vm.show({ title: '确认' })
    await wrapper.vm.$nextTick()

    await wrapper.find('.confirm-btn.danger, .confirm-btn.primary').trigger('click')
    await expect(promise).resolves.toBe(true)
    expect(wrapper.find('.confirm-overlay').exists()).toBe(false)
  })

  it('点击取消按钮 → Promise 解析为 false 并关闭', async () => {
    const promise = wrapper.vm.show({ title: '确认' })
    await wrapper.vm.$nextTick()

    await wrapper.find('.confirm-btn.cancel').trigger('click')
    await expect(promise).resolves.toBe(false)
    expect(wrapper.find('.confirm-overlay').exists()).toBe(false)
  })

  it('按 Escape 等价于取消', async () => {
    const promise = wrapper.vm.show({ title: '确认' })
    await wrapper.vm.$nextTick()

    await wrapper.find('.confirm-overlay').trigger('keydown.escape')
    await expect(promise).resolves.toBe(false)
  })

  it('点击遮罩空白处等价于取消', async () => {
    const promise = wrapper.vm.show({ title: '确认' })
    await wrapper.vm.$nextTick()

    await wrapper.find('.confirm-overlay').trigger('click')
    await expect(promise).resolves.toBe(false)
  })

  it('未提供选项时使用默认文案', async () => {
    wrapper.vm.show()
    await wrapper.vm.$nextTick()

    expect(wrapper.find('.confirm-title').text()).toBe('确认操作')
    const buttons = wrapper.findAll('.confirm-btn')
    expect(buttons[0].text()).toBe('取消')
    expect(buttons[1].text()).toBe('确定')
  })
})
