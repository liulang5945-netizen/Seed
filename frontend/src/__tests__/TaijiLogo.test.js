import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import TaijiLogo from '../components/TaijiLogo.vue'

// F07: TaijiLogo 组件测试——props 驱动、class 切换、可访问性
describe('TaijiLogo', () => {
  it('渲染 img 标签并设置默认尺寸', () => {
    const wrapper = mount(TaijiLogo)
    const img = wrapper.find('img.taiji-logo')
    expect(img.exists()).toBe(true)
    expect(img.attributes('width')).toBe('40')
    expect(img.attributes('height')).toBe('40')
    expect(img.attributes('alt')).toBe('Seed Taiji')
    expect(img.attributes('role')).toBe('img')
  })

  it('接受自定义 size prop', () => {
    const wrapper = mount(TaijiLogo, { props: { size: 72 } })
    const img = wrapper.find('img')
    expect(img.attributes('width')).toBe('72')
    expect(img.attributes('height')).toBe('72')
  })

  it('thinking=false 时添加 is-idle class', () => {
    const wrapper = mount(TaijiLogo, { props: { thinking: false } })
    expect(wrapper.find('img').classes()).toContain('is-idle')
    expect(wrapper.find('img').classes()).not.toContain('is-thinking')
  })

  it('thinking=true 时添加 is-thinking class', () => {
    const wrapper = mount(TaijiLogo, { props: { thinking: true } })
    expect(wrapper.find('img').classes()).toContain('is-thinking')
    expect(wrapper.find('img').classes()).not.toContain('is-idle')
  })

  it('内联 style 设置宽高为 size px', () => {
    const wrapper = mount(TaijiLogo, { props: { size: 48 } })
    const img = wrapper.find('img')
    expect(img.attributes('style')).toContain('width: 48px')
    expect(img.attributes('style')).toContain('height: 48px')
  })
})
