import { describe, it, expect } from 'vitest'
import { mount } from '@vue/test-utils'
import NeedsPentagram from '../components/NeedsPentagram.vue'

// F07: NeedsPentagram 组件测试——五维需求雷达图、props、aria 可访问性
describe('NeedsPentagram', () => {
  const defaultNeeds = { hunger: 0, fatigue: 0, boredom: 0, stress: 0, curiosity: 0 }

  it('渲染 SVG 五边形雷达图', () => {
    const wrapper = mount(NeedsPentagram, { props: { needs: defaultNeeds } })
    expect(wrapper.find('.needs-pentagram').exists()).toBe(true)
    expect(wrapper.find('svg.pentagram-svg').exists()).toBe(true)
  })

  it('默认 props 渲染 5 条引导线', () => {
    const wrapper = mount(NeedsPentagram, { props: { needs: defaultNeeds } })
    const guides = wrapper.findAll('polygon.pentagram-guide')
    expect(guides.length).toBe(5)
  })

  it('aria-label 包含五维数值', () => {
    const needs = { hunger: 60, fatigue: 30, boredom: 80, stress: 10, curiosity: 50 }
    const wrapper = mount(NeedsPentagram, { props: { needs } })
    const label = wrapper.find('.needs-pentagram').attributes('aria-label')
    expect(label).toContain('饿:60')
    expect(label).toContain('累:30')
    expect(label).toContain('闷:80')
    expect(label).toContain('压:10')
    expect(label).toContain('奇:50')
  })

  it('alive=true 时添加 breathing class', () => {
    const wrapper = mount(NeedsPentagram, {
      props: { needs: defaultNeeds, alive: true },
    })
    expect(wrapper.find('.pentagram-body').classes()).toContain('breathing')
  })

  it('alive=false 时无 breathing class', () => {
    const wrapper = mount(NeedsPentagram, {
      props: { needs: defaultNeeds, alive: false },
    })
    expect(wrapper.find('.pentagram-body').classes()).not.toContain('breathing')
  })

  it('渲染 5 条维度轴线', () => {
    const wrapper = mount(NeedsPentagram, { props: { needs: defaultNeeds } })
    const axes = wrapper.findAll('line.pentagram-axis')
    expect(axes.length).toBe(5)
  })

  it('不渲染数据顶点圆点', () => {
    const wrapper = mount(NeedsPentagram, { props: { needs: defaultNeeds } })
    expect(wrapper.findAll('circle').length).toBe(0)
  })

  it('需求值 >70 时对应轴线带 critical class', () => {
    const needs = { hunger: 80, fatigue: 0, boredom: 0, stress: 0, curiosity: 0 }
    const wrapper = mount(NeedsPentagram, { props: { needs } })
    const axes = wrapper.findAll('line.pentagram-axis')
    expect(axes[0].classes()).toContain('critical')
    expect(axes[1].classes()).not.toContain('critical')
  })

  it('数值按 alert/watch/calm 三档分级着色', () => {
    const needs = { hunger: 80, fatigue: 55, boredom: 20, stress: 40, curiosity: 70 }
    const wrapper = mount(NeedsPentagram, { props: { needs } })
    const values = wrapper.findAll('tspan.pentagram-value')
    expect(values.length).toBe(5)
    expect(values[0].classes()).toContain('alert')
    expect(values[1].classes()).toContain('watch')
    expect(values[2].classes()).toContain('calm')
    expect(values[3].classes()).toContain('watch')
    expect(values[4].classes()).toContain('watch')
  })

  it('渲染 5 个维度标签文本', () => {
    const wrapper = mount(NeedsPentagram, { props: { needs: defaultNeeds } })
    const labels = wrapper.findAll('text.pentagram-label')
    expect(labels.length).toBe(5)
    expect(labels[0].text()).toContain('饿')
    expect(labels[1].text()).toContain('累')
  })
})
