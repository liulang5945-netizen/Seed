import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import LifeNeedsDashboard from '../components/LifeNeedsDashboard.vue'

const needRows = [
  { key: 'hunger', label: '饥饿 · 知识摄取', value: 66, state: 'watch' },
  { key: 'fatigue', label: '疲劳 · 睡眠需求', value: null, state: 'none' },
]

describe('LifeNeedsDashboard', () => {
  it('renders the needs chart, expressions and normalized detail rows', () => {
    const wrapper = mount(LifeNeedsDashboard, {
      props: {
        needs: { hunger: 66 },
        alive: true,
        hasNeedsData: true,
        lifeExpressions: [{ emoji: '🍚', text: '需要摄取知识', priority: 'high' }],
        needRows,
        activityLog: [{ type: 'feed', emoji: '🍚', message: '已记录', time: '10:00' }],
      },
      global: {
        stubs: {
          NeedsPentagram: { props: ['needs', 'alive'], template: '<div class="pentagram-stub">{{ needs.hunger }}</div>' },
        },
      },
    })

    expect(wrapper.find('.pentagram-stub').text()).toBe('66')
    expect(wrapper.text()).toContain('需要摄取知识')
    expect(wrapper.text()).toContain('饥饿 · 知识摄取')
    expect(wrapper.text()).toContain('关注')
    expect(wrapper.text()).toContain('已记录')
  })

  it('shows honest empty states when runtime needs are absent', () => {
    const wrapper = mount(LifeNeedsDashboard, {
      props: { hasNeedsData: false, needRows },
      global: {
        stubs: {
          NeedsPentagram: { template: '<div class="pentagram-stub"></div>' },
        },
      },
    })

    expect(wrapper.text()).toContain('暂无需求数据')
    expect(wrapper.text()).toContain('暂无生命事件')
    expect(wrapper.find('.event-item').exists()).toBe(false)
  })
})
