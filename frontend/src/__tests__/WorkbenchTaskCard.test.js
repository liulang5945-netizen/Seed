import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import WorkbenchTaskCard from '../components/WorkbenchTaskCard.vue'

describe('WorkbenchTaskCard', () => {
  it('shows Taiji goal intake and the honest semantic-provider boundary', () => {
    const wrapper = mount(WorkbenchTaskCard, {
      props: {
        interpretation: {
          interpretation: { status: 'candidate' },
          goal: { description: '查看工作区文件', goal_id: 'goal:demo' },
        },
      },
    })

    expect(wrapper.text()).toContain('Taiji 目标证据')
    expect(wrapper.text()).toContain('查看工作区文件')
    expect(wrapper.text()).toContain('等待语义器官')
    expect(wrapper.text()).toContain('语言器官只负责表达')
  })

  it('requires every returned approval before exposing execution', async () => {
    const wrapper = mount(WorkbenchTaskCard, {
      props: {
        plan: {
          plan_id: 'plan-1',
          status: 'needs_approval',
          planning: { steps: [{ step_id: 'step-1', grounding: [{ action_kind: 'workspace.apply_patch' }] }] },
          approval_requirements: [{
            request_id: 'request-1',
            capability_id: 'workspace.apply_patch',
            preview: { summary: '替换一段文本' },
          }],
        },
      },
    })

    expect(wrapper.find('button.action-button.primary').exists()).toBe(false)
    await wrapper.find('button.action-button.secondary').trigger('click')
    expect(wrapper.emitted('approve')).toEqual([['request-1']])

    await wrapper.setProps({ approvalTokens: { 'request-1': 'token-1' } })
    expect(wrapper.find('button.action-button.primary').exists()).toBe(true)
    await wrapper.find('button.action-button.primary').trigger('click')
    expect(wrapper.emitted('execute')).toHaveLength(1)
  })

  it('distinguishes provider evidence from Taiji grounding', () => {
    const wrapper = mount(WorkbenchTaskCard, {
      props: {
        interpretation: {
          interpretation: { status: 'resolved' },
          goal: { description: '读取工作区文件' },
          decomposition: {
            steps: [{ step_id: 'step-1', description: '读取 README.md' }],
          },
        },
      },
    })

    expect(wrapper.text()).toContain('语义步骤证据')
    expect(wrapper.text()).toContain('读取 README.md')
    expect(wrapper.text()).toContain('等待 Taiji grounding')
    expect(wrapper.text()).not.toContain('等待语义器官')
  })
})
