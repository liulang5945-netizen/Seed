import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import { Brain } from 'lucide-vue-next'
import ChatMessageList from '../components/ChatMessageList.vue'

const hints = [{ icon: Brain, text: '解释 Taiji' }]

const baseProps = {
  messages: [],
  displayedMessages: [],
  quickHints: hints,
  connectionStatus: '已连接',
  isTaiji: true,
  runtimeState: 'connected',
}

describe('ChatMessageList', () => {
  it('renders the welcome surface and forwards welcome interactions', async () => {
    const wrapper = mount(ChatMessageList, { props: baseProps })

    expect(wrapper.find('.chat-welcome').exists()).toBe(true)
    expect(wrapper.text()).toContain('Taiji Native 语言通路')
    await wrapper.find('.suggestion').trigger('click')
    await wrapper.find('.example-toggle').trigger('click')

    expect(wrapper.emitted('select-hint')).toEqual([['解释 Taiji']])
    expect(wrapper.emitted('toggle-example')).toHaveLength(1)
  })

  it('renders messages and forwards message actions', async () => {
    const message = { id: 2, role: 'assistant', content: '你好，Seed' }
    const wrapper = mount(ChatMessageList, {
      props: {
        ...baseProps,
        messages: [{ id: 1, role: 'user', content: '你好' }, message],
        displayedMessages: [{ id: 1, role: 'user', content: '你好' }, message],
      },
    })

    expect(wrapper.find('.chat-welcome').exists()).toBe(false)
    expect(wrapper.findAll('article.msg')).toHaveLength(2)
    const actions = wrapper.findAll('.msg-action-btn')
    await actions[0].trigger('click')
    await actions[1].trigger('click')
    await actions[2].trigger('click')

    expect(wrapper.emitted('copy')).toEqual([['你好，Seed']])
    expect(wrapper.emitted('like')).toEqual([[2]])
    expect(wrapper.emitted('regenerate')).toEqual([[2]])
  })

  it('renders the shared native workbench audit phases in an assistant message', () => {
    const message = {
      id: 3,
      role: 'assistant',
      content: '已读取 README',
      workbenchEvents: [
        { sequence: 1, phase: 'planned', payload: { request: { capability_id: 'workspace.read' } } },
        { sequence: 2, phase: 'policy', payload: { policy: { capability_id: 'workspace.read', decision: 'allow' } } },
        { sequence: 3, phase: 'executing', payload: { capability_id: 'workspace.read' } },
        { sequence: 4, phase: 'outcome', payload: { outcome: { capability_id: 'workspace.read', status: 'success' } } },
      ],
    }
    const wrapper = mount(ChatMessageList, {
      props: { ...baseProps, messages: [message], displayedMessages: [message] },
    })

    expect(wrapper.find('[aria-label="Taiji 工作台执行轨迹"]').exists()).toBe(true)
    expect(wrapper.findAll('.workbench-trace-row')).toHaveLength(4)
    expect(wrapper.text()).toContain('workspace.read')
    expect(wrapper.text()).toContain('成功')
  })
})
