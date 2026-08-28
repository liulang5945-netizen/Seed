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
})
