import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import { Brain } from 'lucide-vue-next'
import ChatComposer from '../components/ChatComposer.vue'

const quickHints = [{ icon: Brain, text: '解释 Taiji' }]
const promptTemplates = {
  code: '解释代码',
  summarize: '总结内容',
  translate: '翻译内容',
}

describe('ChatComposer', () => {
  it('forwards model input, send and stop actions', async () => {
    const wrapper = mount(ChatComposer, {
      props: { modelValue: '已有', canSend: true, isReceiving: true, quickHints, promptTemplates },
    })

    await wrapper.find('textarea').setValue('新的输入')
    await wrapper.find('.send').trigger('click')
    await wrapper.find('.stop-btn').trigger('click')

    expect(wrapper.emitted('update:modelValue')).toEqual([['新的输入']])
    expect(wrapper.emitted('send')).toHaveLength(1)
    expect(wrapper.emitted('stop')).toHaveLength(1)
  })

  it('keeps quick actions and attachment selection local to the UI boundary', async () => {
    const wrapper = mount(ChatComposer, {
      props: { quickHints, promptTemplates },
    })

    await wrapper.find('.composer-chip[title="快速"]').trigger('click')
    await wrapper.find('.quick-item').trigger('click')
    await wrapper.find('.composer-chip[title="代码"]').trigger('click')

    const fileInput = wrapper.find('input[type="file"]')
    const file = new File(['hello'], 'note.txt', { type: 'text/plain' })
    Object.defineProperty(fileInput.element, 'files', { value: [file], configurable: true })
    await fileInput.trigger('change')

    expect(wrapper.emitted('apply-quick-hint')).toEqual([['解释 Taiji']])
    expect(wrapper.emitted('insert-template')).toEqual([['解释代码']])
    expect(wrapper.emitted('files-picked')).toEqual([[[file]]])
  })
})
