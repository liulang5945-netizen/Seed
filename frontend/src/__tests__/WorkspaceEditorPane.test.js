import { describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import WorkspaceEditorPane from '../components/WorkspaceEditorPane.vue'

describe('WorkspaceEditorPane', () => {
  it('负责编辑器/终端容器并转发终端尺寸事件', async () => {
    const approvalHandler = vi.fn()
    const wrapper = mount(WorkspaceEditorPane, {
      props: { showTerminal: true, terminalHeight: 180, approvalHandler },
      global: { stubs: { MonacoEditor: true, WebTerminal: true } },
    })

    expect(wrapper.find('.monaco-container').exists()).toBe(true)
    expect(wrapper.find('.ide-terminal').attributes('style')).toContain('height: 180px')
    await wrapper.find('.resize-row').trigger('mousedown')

    expect(wrapper.emitted('resize-terminal')).toHaveLength(1)
  })
})
