import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import WorkspacePathDialog from '../components/WorkspacePathDialog.vue'

describe('WorkspacePathDialog', () => {
  it('emits path, browse, apply and close intents without owning workspace state', async () => {
    const wrapper = mount(WorkspacePathDialog, {
      props: {
        visible: true,
        path: 'E:/Seed',
        quickPaths: [{ label: '桌面', path: 'C:/Users/x/Desktop' }],
      },
    })

    await wrapper.find('.qp-btn').trigger('click')
    await wrapper.find('.browse-btn').trigger('click')
    await wrapper.find('.dlg-input').setValue('E:/Seed/agent_workspace')
    await wrapper.find('.dlg-input').trigger('keydown.enter')
    await wrapper.find('.dlg-btn:not(.primary)').trigger('click')

    expect(wrapper.emitted('update:path')).toEqual([
      ['C:/Users/x/Desktop'],
      ['E:/Seed/agent_workspace'],
    ])
    expect(wrapper.emitted('browse')).toHaveLength(1)
    expect(wrapper.emitted('apply')).toHaveLength(1)
    expect(wrapper.emitted('update:visible')).toEqual([[false]])
  })
})
