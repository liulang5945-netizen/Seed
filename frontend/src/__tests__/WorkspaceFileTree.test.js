import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import { FileText } from 'lucide-vue-next'
import WorkspaceFileTree from '../components/WorkspaceFileTree.vue'

describe('WorkspaceFileTree', () => {
  it('只负责展示树并转发节点与工具栏事件', async () => {
    const nodes = [
      { name: 'src', path: 'src', type: 'directory', depth: 0 },
      { name: 'README.md', path: 'README.md', type: 'file', depth: 0 },
    ]
    const wrapper = mount(WorkspaceFileTree, {
      props: {
        fileTree: nodes,
        flatList: nodes,
        expandedDirs: new Set(['src']),
        getFileIcon: () => FileText,
      },
    })

    await wrapper.find('[title="新建文件"]').trigger('click')
    await wrapper.find('[title="刷新文件树"]').trigger('click')
    await wrapper.find('.tree-item').trigger('click')
    await wrapper.find('.tree-item').trigger('contextmenu')
    await wrapper.find('.resize-col').trigger('mousedown')

    expect(wrapper.findAll('.tree-item')).toHaveLength(2)
    expect(wrapper.emitted('new-file')).toHaveLength(1)
    expect(wrapper.emitted('refresh')).toHaveLength(1)
    expect(wrapper.emitted('select-node')?.[0]).toEqual([nodes[0]])
    expect(wrapper.emitted('context-node')).toHaveLength(1)
    expect(wrapper.emitted('resize')).toHaveLength(1)
  })
})
