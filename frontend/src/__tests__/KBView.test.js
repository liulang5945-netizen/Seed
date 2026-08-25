import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import KBView from '../views/KBView.vue'
import FileUploadQueue from '../components/FileUploadQueue.vue'
import { authFetch } from '../composables/apiClient.js'

// 隔离网络——所有请求经 mock 路由，不发真实请求
vi.mock('../composables/apiClient.js', () => ({
  API_BASE: '',
  authFetch: vi.fn(),
}))

const jsonResponse = (data, ok = true, status = 200) => ({
  ok,
  status,
  json: async () => data,
})

const kbFileList = [
  { name: 'notes.md', size: 2048, mtime: 1700000000, status: 'indexed' },
  // 缺字段条目：前端需回退为 —
  { name: 'draft.txt' },
]

beforeEach(() => {
  authFetch.mockReset()
  authFetch.mockImplementation((url, options = {}) => {
    const method = (options.method || 'GET').toUpperCase()
    if (url.endsWith('/api/rag/config') && method === 'GET') {
      return Promise.resolve(jsonResponse({ status: 'success', config: { candidate_k: 20 } }))
    }
    if (url.endsWith('/api/rag/config') && method === 'PUT') {
      return Promise.resolve(jsonResponse({ status: 'success' }))
    }
    if (url.endsWith('/api/rag/status')) {
      return Promise.resolve(
        jsonResponse({ status: 'ok', doc_count: 1, chunk_count: 5, has_embeddings: true })
      )
    }
    if (url.endsWith('/api/rag/stats')) {
      return Promise.resolve(jsonResponse({ status: 'ok', doc_count: 1, chunk_count: 5 }))
    }
    if (url.endsWith('/api/rag/files')) {
      return Promise.resolve(jsonResponse({ files: kbFileList }))
    }
    if (url.endsWith('/api/rag/clear') && method === 'POST') {
      return Promise.resolve(jsonResponse({ status: 'success', removed: 2 }))
    }
    if (url.includes('/api/rag/preview/')) {
      return Promise.resolve(jsonResponse({ content: 'hello kb preview' }))
    }
    return Promise.resolve(jsonResponse({}))
  })
})

const mountView = ({ toast = vi.fn(), confirm = vi.fn(() => Promise.resolve(true)) } = {}) =>
  mount(KBView, { global: { provide: { toast, $confirm: confirm } } })

const clearCalls = () =>
  authFetch.mock.calls.filter(
    ([u, o]) => u.endsWith('/api/rag/clear') && (o?.method || 'GET') === 'POST'
  ).length

describe('KBView', () => {
  it('渲染 FileUploadQueue 并指向 /api/rag/upload', async () => {
    const wrapper = mountView()
    await flushPromises()
    const uploader = wrapper.findComponent(FileUploadQueue)
    expect(uploader.exists()).toBe(true)
    expect(uploader.props('uploadEndpoint')).toBe('/api/rag/upload')
    // 上传按钮存在（经 kbUploadRef.triggerBrowse 触发选择框）
    expect(
      wrapper.findAll('button').some((b) => b.text().includes('上传文件'))
    ).toBe(true)
  })

  it('列表渲染真实元数据，缺字段回退 —', async () => {
    const wrapper = mountView()
    await flushPromises()
    const rows = wrapper.findAll('tbody tr')
    expect(rows.length).toBe(2)
    // 完整条目：大小格式化 + 状态/时间真实化
    expect(rows[0].text()).toContain('notes.md')
    expect(rows[0].text()).toContain('2.0 KB')
    expect(rows[0].text()).toContain('已索引')
    expect(rows[0].text()).toContain('2023-11')
    // 缺字段条目回退 —
    expect(rows[1].text()).toContain('draft.txt')
    expect(rows[1].text()).toContain('—')
    expect(rows[1].text()).not.toContain('已索引')
    // 索引状态摘要展示（ragStatus）
    expect(wrapper.find('.head-status').text()).toContain('1 文档')
  })

  it('清空需二次确认：确认后调用 POST /api/rag/clear 并刷新', async () => {
    const toast = vi.fn()
    const confirm = vi.fn(() => Promise.resolve(true))
    const wrapper = mountView({ toast, confirm })
    await flushPromises()

    const btn = wrapper.findAll('button').find((b) => b.text().includes('清空知识库'))
    expect(btn).toBeTruthy()
    await btn.trigger('click')
    await flushPromises()

    expect(confirm).toHaveBeenCalledTimes(1)
    expect(confirm.mock.calls[0][0].type).toBe('danger')
    expect(clearCalls()).toBe(1)
    expect(toast).toHaveBeenCalledWith('知识库已清空', 'success')
  })

  it('取消确认时不发起清空请求', async () => {
    const confirm = vi.fn(() => Promise.resolve(false))
    const wrapper = mountView({ confirm })
    await flushPromises()

    const btn = wrapper.findAll('button').find((b) => b.text().includes('清空知识库'))
    await btn.trigger('click')
    await flushPromises()

    expect(confirm).toHaveBeenCalledTimes(1)
    expect(clearCalls()).toBe(0)
  })

  it('预览按钮打开弹窗并展示后端内容', async () => {
    const wrapper = mountView()
    await flushPromises()
    expect(wrapper.find('.dlg-overlay').exists()).toBe(false)

    const previewBtn = wrapper.find('button[aria-label="预览"]')
    expect(previewBtn.exists()).toBe(true)
    await previewBtn.trigger('click')
    await flushPromises()

    expect(
      authFetch.mock.calls.some(([u]) => u.includes('/api/rag/preview/notes.md'))
    ).toBe(true)
    const dlg = wrapper.find('.dlg-overlay')
    expect(dlg.exists()).toBe(true)
    expect(dlg.text()).toContain('notes.md')
    expect(dlg.text()).toContain('hello kb preview')

    await dlg.find('.dlg-btn.primary').trigger('click')
    expect(wrapper.find('.dlg-overlay').exists()).toBe(false)
  })

  it('已移除无后端支撑的死控件', async () => {
    const wrapper = mountView()
    await flushPromises()
    expect(wrapper.text()).not.toContain('批量操作')
    expect(wrapper.text()).not.toContain('筛选')
    expect(wrapper.text()).not.toContain('Embedding 模型')
    expect(wrapper.text()).not.toContain('相似度阈值')
    expect(wrapper.findAll('input.cb').length).toBe(0)
  })
})
