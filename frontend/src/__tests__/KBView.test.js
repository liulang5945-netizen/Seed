import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import KBView from '../views/KBView.vue'
import { authFetch } from '../composables/apiClient.js'

vi.mock('../composables/apiClient.js', () => ({
  API_BASE: '',
  authFetch: vi.fn(),
}))

const jsonResponse = (data, ok = true, status = 200) => ({
  ok,
  status,
  json: async () => data,
})

beforeEach(() => {
  setActivePinia(createPinia())
  authFetch.mockReset()
  authFetch.mockResolvedValue(jsonResponse({
    status: 'ok',
    health: { state: 'connected', model_loaded: true, is_taiji: true, is_seed: true },
    tools: { status: 'ok', tools: [] },
  }))
})

const mountView = () => mount(KBView)

describe('KBView', () => {
  it('默认只呈现真实 native capability 边界，不调用 Legacy RAG', async () => {
    const wrapper = mountView()
    await flushPromises()

    expect(wrapper.text()).toContain('原生能力待接入')
    expect(wrapper.text()).toContain('旧 RAG 接口已从默认客户端路径移除')
    expect(wrapper.text()).not.toContain('上传文件')
    expect(wrapper.text()).not.toContain('检索配置')
    expect(authFetch).not.toHaveBeenCalled()
  })

  it('刷新状态只请求统一 runtime status', async () => {
    const wrapper = mountView()
    await wrapper.find('button').trigger('click')
    await flushPromises()

    expect(authFetch).toHaveBeenCalledWith('/api/runtime/status')
    expect(authFetch.mock.calls.some(([url]) => String(url).includes('/api/rag/'))).toBe(false)
  })

  it('展示 runtime 返回的知识 capability，而不是前端静态假设', async () => {
    authFetch.mockResolvedValue(jsonResponse({
      status: 'ok',
      health: { state: 'connected', model_loaded: true },
      tools: {
        status: 'ok',
        tools: [{ name: 'knowledge.ingest', description: 'ingest source', enabled: true, source_id: 's1' }],
      },
    }))
    const wrapper = mountView()
    await wrapper.find('button').trigger('click')
    await flushPromises()

    expect(wrapper.text()).toContain('原生能力已接入')
    expect(wrapper.text()).toContain('knowledge.ingest')
  })
})
