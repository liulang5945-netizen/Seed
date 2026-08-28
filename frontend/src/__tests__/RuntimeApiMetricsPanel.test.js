import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import RuntimeApiMetricsPanel from '../components/RuntimeApiMetricsPanel.vue'
import { nativeApi, nativeApiMetrics } from '../composables/nativeApi.js'
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

describe('RuntimeApiMetricsPanel', () => {
  beforeEach(() => {
    nativeApiMetrics.reset()
    vi.clearAllMocks()
  })

  it('shows an explicit empty state before any native request', () => {
    const wrapper = mount(RuntimeApiMetricsPanel)

    expect(wrapper.text()).toContain('暂无 nativeApi 请求记录')
    wrapper.unmount()
  })

  it('renders path, counts, latency and final status from facade metrics', async () => {
    authFetch.mockResolvedValueOnce(jsonResponse({ status: 'ok' }, true, 204))
    await nativeApi.workbenchCapabilities()

    const wrapper = mount(RuntimeApiMetricsPanel)

    expect(wrapper.text()).toContain('/api/workbench/capabilities')
    expect(wrapper.text()).toContain('请求 1')
    expect(wrapper.text()).toContain('成功 1')
    expect(wrapper.text()).toContain('HTTP 204')
    expect(wrapper.text()).toContain('平均')
    wrapper.unmount()
  })
})
