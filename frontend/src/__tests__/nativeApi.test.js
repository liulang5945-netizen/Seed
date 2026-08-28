import { describe, expect, it, vi } from 'vitest'
import { nativeApi, nativeApiPaths } from '../composables/nativeApi.js'
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

describe('nativeApi facade', () => {
  it('keeps native endpoint paths explicit and OpenAPI-shaped', () => {
    expect(nativeApiPaths.runtime.status).toBe('/api/runtime/status')
    expect(nativeApiPaths.workbench.loopExecute).toBe('/api/workbench/loop/execute')
    expect(nativeApiPaths.system.selectFolder).toBe('/api/system/select_folder')
  })

  it('serializes workbench JSON payloads at the facade boundary', async () => {
    authFetch.mockResolvedValueOnce(jsonResponse({ status: 'ok' }))

    await nativeApi.workbenchPreview({ kind: 'workspace.rename', parameters: { path: 'README.md' } })

    expect(authFetch).toHaveBeenCalledWith('/api/workbench/preview', expect.objectContaining({
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind: 'workspace.rename', parameters: { path: 'README.md' } }),
    }))
  })

  it('encodes system dialog query through the facade', async () => {
    authFetch.mockResolvedValueOnce(jsonResponse({ status: 'cancel' }))

    await nativeApi.systemSelectFolder('请选择 “Seed” 文件夹')

    expect(authFetch).toHaveBeenCalledWith(
      '/api/system/select_folder?title=%E8%AF%B7%E9%80%89%E6%8B%A9+%E2%80%9CSeed%E2%80%9D+%E6%96%87%E4%BB%B6%E5%A4%B9',
    )
  })

  it('turns non-2xx JSON into a native API error', async () => {
    authFetch.mockResolvedValueOnce(jsonResponse({ detail: 'workspace rejected' }, false, 409))

    await expect(nativeApi.workbenchCapabilities()).rejects.toThrow('workspace rejected')
  })
})
