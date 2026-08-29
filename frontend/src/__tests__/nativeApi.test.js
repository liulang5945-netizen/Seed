import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nativeApi, nativeApiMetrics, nativeApiPaths } from '../composables/nativeApi.js'
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
  authFetch.mockReset()
  nativeApiMetrics.reset()
})

describe('nativeApi facade', () => {
  it('keeps native endpoint paths explicit and OpenAPI-shaped', () => {
    expect(nativeApiPaths.runtime.status).toBe('/api/runtime/status')
    expect(nativeApiPaths.workbench.loopExecute).toBe('/api/workbench/loop/execute')
    expect(nativeApiPaths.workbench.taijiAdmit).toBe('/api/workbench/taiji/admit')
    expect(nativeApiPaths.workbench.taijiExecute).toBe('/api/workbench/taiji/execute')
    expect(nativeApiPaths.workbench.taijiProject).toBe('/api/workbench/taiji/project')
    expect(nativeApiPaths.workbench.taijiReproject).toBe('/api/workbench/taiji/reproject')
    expect(nativeApiPaths.workbench.taijiRecoveryPortfolio).toBe('/api/workbench/taiji/recovery-branch/portfolio')
    expect(nativeApiPaths.chat.workbenchStream).toBe('/api/chat/workbench/stream')
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

  it('keeps Taiji task admission and execution on named native operations', async () => {
    authFetch
      .mockResolvedValueOnce(jsonResponse({ admission: { accepted: true } }))
      .mockResolvedValueOnce(jsonResponse({ execution: { outcome: { status: 'success' } } }))
      .mockResolvedValueOnce(jsonResponse({ affordances: [] }))
      .mockResolvedValueOnce(jsonResponse({ affordances: [] }))
      .mockResolvedValueOnce(jsonResponse({ revision: 3, counts: {} }))

    await nativeApi.taijiWorkbenchAdmit({ snapshot_id: 'snapshot-1' })
    await nativeApi.taijiWorkbenchExecute({ snapshot_id: 'snapshot-1' })
    await nativeApi.taijiWorkbenchProject({ snapshot_id: 'snapshot-1', parameter_bindings: {} })
    await nativeApi.taijiWorkbenchReproject({ snapshot_id: 'snapshot-1' })
    await nativeApi.taijiWorkbenchRecoveryPortfolio('loop-1', 'snapshot-1', 3)

    expect(authFetch).toHaveBeenNthCalledWith(1, '/api/workbench/taiji/admit', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ snapshot_id: 'snapshot-1' }),
    }))
    expect(authFetch).toHaveBeenNthCalledWith(2, '/api/workbench/taiji/execute', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ snapshot_id: 'snapshot-1' }),
    }))
    expect(authFetch).toHaveBeenNthCalledWith(3, '/api/workbench/taiji/project', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ snapshot_id: 'snapshot-1', parameter_bindings: {} }),
    }))
    expect(authFetch).toHaveBeenNthCalledWith(4, '/api/workbench/taiji/reproject', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ snapshot_id: 'snapshot-1' }),
    }))
    expect(authFetch).toHaveBeenNthCalledWith(
      5,
      '/api/workbench/taiji/recovery-branch/portfolio?parent_loop_id=loop-1&snapshot_id=snapshot-1&expected_revision=3',
    )
  })

  it('encodes system dialog query through the facade', async () => {
    authFetch.mockResolvedValueOnce(jsonResponse({ status: 'cancel' }))

    await nativeApi.systemSelectFolder('请选择 “Seed” 文件夹')

    expect(authFetch).toHaveBeenCalledWith(
      '/api/system/select_folder?title=%E8%AF%B7%E9%80%89%E6%8B%A9+%E2%80%9CSeed%E2%80%9D+%E6%96%87%E4%BB%B6%E5%A4%B9',
    )
  })

  it('preserves stream response and cancellation signal without parsing it', async () => {
    const response = { ok: false, status: 503, body: null }
    const signal = { aborted: false }
    authFetch.mockResolvedValueOnce(response)

    await expect(nativeApi.chatStream({ prompt: 'hello' }, { signal })).resolves.toBe(response)

    expect(authFetch).toHaveBeenCalledWith('/api/chat/stream', expect.objectContaining({
      method: 'POST',
      retries: 0,
      signal,
      body: JSON.stringify({ prompt: 'hello' }),
    }))
  })

  it('keeps the structured chat-workbench stream on the raw response boundary', async () => {
    const response = { ok: true, status: 200, body: null }
    const signal = { aborted: false }
    authFetch.mockResolvedValueOnce(response)
    const payload = {
      prompt: '打开 README',
      intent: { intent_id: 'read-1', kind: 'workspace.read', snapshot_id: 'snapshot-1' },
    }

    await expect(nativeApi.chatWorkbenchStream(payload, { signal })).resolves.toBe(response)

    expect(authFetch).toHaveBeenCalledWith('/api/chat/workbench/stream', expect.objectContaining({
      method: 'POST',
      retries: 0,
      signal,
      body: JSON.stringify(payload),
    }))
  })

  it('keeps FormData upload outside the JSON facade', async () => {
    const formData = new FormData()
    formData.append('file', new Blob(['dataset']))
    authFetch.mockResolvedValueOnce({ ok: true, status: 200 })

    await nativeApi.uploadTrainingDataset(formData)

    expect(authFetch).toHaveBeenCalledWith('/api/train/upload_dataset', expect.objectContaining({
      method: 'POST',
      body: formData,
    }))
  })

  it('records request count, status and latency for a minimal SLO snapshot', async () => {
    nativeApiMetrics.reset()
    authFetch.mockResolvedValueOnce(jsonResponse({ status: 'ok' }))

    await nativeApi.workbenchCapabilities()

    expect(nativeApiMetrics.snapshot()['/api/workbench/capabilities']).toEqual(expect.objectContaining({
      requests: 1,
      successes: 1,
      failures: 0,
      last_status: 200,
    }))
  })

  it('turns non-2xx JSON into a native API error', async () => {
    authFetch.mockResolvedValueOnce(jsonResponse({ detail: 'workspace rejected' }, false, 409))

    await expect(nativeApi.workbenchCapabilities()).rejects.toThrow('workspace rejected')
  })
})
