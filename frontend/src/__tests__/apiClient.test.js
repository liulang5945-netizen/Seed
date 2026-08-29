import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'

/**
 * apiClient 测试
 *
 * 测试 resolveApiBase 逻辑和 authFetch 重试/认证行为
 */

// Mock localStorage
const localStorageMock = (() => {
  let store = {}
  return {
    getItem: vi.fn((key) => store[key] ?? null),
    setItem: vi.fn((key, value) => { store[key] = String(value) }),
    removeItem: vi.fn((key) => { delete store[key] }),
    clear: vi.fn(() => { store = {} }),
  }
})()
Object.defineProperty(globalThis, 'localStorage', { value: localStorageMock })

describe('resolveApiBase', () => {
  const originalLocation = window.location

  beforeEach(() => {
    localStorageMock.clear()
    vi.resetModules()
  })

  afterEach(() => {
    // Restore location
    Object.defineProperty(window, 'location', {
      value: originalLocation,
      writable: true,
      configurable: true,
    })
  })

  it('DEV 模式返回空字符串', async () => {
    // jsdom 默认 location 是 localhost，import.meta.env.DEV 在 vitest 中为 true
    const { API_BASE } = await import('../composables/apiClient.js')
    expect(API_BASE).toBe('')
  })

  it('桌面模式复用实际前端端口，不强制回落到 8000', async () => {
    Object.defineProperty(window, 'location', {
      value: new URL('http://127.0.0.1:8137/#/?taiji_client=desktop'),
      writable: true,
      configurable: true,
    })

    const { API_BASE } = await import('../composables/apiClient.js')

    expect(API_BASE).toBe('http://127.0.0.1:8137')
  })
})

describe('authFetch', () => {
  beforeEach(() => {
    localStorageMock.clear()
    vi.resetModules()
  })

  it('自动附加 Authorization 头', async () => {
    localStorageMock.getItem.mockReturnValue('test-jwt-token')

    const mockFetch = vi.fn().mockResolvedValue(new Response('ok', { status: 200 }))
    vi.stubGlobal('fetch', mockFetch)

    const { authFetch } = await import('../composables/apiClient.js')
    await authFetch('http://localhost/api/test')

    expect(mockFetch).toHaveBeenCalledTimes(1)
    const callArgs = mockFetch.mock.calls[0]
    const headers = callArgs[1].headers
    expect(headers.get('Authorization')).toBe('Bearer test-jwt-token')

    vi.unstubAllGlobals()
  })

  it('不覆盖已有的 Authorization 头', async () => {
    localStorageMock.getItem.mockReturnValue('stored-token')

    const mockFetch = vi.fn().mockResolvedValue(new Response('ok', { status: 200 }))
    vi.stubGlobal('fetch', mockFetch)

    const { authFetch } = await import('../composables/apiClient.js')
    await authFetch('http://localhost/api/test', {
      headers: { Authorization: 'Custom token' },
    })

    const callArgs = mockFetch.mock.calls[0]
    const headers = callArgs[1].headers
    expect(headers.get('Authorization')).toBe('Custom token')

    vi.unstubAllGlobals()
  })

  it('401 响应清除 token', async () => {
    localStorageMock.getItem.mockReturnValue('expired-token')

    const mockResponse = { status: 401, headers: new Headers() }
    const mockFetch = vi.fn().mockResolvedValue(mockResponse)
    vi.stubGlobal('fetch', mockFetch)

    const { authFetch } = await import('../composables/apiClient.js')
    await authFetch('http://localhost/api/test')

    expect(localStorageMock.removeItem).toHaveBeenCalledWith('jwt_token')

    vi.unstubAllGlobals()
  })

  it('GET 请求默认重试 2 次', async () => {
    localStorageMock.getItem.mockReturnValue(null)

    let callCount = 0
    const mockFetch = vi.fn().mockImplementation(() => {
      callCount++
      if (callCount <= 2) {
        return Promise.resolve({ status: 503, headers: new Headers() })
      }
      return Promise.resolve(new Response('ok', { status: 200 }))
    })
    vi.stubGlobal('fetch', mockFetch)

    const { authFetch } = await import('../composables/apiClient.js')
    const result = await authFetch('http://localhost/api/test')

    expect(mockFetch).toHaveBeenCalledTimes(3) // 1 initial + 2 retries
    expect(result.status).toBe(200)

    vi.unstubAllGlobals()
  })

  it('POST 写请求默认不重试', async () => {
    localStorageMock.getItem.mockReturnValue(null)

    const mockFetch = vi.fn().mockResolvedValue({ status: 503, headers: new Headers() })
    vi.stubGlobal('fetch', mockFetch)

    const { authFetch } = await import('../composables/apiClient.js')
    await authFetch('http://localhost/api/test', {
      method: 'POST',
      body: JSON.stringify({ data: 'test' }),
    })

    expect(mockFetch).toHaveBeenCalledTimes(1) // no retries for non-idempotent

    vi.unstubAllGlobals()
  })
})

describe('authFetchJSON', () => {
  beforeEach(() => {
    localStorageMock.clear()
    vi.resetModules()
  })

  it('非 JSON 响应抛出错误', async () => {
    localStorageMock.getItem.mockReturnValue(null)

    const mockResponse = {
      status: 200,
      headers: new Headers({ 'content-type': 'text/html' }),
      text: () => Promise.resolve('<html>error</html>'),
    }
    const mockFetch = vi.fn().mockResolvedValue(mockResponse)
    vi.stubGlobal('fetch', mockFetch)

    const { authFetchJSON } = await import('../composables/apiClient.js')

    await expect(authFetchJSON('http://localhost/api/test')).rejects.toThrow(/Expected JSON/)

    vi.unstubAllGlobals()
  })

  it('JSON 响应正常解析', async () => {
    localStorageMock.getItem.mockReturnValue(null)

    const jsonData = { result: 'success' }
    const mockResponse = {
      status: 200,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: () => Promise.resolve(jsonData),
    }
    const mockFetch = vi.fn().mockResolvedValue(mockResponse)
    vi.stubGlobal('fetch', mockFetch)

    const { authFetchJSON } = await import('../composables/apiClient.js')
    const result = await authFetchJSON('http://localhost/api/test')

    expect(result).toEqual(jsonData)

    vi.unstubAllGlobals()
  })
})
