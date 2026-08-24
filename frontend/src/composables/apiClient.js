function resolveApiBase() {
  if (typeof window === 'undefined') return ''

  const params = new URLSearchParams(window.location.search)
  const hashQuery = window.location.hash.includes('?')
    ? new URLSearchParams(window.location.hash.slice(window.location.hash.indexOf('?') + 1))
    : new URLSearchParams()

  if (params.get('taiji_client') === 'desktop' || hashQuery.get('taiji_client') === 'desktop') {
    return `${window.location.protocol}//${window.location.hostname}:8000`
  }

  if (import.meta.env.DEV) return ''

  const { protocol, hostname, port } = window.location
  if (port === '8000') return ''
  return `${protocol}//${hostname}:8000`
}

export const API_BASE = resolveApiBase()

// 可安全重试的状态码：超时/限流/网关类错误。不含 500/501 等可能由请求本身触发的服务端错误。
const RETRYABLE_STATUS = new Set([408, 429, 502, 503, 504])
// 幂等方法默认允许重试；非幂等写请求（POST/PUT/DELETE/PATCH）即使带 body 也不应静默重发。
const IDEMPOTENT_METHODS = new Set(['GET', 'HEAD', 'OPTIONS', 'PUT', 'DELETE'])

export async function authFetch(url, options = {}) {
  const method = (options.method || 'GET').toUpperCase()
  const hasBody = options.body != null
  const idempotent = IDEMPOTENT_METHODS.has(method)
  // 默认重试：仅幂等方法默认 2 次；非幂等写请求默认 0（避免重复副作用，如重复推理）。
  // 调用方显式传入 retries 时以调用方为准。
  const maxRetries = options.retries != null ? options.retries : (idempotent ? 2 : 0)
  const token = localStorage.getItem('jwt_token') || ''
  const headers = new Headers(options.headers || {})
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`)
  }

  let lastError = null
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, { ...options, headers })
      if (response.status === 401) {
        localStorage.removeItem('jwt_token')
        if (typeof window !== 'undefined') {
          window.dispatchEvent(new CustomEvent('taiji-auth-expired', {
            detail: { message: 'JWT token 缺失或已过期，请重新登录' },
          }))
        }
      }
      // 仅对可重试状态码、且未发送 body 的请求重试；避免非幂等写请求（如流式 POST）重复副作用。
      const bodySent = !idempotent && hasBody
      if (maxRetries > 0 && attempt < maxRetries && !bodySent && RETRYABLE_STATUS.has(response.status)) {
        lastError = new Error(`Server error HTTP ${response.status}`)
        await new Promise(r => setTimeout(r, (attempt + 1) * 500))
        continue
      }
      return response
    } catch (e) {
      lastError = e
      // 网络层错误：已发送 body 的非幂等写请求同样不重试，避免重复副作用。
      const bodySent = !idempotent && hasBody
      if (maxRetries > 0 && attempt < maxRetries && !bodySent) {
        await new Promise(r => setTimeout(r, (attempt + 1) * 500))
      }
    }
  }
  throw lastError || new Error('authFetch failed after retries')
}

/**
 * 带 Content-Type 校验的 JSON 请求封装。
 * 若响应非 JSON，抛出错误避免静默解析失败。
 */
export async function authFetchJSON(url, options = {}) {
  const response = await authFetch(url, options)
  const ctype = response.headers.get('content-type') || ''
  if (!ctype.includes('application/json')) {
    const text = await response.text().catch(() => '')
    throw new Error(`Expected JSON but got ${ctype || 'unknown content-type'}: ${text.slice(0, 200)}`)
  }
  return response.json()
}
