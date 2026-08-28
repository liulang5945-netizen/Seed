/** Compute the SHA-256 digest used by the native Workbench write contract. */
export async function sha256Text(text) {
  if (!globalThis.crypto?.subtle || typeof TextEncoder === 'undefined') {
    throw new Error('当前浏览器不支持原生文件写入所需的 SHA-256')
  }
  const bytes = new TextEncoder().encode(String(text))
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes)
  return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('')
}
