import { describe, it, expect, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import FileUploadQueue from '../components/FileUploadQueue.vue'

vi.mock('../composables/apiClient.js', () => ({
  API_BASE: '',
  authFetch: vi.fn(),
}))

describe('blinkDom 门禁', () => {
  it('createElement 拒绝 emoji 标签名（与 Blink 一致）', () => {
    expect(() => document.createElement('📤')).toThrow(/not a valid name/)
    expect(() => document.createElement('1div')).toThrow(/not a valid name/)
  })

  it('合法标签名仍正常创建', () => {
    expect(document.createElement('div').tagName).toBe('DIV')
    expect(document.createElement('my-widget').tagName).toBe('MY-WIDGET')
  })
})

describe('FileUploadQueue 图标契约', () => {
  const mountQueue = (props = {}) =>
    mount(FileUploadQueue, { props: { uploadEndpoint: '/api/rag/upload', ...props } })

  it('默认图标为合法组件，不产生非法原生标签', () => {
    const wrapper = mountQueue()
    expect(wrapper.find('.upload-dropzone').exists()).toBe(true)
    // 图标渲染成 svg 而非 emoji 文本节点
    expect(wrapper.find('.dropzone-icon').element.tagName.toLowerCase()).toBe('svg')
  })

  it('调用方误传字符串图标时兜底为合法组件而非崩溃', () => {
    // 修复前：字符串会被 Vue 当原生标签名，emoji 在 Blink 里渲染期抛
    // InvalidCharacterError 并炸掉整棵 router-view 子树（知识库白屏根因）。
    const wrapper = mountQueue({ uploadIcon: '📤', icon: '📄' })
    expect(wrapper.find('.dropzone-icon').element.tagName.toLowerCase()).toBe('svg')
  })
})
