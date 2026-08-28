import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import MonacoEditor from '../components/MonacoEditor.vue'

// R4: MonacoEditor 组件测试——@monaco-editor/loader 与网络全部隔离

const { fakeMonaco, loaderInit, authFetchMock } = vi.hoisted(() => {
  const fakeEditor = {
    getValue: vi.fn(() => 'hello'),
    setValue: vi.fn(),
    getModel: vi.fn(() => ({})),
    onDidChangeCursorPosition: vi.fn(),
    onDidChangeModelContent: vi.fn(),
    addCommand: vi.fn(),
    layout: vi.fn(),
    dispose: vi.fn(),
  }
  const fakeMonaco = {
    editor: {
      create: vi.fn(() => fakeEditor),
      setModelLanguage: vi.fn(),
      setTheme: vi.fn(),
    },
    KeyMod: { CtrlCmd: 2048 },
    KeyCode: { KeyS: 49 },
  }
  const loaderInit = vi.fn(() => Promise.resolve(fakeMonaco))
  const authFetchMock = vi.fn((url) => {
    if (String(url).includes('/api/workbench/capabilities')) {
      return Promise.resolve({
        ok: true,
        json: async () => ({
          snapshot_id: 'workbench-test-snapshot',
          capabilities: [{ capability_id: 'workspace.read', enabled: true }],
        }),
      })
    }
    return Promise.resolve({ ok: true, json: async () => ({ content: 'print(1)' }) })
  })
  return { fakeMonaco, fakeEditor, loaderInit, authFetchMock }
})

vi.mock('@monaco-editor/loader', () => ({
  default: { init: loaderInit },
}))

vi.mock('../composables/apiClient.js', () => ({
  API_BASE: '',
  authFetch: (...args) => authFetchMock(...args),
}))

// jsdom 缺失的浏览器 API 桩
class ResizeObserverStub {
  constructor(callback) {
    this.callback = callback
  }
  observe(_el) {
    // 立即通知「容器已有尺寸」，驱动 initMonaco
    this.callback([{ contentRect: { width: 800, height: 600 } }], this)
  }
  disconnect() {}
}

describe('MonacoEditor', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.useFakeTimers()
    vi.clearAllMocks()
    loaderInit.mockImplementation(() => Promise.resolve(fakeMonaco))
    authFetchMock.mockImplementation((url) => {
      if (String(url).includes('/api/workbench/capabilities')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            snapshot_id: 'workbench-test-snapshot',
            capabilities: [{ capability_id: 'workspace.read', enabled: true }],
          }),
        })
      }
      return Promise.resolve({ ok: true, json: async () => ({ content: 'print(1)' }) })
    })
    global.ResizeObserver = ResizeObserverStub
    window.matchMedia =
      window.matchMedia ||
      (() => ({ matches: false, addEventListener() {}, removeEventListener() {} }))
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  const mountEditor = async () => {
    const wrapper = mount(MonacoEditor, {
      global: { stubs: { Save: true } },
    })
    await vi.advanceTimersByTimeAsync(50)
    await flushPromises()
    return wrapper
  }

  it('渲染工具栏 / 语言选择器 / 状态栏', async () => {
    const wrapper = await mountEditor()
    expect(wrapper.find('.monaco-toolbar').exists()).toBe(true)
    expect(wrapper.find('select.lang-select').exists()).toBe(true)
    expect(wrapper.find('.monaco-statusbar').exists()).toBe(true)
    expect(wrapper.find('.monaco-statusbar').text()).toContain('行 1')
    // 语言选择器包含全部 17 种语言
    expect(wrapper.findAll('select.lang-select option').length).toBe(17)
  })

  it('Monaco 加载成功后创建编辑器且无降级', async () => {
    const wrapper = await mountEditor()
    expect(loaderInit).toHaveBeenCalled()
    expect(fakeMonaco.editor.create).toHaveBeenCalled()
    expect(wrapper.find('.fallback-editor').exists()).toBe(false)
    expect(wrapper.find('.monaco-error').exists()).toBe(false)
  })

  it('Monaco 加载失败时降级到简易文本编辑器', async () => {
    loaderInit.mockImplementation(() => Promise.reject(new Error('cdn blocked')))
    const wrapper = await mountEditor()
    expect(wrapper.find('.fallback-editor').exists()).toBe(true)
    expect(wrapper.find('textarea.fallback-textarea').exists()).toBe(true)
  })

  it('openFile 拉取内容并创建标签页', async () => {
    const wrapper = await mountEditor()
    await wrapper.vm.openFile('demo/main.py')
    await flushPromises()
    expect(authFetchMock).toHaveBeenCalledWith(
      expect.stringContaining('/api/workbench/file?path=demo%2Fmain.py')
    )
    expect(wrapper.vm.openTabs.length).toBe(1)
    expect(wrapper.vm.openTabs[0].language).toBe('python')
    expect(wrapper.vm.activeTab).toBe('demo/main.py')
    expect(wrapper.find('.monaco-tab .tab-name').text()).toBe('main.py')
  })

  it('已有活动标签时点击树中新文件也会激活新标签', async () => {
    const wrapper = await mountEditor()
    await wrapper.vm.openFile('demo/a.py')
    await flushPromises()
    await wrapper.vm.openFile('demo/b.py')
    await flushPromises()
    expect(wrapper.vm.openTabs.length).toBe(2)
    expect(wrapper.vm.activeTab).toBe('demo/b.py')
  })

  it('saveFile 保存成功后发出 saved 事件并清除脏标记', async () => {
    const wrapper = await mountEditor()
    await wrapper.vm.openFile('demo/main.py')
    await flushPromises()
    wrapper.vm.isDirty = true

    await wrapper.vm.saveFile()
    await flushPromises()

    const postCall = authFetchMock.mock.calls.find((c) => c[1]?.method === 'POST')
    expect(postCall).toBeTruthy()
    expect(wrapper.emitted('saved')).toBeTruthy()
    expect(wrapper.emitted('saved')[0]).toEqual(['demo/main.py'])
    expect(wrapper.vm.isDirty).toBe(false)
  })

  it('saveFile 非 2xx 时返回 false 并发出携带后端 detail 的 save-error', async () => {
    const wrapper = await mountEditor()
    await wrapper.vm.openFile('demo/main.py')
    await flushPromises()
    wrapper.vm.isDirty = true
    authFetchMock.mockImplementation(() =>
      Promise.resolve({
        ok: false,
        status: 403,
        json: async () => ({ detail: '无权限写入' }),
      })
    )

    const ok = await wrapper.vm.saveFile()
    await flushPromises()

    expect(ok).toBe(false)
    expect(wrapper.emitted('save-error')).toBeTruthy()
    expect(wrapper.emitted('save-error')[0][0]).toContain('无权限写入')
    expect(wrapper.emitted('saved')).toBeFalsy()
    // 失败时脏标记保留，内容不丢失
    expect(wrapper.vm.isDirty).toBe(true)
  })

  it('closeTab 关闭最后一个标签后清空活动标签', async () => {
    const wrapper = await mountEditor()
    await wrapper.vm.openFile('a.txt')
    await flushPromises()
    wrapper.vm.closeTab('a.txt')
    expect(wrapper.vm.openTabs.length).toBe(0)
    expect(wrapper.vm.activeTab).toBe('')
  })
})
