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
          programming_languages: [
            { language_id: 'python', label: 'Python', editor_language_id: 'python' },
            { language_id: 'javascript', label: 'JavaScript', editor_language_id: 'javascript' },
            { language_id: 'plaintext', label: 'Plain text', editor_language_id: 'plaintext' },
          ],
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
    authFetchMock.mockImplementation((url, options) => {
      if (String(url).includes('/api/workbench/capabilities')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            snapshot_id: 'workbench-test-snapshot',
            capabilities: [{ capability_id: 'workspace.read', enabled: true }],
            programming_languages: [
              { language_id: 'python', label: 'Python', editor_language_id: 'python' },
              { language_id: 'javascript', label: 'JavaScript', editor_language_id: 'javascript' },
              { language_id: 'plaintext', label: 'Plain text', editor_language_id: 'plaintext' },
            ],
          }),
        })
      }
      if (String(url).includes('/api/workbench/programming-language?path=')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            path: 'demo/main.py',
            file_digest: 'digest',
            programming_language_id: 'python',
            editor_language_id: 'python',
            confidence: 0.99,
            selection_state: 'resolved',
          }),
        })
      }
      if (String(url).includes('/api/workbench/execute')) {
        const request = JSON.parse(options?.body || '{}')
        const cleared = request.parameters?.clear_override === true
        return Promise.resolve({
          ok: true,
          json: async () => ({
            outcome: {
              result: {
                path: 'demo/main.py',
                programming_language_id: cleared ? 'python' : 'javascript',
                editor_language_id: cleared ? 'python' : 'javascript',
                selection_state: cleared ? 'resolved' : 'user_override',
              },
            },
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
    // 语言选择器来自后端 capability projection，而非组件内静态表
    expect(wrapper.findAll('select.lang-select option').length).toBe(4)
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

  it('语言选择提交可逆的 native editor.set_language 覆盖', async () => {
    const wrapper = await mountEditor()
    await wrapper.vm.openFile('demo/main.py')
    await flushPromises()

    await wrapper.find('select.lang-select').setValue('javascript')
    await flushPromises()

    const executeCall = authFetchMock.mock.calls.find(([url, options]) => (
      String(url).includes('/api/workbench/execute') && options?.method === 'POST'
    ))
    expect(executeCall).toBeTruthy()
    expect(JSON.parse(executeCall[1].body)).toMatchObject({
      kind: 'editor.set_language',
      parameters: {
        programming_language_id: 'javascript',
        editor_language_id: 'javascript',
        user_override: true,
      },
    })
    expect(wrapper.vm.openTabs[0].programmingLanguageId).toBe('javascript')
  })

  it('自动检测入口会撤销用户语言覆盖', async () => {
    const wrapper = await mountEditor()
    await wrapper.vm.openFile('demo/main.py')
    await flushPromises()

    await wrapper.find('select.lang-select').setValue('javascript')
    await flushPromises()
    await wrapper.find('select.lang-select').setValue('__auto__')
    await flushPromises()

    const executeBodies = authFetchMock.mock.calls
      .filter(([url, options]) => String(url).includes('/api/workbench/execute') && options?.method === 'POST')
      .map(([, options]) => JSON.parse(options.body))
    expect(executeBodies.at(-1)).toMatchObject({
      kind: 'editor.set_language',
      parameters: { clear_override: true, user_override: false },
    })
    expect(wrapper.vm.openTabs[0].programmingLanguageId).toBe('python')
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
