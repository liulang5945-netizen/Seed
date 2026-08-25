import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import SettingsView from '../views/SettingsView.vue'
import { authFetch } from '../composables/apiClient.js'
import { useAppStore } from '../stores/appStore.js'

// 隔离网络：组件 setup 阶段会请求 /api/health、/api/settings、/api/system/version
vi.mock('../composables/apiClient.js', () => ({
  API_BASE: '',
  authFetch: vi.fn(),
}))

const SETTINGS_KEY = 'taiji_terminal_allow_unauthenticated'

// 按 URL/method 分发 mock 响应；settings 与 health 的 GET 返回值可定制
function setupFetch({ settings = {}, postOk = true, resetOk = true } = {}) {
  authFetch.mockImplementation((url, opts) => {
    if (url.includes('/api/system/reset')) {
      return Promise.resolve({
        ok: resetOk,
        status: resetOk ? 200 : 500,
        json: async () =>
          resetOk
            ? { status: 'ok', message: '已清空 2 个对话会话', removed_sessions: 2 }
            : { detail: 'boom' },
      })
    }
    if (url.includes('/api/settings')) {
      if (!opts || opts.method !== 'POST') {
        return Promise.resolve({ ok: true, json: async () => settings })
      }
      return Promise.resolve({ ok: postOk, status: postOk ? 200 : 500, json: async () => ({}) })
    }
    if (url.includes('/api/health')) {
      return Promise.resolve({ ok: true, json: async () => ({ seed_active: false }) })
    }
    return Promise.resolve({ ok: true, json: async () => ({}) })
  })
}

// 切换到"运行环境"分区并返回开关 input
async function gotoTerminalToggle(wrapper) {
  const navButtons = wrapper.findAll('.settings-nav .sn-item')
  const runtimeBtn = navButtons.find((b) => b.text().includes('运行环境'))
  await runtimeBtn.trigger('click')
  return wrapper
    .find('label[aria-label="允许未认证终端访问开关"]')
    .find('input[type="checkbox"]')
}

describe('SettingsView · 允许未认证终端访问', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
    vi.clearAllMocks()
  })

  const mountView = (provide = {}) =>
    mount(SettingsView, {
      global: {
        provide: {
          toast: vi.fn(),
          $confirm: vi.fn(() => Promise.resolve(true)),
          ...provide,
        },
      },
    })

  it('渲染开关且默认为关闭（false）', async () => {
    setupFetch()
    const wrapper = mountView()
    await flushPromises()
    const toggle = await gotoTerminalToggle(wrapper)
    expect(toggle.exists()).toBe(true)
    expect(toggle.element.checked).toBe(false)
    expect(wrapper.find('.settings-section').text()).toContain('允许未认证终端访问')
  })

  it('开启开关时发起 POST /api/settings 持久化 terminal_allow_unauthenticated', async () => {
    setupFetch()
    const wrapper = mountView()
    await flushPromises()
    const toggle = await gotoTerminalToggle(wrapper)

    await toggle.setValue(true)
    await flushPromises()

    const postCall = authFetch.mock.calls.find(
      ([url, opts]) => url.includes('/api/settings') && opts && opts.method === 'POST'
    )
    expect(postCall).toBeTruthy()
    expect(JSON.parse(postCall[1].body)).toEqual({ terminal_allow_unauthenticated: true })
    expect(localStorage.getItem(SETTINGS_KEY)).toBe('true')
    expect(wrapper.find('label[aria-label="允许未认证终端访问开关"]').find('input').element.checked).toBe(true)
  })

  it('POST 失败时回滚开关并 toast 错误', async () => {
    setupFetch({ postOk: false })
    const toastFn = vi.fn()
    const wrapper = mount(SettingsView, {
      global: { provide: { toast: toastFn } },
    })
    await flushPromises()
    const toggle = await gotoTerminalToggle(wrapper)

    await toggle.setValue(true)
    await flushPromises()

    expect(wrapper.vm.terminalAllowUnauth).toBe(false)
    expect(toggle.element.checked).toBe(false)
    expect(toastFn).toHaveBeenCalledWith(expect.stringContaining('保存终端设置失败'), 'error')
    expect(localStorage.getItem(SETTINGS_KEY)).not.toBe('true')
  })

  it('进入页面时从 GET /api/settings 读取初值', async () => {
    setupFetch({ settings: { terminal_allow_unauthenticated: true } })
    const wrapper = mountView()
    await flushPromises()
    const toggle = await gotoTerminalToggle(wrapper)
    expect(toggle.element.checked).toBe(true)
    expect(localStorage.getItem(SETTINGS_KEY)).toBe('true')
  })

  it('GET 响应迟到于用户切换（POST 已成功）时不用旧值覆盖 UI（竞态防护）', async () => {
    let resolveSettingsGet = null
    authFetch.mockImplementation((url, opts) => {
      if (url.includes('/api/settings')) {
        if (!opts || opts.method !== 'POST') {
          // GET 可手动控制返回时机，模拟迟到响应（返回旧值 false）
          return new Promise((resolve) => {
            resolveSettingsGet = () =>
              resolve({ ok: true, json: async () => ({ terminal_allow_unauthenticated: false }) })
          })
        }
        return Promise.resolve({ ok: true, status: 200, json: async () => ({}) })
      }
      if (url.includes('/api/health')) {
        return Promise.resolve({ ok: true, json: async () => ({ seed_active: false }) })
      }
      return Promise.resolve({ ok: true, json: async () => ({}) })
    })
    const wrapper = mountView()
    await flushPromises()
    const toggle = await gotoTerminalToggle(wrapper)

    // 用户先切换为开启，POST 已成功（localStorage 已回写）
    await toggle.setValue(true)
    await flushPromises()
    expect(localStorage.getItem(SETTINGS_KEY)).toBe('true')

    // 此时迟到的 GET（旧值 false）才返回，不得覆盖用户选择
    resolveSettingsGet()
    await flushPromises()
    expect(wrapper.vm.terminalAllowUnauth).toBe(true)
    expect(toggle.element.checked).toBe(true)
  })
})

// ======================== 持久化控件组 ========================

describe('SettingsView · 设置组持久化（往返/回滚）', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
    vi.clearAllMocks()
  })

  const mountView = (provide = {}) =>
    mount(SettingsView, {
      global: {
        provide: {
          toast: vi.fn(),
          $confirm: vi.fn(() => Promise.resolve(true)),
          ...provide,
        },
      },
    })

  const findSettingsPost = () =>
    authFetch.mock.calls.filter(
      ([url, opts]) => url.includes('/api/settings') && opts && opts.method === 'POST'
    )

  // Taiji 运行设置分区在左侧导航切换后才渲染
  async function gotoNeuron(wrapper) {
    const navButtons = wrapper.findAll('.settings-nav .sn-item')
    await navButtons.find((b) => b.text().includes('Taiji 设置')).trigger('click')
  }

  it('切换时区 → POST body 为 { timezone }，成功后回写 localStorage', async () => {
    setupFetch()
    const wrapper = mountView()
    await flushPromises()

    const select = wrapper.find('select[aria-label="时区"]')
    await select.setValue('Asia/Tokyo')
    await flushPromises()

    const posts = findSettingsPost()
    expect(posts.length).toBe(1)
    expect(JSON.parse(posts[0][1].body)).toEqual({ timezone: 'Asia/Tokyo' })
    expect(localStorage.getItem('taiji_timezone')).toBe('Asia/Tokyo')
  })

  it('刷新后从 GET /api/settings 回读已保存值（持久化往返）', async () => {
    setupFetch({ settings: { timezone: 'Europe/Berlin', ui_density: 'compact', chat_retention_days: '180' } })
    const wrapper = mountView()
    await flushPromises()

    expect(wrapper.find('select[aria-label="时区"]').element.value).toBe('Europe/Berlin')
    expect(wrapper.find('input[name="density"][value="compact"]').element.checked).toBe(true)
    // 对话保留在“数据与隐私”分区，需先切换导航
    const navButtons = wrapper.findAll('.settings-nav .sn-item')
    await navButtons.find((b) => b.text().includes('数据与隐私')).trigger('click')
    expect(wrapper.find('select[aria-label="对话保留"]').element.value).toBe('180')
    expect(localStorage.getItem('taiji_timezone')).toBe('Europe/Berlin')
  })

  it('界面密度切换发起 POST 且失败时回滚 + toast', async () => {
    setupFetch({ postOk: false })
    const toastFn = vi.fn()
    const wrapper = mountView({ toast: toastFn })
    await flushPromises()

    const compact = wrapper.find('input[name="density"][value="compact"]')
    await compact.setValue(true)
    await flushPromises()

    expect(wrapper.vm.uiDensity).toBe('default')
    expect(compact.element.checked).toBe(false)
    expect(toastFn).toHaveBeenCalledWith(expect.stringContaining('保存设置失败'), 'error')
  })

  it('Taiji 开关类控件持久化：自动巩固/睡眠模式各自独立键', async () => {
    setupFetch()
    const wrapper = mountView()
    await flushPromises()
    await gotoNeuron(wrapper)

    const sleep = wrapper.find('label[aria-label="睡眠模式开关"]').find('input')
    await sleep.setValue(true)
    await flushPromises()
    expect(JSON.parse(findSettingsPost().at(-1)[1].body)).toEqual({ taiji_sleep_mode: true })

    const consol = wrapper.find('label[aria-label="自动巩固开关"]').find('input')
    await consol.setValue(false)
    await flushPromises()
    expect(JSON.parse(findSettingsPost().at(-1)[1].body)).toEqual({ taiji_auto_consolidation: false })
    expect(localStorage.getItem('taiji_taiji_auto_consolidation')).toBe('false')
  })

  it('响应超时变更做范围钳制后持久化', async () => {
    setupFetch()
    const wrapper = mountView()
    await flushPromises()
    await gotoNeuron(wrapper)

    const input = wrapper.find('input[aria-label="响应超时"]')
    await input.setValue(99999)
    await input.trigger('change')
    await flushPromises()

    expect(wrapper.vm.responseTimeoutMs).toBe(10000)
    expect(JSON.parse(findSettingsPost().at(-1)[1].body)).toEqual({ taiji_response_timeout_ms: 10000 })
  })

  it('语言切换持久化并双向同步 appStore.currentLang', async () => {
    setupFetch()
    const wrapper = mountView()
    await flushPromises()
    const appStore = useAppStore()

    await wrapper.find('select[aria-label="默认语言"]').setValue('en')
    await flushPromises()

    expect(JSON.parse(findSettingsPost().at(-1)[1].body)).toEqual({ ui_language: 'en' })
    expect(appStore.currentLang).toBe('en')

    await wrapper.find('select[aria-label="默认语言"]').setValue('zh-CN')
    await flushPromises()
    expect(appStore.currentLang).toBe('zh')
  })

  it('用户已修改后迟到的 GET 不得覆盖新控件组值（竞态防护）', async () => {
    let resolveSettingsGet = null
    authFetch.mockImplementation((url, opts) => {
      if (url.includes('/api/settings')) {
        if (!opts || opts.method !== 'POST') {
          return new Promise((resolve) => {
            resolveSettingsGet = () =>
              resolve({ ok: true, json: async () => ({ timezone: 'Asia/Shanghai' }) })
          })
        }
        return Promise.resolve({ ok: true, status: 200, json: async () => ({}) })
      }
      return Promise.resolve({ ok: true, json: async () => ({}) })
    })
    const wrapper = mountView()
    await flushPromises()

    await wrapper.find('select[aria-label="时区"]').setValue('Asia/Tokyo')
    await flushPromises()
    expect(wrapper.vm.timezone).toBe('Asia/Tokyo')

    resolveSettingsGet()
    await flushPromises()
    expect(wrapper.vm.timezone).toBe('Asia/Tokyo')
  })
})

// ======================== 导出 / 重置 / 许可 ========================

describe('SettingsView · 数据操作（导出/重置/许可）', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
    vi.clearAllMocks()
  })

  const mountView = (provide = {}) =>
    mount(SettingsView, {
      global: {
        provide: {
          toast: vi.fn(),
          $confirm: vi.fn(() => Promise.resolve(true)),
          ...provide,
        },
      },
    })

  async function gotoPrivacy(wrapper) {
    const navButtons = wrapper.findAll('.settings-nav .sn-item')
    const btn = navButtons.find((b) => b.text().includes('数据与隐私'))
    await btn.trigger('click')
  }

  it('导出按钮聚合设置与会话并触发 Blob 下载（文件名含日期）', async () => {
    setupFetch({ settings: { timezone: 'Asia/Shanghai' } })
    const createObjectURL = vi.fn(() => 'blob:mock-url')
    const revokeObjectURL = vi.fn()
    global.URL.createObjectURL = createObjectURL
    global.URL.revokeObjectURL = revokeObjectURL
    let downloadName = ''
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(function () {
        downloadName = this.download
      })

    try {
      const wrapper = mountView()
      await flushPromises()
      await gotoPrivacy(wrapper)
      await wrapper.find('button.btn-sm.btn-outline').trigger('click')
      await flushPromises()

      expect(createObjectURL).toHaveBeenCalledTimes(1)
      expect(clickSpy).toHaveBeenCalledTimes(1)
      expect(downloadName).toMatch(/^seed-export-\d{8}\.json$/)
      expect(revokeObjectURL).toHaveBeenCalledWith('blob:mock-url')
      // Blob 内容聚合了设置与会话列表（jsdom Blob 无 text()，用 FileReader 读取）
      const blob = createObjectURL.mock.calls[0][0]
      expect(blob).toBeInstanceOf(Blob)
      const text = await new Promise((resolve, reject) => {
        const fr = new FileReader()
        fr.onload = () => resolve(fr.result)
        fr.onerror = () => reject(fr.error)
        fr.readAsText(blob)
      })
      const parsed = JSON.parse(text)
      expect(parsed.settings).toEqual({ timezone: 'Asia/Shanghai' })
      expect(Array.isArray(parsed.chat_sessions)).toBe(true)
      expect(parsed.exported_at).toBeTruthy()
    } finally {
      clickSpy.mockRestore()
      delete global.URL.createObjectURL
      delete global.URL.revokeObjectURL
    }
  })

  it('重置按钮：二次确认后调用 POST /api/system/reset（scope=chat_sessions）', async () => {
    setupFetch()
    const confirmFn = vi.fn(() => Promise.resolve(true))
    const toastFn = vi.fn()
    const wrapper = mountView({ $confirm: confirmFn, toast: toastFn })
    await flushPromises()
    await gotoPrivacy(wrapper)

    await wrapper.find('button.btn-destructive').trigger('click')
    await flushPromises()

    expect(confirmFn).toHaveBeenCalledTimes(1)
    expect(confirmFn.mock.calls[0][0].type).toBe('danger')
    const resetCall = authFetch.mock.calls.find(([url]) => url.includes('/api/system/reset'))
    expect(resetCall).toBeTruthy()
    expect(JSON.parse(resetCall[1].body)).toEqual({ scope: 'chat_sessions' })
    expect(toastFn).toHaveBeenCalledWith(expect.stringContaining('重置完成'), 'success')
  })

  it('重置按钮：取消二次确认时不发起请求', async () => {
    setupFetch()
    const confirmFn = vi.fn(() => Promise.resolve(false))
    const wrapper = mountView({ $confirm: confirmFn })
    await flushPromises()
    await gotoPrivacy(wrapper)

    await wrapper.find('button.btn-destructive').trigger('click')
    await flushPromises()

    expect(confirmFn).toHaveBeenCalledTimes(1)
    expect(authFetch.mock.calls.find(([url]) => url.includes('/api/system/reset'))).toBeUndefined()
  })

  it('重置接口失败时 toast 错误', async () => {
    setupFetch({ resetOk: false })
    const toastFn = vi.fn()
    const wrapper = mountView({ toast: toastFn })
    await flushPromises()
    await gotoPrivacy(wrapper)

    await wrapper.find('button.btn-destructive').trigger('click')
    await flushPromises()

    expect(toastFn).toHaveBeenCalledWith(expect.stringContaining('重置失败'), 'error')
  })

  it('开源许可按钮打开弹窗，展示 Apache-2.0 声明，可关闭', async () => {
    setupFetch()
    const wrapper = mountView()
    await flushPromises()

    const navButtons = wrapper.findAll('.settings-nav .sn-item')
    await navButtons.find((b) => b.text().includes('关于')).trigger('click')
    expect(wrapper.find('.license-overlay').exists()).toBe(false)

    await wrapper.find('button.btn-sm.btn-ghost').trigger('click')
    expect(wrapper.find('.license-overlay').exists()).toBe(true)
    expect(wrapper.find('.license-body').text()).toContain('Apache License 2.0')
    expect(wrapper.find('.license-body').text()).toContain('NeuroPlex Contributors')

    await wrapper.find('.license-close').trigger('click')
    expect(wrapper.find('.license-overlay').exists()).toBe(false)
  })
})
