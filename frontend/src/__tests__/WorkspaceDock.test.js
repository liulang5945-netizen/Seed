import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { createPinia } from 'pinia'
import WorkspaceDock from '../components/WorkspaceDock.vue'
import { authFetch } from '../composables/apiClient.js'

// 隔离网络——所有请求经 mock 路由，不发真实请求
vi.mock('../composables/apiClient.js', () => ({
  API_BASE: '',
  authFetch: vi.fn(),
}))

const jsonResponse = (data, ok = true, status = 200) => ({
  ok,
  status,
  json: async () => data,
})

const defaultCapabilities = () => ({
  format: 'seed-workbench-contract-v1',
  version: 1,
  snapshot_id: 'workbench-test-snapshot',
  revision: 1,
  workspace_root: 'E:/Seed/agent_workspace',
  capabilities: [
    { capability_id: 'workspace.list', enabled: true },
    { capability_id: 'workspace.read', enabled: true },
    { capability_id: 'workspace.create', enabled: true },
    { capability_id: 'workspace.rename', enabled: true },
    { capability_id: 'workspace.delete', enabled: true },
    { capability_id: 'workspace.apply_patch', enabled: true },
    { capability_id: 'terminal.run', enabled: true },
  ],
})

const defaultEntries = (path = '.') => path === 'src'
  ? [{ name: 'main.py', path: 'src/main.py', type: 'file', size: 10 }]
  : [
      { name: 'src', path: 'src', type: 'directory', size: 0 },
      { name: 'README.md', path: 'README.md', type: 'file', size: 4 },
    ]

// 原生重命名预览的 mock 响应，可按用例覆写（默认成功）
let renameResponse = () => jsonResponse({ status: 'ok' })
const mountedWrappers = []

beforeEach(() => {
  // Dock 初始为分栏档（展开态触发 beginOpen：挂载 body + loadTree）
  localStorage.setItem('taiji_dock_mode', 'split')
  localStorage.removeItem('taiji_dock_width')
  authFetch.mockReset()
  renameResponse = () => jsonResponse({ status: 'ok', path: 'GUIDE.md' })
  authFetch.mockImplementation((url, options = {}) => {
    const method = (options.method || 'GET').toUpperCase()
    if (url.endsWith('/api/workbench/capabilities')) {
      return Promise.resolve(jsonResponse(defaultCapabilities()))
    }
    if (url.endsWith('/api/workbench/events')) {
      return Promise.resolve(jsonResponse({ events: [] }))
    }
    if (url.includes('/api/workbench/files?path=')) {
      const path = new URL(url, 'http://localhost').searchParams.get('path') || '.'
      return Promise.resolve(jsonResponse({ entries: defaultEntries(path) }))
    }
    if (url.endsWith('/api/system/quick_paths')) {
      return Promise.resolve(
        jsonResponse({ paths: [{ label: '桌面', path: 'C:/Users/x/Desktop' }] })
      )
    }
    if (url.endsWith('/api/system/select_folder')) {
      return Promise.resolve(jsonResponse({ status: 'cancel' }))
    }
    if (url.endsWith('/api/workbench/workspace') && method === 'POST') {
      const body = JSON.parse(options.body || '{}')
      return Promise.resolve(jsonResponse({ status: 'ok', path: body.path }))
    }
    if (url.includes('/api/workbench/file?path=')) {
      return Promise.resolve(jsonResponse({ content: 'hello\n', encoding: 'utf-8', digest: 'a'.repeat(64), truncated: false }))
    }
    if (url.endsWith('/api/workbench/preview') && method === 'POST') {
      const body = JSON.parse(options.body || '{}')
      if (body.kind === 'workspace.rename') {
        const rename = renameResponse()
        if (!rename.ok) return Promise.resolve(rename)
        return Promise.resolve(jsonResponse({
          policy: { decision: 'ask_user', reason_code: 'capability_requires_approval' },
          preview: { capability_id: body.kind, mutation: { operation: body.kind, path: body.parameters?.path || '' } },
          approval: { approval_token: 'test-approval-token' },
        }))
      }
      return Promise.resolve(jsonResponse({
        policy: { decision: 'ask_user', reason_code: 'capability_requires_approval' },
        preview: { capability_id: body.kind, mutation: { operation: body.kind, path: body.parameters?.path || '' } },
        approval: { approval_token: 'test-approval-token' },
      }))
    }
    if (url.endsWith('/api/workbench/execute') && method === 'POST') {
      const body = JSON.parse(options.body || '{}')
      const result = body.kind === 'workspace.rename'
        ? { path: 'README.md', new_path: 'GUIDE.md', digest: 'a'.repeat(64) }
        : { path: body.parameters?.path || 'new.py', digest: 'a'.repeat(64), success: true }
      return Promise.resolve(jsonResponse({ outcome: { success: true, result } }))
    }
    return Promise.resolve(jsonResponse({}))
  })
})

afterEach(() => {
  for (const wrapper of mountedWrappers.splice(0)) wrapper.unmount()
  localStorage.removeItem('taiji_dock_mode')
  localStorage.removeItem('taiji_dock_width')
})

// MonacoEditor 依赖真实编辑器运行时，stub 后仅验证视图接线；
// Dock 以展开态挂载（split）触发 beginOpen——body 挂载、loadTree 单发
const mountDock = ({ toast = vi.fn() } = {}) => {
  const wrapper = mount(WorkspaceDock, {
    global: {
      plugins: [createPinia()],
      provide: {
        toast,
        $confirm: vi.fn(() => Promise.resolve(true)),
      },
      stubs: { MonacoEditor: true, WebTerminal: true },
    },
  })
  mountedWrappers.push(wrapper)
  return wrapper
}

const treeCalls = () =>
  authFetch.mock.calls.filter(([u]) => u.includes('/api/workbench/files?path=')).length

// 右键重命名交互流程：打开右键菜单 → 点击重命名 → 输入新名 → 确认
const runRenameFlow = async (wrapper, newName) => {
  const item = wrapper.findAll('.tree-item').find((i) => i.text() === 'README.md')
  await item.trigger('contextmenu')
  await flushPromises()
  const ctxRename = wrapper
    .findAll('.ctx-item')
    .find((i) => i.text().includes('重命名'))
  expect(ctxRename).toBeTruthy()
  await ctxRename.trigger('click')
  await flushPromises()
  expect(wrapper.find('.dlg-box h3').text()).toBe('重命名')
  await wrapper.find('.dlg-input').setValue(newName)
  await wrapper.find('.dlg-btn.primary').trigger('click')
  await flushPromises()
}

const renameCalls = () =>
  authFetch.mock.calls.filter(
    ([u, o]) => u.endsWith('/api/workbench/preview') && (o?.method || '') === 'POST'
  )

const findButtonByTitle = (wrapper, title) =>
  wrapper.findAll('button').find((b) => b.attributes('title') === title)

describe('WorkspaceDock', () => {
  it('展开态挂载 loadTree 只发一次（beginOpen 统一负责）', async () => {
    mountDock()
    await flushPromises()
    expect(treeCalls()).toBe(1)
  })

  it('保存失败时 toast 后端返回的 detail', async () => {
    const toastFn = vi.fn()
    const wrapper = mountDock({ toast: toastFn })
    await flushPromises()
    wrapper.findComponent({ name: 'MonacoEditor' }).vm.$emit('save-error', '无权限写入')
    expect(toastFn).toHaveBeenCalledWith('无权限写入', 'error')
  })

  it('工具条显示当前工作区路径，点击打开文件夹打开对话框并惰性加载快捷路径', async () => {
    const wrapper = mountDock()
    await flushPromises()
    expect(wrapper.find('.dock-path').text()).toBe('E:/Seed/agent_workspace')
    expect(wrapper.find('.dlg-overlay').exists()).toBe(false)

    const btn = findButtonByTitle(wrapper, '打开文件夹')
    expect(btn).toBeTruthy()
    await btn.trigger('click')
    expect(wrapper.find('.dlg-box h3').text()).toBe('打开项目文件夹')
    await flushPromises()
    expect(
      authFetch.mock.calls.some(([u]) => u.endsWith('/api/system/quick_paths'))
    ).toBe(true)
    expect(wrapper.find('.qp-btn').exists()).toBe(true)
  })

  it('切换目录成功后工具条路径更新并重新加载文件树', async () => {
    const wrapper = mountDock()
    await flushPromises()
    const before = treeCalls()
    expect(before).toBeGreaterThan(0)

    const btn = findButtonByTitle(wrapper, '打开文件夹')
    await btn.trigger('click')
    await wrapper.find('.dlg-box .dlg-input').setValue('E:/Seed/data')
    await wrapper.find('.dlg-btn.primary').trigger('click')
    await flushPromises()

    expect(treeCalls()).toBeGreaterThan(before)
    expect(wrapper.find('.dock-path').text()).toBe('E:/Seed/data')
    expect(wrapper.find('.dlg-overlay').exists()).toBe(false)
  })

  it('终端按钮切换底部终端面板显隐且激活态高亮', async () => {
    const wrapper = mountDock()
    await flushPromises()
    expect(wrapper.find('.ide-terminal').exists()).toBe(false)

    const btn = findButtonByTitle(wrapper, '终端 (Ctrl+`)')
    expect(btn.classes()).not.toContain('active')
    await btn.trigger('click')
    expect(wrapper.find('.ide-terminal').exists()).toBe(true)
    expect(btn.classes()).toContain('active')

    await btn.trigger('click')
    expect(wrapper.find('.ide-terminal').exists()).toBe(false)
    expect(btn.classes()).not.toContain('active')
  })

  it('展开目录通过 native Workbench 读取子目录，折叠只本地重算', async () => {
    const wrapper = mountDock()
    await flushPromises()
    const before = authFetch.mock.calls.length
    expect(wrapper.findAll('.tree-item').some((i) => i.text() === 'main.py')).toBe(false)

    const folder = wrapper.findAll('.tree-item').find((i) => i.text() === 'src')
    await folder.trigger('click')
    await flushPromises()
    expect(authFetch.mock.calls.length).toBeGreaterThan(before)
    expect(wrapper.findAll('.tree-item').some((i) => i.text() === 'main.py')).toBe(true)

    const afterExpand = authFetch.mock.calls.length
    await folder.trigger('click')
    expect(authFetch.mock.calls.length).toBe(afterExpand)
    expect(wrapper.findAll('.tree-item').some((i) => i.text() === 'main.py')).toBe(false)
  })

  it('右栏工作区统计基于文件树递归计算', async () => {
    const wrapper = mountDock()
    await flushPromises()
    const statsGroup = wrapper
      .findAll('.panel-right .prop-group')
      .find((g) => g.text().includes('工作区统计'))
    expect(statsGroup).toBeTruthy()
    expect(statsGroup.text()).toContain('文件数')
    // Native Workbench loads child directories on demand; the root projection
    // therefore reports the entries currently known to the client.
    expect(statsGroup.text()).toContain('1')
    expect(statsGroup.text()).toContain('目录数')
    expect(statsGroup.text()).toContain('1')
  })

  it('右键重命名：预填当前名，成功后重新加载文件树并 toast', async () => {
    const toastFn = vi.fn()
    const wrapper = mountDock({ toast: toastFn })
    await flushPromises()
    const before = treeCalls()

    await runRenameFlow(wrapper, 'GUIDE.md')

    const calls = renameCalls()
    expect(calls.length).toBe(1)
    expect(JSON.parse(calls[0][1].body)).toMatchObject({
      kind: 'workspace.rename',
      parameters: { path: 'README.md', new_path: 'GUIDE.md', before_digest: 'a'.repeat(64) },
    })
    expect(treeCalls()).toBeGreaterThan(before)
    expect(toastFn).toHaveBeenCalledWith('已重命名为 GUIDE.md', 'success')
    expect(wrapper.find('.dlg-overlay').exists()).toBe(false)
  })

  it('右键重命名失败时 toast 后端 detail（如 409 目标已存在）', async () => {
    renameResponse = () => jsonResponse({ detail: '目标已存在: GUIDE.md' }, false, 409)
    const toastFn = vi.fn()
    const wrapper = mountDock({ toast: toastFn })
    await flushPromises()

    await runRenameFlow(wrapper, 'GUIDE.md')

    expect(renameCalls().length).toBe(1)
    expect(toastFn).toHaveBeenCalledWith('重命名失败: 目标已存在: GUIDE.md', 'error')
  })

  it('右键重命名未改名（新名与原名相同）时不发请求', async () => {
    const wrapper = mountDock()
    await flushPromises()

    await runRenameFlow(wrapper, 'README.md')

    expect(renameCalls().length).toBe(0)
  })
})
