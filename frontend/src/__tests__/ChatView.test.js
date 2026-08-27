import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import ChatView from '../components/ChatView.vue'
import { useChatStore } from '../stores/chatStore.js'
import { useRuntimeStore } from '../stores/runtimeStore.js'

// R4: 隔离网络——挂载与渲染不应触发真实请求；
// 同时作为 authFetch 的固定档供附件上传失败路径测试使用。
const { mockAuthFetch } = vi.hoisted(() => ({
  mockAuthFetch: vi.fn(() =>
    Promise.resolve({ ok: false, status: 503, json: async () => ({}) })
  ),
}))
vi.mock('../composables/apiClient.js', () => ({
  API_BASE: '',
  authFetch: mockAuthFetch,
}))

describe('ChatView', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    mockAuthFetch.mockClear()
  })

  // toast 兼容组件内两种调用形态：函数式 toast(msg, type) 与对象式 toast.error(...)。
  const makeToast = () => Object.assign(vi.fn(), { success: vi.fn(), error: vi.fn(), info: vi.fn() })

  const mountView = (toast = makeToast()) =>
    mount(ChatView, {
      global: {
        provide: {
          toast,
        },
      },
    })

  it('渲染顶栏与生命体征区', () => {
    const wrapper = mountView()
    expect(wrapper.find('main.chat-workbench').exists()).toBe(true)
    expect(wrapper.find('header.topbar').exists()).toBe(true)
    expect(wrapper.find('[aria-label="生命体征"]').exists()).toBe(true)
  })

  it('无消息时显示欢迎区与建议词', () => {
    const wrapper = mountView()
    expect(wrapper.find('.chat-welcome').exists()).toBe(true)
    expect(wrapper.find('.chat-welcome h1').text()).toContain('有什么我能帮你的吗')
    expect(wrapper.findAll('.suggestions .suggestion').length).toBeGreaterThan(0)
    // 示例对话入口（默认收起的折叠分割线）
    expect(wrapper.find('.thread-divider').text()).toContain('示例对话')
  })

  it('示例对话默认收起，点击后展开首段示例', async () => {
    const wrapper = mountView()
    const toggle = wrapper.find('.thread-divider.example-toggle')
    expect(toggle.exists()).toBe(true)
    // 收起态：不渲染示例消息与假交互按钮
    expect(wrapper.find('.chat-thread-example .msg').exists()).toBe(false)
    expect(wrapper.find('.chat-thread-example .msg-action-btn').exists()).toBe(false)

    await toggle.trigger('click')
    expect(wrapper.findAll('.chat-thread-example .msg').length).toBe(2)
    // 展开后不含第二段示例的代码块内容
    expect(wrapper.html()).not.toContain('局部可塑性没有产生有效学习')

    await toggle.trigger('click')
    expect(wrapper.find('.chat-thread-example .msg').exists()).toBe(false)
  })

  it('点击建议词把文本填入输入框', async () => {
    const wrapper = mountView()
    const chatStore = useChatStore()
    const first = wrapper.find('.suggestions .suggestion')
    await first.trigger('click')
    expect(chatStore.chatInput).toBe(first.text().trim())
  })

  it('输入为空时发送按钮不可用，运行时就绪且输入后可用', async () => {
    const wrapper = mountView()
    const chatStore = useChatStore()
    const runtimeStore = useRuntimeStore()
    const sendBtn = wrapper.find('button.send')
    // disabled 已移除：未就绪时以 unavailable 类门控（点击后 toast 解释原因）
    expect(sendBtn.classes()).toContain('unavailable')

    // 仅输入还不够——还需运行时已连接且模型已加载（canSend 的完整语义）
    chatStore.chatInput = '你好，Seed'
    await wrapper.vm.$nextTick()
    expect(sendBtn.classes()).toContain('unavailable')

    runtimeStore.health.state = 'connected'
    runtimeStore.health.modelLoaded = true
    await wrapper.vm.$nextTick()
    expect(sendBtn.classes()).not.toContain('unavailable')
  })

  it('存在消息时渲染对话列表并隐藏欢迎区', async () => {
    const chatStore = useChatStore()
    chatStore.messages.push({ id: 1, role: 'user', content: '你好' })
    chatStore.messages.push({ id: 2, role: 'assistant', content: '你好，有什么能帮你？' })

    const wrapper = mountView()
    await wrapper.vm.$nextTick()
    expect(wrapper.find('.chat-welcome').exists()).toBe(false)
    const msgs = wrapper.findAll('article.msg')
    expect(msgs.length).toBe(2)
    expect(msgs[0].classes()).toContain('msg-user')
    expect(msgs[1].classes()).toContain('msg-ai')
  })

  it('composer 输入区与功能按钮齐备', () => {
    const wrapper = mountView()
    expect(wrapper.find('.composer textarea').exists()).toBe(true)
    expect(wrapper.findAll('.composer-chip').length).toBeGreaterThanOrEqual(4)
  })

  it('知识库/图像生成/更多无后端支撑的入口已下线', () => {
    const wrapper = mountView()
    expect(wrapper.find('.composer-chip[title="知识库"]').exists()).toBe(false)
    expect(wrapper.find('.composer-chip[title="图像生成"]').exists()).toBe(false)
    expect(wrapper.find('.composer-chip[title="更多"]').exists()).toBe(false)
  })

  it('每个 composer chip 均有真实行为：点击后输入框内容变化', async () => {
    const wrapper = mountView()
    const chatStore = useChatStore()

    // 代码/总结/翻译：插入提示词模板，输入框内容变化
    for (const title of ['代码', '总结', '翻译']) {
      chatStore.chatInput = ''
      await wrapper.find(`.composer-chip[title="${title}"]`).trigger('click')
      expect(chatStore.chatInput.length).toBeGreaterThan(0)
    }
    // 代码模板包含代码块引导
    chatStore.chatInput = ''
    await wrapper.find('.composer-chip[title="代码"]').trigger('click')
    expect(chatStore.chatInput).toContain('```')

    // 快速：展开快捷面板，选中提示词后填入输入框并收起面板
    chatStore.chatInput = ''
    await wrapper.find('.composer-chip[title="快速"]').trigger('click')
    expect(wrapper.find('.quick-panel').exists()).toBe(true)
    const hintText = wrapper.find('.quick-panel .quick-item').text().trim()
    await wrapper.find('.quick-panel .quick-item').trigger('click')
    expect(chatStore.chatInput).toBe(hintText)
    expect(wrapper.find('.quick-panel').exists()).toBe(false)
  })

  it('模板插入为追加语义：已有输入不丢失', async () => {
    const wrapper = mountView()
    const chatStore = useChatStore()
    chatStore.chatInput = '已有内容'
    await wrapper.find('.composer-chip[title="总结"]').trigger('click')
    expect(chatStore.chatInput).toContain('已有内容')
    expect(chatStore.chatInput).toContain('请帮我总结以下内容')
  })

  it('添加 chip 触发文件选择，并经 /api/chat/upload 处理附件', async () => {
    const toast = makeToast()
    const wrapper = mountView(toast)
    const fileInput = wrapper.find('input[type="file"]')
    expect(fileInput.exists()).toBe(true)

    // 点击"添加附件"打开文件选择器（真实入口而非占位 toast）
    const clickSpy = vi.spyOn(fileInput.element, 'click')
    await wrapper.find('.composer-chip[title="添加附件"]').trigger('click')
    expect(clickSpy).toHaveBeenCalled()

    // 模拟选中文件：隔离网络下上传失败 → 错误 toast，不产生假内容注入
    const file = new File(['hello'], 'note.txt', { type: 'text/plain' })
    Object.defineProperty(fileInput.element, 'files', { value: [file], configurable: true })
    await fileInput.trigger('change')
    await flushPromises()
    expect(mockAuthFetch).toHaveBeenCalledWith(
      '/api/chat/upload',
      expect.objectContaining({ method: 'POST' })
    )
    expect(toast).toHaveBeenCalledWith(expect.stringContaining('附件上传失败'), 'error')
  })
})
