import { describe, it, expect, beforeEach, vi } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'
import { setActivePinia, createPinia } from 'pinia'
import { createRouter, createMemoryHistory } from 'vue-router'
import AppSidebar from '../components/AppSidebar.vue'
import { useChatStore } from '../stores/chatStore.js'

// F04: AppSidebar 组件测试——导航完整性、会话区与新建对话
describe('AppSidebar', () => {
  let router
  let chatStore

  beforeEach(() => {
    setActivePinia(createPinia())
    chatStore = useChatStore()
    const Dummy = { render: () => null }
    router = createRouter({
      history: createMemoryHistory(),
      routes: [
        { path: '/', component: Dummy },
        { path: '/workspace', component: Dummy },
        { path: '/agent', component: Dummy },
        { path: '/kb', component: Dummy },
        { path: '/train', component: Dummy },
        { path: '/life', component: Dummy },
        { path: '/settings', component: Dummy },
      ],
    })
  })

  const mountSidebar = async () => {
    await router.push('/')
    await router.isReady()
    return mount(AppSidebar, { global: { plugins: [router] } })
  }

  it('渲染侧边栏主结构（品牌区 / 搜索 / 新建对话 / 会话列表 / 导航）', async () => {
    const wrapper = await mountSidebar()
    expect(wrapper.find('aside.sidebar').exists()).toBe(true)
    expect(wrapper.find('.sidebar-header h2').exists()).toBe(true)
    expect(wrapper.find('.search-field input').exists()).toBe(true)
    expect(wrapper.find('.new-chat-btn').exists()).toBe(true)
    expect(wrapper.find('[aria-label="会话列表"]').exists()).toBe(true)
    expect(wrapper.find('nav[aria-label="主导航"]').exists()).toBe(true)
  })

  it('导航包含全部 6 个入口', async () => {
    const wrapper = await mountSidebar()
    const links = wrapper.findAll('nav a.nav-item')
    expect(links.length).toBe(6)
    const hrefs = links.map((l) => l.attributes('href'))
    for (const path of ['/workspace', '/agent', '/kb', '/train', '/life', '/settings']) {
      expect(hrefs).toContain(path)
    }
  })

  it('当前路由的导航项带 active 类', async () => {
    const wrapper = await mountSidebar()
    await router.push('/settings')
    await wrapper.vm.$nextTick()
    const active = wrapper.findAll('nav a.nav-item.active')
    expect(active.length).toBe(1)
    expect(active[0].attributes('href')).toBe('/settings')
  })

  it('点击新建对话调用 createNewSession 并跳转聊天页', async () => {
    const wrapper = await mountSidebar()
    const spy = vi.spyOn(chatStore, 'createNewSession').mockImplementation(() => {})
    await router.push('/settings')
    await wrapper.vm.$nextTick()

    await wrapper.find('.new-chat-btn').trigger('click')
    await flushPromises()

    expect(spy).toHaveBeenCalled()
    expect(router.currentRoute.value.path).toBe('/')
  })

  it('渲染会话列表中的会话项', async () => {
    const wrapper = await mountSidebar()
    // 直接向 store 注入会话（绕过 API）
    chatStore.sessions = [{ id: 's1', name: '测试会话' }]
    chatStore.sessionsLoaded = true
    await flushPromises()

    expect(chatStore.sessions.length).toBe(1)
    const items = wrapper.findAll('.session-item')
    expect(items.length).toBe(1)
    expect(items[0].text()).toContain('测试会话')
  })
})
