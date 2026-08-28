/**
 * 标签页状态 composable —— 「常驻 + 零动画 + URL 同步」
 *
 * 设计动机（见 plans/active/SEED_DEVELOPMENT_ROADMAP_2026_08.md「标签页交互收敛」）：
 *   1. 常驻：面板一律用 display 切换而非 v-if。v-if 会销毁并重建 DOM，
 *      导致滚动位置、展开状态、输入内容全部丢失，观感上就是"刷新一下"。
 *      同时 v-if + 淡入会让任何真实渲染错误看起来像动画卡顿，掩盖白屏类故障。
 *   2. 零动画：标签内容不是"从别处来的"，它一直就在那里，切换应当是 0ms。
 *      VS Code / Chrome / 主流客户端皆如此。
 *   3. URL 同步：标签状态写入 query 参数，可深链、可前进后退、刷新后保持。
 *
 * 用法：
 *   const { activeTab, isActive, selectTab, onTablistKeydown } = useTabs(['files', 'config', 'test'])
 */
import { onActivated, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

export function useTabs(tabIds, options = {}) {
  const queryKey = options.queryKey || 'tab'
  const fallback = options.default || tabIds[0]

  // 无 vue-router 环境（单元测试/独立挂载）降级为纯状态模式：不做 URL 同步
  const route = useRoute?.() ?? null
  const router = useRouter?.() ?? null
  const syncable = Boolean(route && router && route.query)
  const activeTab = ref(fallback)

  const isValid = (id) => tabIds.includes(id)

  const syncFromUrl = () => {
    if (!syncable) return
    const raw = route.query[queryKey]
    const id = Array.isArray(raw) ? raw[0] : raw
    if (isValid(id) && id !== activeTab.value) activeTab.value = id
  }

  const selectTab = (id) => {
    if (!isValid(id) || id === activeTab.value) return
    activeTab.value = id
  }

  const isActive = (id) => activeTab.value === id

  // 键盘可达性：左右方向键在标签间移动，Home/End 跳到首尾
  const onTablistKeydown = (event) => {
    const keys = { ArrowLeft: -1, ArrowRight: 1 }
    let next = null

    if (event.key in keys) {
      const cur = tabIds.indexOf(activeTab.value)
      next = tabIds[(cur + keys[event.key] + tabIds.length) % tabIds.length]
    } else if (event.key === 'Home') {
      next = tabIds[0]
    } else if (event.key === 'End') {
      next = tabIds[tabIds.length - 1]
    }

    if (!next) return
    event.preventDefault()
    selectTab(next)
    // 焦点跟随选中项，符合 WAI-ARIA tablist 的手动激活模式
    const el = event.currentTarget?.querySelector(`[data-tab-id="${next}"]`)
    el?.focus?.()
  }

  // activeTab → URL：用 replace 避免污染后退栈里堆满同页记录
  if (syncable) {
    watch(activeTab, (id) => {
      if (route.query[queryKey] === id) return
      router.replace({ query: { ...route.query, [queryKey]: id } }).catch(() => {})
    })

    // URL → activeTab：覆盖浏览器前进/后退与外部深链
    watch(() => route.query[queryKey], syncFromUrl)
  }

  onMounted(syncFromUrl)
  // keep-alive 下 onMounted 只触发一次；从别页切回时需重新对齐 URL
  onActivated(() => {
    if (!syncable) return
    const raw = route.query[queryKey]
    const id = Array.isArray(raw) ? raw[0] : raw
    if (isValid(id)) activeTab.value = id
    else router.replace({ query: { ...route.query, [queryKey]: activeTab.value } }).catch(() => {})
  })

  return { activeTab, isActive, selectTab, onTablistKeydown }
}
