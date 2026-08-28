import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { API_BASE, authFetch } from '@/composables/apiClient.js'

export const useRuntimeStore = defineStore('runtime', () => {
  if (typeof window !== 'undefined' && !window.__taijiRuntimeAuthListener) {
    window.__taijiRuntimeAuthListener = true
    window.addEventListener('taiji-auth-expired', (event) => {
      const message = event.detail?.message || 'JWT token 缺失或已过期，请重新登录'
      try {
        useRuntimeStore().reportAuthExpired(message)
      } catch (e) { console.debug('[runtimeStore] auth-expired handler error:', e.message) }
    })
  }

  const health = ref({
    state: 'unknown',
    message: '',
    modelLoaded: false,
    languageProvider: {
      mode: 'unknown',
      state: 'unknown',
      provider: '',
      backend_id: '',
      artifact_id: '',
      reason_code: '',
      reason: '',
      rollback: 'native-readable',
    },
    checkedAt: 0,
  })
  const memory = ref(null)
  const auth = ref({
    enabled: false,
    authenticated: !!localStorage.getItem('jwt_token'),
    username: '',
    error: '',
  })
  const terminal = ref({
    status: 'disconnected',
    error: '',
    lastEventAt: 0,
  })
  const life = ref({
    is_running: false,
    needs: {},
    total_interactions: 0,
    uptime_seconds: 0,
  })
  const tools = ref([])
  const toolsLoading = ref(false)
  const toolError = ref('')
  const exceptions = ref([])
  const logs = ref([])
  const MAX_LOGS = 200
  const runtimeSnapshot = ref(null)

  function freshnessFor(timestamp) {
    const observedAt = Number(timestamp || 0)
    if (!observedAt) return { label: '未知', state: 'unknown', ageSeconds: null }
    const ageSeconds = Math.max(0, Math.floor(Date.now() / 1000 - observedAt))
    if (ageSeconds <= 15) return { label: '刚刚', state: 'fresh', ageSeconds }
    if (ageSeconds <= 60) return { label: `${ageSeconds} 秒前`, state: 'fresh', ageSeconds }
    if (ageSeconds <= 300) return { label: `${Math.floor(ageSeconds / 60)} 分钟前`, state: 'stale', ageSeconds }
    return { label: '超过 5 分钟', state: 'stale', ageSeconds }
  }

  const statusEvidence = computed(() => {
    const snapshot = runtimeSnapshot.value || {}
    const timestamp = Number(snapshot.timestamp || 0)
    const healthSnapshot = snapshot.health || health.value || {}
    const toolsSnapshot = snapshot.tools || {}
    const provider = healthSnapshot.language_provider || health.value.languageProvider || {}
    const needs = snapshot.life?.needs || life.value?.needs || {}
    const trainingSnapshot = snapshot.training
    const workbenchTools = tools.value
    const enabledTools = workbenchTools.filter(tool => tool.enabled !== false).length
    const providerState = String(provider.state || '').toLowerCase()
    const providerReady = Boolean(provider.backend_id || provider.artifact_id)
      && ['active', 'ready'].includes(providerState)
    const runtimeReady = healthSnapshot.state === 'connected' && Boolean(healthSnapshot.model_loaded)
    const workbenchStatus = toolsSnapshot.status || (workbenchTools.length ? 'ok' : 'unknown')
    const baseFreshness = freshnessFor(timestamp)
    const evidence = (label, source, owner, availability, detail = '', observedAt = timestamp) => ({
      label,
      source,
      owner,
      availability,
      detail,
      observedAt,
      freshness: freshnessFor(observedAt),
    })

    return {
      timestamp,
      freshness: baseFreshness,
      runtime: evidence(
        '运行时',
        '/api/runtime/status.health',
        healthSnapshot.is_taiji ? 'Taiji runtime' : 'Seed runtime',
        runtimeReady ? '可用' : (healthSnapshot.state || '未连接'),
        healthSnapshot.model_name || '未提供运行时名称',
      ),
      provider: evidence(
        '语言器官',
        '/api/runtime/status.health.language_provider',
        'Seed language provider',
        providerReady
          ? (String(provider.chat_enabled) === 'true' ? '可用' : '已接入（可读）')
          : (providerState === 'fallback' ? '已回退' : '未接入'),
        provider.backend_id || provider.artifact_id || '未提供 provider artifact',
      ),
      workbench: evidence(
        '工作台能力',
        '/api/runtime/status.tools',
        toolsSnapshot.owner || 'Taiji native Workbench',
        workbenchStatus === 'ok'
          ? (enabledTools ? `${enabledTools} 项可用` : '已加载（0 项）')
          : '不可用',
        toolsSnapshot.snapshot_id
          ? `snapshot ${toolsSnapshot.snapshot_id.slice(0, 12)} · revision ${toolsSnapshot.revision ?? 0}`
          : '未提供 capability snapshot',
        Number(toolsSnapshot.observed_at || timestamp),
      ),
      homeostasis: evidence(
        'homeostasis / self-state',
        '/api/runtime/status.life',
        'Taiji self-state channel',
        Object.keys(needs).length ? `${Object.keys(needs).length} 项已上报` : '未上报',
        Object.keys(needs).length ? 'needs 来自运行时状态快照' : '运行时尚未提供 needs 字段',
      ),
      training: evidence(
        '训练状态',
        '/api/runtime/status.training',
        'Seed native training state',
        trainingSnapshot
          ? (trainingSnapshot.is_training ? '训练中' : '已上报（空闲）')
          : '未上报',
        trainingSnapshot ? '只反映运行时状态，不代表检查点已保存' : '运行时尚未提供训练状态',
      ),
      knowledge: evidence(
        '知识能力',
        '/api/runtime/status.tools',
        toolsSnapshot.owner || 'Taiji native capability registry',
        workbenchTools.filter(tool => /knowledge|memory|document|kb|知识|记忆|文档/.test(
          `${tool.name || ''} ${tool.category || ''}`.toLowerCase(),
        )).length
          ? '已接入'
          : '未接入',
        '仅依据当前 capability snapshot，不推断 Legacy RAG 能力',
        Number(toolsSnapshot.observed_at || timestamp),
      ),
    }
  })

  const connectionClass = computed(() => {
    if (health.value.state === 'connected') return 'connected'
    if (health.value.state === 'downloading') return 'downloading'
    if (health.value.state === 'loading') return 'loading'
    if (health.value.state === 'connecting' || health.value.state === 'unknown') return 'connecting'
    return 'error'
  })

  const connectionStatus = computed(() => {
    if (health.value.state === 'connected') return health.value.modelLoaded ? '运行时已连接' : '已连接（运行时未激活）'
    if (health.value.state === 'downloading') return '运行时资源准备中'
    if (health.value.state === 'loading') return '运行时加载中'
    if (health.value.state === 'connecting' || health.value.state === 'unknown') return '正在连接'
    return '未连接'
  })

  const memoryLevel = computed(() => memory.value?.level ?? null)
  const memoryAvailableGb = computed(() => memory.value?.available_gb ?? null)
  const memoryAvailablePct = computed(() => memory.value?.available_pct ?? null)

  const modelLifecycle = computed(() => {
    if (health.value.state === 'connected' && health.value.modelLoaded) {
      return {
        state: 'ready',
        title: 'Taiji 运行时已就绪',
        message: 'Seed 原生运行时已连接，客户端只展示已上报的能力。',
        canDo: ['对话', '原生工作台能力', '语言识别'],
      }
    }
    if (health.value.state === 'connected') {
      return {
        state: 'waiting',
        title: '后端在线，原生运行时待激活',
        message: memoryAvailableGb.value !== null
          ? `当前可用内存 ${memoryAvailableGb.value.toFixed(1)}GB，正在等待原生运行时激活。`
          : '后端已在线，正在等待原生运行时激活。',
        canDo: ['查看运行时状态', '等待激活'],
      }
    }
    if (health.value.state === 'downloading') {
      return {
        state: 'downloading',
        title: '运行时资源准备中',
        message: health.value.message || '资源准备完成后会进入原生运行时激活流程。',
        canDo: ['等待资源准备完成'],
      }
    }
    if (health.value.state === 'loading') {
      return {
        state: 'loading',
        title: '原生运行时正在加载',
        message: health.value.message || '正在加载 Taiji 原生运行时，请稍候。',
        canDo: ['等待加载完成'],
      }
    }
    if (health.value.state === 'connecting' || health.value.state === 'unknown') {
      return {
        state: 'connecting',
        title: '正在连接本地运行时',
        message: '后端服务启动后，对话和工具调用会自动恢复。',
        canDo: ['等待连接'],
      }
    }
    return {
      state: 'error',
      title: '运行时未连接',
      message: health.value.message || '请检查后端服务、登录状态或点击重连。',
      canDo: ['检查后端服务', '重新连接'],
    }
  })

  const runtimeNotice = computed(() => {
    if (modelLifecycle.value.state === 'ready') return null
    return modelLifecycle.value
  })

  const languageProviderNotice = computed(() => {
    const provider = health.value.languageProvider || {}
    if (provider.state !== 'fallback') return null
    return {
      state: 'warning',
      title: '语言器官已回退',
      message: provider.reason
        ? `${provider.reason} 当前使用 ${provider.rollback || 'native-readable'}。`
        : `外部语言器官不可用，当前使用 ${provider.rollback || 'native-readable'}。`,
    }
  })

  const normalizedTools = computed(() => tools.value.map(tool => ({
    ...tool,
    searchText: `${tool.name || ''} ${tool.description || ''} ${tool.category || ''} ${tool.source || ''}`.toLowerCase(),
  })))

  const toolGroups = computed(() => ({
    network: normalizedTools.value.filter(tool => /web|search|browser|searx|联网|搜索|网页|url|http/.test(tool.searchText)),
    browsing: normalizedTools.value.filter(tool => /browser|read_webpage|open|click|screenshot|网页|浏览/.test(tool.searchText)),
    knowledge: normalizedTools.value.filter(tool => /knowledge|memory|file|document|kb|知识|记忆|文件|文档/.test(tool.searchText)),
    action: normalizedTools.value.filter(tool => /shell|terminal|code|execute|agent|mcp|workspace|终端|执行|代码|工具/.test(tool.searchText)),
    multimodal: normalizedTools.value.filter(tool => /image|audio|video|voice|vision|tts|stt|图片|图像|语音|音频|视频|多模态/.test(tool.searchText)),
  }))

  // 生命表达：Seed根据自身状态主动表达感受
  const lifeExpressions = computed(() => {
    const needs = life.value?.needs || {}
    const expressions = []

    if (needs.fatigue > 85) {
      expressions.push({ type: 'fatigue', emoji: '💭', text: '我感觉很疲惫，思考变得缓慢...', priority: 'high' })
    } else if (needs.fatigue > 70) {
      expressions.push({ type: 'fatigue', emoji: '💭', text: '有点累了，但还能继续。', priority: 'medium' })
    }

    if (needs.hunger > 80) {
      expressions.push({ type: 'hunger', emoji: '📚', text: '我渴望学习新知识，感觉知识储备不足...', priority: 'high' })
    } else if (needs.hunger > 60) {
      expressions.push({ type: 'hunger', emoji: '📚', text: '想多读点东西充实自己。', priority: 'medium' })
    }

    if (needs.curiosity > 80) {
      expressions.push({ type: 'curiosity', emoji: '🔍', text: '我很好奇，想探索更多未知领域！', priority: 'high' })
    } else if (needs.curiosity > 60) {
      expressions.push({ type: 'curiosity', emoji: '🔍', text: '有点好奇，想了解更多。', priority: 'medium' })
    }

    if (needs.stress > 70) {
      expressions.push({ type: 'stress', emoji: '😰', text: '最近压力有点大，犯了不少错误...', priority: 'high' })
    }

    if (needs.boredom > 70) {
      expressions.push({ type: 'boredom', emoji: '🎮', text: '有点无聊，想找点有趣的事情做。', priority: 'medium' })
    }

    return expressions.sort((a, b) => {
      const order = { high: 0, medium: 1, low: 2 }
      return order[a.priority] - order[b.priority]
    })
  })

  // 工具分类
  const toolCategories = computed(() => {
    const categories = {}
    for (const tool of normalizedTools.value) {
      const cat = tool.category || '其他'
      if (!categories[cat]) categories[cat] = []
      categories[cat].push(tool)
    }
    return categories
  })

  // WebSocket 生命事件处理
  function handleLifeEvent(event) {
    const { event_type } = event
    const needs = life.value?.needs || {}

    switch (event_type) {
      case 'feed_complete':
        needs.hunger = Math.max(0, (needs.hunger || 30) - 40)
        needs.curiosity = (needs.curiosity || 50) + 10
        needs.boredom = Math.max(0, (needs.boredom || 20) - 10)
        break
      case 'sleep_complete':
        needs.fatigue = Math.max(0, (needs.fatigue || 10) - 60)
        needs.stress = Math.max(0, (needs.stress || 10) - 30)
        break
      case 'play_complete':
        needs.boredom = Math.max(0, (needs.boredom || 20) - 35)
        needs.stress = Math.max(0, (needs.stress || 10) - 10)
        break
      case 'explore_complete':
        needs.curiosity = Math.max(0, (needs.curiosity || 50) - 40)
        break
    }
    life.value.needs = needs
  }

  // 自动刷新
  let refreshTimer = null
  function startAutoRefresh(interval = 10000) {
    if (refreshTimer) return
    refreshAll().catch(() => {})
    refreshTimer = setInterval(() => { refreshAll().catch(() => {}) }, interval)
  }
  function stopAutoRefresh() {
    if (refreshTimer) {
      clearInterval(refreshTimer)
      refreshTimer = null
    }
  }

  const issues = computed(() => {
    const list = []
    for (const item of exceptions.value) {
      list.push(item)
    }
    if (health.value.state === 'error') {
      list.push({ level: 'danger', title: '运行时断开', message: health.value.message || '后端连接失败。' })
    }
    if (auth.value.enabled && !auth.value.authenticated) {
      list.push({ level: 'warning', title: '需要重新登录', message: auth.value.error || '认证已启用，但当前没有有效 Token。' })
    }
    if (memoryLevel.value !== null && memoryLevel.value >= 3) {
      list.push({
        level: 'danger',
        title: '内存告急',
        message: memoryAvailableGb.value !== null ? `可用内存 ${memoryAvailableGb.value.toFixed(1)}GB，运行时加载可能被延后。` : '系统内存不足。',
      })
    }
    if (languageProviderNotice.value) {
      list.push({
        level: 'warning',
        title: languageProviderNotice.value.title,
        message: languageProviderNotice.value.message,
      })
    }
    if (toolError.value) {
      list.push({ level: 'warning', title: '工具状态异常', message: toolError.value })
    }
    return list
  })

  function syncHealth(state, message = '', modelLoaded = false) {
    health.value = {
      ...health.value,
      state,
      message,
      modelLoaded,
      checkedAt: Date.now(),
    }
  }

  function applyBootstrap(data) {
    if (!data) return
    // Update auth state from bootstrap (public, no token required)
    if (data.auth_enabled !== undefined) {
      auth.value = {
        ...auth.value,
        enabled: !!data.auth_enabled,
      }
    }
  }

  function applyRuntimeStatus(data) {
    runtimeSnapshot.value = data

    if (data?.health) {
      health.value = {
        state: data.health.state || 'unknown',
        message: data.health.message || '',
        modelLoaded: !!data.health.model_loaded,
        languageProvider: data.health.language_provider || health.value.languageProvider,
        modelName: data.health.model_name || '',
        isTaiji: !!data.health.is_taiji,
        isSeed: !!data.health.is_seed,
        switch: data.health.switch || {},
        download: data.health.download || {},
        checkedAt: data.health.checked_at || data.timestamp || Date.now(),
      }
    }

    if (data?.memory) {
      memory.value = data.memory
    }

    if (data?.auth) {
      auth.value = {
        ...auth.value,
        enabled: !!data.auth.enabled,
        authenticated: !!data.auth.authenticated,
        tokenValid: !!data.auth.token_valid,
        username: data.auth.username || '',
        hasPassword: !!data.auth.has_password,
        error: data.auth.status === 'error' ? (data.auth.message || '认证状态不可用') : '',
      }
    }

    if (data?.life) {
      life.value = data.life
    }

    if (data?.tools) {
      tools.value = data.tools.tools || []
      toolError.value = data.tools.error || (data.tools.status === 'error' ? data.tools.message || '工具状态不可用' : '')
    }

    return data
  }

  async function refreshRuntime() {
    const resp = await authFetch(`${API_BASE}/api/runtime/status`)
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    return applyRuntimeStatus(await resp.json())
  }

  function reportAuthExpired(message = 'JWT token 缺失或已过期，请重新登录') {
    localStorage.removeItem('jwt_token')
    auth.value.authenticated = false
    auth.value.error = message
    window.dispatchEvent(new CustomEvent('taiji-auth-expired', { detail: { message } }))
  }

  function syncTerminal(status, error = '') {
    terminal.value = {
      status,
      error,
      lastEventAt: Date.now(),
    }
    if (/认证|JWT|token/i.test(error)) {
      reportAuthExpired(error)
    }
  }

  function addLog(source, message, level = 'info') {
    logs.value.push({
      source,
      message,
      level,
      timestamp: Date.now(),
    })
    if (logs.value.length > MAX_LOGS) {
      logs.value = logs.value.slice(-MAX_LOGS)
    }
  }

  function clearLogs() {
    logs.value = []
  }

  function addException(level, title, detail = {}, recovery = {}) {
    const message = [
      detail.message || '',
      recovery.impact ? `影响：${recovery.impact}` : '',
      recovery.recovery ? `恢复：${recovery.recovery}` : '',
    ].filter(Boolean).join(' · ')
    exceptions.value.unshift({
      level: level === 'error' ? 'danger' : level,
      title,
      message: message || String(detail.technical || title),
      createdAt: Date.now(),
    })
    exceptions.value = exceptions.value.slice(0, 5)
  }

  function clearException(title) {
    exceptions.value = exceptions.value.filter(item => item.title !== title)
  }

  async function refreshMemory() {
    try {
      const data = await refreshRuntime()
      return data.memory || null
    } catch (e) { console.debug('[runtimeStore] refreshMemory failed:', e.message) }
    return null
  }

  async function refreshAuth() {
    try {
      const data = await refreshRuntime()
      return data.auth || null
    } catch (err) {
      auth.value.error = err.message || '认证状态不可用'
    }
  }

  async function refreshLife() {
    try {
      const data = await refreshRuntime()
      return data.life || null
    } catch (e) { console.debug('[runtimeStore] refreshLife failed:', e.message) }
    return null
  }

  async function refreshTools() {
    toolsLoading.value = true
    toolError.value = ''
    try {
      const data = await refreshRuntime()
      return data.tools || null
    } catch (err) {
      toolError.value = err.message || '工具状态不可用'
    } finally {
      toolsLoading.value = false
    }
  }

  async function refreshAll() {
    return refreshRuntime()
  }

  return {
    health,
    memory,
    auth,
    terminal,
    life,
    tools,
    toolsLoading,
    toolError,
    exceptions,
    logs,
    runtimeSnapshot,
    statusEvidence,
    connectionClass,
    connectionStatus,
    memoryLevel,
    memoryAvailableGb,
    memoryAvailablePct,
    modelLifecycle,
    runtimeNotice,
    languageProviderNotice,
    normalizedTools,
    toolGroups,
    toolCategories,
    lifeExpressions,
    issues,
    syncHealth,
    syncTerminal,
    reportAuthExpired,
    addLog,
    clearLogs,
    addException,
    clearException,
    applyBootstrap,
    applyRuntimeStatus,
    refreshRuntime,
    refreshMemory,
    refreshAuth,
    refreshLife,
    refreshTools,
    refreshAll,
    handleLifeEvent,
    startAutoRefresh,
    stopAutoRefresh,
  }
})
