import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'

// Mock localStorage
const localStorageMock = (() => {
  let store = {}
  return {
    getItem: vi.fn((key) => store[key] ?? null),
    setItem: vi.fn((key, value) => { store[key] = String(value) }),
    removeItem: vi.fn((key) => { delete store[key] }),
    clear: vi.fn(() => { store = {} }),
  }
})()
Object.defineProperty(globalThis, 'localStorage', { value: localStorageMock })

// Mock window.dispatchEvent
if (typeof window !== 'undefined') {
  vi.spyOn(window, 'dispatchEvent').mockImplementation(() => {})
}

describe('runtimeStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorageMock.clear()
    vi.clearAllMocks()
  })

  describe('syncHealth', () => {
    it('更新健康状态', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()

      store.syncHealth('connected', '一切正常', true)
      expect(store.health.state).toBe('connected')
      expect(store.health.message).toBe('一切正常')
      expect(store.health.modelLoaded).toBe(true)
      expect(store.health.checkedAt).toBeGreaterThan(0)
    })

    it('默认 modelLoaded 为 false', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()

      store.syncHealth('loading', '加载中')
      expect(store.health.modelLoaded).toBe(false)
    })
  })

  describe('applyBootstrap', () => {
    it('设置 auth.enabled', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()

      store.applyBootstrap({ auth_enabled: true })
      expect(store.auth.enabled).toBe(true)
    })

    it('null 输入不崩溃', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()

      expect(() => store.applyBootstrap(null)).not.toThrow()
      expect(() => store.applyBootstrap(undefined)).not.toThrow()
    })
  })

  describe('applyRuntimeStatus', () => {
    it('应用完整运行时数据', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()

      const data = {
        health: {
          state: 'connected',
          message: 'OK',
          model_loaded: true,
          model_name: 'seed-v1',
          is_taiji: true,
        },
        memory: { level: 1, available_gb: 8.5, available_pct: 53 },
        auth: { enabled: true, authenticated: true, username: 'admin', has_password: true },
        life: { is_running: false, needs: {}, total_interactions: 0, uptime_seconds: 0 },
        tools: { tools: [{ name: 'search', category: 'network' }], status: 'ok' },
      }

      store.applyRuntimeStatus(data)

      expect(store.health.state).toBe('connected')
      expect(store.health.modelLoaded).toBe(true)
      expect(store.memory.level).toBe(1)
      expect(store.auth.username).toBe('admin')
      expect(store.tools).toHaveLength(1)
      expect(store.runtimeSnapshot).toStrictEqual(data)
    })

    it('处理工具错误状态', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()

      store.applyRuntimeStatus({
        tools: { status: 'error', message: '工具不可用' },
      })

      expect(store.toolError).toBe('工具不可用')
    })
  })

  describe('addException / clearException', () => {
    it('添加异常到列表头部', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()

      store.addException('error', '测试错误', { message: '详情' }, { impact: '影响', recovery: '恢复' })
      expect(store.exceptions).toHaveLength(1)
      expect(store.exceptions[0].title).toBe('测试错误')
      expect(store.exceptions[0].level).toBe('danger') // error → danger
    })

    it('最多保留 5 条异常', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()

      for (let i = 0; i < 7; i++) {
        store.addException('info', `错误${i}`, {})
      }
      expect(store.exceptions.length).toBeLessThanOrEqual(5)
    })

    it('按标题清除异常', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()

      store.addException('info', '可清除', {})
      store.addException('info', '保留', {})
      store.clearException('可清除')
      expect(store.exceptions).toHaveLength(1)
      expect(store.exceptions[0].title).toBe('保留')
    })
  })

  describe('addLog / clearLogs', () => {
    it('添加日志', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()

      store.addLog('test', '测试消息', 'info')
      expect(store.logs).toHaveLength(1)
      expect(store.logs[0].source).toBe('test')
      expect(store.logs[0].level).toBe('info')
    })

    it('日志上限 200 条', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()

      for (let i = 0; i < 250; i++) {
        store.addLog('test', `msg-${i}`)
      }
      expect(store.logs.length).toBeLessThanOrEqual(200)
    })

    it('clearLogs 清空日志', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()

      store.addLog('test', 'msg')
      store.clearLogs()
      expect(store.logs).toHaveLength(0)
    })
  })

  describe('computed: connectionClass', () => {
    it('connected 状态返回 connected', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()

      store.syncHealth('connected', '', true)
      expect(store.connectionClass).toBe('connected')
    })

    it('error 状态返回 error', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()

      store.syncHealth('error', '断开')
      expect(store.connectionClass).toBe('error')
    })

    it('connecting/unknown 返回 connecting', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()

      store.syncHealth('connecting', '')
      expect(store.connectionClass).toBe('connecting')
      store.syncHealth('unknown', '')
      expect(store.connectionClass).toBe('connecting')
    })
  })

  describe('computed: connectionStatus', () => {
    it('connected + modelLoaded 返回原生运行时已连接', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()

      store.syncHealth('connected', '', true)
      expect(store.connectionStatus).toBe('运行时已连接')
    })

    it('connected 但无原生运行时返回提示', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()

      store.syncHealth('connected', '', false)
      expect(store.connectionStatus).toContain('运行时未激活')
    })
  })

  describe('computed: statusEvidence', () => {
    it('把运行时、provider、工作台和 self-state 投影为可追溯证据', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()
      const observedAt = Math.floor(Date.now() / 1000)

      store.applyRuntimeStatus({
        timestamp: observedAt,
        health: {
          state: 'connected',
          model_loaded: true,
          model_name: 'seed-native',
          is_taiji: true,
          language_provider: {
            state: 'active',
            backend_id: 'native-readable-v1',
            artifact_id: 'native-readable-v1',
            chat_enabled: 'false',
          },
        },
        tools: {
          status: 'ok',
          snapshot_id: 'snapshot-1234567890',
          revision: 3,
          owner: 'Taiji native Workbench',
          observed_at: observedAt,
          tools: [{ name: 'workspace.read', enabled: true }],
        },
        life: { needs: {}, is_running: false },
        training: { is_training: false },
      })

      expect(store.statusEvidence.runtime.owner).toBe('Taiji runtime')
      expect(store.statusEvidence.runtime.availability).toBe('可用')
      expect(store.statusEvidence.provider.availability).toBe('已接入（可读）')
      expect(store.statusEvidence.workbench.detail).toContain('snapshot snapshot-123')
      expect(store.statusEvidence.homeostasis.availability).toBe('未上报')
      expect(store.statusEvidence.training.availability).toBe('已上报（空闲）')
      expect(store.statusEvidence.runtime.freshness.state).toBe('fresh')
    })
  })

  describe('handleLifeEvent', () => {
    it('feed_complete 降低饥饿值', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()
      store.life = { is_running: true, needs: { hunger: 80, curiosity: 50, boredom: 20, fatigue: 10, stress: 10 } }

      store.handleLifeEvent({ event_type: 'feed_complete', data: {} })
      expect(store.life.needs.hunger).toBe(40) // 80 - 40
    })

    it('sleep_complete 降低疲劳值', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()
      store.life = { is_running: true, needs: { fatigue: 90, stress: 50 } }

      store.handleLifeEvent({ event_type: 'sleep_complete', data: {} })
      expect(store.life.needs.fatigue).toBe(30) // 90 - 60
      expect(store.life.needs.stress).toBe(20) // 50 - 30
    })
  })

  describe('computed: lifeExpressions', () => {
    it('高疲劳度产生疲惫表达', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()
      store.life = { is_running: true, needs: { fatigue: 90 }, total_interactions: 0, uptime_seconds: 0 }

      const expressions = store.lifeExpressions
      expect(expressions.some(e => e.type === 'fatigue' && e.priority === 'high')).toBe(true)
    })

    it('正常状态无特殊表达', async () => {
      const { useRuntimeStore } = await import('../stores/runtimeStore.js')
      const store = useRuntimeStore()
      store.life = { is_running: true, needs: { fatigue: 30, hunger: 30, curiosity: 50, stress: 10, boredom: 20 }, total_interactions: 0, uptime_seconds: 0 }

      const expressions = store.lifeExpressions
      expect(expressions).toHaveLength(0)
    })
  })
})
