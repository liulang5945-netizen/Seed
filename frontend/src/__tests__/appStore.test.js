import { describe, it, expect, beforeEach } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'

// Mock localStorage
const localStorageMock = (() => {
  let store = {}
  return {
    getItem: (key) => store[key] ?? null,
    setItem: (key, value) => { store[key] = String(value) },
    removeItem: (key) => { delete store[key] },
    clear: () => { store = {} },
  }
})()
Object.defineProperty(globalThis, 'localStorage', { value: localStorageMock })

// Mock document.documentElement for applyTheme
if (typeof document !== 'undefined') {
  Object.defineProperty(document, 'documentElement', {
    value: {
      classList: { add: () => {}, remove: () => {} },
      setAttribute: () => {},
      style: { setProperty: () => {} },
    },
    writable: true,
    configurable: true,
  })
}

// _normalizeTheme is not exported, so we replicate the logic for testing
// The actual function lives inside appStore.js
function normalizeTheme(raw) {
  if (!raw) return 'classic'
  if (raw === 'light') return 'classic'
  return raw
}

describe('_normalizeTheme', () => {
  it('returns "classic" for null/undefined', () => {
    expect(normalizeTheme(null)).toBe('classic')
    expect(normalizeTheme(undefined)).toBe('classic')
    expect(normalizeTheme('')).toBe('classic')
  })

  it('maps legacy "light" to "classic"', () => {
    expect(normalizeTheme('light')).toBe('classic')
  })

  it('preserves valid theme values', () => {
    expect(normalizeTheme('dark')).toBe('dark')
    expect(normalizeTheme('classic')).toBe('classic')
    expect(normalizeTheme('teal')).toBe('teal')
    expect(normalizeTheme('violet')).toBe('violet')
    expect(normalizeTheme('warm')).toBe('warm')
    expect(normalizeTheme('auto')).toBe('auto')
  })
})

describe('appStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorageMock.clear()
  })

  it('initializes with default theme "classic"', async () => {
    const { useAppStore } = await import('../stores/appStore.js')
    const store = useAppStore()
    expect(store.currentTheme).toBe('classic')
  })

  it('reads theme from localStorage on init', async () => {
    localStorageMock.setItem('taiji_theme', 'dark')
    // Re-import to get fresh store
    const { useAppStore } = await import('../stores/appStore.js')
    const store = useAppStore()
    expect(store.currentTheme).toBe('dark')
  })

  it('has 5 themes defined', async () => {
    const { useAppStore } = await import('../stores/appStore.js')
    const store = useAppStore()
    expect(store.themes).toHaveLength(5)
    const ids = store.themes.map((t) => t.id)
    expect(ids).toEqual(['classic', 'dark', 'teal', 'violet', 'warm'])
  })

  it('resolvedTheme returns "dark" for dark theme', async () => {
    const { useAppStore } = await import('../stores/appStore.js')
    const store = useAppStore()
    store.currentTheme = 'dark'
    expect(store.resolvedTheme).toBe('dark')
  })

  it('resolvedTheme returns "light" for non-dark themes', async () => {
    const { useAppStore } = await import('../stores/appStore.js')
    const store = useAppStore()
    store.currentTheme = 'classic'
    expect(store.resolvedTheme).toBe('light')
    store.currentTheme = 'teal'
    expect(store.resolvedTheme).toBe('light')
  })

  it('toggleWorkspace flips the flag', async () => {
    const { useAppStore } = await import('../stores/appStore.js')
    const store = useAppStore()
    expect(store.showWorkspace).toBe(false)
    store.toggleWorkspace()
    expect(store.showWorkspace).toBe(true)
    store.toggleWorkspace()
    expect(store.showWorkspace).toBe(false)
  })

  it('t() returns localized text', async () => {
    const { useAppStore } = await import('../stores/appStore.js')
    const store = useAppStore()
    // t() should return the key itself if no translation found
    const result = store.t('nonexistent_key_xyz')
    expect(result).toBe('nonexistent_key_xyz')
  })
})
