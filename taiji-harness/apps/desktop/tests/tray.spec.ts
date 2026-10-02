import { describe, expect, it, vi } from 'vitest'
import { DesktopTray, type DesktopTraySurface, type TrayMenuItem } from '../src/tray.ts'

/** Locale fixture with distinguishable copy, so a relabel is observable. */
function localeOf(language: 'en' | 'zh') {
  return {
    messages: language === 'en'
      ? { productName: 'Seed', openApplication: 'Open Seed', quitApplication: 'Quit Seed' }
      : { productName: 'Seed ZH', openApplication: '打开 Seed', quitApplication: '退出 Seed' },
  }
}

function fixture(language: () => 'en' | 'zh' = () => 'en') {
  const clicks: Array<() => void> = []
  const destroy = vi.fn()
  let tooltip = ''
  let menu: readonly TrayMenuItem[] = []
  const surface: DesktopTraySurface = {
    onClick: (listener) => { clicks.push(listener) },
    setToolTip: (text) => { tooltip = text },
    setMenu: (items) => { menu = items },
    destroy,
  }
  const open = vi.fn()
  const quit = vi.fn()
  const tray = new DesktopTray({
    create: () => surface,
    locale: () => localeOf(language()),
    open,
    quit,
  })
  return { tray, clicks, destroy, open, quit, tooltip: () => tooltip, menu: () => menu }
}

describe('desktop tray', () => {
  it('labels the tooltip and both menu entries from the current locale', () => {
    const b = fixture()
    expect(b.tooltip()).toBe('Seed')
    expect(b.menu()).toHaveLength(3)
    expect(b.menu()[0]).toMatchObject({ label: 'Open Seed' })
    expect(b.menu()[1]).toEqual({ type: 'separator' })
    expect(b.menu()[2]).toMatchObject({ label: 'Quit Seed' })
  })

  it('reopens on a left click and quits from the menu entry', () => {
    const b = fixture()
    b.clicks[0]?.()
    expect(b.open).toHaveBeenCalledOnce()
    const quitItem = b.menu()[2]
    if (quitItem === undefined || 'type' in quitItem) throw new Error('quit entry missing')
    quitItem.click()
    expect(b.quit).toHaveBeenCalledOnce()
    expect(b.open).toHaveBeenCalledOnce()
  })

  it('relabels in place when the shell language changes', () => {
    let language: 'en' | 'zh' = 'en'
    const b = fixture(() => language)
    expect(b.tooltip()).toBe('Seed')
    language = 'zh'
    b.tray.relabel()
    expect(b.tooltip()).toBe('Seed ZH')
    expect(b.menu()[0]).toMatchObject({ label: '打开 Seed' })
    expect(b.menu()[2]).toMatchObject({ label: '退出 Seed' })
  })

  it('destroys the surface once and ignores relabels afterwards', () => {
    const b = fixture()
    b.tray.dispose()
    b.tray.dispose()
    expect(b.destroy).toHaveBeenCalledOnce()

    const tooltip = b.tooltip()
    const menu = b.menu()
    b.tray.relabel()
    expect(b.tooltip()).toBe(tooltip)
    expect(b.menu()).toBe(menu)
  })
})
