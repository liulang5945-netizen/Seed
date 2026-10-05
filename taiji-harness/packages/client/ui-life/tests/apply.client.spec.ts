/**
 * Registration: the life dictionaries, the global main panel, and the sidebar
 * entry all come from one apply. The panel and the entry are contributions
 * that wait for the shell to declare their slots; declaring root is what
 * materializes them, exactly as ui-layout does in the app.
 */

import { Context } from '@taiji/cordis'
import { describe, expect, it, vi } from 'vitest'
import { resolveSlotLabel } from '@taiji/dsh-client-ui-slots'
import { SlotRegistry } from '@taiji/dsh-client-ui-renderer/client'
import { LocaleRuntime } from '@taiji/dsh-client-locale/client'
import { apply, inject, NS, PANEL_ID } from '../src/client/index.ts'
import { LifePanel } from '../src/client/LifePanel.tsx'
import { LifePanelIcon } from '../src/client/LifePanelIcon.tsx'
import { zh } from '../src/client/locales.ts'

async function bench() {
  const ctx = new Context()
  await ctx.plugin(SlotRegistry).await()
  const locale = new LocaleRuntime(ctx)
  locale.setLocale('zh')
  ctx.provide('locale', locale)
  // apply only closes over the facade (the inject face hands it to the
  // panel); a marker double is enough and never gets read here.
  const life = { marker: 'life-facade' }
  ctx.provide('life', life as never)
  return { ctx, life }
}

/** The shell declaration whose children make 'main' and 'sidebar.panellist' exist. */
function declareShell(slots: SlotRegistry): () => void {
  return slots.register({
    name: 'root',
    children: {
      'main': { kind: 'keyed', scope: 'root' },
      'sidebar.panellist': { kind: 'list', scope: 'root' },
    },
  } as never, () => null)
}

describe('ui-life apply', () => {
  it('declares the services it uses', () => {
    expect(inject).toEqual(['slots', 'locale', 'life'])
  })

  it('registers the dictionaries, the main panel and the sidebar entry into a standing shell', async () => {
    const { ctx, life } = await bench()
    const slots = ctx.get('slots') as SlotRegistry
    declareShell(slots)

    await ctx.plugin({ inject: [...inject], apply }).await()

    const panel = slots.entries('main')[0]!
    expect(panel.component).toBe(LifePanel)
    expect(panel.options).toMatchObject({ key: PANEL_ID })
    expect(panel.locale).toBe(NS)
    // The main panel's inject face is what hands LifePanel the life facade.
    const injected = panel.inject?.()
    expect(injected).toEqual({ life })

    const entry = slots.entries('sidebar.panellist')[0]!
    expect(entry.component).toBe(LifePanelIcon)
    expect(entry.options).toMatchObject({ id: PANEL_ID, order: 100 })
    expect(entry.locale).toBe(NS)
    // The label is a locale-following thunk; owners resolve it at read time.
    expect(resolveSlotLabel(entry.options.label)).toBe(zh.panel)
  })

  it('contributes into a shell that arrives after apply', async () => {
    const { ctx } = await bench()
    const slots = ctx.get('slots') as SlotRegistry
    await ctx.plugin({ inject: [...inject], apply }).await()
    expect(slots.entries('main')).toHaveLength(0)

    declareShell(slots)

    await vi.waitFor(() => { expect(slots.entries('main')).toHaveLength(1) })
    expect(slots.entries('sidebar.panellist')).toHaveLength(1)
  })
})
