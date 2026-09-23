/**
 * Life panel, browser half: the **Life** entry of the sidebar and the global
 * page it opens in the main column. The page renders the Taiji local
 * runtime's last reading — where it came from, the life organs, training with
 * its checkpoint roster, the knowledge base, and the host projection — and
 * drives the training and legacy scheduler controls through `ctx.life`.
 */

import type {} from '@taiji/dsh-api-life-controller/client'
import type {} from '@taiji/dsh-client-locale/client'
import type { Context as ClientContext } from '@taiji/cordis'
// Type-only: the root `main` keyed slot the page registers into, declared by
// ui-layout with the panel id brand, and the `sidebar.panellist` list the
// entry registers into, declared by ui-sidebar.
import type { MainPanelId } from '@taiji/dsh-client-ui-layout/client'
import type {} from '@taiji/dsh-client-ui-renderer/client'
import type {} from '@taiji/dsh-client-ui-sidebar/client'
import { LifePanel } from './LifePanel.tsx'
import { LifePanelIcon } from './LifePanelIcon.tsx'
import { en, zh, type LifeLocaleKey } from './locales.ts'

export type { LifePanelInjected, LifePanelProps } from './LifePanel.tsx'
export type { LifeLocaleKey } from './locales.ts'

declare module '@taiji/dsh-client-ui-slots' {
  interface LocaleNamespaceMap {
    /** Life panel copy. */
    'life': LifeLocaleKey
  }
}

/** Dictionary namespace owned by this plugin. */
export const NS = 'life'

/** The id shared by the sidebar entry and the main panel it opens. */
export const PANEL_ID = 'life' as MainPanelId

/** Services required by the sidebar registration and the Life facade; `life` is installed by the Life controller's client half. */
export const inject = ['slots', 'locale', 'life']

/**
 * Contribute the Life entry to the sidebar with the global page it opens.
 * @param ctx - the browser plugin context.
 */
export function apply(ctx: ClientContext): void {
  ctx.effect(() => ctx.locale.register(NS, { zh, en }), 'ui-life: dictionaries')
  const t = ctx.locale.bind(NS)

  // The page is a global panel: it belongs to the deployment, not to a
  // Session, and the sidebar's entry selects it. What the runtime reported
  // and what its controls accept are the page's own.
  ctx.slots.inject('main', () => ctx.slots.register({
    name: 'main',
    key: PANEL_ID,
    locale: NS,
    inject: () => ({ life: ctx.life }),
  }, LifePanel))
  ctx.slots.inject('sidebar.panellist', () => ctx.slots.register({
    name: 'sidebar.panellist',
    id: PANEL_ID,
    order: 100,
    label: () => t('panel'),
    locale: NS,
  }, LifePanelIcon))
}
