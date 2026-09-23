/** The sidebar's Life entry icon; the sidebar owns the button, label, and selected state around it. */

import type { ReactNode } from 'react'
import { IconGaugeOutlineRegular } from '@taiji/dsh-client-ui-primitives'
import type { PropsRuntime } from '@taiji/dsh-client-ui-slots'
import type {} from '@taiji/dsh-client-ui-sidebar/client'

/**
 * Render the life glyph at the size the sidebar asks for.
 * @param props - the sidebar's icon share: the requested edge and whether the panel is selected.
 * @returns the icon element.
 */
export function LifePanelIcon({ size }: PropsRuntime<'sidebar.panellist'>): ReactNode {
  return <IconGaugeOutlineRegular size={size} />
}
