/** Shared activity-family copy used by the archive and delete confirmations. */
import type { SessionActivity } from '@taiji/dsh-api-workspace-controller/client'
import type { PropsLocale } from '@taiji/dsh-client-ui-slots'

/**
 * One family's line: its count and the items' labels (ids when a family
 * carries no label). A family this dictionary does not know — a provider
 * merged into the kind map — falls through to the generic line.
 * @param entry - one activity family reported by the registry providers.
 * @param t - the workspace locale face both confirmations already share.
 * @returns the localized one-line summary of the family.
 */
export function activityLine(entry: SessionActivity, t: PropsLocale<'workspace'>['t']): string {
  const items = entry.items ?? []
  const n = items.length
  const names = items.map(item => item.label ?? item.id).join(t('archive.confirm.listSeparator'))
  const plural = n === 1 ? 'one' : 'other'
  switch (entry.kind) {
    case 'turn': return t('archive.confirm.turn')
    case 'subagent': return t(`archive.confirm.subagents.${plural}`, { n, names })
    case 'job': return t(`archive.confirm.jobs.${plural}`, { n, names })
    case 'schedule': return t(`archive.confirm.schedules.${plural}`, { n, names })
    default: return t(`archive.confirm.other.${plural}`, { kind: entry.kind, n })
  }
}
