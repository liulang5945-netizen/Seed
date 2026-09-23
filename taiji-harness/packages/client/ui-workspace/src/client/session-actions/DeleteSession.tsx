/**
 * The delete action: a `sidebar.workspaces.session.menu.item` row, a
 * `sidebar.workspaces.session.row.action` hover button that arms only on an
 * archived row, and the `shell.overlay` dialog that confirms the physical
 * deletion. The entries only raise the confirmation — the dialog performs the
 * Host call, so a deletion always passes through one explicit consent step,
 * and a refusal for running work escalates to a stop-and-delete confirm
 * inside the same dialog.
 */
import { useState } from 'react'
// Type-only: the family keys each provider merges; a key this program did not compile takes the generic line.
import type {} from '@taiji/dsh-agent/types'
import type {} from '@taiji/dsh-jobs/view'
import type {} from '@taiji/dsh-schedule/client'
import type {} from '@taiji/dsh-subagent/client'
import { Button, IconTrashOutlineRegular, MenuItemButton, Modal, Tooltip } from '@taiji/dsh-client-ui-primitives'
import type {
  DeleteSessionInjected, SessionDeleteConfirmInjected, SessionDeleteConfirmProps, SessionDeleteConfirmRequest,
  SessionMenuItemProps, SessionRowActionProps,
} from '../contract/slots.ts'
import css from '../rows/Rows.module.css'
import browserCss from '../rows/WorkspaceBrowser.module.css'

/**
 * Menu row (order 500): ask for the delete confirmation.
 * @param props - owner share, the delete share, and the menu open state.
 * @returns the row.
 */
export function DeleteSessionMenuItem({
  sessionId, useMenuOpenState, requestSessionDelete, t,
}: SessionMenuItemProps<DeleteSessionInjected>) {
  const [, setMenuOpen] = useMenuOpenState()
  return (
    <MenuItemButton
      icon={<IconTrashOutlineRegular size={14} />}
      onSelect={() => {
        setMenuOpen(false)
        requestSessionDelete(sessionId)
      }}
    >
      {t('menu.deleteSession')}
    </MenuItemButton>
  )
}

/**
 * Hover button (order 300): ask for the delete confirmation, and only on an
 * archived row — the logged, grouped state hides the most destructive affordance.
 * @param props - owner share and the delete share.
 * @returns the button, or nothing while the row is not archived.
 */
export function DeleteSessionRowButton({
  sessionId, useArchived, requestSessionDelete, t,
}: SessionRowActionProps<DeleteSessionInjected>) {
  const archived = useArchived(set => set.has(sessionId))
  if (!archived) return null
  return (
    <Tooltip label={t('menu.deleteSession')} side="bottom" align="end" delayMs={500}>
      <button
        type="button"
        className={css.iconButton}
        aria-label={t('menu.deleteSession')}
        onClick={() => { requestSessionDelete(sessionId) }}
      >
        <IconTrashOutlineRegular size={14} />
      </button>
    </Tooltip>
  )
}

/**
 * The `shell.overlay` entry: nothing while no confirmation is pending,
 * otherwise one dialog per request (keyed by the Session). Confirming asks
 * the Host to delete; a refusal for running work keeps the dialog open, names
 * that work, and turns the confirm action into stop-and-delete. Cancelling
 * leaves everything as it was.
 * @param props - the request hook, its settlement, the delete hop, and the locale seat.
 * @returns the open dialog, or null.
 */
export function SessionDeleteConfirmDialog({
  useDeleteRequest, settleSessionDelete, deleteSession, t,
}: SessionDeleteConfirmProps) {
  const request = useDeleteRequest(pending => pending)
  if (request === null) return null
  return (
    <DeleteConfirmForm
      key={request.sessionId}
      request={request}
      deleteSession={deleteSession}
      onSettle={settleSessionDelete}
      t={t}
    />
  )
}

/** One request's dialog: in-flight and error state die with it. */
function DeleteConfirmForm({ request, deleteSession, onSettle, t }: {
  request: SessionDeleteConfirmRequest
  deleteSession: SessionDeleteConfirmInjected['deleteSession']
  onSettle: () => void
  t: SessionDeleteConfirmProps['t']
}) {
  const [activity, setActivity] = useState<SessionDeleteConfirmRequest['activity']>(request.activity)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const stopFirst = activity !== undefined
  const close = () => {
    if (deleting) return
    onSettle()
  }
  const confirm = () => {
    setDeleting(true)
    setError(null)
    deleteSession(request.sessionId, stopFirst ? { stopActivity: true } : undefined).then(() => {
      setDeleting(false)
      onSettle()
    }).catch((reason: unknown) => {
      setDeleting(false)
      // A Host refusal for running work keeps the dialog open with the
      // reported activity; the confirm action escalates to stop-and-delete.
      const refused = reason instanceof Error && reason.name === 'WorkspaceSessionDeleteError'
        ? (reason as { rpcError?: { code?: string; details?: { activity?: SessionDeleteConfirmRequest['activity'] } } }).rpcError
        : undefined
      if (refused?.code === 'workspace/session-active' && refused.details?.activity !== undefined) {
        setActivity(refused.details.activity)
        return
      }
      // A live Session the Host could not close — another owner holds it — is
      // named in the user's own language rather than by its RPC code.
      if (refused?.code === 'workspace/session-open') {
        setError(t('sessionDelete.confirm.heldOpen'))
        return
      }
      setError(reason instanceof Error ? reason.message : String(reason))
    })
  }
  return (
    <Modal
      open
      onClose={close}
      closeLabel={t('close')}
      title={t('sessionDelete.confirm.title')}
      description={t('sessionDelete.confirm.desc', { title: request.displayTitle })}
      footer={(
        <>
          <Button variant="outline" disabled={deleting} onClick={close}>{t('cancel')}</Button>
          <Button
            variant="outline"
            className={browserCss.deleteAction}
            disabled={deleting}
            onClick={confirm}
          >
            {t(stopFirst ? 'sessionDelete.confirm.stopAction' : 'sessionDelete.confirm.action')}
          </Button>
        </>
      )}
    >
      {stopFirst && (
        <ul className={browserCss.archiveActivity} aria-label={t('sessionDelete.confirm.activity')}>
          {activity?.map((entry, index) => (
            <li key={`${entry.kind}-${String(index)}`}>{activityLine(entry, t)}</li>
          ))}
        </ul>
      )}
      {deleting && <div className={browserCss.deleteStatus} role="status">{t(stopFirst ? 'sessionDelete.confirm.stopPending' : 'sessionDelete.confirm.pending')}</div>}
      {error !== null && <div className={browserCss.renameError} role="alert">{error}</div>}
    </Modal>
  )
}

/**
 * One family's line: its count and the items' labels (ids when a family
 * carries no label). A family this dictionary does not know — a provider
 * merged into the kind map — falls through to the generic line.
 */
function activityLine(entry: NonNullable<SessionDeleteConfirmRequest['activity']>[number], t: SessionDeleteConfirmProps['t']): string {
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
