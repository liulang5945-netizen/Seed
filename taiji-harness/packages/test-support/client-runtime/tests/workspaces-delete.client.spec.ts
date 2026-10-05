/** The recorded `deleteSession` default and its stub replacement. */
import { describe, expect, it } from 'vitest'
import type { SessionId } from '@taiji/dsh-session/types'
import { TestWorkspaces } from '../src/workspaces.ts'
import type { Stabilizer } from '../src/fixtures.ts'

const stabilize: Stabilizer = async (fn): Promise<void> => {
  await fn()
}

describe('TestWorkspaces.deleteSession', () => {
  it('drops the id from the archive and pin sets and records the call', async () => {
    const workspaces = new TestWorkspaces(stabilize)
    const sessionId = 's-1' as SessionId
    await workspaces.update((draft) => {
      draft.archivedSessionIds = [sessionId]
      draft.pinnedSessionIds = [sessionId]
    })

    await workspaces.deleteSession(sessionId)

    expect(workspaces.list.getSnapshot().archivedSessionIds).toEqual([])
    expect(workspaces.list.getSnapshot().pinnedSessionIds).toEqual([])
    expect(workspaces.calls.at(-1)).toEqual({ method: 'deleteSession', args: [sessionId, undefined] })
  })

  it('runs a stub instead of the default and still records the call', async () => {
    const workspaces = new TestWorkspaces(stabilize)
    const sessionId = 's-2' as SessionId
    let stubArgs: unknown[] | undefined
    workspaces.stub('deleteSession', async (id, options) => {
      stubArgs = [id, options]
    })
    await workspaces.update((draft) => {
      draft.pinnedSessionIds = [sessionId]
    })

    await workspaces.deleteSession(sessionId, { stopActivity: true })

    expect(stubArgs).toEqual([sessionId, { stopActivity: true }])
    // The default body must not run, so the pin set still holds the id.
    expect(workspaces.list.getSnapshot().pinnedSessionIds).toEqual([sessionId])
    expect(workspaces.calls.filter(entry => entry.method === 'deleteSession')).toHaveLength(1)
  })
})
