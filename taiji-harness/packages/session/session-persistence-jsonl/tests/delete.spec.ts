/**
 * Physical session removal on the JSONL backend: `delete` unlinks the whole
 * session directory, and the ownership gate is what a Host-side deletion has
 * to get past. Three states matter: a log no handle holds (the state after a
 * live Session was closed), a log whose write handle is still open, and the
 * id that is already gone.
 */

import { existsSync } from 'node:fs'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { Context } from '@taiji/cordis'
import { SessionId } from '@taiji/dsh-session'
import { SessionAlreadyOwnedError, SessionPersistenceNotFoundError } from '@taiji/dsh-session-persistence'
import type { SessionPersistence } from '@taiji/dsh-session-persistence'
import JsonlSessionPersistence from '../src/index.ts'
import { sessionDir } from '../src/format.ts'
import { meta, oneTurnLog } from '../../session-persistence/tests/contract.ts'

const contexts: Context[] = []
const roots: string[] = []

afterEach(async () => {
  await Promise.all(contexts.splice(0).map(ctx => ctx.fiber.dispose()))
  await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true })))
})

async function boot(): Promise<{ persistence: SessionPersistence; root: string }> {
  const root = await mkdtemp(join(tmpdir(), 'dsh-jsonl-delete-'))
  roots.push(root)
  const ctx = new Context()
  contexts.push(ctx)
  await ctx.plugin(JsonlSessionPersistence, { root, compression: 'none' })
  return { persistence: ctx.sessionPersistence, root }
}

/** A stored session under `/work` with one durable turn, closed unless the test holds it. */
async function stored(persistence: SessionPersistence, id: string) {
  const handle = await persistence.create(meta(id, '/work'))
  await handle.append(oneTurnLog())
  return handle
}

describe('JsonlSessionPersistence.delete', () => {
  it('unlinks the session directory of a log no handle holds', async () => {
    const { persistence, root } = await boot()
    const id = SessionId('closed')
    const dir = sessionDir(root, '/work', id)
    const handle = await stored(persistence, 'closed')
    await handle.close()
    expect(existsSync(dir)).toBe(true)

    await persistence.delete(id)

    expect(existsSync(dir)).toBe(false)
    expect(await persistence.stat(id)).toBeUndefined()
    expect(await persistence.list()).toEqual([])
    // The id is gone rather than reopened on demand.
    await expect(persistence.open(id, 'write')).rejects.toBeInstanceOf(SessionPersistenceNotFoundError)
  })

  it('refuses a log whose write handle is still open', async () => {
    const { persistence, root } = await boot()
    const id = SessionId('held')
    const dir = sessionDir(root, '/work', id)
    const handle = await stored(persistence, 'held')
    await handle.flush()
    expect(existsSync(dir)).toBe(true)

    await expect(persistence.delete(id)).rejects.toBeInstanceOf(SessionAlreadyOwnedError)
    expect(existsSync(dir)).toBe(true)

    // Closing the holder is what admits the deletion.
    await handle.close()
    await persistence.delete(id)
    expect(existsSync(dir)).toBe(false)
  })

  it('reports an id that is not stored at all', async () => {
    const { persistence } = await boot()
    await expect(persistence.delete(SessionId('missing'))).rejects.toBeInstanceOf(SessionPersistenceNotFoundError)
  })

  it('never leaves an artifact behind when a write open races the deletion', async () => {
    const { persistence, root } = await boot()
    const id = SessionId('raced')
    const dir = sessionDir(root, '/work', id)
    await (await stored(persistence, 'raced')).close()
    expect(existsSync(dir)).toBe(true)

    // Exactly one side can win — the writer's claim refuses the deletion, or the
    // deletion's lease refuses the open — and a resolved deletion means no log
    // artifact survives it. A race that "resurrected" the directory after the
    // removal would show up here as a resolved deletion beside a live path.
    const [opened, deleted] = await Promise.allSettled([
      persistence.open(id, 'write'),
      persistence.delete(id),
    ])

    if (deleted.status === 'fulfilled') {
      expect(existsSync(dir)).toBe(false)
      expect(await persistence.stat(id)).toBeUndefined()
    } else {
      expect(deleted.reason).toBeInstanceOf(SessionAlreadyOwnedError)
      expect(existsSync(dir)).toBe(true)
    }
    if (opened.status === 'fulfilled') await opened.value.close()
  })
})
