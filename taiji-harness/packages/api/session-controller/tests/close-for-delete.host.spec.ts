/**
 * Closing a live Session for its deletion, against the production pieces a
 * Host-side deletion actually meets: the real AgentLoop, the real JSONL
 * persistence, and the controller that owns the Agent handle. A live Session
 * keeps its log's write ownership, which only `closeSession` can release, so
 * this is where "delete a Session that is open in this Host" either works or
 * the log survives the deletion.
 */

import { existsSync } from 'node:fs'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { Context } from '@taiji/cordis'
import { agentPresetProjectionDefinition } from '@taiji/dsh-agent-preset-registry'
import type { Agent } from '@taiji/dsh-agent'
import { mountAgentLoopTestDependencies, mountAgentLoopTestHarness } from '@taiji/dsh-agent-loop-testkit'
import { SessionId } from '@taiji/dsh-session'
import { SessionAlreadyOwnedError } from '@taiji/dsh-session-persistence'
import TypertRegistry from '@taiji/dsh-typert-registry'
import type {} from '@taiji/dsh-workspace'
// The production JSONL backend is the whole point of this composition test, and
// only its own package declares it: import the sources the sibling test lanes do.
import JsonlSessionPersistence from '../../../session/session-persistence-jsonl/src/index.ts'
import { sessionDir } from '../../../session/session-persistence-jsonl/src/format.ts'
import { ApiSessionAgentController } from '../src/agent.ts'
import { installModelSelectionProjection } from '../src/model-selection-projection.ts'

const contexts: Context[] = []
const roots: string[] = []

afterEach(async () => {
  await Promise.allSettled(contexts.splice(0).map(ctx => ctx.fiber.dispose()))
  for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true })
})

/** The production stack one Web deletion runs against, plus a temp cwd per case. */
async function bench(): Promise<{
  ctx: Context
  controller: ApiSessionAgentController
  harness: Awaited<ReturnType<typeof mountAgentLoopTestHarness>>
  root: string
  cwd: string
}> {
  const root = await mkdtemp(join(tmpdir(), 'dsh-close-delete-log-'))
  const cwd = await mkdtemp(join(tmpdir(), 'dsh-close-delete-cwd-'))
  roots.push(root, cwd)
  const ctx = new Context()
  contexts.push(ctx)
  await ctx.plugin(TypertRegistry)
  await mountAgentLoopTestDependencies(ctx)
  const harness = await mountAgentLoopTestHarness(ctx)
  await ctx.plugin(JsonlSessionPersistence, { root, compression: 'none' })
  ctx.sessionProjections.register(agentPresetProjectionDefinition)
  installModelSelectionProjection(ctx)
  ctx.provide('agentDefaultModel', {
    currentSelection: () => ({ provider: 'fixture', model: 'fixture-model' }),
    saveSelection: () => Promise.resolve(),
  } as never)
  const controller = new ApiSessionAgentController(ctx)
  // The Workspace registry asks this event's listeners to close a live Session
  // before deleting its log; SessionController registers the same hop.
  ctx.on('workspace/session-close', async ({ sessionId }) => {
    await controller.closeSession(sessionId)
  })
  return { ctx, controller, harness, root, cwd }
}

/** Materialize a real log for a live Agent: one committed turn plus its flush. */
async function materialize(ctx: Context, agent: Agent, root: string): Promise<string> {
  agent.session.append('turn/start', { turn: 1 })
  await ctx.sessions.flush(agent.session)
  return sessionDir(root, agent.session.header.cwd!, agent.id)
}

describe('closing a live Session for deletion', () => {
  it('releases the log write ownership, so the deletion removes the directory', async () => {
    const b = await bench()
    const id = SessionId('owned-live')
    const agent = await b.controller.ensureSession(id, b.cwd, false)
    const dir = await materialize(b.ctx, agent, b.root)
    expect(existsSync(dir)).toBe(true)

    expect(await b.controller.closeSession(id)).toBe(true)

    expect(b.ctx.agents.get(id)).toBeUndefined()
    expect(b.ctx.sessions.get(id)).toBeUndefined()
    // The write ownership is gone, so the log removal the registry performs
    // next is admitted rather than refused.
    await b.ctx.sessionPersistence.delete(id)
    expect(existsSync(dir)).toBe(false)
  })

  it('reports a live Session this controller did not publish, and keeps its writer', async () => {
    const b = await bench()
    const id = SessionId('foreign-live')
    await b.harness.create(id, {}, { cwd: b.cwd })

    // No handle here: only the consumer that published the Agent may close it,
    // which is why the registry refuses this deletion instead of racing the
    // writer.
    expect(await b.controller.closeSession(id)).toBe(false)
    expect(b.ctx.agents.get(id)?.id).toBe(id)
    await expect(b.ctx.sessionPersistence.delete(id)).rejects.toBeInstanceOf(SessionAlreadyOwnedError)
  })
})