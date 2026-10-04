import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { execa } from 'execa'
import { Context } from '@taiji/cordis'
import AgentLoop from '@taiji/dsh-agent-loop'
import { mountAgentLoopTestDependencies } from '@taiji/dsh-agent-loop-testkit'
import { afterEach, describe, expect, it, vi } from 'vitest'
import SessionStore, {
  SessionId, TOOL_OUTCOME_UNKNOWN, interruptedTurnClosers,
  type SessionEvent,
} from '@taiji/dsh-session'
import JsonlSessionPersistence from '@taiji/dsh-session-persistence-jsonl'

const repoRoot = fileURLToPath(new URL('../../../../', import.meta.url))
const childScript = fileURLToPath(new URL('./fixtures/crash-child.ts', import.meta.url))
const tsxLoader = fileURLToPath(import.meta.resolve('tsx'))
const sessionId = SessionId('semantic-checkpoint-crash')
const roots: string[] = []
const CHILD_FAILPOINT_TIMEOUT_MS = 30_000

async function waitForMarker(path: string, expected: string): Promise<string> {
  // vi.waitFor retries every callback throw, so terminal states RESOLVE out
  // of the retry loop (complete marker, or content that can no longer become
  // the expected marker) and only the still-in-progress states throw-to-retry.
  const content = await vi.waitFor(async () => {
    const current = await readFile(path, 'utf8').catch((error: unknown) => {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error
      throw new Error(`crash child did not publish failpoint ${JSON.stringify(expected)} at ${path}`, { cause: error })
    })
    if (current === expected || !expected.startsWith(current)) return current
    throw new Error(`crash child has not finished publishing failpoint ${JSON.stringify(expected)}`)
  }, { interval: 10, timeout: CHILD_FAILPOINT_TIMEOUT_MS })
  if (content !== expected) {
    throw new Error(`crash child wrote unexpected failpoint ${JSON.stringify(content)}`)
  }
  return content
}

async function crashAt(mode: 'request' | 'tool'): Promise<{ root: string; markerText: string }> {
  const root = await mkdtemp(join(tmpdir(), `dsh-semantic-${mode}-`))
  roots.push(root)
  const marker = join(root, 'failpoint')
  // Keep the open-before-write window deterministic: readiness is marker content, not path existence.
  await writeFile(marker, '')
  const expectedMarker = mode === 'request' ? 'request-dispatched' : 'tool-side-effect'
  // The SIGKILL-at-failpoint choreography stays custom: the child must die
  // mid-write, so no timeout or graceful termination may reach it first.
  // `--import` takes a module URL; a bare Windows path throws
  // ERR_UNSUPPORTED_ESM_URL_SCHEME before the child reaches the failpoint.
  const child = execa(process.execPath, ['--import', pathToFileURL(tsxLoader).href, childScript, mode, root, marker], {
    cwd: repoRoot,
    env: { TSX_TSCONFIG_PATH: join(repoRoot, 'tsconfig.json') },
    stdin: 'ignore',
    stdout: 'ignore',
    reject: false,
  })
  try {
    const markerText = await waitForMarker(marker, expectedMarker)
    child.kill('SIGKILL')
    const exit = await child
    expect({ code: exit.exitCode ?? null, signal: exit.signal ?? null }).toEqual({ code: null, signal: 'SIGKILL' })
    return { root, markerText }
  } catch (error: unknown) {
    child.kill('SIGKILL')
    throw new Error(`crash child failed: ${(await child).stderr}`, { cause: error })
  }
}

// Read the crashed durable log and balance it the way a resuming reader does:
// the stored events stay untouched; `interruptedTurnClosers` supplies the
// in-memory closers for the interrupted tail turn.
async function load(root: string): Promise<readonly SessionEvent[]> {
  const ctx = new Context()
  await ctx.plugin(SessionStore)
  await ctx.plugin(JsonlSessionPersistence, { root, compression: 'none' })
  try {
    const handle = await ctx.sessionPersistence.open(sessionId, 'read')
    try {
      const { events } = await handle.read()
      return [...events, ...interruptedTurnClosers(events)]
    } finally {
      await handle.close()
    }
  } finally {
    await ctx.fiber.dispose()
  }
}

afterEach(async () => {
  await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true })))
})

describe('semantic checkpoint hard-crash recovery', () => {
  it('persists the complete request before model dispatch', async () => {
    const crashed = await crashAt('request')
    expect(crashed.markerText).toBe('request-dispatched')
    const events = await load(crashed.root)
    expect(events.map(event => event.type)).toEqual([
      'agent/inbox/spliced', 'turn/start', 'agent/inbox/spliced',
      'step/start', 'system/message', 'user/message', 'request/header', 'request/context', 'step/end', 'turn/end',
    ])
    expect(events.at(-1)).toMatchObject({
      type: 'turn/end', data: { reason: { kind: 'interrupted' } },
    })
  })

  it('persists tool intent before a side effect and repairs its missing result as unknown', async () => {
    const crashed = await crashAt('tool')
    expect(crashed.markerText).toBe('tool-side-effect')
    const events = await load(crashed.root)
    expect(events.some(event => event.type === 'assistant/message')).toBe(true)
    expect(events.some(event => event.type === 'tool/call')).toBe(true)
    const result = events.find(event => event.type === 'tool/result')
    expect(result?.type === 'tool/result' && result.data.error).toEqual({
      name: 'ToolOutcomeUnknownError', code: TOOL_OUTCOME_UNKNOWN,
    })
    if (result?.type !== 'tool/result' || result.data.message.content[0]?.type !== 'text') {
      throw new Error('expected a text tool result')
    }
    expect(result.data.message.content[0].text).toContain('Do not retry blindly.')
  })

  it('repairs a real kill -9 tail through the product resume() path', async () => {
    // The manual `interruptedTurnClosers` read above proves the log is readable;
    // this member proves the shipped repair path runs against the same durable
    // artifact: resume() must append the closers itself, not the test.
    const crashed = await crashAt('request')
    const ctx = new Context()
    await mountAgentLoopTestDependencies(ctx)
    await ctx.plugin(AgentLoop, { agents: [] })
    await ctx.plugin(JsonlSessionPersistence, { root: crashed.root, compression: 'none' })
    const probe = await ctx.sessionPersistence.open(sessionId, 'read')
    let before: readonly SessionEvent[]
    try {
      before = (await probe.read()).events
    } finally {
      await probe.close()
    }
    // The durability claim only means something if the crash artifact really has
    // no closure: resume(), not the test, has to write it.
    expect(before.some(event => event.type === 'turn/end'), 'pre-resume log already closed').toBe(false)
    const handle = await ctx.agents.resume({
      resumeSessionId: sessionId,
      agentOptions: { provider: 'crash', model: 'crash' },
    })
    await handle.dispose()
    const reader = await ctx.sessionPersistence.open(sessionId, 'read')
    let stored: readonly SessionEvent[]
    try {
      stored = (await reader.read()).events
    } finally {
      await reader.close()
    }
    const closing = stored.filter(event => event.type === 'turn/end')
    expect(closing.length).toBeGreaterThan(0)
    expect(closing.at(-1)).toMatchObject({ data: { reason: { kind: 'interrupted' } } })
    await ctx.fiber.dispose()
  })

  it('repairs a real kill -9 inside a tool call through the product resume() path', async () => {
    // The tool failpoint leaves an announced call with no result; only the
    // shipped repair may write the unknown-outcome result and the closure.
    const crashed = await crashAt('tool')
    const ctx = new Context()
    await mountAgentLoopTestDependencies(ctx)
    await ctx.plugin(AgentLoop, { agents: [] })
    await ctx.plugin(JsonlSessionPersistence, { root: crashed.root, compression: 'none' })
    const readStored = async (): Promise<readonly SessionEvent[]> => {
      const reader = await ctx.sessionPersistence.open(sessionId, 'read')
      try {
        return (await reader.read()).events
      } finally {
        await reader.close()
      }
    }
    expect((await readStored()).some(event => event.type === 'turn/end'), 'pre-resume log already closed').toBe(false)
    const handle = await ctx.agents.resume({
      resumeSessionId: sessionId,
      agentOptions: { provider: 'crash', model: 'crash' },
    })
    await handle.dispose()
    const stored = await readStored()
    const result = stored.find(event => event.type === 'tool/result')
    expect(result?.type === 'tool/result' && result.data.error).toEqual({
      name: 'ToolOutcomeUnknownError', code: TOOL_OUTCOME_UNKNOWN,
    })
    expect(stored.filter(event => event.type === 'turn/end').at(-1))
      .toMatchObject({ data: { reason: { kind: 'interrupted' } } })
    await ctx.fiber.dispose()
  })
})
