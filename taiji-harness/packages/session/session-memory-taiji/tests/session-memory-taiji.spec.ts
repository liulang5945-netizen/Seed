/**
 * Real-behaviour tests for the session-memory-taiji reporter: a loopback HTTP
 * server stands in for the Taiji runtime and records what the plugin sent.
 */
import { createServer } from 'node:http'
import type { ServerResponse } from 'node:http'
import { basename } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@taiji/cordis'
import type { Agent } from '@taiji/dsh-agent'
import { ToolCallId, createAssistantMessage, createToolResultMessage, createUserMessage } from '@taiji/dsh-llm'
import type { ContentBlock } from '@taiji/dsh-llm'
import SessionStore, { SessionId } from '@taiji/dsh-session'
import type { Session } from '@taiji/dsh-session'
import * as SessionMemory from '../src/index.ts'

const SIGNAL = new AbortController().signal
const PROMPT = 'Summarize the plan.'
const ANSWER = 'Done.'
const TOOL = 'read'
const PROVIDER = 'taiji-local'
const MODEL = 'taiji-1'
const CWD = '/srv/checkout/demo'

const cleanups: Array<() => Promise<unknown>> = []
afterEach(async () => {
  for (const cleanup of cleanups.reverse()) await cleanup()
  cleanups.length = 0
  vi.restoreAllMocks()
})

/** One request the fake runtime received, captured verbatim. */
interface CapturedRequest {
  readonly method: string
  readonly url: string
  readonly contentType: string | undefined
  readonly body: string
}

/** The loopback runtime and the requests it recorded. */
interface FakeRuntime {
  readonly baseURL: string
  readonly requests: CapturedRequest[]
}

/** How a fake runtime answers one captured request. */
type Respond = (request: CapturedRequest, response: ServerResponse) => void

/** Start a loopback runtime that records every request; `respond` defaults to a `recorded` reply. */
async function startRuntime(respond?: Respond): Promise<FakeRuntime> {
  const requests: CapturedRequest[] = []
  const server = createServer((request, response) => {
    let body = ''
    request.setEncoding('utf8')
    request.on('data', (chunk: string) => { body += chunk })
    request.on('end', () => {
      const captured: CapturedRequest = {
        method: request.method ?? '',
        url: request.url ?? '',
        contentType: typeof request.headers['content-type'] === 'string' ? request.headers['content-type'] : undefined,
        body,
      }
      requests.push(captured)
      if (respond === undefined) {
        response.writeHead(200, { 'content-type': 'application/json' })
        response.end(JSON.stringify({ status: 'recorded', entry_id: 'entry-1', digest: 'digest-1' }))
      } else {
        respond(captured, response)
      }
    })
  })
  await new Promise<void>((resolve) => { server.listen(0, '127.0.0.1', resolve) })
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('loopback server bound no port')
  cleanups.push(() => new Promise<void>((resolve, reject) => {
    server.close((error) => {
      if (error === undefined) resolve()
      else reject(error)
    })
  }))
  return { baseURL: `http://127.0.0.1:${String(address.port)}`, requests }
}

/** Mount the session store and the reporter under test. */
async function boot(config: SessionMemory.Config = {}): Promise<Context> {
  const ctx = new Context()
  cleanups.push(() => ctx.fiber.dispose())
  await ctx.plugin(SessionStore)
  await ctx.plugin(SessionMemory, config)
  return ctx
}

/** Append one closed turn whose prompt, answer, and tool the reporter reads. */
function appendTurn(session: Session, turn: number, options: { prompt?: string; answer?: string; tool?: string }): void {
  session.append('turn/start', { turn })
  session.append('step/start', { turn, step: 1 })
  session.append('request/context', { provider: PROVIDER, model: MODEL })
  if (options.prompt !== undefined) {
    session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: options.prompt }],
      source: { kind: 'user' },
    }), { surfaceOp: 'append' })
  }
  const blocks: ContentBlock[] = []
  if (options.answer !== undefined) blocks.push({ type: 'text', text: options.answer })
  const callId = ToolCallId(`call-${String(turn)}`)
  if (options.tool !== undefined) blocks.push({ type: 'tool-call', id: callId, name: options.tool, arguments: '{}' })
  if (blocks.length > 0) {
    session.append('assistant/message', {
      stream: [], turn, step: 1,
      message: createAssistantMessage({ content: blocks, source: { provider: PROVIDER, model: MODEL } }),
    }, { surfaceOp: 'append' })
  }
  if (options.tool !== undefined) {
    const call = session.append('tool/call', { turn, step: 1, callId, name: options.tool, arguments: '{}' })
    session.append('tool/result', {
      turn, step: 1,
      message: createToolResultMessage({ callId, content: [{ type: 'text', text: 'ok' }], isError: false }),
    }, { surfaceOp: 'append', sourceEventSeqs: [call.seq] })
  }
  session.append('step/end', { turn, step: 1 })
  session.append('turn/end', { turn, reason: { kind: 'completed' } })
}

/** A top-level session with a working directory and one recorded turn. */
function reportedSession(ctx: Context, id: string, options: { prompt?: string; answer?: string; tool?: string }): Session {
  const session = ctx.sessions.create(SessionId(id), { meta: { cwd: CWD } })
  appendTurn(session, 1, options)
  return session
}

/** Dispatch the stop boundary the plugin observes. */
async function stop(ctx: Context, session: Session, turn = 1): Promise<void> {
  const agent = { session } as Agent
  await ctx.serial('agent/turn-stopping', { agent, turn, signal: SIGNAL })
}

/** Poll until `predicate` holds, failing after a bounded wait. */
async function until(predicate: () => boolean, message: string): Promise<void> {
  for (let attempt = 0; attempt < 200; attempt += 1) {
    if (predicate()) return
    await new Promise(resolve => setTimeout(resolve, 5))
  }
  throw new Error(message)
}

/** Give a negative assertion time to be disproved. */
function quiet(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 50))
}

describe('session-memory-taiji reporting', () => {
  it('reports one turn as the runtime memory record it expects', async () => {
    const runtime = await startRuntime()
    const ctx = await boot({ baseURL: runtime.baseURL })
    const session = reportedSession(ctx, 'memory-turn', { prompt: PROMPT, answer: ANSWER, tool: TOOL })

    await stop(ctx, session, 1)
    await until(() => runtime.requests.length === 1, 'the report never reached the runtime')

    const [request] = runtime.requests
    if (request === undefined) throw new Error('no request was captured')
    expect(request.method).toBe('POST')
    expect(request.url).toBe('/api/memory/record')
    expect(request.contentType).toBe('application/json')
    expect(JSON.parse(request.body)).toEqual({
      kind: 'interaction',
      text: `问：${PROMPT}\n答：${ANSWER}`,
      session_id: 'memory-turn',
      turn: 1,
      tags: [`provider:${PROVIDER}`, `model:${MODEL}`, `workspace:${basename(CWD)}`, 'tools:1'],
      importance: 0.55,
      source: 'taiji-harness',
      metadata: {
        provider: PROVIDER,
        model: MODEL,
        tools: [TOOL],
        toolCount: 1,
        answerChars: ANSWER.length,
        promptChars: PROMPT.length,
        aborted: false,
      },
    })
  })

  it('does not report the same turn twice', async () => {
    const runtime = await startRuntime()
    const ctx = await boot({ baseURL: runtime.baseURL })
    const session = reportedSession(ctx, 'duplicate', { prompt: PROMPT, answer: ANSWER })

    await stop(ctx, session, 1)
    await stop(ctx, session, 1)
    await until(() => runtime.requests.length === 1, 'the first report never arrived')
    await quiet()
    expect(runtime.requests).toHaveLength(1)
  })

  it('never blocks or throws when the runtime is down, and logs only the first failure', async () => {
    const releases: Array<() => void> = []
    const runtime = await startRuntime((_request, response) => {
      releases.push(() => {
        response.writeHead(500, { 'content-type': 'application/json' })
        response.end('{}')
      })
    })
    const ctx = await boot({ baseURL: runtime.baseURL })
    const debug = vi.spyOn(ctx.logger, 'debug').mockImplementation(() => undefined)

    await stop(ctx, reportedSession(ctx, 'down-1', { prompt: 'one', answer: 'a' }), 1)
    // The stop boundary returned before the runtime had even read the request.
    expect(releases).toHaveLength(0)
    await until(() => runtime.requests.length === 1, 'the first report never arrived')
    releases[0]?.()

    await stop(ctx, reportedSession(ctx, 'down-2', { prompt: 'two', answer: 'b' }), 1)
    await until(() => runtime.requests.length === 2, 'the second report never arrived')
    releases[1]?.()
    await quiet()

    expect(debug).toHaveBeenCalledTimes(1)
    expect(String(debug.mock.calls[0]?.[0])).toContain('HTTP 500')
  })

  it('sends nothing when disabled', async () => {
    const runtime = await startRuntime()
    const ctx = await boot({ baseURL: runtime.baseURL, enabled: false })
    await stop(ctx, reportedSession(ctx, 'disabled', { prompt: PROMPT, answer: ANSWER }), 1)
    await quiet()
    expect(runtime.requests).toHaveLength(0)
  })

  it('sends nothing for a turn with no prompt and no answer', async () => {
    const runtime = await startRuntime()
    const ctx = await boot({ baseURL: runtime.baseURL })
    await stop(ctx, reportedSession(ctx, 'empty', { prompt: '', answer: '' }), 1)
    await quiet()
    expect(runtime.requests).toHaveLength(0)
  })

  it('sends nothing for a subagent or delegated session', async () => {
    const runtime = await startRuntime()
    const ctx = await boot({ baseURL: runtime.baseURL })
    const child = ctx.sessions.create(SessionId('child'), { meta: { cwd: CWD, origin: 'subagent' } })
    appendTurn(child, 1, { prompt: PROMPT, answer: ANSWER })
    await stop(ctx, child, 1)
    const grandchild = ctx.sessions.create(SessionId('grandchild'), { meta: { cwd: CWD, delegationDepth: 1 } })
    appendTurn(grandchild, 1, { prompt: PROMPT, answer: ANSWER })
    await stop(ctx, grandchild, 1)
    await quiet()
    expect(runtime.requests).toHaveLength(0)
  })

  it('fails loud on an impossible budget, timeout, or endpoint', async () => {
    await expect(new Context().plugin(SessionMemory, { maxTextChars: 0 })).rejects.toThrow(/maxTextChars/)
    await expect(new Context().plugin(SessionMemory, { timeoutMs: 0 })).rejects.toThrow(/timeoutMs/)
    await expect(new Context().plugin(SessionMemory, { baseURL: '' })).rejects.toThrow(/baseURL/)
  })
})
