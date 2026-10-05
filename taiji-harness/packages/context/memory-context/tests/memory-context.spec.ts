/**
 * Real-behaviour tests for the memory recall reader: a loopback HTTP server
 * stands in for the Taiji runtime and records what the plugin asked for.
 */
import { createServer } from 'node:http'
import type { Server, ServerResponse } from 'node:http'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@taiji/cordis'
import { agentEvents } from '@taiji/dsh-agent'
import type { Agent, PreStepDecision } from '@taiji/dsh-agent'
import { createUserMessage } from '@taiji/dsh-llm'
import type { UserMessage } from '@taiji/dsh-llm'
import { Session, SessionId } from '@taiji/dsh-session'
import * as memoryContext from '../src/index.ts'
import type { Config } from '../src/index.ts'

const SIGNAL = new AbortController().signal
const QUESTION = '发布检查点'
const ONE_ENTRY = JSON.stringify({
  query: QUESTION,
  entries: [{
    entry_id: 'entry-1',
    kind: 'interaction',
    text: '问：如何发布检查点\n答：我无法发布。',
    session_id: 'earlier-session',
    turn: 7,
    tags: ['provider:taiji-local'],
    importance: 0.55,
    source: 'taiji-harness',
    metadata: {},
    digest: 'digest-1',
    recorded_at: 1_790_225_734.5,
    score: 0.62,
  }],
})
const BLOCK = [
  `memory entries=1 query="${QUESTION}"`,
  memoryContext.RECALL_NOTE,
  '- [0.62 interaction] 问：如何发布检查点 / 答：我无法发布。',
].join('\n')

const cleanups: Array<() => Promise<unknown>> = []
afterEach(async () => {
  for (const cleanup of cleanups.reverse()) await cleanup()
  cleanups.length = 0
  vi.restoreAllMocks()
})

/** One request the fake runtime received, captured verbatim. */
interface CapturedRequest {
  readonly url: string
}

/** The loopback runtime, its server handle, and the requests it recorded. */
interface FakeRuntime {
  readonly baseURL: string
  readonly server: Server
  readonly requests: CapturedRequest[]
}

/** How a fake runtime answers one captured request. */
type Respond = (request: CapturedRequest, response: ServerResponse) => void

/** Answer every request with one body and status. */
function answer(body: string, status = 200): Respond {
  return (_request, response) => {
    response.writeHead(status, { 'content-type': 'application/json' })
    response.end(body)
  }
}

/** Close one loopback server and every connection it still holds. */
function closeServer(server: Server): Promise<void> {
  if (!server.listening) return Promise.resolve()
  server.closeAllConnections()
  return new Promise((resolve, reject) => {
    server.close((error) => {
      if (error === undefined) resolve()
      else reject(error)
    })
  })
}

/** Start a loopback runtime that records every request; `respond` defaults to one ranked entry. */
async function startRuntime(respond: Respond = answer(ONE_ENTRY)): Promise<FakeRuntime> {
  const requests: CapturedRequest[] = []
  const server = createServer((request, response) => {
    const captured: CapturedRequest = { url: request.url ?? '' }
    requests.push(captured)
    respond(captured, response)
  })
  await new Promise<void>((resolve) => { server.listen(0, '127.0.0.1', resolve) })
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('loopback server bound no port')
  cleanups.push(() => closeServer(server))
  return { baseURL: `http://127.0.0.1:${String(address.port)}`, server, requests }
}

/** Mount the plugin under test. */
async function mount(config: Config = {}): Promise<Context> {
  const ctx = new Context()
  cleanups.push(() => ctx.fiber.dispose())
  await ctx.plugin(memoryContext, config)
  return ctx
}

/** A stub Agent carrying only what this plugin reads: the session and its identity. */
function stubAgent(session: Session): Agent {
  return { id: session.id, session, options: {} } as Agent
}

/** One message the user sent. */
function userMessage(text: string): UserMessage {
  return createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } })
}

/** One message this plugin itself injected on an earlier step. */
function injectedContext(text: string): UserMessage {
  return createUserMessage({
    content: [{ type: 'text', text }],
    source: { kind: 'memory-context', form: 'snapshot', sections: [{ name: 'memory-context', text }] },
  })
}

/** Dispatch the pre-step waterfall for one step and return the entered messages. */
async function fire(
  ctx: Context,
  agent: Agent,
  messages: readonly UserMessage[],
  turn = 1,
  step = 1,
  signal: AbortSignal = SIGNAL,
): Promise<readonly UserMessage[]> {
  const proposed = [...messages]
  const decision: PreStepDecision = await agentEvents(ctx, agent).waterfall(
    'agent/pre-step',
    { messages: proposed, turn, step, signal },
    () => Promise.resolve({ kind: 'enter' as const, messages: proposed }),
  )
  if (decision.kind !== 'reject') return decision.messages
  throw new Error('unexpected reject')
}

/** Give a negative assertion time to be disproved. */
function quiet(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 50))
}

describe('memory-context refusal and eviction paths', () => {
  it('refuses malformed recall bodies with one warning and no injection', async () => {
    const bodies: readonly string[] = [
      'null',
      '{"entries":123}',
      '{"entries":[123]}',
      '{"entries":[{"kind":"memory","text":"t","score":"0.5"}]}',
      '{"entries":[{"kind":"memory","text":"t","score":null}]}',
    ]
    for (const [index, body] of bodies.entries()) {
      const runtime = await startRuntime(answer(body))
      const ctx = await mount({ baseURL: runtime.baseURL })
      const warn = vi.spyOn(ctx.logger, 'warn')
      const agent = stubAgent(Session.create(SessionId(`malformed-${String(index)}`)))
      const question = userMessage(QUESTION)

      // A body outside the documented shape is refused rather than rendered thin,
      // and the step still proceeds with what the user sent.
      expect(await fire(ctx, agent, [question], 1, 1)).toEqual([question])
      expect(warn).toHaveBeenCalledTimes(1)
    }
  })

  it('ignores blocks that carry no text when it builds the query', async () => {
    const runtime = await startRuntime()
    const ctx = await mount({ baseURL: runtime.baseURL })
    const agent = stubAgent(Session.create(SessionId('image-block')))
    const message = createUserMessage({
      content: [{ type: 'image', attachment: {} as never }, { type: 'text', text: QUESTION }],
      source: { kind: 'user' },
    })

    expect((await fire(ctx, agent, [message], 1, 1))[0]?.source.kind).toBe('memory-context')
  })

  it('returns an aborted step before the turn is claimed or the runtime is asked', async () => {
    const runtime = await startRuntime()
    const ctx = await mount({ baseURL: runtime.baseURL })
    const agent = stubAgent(Session.create(SessionId('aborted')))
    const question = userMessage(QUESTION)

    expect(await fire(ctx, agent, [question], 1, 1, AbortSignal.abort())).toEqual([question])
    expect(runtime.requests).toHaveLength(0)
  })

  it('skips the recall when the step holds nothing the user actually sent', async () => {
    const runtime = await startRuntime()
    const ctx = await mount({ baseURL: runtime.baseURL })
    const agent = stubAgent(Session.create(SessionId('no-query')))
    const injected = injectedContext(BLOCK)

    expect(await fire(ctx, agent, [injected], 1, 1)).toEqual([injected])
    expect(runtime.requests).toHaveLength(0)
  })

  it('serves the same turn again once the session has been disposed', async () => {
    const runtime = await startRuntime()
    const ctx = await mount({ baseURL: runtime.baseURL })
    const session = Session.create(SessionId('reused'))
    const agent = stubAgent(session)
    const question = userMessage(QUESTION)

    expect(await fire(ctx, agent, [question], 1, 1)).toHaveLength(2)
    expect(await fire(ctx, agent, [question], 1, 2)).toEqual([question])

    ctx.emit('session/disposed', session)
    expect((await fire(ctx, agent, [question], 1, 3))[0]?.source.kind).toBe('memory-context')
  })
})

describe('memory-context recall', () => {
  it('injects one durable message on the first step and nothing on a later step of the same turn', async () => {
    const runtime = await startRuntime()
    const ctx = await mount({ baseURL: runtime.baseURL })
    const agent = stubAgent(Session.create(SessionId('once')))
    const question = userMessage(QUESTION)

    const first = await fire(ctx, agent, [question], 1, 1)
    expect(first).toHaveLength(2)
    expect(first[0]?.source.kind).toBe('memory-context')
    expect(first[0]?.content).toEqual([{ type: 'text', text: BLOCK }])
    expect(first[1]).toBe(question)

    const second = await fire(ctx, agent, [userMessage(QUESTION)], 1, 2)
    expect(second).toHaveLength(1)
    expect(runtime.requests).toHaveLength(1)
  })

  it('still recalls the same turn for a different session', async () => {
    const runtime = await startRuntime()
    const ctx = await mount({ baseURL: runtime.baseURL })

    expect(await fire(ctx, stubAgent(Session.create(SessionId('first'))), [userMessage(QUESTION)], 1, 1)).toHaveLength(2)
    expect(await fire(ctx, stubAgent(Session.create(SessionId('second'))), [userMessage(QUESTION)], 1, 1)).toHaveLength(2)
    expect(runtime.requests).toHaveLength(2)
  })

  it('takes the query from the user-sourced message, not from injected context', async () => {
    const runtime = await startRuntime()
    const ctx = await mount({ baseURL: runtime.baseURL })
    const asked = 'the real question'

    await fire(ctx, stubAgent(Session.create(SessionId('query'))), [
      userMessage(asked),
      injectedContext('recalled memory injected on an earlier step'),
    ], 1, 1)

    const [request] = runtime.requests
    if (request === undefined) throw new Error('no request was captured')
    const url = new URL(request.url, runtime.baseURL)
    expect(url.pathname).toBe('/api/memory/recall')
    expect(url.searchParams.get('query')).toBe(asked)
  })

  it('sends no request and injects nothing when disabled', async () => {
    const runtime = await startRuntime()
    const ctx = await mount({ baseURL: runtime.baseURL, enabled: false })

    const messages = await fire(ctx, stubAgent(Session.create(SessionId('disabled'))), [userMessage(QUESTION)], 1, 1)
    await quiet()
    expect(messages).toHaveLength(1)
    expect(runtime.requests).toHaveLength(0)
  })

  it('injects nothing when recall found no entry', async () => {
    const runtime = await startRuntime(answer(JSON.stringify({ query: QUESTION, entries: [] })))
    const ctx = await mount({ baseURL: runtime.baseURL })

    const messages = await fire(ctx, stubAgent(Session.create(SessionId('empty'))), [userMessage(QUESTION)], 1, 1)
    expect(messages).toHaveLength(1)
    expect(runtime.requests).toHaveLength(1)
  })

  it('injects nothing on a failed recall and warns only once across two failed turns', async () => {
    const runtime = await startRuntime(answer('{}', 500))
    const ctx = await mount({ baseURL: runtime.baseURL })
    const warn = vi.spyOn(ctx.logger, 'warn').mockImplementation(() => undefined)

    expect(await fire(ctx, stubAgent(Session.create(SessionId('down-1'))), [userMessage(QUESTION)], 1, 1)).toHaveLength(1)
    expect(await fire(ctx, stubAgent(Session.create(SessionId('down-2'))), [userMessage(QUESTION)], 1, 1)).toHaveLength(1)

    expect(warn).toHaveBeenCalledTimes(1)
    expect(String(warn.mock.calls[0]?.[0])).toContain('HTTP 500')
  })

  it('injects nothing when the runtime never answers', async () => {
    const runtime = await startRuntime(() => undefined)
    const ctx = await mount({ baseURL: runtime.baseURL, timeoutMs: 50 })

    const messages = await fire(ctx, stubAgent(Session.create(SessionId('silent'))), [userMessage(QUESTION)], 1, 1)
    expect(messages).toHaveLength(1)
    expect(runtime.requests).toHaveLength(1)
  })

  it('injects nothing when the port is closed', async () => {
    const runtime = await startRuntime()
    const ctx = await mount({ baseURL: runtime.baseURL })
    await closeServer(runtime.server)

    const messages = await fire(ctx, stubAgent(Session.create(SessionId('closed'))), [userMessage(QUESTION)], 1, 1)
    expect(messages).toHaveLength(1)
  })

  it('injects nothing when the response body is malformed', async () => {
    const runtime = await startRuntime(answer('not json'))
    const ctx = await mount({ baseURL: runtime.baseURL })

    expect(await fire(ctx, stubAgent(Session.create(SessionId('bad-json'))), [userMessage(QUESTION)], 1, 1)).toHaveLength(1)
    expect(await fire(ctx, stubAgent(Session.create(SessionId('bad-body'))), [userMessage(QUESTION)], 1, 1)).toHaveLength(1)
  })

  it('passes limit to the runtime and keeps the block inside maxChars', async () => {
    const runtime = await startRuntime()
    const ctx = await mount({ baseURL: runtime.baseURL, limit: 3, maxChars: 600 })

    const messages = await fire(ctx, stubAgent(Session.create(SessionId('limit'))), [userMessage(QUESTION)], 1, 1)
    const [request] = runtime.requests
    if (request === undefined) throw new Error('no request was captured')
    expect(new URL(request.url, runtime.baseURL).searchParams.get('limit')).toBe('3')
    const block = messages[0]?.content[0]
    if (block?.type !== 'text') throw new Error('expected an injected text block')
    expect(block.text).toBe(BLOCK)
    expect(block.text.length).toBeLessThanOrEqual(600)
  })

  it('fails loud on an impossible endpoint, entry budget, block budget, or timeout', async () => {
    await expect(new Context().plugin(memoryContext, { baseURL: '' })).rejects.toThrow(/baseURL/)
    await expect(new Context().plugin(memoryContext, { limit: 0 })).rejects.toThrow(/limit/)
    await expect(new Context().plugin(memoryContext, { maxChars: 79 })).rejects.toThrow(/maxChars/)
    await expect(new Context().plugin(memoryContext, { timeoutMs: 0 })).rejects.toThrow(/timeoutMs/)
  })
})
