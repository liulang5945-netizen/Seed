/** Wire shape, chunk sequence, and every failure the runtime can produce. */
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@taiji/cordis'
import LlmRuntime, {
  BlockAssembler,
  createAssistantMessage,
  createDeveloperMessage,
  createSystemMessage,
  createUserMessage,
  ToolCallId,
} from '@taiji/dsh-llm'
import type { GenerateOptions, StreamChunk } from '@taiji/dsh-llm'
import * as LlmTaiji from '../src/index.ts'
import { httpFailure } from '../src/transport.ts'
import type { Behavior } from './mock-runtime.ts'
import { closeMockRuntimes, closedRuntimeUrl, DONE, failureFrame, finalFrame, mockRuntime } from './mock-runtime.ts'

const PROVIDER = 'taiji-local'
const MODEL = 'taiji-local'

const user = (text: string) => createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text }] })
const assistant = (text: string) => createAssistantMessage({
  content: text.length === 0 ? [] : [{ type: 'text', text }],
  source: { provider: PROVIDER, model: MODEL },
})
const call = (overrides: Partial<GenerateOptions> = {}): GenerateOptions => ({
  provider: PROVIDER,
  model: MODEL,
  messages: [user('你好')],
  ...overrides,
})

/** A branded session id; the adapter only ever stringifies it. */
const sessionOf = (id: string) => id as GenerateOptions['sessionId']

async function collect(stream: AsyncIterable<StreamChunk>): Promise<StreamChunk[]> {
  const chunks: StreamChunk[] = []
  for await (const chunk of stream) chunks.push(chunk)
  return chunks
}

function adapterOf(baseURL: string, config: Partial<LlmTaiji.Options> = {}): LlmTaiji.TaijiAdapter {
  return new LlmTaiji.TaijiAdapter({ options: () => LlmTaiji.resolveAdapterOptions({ baseURL, ...config }) })
}

/** Start a stand-in runtime with one scripted chat behavior. */
async function runtimeFor(...behavior: Behavior[]) {
  const runtime = await mockRuntime()
  runtime.script.push(...behavior)
  return runtime
}

/** A composed host with the route mounted, exactly as a profile composes it. */
async function pluginContext(baseURL: string): Promise<Context> {
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(LlmTaiji, { baseURL })
  return ctx
}

afterEach(async () => {
  await closeMockRuntimes()
  vi.unstubAllGlobals()
})

describe('Taiji chat request', () => {
  it('streams one text block and the terminal stop for a final frame', async () => {
    const runtime = await runtimeFor({ kind: 'frames', frames: [finalFrame('你好，世界'), DONE] })

    const chunks = await collect(adapterOf(runtime.url).stream(call()))

    expect(chunks).toEqual([
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: '你好，世界' },
      { type: 'block-end', index: 0, block: { type: 'text', text: '你好，世界' } },
      { type: 'finish', reason: { kind: 'stop' } },
    ])
  })

  it('routes a registry request through the runtime and assembles its answer', async () => {
    const runtime = await runtimeFor({ kind: 'frames', frames: [finalFrame('你好，世界'), DONE] })
    const ctx = await pluginContext(runtime.url)
    const assembler = new BlockAssembler()

    for await (const chunk of ctx.llm.stream(call())) assembler.push(chunk)

    expect(assembler.blocks()).toEqual([{ type: 'text', text: '你好，世界' }])
    expect(assembler.finish).toEqual({ kind: 'stop' })
    // The runtime reports no token accounting, so none is invented.
    expect(assembler.usage).toBeUndefined()
    await ctx.fiber.dispose()
  })

  it('sends the current user turn, the last system prompt, and completed pairs', async () => {
    const runtime = await runtimeFor({ kind: 'frames', frames: [finalFrame('answer'), DONE] })
    const messages = [
      createSystemMessage('你是Seed'),
      user('第一个问题'),
      assistant('第一个回答'),
      user('第二个问题'),
    ]

    await collect(adapterOf(runtime.url).stream(call({ messages })))

    expect(runtime.requests).toHaveLength(1)
    expect(runtime.requests[0]?.path).toBe('/api/chat/stream')
    expect(runtime.requests[0]?.headers.accept).toBe('text/event-stream')
    expect(runtime.requests[0]?.headers['content-type']).toBe('application/json')
    expect(runtime.requests[0]?.headers['user-agent']).toContain('deepseek-harness/')
    expect(runtime.requests[0]?.body).toEqual({
      prompt: '第二个问题',
      system_prompt: '你是Seed',
      history: [['第一个问题', '第一个回答']],
    })
  })

  it('takes the one-shot system field when no system message carries a prompt', async () => {
    const runtime = await runtimeFor({ kind: 'frames', frames: [finalFrame('answer'), DONE] })

    await collect(adapterOf(runtime.url).stream(call({ system: '你是Seed' })))

    expect(runtime.requests[0]?.body).toEqual({ prompt: '你好', system_prompt: '你是Seed', history: [] })
  })

  it('omits the system prompt so the runtime keeps its own default persona', async () => {
    const runtime = await runtimeFor({ kind: 'frames', frames: [finalFrame('answer'), DONE] })

    await collect(adapterOf(runtime.url).stream(call()))

    expect(runtime.requests[0]?.body).toEqual({ prompt: '你好', history: [] })
  })

  it('carries an empty prompt when the request has no user turn', async () => {
    const runtime = await runtimeFor({ kind: 'frames', frames: [finalFrame('answer'), DONE] })

    await collect(adapterOf(runtime.url).stream(call({ messages: [] })))

    expect(runtime.requests[0]?.body).toEqual({ prompt: '', history: [] })
  })

  it('drops the blocks and roles the runtime shape has no slot for', async () => {
    const runtime = await runtimeFor({ kind: 'frames', frames: [finalFrame('answer'), DONE] })
    const reasoningUser = createUserMessage({
      source: { kind: 'user' },
      content: [{ type: 'reasoning', text: 'internal' }, { type: 'text', text: '问题' }],
    })
    const toolTurn = createAssistantMessage({
      content: [{ type: 'tool-call', id: ToolCallId('call-1'), name: 'read', arguments: '{}' }],
      source: { provider: PROVIDER, model: MODEL },
    })
    const messages = [
      createDeveloperMessage({ source: { kind: 'user' }, content: [{ type: 'tool-addition', toolName: 'read' }] }),
      reasoningUser,
      toolTurn,
      user('继续'),
    ]

    await collect(adapterOf(runtime.url).stream(call({ messages })))

    expect(runtime.requests[0]?.body).toEqual({
      prompt: '继续',
      history: [['问题', '']],
    })
  })

  it('keeps a user turn that never received a reply', async () => {
    const runtime = await runtimeFor({ kind: 'frames', frames: [finalFrame('answer'), DONE] })

    await collect(adapterOf(runtime.url).stream(call({ messages: [user('一'), user('二'), assistant('二答'), user('三')] })))

    expect(runtime.requests[0]?.body).toEqual({
      prompt: '三',
      history: [['一', ''], ['二', '二答']],
    })
  })

  it('ignores an assistant turn that opens the history and keeps an unanswered user turn', async () => {
    const runtime = await runtimeFor({ kind: 'frames', frames: [finalFrame('answer'), DONE] })

    await collect(adapterOf(runtime.url).stream(call({ messages: [assistant('早'), user('一'), user('二')] })))

    expect(runtime.requests[0]?.body).toEqual({
      prompt: '二',
      history: [['一', '']],
    })
  })

  it('carries the session, purpose, and offered tool names as request metadata', async () => {
    const runtime = await runtimeFor({ kind: 'frames', frames: [finalFrame('answer'), DONE] })

    await collect(adapterOf(runtime.url).stream(call({
      sessionId: sessionOf('session-1'),
      purpose: 'session-title',
      tools: [
        { name: 'bash', description: 'run a command', parameters: {} },
        { name: 'read', description: 'read a file', parameters: {} },
      ],
    })))

    expect(runtime.requests[0]?.body).toEqual({
      prompt: '你好',
      history: [],
      session_id: 'session-1',
      purpose: 'session-title',
      tools: ['bash', 'read'],
    })
  })

  it('omits the metadata an ordinary request does not carry', async () => {
    const runtime = await runtimeFor({ kind: 'frames', frames: [finalFrame('answer'), DONE] })

    await collect(adapterOf(runtime.url).stream(call({ tools: [] })))

    expect(runtime.requests[0]?.body).toEqual({ prompt: '你好', history: [] })
  })
})

describe('Taiji chat failures', () => {
  it.each([
    { name: 'the in-band failure frame', frames: [failureFrame('boom')] as string[], failure: { code: 'SERVER', message: '生成出错: boom' } },
    { name: 'a frame that is not JSON', frames: ['生成出错'], failure: { code: 'MALFORMED_RESPONSE', message: 'Taiji runtime SSE frame is not JSON' } },
    { name: 'a non-object frame', frames: ['null'], failure: { code: 'MALFORMED_RESPONSE', message: 'Taiji runtime SSE frame has no answer shape' } },
    { name: 'a frame this route does not serve', frames: [JSON.stringify({ type: 'message_start' })], failure: { code: 'MALFORMED_RESPONSE', message: 'Taiji runtime answered with a frame this route does not serve' } },
    { name: 'a final frame without text', frames: [JSON.stringify({ type: 'final', data: { step: 1 } })], failure: { code: 'MALFORMED_RESPONSE', message: 'Taiji runtime final frame carries no answer text' } },
    { name: 'a final frame whose data is not an object', frames: [JSON.stringify({ type: 'final', data: 'answer' })], failure: { code: 'MALFORMED_RESPONSE', message: 'Taiji runtime final frame carries no answer text' } },
    { name: 'an empty answer', frames: [finalFrame('')], failure: { code: 'EMPTY_RESPONSE', message: 'Taiji runtime answered with an empty text' } },
    { name: 'a stream that closes before any answer', frames: [DONE], failure: { code: 'STREAM_CLOSED', message: 'Taiji runtime stream closed before a final answer' } },
  ])('normalizes $name into the terminal error finish', async ({ frames, failure }) => {
    const runtime = await runtimeFor({ kind: 'frames', frames })
    const ctx = await pluginContext(runtime.url)

    expect(await collect(ctx.llm.stream(call()))).toEqual([
      { type: 'finish', reason: { kind: 'error', failure } },
    ])
    await ctx.fiber.dispose()
  })

  it('normalizes a bare HTTP failure into the terminal error finish', async () => {
    const runtime = await runtimeFor({ kind: 'http-error', status: 500, body: JSON.stringify({ detail: 'generation failed' }) })
    const ctx = await pluginContext(runtime.url)

    expect(await collect(ctx.llm.stream(call()))).toEqual([
      { type: 'finish', reason: { kind: 'error', failure: {
        code: 'SERVER', status: 500, message: 'generation failed',
      } } },
    ])
    await ctx.fiber.dispose()
  })

  it('reports the runtime refusal status when its body carries no reason', async () => {
    const runtime = await runtimeFor({ kind: 'http-error', status: 409, body: 'Seed runtime is not active' })

    await expect(collect(adapterOf(runtime.url).stream(call())))
      .rejects.toMatchObject({ code: 'HTTP_409', message: 'Taiji runtime request failed (409)' })
  })

  it('reports a transport failure when nothing listens on the endpoint', async () => {
    await expect(collect(adapterOf(await closedRuntimeUrl()).stream(call())))
      .rejects.toMatchObject({ code: 'TRANSPORT' })
  })

  it('reports a closed stream when the runtime answers with no body', async () => {
    vi.stubGlobal('fetch', () => Promise.resolve(new Response(null, { status: 200 })))

    await expect(collect(adapterOf('http://127.0.0.1:1').stream(call())))
      .rejects.toMatchObject({ code: 'STREAM_CLOSED', message: 'Taiji runtime returned no response body' })
  })

  it('ends the call as aborted when the request is cancelled', async () => {
    const runtime = await runtimeFor({ kind: 'hold' })
    const ctx = await pluginContext(runtime.url)
    const controller = new AbortController()

    const pending = collect(ctx.llm.stream(call({ signal: controller.signal })))
    await vi.waitFor(() => { expect(runtime.requests).toHaveLength(1) })
    controller.abort()

    expect(await pending).toEqual([
      { type: 'finish', reason: { kind: 'aborted', failure: { code: 'ABORTED', message: 'Taiji runtime request aborted' } } },
    ])
    await ctx.fiber.dispose()
  })
})

describe('Taiji HTTP failure classification', () => {
  it.each([
    { status: 401, body: '{}', code: 'AUTH' },
    { status: 402, body: '{}', code: 'QUOTA' },
    { status: 429, body: '{}', code: 'RATE_LIMIT' },
    { status: 400, body: JSON.stringify({ detail: 'maximum context length exceeded' }), code: 'CONTEXT_WINDOW_EXCEEDED' },
    { status: 422, body: JSON.stringify({ message: 'invalid request' }), code: 'INVALID_REQUEST' },
    { status: 503, body: '{}', code: 'SERVER' },
    { status: 404, body: '{}', code: 'HTTP_404' },
    { status: 402, body: JSON.stringify({ detail: 'insufficient balance' }), code: 'QUOTA' },
  ])('maps HTTP $status to $code', async ({ status, body, code }) => {
    const failure = await httpFailure(new Response(body, { status }))
    expect(failure.failure).toMatchObject({ code, status })
    expect(failure.failure?.message).toEqual(expect.any(String))
  })

  it('reads the reason from the detail, then the message field', async () => {
    expect(await httpFailure(new Response(JSON.stringify({ detail: 'from detail' }), { status: 400 })))
      .toMatchObject({ message: 'from detail' })
    expect(await httpFailure(new Response(JSON.stringify({ message: 'from message' }), { status: 400 })))
      .toMatchObject({ message: 'from message' })
    expect(await httpFailure(new Response('null', { status: 400 })))
      .toMatchObject({ message: 'Taiji runtime request failed (400)' })
    expect(await httpFailure(new Response(JSON.stringify({ detail: 7 }), { status: 400 })))
      .toMatchObject({ message: 'Taiji runtime request failed (400)' })
  })
})

describe('Taiji model metadata', () => {
  it('advertises its catalog and resolves exact model metadata', async () => {
    const adapter = adapterOf('http://127.0.0.1:1', {
      models: [{ id: 'runtime', name: 'Runtime', description: 'local runtime', contextWindow: 4096 }, { id: 'bare' }],
    })

    expect(adapter.providerInfo(PROVIDER)).toEqual({ id: PROVIDER, name: LlmTaiji.RUNTIME_DISPLAY_NAME })
    expect(adapter.providerRetryPolicy(PROVIDER)).toMatchObject({ mode: 'normal', maxRetries: 5 })
    // An entry naming neither a label nor a description advertises its id alone.
    expect(await adapter.listModels(PROVIDER)).toEqual([
      { provider: PROVIDER, id: 'runtime', name: 'Runtime', description: 'local runtime', inputModalities: ['text'] },
      { provider: PROVIDER, id: 'bare', name: 'bare', inputModalities: ['text'] },
    ])
    expect(await adapter.resolveModel(PROVIDER, 'runtime')).toEqual({
      provider: PROVIDER,
      id: 'runtime',
      name: 'Runtime',
      description: 'local runtime',
      context: { contextWindow: 4096 },
      inputModalities: ['text'],
    })
    // The endpoint takes no model id, so an unlisted one stays routable.
    expect(await adapter.resolveModel(PROVIDER, 'unlisted')).toEqual({
      provider: PROVIDER,
      id: 'unlisted',
      name: 'unlisted',
      inputModalities: ['text'],
    })
    expect(await adapter.resolveModel(PROVIDER, 'bare')).toEqual({
      provider: PROVIDER,
      id: 'bare',
      name: 'bare',
      inputModalities: ['text'],
    })
  })
})
