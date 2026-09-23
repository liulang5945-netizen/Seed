/** One text answer per request over the Taiji runtime's public HTTP surface. */

import { attributionHeaders, EMPTY_RESPONSE_CODE, LlmAdapter, LlmError } from '@taiji/dsh-llm'
import type { ContentBlock, GenerateOptions, LlmModelInfo, LlmResolvedModelInfo, StreamChunk } from '@taiji/dsh-llm'
import { buildChatRequest } from './chat.ts'
import { CHAT_STREAM_PATH, RUNTIME_DISPLAY_NAME } from './defaults.ts'
import { parseSse } from './sse.ts'
import { httpFailure } from './transport.ts'
import type { TaijiAdapterOptions, TaijiConnectionOptions } from './types.ts'

/** The documented sentinel the runtime writes after its frame stream. */
const DONE = '[DONE]'

/** Strip trailing slashes so one configured spelling yields one request path. */
function root(baseURL: string): string {
  return baseURL.replace(/\/+$/u, '')
}

/**
 * Taiji local-runtime provider. The runtime answers a request with one complete
 * text (`type: "final"`), so this adapter emits one text block and a terminal
 * `stop`; every other frame shape, and every transport failure, becomes the
 * provider-neutral failure the Trajectory records.
 */
export class TaijiAdapter extends LlmAdapter {
  constructor(private readonly dependencies: TaijiAdapterOptions) {
    super()
  }

  override providerInfo(provider: string) {
    return { id: provider, name: RUNTIME_DISPLAY_NAME }
  }

  override providerRetryPolicy(_provider: string) {
    return this.dependencies.options().retryPolicy
  }

  override listModels(provider: string): Promise<LlmModelInfo[]> {
    return Promise.resolve(this.dependencies.options().models.map(model => ({
      provider,
      id: model.id,
      name: model.name ?? model.id,
      ...model.description === undefined ? {} : { description: model.description },
      inputModalities: ['text' as const],
    })))
  }

  override resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    // An unlisted model id stays routable: the endpoint takes no model id at
    // all, so the catalog is a selector list and never a request validator.
    const configured = this.dependencies.options().models.find(entry => entry.id === model)
    return Promise.resolve({
      provider,
      id: model,
      name: configured?.name ?? model,
      ...configured?.description === undefined ? {} : { description: configured.description },
      ...configured?.contextWindow === undefined ? {} : { context: { contextWindow: configured.contextWindow } },
      inputModalities: ['text' as const],
    })
  }

  stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    return this.generate(options)
  }

  /** Emit the finished answer as one text block, then the terminal `stop`. */
  private async * generate(options: GenerateOptions): AsyncGenerator<StreamChunk> {
    const connection = this.dependencies.options()
    let answer: string
    try {
      answer = await this.answer(options, connection)
    } catch (error: unknown) {
      if (error instanceof LlmError) throw error
      if (options.signal?.aborted) throw new LlmError('Taiji runtime request aborted', 'ABORTED', { cause: error })
      throw new LlmError('Taiji runtime transport failed', 'TRANSPORT', { cause: error })
    }
    const block: ContentBlock = { type: 'text', text: answer }
    yield { type: 'block-start', index: 0, blockType: 'text' }
    yield { type: 'text-delta', index: 0, text: answer }
    yield { type: 'block-end', index: 0, block }
    yield { type: 'finish', reason: { kind: 'stop' } }
  }

  /** One HTTP attempt: POST the chat body, then read its whole answer. */
  private async answer(options: GenerateOptions, connection: TaijiConnectionOptions): Promise<string> {
    const response = await fetch(`${root(connection.baseURL)}${CHAT_STREAM_PATH}`, {
      method: 'POST',
      redirect: 'error',
      ...options.signal === undefined ? {} : { signal: options.signal },
      headers: {
        ...attributionHeaders(),
        'accept': 'text/event-stream',
        'content-type': 'application/json',
      },
      body: JSON.stringify(buildChatRequest(options)),
    })
    if (!response.ok) throw await httpFailure(response)
    /* v8 ignore next -- a streaming HTTP response always carries a body; the guard keeps the reader honest */
    if (response.body === null) throw new LlmError('Taiji runtime returned no response body', 'STREAM_CLOSED')
    return await readAnswer(response.body)
  }
}

/** Read the runtime's whole answer from its frame stream. */
async function readAnswer(body: ReadableStream<Uint8Array>): Promise<string> {
  for await (const payload of parseSse(body)) {
    if (payload === DONE) break
    return readFrame(payload)
  }
  throw new LlmError('Taiji runtime stream closed before a final answer', 'STREAM_CLOSED')
}

/**
 * Interpret one frame as the reply, or fail with the frame's own reason.
 * @param payload - one frame's joined `data` payload.
 * @returns the final answer text.
 */
function readFrame(payload: string): string {
  let raw: unknown
  try {
    raw = JSON.parse(payload)
  } catch (_invalidSseJson) {
    throw new LlmError('Taiji runtime SSE frame is not JSON', 'MALFORMED_RESPONSE')
  }
  // An in-band generation failure arrives as a bare JSON string
  // (`data: "生成出错: ..."`); its own wording is the recorded failure.
  if (typeof raw === 'string') throw new LlmError(raw, 'SERVER')
  if (typeof raw !== 'object' || raw === null) {
    throw new LlmError('Taiji runtime SSE frame has no answer shape', 'MALFORMED_RESPONSE')
  }
  const frame = raw as { type?: unknown; data?: unknown }
  if (frame.type !== 'final') {
    throw new LlmError('Taiji runtime answered with a frame this route does not serve', 'MALFORMED_RESPONSE')
  }
  const data = frame.data
  const answer = typeof data === 'object' && data !== null ? (data as { answer?: unknown }).answer : undefined
  if (typeof answer !== 'string') {
    throw new LlmError('Taiji runtime final frame carries no answer text', 'MALFORMED_RESPONSE')
  }
  if (answer.length === 0) {
    throw new LlmError('Taiji runtime answered with an empty text', EMPTY_RESPONSE_CODE)
  }
  return answer
}