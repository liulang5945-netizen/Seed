/** Frame decoding: only a blank-line-terminated `data` payload becomes an event. */
import { describe, expect, it } from 'vitest'
import { parseSse } from '../src/sse.ts'

/** One body delivered as the given chunks, in order. */
function bodyOf(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  })
}

async function payloads(stream: ReadableStream<Uint8Array>): Promise<string[]> {
  const found: string[] = []
  for await (const payload of parseSse(stream)) found.push(payload)
  return found
}

describe('Taiji SSE decoding', () => {
  it('reads one payload per blank-line-terminated frame', async () => {
    expect(await payloads(bodyOf(['data: {"answer":"hi"}\n\ndata: [DONE]\n\n']))).toEqual(['{"answer":"hi"}', '[DONE]'])
  })

  it('joins multiple data lines as the SSE grammar prescribes', async () => {
    expect(await payloads(bodyOf(['data: one\ndata: two\n\n']))).toEqual(['one\ntwo'])
  })

  it('accepts a data line without the optional space', async () => {
    expect(await payloads(bodyOf(['data:{"answer":"hi"}\n\n']))).toEqual(['{"answer":"hi"}'])
  })

  it('ignores frames that carry no data, including comments and other fields', async () => {
    expect(await payloads(bodyOf([': keep-alive\n\nevent: ping\ndata: x\n\nfoo\n\n']))).toEqual(['x'])
  })

  it('does not treat an unterminated tail as an event', async () => {
    expect(await payloads(bodyOf(['data: {"answer":"hi"}\n\ndata: unfinished']))).toEqual(['{"answer":"hi"}'])
  })

  it('reassembles a CRLF pair split across two chunks', async () => {
    expect(await payloads(bodyOf(['data: one\r', '\n\r\n']))).toEqual(['one'])
  })

  it('ends the underlying request when the consumer stops early', async () => {
    let cancelled = false
    const encoder = new TextEncoder()
    const stream = new ReadableStream<Uint8Array>({
      start(controller) { controller.enqueue(encoder.encode('data: first\n\ndata: second\n\n')) },
      cancel() { cancelled = true },
    })
    const iterator = parseSse(stream)[Symbol.asyncIterator]()

    expect((await iterator.next()).value).toBe('first')
    await iterator.return?.(undefined)

    expect(cancelled).toBe(true)
  })
})