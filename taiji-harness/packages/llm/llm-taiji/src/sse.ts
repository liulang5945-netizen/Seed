/**
 * SSE framing for the Taiji runtime's chat stream, decoded here instead of
 * through a parser dependency: the runtime emits one `data:` line per frame
 * terminated by a blank line, and the adapter must read frames that are not
 * objects (its in-band error frame is a bare JSON string), which a general
 * event-source parser would silently reshape.
 */

/**
 * Decode the `data` payload of every complete frame in one response body.
 *
 * A frame is complete only once its blank-line terminator has arrived, so an
 * unterminated tail is not an event — the same rule keeps a connection closed
 * mid-frame from fabricating one. Line endings are normalized on the whole
 * buffer after each read, so a `\r\n` pair split across two chunks still
 * yields the same frame.
 * @param body - response bytes as delivered.
 * @returns each frame's joined `data` payload, in arrival order.
 */
export async function* parseSse(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      const next = await reader.read()
      if (next.done) break
      buffer = (buffer + decoder.decode(next.value, { stream: true })).replaceAll('\r\n', '\n')
      let boundary = buffer.indexOf('\n\n')
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary)
        buffer = buffer.slice(boundary + 2)
        const payload = framePayload(frame)
        if (payload !== undefined) yield payload
        boundary = buffer.indexOf('\n\n')
      }
    }
  } finally {
    // The consumer may stop early (the runtime's final answer is followed by
    // its `[DONE]` sentinel, which nothing here needs); releasing the reader
    // that way must also end the connection instead of leaving it open.
    await reader.cancel()
  }
}

/**
 * Read one frame's `data` payload, or `undefined` for a frame that carries
 * none. Comment and other SSE fields are ignored, and multiple `data` lines
 * join with `\n` as the SSE grammar prescribes.
 */
function framePayload(frame: string): string | undefined {
  const values: string[] = []
  for (const line of frame.split('\n')) {
    if (!line.startsWith('data:')) continue
    const value = line.slice('data:'.length)
    values.push(value.startsWith(' ') ? value.slice(1) : value)
  }
  return values.length === 0 ? undefined : values.join('\n')
}
