/** A loopback Taiji runtime stand-in: scripted chat frames plus a readiness verdict. */
import { createServer } from 'node:http'
import type { IncomingMessage, Server, ServerResponse } from 'node:http'
import { once } from 'node:events'
import { HEALTH_PATH } from '../src/defaults.ts'

/** One scripted behavior for the next chat request the runtime receives. */
export type Behavior =
  /** Write these frames (already JSON-encoded payloads) and close the stream. */
  | { kind: 'frames'; frames: string[] }
  /** Answer the chat request with a bare HTTP error. */
  | { kind: 'http-error'; status: number; body: string }
  /** Send the stream head and then hold the connection open. */
  | { kind: 'hold' }

export interface MockRuntime {
  url: string
  /** Chat requests received, in order. */
  requests: Array<{ path: string; headers: IncomingMessage['headers']; body: Record<string, unknown> }>
  /** Paths of readiness probes received, in order. */
  health: string[]
  /** Readiness `status` the next probe answers with. */
  healthStatus: string
  /** Raw readiness body override; when set it replaces the generated payload. */
  healthBody?: string
  /** HTTP status the readiness endpoint answers with. */
  healthHttpStatus: number
  script: Behavior[]
  close(): Promise<void>
}

/** The runtime's `final` frame, byte-for-byte as `api/routes_chat.py` writes it. */
export function finalFrame(answer: string): string {
  return JSON.stringify({
    type: 'final',
    data: { answer, step: 1, readable: true, language_backend: 'native', runtime: 'seed', workbench: null },
  })
}

/** The runtime's in-band failure frame: a bare JSON string, never an object. */
export function failureFrame(detail: string): string {
  return JSON.stringify(`生成出错: ${detail}`)
}

/** The sentinel the runtime writes after its frames. */
export const DONE = '[DONE]'

const runtimes: Server[] = []

/** Close every runtime opened since the last call; run from each spec's afterEach. */
export async function closeMockRuntimes(): Promise<void> {
  await Promise.all(runtimes.splice(0).map(server => new Promise<void>((resolve) => {
    server.closeAllConnections()
    server.close(() => { resolve() })
  })))
}

/**
 * Start the stand-in on an ephemeral loopback port.
 * @returns the runtime handle; its `healthStatus` and `script` are mutable
 *   between requests so one instance can model a changing runtime.
 */
export async function mockRuntime(): Promise<MockRuntime> {
  const requests: MockRuntime['requests'] = []
  const health: string[] = []
  const runtime: MockRuntime = {
    url: '',
    requests,
    health,
    healthStatus: 'ok',
    healthHttpStatus: 200,
    script: [],
    close: () => Promise.resolve(),
  }
  const server = createServer((request: IncomingMessage, response: ServerResponse) => {
    void handle(request, response).catch((error: unknown) => { response.destroy(error as Error) })
  })
  async function handle(request: IncomingMessage, response: ServerResponse): Promise<void> {
    const parts: Buffer[] = []
    for await (const part of request as AsyncIterable<Buffer>) parts.push(part)
    const url = new URL(request.url ?? '/', 'http://localhost')
    if (url.pathname === HEALTH_PATH) {
      health.push(url.pathname)
      response.writeHead(runtime.healthHttpStatus, { 'content-type': 'application/json' })
      response.end(runtime.healthBody ?? JSON.stringify({ status: runtime.healthStatus, service: 'Taiji API', seed_active: true }))
      return
    }
    requests.push({ path: url.pathname, headers: request.headers, body: asBody(Buffer.concat(parts).toString('utf8')) })
    const behavior = runtime.script.shift()
    if (behavior === undefined) {
      response.writeHead(500, { 'content-type': 'text/plain' }).end('mock script exhausted')
      return
    }
    if (behavior.kind === 'http-error') {
      response.writeHead(behavior.status, { 'content-type': 'application/json' }).end(behavior.body)
      return
    }
    response.writeHead(200, { 'content-type': 'text/event-stream' })
    if (behavior.kind === 'hold') return
    for (const frame of behavior.frames) response.write(`data: ${frame}\n\n`)
    response.end()
  }
  runtimes.push(server)
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('missing loopback port')
  runtime.url = `http://127.0.0.1:${address.port}`
  runtime.close = () => new Promise<void>((resolve) => {
    server.closeAllConnections()
    server.close(() => { resolve() })
  })
  return runtime
}

/**
 * A runtime URL on a loopback port that is closed by the time it is returned:
 * the connection-refused case, which no local listener can file.
 */
export async function closedRuntimeUrl(): Promise<string> {
  const server = createServer()
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('missing loopback port')
  const port = address.port
  await new Promise<void>((resolve) => { server.close(() => { resolve() }) })
  return `http://127.0.0.1:${port}`
}

/** Read one recorded request body as an object, failing loud on anything else. */
function asBody(text: string): Record<string, unknown> {
  const raw: unknown = JSON.parse(text)
  if (typeof raw !== 'object' || raw === null) throw new Error(`mock runtime received a non-object body: ${text}`)
  return raw as Record<string, unknown>
}
