/** A loopback Taiji runtime stand-in: scripted status, control, and training frames. */
import { createServer } from 'node:http'
import type { IncomingMessage, Server, ServerResponse } from 'node:http'
import { once } from 'node:events'

/** One request the stand-in received. */
export interface RecordedRequest {
  readonly method: string
  readonly path: string
  readonly body: unknown
}

/** Scripted behavior for the next training stream. */
export type TrainingScript =
  /** Write these frames (already JSON-encoded payloads); `hold` keeps the stream open afterwards. */
  | { kind: 'frames'; frames: string[]; hold?: boolean }
  /** Answer with a bare HTTP failure instead of opening a stream. */
  | { kind: 'http-error'; status: number; body: string }

export interface MockLifeRuntime {
  url: string
  /** Every request received, in order. */
  requests: RecordedRequest[]
  /** Payload `GET /api/runtime/status` answers with. */
  runtimeStatus: Record<string, unknown>
  /** Reply `GET /api/life/status` gives; 404 models a runtime with Legacy disabled. */
  legacyReply: { status: number; body?: unknown }
  /** Reply `GET /api/rag/status` gives; 404 models a runtime without the knowledge surface. */
  knowledgeReply: { status: number; body?: unknown }
  /** Rows `GET /api/train/checkpoints` answers with. */
  checkpoints: unknown[]
  /** Script consumed by the next `POST /api/train/native`. */
  training: TrainingScript
  /** Script consumed by the next `POST /api/train/resume_checkpoint`. */
  resume: TrainingScript
  /** Reply one control verb gives, keyed by path; absent means `{status: 200}`. */
  controlReplies: Map<string, { status: number; body?: unknown }>
  /** Close the stream of a held training run. */
  closeTraining(): Promise<void>
  close(): Promise<void>
}

const runtimes: Server[] = []
const heldStreams = new Set<ServerResponse>()

/** Close every runtime opened since the last call; run from each spec's afterEach. */
export async function closeMockRuntimes(): Promise<void> {
  for (const response of heldStreams) response.end()
  heldStreams.clear()
  await Promise.all(runtimes.splice(0).map(server => new Promise<void>((resolve) => {
    server.closeAllConnections()
    server.close(() => { resolve() })
  })))
}

/** One runtime `progress` frame as `api/training/resume.py` writes it. */
export function progressFrame(sample: Partial<Record<string, unknown>> = {}): string {
  return JSON.stringify({
    type: 'progress',
    fraction: 0.25,
    desc: 'training',
    step: 250,
    loss: 1.5,
    elapsed: 2.5,
    eta: 10,
    epoch: 1,
    total_epochs: 1,
    samples_per_sec: 100,
    total_steps: 1000,
    ...sample,
  })
}

/** One runtime `completed` frame. */
export function completedFrame(fields: Partial<Record<string, unknown>> = {}): string {
  return JSON.stringify({
    type: 'completed',
    message: 'training complete',
    desc: 'completed',
    step: 1000,
    checkpoint: 'seed_native.pt',
    ticks_added: 750,
    ...fields,
  })
}

/** One runtime `error` frame. */
export function errorFrame(message: string): string {
  return JSON.stringify({ type: 'error', message })
}

/** One runtime `warning` frame, as a corpus-drift notice on a resumed run arrives. */
export function warningFrame(message: string): string {
  return JSON.stringify({ type: 'warning', message })
}

/**
 * Start the stand-in on an ephemeral loopback port.
 * @returns the runtime handle; every scripted field is mutable between requests,
 *   so one instance can model a runtime that changes under the Host.
 */
export async function mockLifeRuntime(): Promise<MockLifeRuntime> {
  const requests: RecordedRequest[] = []
  let training: TrainingScript = { kind: 'frames', frames: [] }
  let resume: TrainingScript = { kind: 'frames', frames: [] }
  let held: ServerResponse | undefined
  const runtime: MockLifeRuntime = {
    url: '',
    requests,
    runtimeStatus: defaultStatus(),
    legacyReply: { status: 404 },
    knowledgeReply: { status: 404 },
    checkpoints: [],
    get training() { return training },
    set training(script: TrainingScript) { training = script },
    get resume() { return resume },
    set resume(script: TrainingScript) { resume = script },
    controlReplies: new Map(),
    async closeTraining() {
      held?.end()
      heldStreams.delete(held as ServerResponse)
      held = undefined
    },
    async close() {
      await new Promise<void>((resolve) => {
        server.closeAllConnections()
        server.close(() => { resolve() })
      })
    },
  }
  const server = createServer((request, response) => {
    void handle(request, response, runtime, requests, (stream) => { held = stream })
  })
  runtimes.push(server)
  server.listen(0, '127.0.0.1')
  await once(server, 'listening')
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('mock runtime has no port')
  runtime.url = `http://127.0.0.1:${String(address.port)}`
  return runtime
}

async function handle(
  request: IncomingMessage,
  response: ServerResponse,
  runtime: MockLifeRuntime,
  requests: RecordedRequest[],
  hold: (stream: ServerResponse) => void,
): Promise<void> {
  const path = new URL(request.url ?? '/', 'http://127.0.0.1').pathname
  const body = await readBody(request)
  requests.push({ method: request.method ?? 'GET', path, body })
  if (request.method === 'GET' && path === '/api/runtime/status') {
    await json(response, 200, runtime.runtimeStatus)
    return
  }
  if (request.method === 'GET' && path === '/api/life/status') {
    await json(response, runtime.legacyReply.status, runtime.legacyReply.body)
    return
  }
  if (request.method === 'GET' && path === '/api/rag/status') {
    await json(response, runtime.knowledgeReply.status, runtime.knowledgeReply.body)
    return
  }
  if (request.method === 'GET' && path === '/api/train/checkpoints') {
    await json(response, 200, { status: 'ok', checkpoints: runtime.checkpoints })
    return
  }
  if (request.method === 'POST' && path === '/api/train/native') {
    await stream(response, runtime.training, hold)
    return
  }
  if (request.method === 'POST' && path === '/api/train/resume_checkpoint') {
    await stream(response, runtime.resume, hold)
    return
  }
  const scripted = runtime.controlReplies.get(path)
  // The real runtime registers its Legacy surface only while `SEED_ENABLE_LEGACY`
  // is on, so an unscripted Legacy path answers 404 exactly as it does there.
  if (scripted === undefined && path.startsWith('/api/taiji/')) {
    await json(response, 404, undefined)
    return
  }
  await json(response, scripted?.status ?? 200, scripted?.body ?? { status: 'ok', message: `${path} accepted` })
}

async function json(response: ServerResponse, status: number, body: unknown): Promise<void> {
  if (body === undefined) {
    response.writeHead(status, { 'content-type': 'application/json' })
    response.end()
    return
  }
  const text = JSON.stringify(body)
  response.writeHead(status, { 'content-type': 'application/json' })
  response.end(text)
}

async function stream(response: ServerResponse, script: TrainingScript, hold: (stream: ServerResponse) => void): Promise<void> {
  if (script.kind === 'http-error') {
    response.writeHead(script.status, { 'content-type': 'application/json' })
    response.end(script.body)
    return
  }
  response.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' })
  for (const frame of script.frames) response.write(`data: ${frame}\n\n`)
  if (script.hold === true) {
    hold(response)
    heldStreams.add(response)
    return
  }
  response.write('data: [DONE]\n\n')
  response.end()
}

async function readBody(request: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = []
  for await (const chunk of request) chunks.push(chunk as Buffer)
  const text = Buffer.concat(chunks).toString('utf8')
  if (text.trim() === '') return undefined
  try {
    return JSON.parse(text)
  } catch {
    return text
  }
}

/** A runtime payload that answers as the Seed native organ while Legacy stays gated. */
function defaultStatus(): Record<string, unknown> {
  return {
    status: 'ok',
    timestamp: 1_760_000_000,
    health: {
      state: 'ok',
      message: '',
      model_loaded: true,
      model_name: 'taiji-native',
      is_taiji: true,
      is_seed: true,
      startup_complete: true,
      startup_error: '',
      language_provider: {},
    },
    memory: { status: 'ok', total_gb: 32, available_gb: 12, used_pct: 62.5 },
    auth: { enabled: false, authenticated: true, token_valid: false, username: '', has_password: false },
    life: {
      status: 'seed',
      is_running: true,
      needs: { curiosity: 42.5, fatigue: 10, stress: 1.5 },
      drives: { exploration: 40, replay: 10, rest: 0, play: 30 },
      mode: 'wake',
      tick: 41,
    },
    tools: { status: 'ok', tools: [], count: 0, error: '', snapshot_id: '', revision: 0, source: '', owner: '', observed_at: 0 },
    training: { is_training: false, publishing: false, pause_requested: false, stop_requested: false },
  }
}
