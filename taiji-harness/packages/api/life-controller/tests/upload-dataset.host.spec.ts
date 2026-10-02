/** Dataset upload: the Host sanitizes one picked name and forwards multipart bytes. */
import { afterEach, describe, expect, it } from 'vitest'
import { Context } from '@taiji/cordis'
import LifeController, { UPLOAD_MAX_BYTES } from '../src/index.ts'
import { closeMockRuntimes, mockLifeRuntime, type MockLifeRuntime } from './mock-runtime.ts'

const roots: Context[] = []

afterEach(async () => {
  await Promise.all(roots.splice(0).map(ctx => ctx.fiber.dispose()))
  await closeMockRuntimes()
})

interface Harness {
  readonly controller: LifeController
  readonly runtime: MockLifeRuntime
}

/** Boot the controller over a loopback stand-in at the schema's floor cadence. */
async function harness(): Promise<Harness> {
  const runtime = await mockLifeRuntime()
  const ctx = new Context()
  roots.push(ctx)
  const dispose = (): void => {}
  ctx.provide('typert', {
    lookups: { configure: () => dispose },
    contexts: { configureHost: () => dispose },
  } as never)
  const controller = new LifeController(ctx, {
    baseURL: runtime.url,
    pollIntervalMs: 250,
    activePollIntervalMs: 250,
    requestTimeoutMs: 2_000,
    maxCheckpoints: 5,
  })
  return { controller, runtime }
}

/** Every upload request the stand-in received. */
function uploads(runtime: MockLifeRuntime): string[] {
  return runtime.requests
    .filter(request => request.path === '/api/train/upload_dataset')
    .map(request => String(request.body))
}

describe('LifeController dataset upload', () => {
  it('forwards one picked file as multipart bytes under its basename', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/train/upload_dataset', {
      status: 200,
      body: { status: 'success', path: 'data/zh.jsonl', message: '数据集 `zh.jsonl` 已成功上传并选中！' },
    })
    const bytes = Buffer.from('{"text":"你好"}\n', 'utf8')

    const value = await controller.uploadDataset(
      { name: 'C:\\Users\\operator\\zh.jsonl', data: bytes.toString('base64') },
      new AbortController().signal,
    )

    expect(value.message).toContain('zh.jsonl')
    const [body] = uploads(runtime)
    expect(body).toContain('name="file"; filename="zh.jsonl"')
    expect(body).toContain(bytes.toString('utf8'))
    expect(body).toContain('application/octet-stream')
  })

  it('refuses names the roster could never show, without calling the runtime', async () => {
    const { controller, runtime } = await harness()
    const signal = new AbortController().signal

    for (const name of ['weights.exe', '..', 'bad:name.jsonl', 'notes']) {
      await expect(controller.uploadDataset({ name, data: 'AA==' }, signal)).rejects.toMatchObject({
        code: 'life/bad-request',
        details: { field: 'name' },
      })
    }
    expect(runtime.requests.some(request => request.path === '/api/train/upload_dataset')).toBe(false)
  })

  it('refuses a payload past the size budget before any transfer', async () => {
    const { controller, runtime } = await harness()
    // One base64 character past the ceiling the panel pre-checks as well.
    const over = 'A'.repeat(Math.ceil(UPLOAD_MAX_BYTES / 3) * 4 + 4)

    await expect(controller.uploadDataset({ name: 'big.jsonl', data: over }, new AbortController().signal))
      .rejects.toMatchObject({ code: 'life/bad-request', details: { field: 'data' } })
    expect(runtime.requests.some(request => request.path === '/api/train/upload_dataset')).toBe(false)
  })

  it('carries a runtime refusal through as its own HTTP status', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/train/upload_dataset', {
      status: 500,
      body: { detail: 'disk full' },
    })

    await expect(controller.uploadDataset({ name: 'zh.jsonl', data: 'AA==' }, new AbortController().signal))
      .rejects.toMatchObject({ code: 'life/runtime-error', details: { status: 500 } })
  })
})
