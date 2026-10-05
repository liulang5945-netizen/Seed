/** Resource cleanup: dataset and checkpoint deletes, and the knowledge files. */
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

/** Every request the stand-in received for one exact path. */
function requestsFor(runtime: MockLifeRuntime, path: string): { method: string; path: string; body: unknown }[] {
  return runtime.requests.filter(request => request.path === path)
}

describe('LifeController dataset cleanup', () => {
  it('deletes one dataset through the roster path it names', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/train/file/consolidated/night-1.jsonl', { status: 200, body: { status: 'success' } })

    const value = await controller.deleteDataset({ path: 'consolidated/night-1.jsonl' }, new AbortController().signal)

    expect(value.message).toBe('')
    const [request] = requestsFor(runtime, '/api/train/file/consolidated/night-1.jsonl')
    expect(request?.method).toBe('DELETE')
  })

  it('percent-encodes each dataset path segment on its own', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/train/file/consolidated/night%201.jsonl', { status: 200, body: { status: 'success' } })

    await controller.deleteDataset({ path: 'consolidated/night 1.jsonl' }, new AbortController().signal)

    expect(requestsFor(runtime, '/api/train/file/consolidated/night%201.jsonl')).toHaveLength(1)
  })

  it('refuses paths that are not roster paths, without calling the runtime', async () => {
    const { controller, runtime } = await harness()
    const signal = new AbortController().signal

    for (const path of ['C:\\tmp\\notes.jsonl', '../../evil.jsonl', '/etc/hosts.jsonl', 'notes', 'weights.exe']) {
      await expect(controller.deleteDataset({ path }, signal)).rejects.toMatchObject({
        code: 'life/bad-request',
        details: { field: 'path' },
      })
    }
    expect(runtime.requests.some(request => request.method === 'DELETE')).toBe(false)
  })

  it('surfaces a 2xx answer whose own body refuses the delete', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/train/file/gone.jsonl', { status: 200, body: { status: 'error', message: '文件不存在' } })

    await expect(controller.deleteDataset({ path: 'gone.jsonl' }, new AbortController().signal))
      .rejects.toMatchObject({ code: 'life/runtime-error', details: { status: 200, detail: '文件不存在' } })
  })
})

describe('LifeController checkpoint cleanup', () => {
  it('deletes one checkpoint and carries the runtime conflict for a protected one', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/train/checkpoint/old_run.pt', {
      status: 200,
      body: { status: 'success', message: '检查点 old_run.pt 已删除' },
    })
    runtime.controlReplies.set('/api/train/checkpoint/live.pt', {
      status: 409,
      body: { detail: '检查点 live.pt 正在使用（活跃模型），不能删除' },
    })

    const value = await controller.deleteCheckpoint({ filename: 'old_run.pt' }, new AbortController().signal)
    expect(value.message).toBe('检查点 old_run.pt 已删除')
    expect(requestsFor(runtime, '/api/train/checkpoint/old_run.pt')[0]?.method).toBe('DELETE')

    await expect(controller.deleteCheckpoint({ filename: 'live.pt' }, new AbortController().signal))
      .rejects.toMatchObject({ code: 'life/conflict', details: { reason: '检查点 live.pt 正在使用（活跃模型），不能删除' } })
  })

  it('refuses names that leave the checkpoint directory, hiding temp files too', async () => {
    const { controller, runtime } = await harness()
    const signal = new AbortController().signal

    for (const filename of ['..', 'sub/evil.pt', 'sub\\evil.pt', 'C:evil.pt', 'notes.txt', '.hidden.pt']) {
      await expect(controller.deleteCheckpoint({ filename }, signal)).rejects.toMatchObject({
        code: 'life/bad-request',
        details: { field: 'filename' },
      })
    }
    expect(runtime.requests.some(request => request.method === 'DELETE')).toBe(false)
  })
})

describe('LifeController knowledge files', () => {
  it('carries the mounted files with their index state', async () => {
    const { controller, runtime } = await harness()
    runtime.knowledgeReply = { status: 200, body: { status: 'ok', doc_count: 2, chunk_count: 9, has_embeddings: true, embed_dim: 8 } }
    runtime.knowledgeFilesReply = {
      status: 200,
      body: {
        files: [
          { name: 'guide.md', size: 2048, mtime: 1_760_000_000, status: 'indexed' },
          { name: 'loose.txt', status: 'pending' },
        ],
      },
    }

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.knowledge?.files).toEqual([
      { name: 'guide.md', sizeBytes: 2048, status: 'indexed' },
      { name: 'loose.txt', status: 'pending' },
    ])
    expect(snapshot.unavailable).toEqual([])
  })

  it('never reads the file list while the knowledge surface is disabled', async () => {
    const { controller, runtime } = await harness()
    // The stand-in answers `GET /api/rag/status` with 404 by default.

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.knowledge).toBeUndefined()
    expect(snapshot.availability.knowledge).toBe('disabled')
    expect(runtime.requests.some(request => request.path === '/api/rag/files')).toBe(false)
  })

  it('records a failed file list as one unavailable line, not a broken snapshot', async () => {
    const { controller, runtime } = await harness()
    runtime.knowledgeReply = { status: 200, body: { status: 'ok', doc_count: 1, chunk_count: 2, has_embeddings: false, embed_dim: 0 } }
    runtime.knowledgeFilesReply = { status: 500 }

    const { snapshot } = await controller.snapshot(new AbortController().signal)

    expect(snapshot.knowledge?.files).toBeUndefined()
    expect(snapshot.knowledge?.docCount).toBe(1)
    expect(snapshot.availability.knowledge).toBe('ok')
    expect(snapshot.unavailable).toContain('knowledge-files: HTTP 500')
  })

  it('uploads one document as multipart bytes under its basename', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/rag/upload', { status: 200, body: { status: 'success', message: '文件 guide.md 已上传，正在后台向量化建库，请稍后查看！' } })
    const bytes = Buffer.from('# 指南\n', 'utf8')

    const value = await controller.uploadKnowledge(
      { name: 'C:\\docs\\guide.md', data: bytes.toString('base64') },
      new AbortController().signal,
    )

    expect(value.message).toContain('guide.md')
    const [body] = requestsFor(runtime, '/api/rag/upload').map(request => String(request.body))
    expect(body).toContain('name="file"; filename="guide.md"')
    expect(body).toContain('# 指南')
  })

  it('refuses a name the filesystem cannot hold and a payload past the budget', async () => {
    const { controller, runtime } = await harness()
    const signal = new AbortController().signal

    await expect(controller.uploadKnowledge({ name: 'bad:name.md', data: 'AA==' }, signal))
      .rejects.toMatchObject({ code: 'life/bad-request', details: { field: 'name' } })
    const over = 'A'.repeat(Math.ceil(UPLOAD_MAX_BYTES / 3) * 4 + 4)
    await expect(controller.uploadKnowledge({ name: 'big.pdf', data: over }, signal))
      .rejects.toMatchObject({ code: 'life/bad-request', details: { field: 'data' } })
    expect(runtime.requests.some(request => request.path === '/api/rag/upload')).toBe(false)
  })

  it('deletes one document by name', async () => {
    const { controller, runtime } = await harness()
    runtime.controlReplies.set('/api/rag/file/loose.txt', { status: 200, body: { status: 'success' } })

    await controller.deleteKnowledge({ name: 'loose.txt' }, new AbortController().signal)

    expect(requestsFor(runtime, '/api/rag/file/loose.txt')[0]?.method).toBe('DELETE')
  })

  it('refuses a delete name that leaves the document directory', async () => {
    const { controller, runtime } = await harness()
    const signal = new AbortController().signal

    // A delete acts on exactly the file it names: an upload may reduce a path
    // to its basename, a delete never does.
    for (const name of ['../escape.md', 'sub/plots.md', 'C:notes.md', 'bad:name.md']) {
      await expect(controller.deleteKnowledge({ name }, signal)).rejects.toMatchObject({
        code: 'life/bad-request',
        details: { field: 'name' },
      })
    }
    expect(runtime.requests.some(request => request.method === 'DELETE')).toBe(false)
  })
})