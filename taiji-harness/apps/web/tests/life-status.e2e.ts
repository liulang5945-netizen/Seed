import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium, type Browser, type Page } from 'playwright'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { closeMockRuntimes, mockLifeRuntime, type MockLifeRuntime } from '../../../packages/api/life-controller/tests/mock-runtime.ts'
import { captureStableAria, compareOrRefreshGolden, launchWebScaffold, webSnapshotMode, type WebScaffold } from './scaffold.ts'

const mode = webSnapshotMode()
const expected = fileURLToPath(new URL('./expected/life-status/', import.meta.url))

describe.skipIf(mode === 'record')('Life state through the shipped web composition', () => {
  let runtime: MockLifeRuntime
  let scaffold: WebScaffold
  let browser: Browser
  let page: Page
  let temporary: string

  beforeAll(async () => {
    runtime = await mockLifeRuntime()
    runtime.controlReplies.set('/api/runtime/status', { status: 500 })
    temporary = await mkdtemp(join(tmpdir(), 'seed-life-status-'))
    const overlay = join(temporary, 'life.overlay.yml')
    await writeFile(overlay, JSON.stringify([
      { id: 'life-controller', config: { baseURL: runtime.url, pollIntervalMs: 60_000, activePollIntervalMs: 60_000, requestTimeoutMs: 1000, maxCheckpoints: 5 } },
      { id: 'llm-taiji', disabled: true },
    ]))
    scaffold = await launchWebScaffold({ extraOverlayPath: overlay })
    browser = await chromium.launch()
    page = await browser.newPage({ viewport: { width: 1280, height: 900 }, locale: 'zh-CN' })
    await page.goto(scaffold.authenticatedUrl, { waitUntil: 'load' })
    const defer = page.getByRole('button', { name: '稍后再说', exact: true })
    if (await defer.isVisible()) await defer.click()
    await page.getByRole('button', { name: '生命', exact: true }).click()
    await page.locator('#life-source').waitFor()
  }, 120_000)

  afterAll(async () => {
    await browser?.close()
    await scaffold?.close()
    await closeMockRuntimes()
    if (temporary !== undefined) await rm(temporary, { recursive: true, force: true })
  })

  it('preserves unknown state, refreshes immediately, and respects training locks', async () => {
    const training = page.locator('#life-training')
    await training.locator('summary').click()
    await training.getByText('训练状态尚未获取，无法判断是否空闲。', { exact: true }).waitFor()
    expect(await training.getByRole('button', { name: '启动训练', exact: true }).isEnabled()).toBe(false)
    expect(await training.getByText('空闲', { exact: true }).count()).toBe(0)
    await compareOrRefreshGolden(join(expected, 'unknown.expected.md'), await captureStableAria(page, '#life-training', scaffold.workspaceCwd), mode)

    runtime.controlReplies.delete('/api/runtime/status')
    await page.locator('#life-source').getByRole('button', { name: '重试', exact: true }).click()
    await training.getByText('空闲', { exact: true }).waitFor()
    expect(await training.getByRole('button', { name: '启动训练', exact: true }).isEnabled()).toBe(true)
    await compareOrRefreshGolden(join(expected, 'idle.expected.md'), await captureStableAria(page, '#life-training', scaffold.workspaceCwd), mode)

    runtime.controlReplies.set('/api/train/checkpoints', { status: 500 })
    await page.locator('#life-source').getByRole('button', { name: '重试', exact: true }).click()
    const checkpoints = page.locator('#life-checkpoints')
    await checkpoints.locator('summary').getByText('尚未获取', { exact: true }).waitFor()
    await checkpoints.locator('summary').click()
    await checkpoints.getByText('检查点列表尚未获取，无法判断是否存在检查点。', { exact: true }).waitFor()
    expect(await checkpoints.getByText('还没有检查点。', { exact: true }).count()).toBe(0)

    runtime.controlReplies.delete('/api/train/checkpoints')
    runtime.checkpoints = [{ filename: 'candidate.pt', step: 20, bytes: 1024 }]
    runtime.runtimeStatus = {
      ...runtime.runtimeStatus,
      training: { is_training: true, pause_requested: false, stop_requested: false, publishing: false },
    }
    await page.locator('#life-source').getByRole('button', { name: '重试', exact: true }).click()
    await training.getByText('训练中', { exact: true }).waitFor()
    expect(await training.getByRole('button', { name: '启动训练', exact: true }).isEnabled()).toBe(false)
    expect(await checkpoints.getByRole('button', { name: '从这里续训 candidate.pt', exact: true }).isEnabled()).toBe(false)
    expect(await checkpoints.getByRole('button', { name: '激活 candidate.pt', exact: true }).isEnabled()).toBe(false)
    expect(runtime.requests.every(request => request.method === 'GET')).toBe(true)
  })
})
