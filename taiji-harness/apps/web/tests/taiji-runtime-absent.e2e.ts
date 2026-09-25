// Fork-side discriminating experiment (added 2026-09-26, see plans/active/roadmap/08 §6 ⑪):
// the four keyless onboarding lanes time out waiting for the first-run "add an API key"
// card while the local Taiji runtime answers on :8000. This lane pins the other side of
// that premise by pointing `llm-taiji` at a dead port through the scaffold's own overlay
// seam, so no shared runtime is stopped.
import { fileURLToPath } from 'node:url'
import type { Browser, Page } from 'playwright'
import { chromium } from 'playwright'
import { afterAll, beforeAll, expect, it } from 'vitest'
import { launchWebScaffold, type WebScaffold } from './scaffold.ts'

const OVERLAY = fileURLToPath(new URL('./taiji-runtime-absent.overlay.yml', import.meta.url))
const CREDENTIAL_STEP = '添加一个 API Key 开始使用'

let scaffold: WebScaffold
let browser: Browser
let page: Page

beforeAll(async () => {
  scaffold = await launchWebScaffold({ deepSeekMissingCredential: true, extraOverlayPath: OVERLAY })
  browser = await chromium.launch()
  page = await browser.newPage({ viewport: { width: 1440, height: 960 }, locale: 'zh-CN' })
  await page.goto(scaffold.authenticatedUrl, { waitUntil: 'load' })
  await page.waitForSelector('[class*="frame"]', { timeout: 30_000 })
}, 120_000)

afterAll(async () => {
  await browser?.close()
  await scaffold?.close()
})

it('withdraws the taiji-local route and asks for a key again when the runtime is unreachable', async () => {
  const ids = scaffold.ctx.llm.listProviders().map(provider => provider.id)
  expect(ids).not.toContain('taiji-local')
  expect(ids).toContain('deepseek-official')
  // The declaration must survive a dead port: only the readiness probe withdraws
  // the route. If this fails, the overlay replaced the row instead of probing it.
  expect(scaffold.ctx.llm.listConfigurableProviders().map(entry => entry.provider)).toContain('taiji-local')
  const card = page.getByRole('dialog', { name: CREDENTIAL_STEP })
  await card.waitFor({ timeout: 15_000 })
  expect(await card.getByLabel('API 密钥', { exact: true }).count()).toBeGreaterThan(0)
})
