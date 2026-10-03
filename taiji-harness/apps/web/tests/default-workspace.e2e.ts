/** First-use startup prepares a durable Workspace through the shipped Web composition. */
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium, type Page } from 'playwright'
import { describe, expect, it, onTestFinished } from 'vitest'
import {
  assertFinalWorkspaceSnapshot, captureStableAria, compareOrRefreshGolden, fixtureUserPrompts, launchWebScaffold,
  selectedSessionFixture, watchConsole, webSnapshotMode,
} from './scaffold.ts'
import { newEnglishPage, saveFailureShot } from './support.ts'

const FIXTURE = fileURLToPath(new URL('../../../snapshots/web/fresh-round-trip/session.v3.jsonl', import.meta.url))
const EXPECTED = fileURLToPath(new URL('../../../snapshots/web/default-workspace/ui.expected.md', import.meta.url))
const FAILURE_EXPECTED = fileURLToPath(new URL('./expected/default-workspace/failure.expected.md', import.meta.url))
const MODE = webSnapshotMode()

/**
 * Select the provisioned Workspace through the Hero's chip, which is what unlocks its
 * composer: startup prepares the Workspace but never opens a Session, so the Hero can
 * come up with nothing selected. A no-op once a Session is current.
 * @param page - the loaded Web page.
 */
async function pickDefaultWorkspace(page: Page): Promise<void> {
  const chip = page.getByRole('button', { name: /Choose workspace|选择工作区/u })
  // The chip only renders once the client has its Workspace list, which arrives
  // after boot's provisioning round-trip.
  if (!await chip.waitFor({ state: 'visible', timeout: 5_000 }).then(() => true, () => false)) return
  await chip.click()
  await page.getByRole('menuitem', { name: 'Default workspace' }).click()
}

describe.skipIf(MODE === 'record')('web e2e: default Workspace', () => {
  it('provisions the default Workspace, starts the first Session from the Hero send, and keeps it across reload', async () => {
    const fixture = await selectedSessionFixture(FIXTURE)
    const scaffold = await launchWebScaffold({
      firstUse: true,
      replayFixture: fixture,
      // The shared recording remains read-only; replay compares the complete persisted Session.
      compareReplaySession: MODE === 'replay',
      paceMs: 5,
    })
    onTestFinished(() => scaffold.close())
    const browser = await chromium.launch()
    try {
      const page = await newEnglishPage(browser)
      const tripwire = watchConsole(page)
      try {
        await page.goto(scaffold.authenticatedUrl)
        // Boot provisions the default Workspace asynchronously, after the page's
        // own load settles, so the Host-side count is polled rather than read.
        await expect.poll(() => scaffold.ctx.workspaceRegistry.list().length, { timeout: 20_000 }).toBe(1)
        // Boot provisions the Workspace but never opens a Session, so the Hero comes
        // up with nothing selected and its composer locked.
        expect(scaffold.ctx.sessions.list()).toEqual([])
        await pickDefaultWorkspace(page)
        const input = page.locator('[data-composer-input][contenteditable="true"]').first()
        await input.waitFor()
        await page.reload()
        expect(scaffold.ctx.workspaceRegistry.list()).toHaveLength(1)
        // Selecting the Workspace is what starts its blank Session, so the restored
        // Session survives the reload and is still blank before the Hero sends.
        const restored = scaffold.ctx.sessions.list()
        expect(restored).toHaveLength(1)
        expect(restored[0]?.snapshotEvents().some(event => event.type === 'user/message')).toBe(false)
        await pickDefaultWorkspace(page)
        await input.waitFor()
        const prompt = fixtureUserPrompts(await readFile(fixture, 'utf8'))[0]!
        await input.fill(prompt)
        const settled = scaffold.whenTurnSettled()
        await input.press('Enter')
        const sessionId = await settled
        const workspace = scaffold.ctx.workspaceRegistry.list()[0]!
        expect(workspace.title).toBe('Default workspace')
        expect(workspace.path).toBe(join(scaffold.workspaceCwd, 'Documents', 'taiji-harness', 'Default workspace'))
        expect((await stat(workspace.path)).isDirectory()).toBe(true)
        expect(workspace.sessionIds).toContain(sessionId)
        expect(scaffold.ctx.sessions.get(sessionId)?.header.cwd).toBe(workspace.path)
        await assertFinalWorkspaceSnapshot(fileURLToPath(new URL('../../../snapshots/web/default-workspace', import.meta.url)), workspace.path)
        await page.getByText('DONE', { exact: true }).waitFor()
        await compareOrRefreshGolden(EXPECTED, await captureStableAria(page, '[class*="centerCol"]', scaffold.workspaceCwd), MODE)
        await page.reload()
        await page.getByText('DONE', { exact: true }).waitFor()
        expect(scaffold.ctx.sessions.list().map(session => session.id)).toEqual([sessionId])
        expect(tripwire.pageErrors).toEqual([])
        expect(tripwire.warnings).toEqual([])
      } catch (error) {
        await saveFailureShot(page, 'web-e2e-default-workspace')
        throw error
      }
    } finally {
      await browser.close()
    }
  })

  it('reports a startup directory conflict and opens the composed folder picker for recovery', async () => {
    const scaffold = await launchWebScaffold({ firstUse: true })
    onTestFinished(() => scaffold.close())
    const parent = join(scaffold.workspaceCwd, 'Documents', 'taiji-harness')
    await mkdir(parent, { recursive: true })
    await writeFile(join(parent, '默认工作区'), 'occupied')
    const chosen = join(scaffold.workspaceCwd, 'chosen')
    await mkdir(chosen)
    const browser = await chromium.launch()
    try {
      const page = await browser.newPage({ locale: 'zh-CN', timezoneId: 'Asia/Shanghai' })
      const tripwire = watchConsole(page)
      try {
        await page.goto(scaffold.authenticatedUrl)
        const input = page.locator('[data-composer-input][contenteditable="true"]').first()
        const notice = page.getByRole('alert').filter({ hasText: '无法创建默认工作区' })
        await notice.waitFor()
        expect(scaffold.ctx.workspaceRegistry.list()).toEqual([])
        expect(scaffold.ctx.sessions.list()).toEqual([])
        await compareOrRefreshGolden(FAILURE_EXPECTED, await notice.ariaSnapshot(), MODE)
        expect(await page.getByRole('dialog').count()).toBe(0)
        expect(await input.count()).toBe(0)
        await page.getByRole('button', { name: '选择工作区', exact: true }).click()
        const picker = page.getByRole('dialog', { name: '选择工作区目录' })
        await picker.waitFor()
        await picker.getByRole('button', { name: '编辑路径', exact: true }).click()
        await picker.getByRole('textbox', { name: '编辑路径' }).fill(chosen)
        await page.keyboard.press('Enter')
        await picker.getByRole('button', { name: '打开', exact: true }).click()
        await picker.waitFor({ state: 'hidden' })
        await expect.poll(() => scaffold.ctx.workspaceRegistry.list().length).toBe(1)
        await input.waitFor()
        expect(await input.textContent()).toBe('')
        expect(scaffold.ctx.workspaceRegistry.list()[0]?.path).toBe(chosen)
        expect(scaffold.ctx.sessions.list()).toHaveLength(1)
        expect(scaffold.ctx.sessions.list().every(session =>
          session.snapshotEvents().every(event => event.type !== 'user/message'))).toBe(true)
        expect(tripwire.pageErrors).toEqual([])
        expect(tripwire.warnings).toEqual([])
      } catch (error) {
        await saveFailureShot(page, 'web-e2e-default-workspace-failure')
        throw error
      }
    } finally {
      await browser.close()
    }
  })

  it('keeps an explicitly deleted default absent on startup', async () => {
    const scaffold = await launchWebScaffold()
    onTestFinished(() => scaffold.close())
    const browser = await chromium.launch()
    try {
      const page = await newEnglishPage(browser)
      const attempted = page.waitForResponse(response => response.url().includes('/api/workspace/initializeDefault'))
      await page.goto(scaffold.authenticatedUrl)
      expect((await attempted).ok()).toBe(true)
      await page.getByRole('button', { name: 'Choose workspace', exact: true }).waitFor()
      // Boot provisioned the default (asynchronously — poll, per the first
      // test); the user now deletes it explicitly.
      await expect.poll(() => scaffold.ctx.workspaceRegistry.list().length, { timeout: 20_000 }).toBe(1)
      const [defaultWorkspace] = scaffold.ctx.workspaceRegistry.list()
      expect(defaultWorkspace).toBeDefined()
      await expect(scaffold.ctx.workspaceRegistry.delete(defaultWorkspace!.id)).resolves.toBe(true)
      expect(scaffold.ctx.workspaceRegistry.list()).toEqual([])
      // The reload re-runs the client boot, whose initializeDefault request must
      // honor the recorded deletion intent instead of re-provisioning.
      const reattempted = page.waitForResponse(response => response.url().includes('/api/workspace/initializeDefault'))
      await page.reload({ waitUntil: 'load' })
      expect((await reattempted).ok()).toBe(true)
      await page.getByRole('button', { name: 'Choose workspace', exact: true }).waitFor()
      expect(await page.getByRole('alert').filter({ hasText: 'Unable to create default workspace' }).count()).toBe(0)
      expect(await page.locator('[data-composer-input][contenteditable="true"]').count()).toBe(0)
      expect(scaffold.ctx.workspaceRegistry.list()).toEqual([])
      expect(scaffold.ctx.sessions.list()).toEqual([])
    } finally {
      await browser.close()
    }
  })
})
