// Keyless browser coverage for the goal bar over the shipped Web composition.
// The command creates a real projected goal in a real Host session. The
// goldens pin the active and disarmed strips, while the clear gesture proves
// the acknowledged tombstone leaves neither stale chrome nor a
// duplicate-mutation error.
import { fileURLToPath } from 'node:url'
import { join } from 'node:path'
import type { Browser, Page } from 'playwright'
import { chromium } from 'playwright'
import { afterAll, beforeAll, describe, expect, it, onTestFailed } from 'vitest'
import type {} from '@taiji/dsh-goal'
import {
  assertFixtureInventory, captureStableAria, compareOrRefreshGolden,
  launchWebScaffold, watchConsole, webSnapshotMode, type WebScaffold,
} from './scaffold.ts'
import { connectFreshWorkspace, newEnglishPage, saveFailureShot } from './support.ts'

const SNAPSHOT_DIR = fileURLToPath(new URL('./expected/goal-bar', import.meta.url))
const ACTIVE_EXPECTED = join(SNAPSHOT_DIR, 'active.expected.md')
const INACTIVE_EXPECTED = join(SNAPSHOT_DIR, 'inactive.expected.md')
const OVERLAY = fileURLToPath(new URL('./goal-bar.overlay.yml', import.meta.url))
const MODE = webSnapshotMode()

describe('web e2e: goal bar clear convergence', () => {
  let scaffold: WebScaffold
  let browser: Browser
  let page: Page
  let tripwire: ReturnType<typeof watchConsole>

  beforeAll(async () => {
    scaffold = await launchWebScaffold({ extraOverlayPath: OVERLAY })
    browser = await chromium.launch()
    page = await newEnglishPage(browser)
    tripwire = watchConsole(page)
    await page.goto(scaffold.authenticatedUrl, { waitUntil: 'load' })
    await page.waitForSelector('[class*="frame"]', { timeout: 30_000 })
    await connectFreshWorkspace(page, scaffold.workspaceCwd)
    // connectFreshWorkspace selects the blank session but leaves the surface on the
    // hero composer, which does not process slash commands (08 §6 ⑱). Open the
    // session row by name so the in-session composer receives the /goal command.
    // Under batch load the row can take longer than the click's own wait (81:
    // 30 s click timeout, file-level fail), so anchor and wait visibly first.
    const sessionRow = page.getByRole('treeitem', { name: /New Session|新会话/u }).first()
    await sessionRow.waitFor({ state: 'visible', timeout: 30_000 })
    await sessionRow.click()
  }, 120_000)

  afterAll(async () => {
    await browser?.close()
    await scaffold?.close()
  })

  it('renders one active goal and clears it without exposing a stale error', async () => {
    onTestFailed(() => saveFailureShot(page, 'web-e2e-goal-bar-clear'))
    const input = page.locator('[data-composer-input][data-placeholder="Describe what you want to build, / commands, @ files or sessions"]')
    await input.waitFor({ timeout: 10_000 })
    await input.fill('/goal guard rapid clear clicks')
    await input.press('Enter')

    const bar = page.locator('[data-goal-bar]')
    await bar.waitFor({ timeout: 10_000 })
    const pause = bar.getByRole('button', { name: 'Pause goal' })
    await expect.poll(() => pause.count(), {
      timeout: 10_000,
    }).toBe(1)
    await pause.hover()
    const pauseTooltip = page.getByRole('tooltip', { name: 'Pause goal', exact: true })
    await pauseTooltip.waitFor()
    const tooltipGeometry = await page.evaluate(() => {
      const element = document.querySelector<HTMLElement>('[role="tooltip"]')
      if (element === null) return null
      const tooltip = element.getBoundingClientRect()
      return {
        declaredLeft: Number.parseFloat(element.style.left),
        declaredTop: Number.parseFloat(element.style.top),
        tooltipCenter: tooltip.left + tooltip.width / 2,
        tooltipTop: tooltip.top,
      }
    })
    expect(tooltipGeometry).not.toBeNull()
    expect(Math.abs(tooltipGeometry!.tooltipCenter - tooltipGeometry!.declaredLeft)).toBeLessThan(2)
    expect(Math.abs(tooltipGeometry!.tooltipTop - tooltipGeometry!.declaredTop)).toBeLessThan(2)
    await page.mouse.move(0, 0)
    await pauseTooltip.waitFor({ state: 'hidden' })
    const snapshot = await captureStableAria(page, '[data-goal-bar]', scaffold.workspaceCwd)
    await compareOrRefreshGolden(ACTIVE_EXPECTED, snapshot, MODE)

    // The connected client auto-opens browsable sessions, and each open starts an
    // agent (08 §6 ⑱, github-ready-review), so the registry's global count is not 1.
    // Assert the meaningful invariant instead: exactly one agent owns an armed goal.
    const armed = scaffold.ctx.agents.list()
      .filter(agent => scaffold.ctx.goals.get(agent)?.activation === 'armed')
    expect(armed).toHaveLength(1)
    scaffold.ctx.goals.disarm(armed[0]!)
    await expect.poll(() => bar.getByRole('button', { name: 'Resume goal' }).count(), {
      timeout: 10_000,
    }).toBe(1)
    const inactive = await captureStableAria(page, '[data-goal-bar]', scaffold.workspaceCwd)
    await compareOrRefreshGolden(INACTIVE_EXPECTED, inactive, MODE)

    const clear = bar.getByRole('button', { name: 'Clear goal' })
    await clear.evaluate((button) => {
      const control = button as HTMLButtonElement
      control.click()
      control.click()
    })
    await expect.poll(() => page.locator('[data-goal-bar]').count(), { timeout: 10_000 }).toBe(0)
    expect(await page.getByText(/no current goal/iu).count()).toBe(0)
    expect(tripwire.pageErrors).toEqual([])
    expect(tripwire.warnings).toEqual([])
  }, 60_000)

  it.skipIf(MODE === 'record')('keeps the fixture inventory closed', async () => {
    await assertFixtureInventory(SNAPSHOT_DIR, ['active.expected.md', 'inactive.expected.md'])
  })
})
