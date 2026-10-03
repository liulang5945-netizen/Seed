import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { Browser, Page } from 'playwright'
import { chromium } from 'playwright'
import { afterAll, beforeAll, describe, expect, it, onTestFailed } from 'vitest'
import type { AgentHandle } from '@taiji/dsh-agent'
import { ToolCallId, createUserMessage } from '@taiji/dsh-llm'
import { SessionId } from '@taiji/dsh-session'
import type { Session } from '@taiji/dsh-session'
import type {} from '@taiji/dsh-agent-preset-registry'
import type {} from '@taiji/dsh-system-prompt'
import {
  assertFixtureInventory,
  captureStableAria,
  compareOrRefreshGolden,
  launchWebScaffold,
  watchConsole,
  webSnapshotMode,
  type WebScaffold,
} from './scaffold.ts'
import { newEnglishPage, saveFailureShot } from './support.ts'

const SNAPSHOT_DIR = fileURLToPath(new URL('../../../snapshots/web/minimal-preset', import.meta.url))
// Platform-tiered corpus (owner-approved ㊵-91 乙档): the win32 tier (session.v4,
// pwsh) and the POSIX tier (session.v3, bash) are both committed and pinned per
// platform — a shared highest-generation pick would hand each platform the
// other one's shell calls.
const FIXTURE = join(SNAPSHOT_DIR, process.platform === 'win32' ? 'session.v4.jsonl' : 'session.v3.jsonl')
const UI_EXPECTED = join(SNAPSHOT_DIR, 'ui.expected.md')
const MODE = webSnapshotMode()
// The scenario is platform-tiered (owner-approved ㊵-91 乙档): the authored
// corpus's shell call and the drive prompt follow the running platform's
// shell tool, so each platform replays a corpus that matches its facts.
const SHELL_TOOL = process.platform === 'win32' ? 'pwsh' : 'bash'
const SCRIPTED_COMMAND = SHELL_TOOL === 'pwsh'
  ? "Write-Output 'MINIMAL_BASH_CARD_OK'"
  : "printf 'MINIMAL_BASH_CARD_OK\\n'"
const PROMPT = `Use the ${SHELL_TOOL} tool to run exactly: ${SCRIPTED_COMMAND}. Then reply exactly MINIMAL_PRESET_REQUEST_OK and stop.`

/** Rendered text of the system prompt surface node, or undefined when the surface carries none. */
function systemPromptText(session: Session): string | undefined {
  const message = session.deriveMessages().find(candidate => candidate.role === 'system')
  return message?.content.flatMap(block => block.type === 'text' ? [block.text] : []).join('')
}

describe('minimal agent preset', () => {
  let scaffold: WebScaffold
  let agentHandle: AgentHandle
  let disposeInjectedPrompt: () => void
  let browser: Browser | undefined
  let page: Page | undefined
  let tripwire: ReturnType<typeof watchConsole> | undefined

  beforeAll(async () => {
    scaffold = await launchWebScaffold({ replayFixture: FIXTURE, compareReplaySession: true, paceMs: 10 })
    disposeInjectedPrompt = scaffold.ctx.systemPrompt.section({
      name: 'test:injected-prompt',
      order: 999,
      text: 'THIS TEXT MUST NOT REACH THE MODEL.',
    })
    agentHandle = await scaffold.ctx.agents.create({
      sessionId: SessionId('minimal-preset-smoke'),
      meta: { cwd: scaffold.workspaceCwd, agentPreset: 'minimal' },
      agentOptions: { provider: 'deepseek-official', model: 'deepseek-v4-flash' },
      setup: agentCtx => scaffold.ctx.agentPresets.mount(agentCtx, 'minimal').then(() => undefined),
    })
    agentHandle.agent.followup(createUserMessage({
      content: [{ type: 'text', text: PROMPT }],
      source: { kind: 'user' },
    }))
    await agentHandle.agent.whenIdle()
  })

  afterAll(async () => {
    const failures: unknown[] = []
    await page?.close().catch((error: unknown) => failures.push(error))
    await browser?.close().catch((error: unknown) => failures.push(error))
    await agentHandle?.dispose().catch((error: unknown) => failures.push(error))
    try {
      disposeInjectedPrompt?.()
    } catch (error: unknown) {
      failures.push(error)
    }
    await scaffold?.close().catch((error: unknown) => failures.push(error))
    if (failures.length === 1) throw failures[0]
    if (failures.length > 1) throw new AggregateError(failures, 'minimal preset smoke teardown failed')
  })

  it('sends the exact RL prompt and shell schema, then executes the persistent shell', async () => {
    const requestHeader = agentHandle.agent.session.requestHeader()
    if (requestHeader === undefined) throw new Error('the minimal agent issued no model request')
    const systemPrompt = systemPromptText(agentHandle.agent.session)
    if (systemPrompt === undefined) throw new Error('the minimal agent issued no system prompt')
    expect(agentHandle.agent.session.snapshotEvents().some(event => event.type === 'user/message'
      && event.data.source.kind === 'runtime-context')).toBe(false)
    expect(scaffold.ctx.agentPresets.serviceFor(agentHandle.agent, 'fs')).toBeUndefined()
    expect(scaffold.ctx.agentPresets.serviceFor(agentHandle.agent, 'compaction')).toBeUndefined()

    const stateDir = join(scaffold.workspaceCwd, 'persistent-state')
    await mkdir(stateDir)
    const signal = new AbortController().signal
    const setupCommand = SHELL_TOOL === 'pwsh'
      ? `Set-Location -LiteralPath ${JSON.stringify(stateDir)}; $env:DSH_MINIMAL_STATE = 'PERSISTED'`
      : `cd ${JSON.stringify(stateDir)} && export DSH_MINIMAL_STATE=PERSISTED`
    await scaffold.ctx.tools.execute({
      signal,
      callId: ToolCallId('minimal-bash-state-setup'),
      name: SHELL_TOOL,
      arguments: { command: setupCommand },
      agent: agentHandle.agent,
    })
    const readCommand = SHELL_TOOL === 'pwsh'
      ? 'Write-Output "$($env:DSH_MINIMAL_STATE):$((Get-Location).Path)"'
      : 'printf \'%s:%s\\n\' "$DSH_MINIMAL_STATE" "$PWD"'

    const bash = await scaffold.ctx.tools.execute({
      signal,
      callId: ToolCallId('minimal-bash-state-read'),
      name: SHELL_TOOL,
      arguments: { command: readCommand },
      agent: agentHandle.agent,
    })
    const text = (result: typeof bash): string => result.content
      .filter(block => block.type === 'text')
      .map(block => block.text)
      .join('')
      .replaceAll(scaffold.workspaceCwd, '{{cwd}}')
      .replaceAll('\\', '/')
      .trimEnd()

    // Platform-tiered expectations: the persistent pwsh result carries no
    // success suffix, and the tool list names the running platform's shell.
    const persistentSuffix = SHELL_TOOL === 'pwsh' ? '' : '\n[Command finished with exit code 0]'
    expect({
      prompt: systemPrompt,
      tools: requestHeader.tools?.map(tool => tool.name),
      goalCommand: scaffold.ctx.commands.find(agentHandle.agent, 'goal') !== undefined,
      [SHELL_TOOL]: text(bash),
    }).toEqual({
      prompt: 'You are a helpful software engineer assistant.',
      tools: [SHELL_TOOL],
      goalCommand: false,
      [SHELL_TOOL]: `PERSISTED:{{cwd}}/persistent-state${persistentSuffix}`,
    })
    expect(requestHeader.tools?.toSorted((left, right) => left.name.localeCompare(right.name)))
      .toEqual(scaffold.ctx.tools.schemas(agentHandle.agent).toSorted((left, right) => left.name.localeCompare(right.name)))
  })

  it.skipIf(MODE === 'record')('expands the completed persistent Bash call in the Web conversation', async () => {
    onTestFailed(() => { if (page !== undefined) void saveFailureShot(page, 'web-minimal-persistent-bash-card') })
    browser = await chromium.launch()
    page = await newEnglishPage(browser)
    tripwire = watchConsole(page)
    await page.goto(scaffold.authenticatedUrl, { waitUntil: 'load' })
    await page.waitForSelector('[class*="frame"]', { timeout: 30_000 })

    // Sidebar groups render collapsed (boot provisions the Default workspace
    // group first); expand every collapsed group, then open the single session
    // row — the treeitem without aria-expanded (the ㊵-146 family recipe).
    const groupRow = page.locator('[role="treeitem"]').first()
    await groupRow.waitFor({ timeout: 15_000 })
    const view = page
    await expect.poll(async () => {
      for (const group of await view.locator('[role="treeitem"][aria-expanded="false"]').all()) {
        await group.click()
      }
      return view.locator('[role="treeitem"]:not([aria-expanded])').count()
    }, { timeout: 15_000 }).toBe(1)
    await page.locator('[role="treeitem"]:not([aria-expanded])').first().click()
    await page.getByText('MINIMAL_PRESET_REQUEST_OK', { exact: true }).waitFor({ timeout: 15_000 })

    const process = page.locator('[data-turn-process]')
    await process.waitFor({ timeout: 15_000 })
    await expect.poll(() => process.getAttribute('aria-expanded')).toBe('false')
    await process.click()
    await expect.poll(() => process.getAttribute('aria-expanded')).toBe('true')

    const group = page.locator('[data-chat-group-key]').filter({ has: page.locator('[data-sample="bash"]') }).first()
    const groupControl = group.locator('[data-process-activity]')
    await groupControl.waitFor({ timeout: 15_000 })
    await expect.poll(() => groupControl.getAttribute('aria-expanded')).toBe('false')
    const row = group.locator('[data-sample="bash"]').first()
    expect(await row.isVisible()).toBe(false)
    await groupControl.click()
    await expect.poll(() => groupControl.getAttribute('aria-expanded')).toBe('true')
    await row.waitFor({ timeout: 15_000 })
    await expect.poll(() => row.getAttribute('aria-expanded')).toBe('false')
    await row.click()

    await expect.poll(() => row.getAttribute('aria-expanded')).toBe('true')
    const call = row.locator('xpath=..')
    await call.getByText('IN', { exact: true }).waitFor()
    await call.getByText('OUT', { exact: true }).waitFor()
    // The persistent pwsh result carries no success suffix (its exit marker is
    // non-zero only), so the card text and the recorded command are
    // platform-tiered alongside the corpus.
    const cardText = SHELL_TOOL === 'pwsh'
      ? 'MINIMAL_BASH_CARD_OK'
      : 'MINIMAL_BASH_CARD_OK\n[Command finished with exit code 0]'
    await call.getByText(cardText, { exact: true }).waitFor()
    const commandNeedle = SHELL_TOOL === 'pwsh'
      ? new RegExp('"command": "?Write-Output \'MINIMAL_BASH_CARD_OK')
      : new RegExp('"command": "?printf \'MINIMAL_BASH_CARD_OK')
    await call.getByText(commandNeedle).waitFor()

    const snapshot = await captureStableAria(page, '[class*="centerCol"]', scaffold.workspaceCwd)
    await compareOrRefreshGolden(UI_EXPECTED, snapshot, MODE)
    expect(tripwire.pageErrors).toEqual([])
    expect(tripwire.warnings).toEqual([])
  }, 60_000)

  it('keeps its snapshot inventory closed', async () => {
    await assertFixtureInventory(SNAPSHOT_DIR, [
      'session.v3.jsonl',
      'system-prompt.expected.md',
      'tool-schemas.expected.json',
      'ui.expected.md',
    ])
  })
})
