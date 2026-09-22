import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { afterEach, describe, expect, it } from 'vitest'
import { Context } from '@taiji/cordis'
import Loader from '@taiji/cordis-plugin-loader'
import Include from '@taiji/cordis-plugin-include'
import { ToolCallId } from '@taiji/dsh-llm'
import { Session, SessionId } from '@taiji/dsh-session'
import AgentRegistry from '@taiji/dsh-agent'
import type { Agent } from '@taiji/dsh-agent'
import SystemPrompt from '@taiji/dsh-system-prompt'
import ToolRuntime from '@taiji/dsh-tools'
import TerminalSessionService from '@taiji/dsh-terminal'
import SandboxProvider from '@taiji/dsh-sandbox'
import type { ConfinedArgv, SandboxPolicy } from '@taiji/dsh-sandbox'
import SandboxPolicyService from '@taiji/dsh-sandbox-policy'
import SessionProjectionRegistry from '@taiji/dsh-session-projection'
import LocalSubprocessRuntime from '@taiji/dsh-subprocess-local'
import * as TerminalLocal from '@taiji/dsh-terminal-bash'
import * as ToolPty from '@taiji/dsh-tool-terminal'
import { unsupportedInbox } from '@taiji/dsh-agent-loop-testkit'

let root: string | undefined
let context: Context | undefined

afterEach(async () => {
  await context?.fiber.dispose()
  context = undefined
  if (root !== undefined) await rm(root, { recursive: true, force: true })
  root = undefined
})

class PassthroughSandbox extends SandboxProvider {
  async confine(argv: readonly string[], _policy: SandboxPolicy): Promise<ConfinedArgv> {
    return { argv: [...argv], enforcement: 'full', denialSignatures: [], runnerFailureRules: [] }
  }
}

async function agent(ctx: Context): Promise<Agent> {
  const scope = ctx.plugin(() => {})
  const id = SessionId('pty-loader-agent')
  const session = Session.create(id)
  const value: Agent = {
    id, options: {}, session, inbox: unsupportedInbox(),
    status: 'idle',
    ctx: scope.ctx,
    send: () => {},
    followup: () => {}, steer: () => {}, inject: () => {}, cancel() {},
    runMaintenance: job => job(new AbortController().signal),
    whenIdle: () => Promise.resolve(),
  }
  await ctx.agents.register(value)
  return value
}

function resultText(result: { content: { type: string; text?: string }[] }): string {
  return result.content.filter(block => block.type === 'text').map(block => block.text).join('')
}

const suite = process.platform === 'linux' || process.platform === 'darwin' ? describe : describe.skip

suite('terminal real Loader composition through cordis.yml', () => {
  it('boots cordis.yml and preserves shell state across real tool calls', async () => {
    root = await mkdtemp(join(tmpdir(), 'dsh-pty-loader-'))
    const configPath = join(root, 'cordis.yml')
    await writeFile(configPath, [
      "- name: '@taiji/dsh-agent'",
      "- name: '@taiji/dsh-system-prompt'",
      "- name: '@taiji/dsh-tools'",
      "- name: '@taiji/dsh-terminal'",
      "- name: '@taiji/dsh-test-sandbox'",
      "- name: '@taiji/dsh-session-projection'",
      "- name: '@taiji/dsh-sandbox-policy'",
      '  config:',
      '    mode: danger-full-access',
      `    workspaceRoot: ${JSON.stringify(root)}`,
      "- name: '@taiji/dsh-subprocess-local'",
      "- name: '@taiji/dsh-terminal-bash'",
      '  config:',
      '    pollIntervalMs: 10',
      '    exactProbeAfterMs: 20',
      '    idleSilenceMs: 250',
      '    handoffGraceMs: 250',
      '    timeoutMs: 2000',
      '    disposeGraceMs: 500',
      "- name: '@taiji/dsh-tool-terminal'",
      '',
    ].join('\n'))

    context = new Context()
    context.baseUrl = pathToFileURL(root).href + '/'
    await context.plugin(Loader)
    context.loader.builtins.include = Include
    const modules = new Map<string, unknown>([
      ['@taiji/dsh-agent', AgentRegistry],
      ['@taiji/dsh-system-prompt', SystemPrompt],
      ['@taiji/dsh-tools', ToolRuntime],
      ['@taiji/dsh-terminal', TerminalSessionService],
      ['@taiji/dsh-test-sandbox', PassthroughSandbox],
      ['@taiji/dsh-session-projection', SessionProjectionRegistry],
      ['@taiji/dsh-sandbox-policy', SandboxPolicyService],
      ['@taiji/dsh-subprocess-local', LocalSubprocessRuntime],
      ['@taiji/dsh-terminal-bash', TerminalLocal],
      ['@taiji/dsh-tool-terminal', ToolPty],
    ])
    context.loader.internal = {
      version: 'v2',
      async import(specifier: string) {
        if (!modules.has(specifier)) throw new Error(`unexpected Loader import: ${specifier}`)
        return modules.get(specifier)
      },
    } as unknown as NonNullable<typeof context.loader.internal>
    await context.loader.create({ name: 'cordis:include', config: { path: pathToFileURL(configPath).href } })
    await context.loader.await()

    const owner = await agent(context)
    const signal = new AbortController().signal
    const spawn = await context.tools.execute({
      signal, callId: ToolCallId('spawn'), name: 'terminal_open', arguments: { type: 'shell', name: 'main', cwd: root }, agent: owner,
    })
    expect(resultText(spawn)).toContain('started terminal session pty-1 (main)')

    await context.tools.execute({
      signal, callId: ToolCallId('state'), name: 'terminal_send', arguments: { sessionId: 'pty-1', text: 'export KEEP=loader; cd /' }, agent: owner,
    })
    const read = await context.tools.execute({
      signal, callId: ToolCallId('read'), name: 'terminal_send', arguments: { sessionId: 'pty-1', text: 'printf "cwd=%s keep=%s\\n" "$PWD" "$KEEP"' }, agent: owner,
    })
    expect(resultText(read)).toContain('cwd=/ keep=loader')
    expect(context.terminals.list(owner)).toHaveLength(1)
  }, 15_000)
})
