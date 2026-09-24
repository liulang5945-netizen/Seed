/**
 * The Taiji runtime's chat shape and the translation from harness messages
 * into it. The translation is deliberately literal: the runtime takes one
 * current user turn, one optional system prompt, and completed `[user,
 * assistant]` pairs, so anything the shape has no slot for — reasoning,
 * tool calls, images, tool results — is dropped instead of being rewritten
 * into prose the runtime never asked for.
 *
 * Tool *names* are the exception: they travel as request metadata so the
 * runtime can record which tools a turn offered. The calls and results
 * themselves are still dropped.
 */

import type { GenerateOptions, RequestMessage } from '@taiji/dsh-llm'

/**
 * One chat request (`api/models.py:ChatRequest`).
 *
 * `system_prompt` is omitted when the request carries none, which is what
 * makes the runtime apply its own default persona.
 *
 * `session_id`, `purpose`, and `tools` are transport metadata: the runtime
 * keeps them for its own learning rings and never feeds them to the model.
 */
export interface TaijiChatRequest {
  /** Plain text of this request's current user turn. */
  prompt: string
  /** Effective system prompt, when the request carries one. */
  system_prompt?: string
  /** Completed turns, each `[user text, assistant text]`, oldest first. */
  history: [string, string][]
  /** Session this request belongs to, when the loop stamped one. */
  session_id?: string
  /** Auxiliary-call classification; an ordinary turn leaves it unset. */
  purpose?: string
  /** Names of the tools this request offers the model. */
  tools?: string[]
}

/** Join one message's visible text blocks; every other block type contributes nothing. */
function textOf(message: RequestMessage): string {
  return message.content.flatMap(block => block.type === 'text' ? [block.text] : []).join('\n')
}

/**
 * Translate one assembled request into the runtime's chat shape.
 *
 * The current turn is the last user-role message; every message before it
 * becomes one `[user, assistant]` pair when a user-side message is followed by
 * an assistant reply. A user-side message with no reply keeps its pair with an
 * empty second element, so the user's words survive rather than being dropped.
 * @param options - the fully assembled request.
 * @returns the runtime's request body.
 */
export function buildChatRequest(options: GenerateOptions): TaijiChatRequest {
  const messages = options.messages
  let promptIndex = -1
  for (const [position, message] of messages.entries()) {
    if (message.role === 'user') promptIndex = position
  }

  const history: [string, string][] = []
  let pending: string | undefined
  let system: string | undefined
  for (const [position, message] of messages.entries()) {
    if (message.role === 'system') {
      // The runtime takes exactly one system prompt, so the newest snapshot
      // wins: a continuing series that re-appends the prompt sends its latest.
      system = textOf(message)
      continue
    }
    // Everything from the current turn on is either its own slot or has none.
    if (promptIndex !== -1 && position >= promptIndex) continue
    if (message.role === 'developer') continue
    if (message.role === 'assistant') {
      if (pending !== undefined) {
        history.push([pending, textOf(message)])
        pending = undefined
      }
      continue
    }
    // `user` and `tool` both read as user-side input to the runtime.
    if (pending !== undefined) history.push([pending, ''])
    pending = textOf(message)
  }
  if (pending !== undefined) history.push([pending, ''])

  const current = promptIndex === -1 ? undefined : messages.at(promptIndex)
  const systemText = system ?? options.system
  // Names only: the runtime records what a turn offered, and every schema
  // would be weight the endpoint has no slot for anyway.
  const toolNames = (options.tools ?? []).map(tool => tool.name)
  return {
    prompt: current === undefined ? '' : textOf(current),
    ...systemText ? { system_prompt: systemText } : {},
    history,
    ...options.sessionId === undefined ? {} : { session_id: String(options.sessionId) },
    ...options.purpose === undefined ? {} : { purpose: options.purpose },
    ...toolNames.length === 0 ? {} : { tools: toolNames },
  }
}
