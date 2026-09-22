import { Context } from '@taiji/cordis'
import { afterEach, describe, expect, it } from 'vitest'
import ToolRuntime from '@taiji/dsh-tools'
import SystemPrompt from '@taiji/dsh-system-prompt'
import UserQuestionService from '@taiji/dsh-user-questions'
import { apply } from '../src/index.ts'

let ctx: Context | undefined

afterEach(async () => {
  await ctx?.fiber.dispose()
  ctx = undefined
})

describe('ui-user-questions node plugin', () => {
  it('mounts no model-facing tool', async () => {
    ctx = new Context()
    await ctx.plugin(SystemPrompt)
    await ctx.plugin(ToolRuntime)
    await ctx.plugin(UserQuestionService)

    await ctx.plugin({ apply }).await()

    // Host UI registration must leave each preset's model-facing tool list intact.
    expect(ctx.tools.get('ask_user_question')).toBeUndefined()
  })
})
