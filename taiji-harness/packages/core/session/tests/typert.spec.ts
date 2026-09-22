import { describe, expect, it } from 'vitest'
import { Context } from '@taiji/cordis'
import SessionStore, { SessionId } from '@taiji/dsh-session'
import TypertRegistry from '@taiji/dsh-typert-registry'

describe('Session Typert provider', () => {
  it('contributes live Session lookup in either service load order', async () => {
    const ctx = new Context()
    const sessionFiber = ctx.plugin(SessionStore)
    await sessionFiber
    await ctx.plugin(TypertRegistry)
    const session = ctx.sessions.create(SessionId('remote-session'))

    const lookup = ctx.typert.lookups.get('session')
    expect(lookup).toMatchObject({
      parameter: 'session',
      wire: 'sessionId',
      hostTypeSymbol: '@taiji/dsh-session#Session',
      wireTypeSymbol: '@taiji/dsh-session/types#SessionId',
    })
    expect(lookup?.resolve(session.id)).toBe(session)

    await sessionFiber.dispose()
    expect(ctx.typert.lookups.get('session')).toBeUndefined()
  })
})
