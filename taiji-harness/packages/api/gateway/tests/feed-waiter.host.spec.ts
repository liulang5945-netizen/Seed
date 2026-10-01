import { describe, expect, it } from 'vitest'
import { FeedWaiter } from '../src/feed-waiter.ts'

describe('FeedWaiter', () => {
  it('parks until wake releases the parked reader', async () => {
    const waiter = new FeedWaiter()
    const abort = new AbortController()
    let settled = false
    const parked = waiter.wait(abort.signal, () => false, () => false).then(() => {
      settled = true
    })
    await Promise.resolve()
    expect(settled).toBe(false)
    waiter.wake()
    await parked
    expect(settled).toBe(true)
  })

  it('settles a parked wait when the reader signal aborts', async () => {
    const waiter = new FeedWaiter()
    const abort = new AbortController()
    const parked = waiter.wait(abort.signal, () => false, () => false)
    await Promise.resolve()
    abort.abort()
    await expect(parked).resolves.toBeUndefined()
  })

  it('settles without parking when the readiness predicate already holds', async () => {
    const waiter = new FeedWaiter()
    const abort = new AbortController()
    await expect(waiter.wait(abort.signal, () => true, () => false)).resolves.toBeUndefined()
    await expect(waiter.wait(abort.signal, () => false, () => true)).resolves.toBeUndefined()
  })

  it('stops owning the callback after settling, so later wakes are inert', async () => {
    const waiter = new FeedWaiter()
    const abort = new AbortController()
    await waiter.wait(abort.signal, () => true, () => false)
    expect(() => {
      waiter.wake()
    }).not.toThrow()
  })
})
