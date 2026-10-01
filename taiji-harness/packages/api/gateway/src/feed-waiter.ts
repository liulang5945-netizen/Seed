/**
 * The wake/await seam shared by the follower queues that publish latest-wins
 * snapshots (life controller) and framed follow streams (workspace
 * controller). Both classes own their pending queue and closed flag; this
 * module owns only the callback handoff — one installed waiter, woken by
 * publication or closure, settled by the same signal the reader passed in.
 * @module @taiji/dsh-api-gateway/feed-waiter
 */

/**
 * One waiter's callback handoff: wake on publication or closure, settle on
 * the reader's abort.
 */
export class FeedWaiter {
  private waiting: (() => void) | undefined

  /** Release the parked reader, if one is parked. */
  wake(): void {
    this.waiting?.()
  }

  /**
   * Park until the next wake, the owning queue's readiness predicates hold,
   * or the signal aborts. The predicates are re-read inside the promise so
   * the poll loop's re-check semantics stay with the queue that owns the
   * state.
   * @param signal - the reader's lifetime.
   * @param hasWork - whether the owning queue currently holds unread work.
   * @param isClosed - whether the owning queue has been closed.
   */
  wait(signal: AbortSignal, hasWork: () => boolean, isClosed: () => boolean): Promise<void> {
    return new Promise((resolve) => {
      const finish = (): void => {
        signal.removeEventListener('abort', finish)
        /* v8 ignore next -- one read owns the sole installed wait callback. */
        if (this.waiting === finish) this.waiting = undefined
        resolve()
      }
      this.waiting = finish
      signal.addEventListener('abort', finish, { once: true })
      /* v8 ignore next -- native signals and the private queue cannot change during this synchronous setup. */
      if (signal.aborted || isClosed() || hasWork()) finish()
    })
  }
}
