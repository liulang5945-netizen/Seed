/**
 * Polling producer for the Life snapshot. One timer serves every consumer:
 * each poll reads the runtime once, stamps the interval in force, and
 * publishes only when the reading changed. A follower always receives the
 * current snapshot first, so a reconnect never renders an empty panel.
 */

import type { Context } from '@taiji/cordis'
import type { LifeRuntimeClient } from './runtime-client.ts'
import type {
  LifeFollowFrame,
  LifeProgressView,
  LifeSnapshot,
  LifeTrainingStreamState,
} from './types.ts'

/** Poll cadence for an idle and for a training runtime. */
export interface LifeFeedOptions {
  /** Interval between reads while nothing runs. */
  readonly pollIntervalMs: number
  /** Interval between reads while training holds the runtime. */
  readonly activePollIntervalMs: number
}

/** Owns the poll loop, the current snapshot, and every follow generation. */
export class LifeFeed {
  private readonly followers = new Set<LifeFollower>()
  private readonly lifetime = new AbortController()
  private snapshot: LifeSnapshot | undefined
  private lastPublished: string | undefined
  private progress: LifeProgressView | undefined
  private warnings: readonly string[] = []
  private stream: LifeTrainingStreamState = 'idle'
  private lastOkAt: string | undefined
  private timer: ReturnType<typeof setTimeout> | undefined
  private polling = false
  private disposed = false

  /**
   * @param ctx - Host context owning this feed's lifetime.
   * @param client - runtime reader shared with the control verbs.
   * @param options - poll cadence.
   */
  constructor(
    private readonly ctx: Context,
    private readonly client: LifeRuntimeClient,
    private readonly options: LifeFeedOptions,
  ) {
    ctx.effect(() => () => { this.dispose() }, 'life-controller.feed')
  }

  /** Latest observed snapshot, absent until the first read settles. */
  get latest(): LifeSnapshot | undefined {
    return this.snapshot
  }

  /**
   * Start the poll loop. Idempotent.
   */
  start(): void {
    if (this.polling || this.stopped) return
    this.polling = true
    void this.cycle()
  }

  /**
   * Read the current snapshot, reading one when the poll loop has none yet.
   * @param signal - caller lifetime.
   * @returns the current snapshot.
   */
  async read(signal: AbortSignal): Promise<LifeSnapshot> {
    return this.snapshot ?? await this.refresh(signal)
  }

  /**
   * Open one generation: the current snapshot first, then every change.
   * @param signal - generation cancellation.
   * @returns the opening snapshot followed by replacement frames.
   */
  async *follow(signal: AbortSignal): AsyncIterable<LifeFollowFrame> {
    signal.throwIfAborted()
    const opening = await this.read(signal)
    const follower = new LifeFollower()
    this.followers.add(follower)
    try {
      yield { type: 'baseline', value: opening }
      for await (const value of follower.read(signal)) yield { type: 'snapshot', value }
    } finally {
      this.followers.delete(follower)
      follower.close()
    }
  }

  /**
   * Record the progress stream's state and re-read once.
   * @param progress - latest sample, absent when the stream carries none.
   * @param stream - how the stream stands.
   * @param warnings - warnings the last run's stream carried, empty when none.
   */
  sync(progress: LifeProgressView | undefined, stream: LifeTrainingStreamState, warnings: readonly string[]): void {
    this.progress = progress
    this.stream = stream
    this.warnings = warnings
    void this.refresh(this.lifetime.signal)
  }

  /** Read once and publish when the reading changed. */
  private async refresh(signal: AbortSignal): Promise<LifeSnapshot> {
    const snapshot = await this.client.readSnapshot(signal, {
      ...(this.progress === undefined ? {} : { progress: this.progress }),
      ...(this.warnings.length === 0 ? {} : { warnings: this.warnings }),
      stream: this.stream,
      ...(this.lastOkAt === undefined ? {} : { lastOkAt: this.lastOkAt }),
    })
    const stamped: LifeSnapshot = { ...snapshot, pollIntervalMs: this.interval() }
    if (snapshot.availability.runtime === 'ok') this.lastOkAt = snapshot.observedAt
    this.snapshot = stamped
    this.publish(stamped)
    return stamped
  }

  private async cycle(): Promise<void> {
    while (!this.stopped) {
      await this.refresh(this.lifetime.signal).catch((error: unknown) => {
        /* v8 ignore next -- readSnapshot converts transport failures into a down snapshot. */
        this.ctx.logger.warn(`life-controller: snapshot read failed: ${String(error)}`)
      })
      // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition -- Disposal can flip while the loop awaits the next tick.
      if (this.stopped) return
      await new Promise<void>((resolve) => {
        this.timer = setTimeout(resolve, this.interval())
      })
    }
  }

  /** Interval in force: the active cadence while training holds the runtime. */
  private interval(): number {
    const training = this.snapshot?.training.isTraining === true || this.stream === 'streaming'
    return training ? this.options.activePollIntervalMs : this.options.pollIntervalMs
  }

  private publish(snapshot: LifeSnapshot): void {
    const encoded = JSON.stringify(snapshot)
    if (encoded === this.lastPublished) return
    this.lastPublished = encoded
    for (const follower of this.followers) follower.push(snapshot)
  }

  private dispose(): void {
    if (this.disposed) return
    this.disposed = true
    this.lifetime.abort(new Error('life-controller: feed disposed'))
    if (this.timer !== undefined) clearTimeout(this.timer)
    this.timer = undefined
    for (const follower of this.followers) follower.close()
    this.followers.clear()
  }

  /** Whether disposal already ran; read through a getter so the poll loop re-reads it. */
  private get stopped(): boolean {
    return this.disposed
  }
}

/** One generation's latest-wins snapshot queue. */
class LifeFollower {
  private pending: LifeSnapshot | undefined
  private waiting: (() => void) | undefined
  private closed = false

  push(snapshot: LifeSnapshot): void {
    if (this.closed) return
    this.pending = snapshot
    this.waiting?.()
  }

  close(): void {
    if (this.closed) return
    this.closed = true
    this.waiting?.()
  }

  async *read(signal: AbortSignal): AsyncIterable<LifeSnapshot> {
    while (!this.closed && !signal.aborted) {
      const next = this.pending
      if (next !== undefined) {
        this.pending = undefined
        yield next
        continue
      }
      await this.wait(signal)
    }
  }

  private wait(signal: AbortSignal): Promise<void> {
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
      if (signal.aborted || this.closed || this.pending !== undefined) finish()
    })
  }
}
