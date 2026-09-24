/**
 * Host owner of the Life Remote namespace: it reads the Taiji local runtime,
 * exposes the reconnect-safe snapshot stream, and carries the control verbs a
 * life panel issues. The runtime stays the single source of truth — this
 * service never derives a life number the runtime did not report.
 */

import { Context } from '@taiji/cordis'
import z from '@taiji/schemastery'
import { Remote, RemoteError, TypertRemoteService } from '@taiji/dsh-typert-protocol'
import { LifeFeed } from './feed.ts'
import { LifeRuntimeClient } from './runtime-client.ts'
import type { LifeTrainingSink } from './runtime-client.ts'
import type {
  LifeActionRequest,
  LifeActivateRequest,
  LifeConsolidateRequest,
  LifeControlValue,
  LifeFollowFrame,
  LifeProgressView,
  LifeResumeCheckpointRequest,
  LifeSnapshotValue,
  LifeTrainStartRequest,
} from './types.ts'

export type * from './types.ts'

/** Runtime address, timeouts, and polling cadence. */
export interface Config {
  /** Base URL of the Taiji local runtime. */
  baseURL?: string
  /** Interval between reads while nothing runs. */
  pollIntervalMs?: number
  /** Interval between reads while training holds the runtime. */
  activePollIntervalMs?: number
  /** Maximum duration of one runtime request. */
  requestTimeoutMs?: number
  /** Checkpoint rows carried into one snapshot. */
  maxCheckpoints?: number
}

/** Configuration after schema defaults have been applied. */
type ResolvedConfig = Config & {
  baseURL: string
  pollIntervalMs: number
  activePollIntervalMs: number
  requestTimeoutMs: number
  maxCheckpoints: number
}

declare module '@taiji/cordis' {
  interface Context {
    /** Host Taiji runtime snapshot and control owner. */
    lifeController: LifeController
  }
}

/** Host service backing the generated `ctx.remote.life` namespace. */
export class LifeController extends TypertRemoteService {
  static inject = ['typert']

  static Config: z<Config, ResolvedConfig> = z.object({
    baseURL: z.string().default('http://127.0.0.1:8000'),
    pollIntervalMs: z.natural().min(250).default(5_000),
    activePollIntervalMs: z.natural().min(250).default(2_000),
    requestTimeoutMs: z.natural().min(1).default(2_000),
    maxCheckpoints: z.natural().min(1).default(20),
  })

  private readonly config: ResolvedConfig
  private readonly client: LifeRuntimeClient
  private readonly feed: LifeFeed
  private active: AbortController | undefined
  private accepted = false
  private progress: LifeProgressView | undefined
  private warnings: string[] = []
  private stream: 'idle' | 'streaming' | 'closed' = 'idle'

  /**
   * @param ctx - Host context owning this service's lifetime.
   * @param config - runtime address, timeouts, and poll cadence.
   */
  constructor(ctx: Context, config: Config = {}) {
    super(ctx, 'lifeController', { namespace: 'life' })
    this.config = LifeController.Config(config)
    this.client = new LifeRuntimeClient(this.config)
    this.feed = new LifeFeed(ctx, this.client, this.config)
    this.feed.start()
    ctx.effect(() => () => {
      this.active?.abort(new Error('life-controller: service disposed'))
      this.active = undefined
    }, 'life-controller.training')
  }

  /**
   * Read the current runtime snapshot.
   * @param signal - caller lifetime.
   * @returns the snapshot, marked `down` when the runtime did not answer.
   */
  @Remote
  async snapshot(signal: AbortSignal): Promise<LifeSnapshotValue> {
    return { snapshot: await this.feed.read(signal) }
  }

  /**
   * Stream the current snapshot first, then every change.
   * @param signal - stream lifetime.
   * @returns the opening snapshot followed by replacement frames.
   */
  @Remote({ mode: 'stream' })
  follow(signal: AbortSignal): AsyncIterable<LifeFollowFrame> {
    return this.feed.follow(signal)
  }

  /**
   * Start a native training run and fold its progress into the snapshot stream.
   * @param request - run parameters; omitted fields keep the runtime's defaults.
   * @returns the runtime's acceptance message; progress arrives through `follow`.
   */
  @Remote
  async trainStart(request: LifeTrainStartRequest): Promise<LifeControlValue> {
    return await this.beginRun((signal, sink) => this.client.streamTraining(request, signal, sink))
  }

  /**
   * Continue training from a saved checkpoint and fold its progress into the
   * snapshot stream, carrying the runtime's corpus-drift warnings through.
   * @param request - checkpoint name and optional datasets and tick cap.
   * @returns the runtime's acceptance message; progress arrives through `follow`.
   */
  @Remote
  async trainResumeCheckpoint(request: LifeResumeCheckpointRequest): Promise<LifeControlValue> {
    return await this.beginRun((signal, sink) => this.client.resumeCheckpoint(request, signal, sink))
  }

  /**
   * Open one training run through the shared exclusivity: one stream per Host,
   * the run accepted or refused as a whole, every later frame folded into the
   * snapshot. A new run clears the previous run's warnings.
   * @param open - how to open this run's stream against a signal and sink.
   * @returns the acceptance message.
   */
  private async beginRun(open: (signal: AbortSignal, sink: LifeTrainingSink) => Promise<void>): Promise<LifeControlValue> {
    if (this.active !== undefined) {
      throw new RemoteError('life/conflict', 'a training stream is already open on this Host', {
        reason: 'start requested while the previous run still streams',
      })
    }
    const controller = new AbortController()
    const accepted = Promise.withResolvers<void>()
    this.active = controller
    this.accepted = false
    this.stream = 'streaming'
    this.progress = undefined
    this.warnings = []
    const run = open(controller.signal, {
      accepted: () => {
        this.accepted = true
        accepted.resolve()
      },
      progress: (sample) => {
        this.progress = sample
        this.feed.sync(this.progress, this.stream, this.warnings)
      },
      warning: (message) => {
        this.warnings.push(message)
        this.feed.sync(this.progress, this.stream, this.warnings)
      },
      completed: (message) => { this.settle(undefined, message) },
      failed: (reason) => { this.settle(reason) },
      closed: () => { this.settle('the runtime ended the progress stream without a terminal event') },
    })
    run.catch((error: unknown) => { this.settle(describe(error)) })
    try {
      await Promise.race([accepted.promise, run])
    } catch (error) {
      this.settle(describe(error))
      throw error
    }
    return { message: 'training accepted' }
  }

  /**
   * Pause the running training.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  @Remote
  async trainPause(signal: AbortSignal): Promise<LifeControlValue> {
    return await this.command(() => this.client.trainPause(signal))
  }

  /**
   * Resume a paused training.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  @Remote
  async trainResume(signal: AbortSignal): Promise<LifeControlValue> {
    return await this.command(() => this.client.trainResume(signal))
  }

  /**
   * Stop the running training after its current step.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  @Remote
  async trainStop(signal: AbortSignal): Promise<LifeControlValue> {
    return await this.command(() => this.client.trainStop(signal))
  }

  /**
   * Force-release the training lock the runtime still holds.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  @Remote
  async trainReset(signal: AbortSignal): Promise<LifeControlValue> {
    return await this.command(() => this.client.trainReset(signal))
  }

  /**
   * Run one native sleep consolidation pass.
   * @param request - pass parameters; omitted fields keep the runtime's defaults.
   * @param signal - caller lifetime.
   * @returns the runtime's pass report message.
   */
  @Remote
  async consolidate(request: LifeConsolidateRequest, signal: AbortSignal): Promise<LifeControlValue> {
    return await this.command(() => this.client.consolidate(request, signal))
  }

  /**
   * Answer later turns from a platform-owned checkpoint; the empty id activates
   * the built-in seed. A failure is the runtime's own refusal — activation
   * swaps the model every later turn runs through.
   * @param request - checkpoint name inside the runtime's checkpoint directory.
   * @param signal - caller lifetime.
   * @returns the runtime's message naming what became active.
   */
  @Remote
  async activateCheckpoint(request: LifeActivateRequest, signal: AbortSignal): Promise<LifeControlValue> {
    return await this.command(() => this.client.activateCheckpoint(request, signal))
  }

  /**
   * Start the Legacy life scheduler.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  @Remote
  async lifeStart(signal: AbortSignal): Promise<LifeControlValue> {
    return await this.command(() => this.client.lifeStart(signal))
  }

  /**
   * Stop the Legacy life scheduler.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  @Remote
  async lifeStop(signal: AbortSignal): Promise<LifeControlValue> {
    return await this.command(() => this.client.lifeStop(signal))
  }

  /**
   * Force one Legacy life activity.
   * @param request - activity and operator-visible reason.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  @Remote
  async lifeAction(request: LifeActionRequest, signal: AbortSignal): Promise<LifeControlValue> {
    return await this.command(() => this.client.lifeAction(request, signal))
  }

  /** Run one control verb, then re-read so the panel sees the effect immediately. */
  private async command(verb: () => Promise<LifeControlValue>): Promise<LifeControlValue> {
    const value = await verb()
    this.feed.sync(this.progress, this.stream, this.warnings)
    return value
  }

  /**
   * Close the progress stream and publish the resulting state. A run that never
   * reached the runtime leaves the panel idle; one that failed after starting is
   * `closed`, because the panel must be able to tell a refusal from a dead run.
   */
  private settle(reason: string | undefined, message?: string): void {
    if (this.active === undefined) return
    const started = this.accepted
    this.active = undefined
    this.accepted = false
    this.progress = undefined
    this.stream = reason !== undefined && started ? 'closed' : 'idle'
    if (reason !== undefined && started) this.ctx.logger.warn(`life-controller: training stream ended: ${reason}`)
    else if (message !== undefined) this.ctx.logger.info(`life-controller: training finished: ${message}`)
    // The settled run's warnings stay folded until a new run clears them: a
    // corpus-drift notice is a fact the operator must still be able to read.
    this.feed.sync(this.progress, this.stream, this.warnings)
  }
}

/** Operator-readable text for one thrown value. */
function describe(error: unknown): string {
  if (error instanceof RemoteError) return `${error.code}: ${error.message}`
  return error instanceof Error ? error.message : String(error)
}

export default LifeController
