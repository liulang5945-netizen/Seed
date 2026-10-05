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
  LifeDeleteCheckpointRequest,
  LifeDeleteDatasetRequest,
  LifeDeleteKnowledgeRequest,
  LifeFollowFrame,
  LifeProgressView,
  LifeResumeCheckpointRequest,
  LifeSnapshotValue,
  LifeTrainStartRequest,
  LifeUploadDatasetRequest,
  LifeUploadKnowledgeRequest,
} from './types.ts'

export type * from './types.ts'

/** Suffixes the runtime's dataset roster scans, mirrored from `api/training/datasets.py`. */
export const NATIVE_DATASET_SUFFIXES: readonly string[] = ['.jsonl', '.ndjson', '.json', '.txt', '.text', '.md', '.csv']

/** Bytes one uploaded dataset may carry; the panel pre-checks the same budget. */
export const UPLOAD_MAX_BYTES = 200 * 1024 * 1024

/** Characters Windows file names cannot hold, refused before the runtime fails on `open`. */
const INVALID_FILE_NAME = /[<>:"|?*\u0000-\u001f]/u

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
   * Upload one dataset file into the runtime's data directory. The name is
   * reduced to its basename and checked against the runtime's trainable
   * suffixes before any bytes leave the Host; the runtime stays the final
   * authority on what it stores.
   * @param request - picked file name and the file's bytes as base64.
   * @param signal - caller lifetime.
   * @returns the runtime's message naming the uploaded dataset.
   */
  @Remote
  async uploadDataset(request: LifeUploadDatasetRequest, signal: AbortSignal): Promise<LifeControlValue> {
    const name = datasetFileName(request.name)
    assertUploadBytes(request.data)
    return await this.command(() => this.client.uploadDataset({ name, data: request.data }, signal))
  }

  /**
   * Delete one dataset file the roster lists. The path is checked to stay a
   * relative roster path with a trainable suffix before the runtime is asked;
   * the runtime's data directories are the only places it may resolve.
   * @param request - POSIX path relative to the runtime's data directory.
   * @param signal - caller lifetime.
   * @returns the runtime's acknowledgement.
   */
  @Remote
  async deleteDataset(request: LifeDeleteDatasetRequest, signal: AbortSignal): Promise<LifeControlValue> {
    const path = datasetPath(request.path)
    return await this.command(() => this.client.deleteDataset({ path }, signal))
  }

  /**
   * Delete one checkpoint; the runtime refuses the active and the configured
   * checkpoint with its own conflict, because removing either breaks the
   * answering model or the next start.
   * @param request - file name inside the runtime's checkpoint directory.
   * @param signal - caller lifetime.
   * @returns the runtime's message naming the deleted checkpoint.
   */
  @Remote
  async deleteCheckpoint(request: LifeDeleteCheckpointRequest, signal: AbortSignal): Promise<LifeControlValue> {
    const filename = checkpointFileName(request.filename)
    return await this.command(() => this.client.deleteCheckpoint({ filename }, signal))
  }

  /**
   * Upload one knowledge document into the runtime's document directory; the
   * name is reduced to its basename and checked before any bytes leave the
   * Host, and the runtime vectorizes the file in the background.
   * @param request - picked file name and the file's bytes as base64.
   * @param signal - caller lifetime.
   * @returns the runtime's message naming the uploaded document.
   */
  @Remote
  async uploadKnowledge(request: LifeUploadKnowledgeRequest, signal: AbortSignal): Promise<LifeControlValue> {
    const name = usableFileName(request.name, 'name')
    assertUploadBytes(request.data)
    return await this.command(() => this.client.uploadKnowledge({ name, data: request.data }, signal))
  }

  /**
   * Delete one knowledge document the file list shows; the runtime removes it
   * from the index as well.
   * @param request - file name inside the runtime's document directory.
   * @param signal - caller lifetime.
   * @returns the runtime's acknowledgement.
   */
  @Remote
  async deleteKnowledge(request: LifeDeleteKnowledgeRequest, signal: AbortSignal): Promise<LifeControlValue> {
    const name = flatFileName(request.name, 'name')
    return await this.command(() => this.client.deleteKnowledge({ name }, signal))
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
  /* v8 ignore next -- the transport and typert layers reject with Error instances; reaching this arm needs a non-Error throw from outside them. */
  return error instanceof Error ? error.message : String(error)
}

/**
 * Reduce one picked file name to the basename the runtime stores, refusing
 * names the file system cannot hold.
 * @param raw - name exactly as the picker reported it.
 * @param field - request field the refusal names.
 * @returns the usable basename.
 */
function usableFileName(raw: string, field: string): string {
  /* v8 ignore next -- String.split returns at least one element, so pop() is never undefined. */
  const name = raw.split(/[\\/]/u).pop() ?? ''
  if (name === '' || name === '.' || name === '..') {
    throw new RemoteError('life/bad-request', `name "${raw}" has no usable file name`, {
      field,
      reason: 'a file name that is not empty',
    })
  }
  if (INVALID_FILE_NAME.test(name) || /[. ]$/u.test(name)) {
    throw new RemoteError('life/bad-request', `name "${name}" is not usable on Windows`, {
      field,
      reason: 'a file name without < > : " | ? * and without a trailing dot or space',
    })
  }
  return name
}

/**
 * Check a name a delete names: a flat file name, never a path or a drive
 * letter. An upload may reduce a picked path to its basename, but a delete
 * must act on exactly the file it names.
 * @param raw - file name exactly as the runtime listed it.
 * @param field - request field the refusal names.
 * @returns the usable flat file name.
 */
function flatFileName(raw: string, field: string): string {
  if (/[\\/:]/u.test(raw)) {
    throw new RemoteError('life/bad-request', `name "${raw}" is not a flat file name`, {
      field,
      reason: 'a flat file name without path separators',
    })
  }
  return usableFileName(raw, field)
}

/**
 * Reduce one picked file name to the dataset basename the runtime stores,
 * refusing names the runtime's roster could never show.
 * @param raw - name exactly as the picker reported it.
 * @returns the usable dataset basename.
 */
function datasetFileName(raw: string): string {
  const name = usableFileName(raw, 'name')
  const suffix = name.slice(name.lastIndexOf('.')).toLowerCase()
  if (!NATIVE_DATASET_SUFFIXES.includes(suffix)) {
    throw new RemoteError('life/bad-request', `dataset name "${name}" has no trainable suffix`, {
      field: 'name',
      reason: `one of ${NATIVE_DATASET_SUFFIXES.join(', ')}`,
    })
  }
  return name
}

/**
 * Check one dataset path a delete names: it must stay a relative roster path
 * with a trainable suffix, so no absolute path, drive letter, or `..` segment
 * ever reaches the runtime's file resolution.
 * @param raw - path exactly as the roster listed it.
 * @returns the normalized relative path.
 */
function datasetPath(raw: string): string {
  const path = raw.replace(/\\/gu, '/').replace(/^\.\//u, '')
  const segments = path.split('/')
  if (
    path === ''
    || path.startsWith('/')
    || /^[A-Za-z]:/u.test(path)
    || segments.some(segment => segment === '' || segment === '.' || segment === '..')
  ) {
    throw new RemoteError('life/bad-request', `dataset path "${raw}" is not a roster path`, {
      field: 'path',
      reason: 'a relative path inside the runtime data directory',
    })
  }
  const suffix = path.slice(path.lastIndexOf('.')).toLowerCase()
  if (!NATIVE_DATASET_SUFFIXES.includes(suffix)) {
    throw new RemoteError('life/bad-request', `dataset path "${path}" has no trainable suffix`, {
      field: 'path',
      reason: `one of ${NATIVE_DATASET_SUFFIXES.join(', ')}`,
    })
  }
  return path
}

/**
 * Check one checkpoint name a delete names: a flat `*.pt` file name, never a
 * path, a drive letter, or one of the dot-prefixed temporary files the roster
 * itself hides.
 * @param raw - file name exactly as the roster listed it.
 * @returns the usable checkpoint file name.
 */
function checkpointFileName(raw: string): string {
  const name = flatFileName(raw, 'filename')
  if (!name.toLowerCase().endsWith('.pt') || name.startsWith('.')) {
    throw new RemoteError('life/bad-request', `checkpoint name "${name}" is not a checkpoint file name`, {
      field: 'filename',
      reason: 'a *.pt file name without a leading dot',
    })
  }
  return name
}

/**
 * Refuse a payload larger than the upload budget, before it travels as an
 * oversized JSON body the connection bridge would reject without detail.
 * @param data - base64 bytes the caller sent.
 */
function assertUploadBytes(data: string): void {
  // base64 carries 3 bytes per 4 characters; the ceiling matches the panel's own check.
  if (data.length > Math.ceil(UPLOAD_MAX_BYTES / 3) * 4) {
    throw new RemoteError('life/bad-request', 'dataset upload exceeds the size budget', {
      field: 'data',
      reason: `at most ${String(UPLOAD_MAX_BYTES / (1024 * 1024))} MB`,
    })
  }
}

export default LifeController
