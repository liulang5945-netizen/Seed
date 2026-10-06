/**
 * HTTP face of the Taiji local runtime (the Python service on
 * `127.0.0.1:8000`). Every read names the endpoint it came from, and every
 * failure keeps its own identity: an unreachable runtime, a non-2xx answer,
 * and a gated source that is simply not mounted are three different facts a
 * panel must be able to tell apart.
 */

import { RemoteError } from '@taiji/dsh-typert-protocol'
import type {
  LifeActionRequest,
  LifeActivateRequest,
  LifeArtifactsView,
  LifeAuthView,
  LifeCheckpointView,
  LifeConsolidateRequest,
  LifeConsolidationView,
  LifeControlValue,
  LifeDatasetView,
  LifeDeleteCheckpointRequest,
  LifeDeleteDatasetRequest,
  LifeDeleteKnowledgeRequest,
  LifeHealthView,
  LifeKnowledgeFileView,
  LifeKnowledgeView,
  LifeLegacyView,
  LifeLifeView,
  LifeMemoryView,
  LifeNativeView,
  LifePassReportView,
  LifeProgressView,
  LifeResumeCheckpointRequest,
  LifeRuntimeState,
  LifeSnapshot,
  LifeSource,
  LifeTrainingView,
  LifeTrainStartRequest,
  LifeUploadDatasetRequest,
  LifeUploadKnowledgeRequest,
  LifeWorkbenchView,
} from './types.ts'

/** Fixed runtime endpoints this client speaks to. */
const STATUS_PATH = '/api/runtime/status'
const CHECKPOINTS_PATH = '/api/train/checkpoints'
const TRAIN_FILES_PATH = '/api/train/files'
const LEGACY_LIFE_PATH = '/api/life/status'
const KNOWLEDGE_PATH = '/api/rag/status'
const KNOWLEDGE_FILES_PATH = '/api/rag/files'
const CONSOLIDATION_PATH = '/api/consolidation/status'
const ARTIFACTS_PATH = '/api/artifacts'
const RUNTIME_ACTIVATE_PATH = '/api/runtime/activate'
const TRAIN_NATIVE_PATH = '/api/train/native'
const RESUME_CHECKPOINT_PATH = '/api/train/resume_checkpoint'
const UPLOAD_DATASET_PATH = '/api/train/upload_dataset'
const DATASET_FILE_PATH = '/api/train/file'
const CHECKPOINT_FILE_PATH = '/api/train/checkpoint'
const UPLOAD_KNOWLEDGE_PATH = '/api/rag/upload'
const KNOWLEDGE_FILE_PATH = '/api/rag/file'
const LEGACY_LIFE_START_PATH = '/api/taiji/life/start'
const LEGACY_LIFE_STOP_PATH = '/api/taiji/life/stop'
const LEGACY_LIFE_ACTION_PATH = '/api/taiji/life/action'

/** Runtime address, timeout, and roster bound after schema defaults. */
export interface LifeRuntimeOptions {
  /** Base URL of the Taiji local runtime. */
  readonly baseURL: string
  /** Maximum duration of one HTTP read or control request. */
  readonly requestTimeoutMs: number
  /** Checkpoint rows carried into one snapshot. */
  readonly maxCheckpoints: number
}

/** State the caller owns and this client only carries through a read. */
export interface LifeTrainingCarry {
  /** Latest progress sample from the open stream, when one is open. */
  readonly progress?: LifeProgressView
  /** Warnings the open (or last settled) stream carried, when there are any. */
  readonly warnings?: readonly string[]
  /** How the progress stream stands. */
  readonly stream: 'idle' | 'streaming' | 'closed'
  /** ISO-8601 instant of the last observation the runtime answered, when known. */
  readonly lastOkAt?: string
}

/** Destinations for one training run's stream. */
export interface LifeTrainingSink {
  /** The runtime accepted the run and the progress stream is open. */
  accepted(): void
  /** One progress sample arrived. */
  progress(sample: LifeProgressView): void
  /** The run reported a non-fatal warning (a corpus-drift notice, for one). */
  warning(message: string): void
  /** The run finished; the runtime names its checkpoint. */
  completed(message: string): void
  /** The run failed after starting. */
  failed(reason: string): void
  /** The stream ended without a terminal event. */
  closed(): void
}

/** One reply from the runtime, kept with its status so callers can branch on it. */
interface LifeRuntimeReply {
  readonly status: number
  readonly body: unknown
}

/**
 * Read and command the Taiji local runtime.
 *
 * A transport failure raises `life/runtime-unreachable`; a non-2xx answer that
 * a verb cannot use raises `life/runtime-error` and carries the runtime's own
 * detail text. Gated endpoints return their 404 unchanged so the caller can
 * report "this source is not mounted" instead of "the runtime is broken".
 */
export class LifeRuntimeClient {
  /** @param options - runtime address, timeout, and roster bound. */
  constructor(private readonly options: LifeRuntimeOptions) {}

  /**
   * Read one complete snapshot from the runtime.
   * @param signal - caller lifetime.
   * @param carry - progress, stream state, and the last answering instant the caller owns.
   * @returns the snapshot, marked `down` when the runtime did not answer.
   */
  async readSnapshot(signal: AbortSignal, carry: LifeTrainingCarry): Promise<LifeSnapshot> {
    let status: LifeRuntimeReply | undefined
    let runtime: LifeRuntimeState = 'ok'
    const unavailable: string[] = []
    try {
      status = await this.read(STATUS_PATH, signal)
    } catch (error) {
      runtime = 'down'
      unavailable.push(describeFailure(error))
    }
    const health = status === undefined ? undefined : healthView(status.body)
    const memory = status === undefined ? undefined : memoryView(status.body)
    const life = status === undefined ? undefined : lifeView(status.body)
    // The workbench and auth sections ride the same always-on status read;
    // a runtime that omits a section keeps it absent rather than zero-filled.
    const workbench = status === undefined ? undefined : workbenchView(status.body)
    const auth = status === undefined ? undefined : authView(status.body)
    const training = trainingView(status?.body, carry)

    // The gated life surface is probed for availability: its numbers already
    // arrive through `runtime/status.life`, and reading it again would either
    // duplicate them or let two sources disagree inside one snapshot.
    const legacy = await this.readGated(LEGACY_LIFE_PATH, signal, 'legacy', unavailable)
    const knowledge = await this.readGated(KNOWLEDGE_PATH, signal, 'knowledge', unavailable)
    // The mounted document list rides the knowledge surface: it is read only
    // while that surface answered, so a disabled knowledge base stays one
    // `disabled` fact instead of a second missing-source line.
    const knowledgeFiles = knowledge.state === 'ok'
      ? await this.readKnowledgeFiles(signal, unavailable)
      : undefined
    const knowledgeReading = knowledge.state === 'ok' ? knowledgeView(knowledge.body, knowledgeFiles) : undefined
    const checkpoints = training === undefined || runtime === 'down'
      ? undefined
      : await this.readCheckpoints(signal, unavailable)
    // The consolidation surface is newer than some running runtimes, so it is
    // read only while the status read answered and a missing endpoint becomes
    // one line in `unavailable` rather than a failure.
    const consolidation = runtime === 'down'
      ? undefined
      : await this.readConsolidation(signal, unavailable)
    const consolidationReading = consolidation === undefined ? undefined : consolidationView(consolidation)
    // The trainable roster feeds the panel's dataset chooser; it answers
    // whenever the runtime is up, and its absence leaves the field absent.
    const trainFiles = runtime === 'down'
      ? undefined
      : await this.readTrainFiles(signal, unavailable)
    const datasetsReading = trainFiles === undefined ? undefined : trainFilesView(trainFiles)
    // The publish surface answers whenever the runtime is up; its absence
    // (an older runtime without the artifacts router) is one unavailable
    // line, not a broken snapshot.
    const artifacts = runtime === 'down'
      ? undefined
      : await this.readArtifacts(signal, unavailable)
    const source = lifeSource(status?.body)

    return {
      source,
      observedAt: new Date().toISOString(),
      fresh: runtime === 'ok',
      pollIntervalMs: 0,
      ...(health === undefined ? {} : { health }),
      ...(memory === undefined ? {} : { memory }),
      ...(workbench === undefined ? {} : { workbench }),
      ...(auth === undefined ? {} : { auth }),
      ...(life === undefined ? {} : { life }),
      training: {
        ...(training ?? emptyTraining()),
        ...(checkpoints === undefined ? {} : { checkpoints }),
        ...(datasetsReading === undefined ? {} : { datasets: datasetsReading }),
      },
      ...(knowledgeReading === undefined ? {} : { knowledge: knowledgeReading }),
      ...(consolidationReading === undefined ? {} : { consolidation: consolidationReading }),
      ...(artifacts === undefined ? {} : { artifacts }),
      availability: {
        runtime,
        legacy: legacy.state,
        knowledge: knowledge.state,
        trainingStream: carry.stream,
      },
      unavailable,
      ...(carry.lastOkAt === undefined ? {} : { lastOkAt: carry.lastOkAt }),
    }
  }

  /**
   * Start a native training run and fold its progress stream.
   * @param request - run parameters; omitted fields keep the runtime's defaults.
   * @param signal - run lifetime; aborting stops folding, not the runtime's run.
   * @param sink - destinations for progress, completion, failure, and an early end.
   */
  async streamTraining(
    request: LifeTrainStartRequest,
    signal: AbortSignal,
    sink: LifeTrainingSink,
  ): Promise<void> {
    await this.foldStream(TRAIN_NATIVE_PATH, trainBody(request), signal, sink)
  }

  /**
   * Continue training from a saved checkpoint over the same progress-stream
   * contract, including the runtime's corpus-drift `warning` events.
   * @param request - checkpoint name and optional datasets and tick cap.
   * @param signal - run lifetime; aborting stops folding, not the runtime's run.
   * @param sink - destinations for progress, warnings, completion, failure, and an early end.
   */
  async resumeCheckpoint(
    request: LifeResumeCheckpointRequest,
    signal: AbortSignal,
    sink: LifeTrainingSink,
  ): Promise<void> {
    await this.foldStream(RESUME_CHECKPOINT_PATH, resumeBody(request), signal, sink)
  }

  /** Open one training stream and fold its shared SSE contract to the sink. */
  private async foldStream(path: string, body: Record<string, unknown>, signal: AbortSignal, sink: LifeTrainingSink): Promise<void> {
    const reply = await this.open(path, {
      method: 'POST',
      headers: { 'content-type': 'application/json', accept: 'text/event-stream' },
      body: JSON.stringify(body),
    }, signal)
    if (!reply.ok || reply.body === null) {
      throw await responseError(reply)
    }
    sink.accepted()
    for await (const frame of readSse(reply.body, signal)) {
      const event = frame as { readonly type?: unknown }
      if (event.type === 'progress') {
        sink.progress(progressView(frame))
        continue
      }
      if (event.type === 'warning') {
        sink.warning(text(frame, 'message'))
        continue
      }
      if (event.type === 'completed') {
        sink.completed(text(frame, 'message'))
        return
      }
      if (event.type === 'error') {
        sink.failed(text(frame, 'message'))
        return
      }
    }
    sink.closed()
  }

  /**
   * Pause the running training.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  trainPause(signal: AbortSignal): Promise<LifeControlValue> {
    return this.command('/api/train/pause', signal)
  }

  /**
   * Resume a paused training.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  trainResume(signal: AbortSignal): Promise<LifeControlValue> {
    return this.command('/api/train/resume', signal)
  }

  /**
   * Stop the running training after its current step.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  trainStop(signal: AbortSignal): Promise<LifeControlValue> {
    return this.command('/api/train/stop', signal)
  }

  /**
   * Force-release the training lock the runtime still holds.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  trainReset(signal: AbortSignal): Promise<LifeControlValue> {
    return this.command('/api/train/reset', signal)
  }

  /**
   * Run one native sleep consolidation pass.
   * @param request - pass parameters; omitted fields keep the runtime's defaults.
   * @param signal - caller lifetime.
   * @returns the runtime's pass report message.
   */
  consolidate(request: LifeConsolidateRequest, signal: AbortSignal): Promise<LifeControlValue> {
    // The runtime's body model is required, so a reason is always sent —
    // defaulting to its own `manual` when the caller named none.
    return this.command('/api/consolidate', signal, request.reason ?? 'manual')
  }

  /**
   * Answer later turns from a platform-owned checkpoint (empty id = built-in).
   * @param request - checkpoint name the runtime resolves inside its directory.
   * @param signal - caller lifetime.
   * @returns the runtime's acceptance message naming what became active.
   */
  async activateCheckpoint(request: LifeActivateRequest, signal: AbortSignal): Promise<LifeControlValue> {
    // The body is always sent: `checkpoint_id` distinguishes an explicit
    // built-in request ('') from an omission the runtime would default-fill.
    const reply = await this.send(RUNTIME_ACTIVATE_PATH, { checkpoint_id: request.checkpointId }, signal)
    return controlValue(reply)
  }

  /**
   * Upload one dataset file into the runtime's data directory. The bytes are
   * sent as multipart/form-data, the shape `POST /api/train/upload_dataset`
   * accepts; the runtime answers with the dataset's new relative path.
   * @param request - sanitized file name and the file's bytes as base64.
   * @param signal - caller lifetime.
   * @returns the runtime's message naming the uploaded dataset.
   */
  async uploadDataset(request: LifeUploadDatasetRequest, signal: AbortSignal): Promise<LifeControlValue> {
    const form = new FormData()
    form.append('file', new Blob([Buffer.from(request.data, 'base64')]), request.name)
    const reply = await this.open(UPLOAD_DATASET_PATH, {
      method: 'POST',
      headers: { accept: 'application/json' },
      body: form,
    }, signal)
    return controlValue({ status: reply.status, body: await parseJson(reply) })
  }

  /**
   * Delete one dataset file from the runtime's data directory.
   * @param request - POSIX path relative to the data directory, as the roster lists it.
   * @param signal - caller lifetime.
   * @returns the runtime's acknowledgement.
   */
  async deleteDataset(request: LifeDeleteDatasetRequest, signal: AbortSignal): Promise<LifeControlValue> {
    return await this.deleteAck(`${DATASET_FILE_PATH}/${encodeRoutePath(request.path)}`, signal)
  }

  /**
   * Delete one checkpoint from the runtime's checkpoint directory. The runtime
   * refuses the active and the configured checkpoint with its own conflict.
   * @param request - file name inside the checkpoint directory.
   * @param signal - caller lifetime.
   * @returns the runtime's message naming the deleted checkpoint.
   */
  async deleteCheckpoint(request: LifeDeleteCheckpointRequest, signal: AbortSignal): Promise<LifeControlValue> {
    return await this.deleteAck(`${CHECKPOINT_FILE_PATH}/${encodeRoutePath(request.filename)}`, signal)
  }

  /**
   * Upload one knowledge document into the runtime's document directory; the
   * bytes are sent as multipart/form-data, the shape `POST /api/rag/upload`
   * accepts, and the runtime vectorizes the file in the background.
   * @param request - sanitized file name and the file's bytes as base64.
   * @param signal - caller lifetime.
   * @returns the runtime's message naming the uploaded document.
   */
  async uploadKnowledge(request: LifeUploadKnowledgeRequest, signal: AbortSignal): Promise<LifeControlValue> {
    const form = new FormData()
    form.append('file', new Blob([Buffer.from(request.data, 'base64')]), request.name)
    const reply = await this.open(UPLOAD_KNOWLEDGE_PATH, {
      method: 'POST',
      headers: { accept: 'application/json' },
      body: form,
    }, signal)
    return controlValue({ status: reply.status, body: await parseJson(reply) })
  }

  /**
   * Delete one knowledge document, both from the document directory and from
   * the index the runtime rebuilds.
   * @param request - file name inside the document directory.
   * @param signal - caller lifetime.
   * @returns the runtime's acknowledgement.
   */
  async deleteKnowledge(request: LifeDeleteKnowledgeRequest, signal: AbortSignal): Promise<LifeControlValue> {
    return await this.deleteAck(`${KNOWLEDGE_FILE_PATH}/${encodeRoutePath(request.name)}`, signal)
  }

  /**
   * Send one DELETE and read its verdict. The runtime's dataset route answers a
   * vanished file with HTTP 200 while its own body says `error`, so a body that
   * refuses the verb is raised as the runtime's own failure instead of being
   * reported as a success.
   * @param path - the runtime path to delete, already encoded.
   * @param signal - caller lifetime.
   * @returns the runtime's acknowledgement.
   */
  private async deleteAck(path: string, signal: AbortSignal): Promise<LifeControlValue> {
    const reply = await this.open(path, { method: 'DELETE', headers: { accept: 'application/json' } }, signal)
    const body = await parseJson(reply)
    if (reply.status >= 200 && reply.status < 300 && text(body, 'status') === 'error') {
      throw runtimeError(reply.status, body)
    }
    return controlValue({ status: reply.status, body })
  }

  /**
   * Start the Legacy life scheduler.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  lifeStart(signal: AbortSignal): Promise<LifeControlValue> {
    return this.gated(LEGACY_LIFE_START_PATH, signal)
  }

  /**
   * Stop the Legacy life scheduler.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  lifeStop(signal: AbortSignal): Promise<LifeControlValue> {
    return this.gated(LEGACY_LIFE_STOP_PATH, signal)
  }

  /**
   * Force one Legacy life activity.
   * @param request - activity and reason.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  lifeAction(request: LifeActionRequest, signal: AbortSignal): Promise<LifeControlValue> {
    return this.gated(`${LEGACY_LIFE_ACTION_PATH}/${request.action}`, signal, request.reason)
  }

  private async command(path: string, signal: AbortSignal, reason?: string): Promise<LifeControlValue> {
    const reply = await this.send(path, reason === undefined ? undefined : { reason }, signal)
    return controlValue(reply)
  }

  private async gated(path: string, signal: AbortSignal, reason?: string): Promise<LifeControlValue> {
    const reply = await this.send(path, reason === undefined ? undefined : { reason }, signal)
    if (reply.status === 404) {
      throw new RemoteError('life/unavailable', 'the running runtime does not serve the Legacy life surface', {
        source: 'life',
        reason: `HTTP 404 from ${path}`,
      })
    }
    return controlValue(reply)
  }

  private async readCheckpoints(signal: AbortSignal, unavailable: string[]): Promise<readonly LifeCheckpointView[] | undefined> {
    try {
      const reply = await this.read(CHECKPOINTS_PATH, signal)
      if (reply.status !== 200) {
        unavailable.push(`checkpoints: HTTP ${String(reply.status)}`)
        return undefined
      }
      const rows = value(reply.body, 'checkpoints')
      if (!Array.isArray(rows)) return undefined
      return rows.slice(0, this.options.maxCheckpoints).map(checkpointView)
    } catch (error) {
      unavailable.push(describeFailure(error))
      return undefined
    }
  }

  private async readConsolidation(signal: AbortSignal, unavailable: string[]): Promise<unknown> {
    try {
      const reply = await this.read(CONSOLIDATION_PATH, signal)
      if (reply.status !== 200) {
        unavailable.push(`consolidation: HTTP ${String(reply.status)}`)
        return undefined
      }
      return reply.body
    } catch (error) {
      unavailable.push(describeFailure(error))
      return undefined
    }
  }

  private async readTrainFiles(signal: AbortSignal, unavailable: string[]): Promise<unknown> {
    try {
      const reply = await this.read(TRAIN_FILES_PATH, signal)
      if (reply.status !== 200) {
        unavailable.push(`train-files: HTTP ${String(reply.status)}`)
        return undefined
      }
      return reply.body
    } catch (error) {
      unavailable.push(describeFailure(error))
      return undefined
    }
  }

  private async readKnowledgeFiles(signal: AbortSignal, unavailable: string[]): Promise<readonly LifeKnowledgeFileView[] | undefined> {
    try {
      const reply = await this.read(KNOWLEDGE_FILES_PATH, signal)
      if (reply.status !== 200) {
        unavailable.push(`knowledge-files: HTTP ${String(reply.status)}`)
        return undefined
      }
      const rows = value(reply.body, 'files')
      if (!Array.isArray(rows)) return undefined
      return rows.map(knowledgeFileView)
    } catch (error) {
      unavailable.push(describeFailure(error))
      return undefined
    }
  }

  private async readArtifacts(signal: AbortSignal, unavailable: string[]): Promise<LifeArtifactsView | undefined> {
    try {
      const reply = await this.read(ARTIFACTS_PATH, signal)
      if (reply.status !== 200) {
        unavailable.push(`artifacts: HTTP ${String(reply.status)}`)
        return undefined
      }
      const runtime = object(reply.body, 'runtime')
      if (runtime === undefined) {
        unavailable.push('artifacts: no runtime block')
        return undefined
      }
      return { activeId: text(runtime, 'active_checkpoint_id'), configuredId: text(runtime, 'configured_checkpoint_id') }
    } catch (error) {
      unavailable.push(describeFailure(error))
      return undefined
    }
  }

  private async readGated(
    path: string,
    signal: AbortSignal,
    source: 'legacy' | 'knowledge',
    unavailable: string[],
  ): Promise<{ state: 'ok' | 'disabled' | 'down'; body: unknown }> {
    try {
      const reply = await this.read(path, signal)
      if (reply.status === 404) return { state: 'disabled', body: undefined }
      if (reply.status !== 200) {
        unavailable.push(`${source}: HTTP ${String(reply.status)}`)
        return { state: 'down', body: undefined }
      }
      return { state: 'ok', body: reply.body }
    } catch (error) {
      unavailable.push(describeFailure(error))
      return { state: 'down', body: undefined }
    }
  }

  private async read(path: string, signal: AbortSignal): Promise<LifeRuntimeReply> {
    const reply = await this.open(path, { headers: { accept: 'application/json' } }, signal)
    return { status: reply.status, body: await parseJson(reply) }
  }

  private async send(path: string, body: unknown, signal: AbortSignal): Promise<LifeRuntimeReply> {
    const reply = await this.open(path, {
      method: 'POST',
      headers: { 'content-type': 'application/json', accept: 'application/json' },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    }, signal)
    return { status: reply.status, body: await parseJson(reply) }
  }

  private async open(path: string, init: RequestInit, signal: AbortSignal): Promise<Response> {
    const timeout = AbortSignal.timeout(this.options.requestTimeoutMs)
    try {
      return await fetch(new URL(path, this.options.baseURL), {
        ...init,
        signal: AbortSignal.any([signal, timeout]),
        redirect: 'error',
      })
    } catch (error) {
      /* v8 ignore next -- an aborted AbortSignal always carries a reason in Node. */
      if (signal.aborted) throw signal.reason ?? error
      throw new RemoteError(
        'life/runtime-unreachable',
        `Taiji runtime is unreachable at ${this.options.baseURL}`,
        /* v8 ignore next 2 -- fetch rejects with Error instances, never bare values. */
        { baseURL: this.options.baseURL, reason: error instanceof Error ? error.message : String(error) },
      )
    }
  }
}

/** Which organ the payload says answered; a detached native organ still counts as native. */
function lifeSource(body: unknown): LifeSource {
  const life = object(body, 'life')
  if (life === undefined) return 'absent'
  const status = text(life, 'status')
  if (status === 'seed') return 'native'
  return status === 'ok' ? 'legacy' : 'absent'
}

/** Health projection from a `RuntimeStatusPayload`. */
function healthView(body: unknown): LifeHealthView | undefined {
  const health = object(body, 'health')
  if (health === undefined) return undefined
  return {
    state: text(health, 'state'),
    modelLoaded: flag(health, 'model_loaded'),
    modelName: text(health, 'model_name'),
    seedActive: flag(health, 'is_seed'),
    startupComplete: flag(health, 'startup_complete'),
  }
}

/** Memory projection from a `RuntimeStatusPayload`. */
function memoryView(body: unknown): LifeMemoryView | undefined {
  const memory = object(body, 'memory')
  if (memory === undefined) return undefined
  return {
    totalGb: number(memory, 'total_gb'),
    availableGb: number(memory, 'available_gb'),
    usedPct: number(memory, 'used_pct'),
  }
}

/** Workbench capability projection from the status payload's `tools` section. */
function workbenchView(body: unknown): LifeWorkbenchView | undefined {
  const tools = object(body, 'tools')
  if (tools === undefined) return undefined
  return {
    status: text(tools, 'status'),
    count: number(tools, 'count'),
    source: text(tools, 'source'),
    owner: text(tools, 'owner'),
    revision: number(tools, 'revision'),
    error: text(tools, 'error'),
  }
}

/** Authentication projection from the status payload's `auth` section. */
function authView(body: unknown): LifeAuthView | undefined {
  const auth = object(body, 'auth')
  if (auth === undefined) return undefined
  return {
    enabled: flag(auth, 'enabled'),
    authenticated: flag(auth, 'authenticated'),
    tokenValid: flag(auth, 'token_valid'),
  }
}

/** Life projection: the payload says which organ answered, and that decides the shape. */
function lifeView(body: unknown): LifeLifeView | undefined {
  const life = object(body, 'life')
  if (life === undefined) return undefined
  const status = text(life, 'status')
  const needs = numericMap(life['needs'])
  if (status === 'seed') {
    if (!flag(life, 'is_running')) return { isRunning: false }
    const native: LifeNativeView = {
      tick: number(life, 'tick'),
      mode: text(life, 'mode'),
      needs,
      drives: numericMap(life['drives']),
    }
    return { isRunning: true, native }
  }
  if (status !== 'ok') return undefined
  const lastHeartbeat = optionalText(life, 'last_heartbeat')
  const lastActivity = optionalText(life, 'last_activity')
  const legacy: LifeLegacyView = {
    isRunning: flag(life, 'is_running'),
    lifeState: text(life, 'life_state'),
    dominantNeed: text(life, 'dominant_need'),
    needs,
    totalHeartbeats: number(life, 'total_heartbeats'),
    totalEvents: number(life, 'total_events'),
    ...(lastHeartbeat === undefined ? {} : { lastHeartbeat }),
    ...(lastActivity === undefined ? {} : { lastActivity }),
  }
  return { isRunning: legacy.isRunning, legacy }
}

/** Training projection; `undefined` means the status read itself failed. */
function trainingView(body: unknown, carry: LifeTrainingCarry): LifeTrainingView | undefined {
  const training = object(body, 'training')
  if (training === undefined) return undefined
  const progress = carry.progress
  const warnings = carry.warnings
  return {
    isTraining: flag(training, 'is_training'),
    pauseRequested: flag(training, 'pause_requested'),
    stopRequested: flag(training, 'stop_requested'),
    publishing: flag(training, 'publishing'),
    ...(progress === undefined ? {} : { progress }),
    ...(warnings === undefined || warnings.length === 0 ? {} : { warnings }),
    checkpoints: [],
  }
}

/** Knowledge projection from `GET /api/rag/status` plus `GET /api/rag/files`. */
function knowledgeView(body: unknown, files: readonly LifeKnowledgeFileView[] | undefined): LifeKnowledgeView {
  return {
    docCount: number(body, 'doc_count'),
    chunkCount: number(body, 'chunk_count'),
    hasEmbeddings: flag(body, 'has_embeddings'),
    embedDim: number(body, 'embed_dim'),
    ...(files === undefined ? {} : { files }),
  }
}

/** One mounted knowledge file from `GET /api/rag/files`. */
function knowledgeFileView(row: unknown): LifeKnowledgeFileView {
  const sizeBytes = optionalNumber(row, 'size')
  return {
    name: text(row, 'name'),
    ...(sizeBytes === undefined ? {} : { sizeBytes }),
    status: text(row, 'status'),
  }
}

/** Trainable roster projection from `GET /api/train/files`; `undefined` when the body carried no roster. */
function trainFilesView(body: unknown): readonly LifeDatasetView[] | undefined {
  const entries = value(body, 'entries')
  if (!Array.isArray(entries)) return undefined
  const rows: LifeDatasetView[] = []
  for (const row of entries) {
    const path = text(row, 'path')
    if (path === '') continue
    rows.push({ path, sizeBytes: number(row, 'size_bytes') })
  }
  return rows
}

/** Memory and consolidation projection from `GET /api/consolidation/status`. */
function consolidationView(body: unknown): LifeConsolidationView {
  const journal = object(body, 'journal') ?? {}
  const spec = object(body, 'spec')
  const report = object(body, 'last_report')
  const reportSpec = report === undefined ? undefined : object(report, 'spec')
  return {
    passes: number(body, 'passes'),
    lastPassAt: number(body, 'last_pass_at'),
    lastCorpus: text(body, 'last_corpus'),
    projectedDigests: number(body, 'projected_digests'),
    running: flag(body, 'running'),
    spec: spec === undefined
      ? null
      : { reason: text(spec, 'reason'), datasets: stringList(spec['datasets']), weaknesses: stringList(spec['weaknesses']) },
    lastReport: report === undefined ? null : passReportView(report, reportSpec),
    journal: {
      entries: number(journal, 'entries'),
      byKind: numericMap(journal['by_kind']),
      sessions: number(journal, 'sessions'),
      lastRecordedAt: number(journal, 'last_recorded_at'),
    },
  }
}

/** One pass report, its gate reason read from the report's own spec block. */
function passReportView(report: Record<string, unknown>, spec: Record<string, unknown> | undefined): LifePassReportView {
  return {
    reason: text(report, 'reason'),
    specReason: spec === undefined ? '' : text(spec, 'reason'),
    durationMs: number(report, 'duration_ms'),
    weaknesses: stringList(report['weaknesses']),
    notes: stringList(report['notes']),
  }
}

/** Training defaults used before the first status answer. */
function emptyTraining(): LifeTrainingView {
  return {
    isTraining: false,
    pauseRequested: false,
    stopRequested: false,
    publishing: false,
    checkpoints: [],
  }
}

/** Request body the runtime's `NativeTrainRequest` accepts. */
function trainBody(request: LifeTrainStartRequest): Record<string, unknown> {
  return {
    ...(request.datasets === undefined ? {} : { datasets: [...request.datasets] }),
    ...(request.parameterBudget === undefined ? {} : { parameter_budget: request.parameterBudget }),
    ...(request.seed === undefined ? {} : { seed: request.seed }),
    ...(request.maxSymbols === undefined ? {} : { max_symbols: request.maxSymbols }),
  }
}

/** Request body the runtime's `ResumeRequest` accepts; `checkpoint` is always sent. */
function resumeBody(request: LifeResumeCheckpointRequest): Record<string, unknown> {
  return {
    checkpoint: request.checkpoint,
    ...(request.datasets === undefined ? {} : { datasets: [...request.datasets] }),
    ...(request.maxTicks === undefined ? {} : { max_ticks: request.maxTicks }),
  }
}

/** One progress sample from a `progress` event. */
function progressView(event: unknown): LifeProgressView {
  const eta = optionalNumber(event, 'eta')
  return {
    fraction: number(event, 'fraction'),
    step: number(event, 'step'),
    loss: number(event, 'loss'),
    elapsed: number(event, 'elapsed'),
    ...(eta === undefined ? {} : { eta }),
    epoch: number(event, 'epoch'),
    totalEpochs: number(event, 'total_epochs'),
    samplesPerSec: number(event, 'samples_per_sec'),
    totalSteps: number(event, 'total_steps'),
  }
}

/** One checkpoint row from `GET /api/train/checkpoints`. */
function checkpointView(row: unknown): LifeCheckpointView {
  return {
    filename: text(row, 'filename'),
    step: number(row, 'step'),
    bytes: number(row, 'bytes'),
    modifiedUtc: text(row, 'modified_utc'),
    savedAtUtc: text(row, 'saved_at_utc'),
    numEpochs: number(row, 'num_epochs'),
  }
}

/** Accepted-verb value, raising the runtime's own failure when it refused. */
function controlValue(reply: LifeRuntimeReply): LifeControlValue {
  if (reply.status < 200 || reply.status >= 300) throw runtimeError(reply.status, reply.body)
  return { message: text(reply.body, 'message') }
}

/** A streaming response that never became a stream, with its body read for the detail. */
async function responseError(response: Response): Promise<RemoteError> {
  const raw = await response.text().catch(() => '')
  if (raw.trim() === '') return runtimeError(response.status, undefined)
  try {
    return runtimeError(response.status, JSON.parse(raw))
  } catch {
    return runtimeError(response.status, raw)
  }
}

/** Non-2xx answer as a Remote failure carrying the runtime's detail text. */
function runtimeError(status: number, body: unknown): RemoteError {
  const detail = (typeof body === 'string' ? body : text(body, 'detail') || text(body, 'message')) || `HTTP ${String(status)}`
  const code = status === 409 ? 'life/conflict' : status === 400 || status === 422 ? 'life/bad-request' : 'life/runtime-error'
  return new RemoteError(code, `Taiji runtime refused the request: ${detail}`, code === 'life/runtime-error'
    ? { status, detail }
    : code === 'life/conflict'
      ? { reason: detail }
      : { field: 'request', reason: detail })
}

/** SSE frames from one streaming response, `[DONE]` excluded. */
async function* readSse(body: ReadableStream<Uint8Array>, signal: AbortSignal): AsyncIterable<unknown> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (!signal.aborted) {
      const chunk = await reader.read()
      if (chunk.done) break
      buffer += decoder.decode(chunk.value, { stream: true })
      let boundary = buffer.indexOf('\n\n')
      while (boundary !== -1) {
        const raw = buffer.slice(0, boundary)
        buffer = buffer.slice(boundary + 2)
        const frame = sseData(raw)
        if (frame !== undefined && frame !== '[DONE]') yield parseFrame(frame)
        boundary = buffer.indexOf('\n\n')
      }
    }
  } finally {
    await reader.cancel().catch(() => undefined)
  }
}

/** Concatenate one SSE event's `data:` lines. */
function sseData(raw: string): string | undefined {
  const lines = raw.split('\n').filter(line => line.startsWith('data:'))
  if (lines.length === 0) return undefined
  return lines.map(line => line.slice(5).trimStart()).join('\n')
}

/** Parse one SSE payload, keeping a bare string payload as its message. */
function parseFrame(frame: string): unknown {
  try {
    return JSON.parse(frame)
  } catch {
    return { type: 'error', message: frame }
  }
}

/** Response body as JSON, `undefined` for an empty or malformed body. */
async function parseJson(reply: Response): Promise<unknown> {
  const raw = await reply.text().catch(() => '')
  if (raw.trim() === '') return undefined
  try {
    return JSON.parse(raw)
  } catch {
    return undefined
  }
}

/** The named nested object, when the payload has one. */
function object(body: unknown, key: string): Record<string, unknown> | undefined {
  if (typeof body !== 'object' || body === null) return undefined
  const value = (body as Record<string, unknown>)[key]
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return undefined
  return value as Record<string, unknown>
}

/** The named nested value, when the payload has one. */
function value(body: unknown, key: string): unknown {
  if (typeof body !== 'object' || body === null) return undefined
  return (body as Record<string, unknown>)[key]
}

/** String field, empty when absent. */
function text(body: unknown, key: string): string {
  return optionalText(body, key) ?? ''
}

/** Optional string field. */
function optionalText(body: unknown, key: string): string | undefined {
  const raw = value(body, key)
  return typeof raw === 'string' && raw !== '' ? raw : undefined
}

/** Number field, zero when absent or not finite. */
function number(body: unknown, key: string): number {
  return optionalNumber(body, key) ?? 0
}

/** Optional number field. */
function optionalNumber(body: unknown, key: string): number | undefined {
  const raw = value(body, key)
  return typeof raw === 'number' && Number.isFinite(raw) ? raw : undefined
}

/** Boolean field, false when absent. */
function flag(body: unknown, key: string): boolean {
  return value(body, key) === true
}

/** Open numeric map, values that are not finite numbers dropped. */
function numericMap(raw: unknown): Readonly<Record<string, number>> {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) return {}
  const out: Record<string, number> = {}
  for (const [key, entry] of Object.entries(raw as Record<string, unknown>)) {
    if (typeof entry === 'number' && Number.isFinite(entry)) out[key] = entry
  }
  return out
}

/** String list, non-strings dropped; the runtime owns these lines verbatim. */
function stringList(raw: unknown): readonly string[] {
  if (!Array.isArray(raw)) return []
  return raw.filter((entry): entry is string => typeof entry === 'string')
}

/**
 * Encode one runtime route suffix: each `/`-separated segment is percent
 * encoded on its own, so a nested dataset path keeps its separators while the
 * runtime still reads the exact names the roster listed.
 * @param relative - a POSIX relative path or a flat file name.
 * @returns the path safe to append to a route prefix.
 */
function encodeRoutePath(relative: string): string {
  return relative.split('/').map(segment => encodeURIComponent(segment)).join('/')
}

/** Operator-readable one-liner for a failed read. */
function describeFailure(error: unknown): string {
  if (error instanceof RemoteError) return `${error.code}: ${error.message}`
  /* v8 ignore next -- every failure on a read path raises an Error. */
  return error instanceof Error ? error.message : String(error)
}
