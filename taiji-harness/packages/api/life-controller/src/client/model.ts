/** Client-side Life snapshot model shared by Remote transport and panel projection. */

import type {} from '@taiji/dsh-api-life-controller/remote'
import { RemoteError, type RemoteFailure, type RemoteResult, type TypertClientRemote } from '@taiji/dsh-typert-protocol'
import type {
  LifeActionRequest,
  LifeActivateRequest,
  LifeConsolidateRequest,
  LifeControlValue,
  LifeResumeCheckpointRequest,
  LifeSnapshot,
  LifeTrainStartRequest,
} from '../types.ts'

/** Complete generated `ctx.remote.life` namespace. */
export type LifeRemote = TypertClientRemote['life']

/** Immutable Client Life state. */
export interface LifeSnapshotState {
  /** Latest snapshot, absent before the first frame. */
  readonly snapshot: LifeSnapshot | undefined
  /** Arrival lifecycle of the state. */
  readonly state: 'loading' | 'ready' | 'error'
  /** Terminal stream failure, absent while the stream is healthy. */
  readonly error: RemoteFailure | null
}

/** State operations emitted by a decoded Life follow generation. */
export interface LifeStreamSink {
  /** Replace all state from the generation baseline. */
  replaceBaseline(snapshot: LifeSnapshot): void
  /** Replace the snapshot with a later observation. */
  replaceSnapshot(snapshot: LifeSnapshot): void
  /** Publish a terminal stream failure without discarding the last snapshot. */
  handleStreamFailure(error: unknown): void
}

/**
 * Owns the Client Life projection: the latest snapshot stays addressable
 * through one identity-stable state object, so a renderer can subscribe with
 * `useSyncExternalStore` and never loop on a freshly built value.
 */
export class ClientLifeModel implements LifeStreamSink {
  private current: LifeSnapshotState = { snapshot: undefined, state: 'loading', error: null }
  private readonly listeners = new Set<() => void>()

  /** @param remote - generated Client Remote face for the life namespace. */
  constructor(private readonly remote: LifeRemote) {}

  /**
   * Read the identity-stable current state.
   * @returns the current snapshot state.
   */
  getSnapshot(): LifeSnapshotState {
    return this.current
  }

  /**
   * Subscribe to state changes.
   * @param listener - invalidation callback.
   * @returns unsubscribe function.
   */
  subscribe(listener: () => void): () => void {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }

  replaceBaseline(snapshot: LifeSnapshot): void {
    this.install({ snapshot, state: 'ready', error: null })
  }

  replaceSnapshot(snapshot: LifeSnapshot): void {
    this.install({ snapshot, state: 'ready', error: null })
  }

  handleStreamFailure(error: unknown): void {
    this.install({ ...this.current, state: 'error', error: failureOf(error) })
  }

  /**
   * Read one snapshot over the wire and reflect it into this model.
   * @param signal - caller lifetime.
   * @returns the snapshot or the Host failure.
   */
  async read(signal?: AbortSignal): Promise<RemoteResult<LifeSnapshot>> {
    const result = await this.remote.snapshot(signal)
    if (!result.ok) {
      this.handleStreamFailure(result.error)
      return result
    }
    this.replaceSnapshot(result.value.snapshot)
    return { ok: true, value: result.value.snapshot }
  }

  /**
   * Start a training run. The Host's verb takes no caller lifetime: the run is
   * accepted or refused as a whole, and its progress arrives through `follow`.
   * @param request - run parameters.
   * @returns the runtime's acceptance message or the Host failure.
   */
  trainStart(request: LifeTrainStartRequest): Promise<RemoteResult<LifeControlValue>> {
    return this.remote.trainStart(request)
  }

  /**
   * Continue training from a saved checkpoint. Same acceptance shape as
   * `trainStart`: the run is accepted or refused as a whole, and its progress
   * — corpus-drift warnings included — arrives through `follow`.
   * @param request - checkpoint name and optional datasets and tick cap.
   * @returns the runtime's acceptance message or the Host failure.
   */
  trainResumeCheckpoint(request: LifeResumeCheckpointRequest): Promise<RemoteResult<LifeControlValue>> {
    return this.remote.trainResumeCheckpoint(request)
  }

  /**
   * Pause the running training.
   * @param signal - caller lifetime.
   * @returns the runtime's message or the Host failure.
   */
  trainPause(signal?: AbortSignal): Promise<RemoteResult<LifeControlValue>> {
    return this.remote.trainPause(signal)
  }

  /**
   * Resume a paused training.
   * @param signal - caller lifetime.
   * @returns the runtime's message or the Host failure.
   */
  trainResume(signal?: AbortSignal): Promise<RemoteResult<LifeControlValue>> {
    return this.remote.trainResume(signal)
  }

  /**
   * Stop the running training.
   * @param signal - caller lifetime.
   * @returns the runtime's message or the Host failure.
   */
  trainStop(signal?: AbortSignal): Promise<RemoteResult<LifeControlValue>> {
    return this.remote.trainStop(signal)
  }

  /**
   * Force-release the training lock.
   * @param signal - caller lifetime.
   * @returns the runtime's message or the Host failure.
   */
  trainReset(signal?: AbortSignal): Promise<RemoteResult<LifeControlValue>> {
    return this.remote.trainReset(signal)
  }

  /**
   * Run one native sleep consolidation pass.
   * @param request - pass parameters.
   * @param signal - caller lifetime.
   * @returns the runtime's pass report message or the Host failure.
   */
  consolidate(request: LifeConsolidateRequest, signal?: AbortSignal): Promise<RemoteResult<LifeControlValue>> {
    return this.remote.consolidate(request, signal)
  }

  /**
   * Answer later turns from a platform-owned checkpoint (empty id = built-in).
   * @param request - checkpoint name inside the runtime's checkpoint directory.
   * @param signal - caller lifetime.
   * @returns the runtime's message or the Host failure.
   */
  activateCheckpoint(request: LifeActivateRequest, signal?: AbortSignal): Promise<RemoteResult<LifeControlValue>> {
    return this.remote.activateCheckpoint(request, signal)
  }

  /**
   * Start the Legacy life scheduler.
   * @param signal - caller lifetime.
   * @returns the runtime's message or the Host failure.
   */
  lifeStart(signal?: AbortSignal): Promise<RemoteResult<LifeControlValue>> {
    return this.remote.lifeStart(signal)
  }

  /**
   * Stop the Legacy life scheduler.
   * @param signal - caller lifetime.
   * @returns the runtime's message or the Host failure.
   */
  lifeStop(signal?: AbortSignal): Promise<RemoteResult<LifeControlValue>> {
    return this.remote.lifeStop(signal)
  }

  /**
   * Force one Legacy life activity.
   * @param request - activity and reason.
   * @param signal - caller lifetime.
   * @returns the runtime's message or the Host failure.
   */
  lifeAction(request: LifeActionRequest, signal?: AbortSignal): Promise<RemoteResult<LifeControlValue>> {
    return this.remote.lifeAction(request, signal)
  }

  /** Swap the state object once, then wake every subscriber. */
  private install(next: LifeSnapshotState): void {
    this.current = next
    for (const listener of this.listeners) listener()
  }
}

/**
 * Normalize a thrown or returned failure into the Remote failure shape.
 * @param error - whatever a call rejected or returned as its failure.
 * @returns the Remote failure carrying its stable code and message.
 */
export function failureOf(error: unknown): RemoteFailure {
  if (typeof error === 'object' && error !== null && 'code' in error && 'message' in error) {
    return error as RemoteFailure
  }
  return new RemoteError('life/stream-failed', error instanceof Error ? error.message : String(error), {
    reason: error instanceof Error ? error.message : String(error),
  })
}
