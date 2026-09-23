/** React-free Client Life service and command facade. */

import { Service, type Context } from '@taiji/cordis'
import type { RemoteFailure, RemoteResult } from '@taiji/dsh-typert-protocol'
import type {
  LifeActionRequest,
  LifeControlValue,
  LifeSnapshot,
  LifeTrainStartRequest,
} from '../types.ts'
import type { ClientLifeModel, LifeSnapshotState } from './model.ts'

/** A Life control verb was refused by the Host or the runtime. */
export class LifeControlError extends Error {
  override readonly name = 'LifeControlError'

  /** @param rpcError - Host business or folded carrier failure. */
  constructor(readonly rpcError: RemoteFailure) {
    super(`life control failed: ${rpcError.code}: ${rpcError.message}`)
  }
}

/** Client Life state and control verbs consumed by the life panel. */
export interface ILife {
  /** Read the identity-stable current snapshot state. */
  getSnapshot(): LifeSnapshotState
  /**
   * Subscribe to snapshot changes.
   * @param listener - invalidation callback.
   * @returns unsubscribe function.
   */
  subscribe(listener: () => void): () => void
  /**
   * Read one snapshot on demand.
   * @param signal - caller lifetime.
   * @returns the observed snapshot.
   */
  refresh(signal?: AbortSignal): Promise<LifeSnapshot>
  /**
   * Start a native training run.
   * @param request - run parameters; omitted fields keep the runtime's defaults.
   * @param signal - caller lifetime.
   * @returns the runtime's acceptance message.
   */
  trainStart(request?: LifeTrainStartRequest, signal?: AbortSignal): Promise<LifeControlValue>
  /**
   * Pause the running training.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  trainPause(signal?: AbortSignal): Promise<LifeControlValue>
  /**
   * Resume a paused training.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  trainResume(signal?: AbortSignal): Promise<LifeControlValue>
  /**
   * Stop the running training after its current step.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  trainStop(signal?: AbortSignal): Promise<LifeControlValue>
  /**
   * Force-release the training lock the runtime still holds.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  trainReset(signal?: AbortSignal): Promise<LifeControlValue>
  /**
   * Start the Legacy life scheduler.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  lifeStart(signal?: AbortSignal): Promise<LifeControlValue>
  /**
   * Stop the Legacy life scheduler.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  lifeStop(signal?: AbortSignal): Promise<LifeControlValue>
  /**
   * Force one Legacy life activity.
   * @param request - activity and operator-visible reason.
   * @param signal - caller lifetime.
   * @returns the runtime's message.
   */
  lifeAction(request: LifeActionRequest, signal?: AbortSignal): Promise<LifeControlValue>
}

/** Client Life state projection and command facade. */
export class LifeClient extends Service implements ILife {
  /**
   * @param ctx - Client root context.
   * @param model - Remote-backed Life state model.
   */
  constructor(ctx: Context, private readonly model: ClientLifeModel) {
    super(ctx, 'life')
  }

  getSnapshot(): LifeSnapshotState {
    return this.model.getSnapshot()
  }

  subscribe(listener: () => void): () => void {
    return this.model.subscribe(listener)
  }

  async refresh(signal?: AbortSignal): Promise<LifeSnapshot> {
    const result = await this.model.read(signal)
    if (!result.ok) throw new LifeControlError(result.error)
    return result.value
  }

  async trainStart(request: LifeTrainStartRequest = {}, signal?: AbortSignal): Promise<LifeControlValue> {
    return await this.unwrap(this.model.trainStart(request, signal))
  }

  async trainPause(signal?: AbortSignal): Promise<LifeControlValue> {
    return await this.unwrap(this.model.trainPause(signal))
  }

  async trainResume(signal?: AbortSignal): Promise<LifeControlValue> {
    return await this.unwrap(this.model.trainResume(signal))
  }

  async trainStop(signal?: AbortSignal): Promise<LifeControlValue> {
    return await this.unwrap(this.model.trainStop(signal))
  }

  async trainReset(signal?: AbortSignal): Promise<LifeControlValue> {
    return await this.unwrap(this.model.trainReset(signal))
  }

  async lifeStart(signal?: AbortSignal): Promise<LifeControlValue> {
    return await this.unwrap(this.model.lifeStart(signal))
  }

  async lifeStop(signal?: AbortSignal): Promise<LifeControlValue> {
    return await this.unwrap(this.model.lifeStop(signal))
  }

  async lifeAction(request: LifeActionRequest, signal?: AbortSignal): Promise<LifeControlValue> {
    return await this.unwrap(this.model.lifeAction(request, signal))
  }

  /** Unwrap one Host result, raising the structured control failure. */
  private async unwrap(pending: Promise<RemoteResult<LifeControlValue>>): Promise<LifeControlValue> {
    const result = await pending
    if (!result.ok) throw new LifeControlError(result.error)
    return result.value
  }
}
