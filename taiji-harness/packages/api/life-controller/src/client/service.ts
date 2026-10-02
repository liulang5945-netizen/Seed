/** React-free Client Life service and command facade. */

import { Service, type Context } from '@taiji/cordis'
import type { RemoteFailure, RemoteResult } from '@taiji/dsh-typert-protocol'
import type {
  LifeActionRequest,
  LifeActivateRequest,
  LifeConsolidateRequest,
  LifeControlValue,
  LifeDeleteCheckpointRequest,
  LifeDeleteDatasetRequest,
  LifeDeleteKnowledgeRequest,
  LifeResumeCheckpointRequest,
  LifeSnapshot,
  LifeTrainStartRequest,
  LifeUploadDatasetRequest,
  LifeUploadKnowledgeRequest,
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
  /**
   * Read the identity-stable current snapshot state.
   * @returns the current snapshot state.
   */
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
   * @returns the runtime's acceptance message.
   */
  trainStart(request?: LifeTrainStartRequest): Promise<LifeControlValue>
  /**
   * Continue training from a saved checkpoint; progress and corpus-drift
   * warnings arrive through the snapshot stream.
   * @param request - checkpoint name and optional datasets and tick cap.
   * @returns the runtime's acceptance message.
   */
  trainResumeCheckpoint(request: LifeResumeCheckpointRequest): Promise<LifeControlValue>
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
   * Upload one dataset file into the runtime's data directory. The Host
   * reduces the name to its basename and forwards the bytes to the runtime.
   * @param request - file name and the file's bytes as base64.
   * @param signal - caller lifetime.
   * @returns the runtime's message naming the uploaded dataset.
   */
  uploadDataset(request: LifeUploadDatasetRequest, signal?: AbortSignal): Promise<LifeControlValue>
  /**
   * Delete one dataset file from the runtime's data directory. The Host checks
   * the path to stay a relative roster path before the runtime is asked.
   * @param request - POSIX path relative to the data directory.
   * @param signal - caller lifetime.
   * @returns the runtime's acknowledgement.
   */
  deleteDataset(request: LifeDeleteDatasetRequest, signal?: AbortSignal): Promise<LifeControlValue>
  /**
   * Delete one checkpoint; the runtime refuses the active and the configured
   * checkpoint, because removing either breaks the answering model or the
   * next start.
   * @param request - file name inside the checkpoint directory.
   * @param signal - caller lifetime.
   * @returns the runtime's message naming the deleted checkpoint.
   */
  deleteCheckpoint(request: LifeDeleteCheckpointRequest, signal?: AbortSignal): Promise<LifeControlValue>
  /**
   * Upload one knowledge document into the runtime's document directory; the
   * Host reduces the name to its basename and the runtime vectorizes the
   * document in the background.
   * @param request - file name and the file's bytes as base64.
   * @param signal - caller lifetime.
   * @returns the runtime's message naming the uploaded document.
   */
  uploadKnowledge(request: LifeUploadKnowledgeRequest, signal?: AbortSignal): Promise<LifeControlValue>
  /**
   * Delete one knowledge document; the runtime removes it from the index too.
   * @param request - file name inside the document directory.
   * @param signal - caller lifetime.
   * @returns the runtime's acknowledgement.
   */
  deleteKnowledge(request: LifeDeleteKnowledgeRequest, signal?: AbortSignal): Promise<LifeControlValue>
  /**
   * Run one native sleep consolidation pass.
   * @param request - pass parameters; omitted fields keep the runtime's defaults.
   * @param signal - caller lifetime.
   * @returns the runtime's pass report message.
   */
  consolidate(request?: LifeConsolidateRequest, signal?: AbortSignal): Promise<LifeControlValue>
  /**
   * Answer later turns from a platform-owned checkpoint; the empty id
   * activates the built-in seed.
   * @param request - checkpoint name inside the runtime's checkpoint directory.
   * @param signal - caller lifetime.
   * @returns the runtime's message naming what became active.
   */
  activateCheckpoint(request: LifeActivateRequest, signal?: AbortSignal): Promise<LifeControlValue>
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

  async trainStart(request: LifeTrainStartRequest = {}): Promise<LifeControlValue> {
    return await this.unwrap(this.model.trainStart(request))
  }

  async trainResumeCheckpoint(request: LifeResumeCheckpointRequest): Promise<LifeControlValue> {
    return await this.unwrap(this.model.trainResumeCheckpoint(request))
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

  async uploadDataset(request: LifeUploadDatasetRequest, signal?: AbortSignal): Promise<LifeControlValue> {
    return await this.unwrap(this.model.uploadDataset(request, signal))
  }

  async deleteDataset(request: LifeDeleteDatasetRequest, signal?: AbortSignal): Promise<LifeControlValue> {
    return await this.unwrap(this.model.deleteDataset(request, signal))
  }

  async deleteCheckpoint(request: LifeDeleteCheckpointRequest, signal?: AbortSignal): Promise<LifeControlValue> {
    return await this.unwrap(this.model.deleteCheckpoint(request, signal))
  }

  async uploadKnowledge(request: LifeUploadKnowledgeRequest, signal?: AbortSignal): Promise<LifeControlValue> {
    return await this.unwrap(this.model.uploadKnowledge(request, signal))
  }

  async deleteKnowledge(request: LifeDeleteKnowledgeRequest, signal?: AbortSignal): Promise<LifeControlValue> {
    return await this.unwrap(this.model.deleteKnowledge(request, signal))
  }

  async consolidate(request: LifeConsolidateRequest = {}, signal?: AbortSignal): Promise<LifeControlValue> {
    return await this.unwrap(this.model.consolidate(request, signal))
  }

  async activateCheckpoint(request: LifeActivateRequest, signal?: AbortSignal): Promise<LifeControlValue> {
    return await this.unwrap(this.model.activateCheckpoint(request, signal))
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
