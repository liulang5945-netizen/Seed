/** Client Life runtime: state model, command facade, and the reconnecting stream. */

import type { Context } from '@taiji/cordis'
import {
  RemoteSnapshotStream,
  RemoteStreamCarrierError,
  type ClientRemote,
} from '@taiji/dsh-api-gateway/client'
import type { LifeFollowFrame } from '../types.ts'
import { ClientLifeModel, type LifeStreamSink } from './model.ts'
import { LifeClient } from './service.ts'

export { ClientLifeModel, failureOf } from './model.ts'
export type { LifeRemote, LifeSnapshotState, LifeStreamSink } from './model.ts'
export { LifeClient, LifeControlError } from './service.ts'
export type { ILife } from './service.ts'
export type * from '../types.ts'

type LifeBaselineFrame = Extract<LifeFollowFrame, { type: 'baseline' }>
type LifeSnapshotFrame = Extract<LifeFollowFrame, { type: 'snapshot' }>

/** Gateway-owned snapshot stream configured for Life state. */
export type LifeStateStream = RemoteSnapshotStream<LifeBaselineFrame, LifeSnapshotFrame>

/** Required Client Remote services. */
export const inject = ['remote', 'remote.life']

declare module '@taiji/cordis' {
  interface Context {
    /** React-free Client Life state and control verbs. */
    life: import('./service.ts').ILife
  }
}

/**
 * Install Client Life state, commands, and reconnecting follow control.
 * @param ctx - Client root Context.
 */
export function apply(ctx: Context): void {
  const model = new ClientLifeModel(ctx.remote.life)
  new LifeClient(ctx, model)
  const control = createLifeStateStream(ctx.remote, {
    accept: model,
    carrierFailed: () => {
      model.handleStreamFailure(new RemoteStreamCarrierError('Life state stream carrier lost'))
    },
    failed: (error) => { model.handleStreamFailure(error) },
  })
  control.start()
  ctx.effect(
    () => async () => { await control.dispose() },
    'life-controller.client.control',
  )
}

/** State destinations used by the Life state stream. */
export interface LifeStateStreamOptions {
  /** Destination for decoded Life snapshots. */
  readonly accept: LifeStreamSink
  /** Observe a retryable carrier loss before reconnection. */
  readonly carrierFailed?: (error: RemoteStreamCarrierError) => void
  /** Publish a terminal business or protocol failure. */
  readonly failed: (error: unknown) => void
}

/**
 * Create the reconnecting Life state stream.
 * @param remote - Client Remote face carrying the life namespace and the stream factory.
 * @param options - Life state destinations.
 * @returns an unstarted stream owned by the Client Life runtime.
 */
export function createLifeStateStream(
  remote: ClientRemote,
  options: LifeStateStreamOptions,
): LifeStateStream {
  const stream = remote.$stream<LifeFollowFrame>({
    name: 'Life state stream',
    open: signal => remote.life.follow(signal),
    ended: accepted => accepted
      ? new RemoteStreamCarrierError('Life state stream ended without a terminal result')
      : new Error('Life state stream ended before its opening snapshot'),
    ...(options.carrierFailed === undefined ? {} : { carrierFailed: options.carrierFailed }),
  })
  return new RemoteSnapshotStream<LifeBaselineFrame, LifeSnapshotFrame>(stream, {
    name: 'Life state stream',
    isSnapshot: (frame): frame is LifeBaselineFrame => frame.type === 'baseline',
    replace: (frame) => { options.accept.replaceBaseline(frame.value) },
    update: (frame) => { options.accept.replaceSnapshot(frame.value) },
    failed: options.failed,
  })
}
