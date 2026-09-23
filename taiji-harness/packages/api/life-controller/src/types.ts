/**
 * Browser-safe vocabulary for the Taiji local-runtime life namespace: the
 * snapshot the Host polls from the runtime, the frames that carry it to a
 * panel, and the control verbs a panel issues back.
 *
 * Every field names something the runtime actually reports. A quantity nobody
 * measures stays absent rather than defaulted, because a permanent zero in the
 * contract reads as a fact and hides that the source never answered.
 */

declare module '@taiji/dsh-typert-protocol' {
  interface RemoteErrorDetailsMap {
    /** The runtime answered with a non-2xx status. */
    'life/runtime-error': { readonly status: number; readonly detail: string }
    /** The runtime never answered: connection, DNS, or timeout failure. */
    'life/runtime-unreachable': { readonly baseURL: string; readonly reason: string }
    /** The verb needs a source the running runtime does not serve. */
    'life/unavailable': { readonly source: 'life' | 'knowledge'; readonly reason: string }
    /** The runtime is in a state that forbids this verb. */
    'life/conflict': { readonly reason: string }
    /** The request is not a shape the runtime accepts. */
    'life/bad-request': { readonly field: string; readonly reason: string }
  }
}

/** Which organ answered the life numbers. */
export type LifeSource = 'native' | 'legacy' | 'absent'

/** Whether the runtime answered the status read. */
export type LifeRuntimeState = 'ok' | 'down'

/** Whether the gated Legacy surface answered. */
export type LifeLegacyState = 'ok' | 'disabled' | 'down'

/** Whether the gated knowledge surface answered. */
export type LifeKnowledgeState = 'ok' | 'disabled' | 'down'

/** How the current training progress stream stands. */
export type LifeTrainingStreamState = 'idle' | 'streaming' | 'closed'

/** Runtime health projection. */
export interface LifeHealthView {
  /** `ok`, `loading`, `downloading`, or `error`. */
  readonly state: string
  /** Whether a model is loaded. */
  readonly modelLoaded: boolean
  /** Loaded model name, empty when none is loaded. */
  readonly modelName: string
  /** Whether the Seed native runtime is active. */
  readonly seedActive: boolean
  /** Whether startup finished. */
  readonly startupComplete: boolean
}

/** Host memory projection. */
export interface LifeMemoryView {
  /** Total memory in GiB. */
  readonly totalGb: number
  /** Available memory in GiB. */
  readonly availableGb: number
  /** Used memory as a percentage. */
  readonly usedPct: number
}

/** Native homeostasis measurements, scaled by the runtime to 0..100. */
export interface LifeNativeView {
  /** Observation count the organ has advanced through. */
  readonly tick: number
  /** Organ mode, for example `wake`. */
  readonly mode: string
  /** Open need map; the runtime owns the vocabulary. */
  readonly needs: Readonly<Record<string, number>>
  /** Open drive map; the runtime owns the vocabulary. */
  readonly drives: Readonly<Record<string, number>>
}

/** Legacy scheduler measurements. */
export interface LifeLegacyView {
  /** Whether the scheduler loop is running. */
  readonly isRunning: boolean
  /** Current activity, for example `idle` or `feeding`. */
  readonly lifeState: string
  /** The need the scheduler acts on first, empty when none dominates. */
  readonly dominantNeed: string
  /** Open need map; the scheduler owns the vocabulary and its scale. */
  readonly needs: Readonly<Record<string, number>>
  /** Heartbeats the scheduler counted. */
  readonly totalHeartbeats: number
  /** Events the scheduler recorded. */
  readonly totalEvents: number
  /** ISO-8601 instant of the last heartbeat, absent before the first one. */
  readonly lastHeartbeat?: string
  /** ISO-8601 instant of the last activity, absent before the first one. */
  readonly lastActivity?: string
}

/** Life numbers from whichever organ answered. */
export interface LifeLifeView {
  /** Whether that organ reports itself running. */
  readonly isRunning: boolean
  /** Native measurements, present only when the native organ answered. */
  readonly native?: LifeNativeView
  /** Legacy measurements, present only when the Legacy scheduler answered. */
  readonly legacy?: LifeLegacyView
}

/** One training progress sample from the runtime's stream. */
export interface LifeProgressView {
  /** Completed fraction, 0..1. */
  readonly fraction: number
  /** Steps completed. */
  readonly step: number
  /** Window-mean loss, as the runtime computes it. */
  readonly loss: number
  /** Seconds elapsed. */
  readonly elapsed: number
  /** Estimated seconds remaining, absent while the runtime cannot estimate. */
  readonly eta?: number
  /** Current epoch. */
  readonly epoch: number
  /** Total epochs. */
  readonly totalEpochs: number
  /** Measured samples per second. */
  readonly samplesPerSec: number
  /** Total steps planned for this run. */
  readonly totalSteps: number
}

/** One checkpoints directory entry. */
export interface LifeCheckpointView {
  /** File name inside the runtime's checkpoint directory. */
  readonly filename: string
  /** Step the checkpoint was written at. */
  readonly step: number
  /** File size in bytes. */
  readonly bytes: number
  /** ISO-8601 file modification instant, empty when the runtime omits it. */
  readonly modifiedUtc: string
  /** ISO-8601 instant the runtime recorded for the save, empty when omitted. */
  readonly savedAtUtc: string
  /** Epochs the runtime reports for the checkpoint. */
  readonly numEpochs: number
}

/** Training state, progress, and the checkpoint roster. */
export interface LifeTrainingView {
  /** Whether a training run holds the runtime's training lock. */
  readonly isTraining: boolean
  /** Whether a pause has been requested and not yet observed. */
  readonly pauseRequested: boolean
  /** Whether a stop has been requested and not yet observed. */
  readonly stopRequested: boolean
  /** Whether a publish holds the runtime's publish lock. */
  readonly publishing: boolean
  /** Latest progress sample, present only while a stream reported one. */
  readonly progress?: LifeProgressView
  /** Checkpoint roster, newest first as the runtime returns it. */
  readonly checkpoints: readonly LifeCheckpointView[]
}

/** Knowledge base size, present only when the gated surface answered. */
export interface LifeKnowledgeView {
  /** Indexed documents. */
  readonly docCount: number
  /** Indexed chunks. */
  readonly chunkCount: number
  /** Whether embeddings exist. */
  readonly hasEmbeddings: boolean
  /** Embedding dimension, zero when the runtime reports none. */
  readonly embedDim: number
}

/** Availability of every source behind one snapshot. */
export interface LifeAvailability {
  /** Runtime status read. */
  readonly runtime: LifeRuntimeState
  /** Gated Legacy life surface. */
  readonly legacy: LifeLegacyState
  /** Gated knowledge surface. */
  readonly knowledge: LifeKnowledgeState
  /** Training progress stream. */
  readonly trainingStream: LifeTrainingStreamState
}

/** One complete reading of the runtime, with its provenance. */
export interface LifeSnapshot {
  /** Which organ answered the life numbers. */
  readonly source: LifeSource
  /** ISO-8601 instant this snapshot was observed. */
  readonly observedAt: string
  /** Whether the observation is recent enough to act on. */
  readonly fresh: boolean
  /** Poll interval in force when this snapshot was observed. */
  readonly pollIntervalMs: number
  /** Health projection, absent when the status read failed. */
  readonly health?: LifeHealthView
  /** Memory projection, absent when the status read failed. */
  readonly memory?: LifeMemoryView
  /** Life numbers, absent when neither organ answered. */
  readonly life?: LifeLifeView
  /** Training projection; empty defaults stand in until the first read. */
  readonly training: LifeTrainingView
  /** Knowledge projection, absent when the gated surface did not answer. */
  readonly knowledge?: LifeKnowledgeView
  /** Availability of every source behind this snapshot. */
  readonly availability: LifeAvailability
  /** One line per source that did not answer, in operator-readable form. */
  readonly unavailable: readonly string[]
}

/** Result of the single-snapshot read. */
export interface LifeSnapshotValue {
  /** The observed snapshot. */
  readonly snapshot: LifeSnapshot
}

/**
 * One frame of the Life state stream. The stream always opens with
 * `baseline`; every later frame replaces the whole snapshot, because a panel
 * renders the reading as a unit and a small snapshot makes field-level
 * increments pure overhead.
 */
export type LifeFollowFrame =
  | { readonly type: 'baseline'; readonly value: LifeSnapshot }
  | { readonly type: 'snapshot'; readonly value: LifeSnapshot }

/** Request to start a native training run. */
export interface LifeTrainStartRequest {
  /** Dataset file names inside the runtime's dataset directory; omitted uses the runtime's default corpus. */
  readonly datasets?: readonly string[]
  /** Parameter budget for the run; omitted uses the runtime's default. */
  readonly parameterBudget?: number
  /** Random seed; omitted uses the runtime's default. */
  readonly seed?: number
  /** Upper bound on symbols consumed; omitted uses the runtime's default. */
  readonly maxSymbols?: number
}

/** Request to force one Legacy life activity. */
export interface LifeActionRequest {
  /** Activity to force. */
  readonly action: 'feed' | 'sleep' | 'play'
  /** Operator-visible reason recorded with the activity. */
  readonly reason?: string
}

/** Result of an accepted control verb, in the runtime's own words. */
export interface LifeControlValue {
  /** Message the runtime returned for the accepted verb. */
  readonly message: string
}
