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
    /** A value the namespace could not carry as a Remote failure at all. */
    'life/stream-failed': { readonly reason: string }
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

/** One trainable dataset file as `GET /api/train/files` lists it. */
export interface LifeDatasetView {
  /** POSIX path relative to the runtime's data directory. */
  readonly path: string
  /** File size in bytes. */
  readonly sizeBytes: number
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
  /** Trainable dataset roster, absent when the roster read did not answer. */
  readonly datasets?: readonly LifeDatasetView[]
  /**
   * Warnings the runtime's own progress stream carried for this Host's last
   * run — a corpus-drift notice on a resumed run is the case that matters.
   * Absent while no run reported one; a new run clears the list.
   */
  readonly warnings?: readonly string[]
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

/** Memory journal counts, as the consolidation status reports them. */
export interface LifeJournalView {
  /** Entries the journal holds. */
  readonly entries: number
  /** Entry counts by kind; the runtime owns the vocabulary. */
  readonly byKind: Readonly<Record<string, number>>
  /** Sessions the journal has seen. */
  readonly sessions: number
  /** Unix seconds of the newest entry, zero while the journal is empty. */
  readonly lastRecordedAt: number
}

/** The data ring's product as the runtime last wrote it. */
export interface LifeSpecView {
  /** Why the readiness gate passed, in the runtime's own words. */
  readonly reason: string
  /** Datasets the spec names, ready to hand to a training request. */
  readonly datasets: readonly string[]
  /** Weaknesses the pass measured, in the runtime's own words. */
  readonly weaknesses: readonly string[]
}

/** The newest sleep pass's own report, in the runtime's words. */
export interface LifePassReportView {
  /** What triggered the pass. */
  readonly reason: string
  /** Why the data ring was ready, or what held it back. */
  readonly specReason: string
  /** How long the pass took, in milliseconds. */
  readonly durationMs: number
  /** Weaknesses the pass measured. */
  readonly weaknesses: readonly string[]
  /** What this pass could not measure, stated instead of guessed. */
  readonly notes: readonly string[]
}

/** Native sleep consolidation and the memory journal it rehearses. */
export interface LifeConsolidationView {
  /** Sleep passes completed. */
  readonly passes: number
  /** Unix seconds of the last pass, zero when none has run. */
  readonly lastPassAt: number
  /** Dataset-relative name of the newest corpus, empty when none was written. */
  readonly lastCorpus: string
  /** Distinct texts rolled forward across passes. */
  readonly projectedDigests: number
  /** Whether a pass is running right now. */
  readonly running: boolean
  /** The spec file as it stands, null while the readiness gate has never passed. */
  readonly spec: LifeSpecView | null
  /** The newest pass's report, null before the first pass. */
  readonly lastReport: LifePassReportView | null
  /** The journal this pass reads from, counted live. */
  readonly journal: LifeJournalView
}

/** Where the running model came from, as `GET /api/artifacts` reports it. */
export interface LifeArtifactsView {
  /** Checkpoint the live runtime answers with; empty means the built-in seed. */
  readonly activeId: string
  /** Checkpoint settings name after the next start; empty means the built-in seed. */
  readonly configuredId: string
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

/** One complete reading of the runtime, naming the organ that answered. */
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
  /** Memory and consolidation projection, absent when the runtime does not serve it. */
  readonly consolidation?: LifeConsolidationView
  /** Model publish surface: which checkpoint answers, and which settings names. */
  readonly artifacts?: LifeArtifactsView
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

/** Request to run one native sleep consolidation pass. */
export interface LifeConsolidateRequest {
  /** Operator-visible reason recorded with the pass; omitted uses the runtime's default. */
  readonly reason?: string
}

/** Request to continue training from a saved checkpoint. */
export interface LifeResumeCheckpointRequest {
  /** Checkpoint file name inside the runtime's checkpoint directory. */
  readonly checkpoint: string
  /** Dataset paths relative to the data directory; omitted trains the runtime default corpus. */
  readonly datasets?: readonly string[]
  /** Upper bound on ticks added; omitted runs the selected corpus through once. */
  readonly maxTicks?: number
}

/** Request to answer later turns from a platform-owned checkpoint. */
export interface LifeActivateRequest {
  /** Checkpoint name inside the runtime's checkpoint directory; the empty string activates the built-in seed. */
  readonly checkpointId: string
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
