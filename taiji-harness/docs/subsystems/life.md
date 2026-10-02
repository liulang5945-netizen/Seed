# Life

English | [中文](life.zh.md)

The Life subsystem is the Harness's window onto the Taiji local runtime — the process that runs beside the Host and owns the Seed native homeostasis organ, the native training loop, and, when its Legacy surface is mounted, the legacy life scheduler and the knowledge index. [dsh-api-life-controller](../../packages/api/life-controller) owns the Host `ctx.lifeController` service and the generated `ctx.remote.life` namespace; the browser reads that namespace through the Client `ctx.life` service and the reconnect-safe state stream the same package installs. The runtime stays the single source of truth: the controller polls it, carries each number through unchanged, and never synthesizes a value the runtime did not report.

Source: [`packages/api/life-controller/src/types.ts`](../../packages/api/life-controller/src/types.ts)

## Runtime sources

The controller reads the runtime's HTTP face in four tiers, and which tier a number came from is part of the data.

- Always-on reads: `GET /api/runtime/status` carries the health, memory, life, training, `tools` (workbench capability snapshot) and `auth` (runtime authentication) sections, `GET /api/train/checkpoints` carries the checkpoint roster, `GET /api/train/files` carries the trainable dataset roster (POSIX paths under the data directory with their sizes), `GET /api/artifacts` carries the publish state — which checkpoint answers and which settings names for the next start — and `GET /api/consolidation/status` carries the memory journal counts with the sleep pass's products — pass counter, latest corpus, data-ring spec, and latest report. All tiers answer whenever the runtime process is up; a runtime that predates the consolidation surface answers `404` there, which the controller records as one `unavailable` line rather than a snapshot failure.
- Gated reads: `GET /api/life/status` (the Legacy scheduler) and `GET /api/rag/status` (the knowledge index) are mounted only while the runtime enables its Legacy surface through `SEED_ENABLE_LEGACY`; `GET /api/rag/files` (the mounted documents with their index state) is read only while that surface answered, so a disabled knowledge base stays one `disabled` fact. An unmounted gated path answers `404`, which the controller records as `disabled` — a source that was never there, not a runtime that broke.
- The training progress stream: `POST /api/train/native` and `POST /api/train/resume_checkpoint` (continuing from a saved checkpoint, datasets and a tick cap included) answer one shared server-sent event stream whose `progress` events carry one `LifeProgressView` each and whose `warning` events — a corpus-drift notice on a resumed run, for one — the controller folds into the snapshot's `training.warnings`; a `completed` or `error` event ends the run.
- The consolidation control: `POST /api/consolidate` runs one native sleep pass (analyse, project, specify, and only on request sleep the substrate) and answers its report as JSON.
- The activation control: `POST /api/runtime/activate` answers later turns from a platform-owned checkpoint — the empty name selects the built-in seed — and persists the choice in settings. The retired `/api/model/publish|published|export_gguf` APIs answer `410`; activation over a checkpoint is the publish surface.
- The resource controls: `POST /api/train/upload_dataset` and `POST /api/rag/upload` carry one picked file into the runtime's data or document directory as multipart bytes, and `DELETE /api/train/file/{path}`, `DELETE /api/train/checkpoint/{filename}`, and `DELETE /api/rag/file/{name}` remove one dataset, checkpoint, or document. The checkpoint route refuses the active and the configured checkpoint with `409`, and the dataset route answers a vanished file with HTTP 200 whose own body says `error`, which the controller raises as the runtime's failure rather than reading as success.

The status section that holds the life numbers names its own organ, and that name decides the shape. `life.status === 'seed'` reports the native homeostasis organ and yields `LifeNativeView` (observation count, organ mode, and the open need and drive maps the runtime scales to 0..100); `life.status === 'ok'` reports the legacy scheduler and yields `LifeLegacyView` (its loop flag, current activity, dominant need, need map, heartbeat and event counters, and the last heartbeat and activity instants). Any other value leaves `life` absent. The controller never rescales a native homeostatic value and never substitutes a native reading for a legacy one: the two organs report on their own scales, and the consumer decides whether the numbers are comparable.

## The snapshot

`LifeSnapshot` is one complete reading, naming the organ that answered: `source` (which organ answered the life numbers), `observedAt` (the ISO-8601 instant this reading was taken), `fresh` (true only when the status read answered this cycle), `pollIntervalMs` (the cadence in force when the reading was taken), the optional `health`, `memory`, `life`, `knowledge`, `consolidation`, `artifacts`, `workbench`, and `auth` projections, the always-present `training` projection, `availability`, and `unavailable`. A status section the running runtime does not include leaves its projection absent rather than zero-filled.

`unavailable` holds one operator-readable line per source that did not answer, so a panel can say which part of the runtime is missing instead of showing a blank. A quantity nobody measured stays absent rather than defaulted to zero, because a permanent zero reads as a measured fact and hides that the source never answered.

`LifeAvailability` classifies each source independently: `runtime` is `ok` or `down`, `legacy` and `knowledge` are `ok`, `disabled`, or `down`, and `trainingStream` is `idle`, `streaming`, or `closed`. `LifeTrainingView` carries the runtime's training lock flag, the pause and stop requests it has not observed yet, the publish lock, the latest `LifeProgressView`, the checkpoint roster newest-first, truncated to the `maxCheckpoints` bound, `datasets` — the trainable roster the panel offers a run (`path` plus the runtime's `size_bytes`), absent when that read did not answer — and `warnings`, the runtime's own messages from this Host's last run, cleared when a new run is accepted and kept after it settles. `LifeKnowledgeView` carries the index size and `files` — the mounted documents with their sizes and `indexed` or `pending` state, absent when the file-list read did not answer.

## Remote methods

| Method | Kind | Behavior |
| --- | --- | --- |
| `snapshot` | unary | Returns the current `LifeSnapshot`; when the poll loop has none yet it reads one, and a runtime that did not answer yields a snapshot marked `down` with `fresh` false rather than a failure. |
| `follow` | stream | Opens with one `baseline` frame carrying the current snapshot, then one `snapshot` replacement frame per changed reading; a reconnect starts a new generation with a fresh baseline. |
| `trainStart` | unary | Starts a native run through the progress stream and resolves with the runtime's acceptance; a second run on the same Host rejects as `life/conflict`, and progress arrives through `follow`. |
| `trainResumeCheckpoint` | unary | Continues training from a saved checkpoint over the same progress stream — accepting its datasets and tick cap, and carrying the runtime's corpus-drift warnings into `training.warnings`; a second run on the same Host rejects as `life/conflict`. |
| `trainPause` | unary | Asks the runtime to pause the running training and returns its message. |
| `trainResume` | unary | Asks the runtime to resume a paused training and returns its message. |
| `trainStop` | unary | Asks the runtime to stop after its current step and returns its message. |
| `trainReset` | unary | Forces the runtime to release a training lock it still holds and returns its message. |
| `uploadDataset` | unary | Moves one picked file into the runtime's data directory through `POST /api/train/upload_dataset` as multipart bytes; the Host reduces the name to its basename, checks its trainable suffix, and refuses a payload past the upload budget before anything travels. |
| `deleteDataset` | unary | Removes one dataset through `DELETE /api/train/file/{path}`; the Host checks the path to stay a relative roster path with a trainable suffix, and a body that says `error` — the route's answer for a vanished file — is raised as the runtime's failure. |
| `deleteCheckpoint` | unary | Removes one checkpoint through `DELETE /api/train/checkpoint/{filename}`; the Host checks a flat `*.pt` name, and the runtime refuses the active and the configured checkpoint with `409` (`life/conflict`). |
| `uploadKnowledge` | unary | Moves one picked document into the runtime's document directory through `POST /api/rag/upload` as multipart bytes; the runtime vectorizes it in the background. |
| `deleteKnowledge` | unary | Removes one document through `DELETE /api/rag/file/{name}`; the Host checks a flat file name, so a delete acts on exactly the file it names. |
| `consolidate` | unary | Runs one native sleep consolidation pass through `POST /api/consolidate` and returns the runtime's report message; the request's `reason` is optional and defaults to the runtime's own. |
| `activateCheckpoint` | unary | Answers later turns from a platform-owned checkpoint through `POST /api/runtime/activate` — the empty name activates the built-in seed — and persists the choice in settings; a missing or unloadable checkpoint is the runtime's own `life/runtime-error` refusal. |
| `lifeStart` | unary | Starts the gated Legacy life scheduler and returns its message. |
| `lifeStop` | unary | Stops the gated Legacy life scheduler and returns its message. |
| `lifeAction` | unary | Forces one gated Legacy activity — `feed`, `sleep`, or `play` — with an operator-visible reason, and returns the runtime's message. |

Every control verb re-reads once after the runtime accepts it, so a panel sees the effect of its own action without waiting for the next poll. The `follow` stream serves every consumer from one poll loop: each cycle reads the runtime once, stamps the interval in force, and publishes only when the rendering changed, so the active cadence applies while training holds the runtime.

## Failures

A Life failure is one `RemoteError` with a stable code; the details carry what tells the cases apart.

| Code | Details | Raised when |
| --- | --- | --- |
| `life/runtime-unreachable` | `baseURL`, `reason` | The runtime never answered: a connection, DNS, or timeout failure. |
| `life/runtime-error` | `status`, `detail` | The runtime answered with a non-2xx status a verb cannot use; `detail` is the runtime's own text. |
| `life/unavailable` | `source`, `reason` | A gated source the running runtime does not serve — an unmounted Legacy life or knowledge surface, or a Legacy control verb refused with `404`. |
| `life/conflict` | `reason` | The runtime state forbids the verb: a `409` from the runtime, or a `trainStart` while a progress stream is already open on this Host. |
| `life/bad-request` | `field`, `reason` | The request is not a shape the runtime accepts: a `400` or `422` from the runtime, or a `lifeAction` naming an activity outside `feed`, `sleep`, and `play`. |

## Client projection

The Client entry installs `ctx.life`, the `ILife` facade, and one reconnecting state stream. `ClientLifeModel` owns the latest snapshot behind one identity-stable state object (`loading`, `ready`, or `error` with the last snapshot retained) and hands out `getSnapshot()` plus `subscribe()` for a renderer; `LifeClient` unwraps each Host result and raises `LifeControlError` with the structured `rpcError` on refusal. `createLifeStateStream()` wraps the Gateway's `RemoteSnapshotStream` over `ctx.remote.life.follow`: a `baseline` frame replaces all state, every later frame replaces the snapshot, and a lost carrier is retried by the Gateway-owned stream rather than by this package; a failure that is not already a Remote failure is normalized to `life/stream-failed`.

<!-- BEGIN GENERATED cordis-surface (gen-cordis-catalog.ts) — do not edit between markers -->

<a id="cordis-surface"></a>

## Cordis API

Generated from source by `scripts/gen-cordis-catalog.ts` (verified fresh by `pnpm run verify-cordis-catalog` in doc-sync; regenerate with `pnpm run gen-cordis-catalog`) — the language sides differ only in locale-specific paired document paths. Signature blocks use a `ts cordis-catalog` fence and keep the original source JSDoc; dispatch modes are defined in the [primer](../cordis-primer.md#dispatch-modes), and the framework-inherited `ctx` API lives in [cordis-api/inherited.md](../cordis-api/inherited.md).

<a id="ctxlifecontroller--lifecontroller"></a>

### `ctx.lifeController` — `LifeController`

Host service backing the generated `ctx.remote.life` namespace.

```ts cordis-catalog
/**
 * Read the current runtime snapshot.
 * @param signal - caller lifetime.
 * @returns the snapshot, marked `down` when the runtime did not answer.
 */
@Remote async snapshot(signal: AbortSignal): Promise<LifeSnapshotValue>

/**
 * Stream the current snapshot first, then every change.
 * @param signal - stream lifetime.
 * @returns the opening snapshot followed by replacement frames.
 */
@Remote({ mode: 'stream' }) follow(signal: AbortSignal): AsyncIterable<LifeFollowFrame>

/**
 * Start a native training run and fold its progress into the snapshot stream.
 * @param request - run parameters; omitted fields keep the runtime's defaults.
 * @returns the runtime's acceptance message; progress arrives through `follow`.
 */
@Remote async trainStart(request: LifeTrainStartRequest): Promise<LifeControlValue>

/**
 * Continue training from a saved checkpoint and fold its progress into the
 * snapshot stream, carrying the runtime's corpus-drift warnings through.
 * @param request - checkpoint name and optional datasets and tick cap.
 * @returns the runtime's acceptance message; progress arrives through `follow`.
 */
@Remote async trainResumeCheckpoint(request: LifeResumeCheckpointRequest): Promise<LifeControlValue>

/**
 * Pause the running training.
 * @param signal - caller lifetime.
 * @returns the runtime's message.
 */
@Remote async trainPause(signal: AbortSignal): Promise<LifeControlValue>

/**
 * Resume a paused training.
 * @param signal - caller lifetime.
 * @returns the runtime's message.
 */
@Remote async trainResume(signal: AbortSignal): Promise<LifeControlValue>

/**
 * Stop the running training after its current step.
 * @param signal - caller lifetime.
 * @returns the runtime's message.
 */
@Remote async trainStop(signal: AbortSignal): Promise<LifeControlValue>

/**
 * Force-release the training lock the runtime still holds.
 * @param signal - caller lifetime.
 * @returns the runtime's message.
 */
@Remote async trainReset(signal: AbortSignal): Promise<LifeControlValue>

/**
 * Upload one dataset file into the runtime's data directory. The name is
 * reduced to its basename and checked against the runtime's trainable
 * suffixes before any bytes leave the Host; the runtime stays the final
 * authority on what it stores.
 * @param request - picked file name and the file's bytes as base64.
 * @param signal - caller lifetime.
 * @returns the runtime's message naming the uploaded dataset.
 */
@Remote async uploadDataset(request: LifeUploadDatasetRequest, signal: AbortSignal): Promise<LifeControlValue>

/**
 * Delete one dataset file the roster lists. The path is checked to stay a
 * relative roster path with a trainable suffix before the runtime is asked;
 * the runtime's data directories are the only places it may resolve.
 * @param request - POSIX path relative to the runtime's data directory.
 * @param signal - caller lifetime.
 * @returns the runtime's acknowledgement.
 */
@Remote async deleteDataset(request: LifeDeleteDatasetRequest, signal: AbortSignal): Promise<LifeControlValue>

/**
 * Delete one checkpoint; the runtime refuses the active and the configured
 * checkpoint with its own conflict, because removing either breaks the
 * answering model or the next start.
 * @param request - file name inside the runtime's checkpoint directory.
 * @param signal - caller lifetime.
 * @returns the runtime's message naming the deleted checkpoint.
 */
@Remote async deleteCheckpoint(request: LifeDeleteCheckpointRequest, signal: AbortSignal): Promise<LifeControlValue>

/**
 * Upload one knowledge document into the runtime's document directory; the
 * name is reduced to its basename and checked before any bytes leave the
 * Host, and the runtime vectorizes the file in the background.
 * @param request - picked file name and the file's bytes as base64.
 * @param signal - caller lifetime.
 * @returns the runtime's message naming the uploaded document.
 */
@Remote async uploadKnowledge(request: LifeUploadKnowledgeRequest, signal: AbortSignal): Promise<LifeControlValue>

/**
 * Delete one knowledge document the file list shows; the runtime removes it
 * from the index as well.
 * @param request - file name inside the runtime's document directory.
 * @param signal - caller lifetime.
 * @returns the runtime's acknowledgement.
 */
@Remote async deleteKnowledge(request: LifeDeleteKnowledgeRequest, signal: AbortSignal): Promise<LifeControlValue>

/**
 * Run one native sleep consolidation pass.
 * @param request - pass parameters; omitted fields keep the runtime's defaults.
 * @param signal - caller lifetime.
 * @returns the runtime's pass report message.
 */
@Remote async consolidate(request: LifeConsolidateRequest, signal: AbortSignal): Promise<LifeControlValue>

/**
 * Answer later turns from a platform-owned checkpoint; the empty id activates
 * the built-in seed. A failure is the runtime's own refusal — activation
 * swaps the model every later turn runs through.
 * @param request - checkpoint name inside the runtime's checkpoint directory.
 * @param signal - caller lifetime.
 * @returns the runtime's message naming what became active.
 */
@Remote async activateCheckpoint(request: LifeActivateRequest, signal: AbortSignal): Promise<LifeControlValue>

/**
 * Start the Legacy life scheduler.
 * @param signal - caller lifetime.
 * @returns the runtime's message.
 */
@Remote async lifeStart(signal: AbortSignal): Promise<LifeControlValue>

/**
 * Stop the Legacy life scheduler.
 * @param signal - caller lifetime.
 * @returns the runtime's message.
 */
@Remote async lifeStop(signal: AbortSignal): Promise<LifeControlValue>

/**
 * Force one Legacy life activity.
 * @param request - activity and operator-visible reason.
 * @param signal - caller lifetime.
 * @returns the runtime's message.
 */
@Remote async lifeAction(request: LifeActionRequest, signal: AbortSignal): Promise<LifeControlValue>
```

Source: [`packages/api/life-controller/src/index.ts`](../../packages/api/life-controller/src/index.ts)
<!-- END GENERATED cordis-surface -->