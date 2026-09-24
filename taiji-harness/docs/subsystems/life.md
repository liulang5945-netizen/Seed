# Life

English | [中文](life.zh.md)

The Life subsystem is the Harness's window onto the Taiji local runtime — the process that runs beside the Host and owns the Seed native homeostasis organ, the native training loop, and, when its Legacy surface is mounted, the legacy life scheduler and the knowledge index. [dsh-api-life-controller](../../packages/api/life-controller) owns the Host `ctx.lifeController` service and the generated `ctx.remote.life` namespace; the browser reads that namespace through the Client `ctx.life` service and the reconnect-safe state stream the same package installs. The runtime stays the single source of truth: the controller polls it, carries each number through unchanged, and never synthesizes a value the runtime did not report.

Source: [`packages/api/life-controller/src/types.ts`](../../packages/api/life-controller/src/types.ts)

## Runtime sources

The controller reads the runtime's HTTP face in four tiers, and which tier a number came from is part of the data.

- Always-on reads: `GET /api/runtime/status` carries the health, memory, life, and training sections, `GET /api/train/checkpoints` carries the checkpoint roster, `GET /api/train/files` carries the trainable dataset roster (POSIX paths under the data directory with their sizes), and `GET /api/consolidation/status` carries the memory journal counts with the sleep pass's products — pass counter, latest corpus, data-ring spec, and latest report. All three tiers answer whenever the runtime process is up; a runtime that predates the consolidation surface answers `404` there, which the controller records as one `unavailable` line rather than a snapshot failure.
- Gated reads: `GET /api/life/status` (the Legacy scheduler) and `GET /api/rag/status` (the knowledge index) are mounted only while the runtime enables its Legacy surface through `SEED_ENABLE_LEGACY`. An unmounted gated path answers `404`, which the controller records as `disabled` — a source that was never there, not a runtime that broke.
- The training progress stream: `POST /api/train/native` answers a server-sent event stream whose `progress` events carry one `LifeProgressView` each, and whose `completed` or `error` event ends the run.
- The consolidation control: `POST /api/consolidate` runs one native sleep pass (analyse, project, specify, and only on request sleep the substrate) and answers its report as JSON.

The status section that holds the life numbers names its own organ, and that name decides the shape. `life.status === 'seed'` reports the native homeostasis organ and yields `LifeNativeView` (observation count, organ mode, and the open need and drive maps the runtime scales to 0..100); `life.status === 'ok'` reports the legacy scheduler and yields `LifeLegacyView` (its loop flag, current activity, dominant need, need map, heartbeat and event counters, and the last heartbeat and activity instants). Any other value leaves `life` absent. The controller never rescales a native homeostatic value and never substitutes a native reading for a legacy one: the two organs report on their own scales, and the consumer decides whether the numbers are comparable.

## The snapshot

`LifeSnapshot` is one complete reading, naming the organ that answered: `source` (which organ answered the life numbers), `observedAt` (the ISO-8601 instant this reading was taken), `fresh` (true only when the status read answered this cycle), `pollIntervalMs` (the cadence in force when the reading was taken), the optional `health`, `memory`, `life`, `knowledge`, and `consolidation` projections, the always-present `training` projection, `availability`, and `unavailable`.

`unavailable` holds one operator-readable line per source that did not answer, so a panel can say which part of the runtime is missing instead of showing a blank. A quantity nobody measured stays absent rather than defaulted to zero, because a permanent zero reads as a measured fact and hides that the source never answered.

`LifeAvailability` classifies each source independently: `runtime` is `ok` or `down`, `legacy` and `knowledge` are `ok`, `disabled`, or `down`, and `trainingStream` is `idle`, `streaming`, or `closed`. `LifeTrainingView` carries the runtime's training lock flag, the pause and stop requests it has not observed yet, the publish lock, the latest `LifeProgressView`, the checkpoint roster newest-first, truncated to the `maxCheckpoints` bound, and `datasets` — the trainable roster the panel offers a run (`path` plus the runtime's `size_bytes`), absent when that read did not answer.

## Remote methods

| Method | Kind | Behavior |
| --- | --- | --- |
| `snapshot` | unary | Returns the current `LifeSnapshot`; when the poll loop has none yet it reads one, and a runtime that did not answer yields a snapshot marked `down` with `fresh` false rather than a failure. |
| `follow` | stream | Opens with one `baseline` frame carrying the current snapshot, then one `snapshot` replacement frame per changed reading; a reconnect starts a new generation with a fresh baseline. |
| `trainStart` | unary | Starts a native run through the progress stream and resolves with the runtime's acceptance; a second run on the same Host rejects as `life/conflict`, and progress arrives through `follow`. |
| `trainPause` | unary | Asks the runtime to pause the running training and returns its message. |
| `trainResume` | unary | Asks the runtime to resume a paused training and returns its message. |
| `trainStop` | unary | Asks the runtime to stop after its current step and returns its message. |
| `trainReset` | unary | Forces the runtime to release a training lock it still holds and returns its message. |
| `consolidate` | unary | Runs one native sleep consolidation pass through `POST /api/consolidate` and returns the runtime's report message; the request's `reason` is optional and defaults to the runtime's own. |
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
 * Run one native sleep consolidation pass.
 * @param request - pass parameters; omitted fields keep the runtime's defaults.
 * @param signal - caller lifetime.
 * @returns the runtime's pass report message.
 */
@Remote async consolidate(request: LifeConsolidateRequest, signal: AbortSignal): Promise<LifeControlValue>

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