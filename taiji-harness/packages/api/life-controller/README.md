---
description: "Host and Client Life control: poll the Taiji local runtime, carry each reading with the organ that answered it, and issue its training and legacy control verbs."
kind: "package-reference"
---
# Life Controller

English | [中文](README.zh.md)

## Summary

`@taiji/dsh-api-life-controller` owns the `ctx.lifeController` service and the Client `ctx.remote.life` namespace. Its Remote methods read the Taiji local runtime — health, memory, each answering organ's life numbers, training state with its checkpoints, the knowledge index, and the memory journal with consolidation products — and carry the control verbs a life panel issues: training start, pause, resume, stop, and reset, one consolidation pass, and the gated legacy life start, stop, and action. The runtime stays the single source of truth: every reading carries its source, timestamp, freshness, and per-source availability, and the service never derives a value the runtime did not report.

## Table of Contents

- [Use this package](#use-this-package)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)
- [Dev Note](#dev-note)

-----

<a id="use-this-package"></a>
## Use this package

The Host controller owns one poll loop over the Taiji local runtime and serves every consumer from it. Each cycle reads the runtime once, stamps the interval in force, and publishes only when the rendering changed, so the [Life subsystem reference](../../../docs/subsystems/life.md) stays the single description of the readings and this README owns the package contract: configuration, the wire verbs, and their failures. `follow()` opens with one `baseline` frame carrying the current snapshot and then emits one `snapshot` replacement frame per change; a reconnect starts a new generation with a fresh baseline, so a consumer never depends on receiving every frame while disconnected.

The runtime is the single source of truth, and the reading says where every number came from. `GET /api/runtime/status` is always available and its `life` section selects its own organ — `seed` for the native homeostasis organ, `ok` for the legacy scheduler — while `GET /api/train/checkpoints` carries the roster of saved runs, `GET /api/train/files` carries the roster of trainable datasets (POSIX paths under the data directory with their sizes), `GET /api/artifacts` carries the publish state (which checkpoint answers and which settings names for the next start), and `GET /api/consolidation/status` carries the memory journal counts with the sleep pass's products (pass counter, latest corpus, data-ring spec, and latest report). The legacy model-publish APIs (`/api/model/publish`, `/api/model/published`, `/api/model/export_gguf`) are retired 410 tombstones — `POST /api/runtime/activate` over a platform-owned checkpoint is the publish surface. The legacy life surface (`/api/life`) and the knowledge index (`/api/rag`) are mounted only while the runtime enables them through `SEED_ENABLE_LEGACY`; an unmounted path answers `404`, which the controller records as `disabled` rather than as a failure, and the legacy control verbs under `/api/taiji/life` are refused as `life/unavailable` in the same situation. A consolidation read a running runtime does not serve becomes one `unavailable` line instead of a snapshot failure, because an older runtime is a fact about the deployment, not a broken reading. Every snapshot therefore carries `source`, `observedAt`, `fresh`, `availability`, and one `unavailable` line per source that did not answer, and a quantity nobody measured stays absent instead of defaulting to zero. The controller never rescales a native homeostatic value and never substitutes a native reading for a legacy one.

### Configuration

| Configuration | Default | Purpose |
| --- | --- | --- |
| `baseURL` | `http://127.0.0.1:8000` | Base URL of the Taiji local runtime |
| `pollIntervalMs` | `5000` | Interval between reads while nothing runs, in milliseconds (minimum `250`) |
| `activePollIntervalMs` | `2000` | Interval between reads while training holds the runtime, in milliseconds (minimum `250`) |
| `requestTimeoutMs` | `2000` | Maximum duration of one runtime request, in milliseconds (minimum `1`) |
| `maxCheckpoints` | `20` | Checkpoint rows carried into one snapshot (minimum `1`) |

The active cadence applies while the runtime reports training or a progress stream is open. `requestTimeoutMs` bounds each read and each control request separately, and a request that exceeds it raises `life/runtime-unreachable`. Every control verb re-reads once after the runtime accepts it, so a panel sees the effect of its own action without waiting for the next cycle.

### Remote methods

| Method | Kind | Purpose | Failures |
| --- | --- | --- | --- |
| `snapshot` | unary | Read the current snapshot; a runtime that did not answer yields a snapshot marked `down` with `fresh` false. | `life/runtime-unreachable` only for an abort that is not the caller's. |
| `follow` | stream | Open a generation with a `baseline` frame, then replace the snapshot per change. | Same as `snapshot`; a lost carrier is retried by the Client stream. |
| `trainStart` | unary | Start a native run over the progress stream and resolve with the runtime's acceptance. | `life/conflict` when a run already streams on this Host, plus the runtime's refusal codes. |
| `trainResumeCheckpoint` | unary | Continue training from a saved checkpoint over the same progress stream — corpus-drift warnings included — and resolve with the runtime's acceptance. | `life/conflict` when a run already streams on this Host, plus the runtime's refusal codes. |
| `trainPause` | unary | Ask the runtime to pause the running training. | `life/runtime-error`, `life/conflict`, `life/bad-request`, `life/runtime-unreachable`. |
| `trainResume` | unary | Ask the runtime to resume a paused training. | `life/runtime-error`, `life/conflict`, `life/bad-request`, `life/runtime-unreachable`. |
| `trainStop` | unary | Ask the runtime to stop after its current step. | `life/runtime-error`, `life/conflict`, `life/bad-request`, `life/runtime-unreachable`. |
| `trainReset` | unary | Force the runtime to release a training lock it still holds. | `life/runtime-error`, `life/conflict`, `life/bad-request`, `life/runtime-unreachable`. |
| `consolidate` | unary | Run one native sleep consolidation pass and return the runtime's report message; the request's `reason` is optional and defaults to the runtime's own. | `life/runtime-error`, `life/conflict`, `life/bad-request`, `life/runtime-unreachable`. |
| `activateCheckpoint` | unary | Answer later turns from a platform-owned checkpoint (`POST /api/runtime/activate`); the empty id activates the built-in seed. | `life/runtime-error` — a missing or unloadable checkpoint is the runtime's own refusal, `life/bad-request` for a name outside the checkpoint directory. |
| `lifeStart` | unary | Start the gated legacy life scheduler. | `life/unavailable` when the legacy surface is not mounted, plus the runtime's refusal codes. |
| `lifeStop` | unary | Stop the gated legacy life scheduler. | `life/unavailable` when the legacy surface is not mounted, plus the runtime's refusal codes. |
| `lifeAction` | unary | Force one gated legacy activity — `feed`, `sleep`, or `play` — with an operator-visible reason. | `life/bad-request` for another activity name, `life/unavailable` when the legacy surface is not mounted. |

The Client entry installs `ctx.life`, the `ILife` facade, and one reconnecting state stream. `ClientLifeModel` keeps the latest snapshot behind one identity-stable state object (`loading`, `ready`, or `error` with the last snapshot retained) and hands out `getSnapshot()` plus `subscribe()`; `LifeClient` unwraps each Host result and raises `LifeControlError` carrying the structured `rpcError` on refusal.

<a id="model-experience"></a>
## Model Experience

None, as the Life subsystem reads local-runtime control state for a panel and registers no prompt, tool, or session event.

#### KV Cache effect

No direct effect; Life reads and control verbs never alter model requests.

## Known Limitations and Deferred Work

<a id="known-limitations-and-deferred-work"></a>

- The controller only polls: it never subscribes to a runtime push channel, so a change becomes visible on the next cycle rather than when it happens.
- `follow()` replaces the whole snapshot instead of emitting field-level increments, so a consumer that wants a delta compares consecutive frames itself.
- Without `SEED_ENABLE_LEGACY` the legacy life and knowledge rows are absent from every snapshot rather than empty, so a panel reads `availability` instead of assuming zero.
- Training progress can lag: after the progress stream drops, the snapshot keeps the last sample it saw and reports the stream as `closed`.
- A consolidation pass shares `requestTimeoutMs` with every other control request: a slower pass keeps running on the runtime while the verb reports the timeout, so the panel re-reads to see the products the pass still wrote.

<a id="dev-note"></a>
### Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

The runtime's HTTP face is the only contract this package speaks: the paths are fixed, and the runtime owns their payload vocabulary. A runtime field this package does not name stays unread rather than defaulted.

</details>

**Runtime invariant:** No companion is published. The Taiji runtime owns every measured value; the controller polls it and carries each reading together with the organ that answered it.