# Life（生命子系统）

[English](life.md) | 中文

Life 子系统是 Harness 观察 Taiji 本地 runtime 的窗口——该 runtime 是运行在 Host 旁边的进程，拥有 Seed native homeostasis 器官、native 训练循环，以及在其 Legacy surface 挂载时启用的 legacy 生命调度器与知识索引。[dsh-api-life-controller](../../packages/api/life-controller)拥有 Host 的 `ctx.lifeController` 服务和生成的 `ctx.remote.life` namespace；浏览器通过 Client 的 `ctx.life` 服务以及同一包安装的可重连状态流读取该 namespace。runtime 始终是唯一事实来源：控制器只轮询它、原样搬运它的每个数值，绝不合成 runtime 未报告的取值。

Source: [`packages/api/life-controller/src/types.ts`](../../packages/api/life-controller/src/types.ts)

## 数据来源

控制器按四个层级读取 runtime 的 HTTP 面，而某个数值来自哪一层本身就是数据的一部分。

- 常开读取：`GET /api/runtime/status` 携带 health、memory、life、training、`tools`（工作台能力快照）与 `auth`（运行时鉴权态）分节，`GET /api/train/checkpoints` 携带 checkpoint 名单，`GET /api/train/files` 携带可训练数据集名单（data 目录下的 POSIX 相对路径及其大小），`GET /api/artifacts` 携带发布态——哪个检查点在应答、settings 为下次启动点名了哪个——`GET /api/consolidation/status` 携带记忆日志计数与睡眠 pass 的产物——pass 计数、最近语料、数据环规格与最近报告。只要 runtime 进程存活，各层都会应答；早于巩固面的 runtime 在该路径应答 `404`，控制器据此记录为一行 `unavailable` 而非快照失败。
- 受门控读取：`GET /api/life/status`（legacy 调度器）与 `GET /api/rag/status`（知识索引）仅在 runtime 通过 `SEED_ENABLE_LEGACY` 启用其 Legacy surface 时才挂载；`GET /api/rag/files`（已挂载文档及其索引状态）仅在该面应答时才读取，因此被禁用知识库只留下一个 `disabled` 事实。未挂载的受门控路径应答 `404`，控制器据此记录为 `disabled`——即"该来源从未存在"，而不是"runtime 坏了"。
- 训练进度流：`POST /api/train/native` 与 `POST /api/train/resume_checkpoint`（从已保存的 checkpoint 续训，可带数据集与 tick 上限）应答同一条 server-sent event 流：每个 `progress` 事件携带一个 `LifeProgressView`，`warning` 事件（例如续训时的语料漂移提醒）由控制器折叠进快照的 `training.warnings`，由 `completed` 或 `error` 事件结束本次运行。
- 巩固控制：`POST /api/consolidate` 运行一次 native 睡眠 pass（分析、投影、规格化，仅在被要求时才让基底入睡），并以 JSON 应答其报告。
- 激活控制：`POST /api/runtime/activate` 让后续回合改由平台所有的检查点应答——空名称选择内置 seed——并把选择写入 settings。已退出的 `/api/model/publish|published|export_gguf` 应答 `410`；对检查点执行激活才是发布面。
- 资源控制：`POST /api/train/upload_dataset` 与 `POST /api/rag/upload` 以 multipart 字节把一个所选文件送进 runtime 的数据目录或文档目录；`DELETE /api/train/file/{path}`、`DELETE /api/train/checkpoint/{filename}` 与 `DELETE /api/rag/file/{name}` 按名删除一个数据集、检查点或文档。检查点路由以 `409` 拒绝活跃与已配置的那一枚；数据集路由对已消失的文件应答 HTTP 200 而响应体自称 `error`，控制器把它抛成 runtime 的失败，而不是读成成功。

承载生命数值的 status 分节会自报其来源器官，而该名称决定数据结构。`life.status === 'seed'` 报告 native homeostasis 器官，产出 `LifeNativeView`（观测计数、器官模式，以及 runtime 已缩放到 0..100 的开放 need 与 drive 映射）；`life.status === 'ok'` 报告 legacy 调度器，产出 `LifeLegacyView`（其循环标志、当前活动、主导 need、need 映射、心跳与事件计数，以及最近心跳和最近活动时刻）。其他取值使 `life` 缺失。控制器不会重新换算 native homeostasis 的数值，也不会用 native 读数顶替 legacy 读数：两个器官各自使用自己的量纲，是否可比由消费方判断。

## 快照

`LifeSnapshot` 是一次完整读数，并指明应答的器官：`source`（哪个器官提供了生命数值）、`observedAt`（本次读数的 ISO-8601 时刻）、`fresh`（仅当本轮 status 读取有应答时为真）、`pollIntervalMs`（取数时生效的轮询间隔）、可选的 `health`、`memory`、`life`、`knowledge`、`consolidation`、`artifacts`、`workbench`、`auth` 投影、始终存在的 `training` 投影、`availability`，以及 `unavailable`。运行中的 runtime 未包含的 status 分节使对应投影保持缺失，而非填零。

`unavailable` 为每个未应答的来源保留一行便于运维阅读的说明，使面板能够指出 runtime 的哪一部分缺失，而不是显示空白。没有人测量过的量保持缺失而非默认零，因为一个永久为零的读数读起来像已测得的事实，反而掩盖了来源从未应答。

`LifeAvailability` 独立分类每个来源：`runtime` 为 `ok` 或 `down`，`legacy` 与 `knowledge` 为 `ok`、`disabled` 或 `down`，`trainingStream` 为 `idle`、`streaming` 或 `closed`。`LifeTrainingView` 携带 runtime 的训练锁标志、它尚未观测到的暂停与停止请求、发布锁、最新的 `LifeProgressView`、按最新在前的 checkpoint 名单（截断到 `maxCheckpoints` 上限）、`datasets`——面板提供给一次运行的可训练名册（`path` 与 runtime 报出的 `size_bytes`），该读取未应答时保持缺失——以及 `warnings`：本 Host 上一次运行中 runtime 自己的消息，新运行被接受时清空、运行结束后仍保留。`LifeKnowledgeView` 携带索引规模与 `files`——已挂载文档及其大小和 `indexed` 或 `pending` 状态，文件清单读取未应答时保持缺失。

## Remote 方法

| 方法 | 类型 | 行为 |
| --- | --- | --- |
| `snapshot` | 一元 | 返回当前 `LifeSnapshot`；当轮询循环尚无快照时会先读取一次，而未能应答的 runtime 产出标记为 `down`、`fresh` 为假的快照，而不是失败。 |
| `follow` | 流 | 先发出一帧携带当前快照的 `baseline`，此后每次读数变化发出一帧替换整份快照的 `snapshot`；重连会以新 baseline 开始新一代。 |
| `trainStart` | 一元 | 通过进度流启动一次 native 运行并以 runtime 的接受应答；同一 Host 上的第二次运行以 `life/conflict` 拒绝，进度经由 `follow` 到达。 |
| `trainResumeCheckpoint` | 一元 | 经同一进度流从已保存的 checkpoint 续训——接受数据集与 tick 上限，并把 runtime 的语料漂移 warning 折叠进 `training.warnings`；同一 Host 上的第二次运行以 `life/conflict` 拒绝。 |
| `trainPause` | 一元 | 请求 runtime 暂停正在运行的训练，返回其消息。 |
| `trainResume` | 一元 | 请求 runtime 恢复已暂停的训练，返回其消息。 |
| `trainStop` | 一元 | 请求 runtime 在当前 step 之后停止，返回其消息。 |
| `trainReset` | 一元 | 强制 runtime 释放其仍持有的训练锁，返回其消息。 |
| `uploadDataset` | 一元 | 经 `POST /api/train/upload_dataset` 以 multipart 字节把一个所选文件送进 runtime 的数据目录；Host 先把名称收敛为 basename、核对可训练后缀，并在任何字节出发前拒绝超出上传预算的载荷。 |
| `deleteDataset` | 一元 | 经 `DELETE /api/train/file/{path}` 删除一个数据集；Host 校验路径保持为带可训练后缀的相对名册路径，而响应体自称 `error`（该路由对已消失文件的应答）会被抛成 runtime 的失败。 |
| `deleteCheckpoint` | 一元 | 经 `DELETE /api/train/checkpoint/{filename}` 删除一个检查点；Host 校验平坦的 `*.pt` 名称，runtime 以 `409`（`life/conflict`）拒绝活跃与已配置的那一枚。 |
| `uploadKnowledge` | 一元 | 经 `POST /api/rag/upload` 以 multipart 字节把一个所选文档送进 runtime 的文档目录；runtime 在后台对其向量化。 |
| `deleteKnowledge` | 一元 | 经 `DELETE /api/rag/file/{name}` 删除一个文档；Host 校验平坦文件名，因此删除只作用于被点名的那一个文件。 |
| `consolidate` | 一元 | 经 `POST /api/consolidate` 运行一次 native 睡眠巩固 pass，返回 runtime 的报告消息；请求的 `reason` 可省略，默认使用 runtime 自己的取值。 |
| `activateCheckpoint` | 一元 | 经 `POST /api/runtime/activate` 让后续回合改由平台所有的检查点应答——空名称激活内置 seed——并把选择写入 settings；检查点缺失或无法加载时是 runtime 自己的 `life/runtime-error` 拒绝。 |
| `lifeStart` | 一元 | 启动受门控的 legacy 生命调度器，返回其消息。 |
| `lifeStop` | 一元 | 停止受门控的 legacy 生命调度器，返回其消息。 |
| `lifeAction` | 一元 | 携带运维可见的原因，强制执行一次受门控的 legacy 活动——`feed`、`sleep` 或 `play`，返回 runtime 的消息。 |

每个控制动词在 runtime 接受后都会重读一次，使面板无需等待下一轮轮询即可看到自身动作的效果。`follow` 流由一个轮询循环服务所有消费方：每轮只读一次 runtime，盖上当时生效的间隔，并仅在渲染结果变化时发布，因此训练占用 runtime 期间生效的是较高频的活动间隔。

## 失败

Life 失败是带稳定错误码的单个 `RemoteError`，其 details 携带用于区分各种情形的信息。

| 错误码 | Details | 触发条件 |
| --- | --- | --- |
| `life/runtime-unreachable` | `baseURL`、`reason` | runtime 始终未应答：连接、DNS 或超时故障。 |
| `life/runtime-error` | `status`、`detail` | runtime 以动词无法使用的非 2xx 状态应答；`detail` 是 runtime 自己的文本。 |
| `life/unavailable` | `source`、`reason` | 运行中的 runtime 不提供某个受门控来源——未挂载的 legacy 生命或知识 surface，或以 `404` 拒绝的 legacy 控制动词。 |
| `life/conflict` | `reason` | runtime 状态禁止该动词：runtime 返回 `409`，或在同一 Host 上已有进度流时调用 `trainStart`。 |
| `life/bad-request` | `field`、`reason` | 请求不是 runtime 接受的结构：runtime 返回 `400` 或 `422`，或 `lifeAction` 指名了 `feed`、`sleep`、`play` 之外的活动。 |

## Client 投影

Client 入口安装 `ctx.life`、`ILife` 门面以及一条可重连状态流。`ClientLifeModel` 用一个身份稳定的状态对象持有最新快照（`loading`、`ready`，或保留最后快照的 `error`），并向渲染方提供 `getSnapshot()` 与 `subscribe()`；`LifeClient` 解开每个 Host 结果，并在被拒绝时抛出携带结构化 `rpcError` 的 `LifeControlError`。`createLifeStateStream()` 在 `ctx.remote.life.follow` 之上包装 Gateway 的 `RemoteSnapshotStream`：`baseline` 帧替换全部状态，其后的每一帧替换快照，载体丢失由 Gateway 拥有的流负责重试而不是本包；本身不是 Remote failure 的失败会被归一化为 `life/stream-failed`。

<!-- BEGIN GENERATED cordis-surface (gen-cordis-catalog.ts) — do not edit between markers -->

<a id="cordis-surface"></a>

## Cordis API

Generated from source by `scripts/gen-cordis-catalog.ts` (verified fresh by `pnpm run verify-cordis-catalog` in doc-sync; regenerate with `pnpm run gen-cordis-catalog`) — the language sides differ only in locale-specific paired document paths. Signature blocks use a `ts cordis-catalog` fence and keep the original source JSDoc; dispatch modes are defined in the [primer](../cordis-primer.zh.md#dispatch-modes), and the framework-inherited `ctx` API lives in [cordis-api/inherited.md](../cordis-api/inherited.md).

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