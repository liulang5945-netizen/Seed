---
description: "Host 与 Client 的生命控制：轮询 Taiji 本地 runtime 并连同应答的器官搬运每份读数，以及下发其训练与 legacy 控制动词。"
kind: "package-reference"
---
# Life Controller

[English](README.md) | 中文

## 概述

`@taiji/dsh-api-life-controller` 拥有 Host 的 `ctx.lifeController` 服务和生成的 Client `ctx.remote.life` namespace。它的 Remote 方法读取 Taiji 本地 runtime——健康、内存、由实际应答器官提供的生命数值、训练状态及其 checkpoint 名单、受门控的知识索引，以及记忆日志与睡眠巩固的产物——并承载生命面板下发的控制动词：训练的启动、暂停、恢复、停止和重置，一次巩固 pass，以及受门控的 legacy 生命启动、停止和活动。runtime 始终是唯一事实来源：每份读数都携带其来源、时间戳、新鲜度和逐来源可用性，而服务绝不合成 runtime 未报告的取值。

## 目录

- [使用本包](#use-this-package)
- [模型体验](#model-experience)
- [已知限制与延期工作](#known-limitations-and-deferred-work)
- [开发备注](#dev-note)

-----

<a id="use-this-package"></a>
## 使用本包

Host 控制器拥有唯一一个面向 Taiji 本地 runtime 的轮询循环，并由它服务所有消费方。每轮只读一次 runtime，盖上当时生效的间隔，并仅在渲染结果变化时发布，因此[生命子系统参考](../../../docs/subsystems/life.zh.md)保持为读数的唯一描述，而本 README 拥有包契约：配置、线上动词及其失败。`follow()` 先发出一帧携带当前快照的 `baseline`，此后每次变化发出一帧替换快照的 `snapshot`；重连会以新的 baseline 开始新一代，因此消费方从不依赖在断线期间收到每一帧。

runtime 是唯一事实来源，而读数会说明每个数值的出处。`GET /api/runtime/status` 始终可用：其 `life` 分节自选来源器官——`seed` 为 native homeostasis 器官，`ok` 为 legacy 调度器——其 `tools` 分节携带工作台能力快照（数量、修订与出处），其 `auth` 分节携带运行时鉴权态；此外`GET /api/train/checkpoints` 携带已保存运行的名单，`GET /api/train/files` 携带可训练数据集名单（data 目录下的 POSIX 相对路径及其大小），`GET /api/artifacts` 携带发布态（哪个检查点在应答、settings 为下次启动点名了哪个），而 `GET /api/consolidation/status` 携带记忆日志计数与睡眠 pass 的产物（pass 计数、最近语料、数据环规格与最近报告）。旧模型发布 API（`/api/model/publish`、`/api/model/published`、`/api/model/export_gguf`）是已退出的 410 墓碑——`POST /api/runtime/activate` 激活平台所有的检查点才是发布面。legacy 生命 surface（`/api/life`）与知识索引（`/api/rag`）仅在 runtime 通过 `SEED_ENABLE_LEGACY` 启用它们时才挂载；未挂载的路径应答 `404`，控制器据此记录为 `disabled` 而非失败，而 `/api/taiji/life` 下的 legacy 控制动词在同样情形下以 `life/unavailable` 拒绝。运行中的 runtime 不提供的巩固读取会成为一行 `unavailable` 而非快照失败，因为较旧的 runtime 是部署的事实，不是坏掉的读数。因此每份快照都携带 `source`、`observedAt`、`fresh`、`availability`，以及每个未应答来源的一行 `unavailable`；没有人测量过的量保持缺失而非默认零。控制器不会重新换算 native homeostasis 的数值，也不会用 native 读数顶替 legacy 读数。

### 配置

| 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `baseURL` | `http://127.0.0.1:8000` | Taiji 本地 runtime 的 Base URL |
| `pollIntervalMs` | `5000` | 空闲时两次读取之间的间隔，单位为毫秒（最小 `250`） |
| `activePollIntervalMs` | `2000` | 训练占用 runtime 期间两次读取之间的间隔，单位为毫秒（最小 `250`） |
| `requestTimeoutMs` | `2000` | 单次 runtime 请求的最大时长，单位为毫秒（最小 `1`） |
| `maxCheckpoints` | `20` | 单份快照携带的 checkpoint 行数（最小 `1`） |

当 runtime 报告正在训练或存在打开的进度流时，生效的是活动间隔。`requestTimeoutMs` 分别限制每次读取和每次控制请求，超过它的请求抛出 `life/runtime-unreachable`。每个控制动词在 runtime 接受后都会重读一次，使面板无需等待下一轮即可看到自身动作的效果。

### Remote 方法

| 方法 | 类型 | 用途 | 失败 |
| --- | --- | --- | --- |
| `snapshot` | 一元 | 读取当前快照；未能应答的 runtime 产出标记为 `down`、`fresh` 为假的快照。 | 仅当中止不是调用方发起时才抛 `life/runtime-unreachable`。 |
| `follow` | 流 | 以 `baseline` 帧开启一代，此后每次变化替换快照。 | 同 `snapshot`；丢失的载体由 Client 流负责重试。 |
| `trainStart` | 一元 | 通过进度流启动一次 native 运行，并以 runtime 的接受应答。 | 当同一 Host 上已有运行在流式输出时为 `life/conflict`，以及 runtime 自身的拒绝码。 |
| `trainResumeCheckpoint` | 一元 | 经同一进度流从已保存的 checkpoint 续训（含 runtime 的语料漂移 warning），并以 runtime 的接受应答。 | 当同一 Host 上已有运行在流式输出时为 `life/conflict`，以及 runtime 自身的拒绝码。 |
| `trainPause` | 一元 | 请求 runtime 暂停正在运行的训练。 | `life/runtime-error`、`life/conflict`、`life/bad-request`、`life/runtime-unreachable`。 |
| `trainResume` | 一元 | 请求 runtime 恢复已暂停的训练。 | `life/runtime-error`、`life/conflict`、`life/bad-request`、`life/runtime-unreachable`。 |
| `trainStop` | 一元 | 请求 runtime 在当前 step 之后停止。 | `life/runtime-error`、`life/conflict`、`life/bad-request`、`life/runtime-unreachable`。 |
| `trainReset` | 一元 | 强制 runtime 释放其仍持有的训练锁。 | `life/runtime-error`、`life/conflict`、`life/bad-request`、`life/runtime-unreachable`。 |
| `consolidate` | 一元 | 运行一次 native 睡眠巩固 pass 并返回 runtime 的报告消息；请求的 `reason` 可省略，默认使用 runtime 自己的取值。 | `life/runtime-error`、`life/conflict`、`life/bad-request`、`life/runtime-unreachable`。 |
| `activateCheckpoint` | 一元 | 让后续回合改由平台所有的检查点应答（`POST /api/runtime/activate`）；空 id 激活内置 seed。 | `life/runtime-error`——检查点缺失或无法加载是 runtime 自己的拒绝；目录外名称为 `life/bad-request`。 |
| `lifeStart` | 一元 | 启动受门控的 legacy 生命调度器。 | legacy surface 未挂载时为 `life/unavailable`，以及 runtime 自身的拒绝码。 |
| `lifeStop` | 一元 | 停止受门控的 legacy 生命调度器。 | legacy surface 未挂载时为 `life/unavailable`，以及 runtime 自身的拒绝码。 |
| `lifeAction` | 一元 | 携带运维可见的原因，强制执行一次受门控的 legacy 活动——`feed`、`sleep` 或 `play`。 | 活动名不在其中时为 `life/bad-request`，legacy surface 未挂载时为 `life/unavailable`。 |

Client 入口安装 `ctx.life`、`ILife` 门面以及一条可重连状态流。`ClientLifeModel` 用一个身份稳定的状态对象持有最新快照（`loading`、`ready`，或保留最后快照的 `error`），并提供 `getSnapshot()` 与 `subscribe()`；`LifeClient` 解开每个 Host 结果，并在被拒绝时抛出携带结构化 `rpcError` 的 `LifeControlError`。

<a id="model-experience"></a>
## 模型体验

无，因为 Life 子系统为面板读取本地 runtime 的控制状态，且不注册提示词、工具或会话事件。

#### KV Cache 影响

无直接影响；Life 的读取与控制动词都不会改变模型请求。

## 已知限制与延期工作

<a id="known-limitations-and-deferred-work"></a>

- 控制器只做轮询：它从不订阅 runtime 的推送通道，因此变化要在下一轮才可见，而非发生即刻可见。
- `follow()` 替换整份快照而不发出字段级增量，因此想要增量的消费方需自行比较相邻帧。
- 未启用 `SEED_ENABLE_LEGACY` 时，legacy 生命与知识行在每份快照中都是缺失而非空值，因此面板应读取 `availability` 而不是假定为零。
- 训练进度可能落后：进度流断开后，快照保留它最后看到的样本，并将该流报告为 `closed`。
- 巩固 pass 与其他控制请求共享 `requestTimeoutMs`：更慢的 pass 会在 runtime 上继续运行，而动词报告超时，因此面板需要重读才能看到 pass 仍然写出的产物。

<a id="dev-note"></a>
### 开发备注

<details>
<summary>维护者工作上下文——点击展开</summary>

runtime 的 HTTP 面是本包唯一使用的契约：路径固定，载荷词表由 runtime 拥有。本包未指名的 runtime 字段保持未读，而非按默认值填充。

</details>

**运行时不变式：** 不发布伴生入口。Taiji runtime 拥有每个已测量的取值；控制器轮询它，并连同应答的器官搬运每份读数。