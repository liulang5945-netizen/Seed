---
description: "把每个顶层 Taiji Host 回合的提问与回答上报到本地运行时的持久记忆库。"
kind: "package-reference"
---

# @taiji/dsh-session-memory-taiji

[English](README.md) | 中文

## 概述

把每个顶层 Taiji Host 回合上报到本地运行时的持久记忆库，让之后训练语料所依据的问答对在发生时就收集下来。插件从会话读取已停止的回合，组合一条记录，携带该回合调用的工具与一个小的显著度启发式，然后 POST 到 `POST /api/memory/record`。上报是尽力而为的：它绝不延迟或使回合失败，运行时不可达时只记录一次，且这里没有任何内容进入模型请求。

## 目录

- [使用本包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [模型体验](#model-experience)
- [已知限制与延期工作](#known-limitations-and-deferred-work)
- [开发备注](#dev-note)

-----

<a id="use-this-package"></a>
## 使用本包

在与会话 Taiji 本地运行时对话的 Host 装配里，把本插件挂在 `@taiji/dsh-life-context` 旁。它不声明任何必需服务，不向组件树贡献任何东西，只从 `agent/turn-stopping` 边界上报。

生成的[配置目录](../../../docs/config-catalog.zh.md#taijidsh-session-memory-taiji)是每个受支持字段及其 JSDoc 的穷尽式真源。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `baseURL` | `http://127.0.0.1:8000` | 运行时根地址；拼接 `/api/memory/record` 之前会去掉结尾的斜杠。 |
| `enabled` | `true` | 为 `false` 时不上报任何回合。 |
| `maxTextChars` | `2000` | 分别作用于所记录提问与回答的字符预算。 |
| `timeoutMs` | `5000` | 单次上报在被放弃前可用的毫秒数。 |

<a id="understand-the-implementation"></a>
## 理解实现

每个合格回合都会向运行时的记忆库追加一条记录。插件观察 turn-stopping 边界，因此只有在模型不再欠响应之后才读取回合：

- 合格性沿用 workspace-change 记录器的规则：subagent 会话，或任何被委派的子会话，都不上报。
- 提问是回合内最后一条 user 角色消息的可见文本；回答拼接其后 assistant 消息的文本块，工具则是它们 `tool-call` 块的名字，去重后保留顺序。
- 组合出的文本是 `问：`、裁剪后的提问、换行、`答：`、裁剪后的回答，即原生语料使用的对话形态。
- 显著度从 `0.3` 起，调用工具的回合加 `0.25`，回答为空加 `0.2`，回合被中止加 `0.25`，最后钳制到 `0..1`。

上报是调度的，绝不 await，因此缓慢或已死的运行时无法延迟回合；首次失败在 debug 级别记录一次，其后失败保持静默。每个会话的记忆会抑制对同一回合的重复 stop，会话释放时丢弃它。会话标题与压缩辅助调用永远不会到达这个接缝，因此永不上报。

**运行时不变式：** 不发布伴随包。本包只在 agent loop 已经记录的回合事件之上提供一次尽力而为的外发上报。

-----

<a id="model-experience"></a>
## 模型体验

### 持久回合日志

#### 模型看到什么

无。插件只通过 `POST /api/memory/record` 写入运行时的记忆库，因此上报的任何部分都不会进入模型请求、工具 schema 或会话接口。

#### Token 影响

无。插件不添加任何提示段、消息或工具，因此回合不因它产生额外的模型输入 token。

#### KV Cache 影响

无。插件记录的内容都不是持久对话状态，因此不会增长或移动任何已缓存的提示前缀。

## 已知限制与延期工作

<a id="known-limitations-and-deferred-work"></a>

这些限制界定了记录在何时不可靠或不完整。它们是当前的包约束，不是任务清单。

- **上报是即发即忘的**——运行时不可达时该回合的记录会丢失，Host 重启会遗忘每会话的抑制记忆。
- **提问取最后一条 user 角色消息**——Host 注入的 user 消息，例如同一步追加的一条 `life-context` 读数，会被当作提问记录下来。
- **重复保护在进程内**——插件从不读取或持久化运行时返回的 digest，因此超出一个 Host 进程的范围就依赖运行时自身的幂等键。

<a id="dev-note"></a>
### 开发备注

<details>
<summary>维护者的工作上下文——点击展开</summary>

无。

</details>