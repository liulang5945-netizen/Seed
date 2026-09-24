---
description: "把 Taiji 运行时的生命读数作为持久上下文并入每次模型步骤。"
kind: "package-reference"
---

# @taiji/dsh-life-context

[English](README.md) | 中文

## Summary

把 Taiji 运行时的生命读数并入模型请求。本插件挂载在 [`@taiji/dsh-api-life-controller`](../../api/life-controller/README.zh.md) 旁，把它的快照流变成两处上下文贡献：系统提示里的一段静态策略，以及每个合格步骤追加的一条持久 `life-state` 用户消息，携带需求、驱力、训练状态与知识库规模。读数有节流、有单行预算，runtime 不可达或快照过期时整块省略。它自身不做任何判断：每个数字都原样来自运行时器官。

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)

-----

<a id="use-this-package"></a>
## Use this package

在 Host 与 Taiji 本地运行时对话的装配里，把本插件挂在 `@taiji/dsh-api-life-controller` 之后。它声明 `inject = ['systemPrompt', 'lifeController']`，此外不向组件树贡献任何东西。

生成的[配置目录](../../../docs/config-catalog.zh.md#taijidsh-life-context)是每个受支持字段及其 JSDoc 的穷尽式真源。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `enabled` | `true` | 为 `false` 时不挂载 section，也不注册 pre-step 监听。 |
| `refreshIntervalMs` | `30000` | 两次持久注入之间的最小毫秒数；`0` 表示每个合格步骤都注入。 |
| `maxChars` | `400` | 一条 `life-state` 行的字符硬预算；超限截断并追加 `truncated=1`。 |

<a id="understand-the-implementation"></a>
## Understand the implementation

每个被接受的模型步骤都会前置一条持久用户消息，其 source kind 为 `life-context`，由 controller 的最新快照渲染：

- `life-state age=5s source=native tick=41 mode=wake needs[curiosity=42.5 fatigue=10 stress=1.5] drives[exploration=40 replay=10 rest=0 play=30] training[off] knowledge[12 docs 340 chunks]` —— native 读数。
- legacy 读数报告 `state`、`dominant`、五项调度器需求与心跳计数。没有内容可报的段被丢弃，绝不补默认值。

处于 `refreshIntervalMs` 之内时跳过注入，除非读数发生了显著变化：训练态翻转、主导需求变化，或任一单项需求漂移超过十分。runtime 读取失败的快照以 `unreachable` 省略；超过六十秒的快照以 `stale` 省略。每种省略原因在进程内只警告一次。

**运行时不变式：** 不发布伴随包。本包只在 life controller 已轮询的快照之上提供两项上下文贡献。

-----

<a id="model-experience"></a>
## Model Experience

### Life-state 读数

#### 模型看到什么

读数以一条持久用户消息的形式前置在该步骤的其它消息之前，系统提示词则在仓库 `LIFE_POLICY` 位置携带一段 `life:policy` section，说明读数是内部遥测、不是证据。

##### 一条原生读数

```markdown
life-state age=5s source=native tick=41 mode=wake needs[curiosity=42.5 fatigue=10 stress=1.5] drives[exploration=40 replay=10 rest=0 play=30] training[off] knowledge[12 docs 340 chunks]
```

#### Token 影响

每条读数的成本是一行，受 `maxChars`（默认 400 字符）约束，外加固定的策略段落。节流把持久成本限制在 runtime 安静时约每 `refreshIntervalMs` 一条读数。

#### KV Cache 影响

读数是持久用户消息，会留在对话里并被模型在后续每一步重读；约束其累积的是节流，而不是淘汰。

## Known Limitations and Deferred Work

<a id="known-limitations-and-deferred-work"></a>

- 节流与最近读数记忆都在进程内：Host 重启会遗忘它们，可能在极短时间内注入两次。
- 读数的新鲜度以 controller 的轮询循环为准；本插件自身不做轮询。
- 注入对 Host 的所有 agent 全局生效。按 agent 退出可以走 system prompt 的 scoped 注册，留待出现消费方再做。
