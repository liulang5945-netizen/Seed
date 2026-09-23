---
description: "把 Taiji 本地运行时（127.0.0.1:8000）接成 harness 的一个 LLM provider 路由。"
kind: "package-reference"
---

# @taiji/dsh-llm-taiji

[English](README.md) | 中文

## Summary

通过 `taiji-local` 路由接入 Taiji 本地运行时。该运行时是本 fork 自己的语言器官：它以纯文本说话，只提供一次 `POST /api/chat/stream` 调用，并在 `GET /api/health` 上公开自身就绪状态。本适配器把 harness 请求装进这个形状，把运行时的回答作为一个文本块产出，并把其它一切结果——失败帧、非 2xx 响应、不可达端点、不服务的帧——归一化为 Trajectory 能记录的 provider 中性失败。它不做任何质量加工：回答原样透传。本包可与 [DeepSeek](../llm-deepseek/README.md) 和 [pi-ai](../llm-pi-ai/README.md) 适配器同时挂载，部署的默认模型由装配决定。

## Table of Contents

- [使用本包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [进一步阅读](#further-exploration)
- [模型体验](#model-experience)
- [已知限制与后续工作](#known-limitations-and-deferred-work)
- [开发者备注](#dev-note)

-----

<a id="use-this-package"></a>
## 使用本包

把本插件与 harness LLM 服务一起挂载即可提供 `taiji-local`。该运行时不需要凭证，因此本插件没有凭证接缝、也没有密钥解析步骤。

### 何时选用

当部署的模型就是本仓构建出的 Taiji 运行时——此时 harness 就是该运行时自己的评测场，每个请求的失效模式都是一条可读的读数——选用本适配器。托管模型请选用 `dsh-llm-deepseek` 或 `dsh-llm-pi-ai`。三者路由名不冲突，可以同时挂载。

### 最小配置

```yaml
- name: '@taiji/dsh-llm-taiji'
  config:
    baseURL: http://127.0.0.1:8000   # 可选；这就是默认值
    models:                          # 可选；默认只有一条
      - id: taiji-local
        name: Taiji（本地运行时）
```

| 字段 | 默认 | 含义 |
|---|---|---|
| `baseURL` | `http://127.0.0.1:8000` | 运行时 HTTP 根；会追加 `/api/chat/stream` 与 `/api/health`。必须是 HTTP(S) 根，且不带凭证、查询串或片段 |
| `models` | 一条（`taiji-local`） | 供发现类消费者展示的提示性目录；未列出的 id 同样可路由 |
| `retryPolicy` | normal，5 次重试 | provider 自有的重试策略，由 `dsh-llm-retry` 执行 |

请求用 `provider: taiji-local` 选择本路由。运行时完全不接受模型 id，因此 `model` 只是本地选择值：它不上线、也不按目录校验。

### 就绪与可路由

只有当运行时能够服务请求时，该路由才注册在 LLM 服务上。插件加载时、以及每次 `loader/volatile-update` 时，适配器探测 `GET /api/health` 并套用一个判定：

| 上报的 `status` | 路由 | 原因 |
|---|---|---|
| `ok` | 注册 | 运行时报告自己可服务请求 |
| `loading` | 注册 | 「还没就绪」不是失败：随后的请求会记录实际发生的事 |
| `downloading` | 注册 | 同 `loading` |
| `error` | 撤回 | 运行时报告自身启动失败 |
| 其它取值、非 2xx、或连不上 | 撤回 | 端点没有给出任何就绪判定，或根本没有应答 |

被撤回的路由是既有注册的「空路由集」，而不是被销毁的注册，因此 provider 回归时两次注册之间没有缝隙。撤回期间 provider 仍列在可配置 provider 目录里，这正是 Models 页面能够显示并编辑其端点的原因。

就绪只在加载时与每次 volatile 更新时采样一次；后台不做轮询。加载之后才起来的运行时，要等到下一次 volatile 更新或 profile 重启才会被注册。

### 端点与线上格式

聊天调用为 `POST {baseURL}/api/chat/stream`，头部 `Accept: text/event-stream` 与 `Content-Type: application/json`。请求体是运行时的 `ChatRequest`：

```json
{
  "prompt": "当前用户轮次的纯文本",
  "system_prompt": "生效的系统提示词，没有则省略",
  "history": [["用户文本", "助手文本"]]
}
```

`prompt` 是最后一条 user 消息的可见文本。`system_prompt` 是最后一条 system 消息的可见文本；若没有任何 system 消息承载提示词，则用一次性调用的 `GenerateOptions.system`；省略该字段会让运行时套用它自己的默认人格。`history` 把此前每条用户侧消息与其后的助手文本配成一对；没有回复的用户侧消息保留该对，第二个元素为空串。

响应是 SSE。成功的流先写一个 `final` 帧，再写哨兵：

```text
data: {"type":"final","data":{"answer":"<全文>","step":1,"runtime":"seed","...":null}}

data: [DONE]

```

带内生成失败是一个**裸 JSON 字符串**，不是对象（`api/routes_chat.py`）：

```text
data: "生成出错: ..."

```

`[DONE]` 结束流。`final` 帧携带完整回答，因此适配器产出 `block-start`(text)、一个携带全文的 `text-delta`、`block-end`，最后是以 `kind: 'stop'` 结束的 `finish`。不产出 `usage` 块：运行时不上报 token 计数，本包也不编造。

### 失败与恢复

每个失败都是 provider 中性的 `LlmError`，因此打在 Taiji 运行时上的失败步骤，其记录与重试方式和打在其它 provider 上的失败完全一致。

| 观察到 | 错误码 | 说明 |
|---|---|---|
| `data: "生成出错: ..."` | `SERVER` | 运行时自己的措辞，原样保留 |
| HTTP 401/403 | `AUTH` | |
| HTTP 402，或响应体含配额措辞 | `QUOTA` | |
| HTTP 429 | `RATE_LIMIT` | |
| 响应体含上下文超限措辞 | `CONTEXT_WINDOW_EXCEEDED` | |
| HTTP 400/413/422 | `INVALID_REQUEST` | |
| HTTP 5xx | `SERVER` | 携带状态码 |
| 其它非 2xx | `HTTP_<status>` | |
| 连接被拒、DNS 失败、超时 | `TRANSPORT` | |
| 调用方取消 | `ABORTED` | |
| 帧不是 JSON、不是对象、不是 `final`，或没有字符串 `answer` | `MALFORMED_RESPONSE` | |
| 流在任何回答之前结束 | `STREAM_CLOSED` | |
| `final` 帧的 `answer` 为空 | `EMPTY_RESPONSE` | 默认策略会重试 |

-----

<a id="understand-the-implementation"></a>
## 理解实现

<details>
<summary>实现内幕——点击展开</summary>

本节解释适配器背后的设计；可观察行为已在[使用本包](#use-this-package)中完整覆盖。

### 设计取舍

两个决定塑造了本包。

第一，翻译是字面的。运行时只接受一个用户轮次、一个可选系统提示词和若干完整配对，因此适配器只映射这些，对于该形状没有位置的内容直接丢弃，而不是围绕它编造散文。运行时随后产出什么就是读数；适配器从不改进、重试或修补它。

第二，可路由性就是注册表自身的成员资格。适配器无法另造一个可用性标志：harness 已经用「该路由是否注册」回答了「这条路由能否服务请求」。因此就绪判定通过注册或撤回路由来套用，用的正是每个适配器都用的那个注册句柄。

### 源码地图

| 文件 | 职责 |
|---|---|
| [`src/index.ts`](src/index.ts) | 注册路由并套用每一次就绪判定 |
| [`src/adapter.ts`](src/adapter.ts) | 请求生命周期与块协议 |
| [`src/chat.ts`](src/chat.ts) | harness 消息到运行时聊天形状的翻译 |
| [`src/sse.ts`](src/sse.ts) | 运行时流的帧解码 |
| [`src/health.ts`](src/health.ts) | 就绪探测及其路由判定 |
| [`src/transport.ts`](src/transport.ts) | HTTP 失败分类 |
| [`src/config.ts`](src/config.ts) | 配置 schema 与那一步显式解析 |

### 线上流程

一次 `stream()` 调用发一次聊天请求；插件加载与每次 volatile 更新各发一次就绪探测。请求携带共享的归属头，且不含任何凭证。适配器只读完整帧：没有终止符的尾部永远不算事件，因此中途断开的连接无法伪造出回答。

</details>

-----

<a id="further-exploration"></a>
## 进一步阅读

- [dsh-llm 服务](../llm/README.md)——本适配器注册所在的 provider 中性服务。
- [llm-deepseek 适配器](../llm-deepseek/README.md)——本包模仿其结构的直连 Messages 实现。
- [LLM 流式子系统](../../../docs/subsystems/llm-streaming.md)——`StreamChunk` 协议与适配器约定。
- [llm-retry](../llm-retry/README.md)——执行本适配器 `retryPolicy` 的重试执行器。

-----

<a id="model-experience"></a>
## 模型体验

### Taiji 请求

#### 模型看到什么

运行时收到的是当前用户轮次、生效的系统提示词，以及对话中已经完成的配对，全部为纯文本。它收不到工具 schema、推理、图片，也收不到 token 计数：harness 内容中该运行时形状没有位置的部分被丢弃，而不是被描述出来。harness 的系统提示词通常以开头那条 system 消息的形式抵达。

- 请求体：`{"prompt","system_prompt","history"}` —— 运行时的三个字段，别无其它。

#### Token 影响

运行时的分词器决定其输入，因此 harness 只能估计而非实测。由于运行时不上报 usage，本适配器不会向 Session 添加任何 token 计数。

#### KV Cache 影响

经由本适配器没有：运行时自行维护会话状态，请求也不携带任何缓存控制或重放元数据。

### Taiji 响应

#### 模型看到什么

运行时的回答原样作为一个文本块回传。harness 永远收不到推理块、工具调用或部分文本，因此经由本路由的轮次无法调用工具。

#### Token 影响

只保留运行时实际产出的生成 token；适配器不添加、也不上报。

#### KV Cache 影响

没有；状态归运行时所有。

## 已知限制与后续工作

<a id="known-limitations-and-deferred-work"></a>

- **工具调用、推理与图片不经过本路由**——运行时的聊天形状只承载纯文本，本适配器不做超出它的翻译。因此 `taiji-local` 上的轮次无法调用工具；这是刻意保留的读数，而不是应当掩饰的缺陷。
- **回答一次性到达**——运行时的 `final` 帧携带完整文本，因此适配器只产出一个 `text-delta`，首个可见 token 的时间戳就等于回答到达的时刻。
- **不上报 token 用量**——运行时不公开任何计数，因此该路由上的 token meter 回落到它自己的估计器。
- **可路由性是采样而非轮询**——就绪探测只在插件加载与每次 `loader/volatile-update` 时运行；之后才就绪的运行时在下一次上述时机之前一直处于撤回状态。
- **目录是提示性的、未经核验**——适配器从不询问运行时提供哪些模型，因为运行时的端点根本不接受模型 id。
- **配对形状之外的历史被丢弃**——当前用户轮次之后的消息，以及任何位置的非文本块，对请求都没有贡献。不会为补偿而改写任何内容，Session 日志保留原始内容。
- **不产出重放状态**——`finish` 块不携带它，因为运行时的帧里没有任何可供重放的不透明 provider 元数据。

<a id="dev-note"></a>
### 开发者备注

无。

**运行时不变式：** 不发布伴随包。本包不暴露超出其所属接缝所强制契约之外的独立事件序列或可变数据关系。