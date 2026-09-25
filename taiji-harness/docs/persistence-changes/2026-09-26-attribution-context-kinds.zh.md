---
description: "记录持久化类型更改及其兼容性确认。"
kind: persistence-change
---

# 2026-09-26-attribution-context-kinds

[English](2026-09-26-attribution-context-kinds.md) | 中文

## 概述

两个消息 source kind——`life-context` 与 `memory-context`——加入持久 user 与 developer 消息的 `source` 联合（`event:user/message`、`event:developer/message`、`event:agent/inbox/spliced`，以及 `event:session/title-llm-request` 的消息投影）。它们由 Taiji 的 life-context 与 memory-context 插件贡献，这两个插件把本地运行时的读数与日志召回，作为一条可归因的上下文行折入每个模型步骤。这两个 kind 只承担归因：消息正文就是上下文文本，source 指明写下它的生产者。

## 目录

- [声明](#declaration)
- [兼容性](#compatibility)
- [验证](#verification)
- [开发备注](#dev-note)

<a id="declaration"></a>
## 声明

```yaml persistence-change
schemaVersion: 1
id: 2026-09-26-attribution-context-kinds
baseline: false
changes:
  - root: "event:agent/inbox/spliced"
    previous: "2026-09-16-session-format-v4"
    after: "0f705b014bd1b8ed583f70fbd9a58eb07306b9d08ea29463c9939466f60d2780"
    decision: same-version
  - root: "event:developer/message"
    previous: "2026-09-16-session-format-v4"
    after: "e27f3253b25a0d3e1a154be8b671dc5513f5d5bb99d78d5fd1f312dde349141a"
    decision: same-version
  - root: "event:session/title-llm-request"
    previous: "2026-09-16-session-format-v4"
    after: "7d3ed761aaf99955e970f732a92d61492e950d1518cdfff3e98b8711a495fa13"
    decision: same-version
  - root: "event:user/message"
    previous: "2026-09-16-session-format-v4"
    after: "406c76d81760316145214b8b31ec6e3fcbe65d354389956e95d126db6b596a08"
    decision: same-version
```

<a id="compatibility"></a>
## 兼容性

裁定：同版本。两个 kind 都声明为「仅归因」（`@persistenceAttribution`，策略 `session-source-attribution`，`unknownKinds: preserve`），与上游 `tmux-context` 生产者在完全相同贡献形态上使用的合同一致。不认识这两个 kind 的读取方仍保留该消息及其正文，只是把 source 视为未识别的生产者，因此用这两个插件写下的日志，在缺少插件的构建里依然可读，也不隐含任何 Session 格式变更。两个插件按步骤注入或整块省略上下文行，从不重写既有消息，所以本次只是给联合增加成员，没有改动任何既有分支的字段。

<a id="verification"></a>
## 验证

用 `tsx scripts/gen-persistence-catalog.ts` 重新生成 `docs/persistence-schema.json`、`docs/persistence-catalog.md` 与 `docs/persistence-catalog.zh.md`，其中为每个受影响 root 记下 source 兼容策略；用 `tsx scripts/persistence-formats.ts --write` 刷新历史格式事实（v0 至 v4 已验证，5 份完整参照）。`tsx scripts/persistence-changes.ts --check` 把八处新增判为 `attribution-kind-added` 且 `requiresVersionBump=false`，`SESSION_FORMAT_VERSION` 保持 4。生产者包自身的用例通过：`packages/context/life-context/tests` 与 `packages/context/memory-context/tests`，其中包含「运行时不可达或无料时整块不注入，而不是留下空归因」的断言。

<a id="dev-note"></a>
## 开发备注

无。
