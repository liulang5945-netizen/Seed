# R5C-S35：artifact batch 输入隔离与部分失败原子性

## 目标

补齐 replay-bound artifact batch 的输入边界：batch API 不得静默忽略未知 candidate/replay mapping key；一个 candidate 的 artifact 解析或 replay 失败时，只能在明确的候选粒度 fail-closed，同时允许同 batch 的其他合法 candidate 按既有 reservation/validation/admission 生命周期继续。

## 计划边界

- 对 `artifacts_by_candidate` 与 `replays_by_candidate` 的未知 key 做 batch 级显式拒绝，拒绝前不改变 candidate state、reservation、artifact ledger、topology、budget 或 cursor。
- 用一个 malformed artifact 与一个合法 measured artifact 组成同一 batch 的部分失败输入，证明失败候选被隔离、合法候选仍可准入，资源 reservation 只按实际候选处理。
- 覆盖错误 candidate binding、缺字段 replay、重复提交和 checkpoint restore；不把异常输入折叠成静默跳过。
- 在 native checkpoint 前后比较 batch state、artifact batch digest、预算、拓扑、evidence cursor 与 retention policy。

## Gate

- unknown artifact/replay key fail-closed 且完全原子。
- malformed candidate 不污染另一 candidate 的 measured artifact、admission 或 topology。
- 合法 candidate 仍能完成准入，失败 candidate 的 reservation/状态变化可审计且可 checkpoint。
- 重复提交只返回既有终态，不产生第二次结构变更。

## 明确不声明

本 slice 只覆盖 native CPU artifact batch 输入隔离与部分失败，不声明开放域收益、无限扩张、自动增预算、CUDA、前端、Windows shell、CI 或通用智能。

## 证据交付

新增定向测试、独立 evaluator、JSON Gate 报告，并同步四份 active/reference/manifest 计划事实。
