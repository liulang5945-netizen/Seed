# R5C-S37：SeedRuntime artifact 失败隔离与并发提交边界

## 目标

在 S36 的 runtime projection 基线上补齐异常路径：SeedRuntime 不得吞掉 tamper/stale/cross-batch 输入，也不得因多个线程同时提交同一 artifact 而产生重复结构变更或 reservation 账本破坏。

## 计划边界

- 通过 SeedRuntime wrapper 提交 tampered artifact、错误 candidate binding、stale parent artifact 和缺字段 replay，确认错误语义与 native adapter 一致。
- 对跨 batch candidate/replay mapping key 验证 fail-closed 且调用前后 checkpoint、candidate state、reservation、topology、budget 不变。
- 用两个并发调用提交同一合法 measured artifact，验证锁只允许一个真实 admission，另一调用返回既有终态，topology/budget 只发生一次预期变化。
- 对部分失败 batch 验证 malformed candidate 不污染合法 sibling，并在 checkpoint 后保留可审计状态。
- 不通过放宽 digest、静默忽略字段或吞异常来“通过” Gate。

## Gate

- tamper、stale parent、错误 binding、缺字段 replay 与跨 batch 输入都 fail-closed，且符合预期的候选级/调用级原子边界。
- 同一 artifact 并发提交最多一次真实 admission，重复调用幂等。
- runtime wrapper 与 native adapter 的 artifact、candidate、batch、budget、topology projection 一致。
- checkpoint restore 后失败记录和成功 provenance 不漂移。

## 明确不声明

本 slice 只覆盖 SeedRuntime/native CPU artifact failure isolation 与并发边界，不声明开放域收益、无限扩张、自动增预算、CUDA、前端、Windows shell、CI 或通用智能。

## 证据交付

新增定向测试、独立 evaluator、JSON Gate 报告，并同步 active/reference/manifest 计划事实。
