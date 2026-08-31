# R5C-S38：多轮 SeedRuntime artifact 生命周期与 retention pressure

## 目标

把 S36–S37 的 runtime artifact contract 推进到连续多轮：同一 SeedRuntime 经历多次 evidence round、artifact admission、部分失败、rollback、重启和显式 retention，验证结构成长不会因轮次增加而混淆 batch、污染 provenance 或突破资源边界。

## 计划边界

- 使用至少三轮真实 Workbench task slices，每轮创建独立 candidate batch，并为成功候选生成 measured replay-bound artifact。
- 在不同轮次混合成功、malformed、stale、rollback 和重复提交，检查 candidate/batch/artifact/evidence 的归属不会跨轮串线。
- 每轮至少一次 SeedRuntime 磁盘 save/load；显式 retention 在小上限下只淘汰终结子图，活动 reservation/pending/rollbackable lineage 继续受保护。
- 统计 bounded resource counters、reservation、structural budget、topology、cursor、artifact digest 和 retention pressure；异常输入必须保持已定义的候选级或调用级原子边界。
- 不用“轮次更多”替代独立 task slice、measured holdout 或可回放 provenance；不把连续运行误报为开放域智能。

## Gate

- 三轮 batch/artifact lineage 独立且可 checkpoint 恢复，重复/旧 artifact 不会跨轮消费。
- malformed/stale/rollback 输入按 contract fail-closed 或幂等，其他轮次和 sibling candidate 不受污染。
- retention 只淘汰无 live lineage 的完整终结子图，活动 lineage 与 pressure 记录准确。
- 多次 restart 后 structural budget、topology、cursor、policy、artifact digest 和状态 projection 一致，资源始终有界。

## 明确不声明

本 slice 只覆盖 native CPU 多轮 SeedRuntime artifact 生命周期与 bounded retention，不声明无限扩张、自动增预算、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

## 证据交付

新增定向测试、独立 evaluator、JSON Gate 报告，并同步 active/reference/manifest 计划事实。
