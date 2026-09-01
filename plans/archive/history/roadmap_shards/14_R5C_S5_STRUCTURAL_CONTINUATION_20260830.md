# R5C-S5：多步 bounded growth 与 checkpoint continuation

> 日期：2026-08-30。该文件记录 S5 的合同、证据和边界；当前执行入口仍由 `03_CURRENT_EXECUTION.md` 唯一维护。

## 目标

证明 Taiji 的结构成长不是只能完成一次的演示，而是可以在保存、关闭、恢复之后沿父 lineage 继续进行，同时受 structural budget 和 validation policy 的硬边界约束。

本阶段只验证三件事：

1. budget=2 时，第一候选 admission 后保存 child checkpoint，恢复后第二候选仍能继续 admission；
2. 两次 admission 的 topology、candidate lineage、decision 和 admission result 全部可恢复，预算精确从 2 降到 0；
3. budget=0 时，第三候选即使 holdout/retention/lesion 指标通过，也必须 fail-closed 拒绝，不得改 topology、增加预算或在重启后复活。

## 合同

- 每次 admission 必须从上一份 parent checkpoint 继续，candidate、proposal、decision、parent/child digest 和预算变更都进入 checkpoint。
- candidate validation 只能在 shadow 中观察，不得因为验证成功直接改变 topology。
- admission 只能通过既有 atomic transaction 执行，resource cost 必须精确扣减。
- 预算不足是结构性拒绝原因；不能自动扩容、隐式追加资源或回退为全量重训。
- rejected candidate 在恢复后不进入 candidate 队列；重新考虑必须有新 evidence 和新 proposal ID。

## 证据

唯一 evaluator：`scripts/training/eval_taiji_structural_continuation.py`。

它验证：

- 两次 admission：`u2`、`u3`；
- topology：`("u0", "u1") → ("u0", "u1", "u2") → ("u0", "u1", "u2", "u3")`；
- budget：`2 → 1 → 0`；
- checkpoint continuation：第一步 child restore 后继续第二步；
- exhausted rejection：`u4` 的 decision 包含 `structural_budget_insufficient`，不产生 admission；
- restore：exhausted checkpoint 保留拒绝状态、预算 0 和两步 topology；
- report：`reports/taiji_w7_r5c_s5_structural_continuation_20260830.json`，`gate.passed=true`。

## 通过标准

所有 metrics 必须为 true：两步 admission、跨 checkpoint continuation、lineage persistence、预算精确扣减、第三候选拒绝、拒绝状态 restore、exhausted topology 不变。

## 明确边界

S5 只证明有界多步结构成长和 checkpoint continuation，不证明无限自进化、自动预算扩容、开放域质量收益、真实 Workbench 长期调度或 CUDA 性能。

下一阶段：R5C-S6 长期增长调度与真实 Workbench evidence 接线。