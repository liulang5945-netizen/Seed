# R5C-S15：实测多区域 validation artifact batch

> 状态：已完成（2026-08-30）
>
> 本 slice 将 S14 的独立 replay measurement owner 接入 S13 的多区域 artifact batch，彻底移除 batch continuation 侧的手工 metrics 集合，并保持逐候选 parent lineage、失败隔离、checkpoint restore 与幂等。

## 1. 目标

S13 已经能消费多个 candidate-bound validation artifact，但调用方仍可能在 artifact batch 外维护一组人工指标。S15 要求每个区域 candidate 都从自己的 baseline/candidate/lesion replay 与原始容量快照生成 measured artifact，再由 batch API 只消费这些 artifact。

## 2. 实现合同

| 对象 | 合同 |
|---|---|
| per-candidate measurement | 每个 candidate 使用自己的 region、task slice、parent checkpoint、trial checkpoint、replay 输入和 resource observation；不能跨候选借用指标 |
| batch input | `continue_structural_candidate_batch_from_validation_artifacts()` 只接收 candidate-bound artifacts 与 holdout replay，不接收 batch-level manual metrics |
| sequential lineage | 第一候选 admission 后，第二候选必须从当前 parent checkpoint 重新生成 measurement/artifact；旧 parent 的 artifact 不能直接复用 |
| failure isolation | 单个 artifact、replay 或 resource digest 错配只关闭对应 candidate 并释放 reservation，不能污染已 admission 区域 |
| recovery | candidate batch、artifact batch、measurement digest、admission、topology 与 budget 一起 checkpoint；恢复后 digest 一致且重复消费幂等 |
| ownership | measurement owner 不拥有 admission；provider、frontend 和 Workbench executor 不拥有结构、预算或 rollback |

## 3. Gate

canary：`scripts/training/eval_taiji_workbench_measured_artifact_batch.py`。

结果：`gate.passed=true`。

覆盖：

- 两个真实 Workbench region candidate 各自生成独立 measured artifact；
- 两侧 policy 分别消费 candidate-specific measured holdout/retention/lesion/resource metrics；
- 第一候选 admission 后，第二候选绑定新的 parent checkpoint 并完成增量 batch admission；
- 原始 resource measurement digest 与 artifact 绑定，禁止只传最终分数；
- 错配 candidate、坏 replay、错误 resource binding 在对应 candidate 上 fail-closed；
- artifact batch checkpoint restore 后 digest 保持，重复消费保持幂等，且没有 batch-level manual metric 注入。

## 4. 未关闭边界

- S15 仍验证受控多区域 canary 和 lifecycle，不声明开放域质量收益、无限预算或自主决定结构目标；
- 多轮 evidence continuation、跨轮 stale artifact 拒绝、第二轮 rollback/restore 由 S16 负责；
- 不扩展 CUDA、CI、provider 自治或训练权重迁移。

## 5. 下一步

R5C-S16：将 measured artifact batch 放入多轮真实 evidence continuation，并验证跨轮 lineage、stale artifact fail-closed、rollback 和恢复后的预算/账本一致性。
