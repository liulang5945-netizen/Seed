# R5C-S13：多区域 batch 的 replay validation artifact continuation

> 状态：已完成（2026-08-30）
>
> 本 slice 把 S12 的单候选 replay-bound validation artifact 扩展到真实多区域 candidate batch，取消批次 continuation 侧的手工 metrics 注入，并保留每个 candidate 独立 parent checkpoint、reservation、失败闭合和恢复语义。

## 1. 目标

S12 已经证明单候选可以消费由 Workbench replay 绑定的 validation artifact。S13 继续处理多区域 batch 的边界：不同候选的 parent checkpoint 可能因前一个 admission 而变化，因此 artifact 不能假设整个 batch 共用一个静态 parent；每个 candidate 必须在实际消费时绑定自己的当前 parent/trial checkpoint。

## 2. 实现合同

| 对象 | 合同 |
|---|---|
| artifact batch ledger | `StructuralValidationArtifactBatch` 按 batch、candidate 和 artifact digest 建立内容寻址集合，支持增量合并、完整性判断与 checkpoint roundtrip |
| batch continuation | `continue_structural_candidate_batch_from_validation_artifacts()` 只接收 artifact 与 holdout replay；不接收 holdout gain、retention regression、lesion effect 或 resource state 的手工集合 |
| parent binding | 每个 artifact 在消费前校验当前 candidate 的 parent checkpoint；前一个 candidate admission 后，后一个 candidate 必须用新 parent 重新生成 artifact |
| failure isolation | replay 错配、artifact candidate key 错配或 payload 无效只关闭对应 candidate、释放对应 reservation，不回滚已准入区域 |
| recovery | artifact batch ledger、candidate batch、admission 和 topology 一起进入 checkpoint；恢复后 artifact batch digest 不变，重复消费返回 `already_applied` |
| public boundary | `SeedRuntime` 提供 batch artifact continuation 公共入口；provider/frontend 不拥有 batch、topology、budget、policy 或 rollback |

## 3. Gate

canary：`scripts/training/eval_taiji_workbench_validation_artifact_batch.py`。

结果：`gate.passed=true`。

覆盖：

- 两个真实 Workbench 区域分别生成自己的 validation artifact；
- 第一候选先消费 artifact 并 admission，第二候选从恢复后的新 parent checkpoint 生成和消费 artifact；
- 第二候选错误 holdout replay 只在独立分支中 failed-closed，第一候选 topology 保持不变；
- 合法第二 artifact 让 batch 完成，两份 artifact digest 合并为完整 artifact batch；
- checkpoint restore 后 artifact batch digest 保持，重复提交两份 artifact 均幂等；
- 把第一候选 artifact 放到第二候选 key 下会 fail-closed，不能跨候选借用证据。

## 4. 未关闭边界

- 当前 canary 仍显式构造 artifact 中的 metric 值；这些值已经不再由 batch continuation 直接注入，但 metric producer 的独立 ownership 和实际测量由 S14 负责；
- 不证明开放域质量收益、无限预算、并行 topology commit、CUDA 或 CI 全量通过；CI 按用户决定继续暂缓。

## 5. 下一步

R5C-S14：建立独立 replay measurement owner，从 baseline/candidate/lesion/recovery 的真实观测计算验证指标，再生成 S12/S13 使用的 artifact。
