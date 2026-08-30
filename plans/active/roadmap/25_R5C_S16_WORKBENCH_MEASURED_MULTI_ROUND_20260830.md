# R5C-S16：多轮 measured evidence continuation 与 rollback

> 状态：已完成（2026-08-30）
>
> 本 slice 将 S15 的 measured artifact batch 放入两轮真实 Workbench evidence 循环，并修复多区域 scheduler 使用全局 cooldown 游标造成的区域饿死；每轮都必须沿 checkpoint lineage 重新测量、重新绑定 artifact，并支持独立回滚。

## 1. 目标

S15 已经证明一轮多区域 measured artifact batch 可以逐候选 admission。S16 继续验证它能否像受约束的生长过程一样跨轮运行：新 evidence 形成新窗口和新候选，旧 artifact 不能跨 parent 状态复用，容量/资源 Gate 不被绕过，rollback 只恢复目标轮次的局部结构。

## 2. 实现合同

| 对象 | 合同 |
|---|---|
| scheduler cursor | `StructuralGrowthScheduleState` 的 cooldown 按 `network_id:region_id` stream 隔离；`last_evaluated_tick` 保留为旧 checkpoint 的聚合兼容字段，不再作为新多区域 stream 的共享阻塞条件 |
| round input | 每轮使用新的真实 Workbench task slice、sealed window digest 和结构目标 identity；历史窗口可参与长期 pressure/provenance，但 scheduler 只消费未评估窗口 |
| parent lineage | 每个 measured artifact 绑定当轮 candidate 的当前 parent/trial checkpoint；前一候选 admission 后，后一候选必须重新测量、重新生成 artifact |
| stale rejection | 旧轮 artifact、旧 parent artifact 或错误 candidate mapping 必须 fail-closed，不得新增 topology、扣预算或污染其他 candidate |
| rollback | rollback 恢复被选候选的 parent topology、budget 和 lifecycle audit；其他已 admission region 保持不变，checkpoint restore 后重复 rollback 幂等 |
| measurement ownership | holdout/retention/lesion/resource metrics 仍由 S14 measurement owner 从 raw replay/resource observation 计算，不接受手工 batch 分数 |

## 3. Gate

canary：`scripts/training/eval_taiji_workbench_measured_multi_round.py`。

结果：`gate.passed=true`。

覆盖：

- 两轮各 6 次真实 `workspace.read` 成功 Outcome，第二轮使用全新的 task slice 和 sealed window digest；
- 两轮各生成两个区域 candidate batch，第二轮 candidate/batch/artifact 与第一轮不重用；
- 第一轮两个候选和第二轮两个候选都使用 measured artifact 完成逐候选 admission；
- 第二轮旧 parent artifact 在第一候选 admission 后 fail-closed，失败分支 topology/budget 不变化；
- 第二轮最后一个 admitted candidate rollback 后恢复到第一候选 admission 的 checkpoint，其他 region 不受污染；
- rollback checkpoint restore 与重复 rollback 幂等，两个 artifact batch 和至少四个 admission lineage 仍可追溯；
- stream-scoped scheduler 回归测试验证 docs tick 12 不会阻塞 code tick 9，旧 scheduler checkpoint 仍保守兼容。

## 4. 根因修复记录

S16 首次 Gate 发现 `schedule_structural_growth_from_evidence()` 使用单一 `last_evaluated_tick`。多区域 batch 先处理 tick 12 的 docs 后，tick 9 的 code 被误判为 cooldown，导致 code region 无候选。修复为 scheduler state 保存 stream cursor，并在 adapter 中以 `f"{network_id}:{region_id}"` 作为 cooldown key；这不是改 canary 顺序，而是修复跨区域调度的所有权边界。

同一 Gate 还发现 canary 必须在“产生 artifact 的 model”上继续消费 artifact，因为容量 pressure measurement 会写入可 checkpoint 的观测 ledger；S16 将 artifact 生成与 continuation 放在同一 checkpoint 分支，避免把合法 artifact 错判成 parent mismatch。

## 5. 未关闭边界

- S16 证明的是受控的两轮结构生命周期，不是无限扩张、自动增加预算、全面自进化或开放域收益；
- pressure projection 仍保留历史 evidence 参与长期聚合，S17 负责把历史 provenance 与新触发事实的完整性校验进一步收紧；
- 不扩展 CUDA、CI、provider 自治或训练权重迁移；CI 全量验收按用户决定继续暂缓。

## 6. 下一步

R5C-S17：对 measurement payload 做 digest 重算与 tamper fail-closed，并闭合多轮 evidence provenance 与 scheduler 消费边界。
