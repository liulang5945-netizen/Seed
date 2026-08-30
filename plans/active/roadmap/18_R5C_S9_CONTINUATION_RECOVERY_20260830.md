# R5C-S9：多轮 continuation、跨区域容量压力与可逆回滚

> 状态：已完成（2026-08-30）
>
> 本 slice 将 S8 的 candidate batch 放进跨 checkpoint 的长期生命周期，补齐容量压力观察、admitted child 回滚、预算重开和新 evidence 再仲裁；不扩展 CUDA、CI 或无限结构预算。

## 1. 目标

S8 已能对多个候选进行确定性排序和 reservation，但仍需要回答三个长期运行问题：

1. 多轮 continuation 中，未处理候选的 reservation 是否能跨 checkpoint 保持；
2. 区域容量、候选队列和 reservation 是否能形成只读压力信号，而不是用规模目标伪造成长理由；
3. admitted candidate 失败后，能否恢复 parent topology、重开预算，同时保留完整 audit，并允许新证据重新竞争。

## 2. 实现合同

| 对象 | 合同 |
|---|---|
| 容量压力 | `StructuralRegionCapacityPressure` 只读记录区域占用、capacity limit、待处理候选、reservation、budget 与派生 pressure；测量不能提交 topology 或扣预算 |
| 多轮 continuation | `continue_structural_candidate_batch()` 只处理本轮显式提供输入的 selected candidate；缺输入的候选保持 `reserved`，不自动释放或复活 |
| 回滚 | `rollback_structural_candidate_batch()` 必须绑定 admitted parent/child checkpoint，恢复 parent topology 和 budget，重复调用返回同一 rollback 结果 |
| 新证据 | 回滚后新 evidence 产生新的 candidate identity；它可以在新的 topology/budget 基线下重新仲裁旧 deferred 候选，但不能覆盖历史 batch 或静默 resurrection |
| checkpoint | batch、reservation、capacity snapshot、rollback audit、candidate state 和 digest 必须可恢复 |

## 3. Gate

唯一 canary：`scripts/training/eval_taiji_structural_continuation_recovery.py`。

覆盖：

- 两个区域的只读容量压力，以及 admission 前后的压力变化；
- 首轮只处理部分 selected candidate，剩余 reservation 跨 checkpoint 保持；
- 恢复后完成第二轮 admission；
- admitted candidate 回滚到 parent，预算由 0 重开为 1，区域 topology 恢复；
- rollback audit checkpoint roundtrip；
- 新 evidence 生成新 candidate，重新仲裁并优先于旧 deferred candidate；
- 重复 rollback 幂等，旧 deferred candidate 仍可审计。

结果：`gate.passed=true`。

## 4. 未关闭边界

- 尚未证明并行 topology admission 的原子性；
- 尚未证明长期预算会自动扩张，预算仍必须由显式合同提供；
- 尚未证明开放域质量收益或全面自进化；
- 真实 Workbench 多区域 evidence 目前仍需 S10 直接生成多个 candidate，S9 canary 中部分候选由测试场景构造；
- CI 全量回归按用户决定暂缓。

## 5. 下一步

R5C-S10：让两个以上真实 Workbench 区域与 task slice 的 Outcome 直接生成多个可追溯 candidate，并一次性接入 S8 arbitration、S9 continuation 与 rollback，清除测试 harness 组装候选的最后边界。
