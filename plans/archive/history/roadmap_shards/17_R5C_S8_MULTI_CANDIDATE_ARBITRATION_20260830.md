# R5C-S8：多候选调度、确定性冲突仲裁与 reservation continuation

> 状态：已完成（2026-08-30）
>
> 本 slice 只解决多个结构候选同时出现时的排序、冲突、预算 reservation 和可恢复 continuation，不并行提交 topology，也不自动扩充长期预算。

## 1. 为什么需要 S8

S7 已能让一个 Workbench scheduler candidate 经过 shadow、policy 和 atomic admission，但真实长期运行不会只产生一个候选。不同 task slice、区域和结构操作可能同时产生候选；如果直接按队列顺序逐个执行，就会把列表顺序变成隐含学习规则，并且可能发生预算超售或同一区域冲突。

S8 将“候选发现”和“候选提交”之间增加一个显式 arbitration ledger：先对候选集建立内容寻址 batch，给出确定性 selected/deferred/rejected 决策和 reservation，再让 selected candidate 逐个复用 S7 的 continuation。

## 2. 决策与所有权边界

| 对象 | owner | S8 行为 | 禁止行为 |
|---|---|---|---|
| candidate batch 身份 | `taiji/structural_arbitration.py` | 绑定候选 payload、当前 topology digest、预算与可用预算 | 不依赖 Python dict 插入顺序或随机数 |
| 候选排序 | `taiji/adapter.py` | priority 降序、source tick 新鲜度降序、resource cost 升序、candidate id 稳定 tie-break | 不用“先到先得”隐藏优先级 |
| 冲突 | candidate `conflict_keys` + 既有 substrate 冲突判断 | 选中高优先级候选，冲突候选记录 deferred 原因 | 不把冲突候选静默丢弃 |
| 预算 | batch reservation ledger | 只预留，不改变实际 structural budget；不同 batch 不能重复占用 | 不自动扩预算、不跨 batch 超售 |
| topology admission | `TSKV8Adapter.continue_structural_candidate()` | selected candidate 逐个走既有 shadow→policy→admission | arbitration 不直接提交 topology |
| continuation | `StructuralCandidateBatch` | batch、状态、原因、剩余 reservation 全部 checkpoint | 不因本轮未提供输入而释放未处理候选 |

## 3. 实现合同

`StructuralCandidateBatch` 持久化：

- `candidate_ids`、selected/deferred/rejected 分组；
- 每个 candidate 的 `reserved/deferred/rejected/admitted/failed_closed/policy_rejected` 状态；
- conflict/budget/missing 的内容可读原因；
- `reserved_resource_cost` 与 `reservation_remaining`；
- batch 建立时的 structural budget、可用预算和 topology digest；
- arbitration digest、revision、batch status。

`arbitrate_structural_candidate_batch()` 的重复请求在候选集、topology 和预算基线相同时返回同一个 batch；reservation 的内部变化不会制造第二个 batch。只有 topology 或预算基线改变，才允许创建新的 arbitration batch。

`continue_structural_candidate_batch()` 只处理本次显式提供 continuation payload 的 selected candidate；未提供 payload 的 selected candidate 保持 `reserved`，支持先完成一个 admission、checkpoint、再继续剩余候选。每个 candidate 的 reservation 从 queue candidate 或已 materialize proposal 读取，避免调用顺序造成预算账本丢失。

## 4. Gate

唯一 canary：

`scripts/training/eval_taiji_structural_arbitration.py`

覆盖：

1. 同区域冲突候选按显式 priority 选出一个，低优先级候选 deferred；
2. 不同区域候选在预算内同时 reserved，预算外候选 deferred；
3. arbitration 不改变 topology 或实际 structural budget；
4. 重复候选集请求返回同一 batch；
5. 首个 selected candidate admission 后，剩余 reservation 保持；
6. native checkpoint restore 后继续第二个 selected candidate；
7. batch/候选状态与 reservation 精确完成，重复 continuation 不重复 admission；
8. deferred candidate 仍留在候选队列，可等待 topology/预算基线变化后重新 arbitration。

结果：`gate.passed=true`。

## 5. 当前边界

S8 仍不证明：

- 多候选并行 admission 的原子性；
- 自动增加 structural budget 或无限神经元扩张；
- 跨区域容量压力的长期预测与回滚策略；
- 开放域质量收益、CUDA 性能、训练 checkpoint 可靠性或 CI 全量通过。

## 6. 下一步

R5C-S9：把多个 batch 串成多轮长期 continuation，接入跨区域容量压力、admitted child checkpoint 回滚和 deferred candidate 的新证据再评估；要求多轮后仍能证明旧任务 retention、预算边界和可逆恢复。CI 按用户决定暂缓。
