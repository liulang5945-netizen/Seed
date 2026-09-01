# R5C-S6：Workbench evidence 驱动的长期增长调度

> 状态：已完成（2026-08-30）
>
> 本 slice 只建立“真实 Workbench Outcome → 结构证据 → candidate-only 调度”的可恢复接线，不把工具成功直接解释成神经元增长，也不执行 candidate admission。

## 1. 为什么需要这一层

R5C-S0–S5 已经具备证据窗口、跨 task slice pressure、candidate bridge、shadow validation、policy、admission、rollback 和跨 checkpoint continuation，但此前主要由 Taiji 内部 evaluator 直接构造 `StructuralRuntimeObservation`。如果真实 Workbench 的执行结果不能进入同一条内容寻址链，结构成长就仍然是孤立实验，而不是项目可用的自进化路径。

S6 的目标是补齐环境边界：Workbench 负责真实执行和返回 `WorkbenchOutcome`；评估器负责提供可审计的 usage、resource pressure、prediction error、learning gain 和 holdout transfer；Taiji 只负责把这些显式指标转成结构证据、聚合压力并排队 candidate。任何一层都不能替另一层发明指标或越权改变拓扑。

## 2. 所有权边界

| 对象 | 唯一 owner | S6 允许的动作 | S6 禁止的动作 |
|---|---|---|---|
| Workbench 执行结果 | `seed_platform/workbench.py` + `api/seed_runtime.py` | 绑定 request/intent/call/capability snapshot 和 outcome digest | 由成功状态推导“应增加神经元” |
| 结构证据窗口 | `taiji/structural_evidence.py` | 追加、封存、去重、checkpoint | 混写 train 与 holdout/retention |
| pressure projection | `taiji/structural_pressure.py` | 跨 task slice 聚合已封存窗口 | 使用单次 demo、目标规模或人工标签 |
| growth scheduler | `taiji/structural_scheduler.py` + `taiji/adapter.py` | 仅消费新窗口、调用 candidate bridge、保存 cursor | 直接 admission、自动扩预算、修改 topology |
| candidate lifecycle | 既有 `taiji/structural_validation.py` / `taiji/adapter.py` | 保持 pending，交给后续 S7 验证 | 绕过 shadow/holdout/lesion/policy |

## 3. 已实现合同

### 3.1 Workbench structural evidence

`WorkbenchStructuralEvidence` 由真实 `WorkbenchOutcome` 创建，内容身份至少绑定：

- `request_id`、`intent_id`、`call_id`；
- capability 与 capability snapshot；
- `outcome_digest`；
- `task_slice_id` 与 `partition`；
- 外部显式提供的 usage/resource/prediction/learning/transfer 指标。

`evidence_id` 使用不包含自身 ID 的 identity payload 计算，再把 ID 写入返回 payload，避免身份计算递归。这个边界由定向 canary 覆盖。

### 3.2 独立结构运行时钟

Workbench `Outcome.tick` 表示动作开始时刻，第一次合法执行可以是 `0`；结构证据要求严格递增的正时钟。因此 S6 使用 adapter 内部可 checkpoint 的 `structural_runtime_tick`，每次外部证据在 append 前取下一个结构 tick。该时钟不覆盖产品动作时钟，也不通过硬编码把动作结果的 tick 强行改写。

### 3.3 可恢复调度

`StructuralGrowthScheduleState` 保存：

- 调度间隔与 `last_evaluated_tick`；
- 已消费的 sealed-window digest 集合；
- scheduler revision。

调度流程为：

```text
real Workbench Outcome
  → WorkbenchStructuralEvidence
  → StructuralRuntimeObservation
  → sealed window
  → train/holdout/retention-isolated projection
  → existing growth controller
  → pending candidate
```

同一窗口不会重复触发；checkpoint restore 后重复调度返回 `waiting/no_new_sealed_window`。调度失败采用 `failed_closed`，不提交 topology。

## 4. Gate 与证据

唯一 canary：

`scripts/training/eval_taiji_workbench_structural_scheduler.py`

它验证：

1. 三次真实 `workspace.read` 均返回成功的 Workbench Outcome；
2. 每次成功结果都返回内容寻址 structural evidence；
3. 两个独立 train task slice 与一个 holdout window 被分别封存；
4. scheduler 通过既有 pressure projection 创建一个 candidate；
5. candidate 仍是 pending，adaptive region topology 不变；
6. scheduler cursor、candidate 和 evidence windows 在 native checkpoint restore 后保留；
7. 重复调度不重新创建 candidate。

结果：`gate.passed=true`。

## 5. 明确未完成

S6 不证明：

- candidate 已经通过 shadow、holdout、retention、lesion 或 admission；
- 结构预算会自动增加；
- 只靠 Workbench 成功状态就能判断增长；
- 无限结构扩张、全面自进化或开放域质量收益；
- CUDA 支持、CI 全量通过或训练 checkpoint 可靠性。

## 6. 下一步

R5C-S7：将 scheduler 创建的 candidate 接入既有 shadow→holdout→retention→lesion→policy→admission 验证闭环，并在多轮 Workbench continuation 中验证旧任务保持、资源成本、重复调度幂等与 rejected candidate 不复活。CI 仍按用户决定暂缓，待功能链路继续收敛后统一处理。