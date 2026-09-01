# R5C-S7：Workbench candidate 验证闭环与长期 continuation

> 状态：已完成（2026-08-30）
>
> 本 slice 只把 S6 调度出来的 candidate 接入既有验证与原子准入生命周期；Workbench 提供可追溯事实，不能用“工具成功”绕过独立质量验证。

## 1. 目标

S6 已能从真实 Workbench Outcome 形成结构证据窗口并创建 candidate-only proposal，但 candidate 仍停留在 pending。S7 补齐 candidate 的可恢复 continuation：候选必须经过 shadow、独立 holdout、retention、lesion、resource 与 structural budget policy，只有 policy 通过才允许一次 atomic admission。

S7 不新增第二套 growth controller，也不把 Workbench 的成功状态直接变成“应该增长”。它只是把 S6 的候选来源与 R5C-S3D 已验证的生命周期连接起来。

## 2. 生命周期合同

```text
sealed Workbench evidence
  → scheduler candidate (pending)
  → parent checkpoint
  → operation-specific shadow validation
  → independent holdout / retention / lesion / resource policy
  → atomic topology admission
  → child checkpoint
```

每个阶段必须保留内容寻址身份和 parent lineage：

- candidate 绑定 pressure projection digest 与 parent checkpoint；
- shadow 只在 trial 中运行，记录 topology/budget before/after digest；
- holdout、retention、lesion 和 resource 指标由 continuation 调用显式提供，不从 Workbench success flag 猜测；
- policy 失败时保持 parent topology，候选进入可恢复 rejected 状态；
- policy 通过后才提交 topology 和精确 structural budget debit；
- 已 admitted candidate 重复执行必须返回同一 admission 结果，不能二次扣预算或重复加单元。

## 3. 实现边界

| 边界 | owner | S7 行为 |
|---|---|---|
| 调度 candidate | `taiji/structural_scheduler.py` + `taiji/adapter.py` | 从 checkpoint 恢复 candidate，交给 continuation，不直接 admission |
| candidate 验证 | `taiji/structural_validation.py` + `taiji/adapter.py` | 复用既有 shadow 与五维 policy，不复制另一套规则 |
| Workbench 接线 | `api/seed_runtime.py` | 暴露薄包装，传递 candidate id 与独立验证输入，不持有结构所有权 |
| 结构提交 | `taiji/adapter.py` | 只由 `admit_structural_candidate()` 原子改变 topology/预算 |
| 运行记录 | native checkpoint | 保存 validation、decision、admission、lineage 和 digest |

## 4. 已实现接口

`TSKV8Adapter.continue_structural_candidate()` 完成一次幂等 continuation：

1. 找到 scheduler candidate 并执行 `validate_structural_candidate_shadow()`；
2. 验证成功后调用 `evaluate_structural_candidate_gate()`；
3. 仅在 `decision.passed` 时调用 `admit_structural_candidate()`；
4. 返回统一的 validation/decision/admission payload；
5. admission 与拒绝记录进入 native checkpoint。

`SeedRuntime.continue_structural_candidate()` 只做 API 层转发，避免 Workbench/API 生成隐含结构规则。

## 5. Gate

唯一 canary：

`scripts/training/eval_taiji_workbench_growth_continuation.py`

覆盖：

1. 三个真实 `workspace.read` Outcome 形成两个 train task slice 和一个 holdout window；
2. scheduler 创建 candidate，candidate materialization 仍不改变 topology；
3. candidate 先通过 operation-specific shadow；
4. 显式 holdout、retention、lesion、resource 指标通过独立 policy；
5. admission 后 topology 增长一次且 structural budget 只扣减一次；
6. native checkpoint restore 保留 admission 记录、topology 和预算；
7. 对同一 candidate 重复 continuation 不重复 admission。

结果：`gate.passed=true`。

## 6. 明确未完成

S7 仍不证明：

- 多个 candidate 同时调度时的确定性排序与冲突仲裁；
- 跨区域并行 admission 或自动扩大 structural budget；
- 无限增长、全面自进化或开放域质量收益；
- CUDA 支持、训练 checkpoint 可靠性或 CI 全量通过。

## 7. 下一步

R5C-S8：建立可恢复的多候选 batch、确定性优先级与区域/预算冲突仲裁，覆盖 deferred/rejected candidate 的恢复、单候选 admission 后的剩余批次 continuation，以及重复调度输出 digest 一致。CI 按用户决定暂缓。
