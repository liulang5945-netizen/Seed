# R5C-S10：真实 Workbench 多区域证据驱动的候选批次调度

> 状态：已完成（2026-08-30）
>
> 本 slice 清除 S8/S9 canary 中“候选由测试 harness 组装”的边界：多个真实 Workbench 区域的成功 Outcome 先形成内容寻址结构证据，再由运行时一次性生成多个 candidate 并进入既有 arbitration；不在本 slice admission topology、不扩预算。

## 1. 目标

S6/S7 已证明单个真实 Workbench evidence stream 可以生成并延续 candidate，S8/S9 已证明多个 candidate 可以仲裁、跨 checkpoint continuation 和 rollback。S10 进一步要求这些 candidate 都从真实 Workbench Outcome 产生，而不是在 evaluator 内直接构造 `StructuralProposalCandidate`。

## 2. 实现合同

| 对象 | 合同 |
|---|---|
| evidence 来源 | 每个结构 observation 必须绑定真实 `WorkbenchOutcome` 的 request/intent/call/capability snapshot 与 outcome digest；provider/frontend 不得伪造 evidence |
| 区域隔离 | `network_id/region_id/task_slice_id/partition` 明确区分两个以上 Workbench 区域；每个区域至少需要两个独立 train task slices 和一个 holdout window |
| candidate 生成 | 每个区域经过既有 sealed-window→pressure→candidate-only bridge，candidate id/evidence ids/source tick/parent checkpoint 可追溯；不允许由前端直接提交 topology 操作 |
| batch 仲裁 | 多个 candidate 在一次 `StructuralCandidateBatch` 中按既有显式排序、冲突和 reservation 合同处理；仲裁不能改变 topology 或实际 structural budget |
| checkpoint | 批次请求 digest、区域、源 window digests、candidate ids、batch id 与 scheduler revision 必须可恢复，重复调度必须幂等 |

## 3. 实现内容

- `taiji/structural_scheduler.py` 新增 `StructuralWorkbenchBatchScheduleResult`，持久化多区域批次调度事实；
- `taiji/adapter.py` 新增 `schedule_structural_candidate_batch_from_workbench_evidence()`，规范化多区域请求，依次消费各区域新 sealed windows，再把新 candidate 交给既有 arbitration；
- `api/seed_runtime.py` 提供同边界的 runtime wrapper，但不把结构控制暴露给前端产品路径；
- checkpoint/restore 保存多区域调度结果；
- `scripts/training/eval_taiji_workbench_multi_region_batch.py` 执行真实 `workspace.read`，不直接构造候选。

## 4. Gate

canary：`scripts/training/eval_taiji_workbench_multi_region_batch.py`。

结果：`gate.passed=true`。

关键证据：

- 6 次真实 `workspace.read` 全部成功；
- `workbench.code` 与 `workbench.docs` 各自产生 2 个 train 窗口和 1 个 holdout 窗口；
- 两个区域生成两个不同的 evidence-derived candidate，并进入同一 batch；
- batch 选中两个候选，reservation 为 2，topology 与 structural budget 保持不变；
- checkpoint restore 后 batch、source windows、scheduler revision 和 request digest 保持；
- 重复多区域调度返回同一 batch 结果，未重复创建候选。

## 5. 未关闭边界

- S10 只证明真实多区域 candidate batch 的生成和仲裁，不证明多区域 admission 已完成；
- holdout/retention/lesion/resource 指标仍需通过 S7 生命周期输入，不能从 Workbench 成功自动推断；
- 不扩展 CUDA、CI、无限 structural budget 或开放域语言质量。

## 6. 下一步

R5C-S11：把 S10 的真实多区域 batch 接入 shadow→policy→admission→rollback 全生命周期，证明一个候选失败时其他区域候选和 reservation 不被污染。
