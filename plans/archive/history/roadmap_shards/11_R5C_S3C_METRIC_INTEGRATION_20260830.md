# W7-R5C-S3C：真实候选指标接入 validation policy

## 目标

把 S3A 产生的真实 candidate shadow/holdout validation record 与 S3B 的独立 policy 绑定，
使 retention、lesion 和资源观测真正参与同一个候选判定，同时继续禁止在 validation 阶段
提交 topology。

## 已实现

- `TSKV8Adapter.evaluate_structural_candidate_gate()` 只接受当前 adapter 保存的、状态为
  `validated` 的 `StructuralCandidateValidation`，且要求对应 topology proposal 仍为 pending。
- 方法将 S3A 的 holdout validation score、调用方提供的 retention regression、lesion effect
  和 resource state 交给 S3B policy；validation evidence 与 metric evidence 合并去重后进入
  content-addressed decision。
- policy 通过时 proposal 继续保持 pending，等待后续 admission；policy 失败时 proposal 被
  原子标为 rejected，parent topology 与 budget 保持不变。
- gate decisions 写入 `structural_runtime` checkpoint；恢复后保留 decision 和 rejected 状态，
  不会重新进入 pending candidate 队列。

## Gate 证据

- `tests/taiji_native/test_structural_pressure.py`：6 个结构 evidence/pressure/bridge/validation
  回归通过，并覆盖 accepted decision 的 checkpoint continuation。
- `scripts/training/eval_taiji_structural_metric_integration.py`：真实 S3A shadow 记录接入
  policy，验证 accepted pending、failed rejected、topology/budget 不变和两条路径的 restore；
  报告 `gate.passed=true`。
- 最后一轮本 slice：36 个结构相关 pytest 通过，Ruff 与 compileall 通过。

## 当前边界

S3C 已完成指标到 policy 的接线，但 retention/lesion 数值仍由上游实验观测提供；本 slice 不
自动采集真实旧任务保持或 lesion 轨迹，也不执行 `commit_*`。因此还不能宣称长期自进化已经
完成。

## 下一阶段准入

R5C-S3D 才能把通过的 decision 接入一次性 admission transaction：trial checkpoint、资源
reservation、topology commit、restore/rollback 和 tombstone 必须同一生命周期闭环，失败时
不能留下半提交结构。
