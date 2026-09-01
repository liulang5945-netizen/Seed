# W7-R5C-S3D：atomic structural admission

## 目标

把通过 S3C 的 candidate validation decision 接入一次真实、有限、可恢复的 topology admission，
验证结构确实只发生一次、预算只扣一次、重复调用幂等，并且 admission lineage 可在 checkpoint
恢复后继续读取。

## 已实现

- `StructuralAdmissionResult` 记录 candidate、proposal、policy decision digest、parent/child
  checkpoint、topology 前后 digest、budget 前后值和结果状态。
- `TSKV8Adapter.admit_structural_candidate()` 只接受当前 adapter 绑定的 validated record、
  passed policy decision 和 pending proposal，然后复用已有 operation-specific `commit_*`。
- 成功 admission 必须同时满足 proposal=`accepted`、topology digest 发生变化、budget 按
  resource cost 精确下降；重复 admission 返回同一结果，不重复增长。
- commit 返回拒绝或 after-state 不满足不变式时记录 `rejected`/`rolled_back`；结果写入
  `structural_runtime` checkpoint，恢复后保留 admission 状态。

## Gate 证据

- `scripts/training/eval_taiji_structural_admission.py`：验证 policy 绑定、单次 neuron growth、
  单次 budget 扣减、topology digest 变化、重复调用幂等和 checkpoint restore；报告
  `gate.passed=true`。
- 最后一轮本 slice：36 个结构相关 pytest、Ruff、compileall 均通过。

## 当前边界

S3D 只证明一个受限 neuron candidate 可以按完整链路 admission；它不代表开放域无限成长，
也没有启动多 seed、多 task slice 的稳定性、长期 retention 或 admission 后 lesion 对照。
现阶段没有训练、没有 CUDA、没有自动循环扩张。

## 下一阶段准入

R5C-S4 需要在独立 seed/task slice 上重复 admission 前后的收益、旧任务 retention、lesion
因果效应、资源和 rollback，并禁止单个 canary 直接扩大长期结构上限。
