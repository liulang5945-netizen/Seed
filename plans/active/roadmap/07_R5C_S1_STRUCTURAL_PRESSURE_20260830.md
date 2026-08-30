# W7-R5C-S1：跨任务结构成长 pressure 投影

## 目标

把已封存的 evidence window 投影为可供后续 growth controller 审核的事实指标，同时明确训练证据、holdout 证据和 retention 证据的边界。S1 仍不直接产生或提交 structural proposal。

## 已实现

- `StructuralRuntimeObservation` 增加显式 `task_slice_id` 与 `partition`：旧 payload 兼容为 `runtime`，原生 adapter 运行时使用当前 episode 作为 task slice，并将 `holdout` 与 `train` 分开记录。
- `StructuralEvidenceLedger` 按 network、region、task slice、partition 分窗，避免 train/holdout 混入同一窗口；digest 索引和 checkpoint 继续覆盖全部上下文。
- `taiji/structural_pressure.py`
  - `project_structural_growth_pressure()` 只接受 sealed summaries。
  - 要求同一 substrate、唯一 window digest、至少两个独立 train task slices 和配置的 train window 数量。
  - holdout/retention 只作为验证计量，不进入 train prediction error 或 resource state 均值。
  - 返回 content-addressed `StructuralGrowthEvidenceProjection`，不调用 controller、不写 topology。
- `scripts/training/eval_taiji_structural_pressure.py`
  - 验证跨任务 train、独立 holdout、浮点稳定性、projection roundtrip、单 task slice 拒绝、ledger 不变。
  - 报告 `taiji_w7_r5c_s1_structural_pressure_20260830.json`，`gate.passed=true`。

## 明确未做

- 未把 projection 直接转换成 growth controller 的 state mutation。
- 未创建 candidate、未消耗 structural budget、未改变 neuron/region/topology。
- 未把 `runtime` 默认分区伪装成可用于泛化证明的 train evidence。

## R5C-S2 准入

下一步才评审 projection 到既有 `AdaptiveStructuralGrowthController` 的单向桥接：需要 projection digest 去重、跨 checkpoint continuation、父结构保存、holdout/retention Gate、candidate-only 输出和失败原子性。S2 不能直接执行 admit；必须继续复用现有 topology ledger 的 shadow/holdout/lesion/rollback。
