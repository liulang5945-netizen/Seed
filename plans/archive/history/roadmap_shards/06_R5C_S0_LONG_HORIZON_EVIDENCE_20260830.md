# W7-R5C-S0：长期结构证据观察窗口

## 目标

把原生运行时产生的 `StructuralRuntimeObservation` 从“最近若干 tick 的调试记录”提升为可审计的长期证据输入。S0 只负责保存、去重、分窗、聚合和恢复，不负责提出或提交结构变更。

## 已实现

- `taiji/structural_evidence.py`
  - `StructuralEvidenceWindow`：单一 network/region 的有界、单调 tick 窗口；满容量自动封存。
  - `StructuralEvidenceWindowSummary`：封存后的均值、tick 区间、预测误差样本数、evidence IDs 和内容 digest。
  - `StructuralEvidenceLedger`：跨窗口去重、全局 evidence digest 索引、窗口容量/封存窗口/索引容量上限、checkpoint roundtrip。
  - 同一 evidence ID 的同内容重放是幂等的；同 ID 不同内容、超出容量、顺序回退均 fail-closed。
- `taiji/adapter.py`
  - standalone adaptive neuron tick 和 cross-region network tick 都进入同一 ledger。
  - `native_checkpoint()` 的 `structural_runtime.evidence_ledger` 保留开放窗口、封存摘要、去重索引和窗口序号。
  - 旧 checkpoint 没有 ledger 时，从仍保留的 runtime observations 兼容重建，不改变旧结构。
- `tests/taiji_native/test_structural_evidence_window.py`
  - 内容寻址和重复 evidence 幂等；冲突重用拒绝。
  - evidence index 容量失败保持 ledger 原子不变。
  - adapter 真实 tick 记录和 checkpoint 恢复 digest 一致。
- `scripts/training/eval_taiji_long_horizon_evidence.py`
  - canary 报告 `taiji_w7_r5c_s0_long_horizon_evidence_20260830.json`，当前 `gate.passed=true`。

## 架构边界

S0 不把 observation 的单次分数直接喂给 `AdaptiveStructuralGrowthController`，不读取 holdout 标签，不把 scale target 当压力，不创建 neuron/region，不改变 topology，不调用 executor，也不把 evidence ledger 当认知主体。窗口只提供后续 S1 使用的可追溯事实。

## 后续准入

R5C-S1 才能评审“窗口摘要如何形成 growth pressure”：必须区分 observation window 与 holdout/retention 评估，明确跨任务 slice 的证据来源、资源压力和 proposal 的父 checkpoint；单个窗口不得直接 admit。CI 完整 Workbench 回归继续作为后置统一门禁，不在本 slice 内处理。
