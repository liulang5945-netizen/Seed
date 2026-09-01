# R5C-S20：跨轮 candidate/artifact/admission/rollback lineage 审计

## 目标

验证 evidence compaction 只是存储层操作，不会改写 Taiji 结构成长链上的 candidate、candidate batch、validation artifact、admission、reservation、budget 或 rollback 语义。

## 红合同

- 压缩前已经生成的 candidate/batch payload、source window digest 和 evidence id 必须保持不变。
- validation artifact 的 parent/trial checkpoint digest 必须继续绑定实际父分支；压缩后的新 checkpoint 不得接受旧 artifact。
- 旧 artifact fail-closed 时不得改变 topology、structural budget、reservation 或 candidate state。
- 从未压缩的原父 checkpoint 恢复后，同一 measured artifact 仍可走 shadow→policy→atomic admission，并且可 rollback。
- compacted checkpoint 恢复后，candidate、evidence consumption audit、pressure snapshot 和 measurement provenance 必须保持一致。

## 实现范围

本 slice 复用既有 Workbench measured artifact batch，不增加新的结构操作或资源预算。新增 `scripts/training/eval_taiji_structural_lineage_compaction.py`，在真实 `workspace.read` Outcome 生成的多区域 batch 上执行：

1. 生成候选与 measured validation artifact，并记录实际父 checkpoint。
2. 压缩已消费的旧 evidence window。
3. 在压缩后的 checkpoint 上重放旧 artifact，要求 parent mismatch fail-closed，并比较 topology/budget 前后摘要。
4. 从压缩前父 checkpoint 恢复，复用同一 artifact 完成 admission 与 rollback。
5. 恢复 compacted checkpoint，比较 candidate、ledger、audit 与 measurement provenance。

## Gate

报告：`reports/taiji_w7_r5c_s20_structural_lineage_compaction_20260830.json`

必须全部满足：

- `compaction_changes_only_evidence_storage`
- `old_artifact_fails_closed_after_compaction`
- `stale_artifact_does_not_mutate_topology_or_budget`
- `uncompacted_parent_can_still_admit_same_artifact`
- `admission_and_rollback_lineage_remain_bound`
- `compacted_checkpoint_restores_audit_and_candidate`
- `measurement_provenance_remains_available_on_parent_branch`

当前结果：`gate.passed=true`。

## 边界

本 slice 不处理 CI、CUDA、前端、物理删除、开放域质量或无限扩张；不提交、不改变 structural budget 默认值，也不把 canary 结果外推成全面自进化能力。

## 后继

唯一后继为 R5C-S21：长序列压缩保留、分支恢复与资源账本压力 Gate。
