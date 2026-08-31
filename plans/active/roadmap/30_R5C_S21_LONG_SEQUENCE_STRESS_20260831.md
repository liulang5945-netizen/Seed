# R5C-S21：长序列压缩保留、分支恢复与资源账本压力 Gate

## 目标

把 S20 的单次跨轮 lineage 审计推进到三轮长序列：验证 active evidence、compacted provenance、pressure snapshot、candidate batch、reservation、admission/rollback 在有限容量下仍保持有界、原子和可恢复。

## 红合同

- 每个 `network_id:region_id` stream 至少保留最新 active window；未消费窗口在调度前不得被压缩。
- compacted window 总数不能超过显式 cap；溢出失败不得产生半提交、digest 漂移或集合变化。
- rollback 只能恢复当前候选绑定的父分支和预算，不能改变同 batch 另一候选的状态。
- 新 evidence 必须在前一轮 rollback/压缩后继续形成新的 source window 与 candidate batch。
- final checkpoint 恢复后，active/compacted window、pressure snapshot、candidate、admission/rollback 和 consumption audit 摘要必须一致。

## 实现范围

新增 `scripts/training/eval_taiji_structural_long_sequence_stress.py`，复用真实 Workbench `workspace.read` Outcome、measured artifact builder 和既有 batch lifecycle：

1. 三轮各记录 6 个真实 evidence window，并使用新的 task slice 与 target identity。
2. 每轮完成真实 measured candidate batch；第二、三轮回滚一个候选，检查预算重开和另一候选隔离。
3. 在 cap=16 下分别压缩，检查每 stream 2 个 active window 和累计 compacted 上限。
4. 在第三轮调度前尝试压缩未消费窗口，确认不会移动；在 cap=15 的临时压力下触发 OverflowError，确认 ledger 原子不变，再恢复 cap=16 完成压缩。
5. 保存并恢复 final native checkpoint，比较完整 audit/lineage 摘要。

## Gate

报告：`reports/taiji_w7_r5c_s21_structural_long_sequence_stress_20260831.json`

必须全部满足：

- `three_real_rounds_create_fresh_windows_and_batches`
- `per_stream_retention_is_bounded_after_each_compaction`
- `unconsumed_round_survives_pre_schedule_compaction`
- `rollback_reopens_budget_and_does_not_contaminate_other_candidate`
- `compaction_overflow_is_atomic`
- `final_long_sequence_stays_within_compacted_cap`
- `checkpoint_restores_long_sequence_audit_and_lineage`
- `round_one_parent_checkpoint_remains_recoverable`

当前结果：`gate.passed=true`；最终为 16 个 compacted windows、2 个 active windows。

## 边界

本 slice 不修改默认 structural budget，不处理 CI、CUDA、前端或物理删除，不把三轮 CPU/native canary 当成无限自进化或开放域质量证明。

## 后继

唯一后继为 R5C-S22：候选与审计账本的有界保留及 lineage 不复活 Gate。
