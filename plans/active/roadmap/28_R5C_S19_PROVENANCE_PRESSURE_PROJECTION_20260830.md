# R5C-S19：压缩后 provenance-aware pressure projection 与跨轮候选边界

> 状态：已完成（2026-08-30）
>
> 本 slice 解决 S18 的关键风险：压缩旧 evidence 后，pressure 聚合不能因为存储形态变化而漂移；同时历史 digest 不能重新变成新的触发事实。

## 1. 目标

S18 已将已消费旧窗口从 active ledger 移入有界 provenance record，但如果 pressure projection 只读取 active summaries，可能丢失构成原候选 identity 所需的 train/holdout/retention 聚合。S19 为已消费历史建立显式、只读、内容寻址的 pressure snapshot，并让 scheduler 仍只根据未评估 active sealed windows 触发新一轮。

## 2. 实现合同

| 对象 | 合同 |
|---|---|
| pressure snapshot | `StructuralEvidencePressureSnapshot` 保存已消费窗口的 source window digest、train/holdout/retention 窗口数、task slice、prediction/error/resource/holdout 聚合和 evidence ids |
| projection | `project_structural_growth_pressure()` 可显式接收 active summaries + `historical_snapshots`，重建 compact 前相同的 pressure projection digest |
| trigger boundary | scheduler 的 `unseen` 仍只从 active `sealed_summaries` 计算；compacted digest 和 snapshot 不会单独触发新的 candidate |
| candidate dedupe | 已生成 candidate 的 projection/source lineage、scheduler evaluated digest 和 checkpoint 不被 compaction 改写；无新 active window 的重复调度返回 `no_new_sealed_window` |
| checkpoint | ledger pressure snapshots、candidate、audit、scheduler state 一起恢复；tampered snapshot 或 ledger digest 失配必须 fail-closed |

## 3. Gate

canary：`scripts/training/eval_taiji_structural_provenance_projection.py`。

结果：`gate.passed=true`。

覆盖：

- 真实 Taiji structural scheduler 从 active train/holdout evidence 创建 candidate；
- 压缩已消费旧窗口后，active summaries + pressure snapshot 的 projection digest 与压缩前完全一致；
- compacted source digest 不再出现在 active trigger 集合；
- 重复 schedule 不新增 candidate；
- candidate、pressure snapshot、consumption audit 和 ledger checkpoint 恢复等价；
- tampered pressure snapshot 在 native checkpoint 恢复时 fail-closed。

定向回归：`tests/taiji_native/test_structural_evidence_compaction.py`、`test_structural_evidence_window.py`、`test_structural_pressure.py`、`test_structural_scheduler.py` 共 `17 passed`；Ruff 与 py_compile 通过。本 slice 未运行 CI。

## 4. 未关闭边界

- S19 只保证 evidence/pressure/candidate 边界，不把 compacted history 变成新的 runtime evidence，也不扩大开放域质量结论；
- candidate batch、validation artifact、admission、rollback 在多轮压缩前后的全链路 lineage 摘要由 S20 审计；
- 不扩展 CUDA、CI、provider 自治、无限预算或开放域语言质量。

## 5. 下一步

R5C-S20：跨轮 candidate/artifact/admission/rollback lineage 审计，验证 evidence 压缩不会改变 parent/child、reservation、预算恢复或旧 artifact 的 stale 拒绝边界。
