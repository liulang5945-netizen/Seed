# R5C-S18：多轮 ledger compactness 与跨轮 evidence 消费审计

> 状态：已完成（2026-08-30）
>
> 本 slice 把“可追溯历史”和“当前可消费 evidence”分成两个有界层次，避免多轮运行把历史 lineage 重复送入 pressure projection。

## 1. 目标

S16/S17 已完成多轮 measured continuation 以及 measurement/artifact provenance integrity，但 evidence ledger 仍把 sealed summaries 和 active evidence index 持续保留在同一层。S18 建立可恢复的跨轮消费审计，并允许只压缩已经被 scheduler 消费的旧窗口。

## 2. 实现合同

| 对象 | 合同 |
|---|---|
| consumption audit | `StructuralEvidenceLedger.audit_consumption()` 区分 evaluated、consumed、unconsumed、retained、compacted 和 orphaned digest，并按 network/region stream 给出消费状态 |
| compaction boundary | `compact_consumed_windows()` 只处理 scheduler 已消费且不是每个 stream 最新保留窗口的 sealed summary；未消费窗口不移动 |
| provenance record | 被压缩窗口保留 window/evidence digest、network/region、task slice、partition、tick 范围、observation count 和 consumed scheduler revision；不再作为 active pressure 输入 |
| active capacity | 被压缩窗口的 evidence digest 从 active index 移入有界 compacted provenance index，释放 active evidence capacity；重复的 compacted evidence 仍按原 digest 幂等，内容冲突 fail-closed |
| checkpoint | compacted provenance、active windows、两个 evidence index 和 ledger digest 一起进入 native checkpoint；旧无 compaction 字段的 ledger payload 继续读取 |
| adapter ownership | adapter 只暴露 audit/compaction 入口；scheduler 继续拥有消费 digest，pressure projection 继续只消费 active sealed summaries |

## 3. Gate

canary：`scripts/training/eval_taiji_structural_evidence_compaction.py`。

结果：`gate.passed=true`。

覆盖：

- 多 stream、多轮窗口的 evaluated/consumed/unconsumed 审计；
- 只压缩已消费旧窗口，并保留 task slice、partition、来源和消费 revision；
- 未消费窗口仍留在 active projection 集合；
- 压缩释放 active evidence capacity，新一轮 evidence 可继续进入；
- compacted evidence 重复 replay 幂等，篡改 provenance fail-closed；
- ledger payload roundtrip 等价，adapter audit/compaction 通过 native checkpoint 恢复；
- `StructuralEvidenceConsumptionAudit` 和 `StructuralEvidenceCompactionResult` 都是内容寻址对象。

定向回归：`tests/taiji_native/test_structural_evidence_window.py` 与 `tests/taiji_native/test_structural_evidence_compaction.py` 共 `7 passed`；Ruff 与 py_compile 通过。本 slice 未运行 CI。

## 4. 未关闭边界

- S18 没有把 compacted provenance 重新纳入 pressure projection，也没有声称跨轮开放域质量提升；
- candidate/batch/artifact/admission/rollback 的 compact 前后等价 pressure identity 由 S19 显式验证；
- 不扩展 CUDA、CI、provider 自治、无限预算或开放域语言质量。

## 5. 下一步

R5C-S19：验证压缩后 provenance-aware pressure projection 与跨轮候选边界，确保 compacted digest 只用于 lineage/消费审计，不会成为新的 pressure 输入。
