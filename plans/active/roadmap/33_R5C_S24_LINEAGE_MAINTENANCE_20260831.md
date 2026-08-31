# R5C-S24：运行时维护边界接入协同 lineage 压缩

状态：已完成（2026-08-31）

## 目标

把 S23 的跨账本 lineage retention 接入现有 structural maintenance boundary，同时保持维护行为显式、可恢复、可审计。S24 不新增后台线程，不扩大 structural budget，不执行物理删除，也不把 retention pressure 直接变成结构成长或准入信号。

## 实现边界

- `TSKV8Adapter.run_structural_maintenance_cycle()` 新增可选的 `lineage_retention_max_batches`。
- 调用方未提供该参数时，维护行为保持原有 candidate-only 路径，checkpoint 不增加 retention audit。
- 调用方显式提供正整数上限时，候选处理完成后运行 `compact_structural_lineage_history()` 一次。
- `StructuralLineageRetentionResult` 进入 structural checkpoint；source/target digest 排除 audit 自身，保证恢复后重复维护的结果可比较且幂等。
- 非法上限在进入结构变更前拒绝；异常路径保留 candidate、ledger、topology、budget 和上一份 audit。

## Gate

真实 Workbench evidence 的 native/CPU canary：

`scripts/training/eval_taiji_structural_lineage_maintenance.py`

必须同时证明：

1. 默认 maintenance 不触发 retention；
2. 显式 maintenance 只淘汰终结 lineage，且协同删除关联账本；
3. topology、structural budget、活动 reservation、pending/deferred/rollbackable lineage 不变；
4. retention audit 的 digest、保护项和 pressure 可随 checkpoint 恢复；
5. 非法上限原子失败；
6. 保护 lineage 超限时 pressure 可观测、重复维护幂等且不误删。

## 证据

- 报告：[taiji_w7_r5c_s24_structural_lineage_maintenance_20260831.json](../../../reports/taiji_w7_r5c_s24_structural_lineage_maintenance_20260831.json)
- Gate：`gate.passed=true`
- 定向回归：S18–S24 相关测试 `24 passed`
- 语法/lint：compileall、Ruff、`git diff --check` 通过

## 明确未覆盖

- 不声明无限增长、自动增加预算或全面自进化收益。
- 不声明 CUDA、Windows shell、前端视觉或完整 CI 通过。
- 不开放 SeedRuntime 的自动清理调度；显式 runtime 可观测入口留给 S25。

## 唯一后继

R5C-S25：把 lineage maintenance audit 接入 SeedRuntime 的显式可观测契约，向产品/runtime 层提供稳定 payload，同时保持 Taiji 对结构决策和 retention policy 的所有权。
