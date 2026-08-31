# R5C-S28：retention policy 可迁移生命周期与回滚

状态：已完成（2026-08-31）

## 目标

验证 retention policy 的 schema 演进可以显式进行、可审计、可恢复和可回滚，不因为版本变化重写既有 lineage、retention result、topology 或 budget。

## 实现边界

- `StructuralLineageRetentionPolicy` 支持明确的 v1→v2 相邻迁移；v2 增加 `mode=terminal_only`，不改变 v1 的安全语义。
- `StructuralLineageRetentionPolicyMigration` 保存 source/target/status/digest，并通过 Taiji adapter 与 SeedRuntime 提供显式 commit/rollback。
- 迁移要求 max_batches、mode 和 protection rules 保持一致；不能借 migration 关闭活动 lineage 保护。
- migration 与 policy/result 一起 checkpoint；加载时校验 digest、相邻版本和 safety semantics。
- 没有显式 migration 请求时不隐式升级；回滚只恢复 policy/migration 状态，不删除或重建旧 retention result。

## Gate

真实 Workbench evidence 的 native/CPU canary：

`scripts/training/eval_taiji_structural_lineage_policy_migration.py`

必须同时证明：

1. v1→latest migration 是显式、相邻且安全语义不变；
2. policy、migration、retention result、status checkpoint restore 一致；
3. rollback 恢复旧 policy，旧 audit、topology 和 budget 不变；
4. 无迁移请求不发生隐式升级；
5. 非法安全语义、篡改和不一致 checkpoint fail-closed 且原子不变。

## 证据

- 报告：[taiji_w7_r5c_s28_structural_lineage_policy_migration_20260831.json](../../../reports/taiji_w7_r5c_s28_structural_lineage_policy_migration_20260831.json)
- Gate：`gate.passed=true`
- 定向用例：`tests/taiji_native/test_structural_lineage_policy_migration.py` 为 `3 passed`
- S18–S28 相关回归：`37 passed`
- 语法/lint：Ruff、compileall、`git diff --check` 通过

## 明确未覆盖

- 不支持未知 revision 的自动迁移。
- 不把 policy migration 变成 structural growth、candidate admission 或后台清理触发器。
- 不声明无限增长、自动增加预算、开放域质量或全面自进化。
- 不声明 CUDA、Windows shell、前端视觉或完整 CI 通过。

## 唯一后继

R5C-S29：通过真实 SeedRuntime 磁盘 checkpoint 保存/加载验证 policy、migration、audit、status 和 lineage 的继续与回滚。
