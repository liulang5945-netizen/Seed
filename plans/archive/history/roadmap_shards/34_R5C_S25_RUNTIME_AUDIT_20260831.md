# R5C-S25：SeedRuntime 显式 lineage maintenance audit 可观测契约

状态：已完成（2026-08-31）

## 目标

把 S24 的 Taiji adapter maintenance 结果投影到 SeedRuntime 的显式调用面，提供稳定、可校验、可 checkpoint 恢复的 audit payload。SeedRuntime 只做编排和投影；candidate lifecycle、retention policy、topology 与 structural budget 仍由 Taiji 独占。

## 实现边界

- 新增 `StructuralMaintenanceAudit`，统一保存本次 candidate maintenance results、当次 lineage retention result 和 structural runtime tick。
- 新增 `SeedRuntime.run_structural_maintenance_cycle()`，传递显式 retention 上限并返回 audit payload。
- 默认调用只返回当次 candidate maintenance 结果，不把 checkpoint 中恢复的旧 retention audit 伪装成新动作。
- audit payload 使用 canonical digest；解析时重建 candidate result 和 retention result，digest 不一致即 fail-closed。
- runtime 不启动后台维护、不自动保存外部工作区、不执行 provider/frontend/Workbench 副作用。

## Gate

真实 Workbench evidence 的 native/CPU canary：

`scripts/training/eval_taiji_structural_lineage_runtime.py`

必须同时证明：

1. 默认 runtime 调用返回合法 audit 且不触发 retention；
2. 显式正上限能投影本次 retention result；
3. Seed checkpoint restore 保留 Taiji audit state 与活动 lineage；
4. 恢复后默认调用不重放旧 audit；
5. topology/budget 不因 projection 变化；
6. 非法上限和篡改 audit payload fail-closed。

## 证据

- 报告：[taiji_w7_r5c_s25_structural_lineage_runtime_20260831.json](../../../../reports/taiji_w7_r5c_s25_structural_lineage_runtime_20260831.json)
- Gate：`gate.passed=true`
- 定向回归：S18–S25 相关测试 `27 passed`
- 语法/lint：Ruff、compileall、`git diff --check` 通过

## 明确未覆盖

- 不把 status projection 当作结构决策输入，不开放 runtime 自动清理。
- 不声明无限增长、自动增加预算、开放域质量或全面自进化。
- 不声明 CUDA、Windows shell、前端视觉或完整 CI 通过。

## 唯一后继

R5C-S26：将最近一次 structural maintenance audit 以纯只读状态投影接入 runtime status，验证没有 audit 时的空态、恢复一致性与零副作用。
