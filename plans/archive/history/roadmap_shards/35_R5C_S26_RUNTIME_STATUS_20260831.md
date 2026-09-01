# R5C-S26：runtime 只读 structural maintenance 状态投影

状态：已完成（2026-08-31）

## 目标

让产品/runtime 层可以通过现有 `SeedRuntime.status()` 观察 Taiji structural maintenance 的最近 audit，而不读取内部 ledger，也不把状态查询变成维护或结构决策入口。

## 实现边界

- `SeedRuntime.structural_maintenance_status()` 返回版本化、只读的 status projection。
- `SeedRuntime.status()` 在 `structural_maintenance` 字段挂载同一 projection。
- 有 audit 时返回 structural runtime tick、完整 retention audit payload 和 pressure；无 audit 时返回明确 `no_audit` 空态。
- 查询不调用 `run_structural_maintenance_cycle()`，不生成 candidate、不执行 retention、不保存 checkpoint、不改变 topology/budget。
- status projection 只向上提供事实，不能作为结构成长、准入或 provider/frontend 行为的输入。

## Gate

真实 Workbench evidence 的 native/CPU canary：

`scripts/training/eval_taiji_structural_lineage_status.py`

必须同时证明：

1. 没有 audit 时空态字段完整且 checkpoint digest 不变；
2. 显式维护后 status 返回最近 audit、digest、保护项、删除项和 pressure；
3. status 查询本身无 structural side effect；
4. Seed checkpoint restore 后 status projection 一致；
5. status 查询不改变 topology 或 structural budget。

## 证据

- 报告：[taiji_w7_r5c_s26_structural_lineage_status_20260831.json](../../../../reports/taiji_w7_r5c_s26_structural_lineage_status_20260831.json)
- Gate：`gate.passed=true`
- 定向用例：`tests/taiji_native/test_structural_lineage_status.py` 为 `3 passed`
- 语法/lint：Ruff、compileall、`git diff --check` 通过

## 明确未覆盖

- 不让 status 反馈到 structural decision，不开放后台自动维护。
- 不声明无限增长、自动增加预算、开放域质量或全面自进化。
- 不声明 CUDA、Windows shell、前端视觉或完整 CI 通过。

## 唯一后继

R5C-S27：把裸 `max_batches` 收敛成版本化、内容寻址的 Taiji-owned lineage retention policy，统一 policy snapshot、checkpoint 和兼容入口语义。
