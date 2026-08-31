# R5C-S33：artifact provenance 与 rollback/恢复闭环

## 目标

把 S32 的 replay-bound validation artifact 接入既有 rollback、retention maintenance 与 checkpoint continuation，明确区分“历史 artifact 可审计”和“候选仍可继续变更”，阻断已回滚候选被旧 artifact 重新激活。

## 实现结果

- `TSKV8Adapter.continue_structural_candidate_from_validation_artifact()` 在发现已回滚 lineage 时返回明确的 `rolled_back` 终态，不再把它误报为 `already_applied`。
- artifact 准入、第二候选准入和 rollback 均在 native checkpoint 前后保持 candidate、admission、rollback 与 artifact digest 绑定。
- 双候选回滚后，显式 retention 只淘汰无 live lineage 的完整 batch 子图，并同步淘汰 artifact、artifact-batch、admission、rollback 等关联记录。
- 淘汰后旧 batch/artifact 回放 fail-closed 且不改变当前 checkpoint；再次保存/加载不复活任何 rollback lineage。

## 验收证据

- 定向测试：`tests/taiji_native/test_structural_lineage_artifact_rollback.py` 与 S32 artifact 测试，`2 passed`。
- 静态检查：`ruff check taiji/adapter.py tests/taiji_native/test_structural_lineage_artifact_rollback.py scripts/training/eval_taiji_structural_lineage_artifact_rollback.py` 通过。
- 报告：`reports/taiji_w7_r5c_s33_structural_lineage_artifact_rollback_20260831.json`，`gate.passed=true`。
- 通过条件：artifact provenance 跨 rollback checkpoint 保留；旧 artifact 回放只能返回 `rolled_back`；终结子图可原子压缩；压缩后回放 fail-closed 且 checkpoint restore 不复活。

## 明确不声明

本 slice 只证明 native CPU artifact rollback、retention 与 checkpoint recovery，不声明开放域收益、无限扩张、自动增预算、CUDA、前端、Windows shell、CI 或通用智能。

## 下一步

**R5C-S34：多批次 artifact lineage 隔离与 retention pressure Gate**。在一个活动 batch 与多个终结 batch 并存时，验证压缩只作用于无 live lineage 的 artifact 子图，不污染活动 batch 的 replay、预算、拓扑和 checkpoint continuation。
