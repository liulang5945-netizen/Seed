# R5C-S31：重启后的候选准入、回滚与 checkpoint continuation

状态：已完成（2026-08-31）

## 目标

把 S30 的“新 evidence 可继续流动”推进到已有结构成长合同中的一次受限候选生命周期，确认 restart 不会绕过 holdout、validation、资源预算、topology transaction 或 rollback lineage。

## 实现边界

- 新 batch 的 candidate 在重启后先通过 candidate-only holdout replay，再进入既有五类 validation gate。
- 第一 candidate admission 后保存中间 checkpoint；恢复后继续第二 candidate。
- 第二 candidate admission 后显式 rollback，恢复父结构和 structural budget，再保存/加载最终 checkpoint。
- `continue_structural_candidate_batch()` 拒绝不属于当前 selected batch 的 candidate，避免跨批次请求被静默忽略；拒绝前后 native checkpoint digest 不变。
- policy migration provenance、admission、rollback 和 topology/budget 状态随 checkpoint 保持。

## Gate

真实 Workbench evidence + native/CPU restart candidate canary：

`scripts/training/eval_taiji_structural_lineage_restart_admission.py`

必须证明：

1. 第一 candidate 可在重启后完成 holdout replay、validation gate 和 atomic admission；
2. 中间 checkpoint 恢复后第二 candidate 可完成同一批次 continuation；
3. rollback 只回退目标 candidate，恢复父 topology/budget，并可再次 checkpoint；
4. policy migration/rollback lineage、candidate admission/rollback 记录可恢复；
5. 跨批次 candidate continuation fail-closed 且不改变当前 checkpoint。

## 证据

- 定向用例：[test_structural_lineage_restart_admission.py](../../../tests/taiji_native/test_structural_lineage_restart_admission.py) 为 `1 passed`。
- Canary：[taiji_w7_r5c_s31_structural_lineage_restart_admission_20260831.json](../../../reports/taiji_w7_r5c_s31_structural_lineage_restart_admission_20260831.json)，`gate.passed=true`。
- Ruff 定向检查通过；本轮不运行 CI，不把本地 CPU Gate 扩大为远端结论。

## 明确未覆盖

- 不把两次 admission 视为无限自扩张、开放域质量或完整自进化证明。
- 不开始 CUDA、前端现场取证、完整 CI 或从零训练。

## 唯一后继

R5C-S32：把 restart candidate continuation 的 holdout 事实提升为 replay-bound validation artifact 与独立 measured continuation。
