# R5C-S32：重启后 replay-bound validation artifact 与 measured continuation

## 目标

把 S31 的 candidate-only replay 收紧为可寻址、可校验、可跨重启消费的 Workbench validation artifact。指标必须由独立 replay measurement owner 产生，准入层只能消费已绑定的事实，不能在恢复后临时重算指标来替代原始测量。

## 本 slice 的实现边界

- `WorkbenchStructuralValidationArtifact` 继续绑定 candidate、network/region、evidence ids、parent/trial checkpoint、holdout/retention/lesion/resource digest 和 measured metrics。
- 新增 native CPU 磁盘 canary：先生成真实 Workbench evidence，再由独立 measurement owner 生成 artifact，将 artifact 以 JSON 内容寻址 payload 保存，重启后只消费保存的 artifact 与 replay。
- 容量测量会记录 pressure snapshot，因此 artifact 的 parent checkpoint 必须在测量完成后保存；否则 artifact 会正确地因父状态不匹配而拒绝。
- 篡改 `measurement_digest` 必须在候选粒度 fail-closed，且不改变该分支的预算和拓扑；合法 artifact 必须能完成 validation→policy→atomic admission。
- 中间 checkpoint 恢复后继续第二候选；最终再次 checkpoint 后重复消费两个 artifact 必须返回 `already_applied`，artifact batch 必须保持 complete。

## 验收证据

- 定向测试：`tests/taiji_native/test_structural_lineage_restart_artifact.py`，`1 passed`。
- R5C-S18–S32 相关回归：`31 passed`；新增 S32 与 S29–S32 canary 均返回退出码 0。
- 报告：`reports/taiji_w7_r5c_s32_structural_lineage_restart_artifact_20260831.json`，`gate.passed=true`。
- 通过条件：measured parent 与 checkpoint 一致；artifact payload 内容往返一致；篡改 fail-closed；预算/拓扑保持；两个 measured artifact 跨 checkpoint 完成准入；重复消费幂等。

## 明确不声明

本 slice 只证明 native CPU checkpoint 与 replay-bound measured artifact 生命周期，不声明开放域能力、无限扩张、自动增预算、CUDA、前端、Windows shell、CI 或通用智能。

## 下一步

**R5C-S33：artifact provenance 与 rollback/恢复闭环**。在不复活已 rollback candidate 的前提下，验证 artifact ledger、rollback lineage、再次 checkpoint 和 retention maintenance 的联合语义。
