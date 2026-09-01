# R5C-S51：verified measurement artifact bridge Gate

## 目标

把 S49 的 measurement-fact sidecar 与 runtime 消费边界连接起来：外部 store bridge 增加显式 `require_verified_measurements` 严格模式，只有 artifact 本体和匹配 sidecar 均通过独立校验时才允许消费。S51 时代的兼容调用继续支持已有 artifact-only legacy 证据；当前默认边界已由 S52 的显式 policy 收口，避免历史 checkpoint 被隐式破坏。

## 设计边界

- `StructuralValidationArtifactStore.load_verified_artifact()` 先加载 artifact，再加载并验证对应 measurement sidecar，digest 不匹配、缺失、篡改或 legacy 均 fail-closed。
- `SeedRuntime.continue_structural_candidate_batch_from_artifact_store()` 增加显式布尔门；严格模式只改变 store resolution，不改变 native replay、admission、rollback、retention 或预算语义。
- 多 candidate bridge 必须先完成全部 strict resolution，再进入既有 all-or-nothing batch contract；任一 legacy/缺失 sidecar 不得提前消费 sibling。
- 默认 `require_verified_measurements=False` 保留向后兼容；严格模式由调用方明确选择，不把 store 变成认知主体或自动升级器。

## Gate

真实 native/CPU canary 必须证明：

1. verified bundle 在严格模式可通过并完成既有 runtime contract；
2. legacy artifact 在默认模式仍可消费，在严格模式 fail-closed；
3. 多 candidate 中任一 strict resolution 失败时不发生部分 admission、预算扣减或 checkpoint 变化；
4. sidecar 篡改/缺失在 runtime 变更前被拒绝，重复 verified 消费仍保持既有幂等语义。

## 验证入口

- 定向测试：`tests/taiji_native/test_runtime_verified_measurement_bridge.py`
- CPU canary：`scripts/training/eval_taiji_runtime_verified_measurement_bridge.py`
- 报告：`reports/taiji_w7_r5c_s51_runtime_verified_measurement_bridge_20260831.json`

本 slice 仍不处理 CI、CUDA、前端、Windows shell、默认强制迁移、自动删除、无限扩张、开放域收益或通用智能声明。
