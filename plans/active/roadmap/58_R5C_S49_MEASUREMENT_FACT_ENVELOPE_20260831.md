# R5C-S49：measured artifact 的 measurement-fact sidecar Gate

## 目标

补齐 S46–S48 暴露出的完整性边界：`WorkbenchStructuralValidationArtifact` 当前携带 measurement digest 和聚合指标，但外部 store 没有 measurement facts 本体，因此只能验证 digest 字符串与 artifact 内容寻址，不能独立重算 measurement digest。S49 增加与 `measurement_digest` 内容寻址的 canonical measurement sidecar，使新写入的 measured artifact 可被 store 独立复核。

## 设计边界

- 新入口 `put_measured_artifact(artifact, measurements)` 同时校验两者的 `measurement_digest`，并分别以 artifact digest、measurement digest 写入 canonical JSON。
- `load_measurements()` 通过 `StructuralValidationMeasurements.from_payload()` 重算 measurement digest，再验证文件名和 canonical bytes。
- `inventory()` 识别并审计 sidecar；新 bundle 的 measurement status 为 `verified`。保留早期 `put(artifact)` 产生的 legacy artifact，但只标记为 `legacy_unverified`，不伪造已经不存在的测量事实。
- sidecar 与 artifact 都是 immutable、内容寻址、不可覆盖；冲突、篡改、非法 sidecar 和未被 artifact 引用的 sidecar fail-closed，不自动删除或修复。
- runtime bridge 继续只消费 artifact；sidecar 是完整性审计证据，不成为新的认知主体或自动注册入口。

## Gate

真实 native/CPU canary 必须证明：

1. 新 measured bundle 的 artifact/measurement 两个 digest roundtrip 通过，inventory 独立报告 `verified`；
2. measurement facts、measurement 文件名或 artifact 的 digest 关系被篡改时，load/inventory fail-closed，原始文件仍保留；
3. 旧 legacy artifact 仍可被既有 runtime contract 消费，但明确标记为 `legacy_unverified`，不被误报为 verified；
4. sidecar 审计、重复写入和 checkpoint restore 不改变 runtime、budget、topology、lineage 或任何外部字节。

## 验证入口

- 定向测试：`tests/taiji_native/test_structural_artifact_measurement_sidecar.py`
- CPU canary：`scripts/training/eval_taiji_structural_artifact_measurement_sidecar.py`
- 报告：`reports/taiji_w7_r5c_s49_structural_artifact_measurement_sidecar_20260831.json`

本 slice 仍不处理 CI、CUDA、前端、Windows shell、自动删除、自动修复、无限扩张、开放域收益或通用智能声明。
