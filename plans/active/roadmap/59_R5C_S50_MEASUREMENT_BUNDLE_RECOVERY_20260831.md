# R5C-S50：measurement bundle 部分写入后的显式恢复 Gate

## 目标

验证 S49 的 artifact 与 measurement sidecar 双文件 bundle 在进程中断或只完成一侧写入时的行为边界：不把不完整 bundle 当作 verified，不自动删除、不自动修复；同一调用方提供完整 artifact + measurements 后可以幂等重试，恢复为 verified bundle。

## 设计边界

- sidecar-only、非法 sidecar 或篡改 bytes 继续由 inventory/load fail-closed，并保留现场供人工或上层流程决定。
- artifact-only legacy 文件保持 `legacy_unverified`，不能伪造 measurement facts；补交匹配 measurements 后才升级为 `verified`。
- `put_measured_artifact()` 的重试只允许相同内容寻址字节，冲突仍拒绝；不执行自动垃圾回收、目录清理或 runtime 注册。
- recovery 前后 runtime bridge、checkpoint、budget、topology、lineage 和 store 中已有 immutable bytes 均不被隐式改变。

## Gate

真实 native/CPU canary 必须证明：

1. 只写 sidecar 时 audit fail-closed 但不删除，补交完整 artifact 后可恢复并显示 `verified`；
2. 只写 artifact 时显示 `legacy_unverified`，补交匹配 sidecar 后可升级为 `verified`；
3. 重试是幂等的，冲突 sidecar/artifact 不覆盖原 bytes；
4. 篡改或恢复过程不触发 runtime 消费、checkpoint 变化或自动清理。

## 验证入口

- 定向测试：`tests/taiji_native/test_structural_artifact_measurement_bundle_recovery.py`
- CPU canary：`scripts/training/eval_taiji_structural_artifact_measurement_bundle_recovery.py`
- 报告：`reports/taiji_w7_r5c_s50_structural_artifact_measurement_bundle_recovery_20260831.json`

本 slice 仍不处理 CI、CUDA、前端、Windows shell、自动修复、自动删除、无限扩张、开放域收益或通用智能声明。
