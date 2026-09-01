# R5C-S47：SeedRuntime 外部 artifact store audit 只读投影 Gate

## 目标

把 S46 的外部 `StructuralValidationArtifactStore` inventory / audit 接入一个显式的 SeedRuntime 只读观察入口。该入口只投影外部 artifact 的完整性事实，并与当前 checkpoint 中可见的 runtime validation-artifact lineage 做对照；它不把外部文件注册回 Taiji，不触发 replay、retention、budget、candidate/batch 或 checkpoint 变化。

## 设计选择

- 入口放在 `SeedRuntime`，因为它是客户端/工作台能够调用的运行时边界；artifact store 仍由 Taiji store 类型负责文件完整性。
- 输出使用稳定的 `audit_digest`，由 store inventory 与 runtime lineage visibility 共同内容寻址；相同 checkpoint、相同 store 内容必须得到相同投影。
- visibility 只表达事实关系：`runtime_recorded`、`runtime_batch_referenced` 或 `external_orphan`。它不把“外部存在”推断为“可消费”，也不自动创建 batch 映射。
- store audit 异常直接 fail-closed；SeedRuntime 不吞掉错误、不修复文件、不追加 audit 记录。

## Gate

真实 native/CPU canary 必须证明：

1. 同一 runtime checkpoint 对同一 store 重复查询返回完全相同的 `audit_digest` 与 entries；
2. checkpoint save/load 后投影仍一致；runtime orphan 被标识为外部存在但 runtime 不可消费，活动/已知 lineage 不被错误推断；
3. 缺失、篡改或非法 store 文件通过 runtime 入口 fail-closed，且查询前后 runtime checkpoint、拓扑、budget、retention audit 和 store 文件字节不变；
4. 只读投影不注册任何外部 artifact，不改变 `structural_validation_artifacts` 或 artifact batch 记录。

## 验证入口

- 定向测试：`tests/taiji_native/test_runtime_artifact_store_audit_projection.py`
- CPU canary：`scripts/training/eval_taiji_runtime_artifact_store_audit_projection.py`
- 报告：`reports/taiji_w7_r5c_s47_runtime_artifact_store_audit_projection_20260831.json`

本 slice 仍不处理 CI、CUDA、前端、Windows shell、自动注册、自动删除、无限扩张、开放域收益或通用智能声明。
