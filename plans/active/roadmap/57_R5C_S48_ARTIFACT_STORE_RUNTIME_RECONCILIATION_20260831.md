# R5C-S48：artifact store 与 runtime lineage 只读对账 Gate

## 目标

在 S47 的外部 store audit 只读投影上增加反向引用对账：除了列出 store 中的 artifact 及其 runtime visibility，还明确列出 runtime 已记录或 artifact batch 已引用、但外部 store 当前缺失的 digest。这样“外部 orphan”和“runtime 缺失外部文件”不会被混为一谈。

## 设计边界

- 保留 S46 的 store integrity audit 和 S47 的 `runtime_recorded`、`runtime_batch_referenced`、`external_orphan` 事实语义。
- projection 增加稳定排序的 runtime artifact digest 集合、runtime batch-reference digest 集合，以及两者各自的 missing-store digest 集合。
- projection schema 升级为显式 v2；旧 v1 报告保留为历史证据，不在运行时隐式迁移或伪造。
- 对账只观察 store 与当前 checkpoint，不注册缺失文件、不从 runtime 回写 store、不删除 orphan、不运行 replay/retention，也不改变 budget、topology 或 checkpoint。

## Gate

真实 native/CPU canary 必须证明：

1. store 完整时，runtime 与 batch 引用集合的 missing-store 集合为空，重复查询及 checkpoint restore 的 projection digest 稳定；
2. 构造一个 runtime 已记录但未进入外部 store 的 artifact 时，只报告对应 missing digest，不影响已有 store entry 的 visibility；
3. 外部 orphan 与 runtime-missing 两类不互相误报；
4. 缺失/篡改 store 文件仍由 S46 integrity audit fail-closed，整个对账查询保持只读且不改变 runtime 或 store 字节。

## 验证入口

- 定向测试：`tests/taiji_native/test_runtime_artifact_store_runtime_reconciliation.py`
- CPU canary：`scripts/training/eval_taiji_runtime_artifact_store_runtime_reconciliation.py`
- 报告：`reports/taiji_w7_r5c_s48_artifact_store_runtime_reconciliation_20260831.json`

本 slice 仍不处理 CI、CUDA、前端、Windows shell、自动修复、自动删除、无限扩张、开放域收益或通用智能声明。
