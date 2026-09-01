# R5C-S46：外部 structural artifact store 只读 inventory / audit Gate

## 目标

在 S45 已明确 runtime lineage retention 与外部 immutable artifact store 生命周期分离后，为外部 store 增加只读、可重复的 inventory / audit 能力。审计只回答“有哪些文件、文件名 digest、artifact digest、measurement digest 是否一致，以及 canonical bytes 是否完整”，不决定 artifact 是否仍属于 runtime，不自动删除文件，也不改变任何 runtime 状态。

## 为什么现在做

S45 允许终结 batch 的外部 artifact 继续保留，同时要求这些 artifact 不能复活已被 runtime retention 淘汰的 batch。若没有独立 inventory，外部保留物只能靠目录枚举或直接读取，无法稳定区分完整 artifact、篡改文件、非法命名文件和 runtime 已不可消费的 orphan。S46 把“可审计”和“可消费”明确分开，避免把审计误当成垃圾回收或重新激活机制。

## 实现边界

- `StructuralValidationArtifactStore` 提供只读 `inventory()` / `audit()` 视图，按稳定的 digest 文件名顺序返回 artifact digest 与 measurement digest 事实摘要。
- 审计必须重新验证文件名 digest、payload 的 artifact digest、canonical JSON 字节和 measurement digest；任一文件异常都 fail-closed。
- 外部文件只读审计不触发 `SeedRuntime`、candidate、batch、budget、provenance、retention audit 或 checkpoint 变化。
- runtime retention 产生的 orphan 仍可出现在 inventory，但只能作为外部事实被观察；尝试把它交回已删除 batch 仍由 native batch contract 拒绝。
- 不实现自动垃圾回收、物理删除、tombstone、store 重写、跨机器索引服务或并发写入协议扩展。

## Gate

真实 native/CPU canary 必须证明：

1. healthy store 的 inventory 与逐项 `load()` 事实一致，并在 checkpoint restore 前后保持稳定；
2. runtime retention 后外部 orphan 仍可被审计，但不存在于 runtime 可消费 lineage，旧 batch replay fail-closed 且 checkpoint digest 不变；
3. 篡改 payload、measurement facts 或文件名 digest，以及额外非法文件，均被只读 audit 拒绝，不发生自动删除或修复；
4. inventory/audit 不改变 store 文件字节、runtime 拓扑、budget、candidate/batch 状态或 retention audit。

## 验证入口

- 定向测试：`tests/taiji_native/test_runtime_retention_store_audit.py`
- CPU canary：`scripts/training/eval_taiji_runtime_retention_store_audit.py`
- 报告：`reports/taiji_w7_r5c_s46_runtime_retention_store_audit_20260831.json`

本 slice 仍不处理 CI、CUDA、前端、Windows shell、自动删除、无限扩张、开放域收益或通用智能声明。
