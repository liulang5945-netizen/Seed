# R5C-S45：runtime retention 与外部 artifact store 生命周期分离 Gate

## 目标

明确并验证两类生命周期的所有权边界：runtime retention 只压缩 batch/candidate/artifact 的 lineage 引用，外部 immutable artifact store 不因 runtime maintenance 被物理删除；但 store 中保留的旧 artifact 也不能绕过已删除 batch 重新进入 runtime。

## 计划边界

- 创建活动 batch 与终结 batch，外部 store 保存终结 batch 的完整 measured artifact 文件。
- 运行小上限 terminal-only retention，确认 runtime 终结 lineage 被移除而 store 文件字节保持不变。
- 从 store 重新读取旧 artifact，尝试通过 SeedRuntime bridge 回放；已删除 batch 必须 fail-closed 且 runtime checkpoint digest 不变。
- 不实现自动垃圾回收；store 文件的后续清理必须以独立、显式、可审计的策略处理。

## Gate

- runtime retention 不直接删除外部 artifact 文件，artifact digest/measurement facts 可继续审计。
- 已删除 batch 的旧 artifact 即使仍在 store 中，也不能复活 candidate、topology、budget 或 provenance。
- 活动 batch、store artifact 和 retention audit 的所有权边界在 checkpoint restore 后一致。

## 明确不声明

本 slice 只覆盖 native CPU runtime retention 与外部 immutable store 的生命周期隔离，不声明自动垃圾回收、无限存储、无限结构扩张、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

## 证据交付

新增定向测试、独立 evaluator、JSON Gate 报告，并同步 active/reference/manifest；按用户决定不在本 slice 处理 CI，也不提交或推送。
