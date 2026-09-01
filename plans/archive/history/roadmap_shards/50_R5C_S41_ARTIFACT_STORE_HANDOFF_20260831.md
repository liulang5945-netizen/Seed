# R5C-S41：外部 measured artifact store 与跨进程交接 Gate

## 目标

把 measured artifact 从调用栈里的 Python 对象提升为可独立交接的不可变 artifact：artifact 以自身 digest 命名，能够写入磁盘、从另一运行实例读取并再次通过 Taiji contract 校验，损坏或碰撞不能被静默接受。

## 计划边界

- 为 `WorkbenchStructuralValidationArtifact` 增加独立的 content-addressed JSON store；文件名只由 artifact digest 决定，不使用 candidate/path 等可变字段作为存储身份。
- 写入采用同目录原子替换；同一 digest 的重复写入必须幂等，字节冲突必须拒绝；读取必须重新解析 artifact 并验证内外 digest。
- 验证 store 产生的 payload 可脱离原 runtime 交接，再由 SeedRuntime artifact batch API 消费；并发重复写不能产生半文件或不同内容。
- store 不负责物理删除、不绕过 batch/parent/replay/admission/rollback/retention contract；retention 只管理 lineage 引用，artifact store 的清理另行设计。

## Gate

- artifact JSON roundtrip 后 digest、measurement digest、evidence 与所有 validation facts 不变。
- 同一 artifact 重复/并发写入幂等，恶意篡改、非法 digest 文件和内容碰撞 fail-closed。
- 读取的外部 payload 可经过 SeedRuntime 重启后的现有 batch contract，不能跳过当前 parent、candidate、batch 或 replay 校验。
- 任何失败不修改已存在的 artifact，也不改变 runtime topology、budget、candidate 或 batch 状态。

## 明确不声明

本 slice 只覆盖 native CPU measured artifact 的外部持久化与交接，不声明 artifact 无限存储、自动垃圾回收、开放域收益、无限结构扩张、CUDA、前端、Windows shell、CI 或通用智能。

## 证据交付

新增 store 实现、定向测试、独立 evaluator、JSON Gate 报告，并同步 active/reference/manifest；按用户决定不在本 slice 处理 CI，也不提交或推送。
