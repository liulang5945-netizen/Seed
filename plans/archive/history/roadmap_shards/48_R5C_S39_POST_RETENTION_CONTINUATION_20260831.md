# R5C-S39：retention 后活动 lineage 继续执行 Gate

## 目标

验证 S38 压缩后的活动 batch 不是静态快照：SeedRuntime 从 retention 后的磁盘 checkpoint 恢复时，仍能对受保护 reservation 生成新的 measured artifact，继续完成 candidate admission，再执行可逆 rollback 和第二次 checkpoint。

## 计划边界

- 重建三轮真实 Workbench evidence：保留至少一个活动 batch，另一个 batch 在 retention 前成为可淘汰终结子图，第三轮保持活动 reservation。
- retention 后从磁盘恢复，使用当前恢复状态重新测量受保护 batch 的候选；不能复用 retention 前的 parent digest，也不能把旧 artifact 重新激活。
- 顺序完成两个 candidate 的 measured admission，核对 reservation、topology、budget、artifact provenance 和 batch 状态；随后 rollback 两个 candidate，确认恢复后的结构与预算可逆。
- 至少两次 post-retention save/load，第二次显式 maintenance 只能保留活动 lineage，不得复活已删除 batch/artifact，也不得产生隐式重放。

## Gate

- retention 后受保护 batch 可继续消费新的 measured artifact，且 parent/trial digest 绑定当前 checkpoint。
- 两个 admission 只发生一次，预算精确扣减；重复提交或 rollback 幂等，rollback 后 topology/budget 恢复。
- 已淘汰终结 batch 的旧 artifact 仍 fail-closed；活动 sibling、reservation、cursor、policy 与 retention audit 不受污染。
- post-retention checkpoint 恢复后所有关键 projection 一致，资源计数始终受 lineage policy 和 structural budget 约束。

## 明确不声明

本 slice 只覆盖 native CPU SeedRuntime retention 后的活动 lineage continuation，不声明无限扩张、自动增加预算、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

## 证据交付

新增定向测试、独立 evaluator、JSON Gate 报告，并同步 active/reference/manifest；按用户决定不在本 slice 处理 CI，也不提交或推送。
