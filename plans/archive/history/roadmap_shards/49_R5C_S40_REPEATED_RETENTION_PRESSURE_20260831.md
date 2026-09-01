# R5C-S40：重复 retention pressure 循环与有界增长 Gate

## 目标

把 S38–S39 的一次性三轮验证推进到多次 retention pressure 循环：在一个活动 reservation 持续存在时，连续创建、准入、回滚并淘汰多个终结 batch，确认 Taiji 的结构 lineage 会随生命周期受控增长，而不是靠扩大内存或无限保留历史记录。

## 计划边界

- 使用至少五轮新的 Workbench task slices；第一轮保留活动 batch，后续至少四轮依次完成 measured artifact admission、rollback 和显式 terminal-only retention。
- 每轮在 schedule、measured parent、admission/rollback 或 maintenance 后至少完成一次 SeedRuntime save/load，检查 cursor、topology、budget、policy 和 batch 状态。
- 使用小 retention 上限（`max_batches=1`）反复淘汰当前终结 batch；活动第一轮 reservation 必须持续受保护。
- 记录 batch/artifact/candidate/maintenance 数量、已删除 batch/artifact replay、runtime tick、结构预算和 topology，确认 lineage ledger 与资源计数保持有界。

## Gate

- 至少四次 retention pressure 循环都只淘汰当轮终结 batch，不删除活动 reservation，也不改变其 topology/budget。
- 每轮 admission 的预算扣减和 rollback 恢复精确；重启不重复 admission、不复活已删除 lineage。
- 最终 checkpoint 只保留受保护活动 batch，任一已删除终结 batch 的 artifact replay fail-closed 且无状态变化。
- 多轮 cursor 单调、policy digest 稳定、结构记录数量受 retention/lineage 上限约束，不以规模膨胀冒充能力提升。

## 明确不声明

本 slice 只覆盖 native CPU SeedRuntime 重复 retention pressure 与有界 structural lineage，不声明无限扩张、自动增加预算、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

## 证据交付

新增定向测试、独立 evaluator、JSON Gate 报告，并同步 active/reference/manifest；按用户决定不在本 slice 处理 CI，也不提交或推送。
