# R5C-S44：多 artifact 外部 batch 交接与 parent 顺序 Gate

## 目标

验证两个 candidate 的 measured artifact 可以分别从外部 store 交接，严格按照 admission 后 parent checkpoint 变化重新绑定第二个 artifact，最终完成同一 batch；批量重复提交不得重复 admission 或预算扣减。

## 计划边界

- 创建真实双 candidate Workbench batch；第一个 artifact 从 store 重启后 admission，第二个 artifact 必须在第一个 admission 后重新测量并保存。
- 通过 SeedRuntime store bridge 在 checkpoint 间分阶段交接两个外部 artifact，再提交完整 artifact mapping 验证已完成 candidate 与待处理 candidate 的混合幂等语义。
- 核对每个 artifact 的 parent/trial digest、measurement/evidence provenance、batch completion、topology 和 budget；不得用旧 parent artifact 伪造并行 admission。
- 不改变 native 候选级 replay 失败、retention、rollback 或 store 所有权。

## Gate

- 第一 candidate 只 admission 一次；第二 candidate 只消费绑定第一步 child checkpoint 的新 artifact。
- 外部双 artifact 交接后 batch complete，重复全量提交只返回既有终态，预算精确扣减。
- checkpoint restore 后两项 artifact provenance、parent/trial digest 和 batch state 一致，旧 parent artifact 不会重新激活。

## 明确不声明

本 slice 只覆盖 native CPU SeedRuntime 多 artifact 外部 batch continuation，不声明并行无序 admission、无限结构扩张、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

## 证据交付

新增定向测试、独立 evaluator、JSON Gate 报告，并同步 active/reference/manifest；按用户决定不在本 slice 处理 CI，也不提交或推送。
