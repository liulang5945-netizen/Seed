# R5C-S34：多批次 artifact lineage 隔离与 retention pressure

## 目标

验证多个 Workbench candidate batch 并存时，retention 只按完整 lineage 子图工作：无 live lineage 的终结 batch 可以被淘汰，活动 batch、pending/reserved candidate 和 rollbackable admission 必须保持可继续、可恢复，且任何 batch 的 artifact 不得污染另一 batch。

## 计划边界

- 构造至少两个独立 batch：一个保持活动 reservation/pending candidate，另一个完成 measured artifact admission 后回滚为终结 lineage。
- 使用小 retention limit 触发 pressure，检查保护集合、artifact/artifact-batch digest 与 batch 归属；只允许完整终结子图被淘汰。
- 对终结 batch 的 artifact、replay、candidate 和 batch 入口做 fail-closed 验证；对活动 batch 完成一次 measured continuation，证明压缩没有跨 batch 修改 topology、budget、cursor 或 reservation。
- 在压缩前后保存/加载 native checkpoint，比较 retention pressure、保护集合、artifact digest、candidate states、topology 与 structural budget。
- 将跨 batch mapping key、错误 artifact batch、stale parent 和重复 maintenance 纳入 red evidence，任何失败必须保持原子。

## Gate

- 活动 lineage 在小上限下保持完整，且 protected pressure 可观察。
- 终结 batch 的 candidate/artifact/artifact-batch/validation/gate/admission/rollback/schedule audit 同步淘汰，不出现孤儿记录。
- 终结 batch 的旧 replay fail-closed；活动 batch 仍可准入/回滚。
- checkpoint restore 后上述结论不漂移，且重复 retention 不产生新的结构变化。

## 明确不声明

本 slice 只覆盖 native CPU 多 batch artifact retention 与隔离，不声明无限扩张、自动增预算、开放域收益、CUDA、前端、Windows shell、CI 或通用智能。

## 证据交付

新增定向测试、独立 evaluator、JSON Gate 报告，并同步 `03_CURRENT_EXECUTION.md`、`04_EXECUTION_PLAN.md`、`IMPLEMENTATION_STATUS_2026_08.md` 和 manifest。
