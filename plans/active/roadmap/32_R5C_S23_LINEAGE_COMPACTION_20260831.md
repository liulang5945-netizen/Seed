# R5C-S23：跨账本协同终结保留

## 目标

把 S22 的“受保护记录不能盲删”推进为一个显式的 lineage graph retention 操作：以 candidate batch 为根，协调 candidate、artifact、artifact batch、validation、gate、admission、rollback、proposal 与相关 schedule audit 的终结生命周期。

## 实现

- `taiji/structural_lineage.py` 新增 content-addressed `StructuralLineageRetentionResult`，记录 source/target checkpoint、保护/保留/删除 batch 与 candidate、各 ledger 删除计数和 retention pressure。
- `TSKV8Adapter.compact_structural_lineage_history()` 只选择无 active reservation、reserved/deferred candidate、pending topology proposal 和 rollback 依赖的终结 batch；按插入顺序选择删除目标。
- 删除以一个 adapter 状态事务完成：batch、candidate/proposal、validation、artifact、artifact batch、gate、admission、rollback、maintenance 与 schedule audit 同步处理；checkpoint digest 失败时恢复所有受影响容器。
- 被删除候选的旧 replay 无法重新 materialize，旧 batch rollback 解析失败；保护 batch 即使超过目标上限也不被静默清除，结果显式返回 `retention_pressure`。

## Gate

报告：`reports/taiji_w7_r5c_s23_structural_lineage_compaction_20260831.json`

真实 Workbench canary 必须全部满足：

- 终结 batch 及六类关联验证/准入/回滚记录和 artifact batch 协同移除；
- active reservation 与 candidate lineage 保留；
- checkpoint restore digest 确定；
- 旧 replay/rollback fail-closed 且不复活旧链；
- retention result 可 payload roundtrip 且 content-addressed；
- 压缩后新 evidence 能在原 checkpoint 分支上确定性地产生同一 schedule；
- protected lineage 超限返回 pressure 且不删除保护 batch。

结果：`gate.passed=true`；S23 新增定向用例 `4 passed`，R5C 相关回归 `31 passed`，Ruff/py_compile 通过。

## 边界

本 slice 只实现 native/CPU lineage graph 的逻辑压缩与 checkpoint 审计，不执行物理文件删除，不扩展 structural budget，不触及 CI、CUDA、前端、provider 或 Workbench 副作用。

## 后继

唯一后继为 R5C-S24：把显式协同压缩接入结构运行时维护周期，建立自动触发、幂等、失败原子和上层 pressure 观测 Gate。
