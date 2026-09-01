# R5C-S11：真实多区域 Workbench batch 的完整生命周期

> 状态：已完成（2026-08-30）
>
> 本 slice 把 S10 由真实 Workbench Outcome 生成的多区域 batch 接入既有 shadow、policy、atomic admission、checkpoint continuation 与 rollback；不新增第二套结构生命周期，也不并行提交 topology。

## 1. 目标

S10 已证明真实 Workbench 可以在两个区域产生两个不同 candidate 并进入同一 arbitration batch。S11 要求这些 candidate 真正走完“候选→验证→准入→回滚”的生命周期，同时证明局部失败不会污染其他区域。

## 2. 实现合同

| 对象 | 合同 |
|---|---|
| candidate continuation | 每个 selected candidate 独立使用 holdout、retention、lesion、resource 输入，仍由既有 validation policy 决定是否 admission |
| 失败隔离 | malformed holdout、policy rejection 或 admission failure 只能改变对应 candidate 状态并释放对应 reservation，不能回滚或污染其他 candidate |
| checkpoint | 首个 candidate admission 后必须能保存 batch reservation 与 topology，恢复后继续另一个 candidate |
| rollback | 每个 admitted candidate 绑定自己的 parent/child checkpoint；回滚只恢复该 candidate 对应的结构变化并重开其预算 |
| ownership | Workbench 只产生事实证据，provider/frontend 不拥有 topology、budget、policy 或 rollback 控制权 |

## 3. Gate

canary：`scripts/training/eval_taiji_workbench_multi_region_lifecycle.py`。

结果：`gate.passed=true`。

覆盖：

- 6 次真实 `workspace.read` 形成 S10 多区域 batch；
- 第一候选先 admission，剩余候选 reservation 跨 checkpoint 保持；
- 独立恢复分支让第二候选使用空 holdout，验证 fail-closed 且第一候选保持 admitted；
- 另一独立恢复分支使用合法 holdout 完成第二候选 admission；
- 第二候选 rollback 恢复其真实 region topology、重开一个 structural budget；
- rollback checkpoint restore 后重复 rollback 返回相同内容寻址结果。

## 4. 未关闭边界

- validation metrics 仍由 continuation 调用方显式提供，真实 Workbench replay artifact 由 S12 负责；
- 不证明并行 topology commit、自动预算扩张、开放域质量收益或全面自进化；
- CUDA 和 CI 全量验收按用户决定继续暂缓。

## 5. 下一步

R5C-S12：建立真实 Workbench replay 驱动的 `WorkbenchStructuralValidationArtifact`，让 holdout、retention、lesion、resource 事实具备统一内容寻址与 checkpoint lineage，再交给现有 policy。
