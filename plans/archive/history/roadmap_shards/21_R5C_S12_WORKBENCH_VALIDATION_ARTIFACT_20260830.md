# R5C-S12：真实 Workbench replay 驱动的 validation artifact

> 状态：已完成（2026-08-30）
>
> 本 slice 把单候选的验证事实从 continuation 调用方显式传参，收敛为由真实 Workbench replay 生成、内容寻址并绑定 checkpoint lineage 的不可变 artifact；artifact 只提供事实，既有 validation policy 仍拥有准入决策。

## 1. 目标

S11 已经证明真实多区域 batch 可以逐候选完成 shadow→policy→atomic admission→rollback，但 holdout、retention、lesion、resource 指标仍由 canary 直接注入。S12 消除这条证据边界，要求验证事实能被重放、校验、审计和恢复，而不是因为调用方传入了一个高分就通过。

## 2. 实现合同

| 对象 | 合同 |
|---|---|
| validation artifact | `WorkbenchStructuralValidationArtifact` 对 candidate、network/region/task slice、Outcome/evidence digest、holdout 输入/输出、retention/lesion 对照、resource measurement 与 parent/trial checkpoint digest 做内容寻址绑定 |
| replay verification | 消费 artifact 前必须验证当前 candidate、区域、资源成本、证据集合、parent checkpoint、holdout replay 和候选 trial checkpoint 全部匹配 |
| policy ownership | artifact 只提供不可变事实；`continue_structural_candidate()` 仍调用既有 shadow→policy→atomic admission，不把 artifact 分数直接视为准入 |
| failure isolation | replay 不匹配、篡改或 checkpoint 错配必须 fail-closed，且不能新增 admission、改变 topology 或扣减预算 |
| recovery | artifact 与 admission/rollback lineage 一并进入 checkpoint；恢复后重复消费必须幂等，不能重复提交 topology |
| public boundary | `SeedRuntime` 提供同一 artifact continuation 入口；provider/frontend 不获得结构、预算、policy 或 rollback 所有权 |

## 3. Gate

canary：`scripts/training/eval_taiji_workbench_validation_artifact.py`。

结果：`gate.passed=true`。

覆盖：

- 6 次真实 `workspace.read` 成功 Outcome，形成两个区域各自的 train/holdout evidence stream；
- artifact digest 绑定真实 Outcome/evidence、parent checkpoint、候选 trial checkpoint 与独立 holdout replay；
- altered holdout output 在独立 clone 中返回 `failed_closed`，不产生新增 admission 或 topology 变化；
- 合法 artifact 经过既有 validation policy 完成 admission；
- artifact、admission 与 checkpoint restore 后保持可读，重复消费返回 `already_applied`；
- payload 修改 `holdout_gain` 后由内容寻址校验拒绝。

## 4. 未关闭边界

- S12 只完成单候选 artifact 消费；多区域 batch 中每个 candidate 的 artifact 集合与 batch digest 绑定由 S13 负责；
- artifact 中的 metrics 仍需由真正的 replay/measurement owner 产生，本 slice 不声明开放域质量收益或全面自进化；
- 不扩展 CUDA、CI、无限预算或 provider 自治；CI 全量验收按用户决定继续暂缓。

## 5. 下一步

R5C-S13：把 replay-bound validation artifact 扩展到多区域 batch continuation，移除批次级手工验证指标注入并保持候选间失败隔离。
