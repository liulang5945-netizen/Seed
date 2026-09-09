# M5 K 轴 scorecard 合同：K1/K2 证据收束与 A8 晋级边界

> 注册日期：2026-09-09。输入只允许使用已经完成的
> `M5_K1` 与 `M5_K2` formal reports；本合同是审计/收束，不是新的训练实验。

## 1. 目的

把 K1 单步组合与 K2 多步 autoregressive 组合放进同一个可寻址 scorecard，同时严格区分：

1. 能力绝对值是否存在；
2. A 相对 frozen/lesion 对照是否有因果分离；
3. 是否有同一 parent checkpoint、同一默认 runtime、连续学习后的 parent retention。

前两项可以关闭 K 轴证据线，第三项缺失时必须拒绝“结构已经可以进入默认 Taiji 路径”这一结论。

## 2. 固定输入

- `reports/taiji_m5_k1_skill_composition_formal_20260909.json`；
- `reports/taiji_m5_k2_multistep_formal_20260909.json`；
- `taiji.continual_evaluation.ContinualScorecard` 的 content-addressed absolute capability snapshots。

脚本不得重训、重算 cell、修改 formal 阈值或从历史 canary 替代 formal report。源报告 digest 写入 scorecard，源文件改变时审计必须失效。

## 3. Scorecard 字段与 Gate

### 3.1 能力证据

- K1 `single_step_composition_success`：formal A unseen success 的 mean/min/max；
- K2 `multistep_composition_success`：formal A holdout episode success 的 mean/min/max；
- K1/K2 各自的 `A-B`、`A-C` min/mean/max；
- 每一阶段的 robust、technical cells、read admission、reward variance。

### 3.2 否决边界

以下任一项成立，最终 `can_promote=false`：

- 任一 formal report 不是 `status=passed` 且 `robust=true`；
- scorecard 的 absolute capability snapshot 缺少或 digest 校验失败；
- K1/K2 parent retention 为 `null`（当前事实）；
- learner 是 standalone shadow，未接入默认 Taiji runtime；
- 没有同一 parent 上的 S/G/K 连续课程、资源等价、rollback 和旧能力非劣证据。

因此本次预期结论是：`k_evidence_closed=true`，`promotion_gate=false`，`can_promote=false`。这不是 K1/K2 失败，而是防止把组合课程证据误称为连续结构成长。

## 4. 下一出口

scorecard 通过只允许两个方向：

- 若要继续 K 课程，另行预注册 K3（outcome→world/任务间依赖）；
- 若要进入默认路径，建立同一 parent、同一 runtime 的 A8/R6 课程，把 K1/K2 作为 shadow evidence 输入，并补齐 parent retention、资源、rollback 与旧能力 Gate。

在上述出口重新预注册前，不解冻 K2 shadow，不引入 MCP/provider/client/CUDA 变量。
## 5. 执行记录（2026-09-09）

脚本 `scripts/training/audit_taiji_m5_k_scorecard.py` 已读取 K1/K2 formal 报告，验证两条证据线均 `status=passed`、`robust=true`，并用 `ContinualScorecard` 生成 content-addressed absolute capability snapshots。报告：`reports/taiji_m5_k_axis_scorecard_20260909.json`。

- `k_evidence_closed=true`：K1 单步组合与 K2 三步组合证据均闭合；A-B/A-C 对照信息保留在 scorecard 外层 comparison evidence。
- `parent_retention_missing=true`、`standalone_shadow=true`：两个 formal 都明确没有 parent baseline，且 learner 是 `learn=False` 的 standalone shadow。
- `promotion_gate=false`、`can_promote=false`：默认 runtime owner、同一 parent 的连续 S/G/K、资源/rollback/旧能力 Gate 均未被伪造为通过。

**结论**：K1/K2 作为 A8 K 轴的组合能力证据已经收束，但还不是可进入默认 Taiji 路径的成长结构。下一步必须在新的预注册中选择 K3 证据或同一 parent 的 A8/R6 晋级课程；当前继续保持 shadow 隔离。

**历史边界更新（2026-09-09）**：K3 formal 已完成，当前 K 轴统一入口改由 [M5_K_AXIS_SCORECARD_V2_CONTRACT_20260909.md](M5_K_AXIS_SCORECARD_V2_CONTRACT_20260909.md) 管理。本文与 v1 报告保持不可覆盖的 K1/K2-only 历史记录。
