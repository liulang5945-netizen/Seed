# M5 K 轴 scorecard v2：K1/K2/K3 shadow 证据收束与晋级边界

> 注册日期：2026-09-09。v1 scorecard 已关闭 K1/K2 standalone composition evidence；本 v2 只增加已经完成 formal 的 K3 outcome→world/dependency evidence，不重训、不重算任何 cell，不把 shadow 结构接入默认 runtime。

## 1. 目的

建立唯一的 K 轴审计入口，分别保存三种能力证据：

1. K1：结构化语义到真实只读动作的单步组合；
2. K2：预测 world 驱动的三步 autoregressive 组合；
3. K3：真实 success/failure outcome 进入 observed world/dependency，并通过 lineage gate 约束下一步任务。

scorecard 必须把 absolute capability、A-B/A-C 因果对照、来源报告 digest 和 promotion veto 分开记录。K3 的加入不等价于“Taiji 已经自进化”，因为三条证据仍然是 standalone shadow learner。

## 2. 固定输入与内容寻址

只允许读取以下已完成 formal reports：

- `reports/taiji_m5_k1_skill_composition_formal_20260909.json`；
- `reports/taiji_m5_k2_multistep_formal_20260909.json`；
- `reports/taiji_m5_k3_outcome_dependency_formal_20260909.json`。

脚本必须把三个 source report 的 content digest 写入输出，并使用 `ContinualScorecard` 生成 K1/K2/K3 各自的 absolute capability snapshot。任何 source report 改变后，scorecard digest 必须改变；不能使用历史 canary、手工数值或重新执行训练替代 formal report。

## 3. evidence 字段

### 3.1 absolute capability

- K1：A unseen success 的 min/mean/max；
- K2：A holdout multistep success 的 min/mean/max；
- K3：A holdout outcome-dependent success 的 min/mean/max；
- 每阶段的 robust、technical cells、admission、reward variance 分别保存。

### 3.2 comparison evidence

分别保存 K1/K2/K3 的 A-B、A-C min/mean/max。K3 额外保存 probe admission、feedback lineage admission 和 feedback reward variance，不能将 lineage 与普通执行成功合并成一个分数。

### 3.3 promotion veto

以下任一项成立，v2 的 `promotion_gate=false`、`can_promote=false`：

- 任一 formal report 不是 `status=passed` 且 `robust=true`；
- 任一 scorecard snapshot 缺失、content digest 不可验证或来源 digest 失配；
- 任一阶段的 parent retention 为 `null`；
- learner 仍是 standalone shadow，未附着同一默认 Taiji runtime owner；
- 没有同一 parent 的连续 S/G/K 课程、资源等价、rollback 和旧能力非劣证据。

预期结论是：`k_evidence_closed=true`，但 `promotion_gate=false`、`can_promote=false`。这是证据边界，不是 K1/K2/K3 formal 失败。

## 4. 不做的事

- 不重训、重跑或修改 K1/K2/K3 formal 判据；
- 不把 projection/transition/planner shadow 接入默认 runtime；
- 不引入 provider、MCP、联网、真实客户端写入、CUDA 或视觉变量；
- 不把三条 evidence 的 absolute success 平均成一个“智能分数”；
- 不用 K1/K2/K3 formal 证据伪造 parent retention 或连续生长。

## 5. 产物与唯一后续动作

- auditor：`scripts/training/audit_taiji_m5_k_axis_scorecard_v2.py`；
- report：`reports/taiji_m5_k_axis_scorecard_v2_20260909.json`；
- 完成后同步 `plans/active/roadmap/03_CURRENT_EXECUTION.md`，保留 v1 历史报告不覆盖；
- scorecard v2 之后，唯一下一步是另行预注册同一 parent 的 A8/R6 promotion course；在该预注册前不解冻任何 K shadow owner。

## 6. 执行记录（2026-09-09）

审计脚本 `scripts/training/audit_taiji_m5_k_axis_scorecard_v2.py` 已读取 K1/K2/K3 formal reports，没有重训或重跑 cell；报告为 `reports/taiji_m5_k_axis_scorecard_v2_20260909.json`。

- 三个 source report digest 已写入 scorecard，`k_evidence_closed=true`；K1/K2/K3 的 absolute A success min/mean/max 均为 `1.0/1.0/1.0`；
- K1/K2/K3 的 A-B、A-C comparison evidence 均保留为独立字段；K3 的 probe admission 与 feedback lineage 均为 `9/9`，feedback reward variance 也独立保留；
- `parent_retention_missing=true`、`standalone_shadow=true`；`default_runtime_owner_attached=false`、`same_parent_continual_s_g_k_evidence=false`、`resource_rollback_old_capability_gate=false`；
- 因而 `promotion_gate=false`、`can_promote=false`。K 轴 evidence 已闭合，但没有任何 K shadow owner 被解冻或接入默认 Taiji runtime。

该 v2 scorecard supersede 旧的 K1/K2-only scorecard 作为当前 K 轴入口，但保留 v1 报告作为历史、不可覆盖的审计产物。
