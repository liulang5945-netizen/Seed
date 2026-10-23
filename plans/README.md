# Seed / Taiji 计划与架构入口

当前执行是 **N 主线：优先提升态极（Taiji）原生自然中文多轮对话能力**。Seed 是产品与运行环境。当前只运行warm_start，版本化累计上限为1.55M CPU秒/臂、4.4M总额；当前结果和唯一下一步只看[03当前执行](active/roadmap/03_CURRENT_EXECUTION.md)。

## 项目位置

当前实验尚未到完整终点，也没有新的语言能力通过结论。N1-165A有限结果和R2早期自然回复失败均保留；N1-166、W5b与能力晋级当前不放行。大阶段继承资产和整体退出条件见01，不能由本次训练或局部测试替代。

执行顺序已收束为：**训练期间补齐extension评测接线 → 核验真实终点与来源/账本 → 冻结生成与人工判读 → 依据结果确定唯一后继方向 → 同bundle整模型验收**。训练触顶仍未完成时按真实partial结案，不能改称完整终点。

2026-10-23的[项目改进与唯一持续目标建议](reference/delivery-runs/20261021-n-mainline/artifacts/16-project-review-and-continuous-objective.md)把方向收束为同一模型的可靠任务能力与持续积累，按既有L2→L3→L4推进；目标表述为建议，不改已冻结评价门或自动启动新方向。

## 文档职责与证据入口

| 要查什么 | 唯一主要入口 |
|---|---|
| 当前状态、预算、运行与下一步 | [03当前执行](active/roadmap/03_CURRENT_EXECUTION.md) |
| 三臂合同、恢复与技术证据 | [续训活动摘要](reference/delivery-runs/20261021-n-mainline/artifacts/02-continuation-extension-plan.md) |
| 交付阶段和待办 | [专家团黑板](reference/delivery-runs/20261021-n-mainline/board.md) |
| 总能力阶段及缺口 | [01 M0–M8阶段地图](active/roadmap/01_SCOPE_AND_PHASES.md) |
| 晋级、CI和发布规则 | [02门禁](active/roadmap/02_GATES_AND_CI.md) |
| 整模型与用户验收 | [07交付验收](active/roadmap/07_MINI_MODEL_DELIVERY.md) |
| N系列设计与高上限后继方向 | [09主线规划](active/roadmap/09_NEXT_MAINLINE_PLAN.md) |
| 长期目标与原生架构 | [根需求](active/TAIJI_CORE_REQUIREMENTS.md)、[架构](active/TAIJI_NATIVE_ARCHITECTURE_V1.md)、[VISION](reference/VISION_FUTURE_TECHNOLOGY.md) |
| 历史实验正负结果 | [结果总览](reference/PROJECT_RESULTS_AND_CONCLUSIONS_20260917.md)与具体冻结结案报告；旧总览不得覆盖新结果 |
| 本次收束说明与历史索引 | [计划收束记录](reference/delivery-runs/20261021-n-mainline/artifacts/12-plan-consolidation-20261022.md) |

## 推进原则

- 当前队首只由03维护；README做导航，黑板记阶段，活动合同记执行细节。旧“当前状态/下一步/额度”移入历史，不并列发号施令。
- 保持原生Taiji能力方向，以真实输入到输出、恢复和同bundle验收证明模型能力；工程绿灯不冒充语言通过。
- 已授权且未触发停止线的工作连续推进；训练前检查checkpoint保存/恢复。预算、数据权限和冻结评价边界按对应合同执行。
- 依据完整结果选择能解除根因、上限更高的后继方案；不为尽快通过小测试而降低终点、阈值或替换目标。
- 保留失败、负结果与不可判结果；运行状态以真实句柄及耐久证据为准。正常训练期间文档整理不改变训练源码。

## 历史入口

整理前本页的逐轮快照完整保存在[历史索引快照](archive/history/PLANS_INDEX_BEFORE_N_CONSOLIDATION_20261022.md)。[旧执行快照](archive/history/N_MAINLINE_EXECUTION_BEFORE_CONSOLIDATION_20261022.md)保留N1和R2全过程；冻结合同、架构报告、原始报告与checkpoints继续保留在原路径。
