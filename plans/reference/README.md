# 研究合同、裁决与历史证据索引

本页按主题导航，不再复制逐轮训练数字、审批和“下一步”。当前队首见[03](../active/roadmap/03_CURRENT_EXECUTION.md)，详细工作包见[10](../active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)。原滚动索引全文可从`d12a8679a:plans/reference/README.md`恢复。

## 当前实施与交接

| 主题 | 入口 |
|---|---|
| 持续Goal与旧计划过渡 | [17过渡矩阵](delivery-runs/20261021-n-mainline/artifacts/17-plan-transition-and-reconciliation.md)、[当前交付黑板](delivery-runs/20261021-n-mainline/board.md) |
| 当前R2工程交接 | [02工程计划](delivery-runs/20261021-n-mainline/artifacts/02-plan.md)、[R2技术摘要](delivery-runs/20261021-n-mainline/artifacts/02-continuation-extension-plan.md) |
| extension评价适配 | [独立provider/worker架构](delivery-runs/20261021-n-mainline/artifacts/04-continuation-evaluation-adapter.md) |
| 单入口续训与来源 | [15精简](delivery-runs/20261021-n-mainline/artifacts/15-training-system-simplification.md)、[原extension架构](delivery-runs/20261021-n-mainline/artifacts/04-continuation-extension-architecture.md) |
| 原全轮/R2配对合同 | [全轮预训练](PLAN-R2-DAILY-DIALOGUE-FULL-TURN-PRETRAINING_20261021.md)、[恢复合同](PLAN-R2-DAILY-DIALOGUE-RECOVERY_20261021.md)、[warm/fresh合同](PLAN-R2-DAILY-DIALOGUE-WARMSTART-MATCHED_20261021.md) |
| 设计储备与原生目标 | [VISION](VISION_FUTURE_TECHNOLOGY.md)、[09机制映射](../active/roadmap/09_NEXT_MAINLINE_PLAN.md)、[根需求](../active/TAIJI_CORE_REQUIREMENTS.md) |

## 已结案证据与批准范围

| 主题 | 权威记录 | 使用边界 |
|---|---|---|
| M阶段收束 | [M0–M7收束](M0_M7_PROJECT_CONSOLIDATION_20261006.md)、[M5批准](M5_EXIT_APPROVAL_20260920.md)、[M6批准](M6_SCOPE_EXCLUSION_APPROVAL_20261006.md) | 保留限定范围/排除，不补发新模型L2/L3/L4 |
| R2旧自然对话负结果 | [对话架构审计结案](PLAN-R2-DAILY-DIALOGUE-ARCHITECTURE-AUDIT_20261021_execution_closeout.md) | 旧0/12只属对应候选，不能代替新评测 |
| N1局部训练及泛化边界 | [N1-162B结案](PLAN-N1-162B_execution_closeout_20261020.md)及各PLAN/EXP原件 | 具体进展和166冻结看03；历史“已批准下一格”不调度 |
| 蒸馏与已删除过程材料 | [M5墓碑](M5_DISTILLATION_TOMBSTONE.md)、[20261007墓碑](DISTILLATION_TOMBSTONE_20261007.md) | 原结论、失败及精确Git恢复优先，不重复复制全文 |
| 机器结果与来源 | [reports导航](../../reports/README.md)、`plans/manifests/` | 报告日期/模型身份/判据分账，不混合跨版本成绩 |
| 本轮历史整理 | [整理与Git恢复](PROJECT_HISTORY_CONSOLIDATION_20261023.md) | 路径/字节/恢复提交可查；当前冻结证据保留 |
| 既有冷归档 | [旧整理记录](PROJECT_ORGANIZATION_20261014.md)、[完成回执](PROJECT_COLD_ARCHIVE_COMPLETION_20261014.json) | 原件已在10月14日删除，ZIP恢复包保留，不重复计算成果 |

## 如何查具体实验

文件按`PLAN-N1/N2/N3/N5`、`SPEC`、`EXP`、`execution_closeout`、`APPROVAL`等既有命名族查找；使用`rg --files plans/reference`定位，再读对应合同和结案。原件保持现有路径，避免破坏脚本、冻结SHA和外部引用。历史结构入口见[archive索引](../archive/README.md)，运行流水按08登记与Git定位。

未完成条款保留在当前计划/债表，失败与未消费题面不因整理改变状态。历史记录中的预算、审批或“下一步”只解释其时点，当前调度统一由03管理。
