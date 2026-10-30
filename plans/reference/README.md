# 研究合同、裁决与历史证据索引

本页按主题导航，不再复制逐轮训练数字、审批和“下一步”。当前队首见[03](../active/roadmap/03_CURRENT_EXECUTION.md)，详细工作包见[10](../active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)。原滚动索引全文可从`d12a8679a:plans/reference/README.md`恢复。

## 当前主实验的合同分工

| 文档 | 文档状态 | 维护职责 |
|---|---|---|
| [20261030核心设计重审](TAIJI_CORE_DESIGN_REVIEW_20261030.md) | 设计评审建议 | 核心计算图、旧正证、F/N178混杂、保留/重构/退出与新合同前提 |
| [20261029阶段收束](PROJECT_CONSOLIDATION_20261029.md) | 当前收束记录 | E/F0a资产、E3子门、失败、CI/契约待办及用户执行边界 |
| [E1消费合同](SPEC-N1-177_E1_state_consumption_contract_20261029.md) | 冻结原规则 | 状态依赖任务与内容消费；进度以03为准 |
| [F值编码器合同](SPEC-N1-177_F_value_encoder_20261029.md) | 方向/四臂及预检规则冻结 | F0a/F0b分账；F1数字/预算及曝光冲突仍须定稿 |
| [F1冻结提案](SPEC-N1-177_F1_freeze_proposal_20261029.md) | 待owner签署（未启动） | 候选范围、12-run预算(≈121,500 CPU秒)、数值门与停止线；签后写回F合同 |
| [N178解析覆盖扩展提案](SPEC-N1-178_parse_coverage_extension_proposal_20261030.md) | 待owner签署（未启动） | F1/F1b 负结果的共同瓶颈；四格覆盖率基线；先测价再定预算 |
| [E状态消费/信用决策](PLAN-N1-177_E_state_consumption_and_credit_decision_20261029.md) | 当前设计决策 | A–D证据复核/勘误、选项取舍及E0–E3出口 |
| [N177实施计划](PLAN-N1-177_joint_cognition_slice_20261026.md) | 活动交接 | 已有资产、A–F实施边界及后继接口 |
| [177.5 T1 SPEC](SPEC-N1-177_5_T1_F0_F1_pilot_20261027.md) | 原合同冻结 | 五模式×三seed、数据/更新权限、阈值和预算 |
| [03执行看板](../active/roadmap/03_CURRENT_EXECUTION.md) | 随耐久进度维护 | 当前运行、臂矩阵及累计资源 |
| [10工作包](../active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md) | 随阶段交接维护 | 训练＋并行准备、评价、诊断、后继候选的依赖与验收 |

PLAN的排期表可以随真实推进修订，SPEC与已结案证据保持原规则；同一件事不在多份表里复制状态和预算。当前模式/臂进度只登记03。

## 当前实施与交接

| 主题 | 入口 |
|---|---|
| 持续Goal与旧计划过渡 | [17过渡矩阵](delivery-runs/20261021-n-mainline/artifacts/17-plan-transition-and-reconciliation.md)、[当前交付黑板](delivery-runs/20261021-n-mainline/board.md) |
| 历史R2工程交接 | [02工程计划](delivery-runs/20261021-n-mainline/artifacts/02-plan.md)、[R2技术摘要](delivery-runs/20261021-n-mainline/artifacts/02-continuation-extension-plan.md) |
| extension评价适配 | [独立provider/worker架构](delivery-runs/20261021-n-mainline/artifacts/04-continuation-evaluation-adapter.md) |
| 单入口续训与来源 | [15精简](delivery-runs/20261021-n-mainline/artifacts/15-training-system-simplification.md)、[原extension架构](delivery-runs/20261021-n-mainline/artifacts/04-continuation-extension-architecture.md) |
| 原全轮/R2配对合同 | [全轮预训练](PLAN-R2-DAILY-DIALOGUE-FULL-TURN-PRETRAINING_20261021.md)、[恢复合同](PLAN-R2-DAILY-DIALOGUE-RECOVERY_20261021.md)、[warm/fresh合同](PLAN-R2-DAILY-DIALOGUE-WARMSTART-MATCHED_20261021.md) |
| 下一主线统一架构、驱动力与VX0–VX7工程包 | [VISION现行方案](VISION_FUTURE_TECHNOLOGY.md)；旧全文由其固定Git身份恢复 |
| 当前共同认知纵切片实施 | [N177接口与顺序合同](PLAN-N1-177_joint_cognition_slice_20261026.md)；唯一动作由03维护 |
| 七类架构缺陷与改造建议核实 | [主张与来源审计](PLAN-N1-176_architecture_claims_audit_20261026.md) |
| N176已训课程资产与收束 | [N176实施计划](PLAN-N1-176_foundation_curriculum_20261026.md)、[固定F0/F1合同](SPEC-N1-176_F0_F1_pilot_20261026.md)、[材料审计](../../reports/taiji_n1_plan175_curriculum_data_audit_20261026.json) |
| N175当前评价与匿名判读 | [评价交接](PLAN-N1-175_evaluation_handoff_20261026.md)、[匿名判读表](PLAN-N1-175_blind_review_20261026.md) |
| 前序语言负结果的原因与审计 | [N175原因复盘](PLAN-N1-175_prior_results_cause_review_20261026.md)、[初始化/历史读数审计](../../reports/taiji_n1_plan175_prior_results_cause_audit_20261026.json) |
| 当前N的必要交汇与原生目标 | [09机制/交汇映射](../active/roadmap/09_NEXT_MAINLINE_PLAN.md)、[根需求](../active/TAIJI_CORE_REQUIREMENTS.md) |
| 生命页发展运行台与VX6.0–6.4 | [实施设计](UI_LIFE_UX_SPEC_20261003.md)、[视觉规则](UI_LIFE_VISUAL_DIRECTION_20261003.md) |

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
