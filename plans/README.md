# Seed / Taiji 计划与架构入口

> 更新：2026-09-13；结果审查基线 `102b81e1`。执行顺序仅由[当前推进方案](active/roadmap/03_CURRENT_EXECUTION.md)决定。

## 当前阶段

**K 轴已限定范围晋级，P5.1 内容收益与 P5.2 合同/预测 Gate 已通过；当前重点转向预测实际驱动执行、群体因果语料与在线学习。**

三个边界必须同时保留：K/G 附着为 opt-in、进程内状态；P5.1g 的真实语料成绩来自未准入 trial learner；P5.2 执行仍为 scripted 路径，readout 独立测准确率，interaction-group 报告实际 groups=0。结构成长尚未触发。

[本轮结果复审](reference/M5_POST_P5_2_REVIEW_20260913.md)解释这些结论与源码依据。[推进方案](active/roadmap/03_CURRENT_EXECUTION.md)给出 P5.2a–d、真实语料准入、runtime 采用及后续工程的依赖、验收和停止点。

**唯一推荐下一步：P5.2a 预测驱动执行预注册**。先明确模型输出如何成为真实 ActionIntent、参数来源、outcome 反馈和对照；新的实验阈值与默认行为切换仍需对应设计/批准记录。

## 最新证据入口

| 证据 | 结论与使用范围 |
|---|---|
| [K 轴晋级宣布](reference/M5_K_PROMOTION_DECLARATION_20260912.md) / [scorecard v8](../reports/taiji_m5_k_axis_scorecard_v8_20260912.json) | K 轴限定晋级与 13 项 promotion gate；不等于默认产品采用或结构成长 |
| [P4.14 联合课程](reference/M5_K_P4_14_JOINT_COURSE_PREREGISTRATION_20260911.md) | K continuation → post-K 重构输入 → G 求解器，4/4 cell |
| [P5.1 内容迁移](reference/M5_P5_1_SOURCED_KNOWLEDGE_TRANSFER_PREREGISTRATION_20260912.md) | 词表重叠下的受控内容收益 |
| [P5.1b 语义改写](reference/M5_P5_1B_SEMANTIC_PARAPHRASE_TRANSFER_PREREGISTRATION_20260912.md) | 失败证据；小语料下辨别力不足 |
| [P5.1c 对比训练](reference/M5_P5_1C_CONTRASTIVE_DISCRIMINATION_PREREGISTRATION_20260912.md) | 内容家族辨别改善 |
| [P5.1d encoder 注入](reference/M5_P5_1D_SEMANTIC_ENCODER_INJECTION_PREREGISTRATION_20260912.md) | 锚定语义 encoder 接入与回归 |
| [P5.1e 构造同预算](reference/M5_P5_1E_SAME_BUDGET_CONTENT_BENEFIT_PREREGISTRATION_20260912.md) | 构造载体内容收益 |
| [P5.1f 真实语料失败](reference/M5_P5_1F_REAL_CORPUS_SAME_BUDGET_PREREGISTRATION_20260912.md) | 配额口径、准入、指标饱和与超预算问题 |
| [P5.1g 配额对照](reference/M5_P5_1G_REAL_CORPUS_QUOTA_BUDGET_PREREGISTRATION_20260912.md) / [报告](../reports/taiji_p5_1g_real_corpus_quota_budget_20260912.json) | trial 内容收益成立；两臂 admission 仍失败 |
| [P5.2 Workbench 合同](reference/M5_P5_2_WORKBENCH_SIMULATION_CONTRACT_PREREGISTRATION_20260912.md) / [报告](../reports/taiji_p5_2_workbench_simulation_contract_20260912.json) | 真实合同执行与动作预测分别验证；尚未闭合预测执行与群体迁移 |

## 文档职责

| 文档 | 职责 |
|---|---|
| [当前推进方案](active/roadmap/03_CURRENT_EXECUTION.md) | 唯一下一步、依赖、交付、验收、讨论节点 |
| [本轮结果复审](reference/M5_POST_P5_2_REVIEW_20260913.md) | 证据核验、边界修正、方案选择依据 |
| [核心需求](active/TAIJI_CORE_REQUIREMENTS.md) | 认知所有权、协作、行动、记忆、持续学习与成长的长期目标 |
| [原生架构](active/TAIJI_NATIVE_ARCHITECTURE_V1.md) / [继承式成长](active/architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md) | 状态、学习与成长机制约束 |
| [Seed 架构](active/SEED_ARCHITECTURE.md) / [架构方向](active/ARCHITECTURE_DIRECTION_2026_08.md) | 产品/认知职责和 legacy 边界 |
| [P4 结果复审](reference/M4V2_POST_C_STAGE_RESULT_REVIEW_20260910.md) | 历史容量、表征、学习规则等结论来源 |
| [整理前执行快照](archive/history/20260913_plan_reorganization/EXECUTION_BEFORE_REVIEW.md) / [整理前索引](archive/history/20260913_plan_reorganization/PLAN_INDEX_BEFORE_REVIEW.md) | 完整历史流水；旧“下一步”不授予执行许可 |

## 维护与 CI

active 只保留当前决策与核心约束；reference 保留预注册/结果解释；archive 保存历史；manifests 保存冻结数据合同。失败报告不覆盖，临时产物逐项核验后清理。

截至本次查询，最新远端 CI 在旧提交 `c7bbd389` 的 Ruff 阶段失败，当前审查 HEAD 尚无该次远端验证；P5 历史“零新增”检查与 CI 正式门槛分开记录。详细处理顺序见推进方案 §11。
