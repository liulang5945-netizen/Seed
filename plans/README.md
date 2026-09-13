# Seed / Taiji 计划与架构入口

> 更新：2026-09-13；结果审查基线 `f9825943`。执行顺序仅由[当前推进方案](active/roadmap/03_CURRENT_EXECUTION.md)决定。

## 当前阶段

**K 轴已限定晋级；路线 C、A 均已完成。A 表征机械生效，但迁移仍未通过；下一步进入路线 B 的评分与任务可达性审查。**

边界：K/G 为 opt-in、进程内状态；P5.1g 为未准入 trial；P5.2a 已接真实预测执行但泛化门未过；修复后的 P5.2b 当前 groups=0、rejected=6。A 的预测两档不代表正确排序，跨任务覆盖不代表同任务协作。结构成长未触发。

[本轮结果复审](reference/M5_POST_ROUTE_A_REVIEW_20260913.md)核对评分参照、预测方向及任务上界。[详细方案](active/roadmap/03_CURRENT_EXECUTION.md)给出 B0/B1/B2、在线学习、真实语料准入与产品采用的步骤、对照、验收和停止点。

**唯一推荐下一步：路线 B0 —— 评分与任务可达性审查，形成路线 B 预注册草案。** 先统一收益参照并确认协作主张可测，再冻结条件化关系表示与新测试；不直接加列重跑旧 Gate。

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
| [P5.2a 预测驱动执行](reference/M5_P5_2A_PREDICTIVE_EXECUTION_PREREGISTRATION_20260913.md) / [报告](../reports/taiji_p5_2a_predictive_execution_20260913.json) | `predictive_execution_insufficient`；八门过、门 9 迁移失败；失败全为动作选择错误 |
| [P5.2b 群体因果语料](reference/M5_P5_2B_GROUP_CAUSAL_CORPORA_PREREGISTRATION_20260913.md) / [报告](../reports/taiji_p5_2b_group_causal_corpora_20260913.json) | 修复后语料合同九门通过；当前 groups=0、rejected=6，旧 admitted pair 无效 |
| [P5.2c′ 未见组合迁移结果](../reports/M5_P5_2C_PRIME_UNSEEN_COMBINATION_TRANSFER_RESULT_20260913.md) | transfer_no_gain；未见面只有一对且冗余，非一般选择能力结论 |
| [P5.2c″ 未见组合迁移结果](../reports/M5_P5_2C_DOUBLE_PRIME_UNSEEN_COMBINATION_TRANSFER_RESULT_20260913.md) | 路线 C 完成，两对留出、旧表示常量；block-3 只量化未修可达性 |
| [P5.2c‴ 表征修复结果](../reports/M5_P5_2C_TRIPLE_PRIME_REPRESENTATION_REPAIR_RESULT_20260913.md) / [本轮复审](reference/M5_POST_ROUTE_A_REVIEW_20260913.md) | 路线 A 完成，秩和区分度提升但收益未过；评分参照和任务上界需审查 |

## 文档职责

| 文档 | 职责 |
|---|---|
| [当前推进方案](active/roadmap/03_CURRENT_EXECUTION.md) | 唯一下一步、依赖、交付、验收、讨论节点 |
| [技术债登记册](active/roadmap/05_TECH_DEBT_REGISTER.md) | A1/A2 已结项，历史 SystemExit 待定位；新评分/CI 约束见最新状态补充 |
| [P5.2c′ 决策历史](active/roadmap/06_P5_2C_PRIME_NEXT_STEP_DECISION.md) | 用户既定顺序先 C 后 AB；C/A 已完成，原选项不是当前待选事项 |
| [P5.2c″ 预注册](reference/M5_P5_2C_DOUBLE_PRIME_UNSEEN_COMBINATION_TRANSFER_PREREGISTRATION_20260913.md) | 路线 C 的冻结判据：两个已验证互补的未见组合、block-3 量化、**声明只修测量仪器不修表征** |
| [P5.2c″ 结果报告](../reports/M5_P5_2C_DOUBLE_PRIME_UNSEEN_COMBINATION_TRANSFER_RESULT_20260913.md) | `transfer_signal_constant`；九门 8 过；三项目标全达成；含「先 C 后 A」排期验证 |
| [本轮结果复审](reference/M5_POST_P5_2_REVIEW_20260913.md) | 证据核验、边界修正、方案选择依据 |
| [核心需求](active/TAIJI_CORE_REQUIREMENTS.md) | 认知所有权、协作、行动、记忆、持续学习与成长的长期目标 |
| [原生架构](active/TAIJI_NATIVE_ARCHITECTURE_V1.md) / [继承式成长](active/architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md) | 状态、学习与成长机制约束 |
| [Seed 架构](active/SEED_ARCHITECTURE.md) / [架构方向](active/ARCHITECTURE_DIRECTION_2026_08.md) | 产品/认知职责和 legacy 边界 |
| [P4 结果复审](reference/M4V2_POST_C_STAGE_RESULT_REVIEW_20260910.md) | 历史容量、表征、学习规则等结论来源 |
| [整理前执行快照](archive/history/20260913_plan_reorganization/EXECUTION_BEFORE_REVIEW.md) / [整理前索引](archive/history/20260913_plan_reorganization/PLAN_INDEX_BEFORE_REVIEW.md) | 完整历史流水；旧“下一步”不授予执行许可 |

## 维护与 CI

active 只保留当前决策与核心约束；reference 保留预注册/结果解释；archive 保存历史；manifests 保存冻结数据合同。失败报告不覆盖，临时产物逐项核验后清理。

当前 f9825943 的 CI 34753643532 在本轮查询时尚在运行，两条 Linux 已在 Ruff 失败；前端、Docker、启动 smoke 成功。架构迁移的局部测试通过不等于全仓 CI 通过，处理顺序见推进方案 §9。
