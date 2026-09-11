# Seed / Taiji 计划与架构入口

> 2026-09-11 依据当前 main 的源码、checkpoint 与报告复审。唯一执行顺序：[当前计划](active/roadmap/03_CURRENT_EXECUTION.md)。

当前主线是 M5 K 轴持续学习与 G 选择机制。K1/K2/K3 standalone 证据已由 scorecard v3 收束；P4.7–P4.10 关闭容量/旧表示路线并定位 parent-relative 特征因子化；P4.11 求解器、P4.12 课程级验证、P4.13 两阶段晋级课程均已完成。最新 scorecard v4 只读汇总了这条 G 侧证据链，`g_solver_mechanism_course_closed=true`，但 `promotion_gate=false`、`can_promote=false`、`growth_admitted=false`。

**P0 等 replay 配对诊断已完成；P1.1 数据合同已修复并通过；P2.7 局部跨项目/路径泛化、P3.0–P3.6 状态/所有权/行为保持链、P4.0–P4.6 固定容量与更新规则诊断、P4.7 容量干净检验、P4.8 表征合同、P4.9 特征空间探针、P4.10 特征因子化、P4.11 投影求解器、P4.12 课程级验证、P4.13 两阶段晋级课程均已完成。** P4.11 在两个 seed 上同时通过新任务与双保持门；P4.12 为 3 个身份批次 × 3 个 seed 的 9/9 projected cell；P4.13 为 9/9 两阶段 cell，A+B+保持累积约束投影零违反、向后保持零失败，机械 checkpoint/rollback 和绝对资源预算全通过。机制证据闭合不等于晋级：K worker 联合课程尚未预注册/运行，默认 runtime rollout review 尚未执行，所有 owner 与默认 runtime 仍保持 fail-closed。

阶段已按“研究审计完成、模型能力尚未验收”收束；固定容量与旧表示路线已关闭，求解器机制下的 G 侧课程已闭合。**下一步唯一动作是预注册 K worker 联合课程，把 P2.6/P2.7 continuation 机械与求解器机制接到同一父代；在此之前不解冻 owner、不接默认 runtime、不召开晋级评审。** P5、CUDA、客户端外围保留为后续路线，不并行开工；完整成果分类、停止条件及文档边界均在当前计划中。

- [本轮结果复审](reference/M4V2_POST_C_STAGE_RESULT_REVIEW_20260910.md)：实际收益、预算混淆、父 worker 重复、seed 循环、恢复缺口与修订依据。
- [M5 K 轴 scorecard v4 合同](reference/M5_K_AXIS_SCORECARD_V4_CONTRACT_20260911.md)：P4.12/P4.13 求解器证据的只读汇总、晋级边界与唯一后续动作。
- [P4.11 投影求解器预注册](reference/M5_K_P4_11_PROJECTION_SOLVER_PREREGISTRATION_20260911.md)：固定容量路线重开后的更新机制证据。
- [P4.12 课程级验证预注册](reference/M5_K_P4_12_COURSE_LEVEL_VALIDATION_PREREGISTRATION_20260911.md)：9-cell 课程级验证。
- [P4.13 两阶段晋级课程预注册](reference/M5_K_P4_13_PROMOTION_COURSE_PREREGISTRATION_20260911.md)：A → B 累积投影与向后保持。
- [M5 K 轴 scorecard v4 报告](../reports/taiji_m5_k_axis_scorecard_v4_20260911.json)：机器可读的最终边界与 veto 状态。
- [P0 等 replay 诊断报告](../reports/taiji_m5_k_p0_equal_replay_diagnostic_20260910.json)：同课程、同 replay、轨迹差异和 checkpoint preflight 结果。
- [P1 数据契约审计报告](../reports/taiji_m5_k_p1_data_contract_audit_20260910.json)：typed mask、五类有效输入、split、course seed 与 model seed 独立性结果；当前 `can_start_p2=false`。
- [P1 数据 manifest](manifests/taiji_m5_k_p1_data_manifest_v1.json)：450 条可重建 audit record 及版本化 mask contract。
- [P1 v2 数据契约审计报告](../reports/taiji_m5_k_p1_data_contract_audit_v2_20260910.json)：修复后 Gate 为 `passed`，`can_start_p2=true`。
- [P1 v2 数据 manifest](manifests/taiji_m5_k_p1_data_manifest_v2.json)：450 条 train + 10 条 validation 的可重建 contract。
- [P2 v2 validation pilot](../reports/taiji_m5_k_p2_validation_pilot_v2_20260910.json)：50 条均衡 wake + 10 条 replay，独立 checkpoint preflight 通过；连续 MSE 改善但命中率未改善，`can_promote=false`。
- [P2.1 输出/行动链诊断](../reports/taiji_m5_k_p2_output_action_diagnostic_20260910.json)：只读复建 10 条 validation，K1→K2 级联、planner、隔离 Workbench 与资源/恢复证据；未训练、未读 sealed、`can_promote=false`。
- [P2.2 安全 bridge canary](../reports/taiji_m5_k_p2_2_safety_bridge_canary_20260910.json)：typed abstention、根目录 `workspace.list` recovery、K2 世界对齐和隔离 Workbench 结果；工程控制通过，但 recovery/alignment 明确为 oracle，不代表模型能力。
- [P2.3 continuation manifest](manifests/taiji_m5_k_p2_3_recovery_continuation_manifest_v1.json)：6 条 train + 2 条 validation，host recovery 与 fit-eligible candidate 分层，K1/K2 digest 和 project/path/template 隔离。
- [P2.3 continuation contract](../reports/taiji_m5_k_p2_3_recovery_continuation_contract_20260910.json)：数据合同通过；只建 artifact、未训练、未读 sealed、`can_promote=false`。
- [P2.3 targeted learning pilot](../reports/taiji_m5_k_p2_3_targeted_learning_pilot_20260910.json)：训练前后 checkpoint 独立恢复通过；continuation 2/2 无新增能力，candidate-only fit 造成旧类 K1/K2 退化，`can_promote=false`。
- [P2.4 retention canary](../reports/taiji_m5_k_p2_4_retention_canary_20260910.json)：50 条 P2 rehearsal digest/平衡复现，交错 continuation 后旧类保持和安全 abstention 通过；continuation 仍无新增能力，`can_promote=false`。
- [P2.5 novel-composition probe](../reports/taiji_m5_k_p2_5_novel_composition_probe_20260910.json)：TypeScript＋可用 toolchain 组合与 P1/P2 路径隔离；frozen parent 的 K1 goal/content 为 2/2、K2 goal 为 2/2 但 K2 content 为 0/2，Workbench 为 2/2；checkpoint 保存/独立恢复通过，validation-only，`can_promote=false`。
- [P2.6 novel K2 learning](../reports/taiji_m5_k_p2_6_novel_learning_20260910.json)：6 条 novel train + 2 条 disjoint validation 与 50 条均衡 rehearsal 交错；新组合 K2 content 从 parent 0/2 到 interleaved 2/2，旧类 K1/K2 与安全 abstention 保持，参数未增长，独立恢复通过；仅为局部学习证据，`can_promote=false`。
- [P2.7 independent holdout generalization](../reports/taiji_m5_k_p2_7_generalization_20260910.json)：P2.6 learned checkpoint 在 2 个全新 project、4 条全新 path 上 validation-only 达到 K1/K2 goal/content `4/4`、Workbench `4/4`；父模型同一 holdout 的 K2 content 为 `0/4`，独立恢复和参数稳定 Gate 通过；满足 P3 局部泛化入场条件，仍不等于 promotion。
- [P3.0 checkpoint/interrupt-resume contract](../reports/taiji_m5_k_p3_0_checkpoint_contract_20260910.json)：以 P2.6 learned checkpoint 为 parent，uninterrupted、wake 中段中断恢复、replay 边界中断恢复三条轨迹的 worker/budget/RNG/stream digest/final cursor 均一致；tampered cursor、wrong parent、missing lineage 全部拒绝，rollback 独立恢复通过；没有新增 S/G worker、没有参数增长，`can_promote=false`。
- [P3.1 single-cell state preflight](../reports/taiji_m5_k_p3_1_single_cell_20260910.json)：在 P3.0 parent 上回放 4 条 P2.7 holdout、20 个 observation→S update→G selection→K readout→action 事件；uninterrupted、事件中点、阶段边界及两条 resumed 轨迹的 event/owner/worker/budget/RNG/cursor/logical digest 全部一致，篡改 cursor、wrong base、wrong manifest 全部拒绝。S/G 明确为 control-only，K 复用既有 K1/K2；无 fit、无参数增长、`can_promote=false`。对应 owner/事件清单见 [P3.1 manifest](manifests/taiji_m5_k_p3_1_single_cell_manifest_v1.json)。
- [P3.2 K→G owner-transfer preflight](../reports/taiji_m5_k_p3_2_owner_transfer_20260910.json)：K1 只提供 inherited candidate，GSelectionState 持有最终 goal/content，外部 target 不进入运行时；4/4 holdout 上 K-only 与 owner-transfer 的 K1 selection、K2 output、safe abstention、Workbench 全部等价，参数未增长，事件/事件边界/case 边界独立恢复和错误 mask/base/manifest 拒绝全部通过。对应 owner/lineage 清单见 [P3.2 manifest](manifests/taiji_m5_k_p3_2_owner_transfer_manifest_v1.json)。
- [P3.3 G candidate data-signal canary](../reports/taiji_m5_k_p3_3_g_signal_canary_20260911.json)：用 P3.2 冻结 K、P1 重建数据生成 40 条 train + 10 条 validation candidate set；候选数、target coverage、pair/abstain 多样性、竞争干扰、project/path 隔离、runtime target 排除和独立 checkpoint restore 全部通过，`fit_called=false`、K 参数未变、P2.7 holdout 未用于 fit。对应不可变候选清单见 [P3.3 manifest](manifests/taiji_m5_k_p3_3_g_candidate_manifest_v1.json)。
- [P3.3 G-only learning Gate](../reports/taiji_m5_k_p3_3_g_learning_20260911.json)：冻结 P3.2 K，只训练 13 参数 G 320 steps；G 零步/训练后 checkpoint 独立恢复、K digest 前后相同、篡改/错误 lineage 拒绝、P2.7 四条 Workbench 全部 `4/4`。K-only、zero-step、trained-G 在 train/validation/holdout 均无行为差异，故 `can_promote=false`，不追加同质 epoch。候选 identity 已绑定 course/index/项目/模板/输入摘要，避免把重复观测误当同一经验。
- [P3.4 behavior signal Gate](../reports/taiji_m5_k_p3_4_behavior_signal_20260911.json)：冻结 P3.2 K、只做 behavior utility canary，生成 40 条 train + 10 条 validation、每条 2–6 个候选；`candidate_roles={abstain, proposal, reobserve}`，40/50 条有非零 utility margin 和 K-only 行为分歧，30 条 target 为 typed `reobserve`，`fit_called=false`、`can_promote=false`。10 条零 margin B/D 平局不进入训练；对应不可变清单见 [P3.4 manifest](manifests/taiji_m5_k_p3_4_behavior_manifest_v1.json)。
- [P3.5 reobserve-aware G-only learning Gate](../reports/taiji_m5_k_p3_5_g_learning_20260911.json)：只用 P3.4 的 32 条非零 margin train candidate 做 256 steps/13 参数 G-only fit；8 条 validation、P2.7 四条 holdout 未 fit，K1/K2 digest 前后相同，zero-margin train/validation 排除，tamper/lineage 拒绝，30 条 reobserve target 和所有实际 reobserve selection 均投影为可往返、无 `ActionIntent` 的 `workspace.list` typed abstention。contested utility `16.5→30`、behavior target hit `0→30`，Gate 通过但 `can_promote=false`；对应 fit manifest 见 [P3.5 manifest](manifests/taiji_m5_k_p3_5_g_learning_manifest_v1.json)。
- [P3.6 独立行为 holdout 与保持 Gate](../reports/taiji_m5_k_p3_6_behavior_holdout_20260911.json)：不调用 `fit`，加载 P3.5 zero-step/trained-G，在 2 个全新 project、4 条全新 path 上重新计算 behavior utility；trained-G utility `4.0` 高于 zero-step `2.65`，behavior target hit `4/4` 高于 `1/4`，K-only 与 zero-step 相同。3 条 reobserve target/实际 selection 均通过 typed `workspace.list` projection；P1 五类旧类、安全出口、P2.7 Workbench `4/4`、K1/K2 digest、G 独立恢复和 rollback 均通过，`can_promote=false` 保持；对应 holdout manifest 见 [P3.6 manifest](manifests/taiji_m5_k_p3_6_behavior_holdout_manifest_v1.json)。
- [P4.0 固定容量压力扫描](../reports/taiji_m5_k_p4_0_capacity_pressure_20260911.json)：validation-only，冻结 P3.5 trained-G 与 P3.2 K1/K2，在 5 个新案例、3 个新 project、width `2/4/8/12` 和 sequence `1/4/16` 上扫描；trained-G residual 为 `0.32/0/0.54/0.59`，序列 utility 均为 `0.6375`，feature collision 为 `0`，K digest、G lineage 和独立恢复通过，`growth_admitted=false`；对应压力 manifest 见 [P4.0 manifest](manifests/taiji_m5_k_p4_0_capacity_pressure_manifest_v1.json)。
- [P4.1 候选集上下文与公平容量合同预检](../reports/taiji_m5_k_p4_1_context_contract_20260911.json)：validation-only，绑定 P4.0 的 20 个 candidate/behavior artifact，建立 9 维候选集上下文；22 参数 fixed-large reference 与 context-lesion reference 独立恢复，零 fit 下两者选择均与 13 参数 G 一致，归因 `inconclusive`；对应 manifest 见 [P4.1 manifest](manifests/taiji_m5_k_p4_1_context_contract_manifest_v1.json)。
- [P4.2 隔离训练与公平容量归因](../reports/taiji_m5_k_p4_2_capacity_attribution_20260911.json)：两个 deterministic seed、14 条 fit-eligible 新 train、fixed-small/context-aware-small/fixed-large 三臂均完成独立 checkpoint 保存/恢复；新 holdout 没有稳定容量收益，旧行为 retention 从 parent `4/4` 在多数训练臂/seed 降为 `3/4`，`experiment_passed=false`、`growth_admitted=false`；对应 manifest 见 [P4.2 manifest](manifests/taiji_m5_k_p4_2_capacity_attribution_manifest_v1.json)。下一步是 P4.3 保持约束下的增量学习 Gate。
- [P4.3 保持约束下的增量学习](../reports/taiji_m5_k_p4_3_retention_incremental_20260911.json)：两个 deterministic seed、new-only 与 rehearsal-mix 两臂均完成独立 checkpoint/lineage/篡改/rollback 检查；新 holdout 与 fresh retention 都达到 utility `0.68`，但两臂完全相同，`rehearsal_specific_gain=false`、`outcome=signal_insufficient`、`experiment_passed=false`、`growth_admitted=false`；对应 manifest 见 [P4.3 manifest](manifests/taiji_m5_k_p4_3_retention_incremental_manifest_v1.json)。下一步是 P4.4 保持身份与结构校准 Gate。
- [P4.4 保持身份与结构校准](../reports/taiji_m5_k_p4_4_retention_identity_calibration_20260911.json)：validation-only 从 P3.6 提取结构摘要，在 2 个新 project、4 条新 path 的 sibling retention 上评估 parent、P4.2 三臂和 P4.3 两臂；P4.2 历史退化的 5 个 arm/seed 全部复现，P4.3 两臂/两 seed 也退化，`outcome=retention_failure_reproduced`、`experiment_passed=false`、`growth_admitted=false`；对应 manifest 见 [P4.4 manifest](manifests/taiji_m5_k_p4_4_retention_identity_calibration_manifest_v1.json)。下一步是 P4.5 保持约束与更新规则对照 Gate。
- [P4.5 保持约束与更新规则对照](../reports/taiji_m5_k_p4_5_update_rule_gate_20260911.json)：固定 13 参数 G、两个 deterministic seed、new-only/rehearsal-interleaved/constrained-update 三臂均完成独立 checkpoint/lineage/tamper/rollback 检查；rehearsal 与 new-only 完全相同，trust-region 只在 seed-0 恢复 retention、却退回 parent 新任务表现，seed-1 新任务通过时 retention 仍为 utility `0.8`、target `3/4`，`outcome=update_rule_unresolved`、`experiment_passed=false`、`growth_admitted=false`；对应 manifest 见 [P4.5 manifest](manifests/taiji_m5_k_p4_5_update_rule_gate_manifest_v1.json)。P4.6 已完成，当前不再自动追加实验。
- [P4.6 功能性 parent-preserving objective 对照](../reports/taiji_m5_k_p4_6_functional_parent_objective_20260911.json)：固定 13 参数 G、两个 deterministic seed、new-only/functional 两臂均完成独立 checkpoint/lineage/tamper/rollback 检查；seed-0 的 functional 臂保持达到 utility `1.0`、target `4/4`，但新任务降至 utility `0.585`、target `9/20`、safe violation `6`；seed-1 新任务达到 utility `0.68`、target `12/20`、safe violation `0`，但 retention 仍为 utility `0.8`、target `3/4`；`outcome=functional_update_unresolved`、`experiment_passed=false`、`growth_admitted=false`、`can_promote=false`。固定容量路线停止自动推进，当前入口是架构/目标决策点；对应 [P4.6 manifest](manifests/taiji_m5_k_p4_6_functional_parent_objective_manifest_v1.json)。
- [机器审计](../reports/taiji_m4v2_plan_result_review_20260910.json)：源码摘要、权重对比、资源计数和 retention 反例。
- [历史执行流水](archive/history/20260910_result_review/EXECUTION_HISTORY.md)：保留此前全部记录，旧“下一步”不再授权执行。
- [历史计划入口](archive/history/20260910_result_review/PLAN_INDEX_HISTORY.md)：此前入口及研究进展记录。

## 权威文档

| 文档 | 职责 |
|---|---|
| [当前执行计划](active/roadmap/03_CURRENT_EXECUTION.md) | 唯一下一步、依赖顺序和验收条件 |
| [核心需求](active/TAIJI_CORE_REQUIREMENTS.md) | 长期目标 |
| [原生架构](active/TAIJI_NATIVE_ARCHITECTURE_V1.md) | 核心对象和学习体系 |
| [继承式成长 v2](active/architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md) | 持续学习与结构成长研究依据 |
| [架构方向](active/ARCHITECTURE_DIRECTION_2026_08.md) | 原生身份、成熟技术与 legacy 边界 |
| [Seed 架构](active/SEED_ARCHITECTURE.md) | 客户端、Workbench、provider 所有权 |
| [结果复审](reference/M4V2_POST_C_STAGE_RESULT_REVIEW_20260910.md) | 本轮实测结论与限制 |

## 维护规则

active 保留核心需求、架构和唯一执行计划；reference 保留证据解释；archive 保存完成/失效的执行与调试流水；manifests 保存版本化合同。被替代的预注册须标明状态，历史 JSON/权重保留，不通过改写旧成绩制造通过。

每轮按实际结果更新当前计划并提交。知识语料、MCP/IDE、客户端热插拔、watchdog、CUDA 和视觉发布已排入当前计划 P5。
