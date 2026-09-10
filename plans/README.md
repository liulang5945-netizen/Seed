# Seed / Taiji 计划与架构入口

> 2026-09-10 依据 main cccea110 的源码、checkpoint 与报告复审。唯一执行顺序：[当前计划](active/roadmap/03_CURRENT_EXECUTION.md)。

目前已有五类 K worker 和 fast/slow＋replay 实现。C-stage v2 在 D/R/A 小型评估上四门通过，FS 比 C 的平均 MSE 改善约 0.001868，弱类改善约 0.002495。该组合使用额外 replay；三组 v4 K worker 权重实测相同，全五类泛化与独立拆分收益仍待验证，can_promote=false。

**P0 等 replay 配对诊断已完成；P1.1 数据合同已修复并通过；P2 pilot 未晋级；P2.1 输出/行动链诊断、P2.2 安全 bridge、P2.3 continuation 数据合同/targeted learning、P2.4 retention、P2.5 novel-composition probe 和 P2.6 novel K2 learning 已完成。唯一下一步：P2.7 independent holdout generalization probe。** P0 已证明当前 FS 有效轨迹与直接 continuation＋相同 replay 在 `4.76837158203125e-7` 峰值差内等价，因此后续效果基线采用直接 continuation＋replay；FS 保留为可恢复状态实现候选。P2 显示连续 MSE 下降但 goal/content 命中未提升；P2.1 确认 6/10 行低于原生 `0.55` confidence floor；P2.2 的 recovery/alignment 是 oracle control；P2.3 candidate-only fit 未增加能力且损害原五类高证据读出；P2.4 已修复保持；P2.5 定位了 TypeScript＋可用 toolchain 新组合的 K2 content 缺口；P2.6 在 retention 约束下将该目标从 0/2 学到 2/2，但样本仍小，尚不能宣称泛化或晋级。SGK v1 继续暂停。

阶段已按 601413cd 收束为“研究审计完成、模型能力尚未验收”；本轮完成了 P2 小预算训练 pilot、P2.1 只读诊断、P2.2 安全 bridge、P2.3 continuation 数据合同/targeted learning、P2.4 retention、P2.5 novelty probe 和 P2.6 novel learning，但没有 promotion 成绩。**当前只执行 P2.7 independent holdout generalization probe：加载 P2.6 交错学习 checkpoint，在全新项目/路径上 validation-only 评分；不追加 fit、不降低 confidence floor、不把 host/oracle 控制当模型能力、不进入 P3。** P3–P5 保留为后续路线，不并行开工；完整成果分类、停止条件及文档边界均在当前计划中。

- [本轮结果复审](reference/M4V2_POST_C_STAGE_RESULT_REVIEW_20260910.md)：实际收益、预算混淆、父 worker 重复、seed 循环、恢复缺口与修订依据。
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
