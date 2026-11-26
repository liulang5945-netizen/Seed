# Seed / Taiji 活动计划索引

> 文档状态：[索引] — 只分类和导航。当前唯一入口是[MAINLINE](../MAINLINE.md)，唯一权威主线是[MASTER-PLAN](../MASTER-PLAN.md)；本页只做指针，不写结论。

## 唯一分类

| 状态 | 文档 | 维护边界 |
|---|---|---|
| 主线 | [MAINLINE](../MAINLINE.md) | 单入口：只留状态、队首、链接、待裁定指针 |
| 权威 | [MASTER-PLAN](../MASTER-PLAN.md) | 唯一权威主线：状态表、依赖、权威源、冲突消解、盘点 |
| 状态 | [03_CURRENT_EXECUTION.md](roadmap/03_CURRENT_EXECUTION.md) | 证据与审批台账；唯一队首标题只静态跳转总计划 |
| 索引 | [10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md](roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md) | 兼容旧章节入口；不另排工作包或复制阶段状态 |
| 规范 | [TAIJI_CORE_REQUIREMENTS.md](TAIJI_CORE_REQUIREMENTS.md)、[TAIJI_NATIVE_ARCHITECTURE_V1.md](TAIJI_NATIVE_ARCHITECTURE_V1.md) | 长期约束与认知所有权，不冒称实现 |
| 规范 | [STAGE_GATES_AND_STOPPING.md](STAGE_GATES_AND_STOPPING.md)、[MECHANISM_ACCEPTANCE.md](MECHANISM_ACCEPTANCE.md) | 阶段成绩单／停止规则与机制终态／训练放行（D-20261124-464）；N31 起所有卡适用 |
| N31 结案 | [CLOSEOUT-N31](../reference/CLOSEOUT-N31_20261124.md) | 有界交付；能力边界、负结果、缺口清单、资产状态（D-20261124-499） |
| N32 收口 | [CLOSEOUT-N32](../reference/CLOSEOUT-N32_20261125.md) | 有界交付＋R1 确认正例；终态、负结果、缺口、资源、N33 准入条件（D-20261125-552） |
| N33 收口 | [CLOSEOUT-N33](../reference/CLOSEOUT-N33_20261126.md) | 三正例＋一条部分确认；生长收益未签署；资源、seed 状态、N34 准入条件（D-20261126-609） |
| **N34 收口** | [CLOSEOUT-N34](../reference/CLOSEOUT-N34_20261126.md) | N34.9.90 六轴＋N34.5.90 百会话同主体联合全部确认通过；零 `check/final`（D-20261126-664） |
| **N35 收口** | [CLOSEOUT-N35](../reference/CLOSEOUT-N35_20261126.md) | N35 有界部分确认收口（`N35.1.91` 8/9 门 3/3 过，`944.33/1,500 CPU`，零 `check/final`；D-20261126-682） |
| **N36 有界交付** | [N36-ADMISSION](../reference/N36-ADMISSION_20261126.md)、[DESIGN-N36-1-01](../reference/DESIGN-N36-1-01_20261126.md)、[RESULT-N36-1-01](../reference/RESULT-N36-1-01_20261126.md) | 有界可运行交付/接管包就绪（`delivery_package_ready`，`D1..D5` 5/5 过，`23.06/100 CPU`，零 `check/final`；D-20261126-683/684） |
| 目标覆盖矩阵 | [COVERAGE-MATRIX_20261126](../reference/COVERAGE-MATRIX_20261126.md) | v4 条款→✅/🟡/❌、证据与缺口、下一步优先级（D-20261126-615） |
| 探索证据汇总 | [EVIDENCE-SUMMARY-EXPLORATION](../reference/EVIDENCE-SUMMARY-EXPLORATION_20261126.md) | N33＋N34.9.90＋N34.5.90 确认级汇总＋缺口表（D-20261126-645/659/664） |
| **N34.5.90 长跑联合确认** | [RESULT-N34-5-90](../reference/RESULT-N34-5-90_20261126.md) | 确认 seed `20261811–13` 同主体百会话联合确认通过（`confirm_positive`，`65.83 CPU`；D-20261126-662/663） |
| **N34.9.90 六轴确认** | [RESULT-N34-9-90](../reference/RESULT-N34-9-90_20261126.md) | 确认 seed `20261801–03` 六轴全过（`confirm_all_passed`，`116.59 CPU`；D-20261126-658/659） |
| 生长决策学习线 | [RESULT-N34-3-04](../reference/RESULT-N34-3-04_20261126.md) | 状态相关生长决策 v2 探索正例（3/3 达按型 oracle，D-20261126-657；已入 N34.9.90 确认） |
| 中文任务设施 | [DESIGN-NCH-1-01](../reference/DESIGN-NCH-1-01_20261126.md) | 封闭词表自然中文连续任务 harness（零训练 `env_ready`；D-20261126-616） |
| 证据可信度设施 | [DESIGN-ECL-1-01](../reference/DESIGN-ECL-1-01_20261126.md) | 证据可信度账本与纠正（零训练 `ledger_ready`；D-20261126-617） |
| 可信度学习线 | [RESULT-ECL-1-03](../reference/RESULT-ECL-1-03_20261126.md) | 在线可信度学习探索正例（`credibility_learned_positive`；D-20261126-639/640） |
| 不确定性设施 | [DESIGN-UNC-1-01](../reference/DESIGN-UNC-1-01_20261126.md) | ECE/Brier＋温度校准＋中文分级表达（零训练 `calibration_ready`；D-20261126-618） |
| 不确定性学习线 | [RESULT-UNC-1-02](../reference/RESULT-UNC-1-02_20261126.md) | 学习式不确定性估计探索正例（`uncertainty_learned_positive`；D-20261126-641/642） |
| 主体循环设施 | [DESIGN-SUB-1-01](../reference/DESIGN-SUB-1-01_20261126.md) | 四项设施集成冒烟（零训练 `loop_ready`；D-20261126-619） |
| 多模态物理设施 | [DESIGN-PHY-1-01](../reference/DESIGN-PHY-1-01_20261126.md) | 两族动力学＋三模态＋可纠正因果预测（零训练 `physics_ready`；D-20261126-621） |
| 多模态学习线 | [RESULT-PHY-2-01](../reference/RESULT-PHY-2-01_20261126.md) | 跨族物理因果迁移 v2 探索正例（无泄露，碰撞召回 0.975–0.992；`multimodal_v2_positive`；D-20261126-652） |
| 中文任务学习线 | [RESULT-NCH-2-01](../reference/RESULT-NCH-2-01_20261126.md) | 中文连续指代任务 v2 探索正例（跨回合 GRU vs 无状态；`chinese_continuous_v2_positive`；D-20261126-653） |
| 开放中文探索线 | [DESIGN-NCH-3-01](../reference/DESIGN-NCH-3-01_20261126.md)、[RESULT-NCH-3-04](../reference/RESULT-NCH-3-04_20261126.md)、[RESULT-NCH-3-05](../reference/RESULT-NCH-3-05_20261126.md)、[RESULT-NCH-3-06](../reference/RESULT-NCH-3-06_20261126.md) | 开放自然中文连续任务设施（`open_env_ready`）、`v3` 探索正信号（`signal_positive`；D-665..669）与 `v4/v5` 探索（`signal_partial`；D-675..677） |
| **N35 准入备忘** | [MEMO-N35-ADMISSION-DECISION](../reference/MEMO-N35-ADMISSION-DECISION_20261126.md) | 模型级联合确认＋首次 `check` 准入四项签字决策（D-20261126-670） |
| **N35 准入（已生效）** | [N35-ADMISSION](../reference/N35-ADMISSION_20261126.md)、[SCORECARD-N35](../reference/SCORECARD-N35_20261126.md) | owner 批复方案 A；确认 seed `20261901–03`；`R0` 成绩单冻结（D-20261126-671） |
| **N35.1.90 联合确认** | [DESIGN-N35-1-90](../reference/DESIGN-N35-1-90_20261126.md)、[RESULT-N35-1-90](../reference/RESULT-N35-1-90_20261126.md) | `engineering_failure`（`650.03/650 CPU`：`20261903` 超时且 `20261901` 未过 `R0-1` `coref=0.6250<0.70`；`20261901–03` 作废，备 `20261911–13` 未读；D-20261126-673/674） |
| **N35 口径重估** | [MEMO-N35-SCOPE-REEVAL](../reference/MEMO-N35-SCOPE-REEVAL_20261126.md) | 暂停 `NCH.3` 探索线（`2,490.72/2,500 CPU`），呈报 `B1 / B2 / B3` 待裁定（D-20261126-678） |
| **N35.1.91（方案 B1）** | [SCORECARD-N35-B1](../reference/SCORECARD-N35-B1_20261126.md)、[DESIGN-N35-1-91](../reference/DESIGN-N35-1-91_20261126.md)、[RESULT-N35-1-91](../reference/RESULT-N35-1-91_20261126.md) | `confirm_partial`（`294.30/450 CPU`，8/9 门 3/3 过；`20261911–13` 作废，零 `check/final`；D-20261126-679..681） |
| 双任务主体循环 | [DESIGN-SUB-1-02](../reference/DESIGN-SUB-1-02_20261126.md) | 中文＋物理交替集成冒烟（零训练 `mixed_loop_ready`；D-20261126-622） |
| N34 准入备忘 | [MEMO-N34-ADMISSION-DECISION](../reference/MEMO-N34-ADMISSION-DECISION_20261126.md) | 四项签字解锁/建议值/风险（D-20261126-623） |
| N34 就绪包 | [READY-N34](../reference/READY-N34_20261126.md) | 一键复验全部零训练资产（`bundle_ready`，11/11；D-20261126-624） |
| 目标修订 | [主张对照清单](../reference/REVIEW-GOAL-CLAIMS_20261125.md)、[修订稿 v4 草案](../reference/DRAFT-GOAL-REVISION-V4_20261125.md)（未生效） | owner 签字后合并进核心需求 §2.1（D-20261125-501） |
| N32.2（R1） | [开线卡](../reference/DESIGN-N32-2-00_20261125.md)、[结线卡](../reference/RESULT-N32-2-CLOSE_20261125.md) | 已结线：`budget_stop`「R1 未检验」（D-20261125-514） |
| N32 放行 | [REVIEW-N32-ADMISSION](../reference/REVIEW-N32-ADMISSION_20261124.md)、[SCORECARD-N32.md](SCORECARD-N32.md)（已实现、未放行） | 放行01–07 逐条状态；N32 成绩单草案（D-20261124-495） |
| 运营 | [02_GATES_AND_CI.md](roadmap/02_GATES_AND_CI.md)、[05_TECH_DEBT_REGISTER.md](roadmap/05_TECH_DEBT_REGISTER.md)、[07_MINI_MODEL_DELIVERY.md](roadmap/07_MINI_MODEL_DELIVERY.md)、[08_UPSTREAM_SYNC_PLAYBOOK.md](roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md) | 门禁、债、能力/交付、上游重放；不定义方向 |
| 组件 | [09_NEXT_MAINLINE_PLAN.md](roadmap/09_NEXT_MAINLINE_PLAN.md) | N1–N6 机制词典与历史映射；**已 superseded，不是当前主计算图** |
| 历史 | [01_SCOPE_AND_PHASES.md](roadmap/01_SCOPE_AND_PHASES.md)、[04_EXECUTION_PLAN.md](roadmap/04_EXECUTION_PLAN.md)、[06_P5_2C_PRIME_NEXT_STEP_DECISION.md](roadmap/06_P5_2C_PRIME_NEXT_STEP_DECISION.md)、[ARCHITECTURE_DIRECTION_2026_08.md](ARCHITECTURE_DIRECTION_2026_08.md)、[SEED_ARCHITECTURE.md](SEED_ARCHITECTURE.md)、[SEED_DEVELOPMENT_ROADMAP_2026_08.md](SEED_DEVELOPMENT_ROADMAP_2026_08.md)、[TAIJI_CONTINUAL_DEVELOPMENT_V2.md](architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md) | 解释当时状态、批准与旧路线；不发布当前指令 |
| 证据 | [11_CI_FLOAT_EVIDENCE_FOR_OWNER_20261025.md](roadmap/11_CI_FLOAT_EVIDENCE_FOR_OWNER_20261025.md)、[12_LINE2_EVIDENCE_CLOSEOUT_20261025.md](roadmap/12_LINE2_EVIDENCE_CLOSEOUT_20261025.md) | 机器读数与结案回执；不改写为能力通过 |
| 索引 | [PLAN_INDEX.md](PLAN_INDEX.md)（本页）、[plans/README.md](../README.md)、[ID_MAP_N](../reference/ID_MAP_N.md)（N 编号对照） | 只导航 |

## 步骤与原合同映射（不登记阶段状态）

> D-20261124-464 起新工作统一用 N31 起的阶段编号；下表旧步骤 01–13 的映射见 [ID_MAP_N](../reference/ID_MAP_N.md)。

实际状态、准入与实施次序只见[总计划](../MAINLINE.md#3-唯一执行队列)；原合同路径和实验身份保持不变。

| 总计划步骤 | 原合同身份 | 文档 |
|---|---|---|
| [01](../MAINLINE.md#3-唯一执行队列) | R1 语义 | [SPEC-NATIVE-CONTENT-CORE_R1_SEMANTIC_CONTRACT_20261030.md](../reference/SPEC-NATIVE-CONTENT-CORE_R1_SEMANTIC_CONTRACT_20261030.md) |
| [02](../MAINLINE.md#3-唯一执行队列) | R2 候选与比较 | [SPEC-NATIVE-CONTENT-CORE_R2_CANDIDATE_COMPARISON_20261030.md](../reference/SPEC-NATIVE-CONTENT-CORE_R2_CANDIDATE_COMPARISON_20261030.md) |
| [03](../MAINLINE.md#3-唯一执行队列) | R3 新核心实现 | [SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md](../reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md) |
| [04](../MAINLINE.md#3-唯一执行队列) | R3b-1 修正旧图基线 | [R3 记录 §6](../reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md#6-r3b1-修正旧图基线20261030) |
| [05](../MAINLINE.md#3-唯一执行队列) | R3b-2 普通端到端基线 | [R3 记录 §7](../reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md#7-r3b2-普通端到端基线20261030) |
| [06](../MAINLINE.md#3-唯一执行队列) | R4，含课程准备 | [定价与 R5 数值门](../reference/SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md) |
| [07](../MAINLINE.md#3-唯一执行队列) | R5 比较与结案 | [R4 合同 §9](../reference/SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md#9-r5-执行结果20261030) |
| [08](../MAINLINE.md#3-唯一执行队列) | R5b 复制轨候选 | [SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md](../reference/SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md) |
| [09](../MAINLINE.md#3-唯一执行队列) | 连续生命/递归进化共同内核 | [算法主稿](../reference/TAIJI_CORE_ALGORITHM_V1_DRAFT_20261031.md)＋[R021](../reference/SPEC-TAIJI-CORE_REFERENCE_GRAPH_R021_20261031.md)＋[R022](../reference/SPEC-TAIJI-CORE_CREDIT_R022_20261031.md)＋[R023](../reference/SPEC-TAIJI-CORE_GROWTH_R023_20261031.md)＋[R024](../reference/SPEC-TAIJI-CORE_RESTORE_R024_20261031.md)＋[R025](../reference/SPEC-TAIJI-CORE_ACTIVITY_R025_20261031.md)＋[R026](../reference/SPEC-TAIJI-CORE_PLASTICITY_R026_20261031.md)＋[R027恢复门](../reference/SPEC-TAIJI-CORE_GATES_R027_20261031.md)；实验权限与正式冻结合同另定 |
| [10](../MAINLINE.md#3-唯一执行队列) | 按任务需要的 T3 | 工作卡统一见总计划，合同随对应准入建立 |
| [11](../MAINLINE.md#3-唯一执行队列) | 自然中文连续任务 / L3 | 工作卡统一见总计划，合同随对应准入建立 |
| [12](../MAINLINE.md#3-唯一执行队列) | 跨会话学习与保持 / L4 | 工作卡统一见总计划，合同随对应准入建立 |
| [13](../MAINLINE.md#3-唯一执行队列) | 条件自主选择与扩域 / L5 | 工作卡统一见总计划，合同随对应准入建立 |

[重构方案](../reference/PLAN-NATIVE-CONTENT-CORE_REDESIGN_20261030.md)仅作设计依据，不是另一份实施排期。

## 旧资产

N177/F/N178、VISION、旧 T2/T3、M0–M8、R2 日对话、UI 生命页等旧文档全部登记在[LEGACY_ASSET_INDEX](../reference/LEGACY_ASSET_INDEX.md)。它们只作证据、基线或长期储备，其“当前/下一步”只解释写作时点；总计划步骤 09/10 的新工作卡与旧实现身份分开。

## 维护规则

1. 新方向只能进[MAINLINE](../MAINLINE.md)第 3 节，不新增并列主线。
2. 03 的“当前唯一下一步”入口只跳转总计划；本页不登记动作、审批或阶段状态。
3. active 文档前 10 行必须有 `> 文档状态：[...]`。
4. 本页必须覆盖 `plans/active/` 下全部 `.md`；由单主线审计脚本检查。
5. 旧文不因收敛删除或改名；只降级调度权。

本轮项目级整理的删除/归档与恢复索引见[历史整理](../archive/history/n_mainline_20261030/README.md)；旧N1归因交接已由原收官判读取代，不再留活动快照。

2026-10-31根目标纠偏：09.1共同内核合同设计前置；12/13是长期与成熟自主范围验收，不是首次接入生命、学习或生长。局部固定容量实验不作为最终主体形态。
