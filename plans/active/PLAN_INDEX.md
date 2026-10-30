# Seed / Taiji 活动计划索引

> 文档状态：[索引] — 只分类和导航。方向见[唯一主线](../MAINLINE.md)，队首见[03](roadmap/03_CURRENT_EXECUTION.md)。

## 唯一分类

| 状态 | 文档 | 维护边界 |
|---|---|---|
| 主线 | [MAINLINE](../MAINLINE.md) | 唯一方向、主计算图、执行队列、旧方向处置 |
| 状态 | [03_CURRENT_EXECUTION.md](roadmap/03_CURRENT_EXECUTION.md) | 唯一队首、运行/预算/停止/恢复事实 |
| 工作包 | [10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md](roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md) | 工作包依赖、出口、失败回流、条件后继 |
| 规范 | [TAIJI_CORE_REQUIREMENTS.md](TAIJI_CORE_REQUIREMENTS.md)、[TAIJI_NATIVE_ARCHITECTURE_V1.md](TAIJI_NATIVE_ARCHITECTURE_V1.md) | 长期约束与认知所有权，不冒称实现 |
| 运营 | [02_GATES_AND_CI.md](roadmap/02_GATES_AND_CI.md)、[05_TECH_DEBT_REGISTER.md](roadmap/05_TECH_DEBT_REGISTER.md)、[07_MINI_MODEL_DELIVERY.md](roadmap/07_MINI_MODEL_DELIVERY.md)、[08_UPSTREAM_SYNC_PLAYBOOK.md](roadmap/08_UPSTREAM_SYNC_PLAYBOOK.md) | 门禁、债、能力/交付、上游重放；不定义方向 |
| 组件 | [09_NEXT_MAINLINE_PLAN.md](roadmap/09_NEXT_MAINLINE_PLAN.md) | N1–N6 机制词典与历史映射；**不是当前主计算图** |
| 历史 | [01_SCOPE_AND_PHASES.md](roadmap/01_SCOPE_AND_PHASES.md)、[04_EXECUTION_PLAN.md](roadmap/04_EXECUTION_PLAN.md)、[06_P5_2C_PRIME_NEXT_STEP_DECISION.md](roadmap/06_P5_2C_PRIME_NEXT_STEP_DECISION.md)、[HANDOVER_N1_ATTRIBUTION_20261007.md](roadmap/HANDOVER_N1_ATTRIBUTION_20261007.md)、[ARCHITECTURE_DIRECTION_2026_08.md](ARCHITECTURE_DIRECTION_2026_08.md)、[SEED_ARCHITECTURE.md](SEED_ARCHITECTURE.md)、[SEED_DEVELOPMENT_ROADMAP_2026_08.md](SEED_DEVELOPMENT_ROADMAP_2026_08.md)、[TAIJI_CONTINUAL_DEVELOPMENT_V2.md](architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md) | 解释当时状态、批准与旧路线；不发布当前指令 |
| 证据 | [11_CI_FLOAT_EVIDENCE_FOR_OWNER_20261025.md](roadmap/11_CI_FLOAT_EVIDENCE_FOR_OWNER_20261025.md)、[12_LINE2_EVIDENCE_CLOSEOUT_20261025.md](roadmap/12_LINE2_EVIDENCE_CLOSEOUT_20261025.md) | 机器读数与结案回执；不改写为能力通过 |
| 索引 | [PLAN_INDEX.md](PLAN_INDEX.md)（本页）、[plans/README.md](../README.md) | 只导航 |

## 当前合同（位于 reference，不在 active）

| 合同 | 文档 | 状态 |
|---|---|---|
| 重构方案 | [PLAN-NATIVE-CONTENT-CORE_REDESIGN_20261030.md](../reference/PLAN-NATIVE-CONTENT-CORE_REDESIGN_20261030.md) | 现行主方案 |
| R1 语义 | [SPEC-NATIVE-CONTENT-CORE_R1_SEMANTIC_CONTRACT_20261030.md](../reference/SPEC-NATIVE-CONTENT-CORE_R1_SEMANTIC_CONTRACT_20261030.md) | 设计级闭合 |
| R2 基线 | [SPEC-NATIVE-CONTENT-CORE_R2_CANDIDATE_COMPARISON_20261030.md](../reference/SPEC-NATIVE-CONTENT-CORE_R2_CANDIDATE_COMPARISON_20261030.md) | 待用户评审 |
| R3 实现 | [SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md](../reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md) | 结构门通过，未训练 |
| R3b-1 修正旧图基线 | [R3 记录 §6](../reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md#6-r3b1-修正旧图基线20261030) | 22/22 结构门；仅 value bias 可训练；未训练 |
| R3b-2 普通端到端基线 | [R3 记录 §7](../reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md#7-r3b2-普通端到端基线20261030) | 11/11 结构门；全参数可训练；未训练 |

## 旧资产

N177/F/N178、VISION、T2/T3、M0–M8、R2 日对话、UI 生命页等旧文档全部登记在[LEGACY_ASSET_INDEX](../reference/LEGACY_ASSET_INDEX.md)。它们只作证据、基线或长期储备，其“当前/下一步”只解释写作时点。

## 维护规则

1. 新方向只能进[MAINLINE](../MAINLINE.md)第 3 节，不新增并列主线。
2. 只有 03 写“当前唯一下一步”。
3. active 文档前 10 行必须有 `> 文档状态：[...]`。
4. 本页必须覆盖 `plans/active/` 下全部 `.md`；由单主线审计脚本检查。
5. 旧文不因收敛删除或改名；只降级调度权。
