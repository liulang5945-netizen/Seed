# 研究合同、裁决与历史证据索引

> 文档状态：[索引] — 本页只区分“当前合同”与“旧资产”。方向见[唯一主线](../MAINLINE.md)，队首见[03](../active/roadmap/03_CURRENT_EXECUTION.md)。

## 1. 当前合同（唯一可被当前队列引用）

| 文档 | 状态 | 维护职责 |
|---|---|---|
| [重构方案](PLAN-NATIVE-CONTENT-CORE_REDESIGN_20261030.md) | 现行主方案 | 共同内容主图、动态读取/生成、三类基线、有限退出 |
| [R1 统一语义合同](SPEC-NATIVE-CONTENT-CORE_R1_SEMANTIC_CONTRACT_20261030.md) | 设计级闭合 | 任务/时间/byte目标/固定分母/cut-rescue/来源/fixture |
| [R2 候选与基线合同](SPEC-NATIVE-CONTENT-CORE_R2_CANDIDATE_COMPARISON_20261030.md) | 待用户评审 | 新核心/修正旧图/普通基线、Q1 与 Q2、公平性 |
| [R3 隔离实现与零步恢复](SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md) | 结构门通过，未训练 | 新核心模块、信用组、bundle 恢复、结构探针；§6 含 R3b-1 修正旧图基线 |
| [20261030 核心设计重审](TAIJI_CORE_DESIGN_REVIEW_20261030.md) | 设计依据 | 目标/底座/当前候选分层、旧实验混杂、重构建议 |
| [20261029 阶段收束](PROJECT_CONSOLIDATION_20261029.md) | 当前证据账 | E/F0a 资产、E3 子门、失败、CI/契约待办 |

论文、报告和机器读数的身份仍以 `plans/manifests/` 与 `reports/` 为准；合同不复制逐轮数字。

## 2. 旧资产（不是当前合同）

| 资产族 | 代表文档 | 处置 |
|---|---|---|
| N177 A–F / E1 / F0a-F1b | [N177 实施计划](PLAN-N1-177_joint_cognition_slice_20261026.md)、[E1 合同](SPEC-N1-177_E1_state_consumption_contract_20261029.md)、[F 编码器](SPEC-N1-177_F_value_encoder_20261029.md)、[F1 冻结](SPEC-N1-177_F1_freeze_proposal_20261029.md)、[177.5 T1](SPEC-N1-177_5_T1_F0_F1_pilot_20261027.md) | 旧基线与证据；只用于 `corrected_old_v1`，不续推 |
| N178 parser | [解析覆盖提案](SPEC-N1-178_parse_coverage_extension_proposal_20261030.md) | 诊断；不自动扩量 |
| N175/N176 | [N175 交接](PLAN-N1-175_evaluation_handoff_20261026.md)、[N176 课程](PLAN-N1-176_foundation_curriculum_20261026.md)、[N176 冻结合同](SPEC-N1-176_F0_F1_pilot_20261026.md) | 冻结/证据；不重判、不重生成 |
| N120–N174 家族 | 以 `PLAN-N1-*`、`SPEC-N1-*`、`*_closeout*` 检索 | 历史实验；结论按原范围保留 |
| R2 日对话 / CC-1 / extension | `PLAN-R2-*`、`delivery-runs/20261021-n-mainline/` | 冻结工程资产；重启需新合同 |
| VISION / VX0–VX7 | [VISION 长期方案](VISION_FUTURE_TECHNOLOGY.md) | 长期设计储备；不产生当前工单 |
| M0–M8 / 批准 / 墓碑 | [M0–M7 收束](M0_M7_PROJECT_CONSOLIDATION_20261006.md)、[M5 批准](M5_EXIT_APPROVAL_20260920.md)、[M6 排除](M6_SCOPE_EXCLUSION_APPROVAL_20261006.md) | 原批准范围与排除；不补发能力 |
| UI 生命页 | [UX 规范](UI_LIFE_UX_SPEC_20261003.md)、[视觉方向](UI_LIFE_VISUAL_DIRECTION_20261003.md) | 发布面；不属于研究主线 |
| 历史整理 | [整理与 Git 恢复](PROJECT_HISTORY_CONSOLIDATION_20261023.md)、[墓碑](DISTILLATION_TOMBSTONE_20261007.md) | 追溯身份；不重复全文 |

完整清单与统一规则见[LEGACY_ASSET_INDEX](LEGACY_ASSET_INDEX.md)。

## 3. 使用规则

- 当前工作包只能引用本页第 1 节的合同；引用第 2 节时只能作证据、基线或背景。
- 旧文档中的“当前/下一步/唯一动作”只解释其写作时点，不产生当前指令。
- 冻结 SPEC 与报告保持路径、字节和 Git 身份；本页不因收敛改写原规则。
- 新合同必须先在[MAINLINE](../MAINLINE.md)第 3 节登记阶段，才能成为“当前合同”。

## 4. 如何查具体实验

按 `PLAN-N1/N2/N5`、`SPEC`、`EXP`、`execution_closeout`、`APPROVAL` 命名族检索 `plans/reference`，再读对应合同/结案。机器结果看 `reports/README.md`；历史结构见 `plans/archive/README.md`。未完成条款保留在 03/10 或债表，不因整理改变状态。
