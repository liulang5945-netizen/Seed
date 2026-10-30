# 研究合同、裁决与历史证据索引

> 文档状态：[索引] — 本页只导航合同、设计依据和旧资产。按序推进只读[MAINLINE 总计划](../MAINLINE.md)，从其[执行位置](../MAINLINE.md#执行位置)进入。

## 1. 总计划步骤对应的原合同

R 编号保留原合同与实验身份；实施统一使用总计划步骤 01–13。本页不复制阶段状态、审批或队首，实际进度与准入见[总计划](../MAINLINE.md#3-唯一执行队列)，原始批准与读数见[03 台账](../active/roadmap/03_CURRENT_EXECUTION.md)。

| 总计划步骤 | 原合同 | 约定范围 |
|---|---|---|
| [01](../MAINLINE.md#3-唯一执行队列) | [R1 统一语义合同](SPEC-NATIVE-CONTENT-CORE_R1_SEMANTIC_CONTRACT_20261030.md) | 任务、时间、byte 目标、固定分母、cut/rescue、来源与 fixture |
| [02](../MAINLINE.md#3-唯一执行队列) | [R2 候选与基线合同](SPEC-NATIVE-CONTENT-CORE_R2_CANDIDATE_COMPARISON_20261030.md) | 新核心、修正旧图、普通基线，Q1 与 Q2、公平性 |
| [03](../MAINLINE.md#3-唯一执行队列) | [R3 隔离实现与零步恢复](SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md) | 新核心模块、信用组、bundle 恢复与结构探针 |
| [04](../MAINLINE.md#3-唯一执行队列) | [R3b-1（R3 §6）](SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md#6-r3b1-修正旧图基线20261030) | 修正旧图基线 |
| [05](../MAINLINE.md#3-唯一执行队列) | [R3b-2（R3 §7）](SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md#7-r3b2-普通端到端基线20261030) | 普通端到端基线 |
| [06](../MAINLINE.md#3-唯一执行队列) | [R4 定价与 R5 数值门](SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md) | 课程准备、成本、预算、行为/机制/保持/恢复/成本门 |
| [07](../MAINLINE.md#3-唯一执行队列) | [R5 执行记录（R4 §9）](SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md#9-r5-执行结果20261030) | 有界比较、评价与原结案 |
| [08](../MAINLINE.md#3-唯一执行队列) | [R5b 复制轨候选合同](SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md) | 复制轨、自前缀日程、基线复用与有限出口 |
| [09](../MAINLINE.md#3-唯一执行队列) | T2 合同按对应准入建立 | 经历选择 |
| [10](../MAINLINE.md#3-唯一执行队列) | T3 合同按任务需要建立 | 自然目标、有限行动与后果 |
| [11](../MAINLINE.md#3-唯一执行队列) | 自然连续任务合同按对应准入建立 | 自然中文连续任务与 L3 |
| [12](../MAINLINE.md#3-唯一执行队列) | 跨会话合同按对应准入建立 | 反馈学习、旧能力保持与 L4 |
| [13](../MAINLINE.md#3-唯一执行队列) | 自主扩域合同按对应准入建立 | 有边界主动选择、条件扩域与 L5 |

设计依据：[重构设计方案](PLAN-NATIVE-CONTENT-CORE_REDESIGN_20261030.md)、[20261030 核心设计重审](TAIJI_CORE_DESIGN_REVIEW_20261030.md)。它们解释设计选择，不另排实施队列。历史证据账见[20261029 阶段收束](PROJECT_CONSOLIDATION_20261029.md)。

论文、报告和机器读数的身份仍以 `plans/manifests/` 与 `reports/` 为准。课程与分区入口：[split](../manifests/native_core_r1_split_v1.json)、[train](../manifests/native_core_r1_train_v1.json)、[query](../manifests/native_core_r1_query_v1.json)、[holdout](../manifests/native_core_r1_holdout_v1.json)、[审计报告](../../reports/taiji_native_core_r1_course_audit_20261030.json)；数据访问按对应冻结合同，不因索引可见而扩权。

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

- 当前工作包以总计划中链接的合同为准；引用本页第 2 节时只能作证据、基线或背景。
- 旧文档中的“当前/下一步/唯一动作”只解释其写作时点，不产生当前指令。
- 冻结 SPEC 与报告保持路径、字节和 Git 身份；本页不因收敛改写原规则。
- 新合同必须先在[MAINLINE](../MAINLINE.md)第 3 节登记阶段，才能成为“当前合同”。

## 4. 如何查具体实验

按 `PLAN-N1/N2/N5`、`SPEC`、`EXP`、`execution_closeout`、`APPROVAL` 命名族检索 `plans/reference`，再读对应合同/结案。机器结果看 `reports/README.md`；历史结构见 `plans/archive/README.md`。未完成步骤维护在总计划，证据和审批维护在 03，运营债务沿原债表；整理不改变原结论或授权。
