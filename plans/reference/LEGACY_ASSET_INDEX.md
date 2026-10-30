# 旧资产总索引

> 文档状态：[历史] — 本页只登记旧方向、旧实验和冻结合同的处置，不发布当前指令。当前方向见[MAINLINE](../MAINLINE.md)，队首见[03](../active/roadmap/03_CURRENT_EXECUTION.md)。

## 1. 判定规则

一份文档属于旧资产，当且仅当它**不在**[reference/README §1 合同映射](README.md#1-总计划步骤对应的原合同)的表中。旧资产可以作证据、基线、组件或长期储备，但不能：

- 被当前工作包引用为“下一步”；
- 作为第二套主线或第二套能力验收；
- 因文件名含 `PLAN`/`SPEC`/`NEXT` 就自动获得调度权。

旧文的“当前/下一步/唯一动作”只解释其写作时点。

## 2. 资产族与处置

| 族 | 范围/代表入口 | 处置 | 可复用部分 |
|---|---|---|---|
| N120–N174 | `PLAN-N1-17x_*`、`SPEC-N1-17x_*`、`*_execution_closeout*` | 历史实验 | 数据构造、恢复/计时仪器、失败原因 |
| N175 | [评价交接](PLAN-N1-175_evaluation_handoff_20261026.md)、[盲评](PLAN-N1-175_blind_review_20261026.md)、[原因复盘](PLAN-N1-175_prior_results_cause_review_20261026.md) | 冻结/未判面保留 | R 0/3、S 6/312 的边界；不代判、不重生成 |
| N176 | [课程计划](PLAN-N1-176_foundation_curriculum_20261026.md)、[F0/F1 合同](SPEC-N1-176_F0_F1_pilot_20261026.md)、[架构主张审计](PLAN-N1-176_architecture_claims_audit_20261026.md) | 冻结资产/证据 | 已生成课程与消费登记；不重生成 |
| N177 A–F | [实施计划](PLAN-N1-177_joint_cognition_slice_20261026.md)、[177.5 T1](SPEC-N1-177_5_T1_F0_F1_pilot_20261027.md) | 旧基线来源 | 接口、来源、恢复、失败模式 |
| N177 E1 | [E1 消费合同](SPEC-N1-177_E1_state_consumption_contract_20261029.md)、[E 决策](PLAN-N1-177_E_state_consumption_and_credit_decision_20261029.md) | 证据/历史设计 | 状态依赖任务、固定分母、cut/rescue 语义 |
| N177 F | [F 编码器](SPEC-N1-177_F_value_encoder_20261029.md)、[F1 冻结提案](SPEC-N1-177_F1_freeze_proposal_20261029.md) | 负结果/旧基线 | `corrected_old_v1` 的字节编码与失败对照 |
| N178 | [解析覆盖提案](SPEC-N1-178_parse_coverage_extension_proposal_20261030.md) | 诊断 | 解析缺口解释；不自动扩量 |
| R2 日对话/CC-1/extension | `PLAN-R2-*`、[工程计划](delivery-runs/20261021-n-mainline/artifacts/02-plan.md)、[过渡矩阵](delivery-runs/20261021-n-mainline/artifacts/17-plan-transition-and-reconciliation.md) | 冻结工程 | 恢复、评测适配、来源纪律 |
| VISION/VX | [VISION](VISION_FUTURE_TECHNOLOGY.md)、[持续发展 V2](../active/architecture/TAIJI_CONTINUAL_DEVELOPMENT_V2.md) | 长期储备 | 42 条设计条款作输入；不产生当前工单 |
| M0–M8/批准 | [M0–M7 收束](M0_M7_PROJECT_CONSOLIDATION_20261006.md)、[M5 批准](M5_EXIT_APPROVAL_20260920.md)、[M6 排除](M6_SCOPE_EXCLUSION_APPROVAL_20261006.md) | 冻结批准 | 原范围、排除、重启条件 |
| UI/产品面 | [UX 规范](UI_LIFE_UX_SPEC_20261003.md)、[视觉方向](UI_LIFE_VISUAL_DIRECTION_20261003.md) | 发布面 | 生命页字段与 VX6 子包 |
| 09 机制词典 | [09 N1–N6 映射](../active/roadmap/09_NEXT_MAINLINE_PLAN.md) | 组件参考 | 机制责任/历史映射；不是当前主计算图 |
| 01 阶段地图 | [01](../active/roadmap/01_SCOPE_AND_PHASES.md) | 历史 | M0–M8 与 N 层次追溯 |

## 3. 唯一复入口

旧资产若要重新成为当前主线，必须：

1. 在[MAINLINE](../MAINLINE.md)第 3 节登记为“候选阶段”；
2. 写一份新的当前合同，明确任务、数据、基线、预算、失败出口；
3. 通过 R4 定价与 R5 比较，或由用户明确改序。

不允许通过新增一份 `PLAN-NEXT-*` 文档自动复辟。

## 4. 保留纪律

- 不删除、不改名、不重写冻结件；路径与 Git 身份是证据。
- 旧报告的数字不并入新候选成绩；同一 bundle 才能拼能力。
- 旧 sealed 面不因新主线重读；新面另立合同。
- 旧“下一步”不再调度；当前实际动作在MAINLINE的执行位置，03只提供跳转。

本轮已实际迁移/删除的旧交接、根目录CI材料及恢复身份见[2026-10-30整理索引](../archive/history/n_mainline_20261030/README.md)。被runtime/CI/测试直接读取的旧文档与源文件保留原位，不凭旧资产分类删除。
