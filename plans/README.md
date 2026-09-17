# Seed / Taiji 计划与架构入口

> 更新：2026-09-16；R0证据与门禁审计已完成，R2结构化语言入口、保存恢复前置、paired诊断、P2序列级只读评价、G1条件接口 smoke、H2/H3内部表示/读出审计、H3.1序列路径对照、H3.2 response-start、H3.3泛化控制、H3.3-C response-phase候选、H3.4逐位置条件信用审计与H3.5目标/数据/表示合同复审均已完成；本地未提交观察单列。旧B0/WP-1待决策执行指令已失效。

## 项目位置

先读[当前结果与结论总览](reference/PROJECT_RESULTS_AND_CONCLUSIONS_20260917.md)：按正向成果、负结果、不可判结果、真实输出和结论边界整理；运行收束状态见[收束记录](reference/PROJECT_CONSOLIDATION_20260917.md)。

当前仍为M5知识与身体。2026-09-17按用户要求[阶段收束并暂停](reference/PROJECT_CONSOLIDATION_20260917.md)：H3.7B已结项，H3.8 v1/v2停止投入，P3b-v2中断未判定；没有阶段晋级或Mini交付。不执行下方历史文档的旧“下一步”，唯一执行状态以[03当前裁决](active/roadmap/03_CURRENT_EXECUTION.md)为准。

P3b两臂已停止，数据分布效应只有一个共同检查点，阶段结论应按“不可判”封存；64预算在线复测是未提交候选，不能继续仅写“等训练结束”或“放宽预算即可通过”。

## 阅读顺序

| 文档 | 职责 |
|---|---|
| [01 总阶段地图](active/roadmap/01_SCOPE_AND_PHASES.md) | M0–M8、长期目标与能力轴缺口 |
| [02 晋级与发布](active/roadmap/02_GATES_AND_CI.md) | 实验、轴晋级、阶段退出、发布各自门槛 |
| [03 详细推进方案](active/roadmap/03_CURRENT_EXECUTION.md) | 最新证据、R0–R6依赖、逐包交付/验收/停止、唯一下一步 |
| [R0证据与门禁审计](reference/M5_R0_EVIDENCE_GATE_AUDIT_20260916.md) | P3b终态、P5.2d v2逐门可用性与下一版仪器修正边界 |
| [R2语言目标与信用分配设计](reference/M5_R2_LANGUAGE_TARGET_CREDIT_DESIGN_20260916.md) | 从字节预测到可回答模型的根因、目标、信用分配、checkpoint前置、状态审计和晋级计划 |
| [R2-H3.5分层回答计划合同](reference/M5_R2_H3_5_HIERARCHICAL_RESPONSE_PLAN_CONTRACT_20260916.md) | 回答计划与byte渲染分层、v3数据课程、预算、恢复、消融和最小可证伪对照 |
| [R2-H3.5-A候选预注册](reference/M5_R2_H3_5A_RESPONSE_PLAN_PREREGISTRATION_20260916.md) | 32维plan、监督编码、运行时oracle禁令、checkpoint前置和smoke出口 |
| [R2-H3.5-A matched预注册](reference/M5_R2_H3_5A_MATCHED_RUN_PREREGISTRATION_20260916.md) | v3控制集、三seed对照、plan消融、final延迟权限和停止门 |
| [07 整模型验收](active/roadmap/07_MINI_MODEL_DELIVERY.md) | 真实输出评价、L3触发最小用户版本 |
| [05 技术债](active/roadmap/05_TECH_DEBT_REGISTER.md) | 隔离、仪器、恢复与CI；最新日期优先 |
| [唯一完整VISION](reference/VISION_FUTURE_TECHNOLOGY.md) | 成熟发展训练＋运行期适应候选架构，不自动改变主线合同 |
| [06 历史决策](active/roadmap/06_P5_2C_PRIME_NEXT_STEP_DECISION.md) | 旧C/A/B路线依据，不是当前执行授权 |

## 推进原则

统一能力主张以[认知闭环验收原则](active/roadmap/02_GATES_AND_CI.md#21-统一认知闭环验收原则2026-09-17用户确认)为准：同一模型、同一真实任务，记忆/世界模型/规划的联合行为贡献；原生语言与行动能力分账，不以模块或界面代替。本条不改变当前收束暂停状态。

1. R0已统一P3b终态与在线复测门禁；当前先推进R2条件回答主线和可证伪readout/序列路径，不把局部代理分当作能力。
2. P5.2d修正仪器和Mini模型验收均不阻塞当前R2实现；在线适应满足隔离、可信底座和R2能力依赖后推进，Mini验收在主线bundle形成后统一执行。
3. 基本对话、同bundle代表能力、安全/恢复/性能达到L3才交用户自由验收；不立即赶演示，也不等待完整VISION/M8。
4. M5退出仍须知识、协作/选择、在线/身体与共同门齐备；排除项显式批准。
5. 历史失败保留，新结果不追改旧阈值。新训练、核心合同、默认行为与资源投入需相应批准；训练前必须先通过checkpoint保存/恢复前置检查。

9月16日查询的最新远端CI run34869725409总体失败（3.10腿），对应d09dcc21而非当前HEAD。局部通过不等于远端全绿。
