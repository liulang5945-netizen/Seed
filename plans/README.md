# Seed / Taiji 计划与架构入口

> 更新：2026-09-16；R0证据与门禁审计已完成，R2结构化语言入口、保存恢复前置、paired诊断、P2序列级只读评价、G1条件接口 smoke、H2/H3内部表示/读出审计、H3.1序列路径对照、H3.2 response-start、H3.3泛化控制、H3.3-C response-phase候选、H3.4逐位置条件信用审计与H3.5目标/数据/表示合同复审均已完成；本地未提交观察单列。旧B0/WP-1待决策执行指令已失效。

## 项目位置

当前为M5知识与身体。K轴限定晋级；B2-v4在冻结create族内支持协作，但不等于轴独立晋级或产品采用。在线回写仍失败，且[R0审计](reference/M5_R0_EVIDENCE_GATE_AUDIT_20260916.md)确认v2存在门禁假阳性与不可判门。整模型语言能力未达到用户验收线。当前先推进R2目标对齐训练、原生语言读出和持续适应架构；结构化episode入口、checkpoint前置、paired诊断、static/slow/fast/fast_slow对照、P2序列级只读评价、G1策略显式化 smoke、H2/H3状态审计、H3.1 beam、H3.2 response-start、H3.3-A/B/C泛化与候选owner、H3.4逐位置条件信用审计均已落地。预算一致的H3.3-B/C均为总active 273,890≤300k、preflight/恢复通过；H3.4显示UTF-8合法率、无替换率和end-marker边界率均为1.0，但train/dev/final exact与sequence仍为0，dev/final首字节与未见continuation不可迁移。H3.5据此选定`prefix encoder -> response_plan_state -> plan-conditioned byte renderer`的分层原生方案；H3.5-A已冻结32维plan、span监督编码、300k预算、oracle禁令、preflight和消融。当前唯一下一步是隔离candidate实现与plumbing smoke。Mini模型验收后置，P5.2d修正仪器降为并行债务。

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
| [07 整模型验收](active/roadmap/07_MINI_MODEL_DELIVERY.md) | 真实输出评价、L3触发最小用户版本 |
| [05 技术债](active/roadmap/05_TECH_DEBT_REGISTER.md) | 隔离、仪器、恢复与CI；最新日期优先 |
| [唯一完整VISION](reference/VISION_FUTURE_TECHNOLOGY.md) | 成熟发展训练＋运行期适应候选架构，不自动改变主线合同 |
| [06 历史决策](active/roadmap/06_P5_2C_PRIME_NEXT_STEP_DECISION.md) | 旧C/A/B路线依据，不是当前执行授权 |

## 推进原则

1. R0已统一P3b终态与在线复测门禁；当前先推进R2条件回答主线和可证伪readout/序列路径，不把局部代理分当作能力。
2. P5.2d修正仪器和Mini模型验收均不阻塞当前R2实现；在线适应满足隔离、可信底座和R2能力依赖后推进，Mini验收在主线bundle形成后统一执行。
3. 基本对话、同bundle代表能力、安全/恢复/性能达到L3才交用户自由验收；不立即赶演示，也不等待完整VISION/M8。
4. M5退出仍须知识、协作/选择、在线/身体与共同门齐备；排除项显式批准。
5. 历史失败保留，新结果不追改旧阈值。新训练、核心合同、默认行为与资源投入需相应批准；训练前必须先通过checkpoint保存/恢复前置检查。

9月16日查询的最新远端CI run34869725409总体失败（3.10腿），对应d09dcc21而非当前HEAD。局部通过不等于远端全绿。
