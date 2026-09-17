# Seed / Taiji 计划与架构入口

> 更新：2026-09-17；当前任务是完善、补全用于指导开发的计划，不是立即选择或实施架构。03 §5给出能力/工作/交付路线，01 §6给出全项目依赖，唯一VISION §15–17提供高上限方案与演进设计。A-R2仅为备选示例；研究仍暂停。

## 项目位置

先读[当前结果与结论总览](reference/PROJECT_RESULTS_AND_CONCLUSIONS_20260917.md)：按正向成果、负结果、不可判结果、真实输出和结论边界整理；运行收束状态见[收束记录](reference/PROJECT_CONSOLIDATION_20260917.md)。

当前仍为M5，工作主线为R2原生语言，研究执行暂停。H3.7/H3.8旧候选已结案；随后目标、容量、课程扫描与672档六seed巩固已完成。最新均值0.2242±0.1642是开发集exact及总体标准差，不是置信区间；固定回答“未知”在同集即可得0.3810，非零不等于内容能力成立。672行只有448个不同输入/答案对，完整逐题输出/检查点未由巩固脚本保存。详见[设计依据复核](reference/M5_R2_DESIGN_EVIDENCE_AUDIT_20260917.md)。没有阶段晋级或Mini交付，唯一顺序以03为准。

当前核心缺口是同任务中的共同表示、记忆消费、世界预测、规划选择与学习信用未形成统一能力证据。原P3b的数据分布效应只有一个共同检查点，不能判定；P5.2d后继报告虽有仪器修正，A1真实收益仍未闭合，不以预算放宽或汇总布尔代替。

## 怎样用这套计划指导开发

| 你要回答的问题 | 主要入口 | 应得到的指导 |
|---|---|---|
| 项目整体还缺什么，何时晋级？ | [01 §2–6](active/roadmap/01_SCOPE_AND_PHASES.md) | M0–M8状态、能力缺口、依赖及交付，未完成的长期目标不被局部研究隐藏 |
| 下一段开发具体要做什么？ | [03 §5](active/roadmap/03_CURRENT_EXECUTION.md) | R2能力覆盖、开发顺序、工作包模板、结果处置；不是预选A的开工指令 |
| 现有设计受限时有哪些高上限选择？ | [VISION §15–17](reference/VISION_FUTURE_TECHNOLOGY.md) | 不同核心方案的计算/学习机制、适用证据、代价、替代与组合演进 |
| 什么算完成，何时给用户模型？ | [02 §2–3](active/roadmap/02_GATES_AND_CI.md)、[07](active/roadmap/07_MINI_MODEL_DELIVERY.md) | 规划/实验/能力/阶段/采用分别记账，L3触发后置用户验收 |

先看目标与缺口，再确定交付和选择设计，不从算法名字倒推项目必须做什么。当前仅完善这些规划；恢复开发时将选定范围展开为一个工作包，不把整份路线当作一次授权。

## 文档职责与证据入口

| 文档 | 职责 |
|---|---|
| [根需求](active/TAIJI_CORE_REQUIREMENTS.md) | CR-1–CR-10长期目标；成熟能力与原型、方法选择分开 |
| [架构与规则](active/TAIJI_NATIVE_ARCHITECTURE_V1.md) | §0.2四层规则；§4.6–4.7统一任务/行动合同；§6.3–6.4学习与恢复；§8.5规则实施清单 |
| [01 总阶段地图](active/roadmap/01_SCOPE_AND_PHASES.md) | M0–M8、长期目标与能力轴缺口 |
| [02 晋级与发布](active/roadmap/02_GATES_AND_CI.md) | 实验、轴晋级、阶段退出、发布各自门槛 |
| [03 详细推进方案](active/roadmap/03_CURRENT_EXECUTION.md) | 当前规划模式、R2能力路径、开发包输入/交付/验证/退出，以及唯一活动入口 |
| [R2设计依据复核](reference/M5_R2_DESIGN_EVIDENCE_AUDIT_20260917.md) | 课程与六seed结果、固定回答/重复样本风险、归因收紧与证据缺口 |
| [R0证据与门禁审计](reference/M5_R0_EVIDENCE_GATE_AUDIT_20260916.md) | P3b终态、P5.2d v2逐门可用性与下一版仪器修正边界 |
| [R2语言目标与信用分配设计](reference/M5_R2_LANGUAGE_TARGET_CREDIT_DESIGN_20260916.md) | 历史语言目标与信用假设、实现来源；不覆盖后继证据或03执行裁决 |
| [R2-H3.5分层回答计划合同](reference/M5_R2_H3_5_HIERARCHICAL_RESPONSE_PLAN_CONTRACT_20260916.md) | 回答计划与byte渲染分层、v3数据课程、预算、恢复、消融和最小可证伪对照 |
| [R2-H3.5-A候选预注册](reference/M5_R2_H3_5A_RESPONSE_PLAN_PREREGISTRATION_20260916.md) | 32维plan、监督编码、运行时oracle禁令、checkpoint前置和smoke出口 |
| [R2-H3.5-A matched预注册](reference/M5_R2_H3_5A_MATCHED_RUN_PREREGISTRATION_20260916.md) | v3控制集、三seed对照、plan消融、final延迟权限和停止门 |
| [07 整模型验收](active/roadmap/07_MINI_MODEL_DELIVERY.md) | 真实输出评价、L3触发最小用户版本 |
| [05 技术债](active/roadmap/05_TECH_DEBT_REGISTER.md) | 隔离、仪器、恢复与CI；按证据范围及明确后继修订解释 |
| [唯一完整VISION](reference/VISION_FUTURE_TECHNOLOGY.md) | §15多架构设计；§16选择/迁移与A-R2示例；§17共用开发主干、数据训练与演进依赖 |
| [06 历史决策](active/roadmap/06_P5_2C_PRIME_NEXT_STEP_DECISION.md) | 旧C/A/B路线依据，不是当前执行授权 |
| [旧执行计划快照](archive/history/EXECUTION_BEFORE_RULE_REPAIR_20260917.md) | 原逐轮台账与当时判断，保留追溯；全部旧“下一步”失效 |
| [R2逐轮诊断快照](archive/history/R2_DIAGNOSTIC_QUEUE_BEFORE_DESIGN_OPTIONS_20260917.md) | 七次后继追加的旧队首与当时解释；当前结论由设计依据复核限定 |

## 推进原则

统一能力主张以[认知闭环验收原则](active/roadmap/02_GATES_AND_CI.md#21-统一认知闭环验收原则2026-09-17用户确认)为准：同一模型、同一真实任务，记忆/世界模型/规划的联合行为贡献；原生语言与行动能力分账，不以模块或界面代替。本条不改变当前收束暂停状态。

1. 当前只完善开发指导计划，研究主线保持R2且执行暂停。唯一后续建议是审阅03 §5与01 §6的阶段覆盖、依赖和交付安排；不是要求现在批准A-R2或启动训练。A–E保留为设计储备，不依次试遍，不重开P3b结案或续训已停止配置。
2. 最小任务族属于方法而非主线：直接服务R2判别的小型文本实验可在批准包内使用；需要新Workbench环境或新增行动/适应能力的方案保留VISION，未批准不执行、不成为R2前置。讨论设计不自动改变执行顺序。
3. 基本对话、同bundle代表能力、安全/恢复/性能达到L3才交用户自由验收；不立即赶演示，也不等待完整VISION/M8。
4. M5退出仍须知识、协作/选择、在线/身体与共同门齐备；排除项显式批准。
5. 历史失败保留，新结果不追改旧阈值。新训练、核心合同、默认行为与资源投入需相应批准；训练前必须先通过checkpoint保存/恢复前置检查。
6. 不可妥协边界、能力合同、工程合同、实验选择分层管理；规则写明目的、范围、实现状态和验证方式。允许版本化替换方法，不事后放松确认门，不把设计文本当已实现能力。

9月16日查询的最新远端CI run34869725409总体失败（3.10腿），对应d09dcc21而非当前HEAD。局部通过不等于远端全绿。
