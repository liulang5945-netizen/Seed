# Seed / Taiji 计划与架构入口

> 更新：2026-09-18。唯一VISION §19给出项目缺口实际方案，§20补齐多模态共享认知，§21补齐自主学习的信息准入、抗污染和恢复；本轮§22继续补齐不确定性、信念修订、有限记忆、技能/调度、版本一致性及长期评价，根需求/架构/01/02/07同步设计与验收入口。推荐设计不等于架构采用或新运行授权，局部实验仍按02 §2.5有条件准入；最新研究状态与唯一下一步只看03及有效合同/裁决。

## 项目位置

先读[当前结果与结论总览](reference/PROJECT_RESULTS_AND_CONCLUSIONS_20260917.md)：按正向成果、负结果、不可判结果、真实输出和结论边界整理；运行收束状态见[收束记录](reference/PROJECT_CONSOLIDATION_20260917.md)。

当前仍为M5/R2原生语言。D1测量底座可继承；本次续写核对的[D8/v3 matched报告](../reports/r2_d8_matched_dev_20260918.json)多字值完整率均值0.3621达到K1，但M4/flip为0，整体outcome仍failed、未准晋级，不能抹去复制正信号或写成完整语言通过。最新逐包裁决见[03](active/roadmap/03_CURRENT_EXECUTION.md)及对应有效合同；若摘要与新报告不同步，先核对身份/裁决，不用旧“待启动”覆盖已发生结果。首页不另设活动队首；旧672结果仍按[历史复核](reference/M5_R2_DESIGN_EVIDENCE_AUDIT_20260917.md)保留。

当前核心缺口是同任务中的共同表示、记忆消费、世界预测、规划选择与学习信用未形成统一能力证据。原P3b的数据分布效应只有一个共同检查点，不能判定；P5.2d后继报告虽有仪器修正，A1真实收益仍未闭合，不以预算放宽或汇总布尔代替。

## 怎样用这套计划指导开发

| 你要回答的问题 | 主要入口 | 应得到的指导 |
|---|---|---|
| 项目整体还缺什么，何时晋级？ | [01 §2–6](active/roadmap/01_SCOPE_AND_PHASES.md) | M0–M8状态、能力缺口、依赖及交付，未完成的长期目标不被局部研究隐藏 |
| 下一段开发具体要做什么？ | [03 §5](active/roadmap/03_CURRENT_EXECUTION.md)、[01 §6.4–6.5](active/roadmap/01_SCOPE_AND_PHASES.md) | 活动包与局部准入看03；转段条件及数据/模型/集成/评价交付看01，不把整条路线当一次授权 |
| 现有设计受限时有哪些高上限选择？ | [VISION §15–19](reference/VISION_FUTURE_TECHNOLOGY.md) | 五类备选及推荐双通道组合；具体计算/训练、取舍与否决，不能按模块名称或实现难度排名 |
| 七项项目缺口怎样实际解决？ | [VISION §19](reference/VISION_FUTURE_TECHNOLOGY.md#19-项目缺口的实际解决方案推荐核心学习配方演进与交付) | 核心/数据/学习/反馈/消费者/交付逐项映射，依赖与当前R2优先项明确，不自动开多条主线 |
| 多模态怎样形成同一个模型？ | [VISION §20](reference/VISION_FUTURE_TECHNOLOGY.md#20-多模态共同认知专用编码共享事件跨模态学习与验收) | 专用编码、共享对象/事件、区域/时间引用、对齐课程与跨模态干预；不以OCR/ASR或答案拼接代替理解 |
| 自主学习遇到恶意信息怎么办？ | [VISION §21](reference/VISION_FUTURE_TECHNOLOGY.md#21-自主学习的信息治理不可信输入分层准入抗污染与恢复)、[02 §2.6](active/roadmap/02_GATES_AND_CI.md) | 观察/相信/学习/行动分权，来源继承、候选隔离、快慢准入、污染撤销与回退；不承诺过滤绝对安全 |
| 长期运行还有哪些机制缺口？ | [VISION §22](reference/VISION_FUTURE_TECHNOLOGY.md#22-长期运行的补全设计不确定性信念修订记忆预算技能与持续评价) | 何时求证、冲突/历史信念、有限记忆、技能与目标、私有状态隔离、自主调度和更新/恢复；同一压力序列检验共同收益 |
| 怎样证明越用越会，而不是测试时偷学答案？ | [07 §4.4](active/roadmap/07_MINI_MODEL_DELIVERY.md#44-冻结能力与在线学习分账2026-09-18设计补充)、[VISION §22.8](reference/VISION_FUTURE_TECHNOLOGY.md#228-长期评价冻结能力和在线学习必须分账) | 固定能力测试保持冻结；在线流先记录预测再释放合法反馈，只评价后续适应，另测旧能力与资源成本 |
| 什么算完成，何时给用户模型？ | [02 §2–3](active/roadmap/02_GATES_AND_CI.md)、[07](active/roadmap/07_MINI_MODEL_DELIVERY.md) | 规划/实验/能力/阶段/采用分别记账，L3触发后置用户验收 |
| 如何从研究runner接成一个模型？ | [07 §5.2](active/roadmap/07_MINI_MODEL_DELIVERY.md)、[VISION §17.7](reference/VISION_FUTURE_TECHNOLOGY.md) | 权重/编码身份、会话映射、真实消费者、恢复与同bundle评价；候选接入不等于默认采用 |

先看目标与缺口，再确定交付和选择设计，不从算法名字倒推项目必须做什么。本轮仅完善这些规划；项目研发的当前范围和停止条件沿用03与冻结合同，不因文档更新撤销旧成果或自动放行新训练。

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
| [唯一完整VISION](reference/VISION_FUTURE_TECHNOLOGY.md) | §15–18架构选择与开发；§19实际方案；§20多模态；§21信息治理；§22长期认知、资源/版本与持续评价 |
| [06 历史决策](active/roadmap/06_P5_2C_PRIME_NEXT_STEP_DECISION.md) | 旧C/A/B路线依据，不是当前执行授权 |
| [旧执行计划快照](archive/history/EXECUTION_BEFORE_RULE_REPAIR_20260917.md) | 原逐轮台账与当时判断，保留追溯；全部旧“下一步”失效 |
| [R2逐轮诊断快照](archive/history/R2_DIAGNOSTIC_QUEUE_BEFORE_DESIGN_OPTIONS_20260917.md) | 七次后继追加的旧队首与当时解释；当前结论由设计依据复核限定 |

## 推进原则

统一能力主张以[认知闭环验收原则](active/roadmap/02_GATES_AND_CI.md#21-统一认知闭环验收原则2026-09-17用户确认)为准：同一模型、同一真实任务，记忆/世界模型/规划的联合行为贡献；原生语言与行动能力分账，不以模块或界面代替。本条不覆盖具体包的执行/停止状态。

1. 本轮只完善开发指导，仓库研究主线保持R2，最新阶段裁决与唯一入口见03。必要局部实验按[02 §2.5](active/roadmap/02_GATES_AND_CI.md)有条件准入：明确判别价值、对照、累计预算及停止/返回动作；已授权且未触发停止线可按约定推进，不逐次全架构审批。当前已停止配置不会因本轮规则修订自动重开，A–E也不依次试遍。
2. 最小任务族属于方法而非主线：直接服务R2判别的小型文本实验可在批准包内使用；需要新Workbench环境或新增行动/适应能力的方案保留VISION，未批准不执行、不成为R2前置。讨论设计不自动改变执行顺序。
3. 基本对话、同bundle代表能力、安全/恢复/性能达到L3才交用户自由验收；不立即赶演示，也不等待完整VISION/M8。
4. M5退出仍须知识、协作/选择、在线/身体与共同门齐备；排除项显式批准。
5. 历史失败保留，新结果不追改旧阈值。新训练、核心合同、默认行为与资源投入需相应批准；训练前必须先通过checkpoint保存/恢复前置检查。
6. 不可妥协边界、能力合同、工程合同、实验选择分层管理；规则写明目的、范围、实现状态和验证方式。允许版本化替换方法，不事后放松确认门，不把设计文本当已实现能力。

9月16日查询的最新远端CI run34869725409总体失败（3.10腿），对应d09dcc21而非当前HEAD。局部通过不等于远端全绿。
