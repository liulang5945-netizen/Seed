# Seed / Taiji 计划与架构入口

正式持续Goal：**沿N主线，把态极发展成由自身认知机制驱动、能通过自然中文可靠完成连续任务、跨会话积累和修正经验、从反馈中学会新能力并保持旧能力的原生认知模型；以同一可恢复版本的未见表现、旧能力保持和实际成本验收每轮进步。** Seed负责产品与运行承载。

## 当前路线与文档职责

2026-10-29进入E决策：N177.6的15臂与数值评价完成，D已发布；B实际链/隔离和终点恢复回执的余项转E0补证。采用[状态因果消费优先的E方案](reference/PLAN-N1-177_E_state_consumption_and_credit_decision_20261029.md)：E0补证与E1新端点/消费合同→E2任务信用on/off同图候选→E3零步验收→F独立预算及有界验证。旧切片正向收益保留；新初始化复验与旧题直接接梯度暂不排首位。当前状态见[03](active/roadmap/03_CURRENT_EXECUTION.md)，工作包见[10](active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)。

| 要查什么 | 主要入口 |
|---|---|
| 当前队首、运行/预算事实及停止/恢复 | [03当前执行](active/roadmap/03_CURRENT_EXECUTION.md) |
| 详细分阶段工作包、依赖、产物、证据、失败回流和决策 | [10持续能力执行计划](active/roadmap/10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md) |
| 前序语言负结果的原因、初始化与评价边界 | [原因复盘](reference/PLAN-N1-175_prior_results_cause_review_20261026.md) |
| CC-2早期诊断及模型路线历史 | [N1-167](reference/PLAN-N1-167_execution_closeout_20261025.md)、[N1-168](reference/PLAN-N1-168_execution_closeout_20261025.md)、[N1-169候选设计](reference/PLAN-N1-169_shared_language_chain_design_20261025.md) |
| 旧计划继承/合并/退役/历史/条件保留 | [17过渡与收束](reference/delivery-runs/20261021-n-mainline/artifacts/17-plan-transition-and-reconciliation.md) |
| N1–N6及42条设计责任 | [09主线设计](active/roadmap/09_NEXT_MAINLINE_PLAN.md) |
| 根需求、认知所有权、现有架构 | [核心需求](active/TAIJI_CORE_REQUIREMENTS.md)、[原生架构](active/TAIJI_NATIVE_ARCHITECTURE_V1.md) |
| 下一主线统一架构、驱动力及VX开发包 | [VISION现行方案](reference/VISION_FUTURE_TECHNOLOGY.md) |
| 生命页重设计、字段来源及VX6子包 | [生命页实施设计](reference/UI_LIFE_UX_SPEC_20261003.md)、[视觉方向](reference/UI_LIFE_VISUAL_DIRECTION_20261003.md) |
| M0–M8资产与范围排除 | [01阶段地图](active/roadmap/01_SCOPE_AND_PHASES.md)、[既有项目收束](reference/M0_M7_PROJECT_CONSOLIDATION_20261006.md) |
| 能力、用户验收与采用/发布 | [07整模型交付](active/roadmap/07_MINI_MODEL_DELIVERY.md)、[02门禁/CI](active/roadmap/02_GATES_AND_CI.md) |
| 历史CC-1实施交接与R2冻结技术约束 | [工程交接](reference/delivery-runs/20261021-n-mainline/artifacts/02-plan.md)、[历史技术摘要](reference/delivery-runs/20261021-n-mainline/artifacts/02-continuation-extension-plan.md) |
| 活动文件索引、交付记录 | [PLAN_INDEX](active/PLAN_INDEX.md)、[黑板](reference/delivery-runs/20261021-n-mainline/board.md) |

已有实施资产与后继衔接见[N177合同](reference/PLAN-N1-177_joint_cognition_slice_20261026.md)；已训资产收束仍按[N176冻结合同](reference/SPEC-N1-176_F0_F1_pilot_20261026.md)，N175原合同与未完成S留原位，不混用资格。

## plans的分层与维护入口

按“目标约束→当前执行→工作包→合同→证据/历史”查找；沿既有目录细化职责，冻结原件保持路径与身份。

| 层次 | 位置/入口 | 保存内容 | 更新触发 |
|---|---|---|---|
| 项目约束 | active核心需求/架构、roadmap的01/02/07/09 | 目标、机制、范围与能力规则 | 真实设计或采用范围变化 |
| 当前执行 | active/roadmap/03 | 主实验、阶段状态、臂矩阵、资源快照 | durable/运行/停止状态变化 |
| 工作包 | active/roadmap/10 | 父子关系、依赖、产物、验收与决策卡 | 阶段交接或方案决策 |
| 合同与交接 | reference的PLAN/SPEC/APPROVAL；[分类索引](reference/README.md) | PLAN实施衔接；SPEC原冻结；批准范围 | PLAN按实际推进；冻结件按版本纪律 |
| 身份与证据 | manifests、仓根reports | 候选/数据身份与机器原始读数 | 有实际来源的报告产出 |
| 历史 | archive、原结案件及Git恢复身份 | 旧路线、失败、审批与过程快照 | 有明确恢复索引的归档/收束 |

### 活动计划表格约定

| 表类型 | 推荐字段 | 使用原则 |
|---|---|---|
| 阶段总表 | 阶段、状态、前置、产物、转交 | 每行一个可交付阶段；控制在五列，细节下沉卡片 |
| 工作包卡片 | 字段、设计 | 分别写输入/来源、动作/交付、验收、失败出口与权限；不把几百字塞进总表 |
| 实验进度 | 模式、各seed耐久步 | 只在03维护；正式阈值取SPEC，其他计划引用03 |
| 证据/决策 | 主张、适用范围、证据/裁决 | 事实、假说、候选方案及能力通过分开 |

状态统一使用“进行中、可并行准备、待前置、待裁决、条件触发、已完成、暂停、已结案”。“可并行准备”不是已实施完成；并行项写清资源/源码/writer边界。总表引用工作包，工作包引用SPEC/报告；进度与数值门各有一个维护处。当前主线先按此规范整理，历史结案及冻结表格保留原义与身份，不为排版重新冻结。

## 继承与纪律

- M5批准退出、M6限定收官和成长/发布/设备等排除继续按原批准范围继承，不宣称原生对话或L3/L4已通过。N1-166与原封存面边界保持；W5b恢复v2按原结案继承，不重新当未来战役。
- 训练先验证checkpoint保存/独立恢复；未变合同内已授权工作自动推进，实质预算/题面/冻结协议变动到10对应D节点裁决。
- 成功、负结果、partial和不可判分别记账；同一bundle的真实能力是验收对象，不用多个研究runner成绩拼出一个模型。
- 旧方向退出活动指令与队首，冻结合同、失败、原始输出与谱系保留。目录与产物按[目录规范](../docs/FOLDER_STRUCTURE_RULES.md)管理。

## 历史

2026-10-23已执行[全项目历史整理](reference/PROJECT_HISTORY_CONSOLIDATION_20261023.md)：删除旧测试/缓存与6份整篇重复快照，08保留重放手册及登记标题，reference改为主题索引；完整旧文按Git精确恢复，不再重复存放。

16保留[目标提出时的评审](reference/delivery-runs/20261021-n-mainline/artifacts/16-project-review-and-continuous-objective.md)，17登记本轮正式衔接。更早逐轮状态见[20261022索引快照](archive/history/PLANS_INDEX_BEFORE_N_CONSOLIDATION_20261022.md)与[旧执行快照](archive/history/N_MAINLINE_EXECUTION_BEFORE_CONSOLIDATION_20261022.md)，本轮六份过渡前全文由17导航；旧“下一步”不再调度。
