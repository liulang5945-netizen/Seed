# N主线当前执行

> 文档状态：[状态] — 唯一队首登记处；方向由[MAINLINE](../../MAINLINE.md)定义。

更新：2026-10-30，Asia/Shanghai。03是当前主实验状态的唯一登记处；方向与队列只由[MAINLINE](../../MAINLINE.md)定义，[10](10_CONTINUOUS_CAPABILITY_EXECUTION_PLAN.md)负责阶段依赖与工作包。旧N177/F/N178只作基线/证据，不再负责实施衔接。进度数字只在本页登记，其他计划引用，不复制快照。

## 长期目标

持续Goal（2026-10-27修订）：沿N主线，发展由态极自身可学习、可恢复的认知机制参与决策与表达的原生认知模型，使其能够以自然中文可靠完成连续任务，跨会话积累和纠正经验，从可核验的经验与反馈中学习并保持旧能力；在授权和资源边界内，必要时即使没有新提示，也能根据目标与状态自主选择继续、求证、求助或等待。各阶段预先声明任务范围和收官条件，以身份可追溯的同一候选及其前后版本，对照合理基线，按范围验证未见任务表现、必要机制的实际贡献、反馈对负责模块的作用、旧能力保持、主动选择效果及真实资源成本；成功、负结果、不可判、工程失败或预算停止均按预定出口结案。

## 当前唯一下一步：执行 R5b（合同已批准 2026-10-30：实现→工程门→pilot 定价→9 run→评价→结案）

用户已采纳共同内容核心为唯一计划方向，现按[重构方案](../../reference/PLAN-NATIVE-CONTENT-CORE_REDESIGN_20261030.md)推进：语义合同→共同主图/基线→隔离实现/零步恢复→预算冻结→一次比较→条件能力接力。旧N177/F/N178结果与费用继续保留，已有授权N178b按原终点/预算结案；它们不再是主候选。R4 定价与 R5 比较均已执行；R5 以 `FAIL_BEHAVIOR` 结案，R6 未准入。R5b 复制轨候选合同已按 R4 §9.3 预定义的唯一后继起草：[SPEC](../../reference/SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md)——候选 `new_core_v2`＝复制轨（选中值原始字节进解码分布，通道不接触 gold）＋自前缀日程（257–512 更新以 0.5 概率换自身贪心前缀），针对 R5 归因的两处架构缺口；同数据/同门逐字沿用，基线复用 R5 端点不重训，预算 120,000 CPU 秒/48 小时；待用户批准（`--approve-r5b`）。

R1设计级已闭合。R2合同已评审：[候选与三类基线比较合同](../../reference/SPEC-NATIVE-CONTENT-CORE_R2_CANDIDATE_COMPARISON_20261030.md)，用户于2026-10-30批准R2/R4与R5；R2内容不再改动，R5已执行并结案。

R3新核心已实现并通过结构门：[R3记录](../../reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md)、[模块](../../../taiji/native_content_core.py)、[测试](../../../tests/taiji_native/test_native_content_core.py)、[结构探针报告](../../../reports/taiji_native_core_r3_probe_20261030.json)。公开fixture 17案结构检查53/53通过；零步bundle在独立进程恢复digest一致；0 optimizer updates、未读新sealed。零步相关/无关比1.57，明确不构成判别力证据。

R3b-1修正旧图基线已接入：[模块](../../../taiji/corrected_old_baseline.py)、[测试](../../../tests/taiji_native/test_corrected_old_baseline.py)、[结构探针报告](../../../reports/taiji_native_core_r3b_baseline_20261030.json)。22/22结构检查通过；硬解析覆盖16/16公开查询；首字节监督、cut 0.08798/rescue 0.0、固定分母17/17；仅value_bias可训练（3,341），语言头冻结；零步独立进程恢复一致。`unknown`仍退回current、解析失败仍硬丢弃，作为基线限制记录，不算新核心优势。

R3b-2普通端到端基线已接入：[模块](../../../taiji/ordinary_baseline.py)、[测试](../../../tests/taiji_native/test_ordinary_baseline.py)、[结构探针报告](../../../reports/taiji_native_core_r3b2_ordinary_20261030.json)。11/11结构检查通过；历史全文进入prompt、无硬解析门/无离散store；cut 0.02211/rescue 0.0、首字节监督、固定分母17/17；120,097参数全部可训练；零步独立进程恢复一致。零步三组answer_exact均为0，不构成能力证据。

三组对象（R3新核心、R3b-1修正旧图、R3b-2普通端到端）现已齐备，R4 定价与 R5 比较均已完成；R5 结案 `FAIL_BEHAVIOR`，R6 未准入。

R4-prep数据已物化：[分区](../../../plans/manifests/native_core_r1_split_v1.json)、[train](../../../plans/manifests/native_core_r1_train_v1.json)、[query](../../../plans/manifests/native_core_r1_query_v1.json)、[holdout](../../../plans/manifests/native_core_r1_holdout_v1.json)、[审计报告](../../../reports/taiji_native_core_r1_course_audit_20261030.json)。432个可能历史取360：216/72/72，另72 unused；每槽6/2/2，反事实成对不跨分区；4 episodes/history、8 turns/episode，1,440 episodes/11,520 turns/5,760 answer turns；实际文本452,160字节；train/query/holdout分文件写入，仅train可读；无模型调用、无sealed读取。精确成本已由R4定价给出（见下）。

R4合同与定价已执行：[定价与R5数值门](../../reference/SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md)、[定价报告](../../../reports/taiji_native_core_r4_pricing_20261030.json)、[训练模块](../../../taiji/native_content_training.py)、[定价脚本](../../../scripts/training/price_native_content_core_r4.py)、[测试](../../../tests/taiji_native/test_native_content_training.py)。三组同train/同曝光/同seed/同预算；512更新×15 run（Q1 9 + Q2 6）；200,000 CPU秒/48小时上限。实测单价：新核心5.45 CPU秒/更新、修正旧图0.94、普通端到端8.20；零步与第25步各对象同进程+新进程恢复6/6一致；pilot loss均有限、无NaN；投影78,750 CPU秒/1.88小时，冻结512更新。定价只证明成本与恢复，不代表能力；query/holdout/sealed未读。

R4预检已通过（不训练）：[报告](../../../reports/taiji_native_core_r4_preflight_20261030.json)。4个真实train episode、每对象16应答轮/460目标字节；三组前向+反向有限；新核心reader/decoder/core/query/attention有梯度、content_encoder无（持久latent边界）；修正旧图仅bias可训练；普通基线全参数可训练；零步独立进程恢复一致；`optimizer_updates=0`、query/holdout未读。forward+backward的17.27 CPU/4.69 wall只是下界，不是R4单价。

R5固定分母评分器已实现：[模块](../../../taiji/native_content_evaluator.py)、[测试](../../../tests/taiji_native/test_native_content_evaluator.py)。分母由声明的answer-turn列表决定，缺失/解析失败/拒答计错；cut/rescue只在baseline正确且cut错误的case上计算；swap按期望答案判对。3/3测试通过。

R5训练与评价已执行并结案：[训练运行器](../../../scripts/training/run_native_content_core_r5.py)、[评价运行器](../../../scripts/training/evaluate_native_content_core_r5.py)、[训练报告](../../../reports/taiji_native_core_r5_training_20261030.json)、[评价报告](../../../reports/taiji_native_core_r5_evaluation_20261030.json)。15/15 run、7,680更新、1.09墙钟小时、42,109 CPU秒；恢复15/15；无NaN/无预算停止。结案 `FAIL_BEHAVIOR`：holdout answer_exact 0.323（门槛0.80）、current 0.173、previous 0.207、unknown 0.737、state 0.750；cut_drop 0.166（门槛0.50）；训练内插0.329（门槛0.85）；收据0.429（门槛0.95）。通过项：比最好基线高0.169（CI下界0.150）、query高0.150、Q2 credit_on−stop差0.117（CI [0.095,0.137]）、rescue 1.0、交换差CI下界0.020、保持差距−0.006、成本0.714、恢复全过。归因：存在真实但不足的基线净收益；答案未稳定经过选中内容（收据0.429、cut 0.166，Q2 +0.117 说明reader/decoder信用仍有部分作用）；自由生成与teacher forcing差距显著（train loss 0.08 vs 内插exact 0.329）。

R1完成证据：[统一语义合同](../../reference/SPEC-NATIVE-CONTENT-CORE_R1_SEMANTIC_CONTRACT_20261030.md)五项边界确认；世界审计（360-history/216-72-72/1,440 episodes/11,520 turns/0.80–1.11MB）；公开fixture 17案通过；§2.2接口审计与§2.3合同矛盾自检完成。

F1（holdout 面）与 F1b（query 面，从未读过）都结案为负结果：F1 判 `FAIL_COPY_ONLY`、F1b 判 `FAIL_NO_STATE_EFFECT`（[收束记录 §13–§14](../../reference/PROJECT_CONSOLIDATION_20261029.md)）。F1b 的门表给出真数：`cut_drop 0.333` 与 `rescue_gain 1.0` 过线、`relevant_swap_delta` 0.096（F1 为 0），但 `answer_exact 0.041`、三条差值为 0；**54/123（44%）查询轮连实体都没解析出来**，两套 sealed 面同比例。

解析覆盖确是重要缺口，但不能再将两次低分唯一解释为“状态路径没接上”。[20261030核心重审](../../reference/TAIJI_CORE_DESIGN_REVIEW_20261030.md)进一步确认：当前/旧值目标与current-only接口不匹配，F字节目标相对原trainer错位，历史可能经hidden旁路进入，cut/rescue也不是正确性因果指标。原FAIL与费用保留，具体归因须受这些混杂限制；原N178提案仅作解析诊断，不替代核心方案重审。

本轮仅形成核心设计建议与证据账，不修代码或重跑评价；已授权N178b按原预算/终点有限结案，不被本文静默停止、恢复或追加。后继新训练先完成重审所要求的语义合同、基线与比较设计，不用parser覆盖提高自动重置跨包重审计数。

## 本轮资产与状态

| 项目 | 当前状态 | 已成立结论 | 待交接 |
|---|---|---|---|
| 原N177.6 A–D | 原数值结果完成 | 固定切片收益保留，原恢复余证仍未齐 | E0历史源补证 |
| N176 | 固定输出已生成、消费登记完成 | 1360份原输出本地hash保留，F2未解析 | 不重生成、不再称F0/F1未消费 |
| E1/E2a | 合同冻结、课程生成 | 状态依赖端点/manifest存在，新sealed未物化 | 原数据边界继承 |
| E2b消费接口 | 有限完成 | 存储、选择、query-only及读出可用 | 保留负结果与复制下界 |
| F0a | 已完成训练侧标定 | 64条train/200次更新；train1.0、copy1.0、no-state0.078125 | 不代替F0b/泛化证据 |
| E3推理恢复子门 | v11/v12通过 | parse/span/base及下一字节logits严格一致 | 不代签全部训练态E3 |
| E0 | 已闭合（2026-10-29，只读） | 15臂fresh-process 15/15；历史revision逐臂复算60/60；隔离/mask矩阵8/8 | 只覆盖固定train输入；不代签能力或训练态往返 |
| E2c/训练态E3余项 | 已闭合（2026-10-29，零步） | 信用on/stop 12/12×2；零步恢复17/17；禁写组3/3 | 选择器不吃回答信用；真实链解析覆盖1/4（DEBT-G184） |
| F0b | 已完成（2026-10-29，超 wall 96 秒） | 12/12 run；每步重跑冻结链；实测 4.68 CPU 秒/更新；零/端点恢复 24/24 全真 | CPU 1,452≤2,000，wall 1,596>1,500（协议漏算终点尾段，已补预留，见收束记录§10）；F1 投影≈121,500 CPU 秒 |
| F1 | 已结案 FAIL_COPY_ONLY | holdout 面：全臂贴地板、43%查询轮未解析 | 训练面只有 16 条可解析用例 |
| F1b | 已结案 FAIL_NO_STATE_EFFECT | query 面：108 条训练用例后 answer 4.1%、cut 0.333、swap 0.096、44%未解析 | 下一步补 producer 解析覆盖 |

详细证据、失败与来源见[20261029收束记录](../../reference/PROJECT_CONSOLIDATION_20261029.md)。已确认工程与负结果作为资产继承，不按旧“下一步”重做。

## 保留的失败与限制

- 原producer三种连续值特征在对应探针中rank2、8色原型25%；新实体抽取缺口仍在，不外推所有未来表征必败。
- value-token fixture控制/恢复通过但copy-only同等，是离散复制下界。
- E3 v6 lineage gap保留；span按字节排序后v11/v12推理恢复通过，未重写旧失败。
- F0a train1.0不是未见能力；预检的model_calls=0只表示未调用完整producer/V4链，并非没有训练更新。

## 执行清单与准入

| 工作包 | 具体任务 | 验收/边界 |
|---|---|---|
| 重构-1 | [统一语义合同](../../reference/SPEC-NATIVE-CONTENT-CORE_R1_SEMANTIC_CONTRACT_20261030.md) | 设计级闭合：五项边界、世界/17案fixture、§2.2接口审计；未实现/训练 |
| 重构-2 | [候选与三类基线合同](../../reference/SPEC-NATIVE-CONTENT-CORE_R2_CANDIDATE_COMPARISON_20261030.md) | 已评审：用户2026-10-30批准R2/R4；R2内容冻结，R5仍需单独批准 |
| 重构-3 | [新核心隔离实现与零步恢复](../../reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md) | 结构门通过：53/53检查、cut/rescue、首字节监督、独立进程恢复；未训练、不称学会 |
| 重构-3b-1 | [修正旧图基线](../../reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md#6-r3b1-修正旧图基线20261030) | 结构门通过：22/22、固定分母17/17、仅value bias可训练；未训练 |
| 重构-3b-2 | [普通端到端基线](../../reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md#7-r3b2-普通端到端基线20261030) | 结构门通过：11/11、固定分母17/17、全参数可训练；未训练 |
| 重构-4-prep | [课程与分区物化](../../reference/SPEC-NATIVE-CONTENT-CORE_R1_SEMANTIC_CONTRACT_20261030.md#31-首版世界形状本轮确认) | 已完成：216/72/72、1,440 episodes、5,760 answer turns；无模型调用 |
| 重构-4 | [定价与R5数值门合同](../../reference/SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md) | 已执行（`r4_priced`）：单价5.45/0.94/8.20 CPU秒、恢复6/6、投影78,750 CPU秒/1.88h、冻结512更新；未读query/holdout |
| 重构-5 | 一次有界独立比较：Q1 9 run + Q2 6 run | 已执行并结案 `FAIL_BEHAVIOR`：holdout 0.323、cut 0.166、内插 0.329、收据 0.429 未达门；比基线高 0.169、Q2 差 0.117、成本/恢复过门 |
| 重构-5b | 内容落地生成候选：答案必须消费选中内容并缩小自由生成差距 | R5b 合同已起草（[SPEC](../../reference/SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md)）待批准；未授权实现/训练 |
| E0 | 已完成：15臂终点fresh-process parse/answer回执；B真实输入、有效mask及隔离矩阵 | 三份件＋三支仪器自带机检判据（见收束记录§8）；未重训/未重评holdout |
| CI | 修复四模块9处mypy类型错误 | 精确记录当前源码身份；本机核心类型/受影响回归及实际CI重新核验 |
| 训练态E3 | 已完成：credit-on/stop、禁写组及optimizer/RNG/cursor往返 | E2c 12/12×2 用例、E3 17/17；选择器无回答信用、真实链解析覆盖仅1/4 单列（收束记录§9） |
| F0b准备 | 12-run清单、独占计时/恢复方案、唯一曝光口径 | 不以F0a单体118.25 CPU估整候选成本 |
| F0b执行 | 在原2,000 CPU/1,500 wall限额内完整候选短标定 | 实际采样不与E0/恢复争资源 |
| F1冻结 | 数值门、预算、更新/曝光与跨包停止线 | E0、训练态E3、F0b齐备且判据可判失败后才进入 |

F合同初拟“1280 updates＝5 epochs（128 episodes/batch4）”有算术冲突：5 epochs为160步，1280为40 epochs。本轮只登记勘误，原冻结文本保留；正式定价前须选唯一口径。F0a copy-only已1.0，F1要求main超过copy-only的判别力也须在读取holdout前澄清。

## 收束实施边界

当前HEAD基线85e7267ef的fast lint失败在mypy，共9处；该修复已在`0193fc505`落地并推送，远端`CI fast lint`于`7a53ce647`转绿（更早的同job读数见收束记录§5）。测试目录与原有untracked材料未实际清理，清理候选/核验步骤见收束记录，执行留给用户。

## 训练、评价和归因的共用边界

新训练依赖所需数据/状态/消费者/更新责任、完整checkpoint零步保存与独立恢复、成本标定及固定合同。按用户2026-10-29更正，E0/E3/F0b的执行由助手承担：只读补证与零步工程验收直接做，计时/训练类工作在原授权预算内串行执行；超预算、需改冻结范围或影响他人工作树时先报具体影响。

每轮结案必须交付10的架构判决、适用任务、信息与信用路径、因果证据、竞争原因和唯一后继。工程接通、loss下降、参数变化、toy通过都不能替代完整原生认知或能力晋升。旧实耗、失败和未知尾账继续保留。

## 历史去向与文档职责

本页原逐轮流水及旧“当前/下一步”已退出活动调度，原文可由[历史恢复登记](../../reference/PROJECT_HISTORY_CONSOLIDATION_20261023.md#6-2026-10-26活动计划重编)中的固定提交、blob与LF SHA精确恢复。用户于2026-10-27要求清理已废弃checkpoint：已结案R2恢复v2的2,869个中间快照经回执哈希核验后删除，45个首尾/直接引用/正式端点和全部回执、流水保留；[清理清单](../../../reports/taiji_r2_historical_checkpoint_prune_20261027.json)逐项记录。已删中间模型不能从本机恢复或重新按文件校验，原失败和费用没有改写。

03发布唯一当前动作；10维护依赖/WBS/有限出口；09维护机制及42条来源；07维护能力门；N177合同落实施工；VISION维护更广设计和有条件候选。UI、驱动力与全自主性按各自承载，不成为本轮共同链的额外无限门。CI已有他人负责的修复继续独立协作，只检查本次影响面。

N177.1、N177.2-T1/T2/T3有限完成。T2仅覆盖同一JointTaskState仍保存输入来源的记忆消费；旧V1状态缺时间来源时歧义读取会拒绝，MemoryRecord无独立owner/scope身份。T3的结构化Goal target来自接口fixture；未测试中文目标解释、环境执行或结果学习。默认Adapter未替换，不代表自然理解或跨会话学习。

N177.1状态机纳入response_complete、awaiting_outcome、cancelled及resolved complete四种语义。结束标记只关闭本轮答复；若ActionIntent有待结Outcome，重复用户输入/turn reset/cancel均拒绝，unknown结果等到同intent receipt结算，非终态失败允许继续规划。该状态名与回归由[JointTaskStateV1](../../../taiji/joint_task_state.py)实现，30项退出规范用例通过。
