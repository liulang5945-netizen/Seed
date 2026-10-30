# N主线执行证据与审批台账

> 文档状态：[状态] — 只登记执行证据、审批、费用与适用限制；统一编号、阶段状态和实际动作集中在[总计划](../../MAINLINE.md)。

更新：2026-10-30，Asia/Shanghai。本页原执行表已整合进总计划，不再发布第二套顺序或复制待办。以下按统一步骤关联证据，阶段记录只解释当时，不能把零步时“未训练”当作后续现状。

## 当前唯一下一步：见总计划的执行位置

直接按[总计划「执行位置」](../../MAINLINE.md#执行位置)推进。本标题仅保留导航兼容，不独立维护候选、状态、审批或下一动作。

## 当前候选的审批依据

| 统一步骤 | 批准/边界事实 | 原始依据 |
|---|---|---|
| 02、06、07 | 用户于2026-10-30批准R2/R4与R5，R5现已按原门结案 | [R4页首及§9](../../reference/SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md) |
| 08 | 用户于2026-10-30选择“批准，按合同执行”；该合同实现与训练授权已成立 | 提交`8780f858e`与[R5b §10执行记录](../../reference/SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md#10-审批方式)；本次整理不实施 |
| 09–13 | 原R6能力接力不由R5/R5b批准自动授权 | 对照总计划各步所需新范围/合同，不借原批准越门 |

## 步骤01–06的交付证据

| 步骤 | 交付出处 | 读数与适用限制 |
|---|---|---|
| 01 | [R1合同](../../reference/SPEC-NATIVE-CONTENT-CORE_R1_SEMANTIC_CONTRACT_20261030.md)、[世界审计](../../../reports/taiji_native_core_r1_world_audit_20261030.json)、[17案fixture审计](../../../reports/taiji_native_core_r1_public_fixture_audit_20261030.json) | 设计语义及公开反例，不是未见任务能力 |
| 02 | [R2合同](../../reference/SPEC-NATIVE-CONTENT-CORE_R2_CANDIDATE_COMPARISON_20261030.md) | Q1三类架构组，Q2图内reader/decoder信用比较；架构包与单因素分账 |
| 03 | [R3记录](../../reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md)、[结构探针](../../../reports/taiji_native_core_r3_probe_20261030.json) | 53/53结构门，公开fixture、零步独立恢复；本阶段0更新，后续训练见07 |
| 04 | [修正旧图探针](../../../reports/taiji_native_core_r3b_baseline_20261030.json)、[R3 §6](../../reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md#6-r3b1-修正旧图基线20261030) | 22/22，固定分母17/17，仅value_bias可训3,341；unknown回current、解析失败丢弃等基线限制保留 |
| 05 | [普通基线探针](../../../reports/taiji_native_core_r3b2_ordinary_20261030.json)、[R3 §7](../../reference/SPEC-NATIVE-CONTENT-CORE_R3_ISOLATED_IMPLEMENTATION_20261030.md#7-r3b2-普通端到端基线20261030) | 11/11，固定分母17/17，全文prompt、无硬解析门、120,097参数全可训 |
| 06数据 | [课程审计](../../../reports/taiji_native_core_r1_course_audit_20261030.json)、[分区身份](../../manifests/native_core_r1_split_v1.json) | 360历史取216/72/72；1,440 episodes、11,520 turns、5,760应答轮；反事实成对不跨分区；本台账不加载分区正文 |
| 06工程 | [零更新预检](../../../reports/taiji_native_core_r4_preflight_20261030.json) | reader/decoder等组有任务梯度，content_encoder无（持久latent边界）；0 optimizer updates，不称已学会 |
| 06定价 | [定价报告](../../../reports/taiji_native_core_r4_pricing_20261030.json) | 新核心/旧图/普通单价5.45/0.94/8.20 CPU秒/更新；3×50 pilot；零步/第25步恢复6/6；冻结512更新/run。仅该定价阶段未读query/holdout/sealed |

## 步骤07的结果、费用与限制

原始来源：[训练报告](../../../reports/taiji_native_core_r5_training_20261030.json)、[评价报告](../../../reports/taiji_native_core_r5_evaluation_20261030.json)、[原合同§9](../../reference/SPEC-NATIVE-CONTENT-CORE_R4_PRICING_AND_GATES_20261030.md#9-r5-执行结果20261030)。固定分母评分器为[native_content_evaluator.py](../../../taiji/native_content_evaluator.py)。

| 读数 | 实际结果 | 解释边界 |
|---|---|---|
| 执行与恢复 | 15/15 run，7,680更新；恢复15/15；无NaN/无预算停止 | 完成训练不代表过能力门 |
| 整体结果 | `FAIL_BEHAVIOR` | 按原冻结门；不追改阈值 |
| 相对比较 | holdout约0.323，比最好基线高约0.169；Q2信用差约0.117 | 有真实但不足的相对收益，绝对行为仍不合格 |
| 关键失败项 | cut_drop约0.166，训练内插约0.329，收据约0.429 | 内容选择和生成路径值得优先检验；不能凭这些单项穷尽全部原因 |
| 父能力保持 | 报告`retention_parent=none` | 训练内插不代证旧能力保持 |
| CPU费用 | run CPU和42,108.75秒；训练进程总CPU42,224.15625秒；评价另6,529.890625秒 | 不将run和当全部实耗，不漏评价；完整ledger取原报告 |
| 数据使用 | 评价已读query/holdout，sealed未读 | 后继复用相同面是回归比较，不重新称独立确认 |

## 步骤08的证据入口

[已批准合同](../../reference/SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md)保留原候选、两项改动、9-run清单、门值、预算及审批。实现、定价与结果的现状只更新[总计划](../../MAINLINE.md#执行位置)；有真实交付后在此追加对应代码revision、报告与费用，不用“计划已写”填成完成。

## 继承的历史事实与限制

| 历史资产 | 已有事实 | 保留边界 |
|---|---|---|
| N175/N176 | N175 R为0/3，S未全判；N176固定1,360份生成与消费登记已完成，F2未解析 | 按原输出/hash/合同留存，不重复生成或借新计划代判 |
| N177.1–177.5 | 状态机、来源消费、结构化T2/T3、T1 producer与恢复已有有限工程证据 | 结构化Goal/状态不代证自然目标理解、环境学习或跨会话能力；默认Adapter未替换 |
| E0 | 15臂fresh-process 15/15，历史revision复算60/60，隔离/mask 8/8 | 仅固定train输入，不代签任务能力 |
| E2c/训练态E3 | 信用on/stop 12/12×2，零步恢复17/17，禁写组3/3 | 旧选择器无回答信用，真实链解析覆盖1/4单列 |
| F0a/F0b | F0a train/copy均1.0；F0b 12/12 run，实测4.68 CPU秒/更新，恢复24/24 | F0a不代泛化；F0b wall超96秒如实留账 |
| F1/F1b | 分别`FAIL_COPY_ONLY`/`FAIL_NO_STATE_EFFECT`，低分与费用不撤销 | 目标字节错位、时态接口、hidden旁路和指标混杂限制归因；旧cut/rescue不代正确性因果收益 |
| N178 | face2解析局部改善，22/29/24/64，非完整回答收益 | 采样seed不等于全参数独立初始化；面已用于选剂量，后继属开发诊断，不重置F累计停止线 |

详账见[阶段收束](../../reference/PROJECT_CONSOLIDATION_20261029.md)、[核心设计重审](../../reference/TAIJI_CORE_DESIGN_REVIEW_20261030.md)、[旧资产索引](../../reference/LEGACY_ASSET_INDEX.md)。历史表的处置不构成当前待办。

## 历史恢复、CI与资料归属

原逐轮正文可从[历史恢复登记](../../reference/PROJECT_HISTORY_CONSOLIDATION_20261023.md#6-20261026活动计划重编)与Git恢复。历史R2中间checkpoint清理已有[逐项回执](../../../reports/taiji_r2_historical_checkpoint_prune_20261027.json)，该既有事实不授权本轮进一步删除。

历史fast lint修复及回执见收束记录；历史CI绿不代当前提交CI绿。每包只检查受影响范围，保留无关未提交工作。计划放plans，机器报告放reports，模型和数据沿受控根；当前排期全部回总计划，03不再复制执行清单。
