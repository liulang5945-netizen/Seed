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

[已批准合同](../../reference/SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md)保留原候选、两项改动、9-run清单、门值、预算及审批。提交`dc7a43354`提供[native_content_core_v2.py](../../../taiji/native_content_core_v2.py)、[训练适配器](../../../taiji/native_content_training_v2.py)、评价适配及[11个定向测试定义](../../../tests/taiji_native/test_native_content_core_v2.py)；本轮只读代码与提交，不运行测试，不能据文件存在签工程门通过。

静态对照发现的待核项：批准合同按episode抽样，适配器按turn抽样且缩短自前缀路径的实际监督字节；50步pilot不覆盖257后分支，草拟额外分支探针需具体方法/更新账修订。原合同不在本轮改写，细项及处置顺序已写入总计划08。后续真实工程/定价/训练/评价回执到位后再追加身份与费用；并行草稿不计完成。

提交`b78bdfdce`追加[定价执行件](../../../scripts/training/price_native_content_core_r5b.py)、[训练执行件](../../../scripts/training/run_native_content_core_r5b.py)、[评价执行件](../../../scripts/training/evaluate_native_content_core_r5b.py)；本轮未运行；执行件存在不等于工程/训练完成。收尾核对时工作区出现`reports/taiji_native_core_r5b_pricing_20261030.json`，status为`r5b_priced`，登记60更新、约572.56 CPU秒、约59.00墙钟秒，零步/两阶段端点同进程和新进程恢复均记相等，query/holdout/sealed读取均记false。本次未提交该并行机器件；原50步与60步差异、实际日程分支及监督字节仍需按总计划08审计，不凭status补签全部前置门。

项目所有者已正式将[总计划](../../MAINLINE.md)定位为新的N主线。03继续仅登记证据与审批；09–13详细方案不因此成为已冻结训练合同或默认模型采用批准。

### 08.1 合同—实现对照表（2026-10-31 实施方交付，偏差已按原合同纠正）

「原要求→实现→处置→证据」四栏对照。原实现（`dc7a43354`）存在三处与批准合同的偏差，均已纠正；纠正后 12/12 定向测试＋13/13 R5 回归测试绿（本轮实测运行，非仅文件存在）。

| 合同要求（R5b §2.1/§5/§7） | 原实现 | 处置 | 证据（定向测试，本轮实测绿） |
|---|---|---|---|
| §2.1-A 复制轨：读取路径选记录，被选记录 value span 原始字节经指针分布进入字节分布，与词表 softmax 按 gate 混合 | `next_logits`：`_read` 选 top-1→span 字节 `copy_key/copy_query` 位置 softmax→`index_add` 成 257 维 copy 分布→`copy_gate(cat(updated,read))` 混合取 log | 一致，未改 | `test_copy_track_math_is_the_declared_mixture`（混合公式逐步复算 allclose） |
| §2.1-A/§5 通道不接触 gold；宿主伪造选择必须被拒 | teacher/answer/scheduled 全部要求 store 绑定；`content_read_cache` 校验每条 condition 记录的 `record_sha256`+`raw_span` 属于声明 store | 一致，未改 | `test_forged_condition_record_is_rejected` |
| §2.1-A copy 参数归 decoder 信用组（Q2 stop 仍恰为 reader+decoder） | `credit_group_of("copy_*")→decoder` | 一致，未改 | `test_credit_stop_freezes_copy_parameters` |
| 空 store/切断退化＝v1 逐字节 | 空 content_ids 直接返回 v1 raw logits | 一致，未改 | `test_copy_track_reduces_to_v1_without_content`、`test_cut_store_degenerates_to_v1` |
| §2.1-B 抽样单位＝**每 episode** 以 0.5 概率 | coin 按 `episode_id+turn_index` 逐应答轮抽样 | **偏差①纠正**：coin 去掉 `turn_index`，只按 (seed, update, episode_id) | `test_own_prefix_coin_is_pure_and_declared` |
| §2.1-B 教师强制**目标字节不变，仅换前缀来源** | 自前缀分支只监督后半段（监督字节缩短为 len−len//2） | **偏差②纠正**：scheduled 分支对**全目标**产出监督行，仅前半段的输入源换成自身贪心字节；曝光（监督字节数与目标）与纯 TF 分支逐字节一致；行数≠目标长度即抛错 | `test_scheduled_forced_conditioned_is_deterministic_and_byte_exact`（rows=len(target)）、`test_v2_adapter_loss_terms_byte_accounting`（两分支 supervised_bytes 相等） |
| §7/08.3 探针须证明**实际执行的分支**、日程位置与监督字节 | 原 phase B 强设 `schedule_update=300` 被 `R5BRunTrainer.update()` 每次重置回 `steps+1`（**phase B 实测的是纯 TF**；已产出的定价报告失真） | **偏差③方法修订**：新增 `R5BRunTrainer.schedule_position_override` 声明式探针机制，逐更新记账 `schedule_position/schedule_override_active/episodes_scheduled/scheduled_supervised_bytes`；定价脚本 phase A 断言 0 scheduled、phase B 断言 >0，否则工程失败 | `test_trainer_override_keeps_probe_on_declared_branch`；失真报告 `reports/taiji_native_core_r5b_pricing_20261030.json`（未提交）已删除，定价按修订方法在 08.3 重跑 |

runner（`b78bdfdce`）另补 `--run-id`（默认 20261031），避免与已停草稿的写一次目录冲突。

**并行草稿账（不计任何回执）**：按偏差纠正前的实现启动的 9-run 训练已按 08.1 停止，停在 `checkpoints/native_core_r5b/20261030/`（3 个端点 seed0/1/2＋q2_credit_on_seed0 零步，无报告）；耗 CPU 约 1.5 万秒，按 R5b §7 计入同一 120,000 CPU 秒总预算账。作废定价件 60 更新约 572.56 CPU 秒同样计入。预算余量约 10.4 万 CPU 秒，覆盖修订后重定价（≈0.1 万）、正式 9×512 训练（≈4.3 万）与评价（≈1 万）。

08.1 对照表不能代替 08.2 工程回执；下一步按总计划 08.2 交付信用/隔离/训练态恢复证据（含 256→257 边界与恢复后下一次更新）。

### 08.2 工程回执（2026-10-31 实测，`r5b_082_ok`）

[回执](../../../reports/taiji_native_core_r5b_082_engineering_20261031.json)、[探针脚本](../../../scripts/training/probe_native_content_core_r5b_082.py)、实现 `1315ff959`。五族门全过（无 failed_gates）：

- **信用**：同初态 credit_on/stop 前向逐位一致；逐 loss 组读数——on 侧 reader/decoder/core/attention/output/query_encoder 全部有梯度，stop 侧 reader+decoder 零梯度、冻结参数逐位不变、其余组确认位移（optimizer 真实生效）。
- **gold 与隔离**：宿主伪造 condition 记录（含可复制 span 的外来记录）被 store 绑定校验拒绝；跨 scope 读取为空、本 scope 恰 1 条。
- **退化与切断/救援**：带内容时复制轨激活（与 v1 分歧）；空 store 下 v2 与 v1 在**同一 cut condition** 下逐字节相等；救援答案与基线逐字节一致。
- **旧 v1 可加载**：R5 冻结端点 `q1_new_core_v1_seed0` 以原报告记录 digest（e8550592…）一致加载。
- **256→257 日程边界与训练态恢复**：连续 257 更新 vs 256 处 checkpoint 新进程恢复后下一更新——schedule 位置 256（0 scheduled）→257（2 scheduled，coin 按声明触发）；恢复后下一更新 loss 与连续 run **逐位相等**（0.3529213984619896）；恢复后端点 probe digest 与连续 run 端点一致（8ee2b741…）；cursor（run_id/seed/frozen_groups）、optimizer 步数、exposures 恢复记账在回执。

账目与作废记录：工程探针共 515 优化器更新、正式回执 4,707 CPU 秒/405 墙钟秒；此前三次脚本缺陷（函数序 NameError、GATES 键名 KeyError、写一次残留）与一次探针比较 bug（误把 V2 空 store 对 V1 带内容比较；实现无缺陷）各作废一跑，累计约 1.4 万 CPU 秒，连同作废回执（未提交已删）一并计入预算账。**累计已耗约 3.5 万/120,000 CPU 秒，余约 8.5 万**，覆盖 08.3 重定价（≈0.06 万）、08.4 正式 9×512（≈4.3 万）与评价（≈1 万）。08.2 是工程门，不代能力；query/holdout/sealed 未读。

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
