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
| [08 同面后继](../MAINLINE.md#3-唯一执行队列) | [R5c 复制执行精度候选合同](SPEC-NATIVE-CONTENT-CORE_R5C_COPYEXPR_CANDIDATE_20261031.md) | 已批准并按原合同结案`FAIL_BEHAVIOR`，证据见总计划与原合同；保留内容/表达资产，不再发布待批准指令 |
| [09](../MAINLINE.md#3-唯一执行队列) | [核心算法主稿：v1.1底座＋v1.2生长信用候选](TAIJI_CORE_ALGORITHM_V1_DRAFT_20261031.md) | §24集中局部秩界、覆盖出生、重放耦合假说与反例；不是冻结实验或已实现模型 |
| [09工程接口](../MAINLINE.md#3-唯一执行队列) | [R021正式参考图接口](SPEC-TAIJI-CORE_REFERENCE_GRAPH_R021_20261031.md) | 六项唯一接口、凸分区软门、覆盖工程界和阶段边界；不授予训练 |
| [09信用判别](../MAINLINE.md#3-唯一执行队列) | [R022信用与数值判别合同](SPEC-TAIJI-CORE_CREDIT_R022_20261031.md) | 09.2.2三负责目标、FD/eps/二阶判别、零类与负对照、fp32对比与声明更新表达式；不授予训练/出生 |
| [09覆盖出生](../MAINLINE.md#3-唯一执行队列) | [R023信用覆盖出生与坐标迁移合同](SPEC-TAIJI-CORE_GROWTH_R023_20261031.md) | 09.2.3门控信用/标定/分配/出生/全局基线与§10.3迁移；校准不可分辨发现入账；不授予训练/恢复 |
| [09主体恢复](../MAINLINE.md#3-唯一执行队列) | [R024完整主体恢复合同](SPEC-TAIJI-CORE_RESTORE_R024_20261031.md) | 09.2.4 bundle schema/五边界/独立进程续推/声明更新/幂等闸；恢复通过不签语义保持；不授予训练/定价 |
| [09活动循环](../MAINLINE.md#3-唯一执行队列) | [R025活动动力学合同](SPEC-TAIJI-CORE_ACTIVITY_R025_20261031.md) | 09.3六活动循环/模型所有选择头/想象隔离/对照臂；不签学会选择或自主性；不授予训练/定价 |
| [09六臂比较](../MAINLINE.md#3-唯一执行队列) | [R026六臂预注册比较合同](SPEC-TAIJI-CORE_PLASTICITY_R026_20261031.md) | 09.4同父六臂（覆盖出生/GradMax/条件SVD/随机门/仅重放/同容量）预注册比较；裁决negative如实入账；不授予训练/定价 |
| [09恢复门](../MAINLINE.md#3-唯一执行队列) | [R027主体恢复门合同](SPEC-TAIJI-CORE_GATES_R027_20261031.md) | 09.5八边界合拢/清单核验/query-only隔离/分账；不签语义保持或自主性；不授予训练/定价 |
| [09定价冻结](../MAINLINE.md#3-唯一执行队列) | [09.6生命片段定价合同](SPEC-TAIJI-CORE_PRICING_20261101.md) | 09.6实测单价、09.7六臂/3 seed数值门、探针暴露与预算；训练准入待owner裁定；不签任务能力或成长收益 |
| [09工程补齐](../MAINLINE.md#3-唯一执行队列) | [09.6.1工程补齐合同](SPEC-TAIJI-CORE_ENGINEERING_20261101.md) | 09.6后置五项缺口（§10.2约束更新/活动策略学习/lesion/中文夹具/子进程CPU）；owner已批准实现、未批准09.7训练；不签任务能力或自主性 |
| [09夹具改造](../MAINLINE.md#3-唯一执行队列) | [09.6.2夹具改造合同](SPEC-TAIJI-CORE_FIXTURES_20261101.md) | 反复触发到达日程＋2×3生长网格；查明活动面失败源于`TRACE_LAMBDA=0`的两步信用、生长面四次独立否证且排除"训练太慢"；不签自主性或结构收益 |
| [09方向规则](../MAINLINE.md#3-唯一执行队列) | [09.6.4方向规则候选合同](SPEC-TAIJI-CORE_DIRECTION_20261101.md) | fit生涯多子窗共识方向（不泄漏评估集）；预注册份额门被自身随机对照否掉，判`metric_not_discriminating`；不签结构收益 |
| [09绝对判据](../MAINLINE.md#3-唯一执行队列) | [09.6.5绝对贡献判据合同](SPEC-TAIJI-CORE_DIRECTION_ABS_20261101.md) | 以"绝对贡献≥1e-4且随机对照<1e-4、3种子≥2支持"替换失效的份额门；**种子轴被实测证明无效（父模型跨全局种子逐位相同），结论作废、旧报告保留** |
| [09方向复现](../MAINLINE.md#3-唯一执行队列) | [09.6.6真实变化轴复现合同](SPEC-TAIJI-CORE_DIRECTION_REPLICATION_20261101.md) | 变化轴改为规则窗口数parts∈{2,3,4}；2/3格过绝对门（1.426e-4／1.343e-4，parts=4为7.3e-5）、随机对照全在门下、基线逐位复现；薄边际，不签任务能力或09.7准入 |
| [09稳定性面](../MAINLINE.md#3-唯一执行队列) | [09.6.7稳定性面合同](SPEC-TAIJI-CORE_DIRECTION_SURFACE_20261101.md) | `parts∈{2..6}`×`units∈{1,2,3}`共15格全发布，预注册"平台≥6格且≥3个parts"；随机对照与基线同轮；不签成本偿付或09.7准入 |
| [09净收益](../MAINLINE.md#3-唯一执行队列) | [09.6.8净收益对照合同](SPEC-TAIJI-CORE_NETBENEFIT_20261101.md) | 同父同8步"只重放 vs 共识出生"；净收益 +1.739e-4、出生额外11.50 CPU秒（≈6.6秒/每1e-4 NLL，定价输入非门槛）；单一确定性主体，不签任务能力或09.7准入 |
| [09跨生涯复核](../MAINLINE.md#3-唯一执行队列) | [09.6.9跨评估生涯复核合同](SPEC-TAIJI-CORE_NETBENEFIT2_20261101.md) | 同一两臂在HOLDOUT与新声明的第二评估生涯上复测：+1.739e-4／+1.903e-4 ⇒ 净收益不是单条夹具产物；两条均为合成夹具，不签任务能力或09.7准入 |
| [09.7片段](../MAINLINE.md#3-唯一执行队列) | [09.7一次有界联合片段合同](SPEC-TAIJI-CORE_JOINT_FRAGMENT_20261101.md) | 单点候选parts=5/units=3，32步分四检查点；主判据=两条评估生涯净收益在片段末仍≥1e-4且保持≤0.030；parts=4预先排除；不签任务能力或长程偿付 |
| [09.7b可持续点](../MAINLINE.md#3-唯一执行队列) | [09.7b可持续片段长度定位合同](SPEC-TAIJI-CORE_HORIZON_20261101.md) | 逐检查点记保持漂移；仅8步在门内（净收益1.739e-4／1.903e-4），16步起两臂同步越线 ⇒ 卡点是训练预算不是出生 |
| [09.8收束](../MAINLINE.md#3-唯一执行队列) | [09.8全局收束与能力/资源卡合同](SPEC-TAIJI-CORE_CLOSEOUT_09_20261101.md) | 六件收束物（bundle／活动轨迹／父子映射／参数继承／原始输出／能力资源卡）＋三项一致性（恢复、继承、轨迹非退化）；卡片禁出现能力声称；不授予训练 |
| [10.1任务环境](../MAINLINE.md#3-唯一执行队列) | [10.1 T3自然目标与有限行动范围冻结合同](SPEC-TAIJI-CORE_T3_GOAL_20261101.md) | 2对象×3位置、动作ID逐步重排、终态评分只读终态；三条设计判据（换Goal改最优、分区完整、四条捷径均不能满分）；不涉及模型、不授予训练 |
| [10.2目标绑定](../MAINLINE.md#3-唯一执行队列) | [10.2自然Goal绑定合同](SPEC-TAIJI-CORE_GOAL_BINDING_20261101.md) | 绑定契约＋五类公开反例（改写/否定/合取/缺条件/歧义）＋规则基线；缺条件与歧义强制拒答；模型侧绑定未交付 |
| [10.3预测选择](../MAINLINE.md#3-唯一执行队列) | [10.3预测与选择合同](SPEC-TAIJI-CORE_GOAL_PLANNER_20261101.md) | 预测/想象/实际Outcome分账＋五条对照臂；契约与判别力成立，但规划与贪心打平 ⇒ `planner_no_advantage`，任务需加深 |
| [10.1b任务加深](../MAINLINE.md#3-唯一执行队列) | [10.1b位置容量与阻挡合同](SPEC-TAIJI-CORE_T3_GOAL_10_1B_20261101.md) | 位置容量1使贪心失效；默认路径逐位不变；加深后规划在2格严格优于贪心；不涉及模型 |
| [10.4模型侧选择](../MAINLINE.md#3-唯一执行队列) | [10.4主体自身参与选择合同](SPEC-TAIJI-CORE_SUBJECT_SELECTION_20261101.md) | 09.8同一候选对动作句算条件NLL取最低；主体2.81=随机2.81<贪心3.86 ⇒ `model_below_random`；19/19全选杯子句=固定字面偏好，P2仍未满足 |
| [10.5目标条件化](../MAINLINE.md#3-唯一执行队列) | [10.5目标条件化选择接口合同](SPEC-TAIJI-CORE_GOAL_CONDITIONED_20261101.md) | 匹配/错配目标下的 NLL 对比度；两动作 contrast 差仅 8.018e-5＜1e-3 ⇒ `contrast_below_floor`；目标效应是共模的，P2 仍未满足 |
| [10.6动作特定](../MAINLINE.md#3-唯一执行队列) | [10.6动作特定构造合同](SPEC-TAIJI-CORE_ACTION_VERDICT_20261101.md) | 动作进前缀的是非题；动作差 4.804e-4＜1e-3 且方向相反 ⇒ `action_gap_below_floor`；三条免训练接口均未给出随目标区分动作的信号 |
| [10.7有界训练](../MAINLINE.md#3-唯一执行队列) | [10.7第一次有界训练合同](SPEC-TAIJI-CORE_TRAINING_10_7_20261101.md) | 12样本按局面6/6划分；lr=1e-4损失降37%但留出仍3/6、漂移0.0322破门；机制=训练把常量偏好从印章换成杯子，未造出条件化；权重不发布 |
| [10.11收束卡](../MAINLINE.md#3-唯一执行队列) | [10阶段收束依据与负结果卡合同](SPEC-TAIJI-CORE_CLOSEOUT_10_20261101.md) | 只读复核 10.4–10.7 与 11：同一候选、四项出口在册、无能力声称、命中序列交替（各3/6）、保持门单侧破门、损失比0.6335、权重不出库、11仍不准入 ⇒ `closeout_10_evidence_complete`；`p2_satisfied=false`、`stage_close_decision=owner_pending`，不训练 |
| [10.7e读出实验](../MAINLINE.md#3-唯一执行队列) | [10.7e输入条件读出合同](SPEC-TAIJI-CORE_READOUT_10_7E_20261101.md) | 主体冻结、只训 1,024 参数秩 4 双线性头；留出 6/6、标签对调对照 0/6、未训练 6 平局 ⇒ `readout_binding_supported`；夹具级、不构成 P2，不授权追加训练 |
| [10.7f六选一](../MAINLINE.md#3-唯一执行队列) | [10.7f六选一选择合同](SPEC-TAIJI-CORE_SELECTION_10_7F_20261101.md) | 主体冻结、1,536 参数头读目标＋局面＋动作；训练损失 5e-06 完全拟合但留出 top-1 只有 2/6（＝未训练常量）、错配对照 0/6 ⇒ `readout_selection_absent`；能背不能迁 |
| [11准入门](../MAINLINE.md#3-唯一执行队列) | [11 L3用户验收准入合同](SPEC-TAIJI-CORE_L3_ADMISSION_20261101.md) | 四项前置机检（L2／模型侧代表能力／安全恢复性能／同候选身份）；当前四项全假 ⇒ `step11_not_admitted`；不授予训练 |
| [11.7仪器读数](../MAINLINE.md#3-唯一执行队列) | [11.7候选仪器读数合同](SPEC-TAIJI-CORE_INSTRUMENT_11_7_20261101.md) | 20 次打分逐位可重复、p50 0.120 s／p95 0.125 s、峰值 1.483 GB（打分再抬 0）、评分期 0 写／0 socket／0 子进程 ⇒ `instrument_measured_quiet`；并入 11 门后 P3 翻真，准入仍 `step11_not_admitted` |
| [09历史证据](../MAINLINE.md#3-唯一执行队列) | [v1.1/E1独立审计](TAIJI_CORE_ALGORITHM_V1_EVALUATION_20261031.md) | 对象64fdf8576；原字节/可读件映射见入库身份，不是v1.2重新裁决 |
| [算法重审设计输入](../MAINLINE.md#执行位置) | [09/10结果与算法修订建议](TAIJI_ALGORITHM_REVIEW_20261101.md) | 生成成功门可达性、条件读取/联合信用、保持语义与生长正例范围；不批准训练 |
| [10.21](../MAINLINE.md#执行位置) | [目标条件读取与绑定学习原规格](SPEC-TAIJI-CORE_COMPOSITION_10_21_20261101.md) | 原合同与reader_selection_absent负结果保留，现已由后继阶段接力 |
| [14.3](../MAINLINE.md#执行位置) | [读源条件化原合同](SPEC-TAIJI-CORE_READ_SOURCE_14_3_20261102.md) | 原pathway_unreachable退出保留；旧读取与新读取的路径差异、小距离的可声称边界见14.4起修订 |
| [复审结论](../MAINLINE.md#执行位置) | [15.1结案与原版N16在途复审](REVIEW-PACKET_20261102.md) | 子集/配对/gamma解释核对，按当前运行结果形成后继 |
| [14.4–14.8](../MAINLINE.md#执行位置) | [记录关系读取与原生字节表达](SPEC-TAIJI-CORE_RECORD_RELATION_14_4_20261102.md) | 已按原出口执行/收束，14.7未释放，局部正读数与任务负结果保留 |
| [15.1](../MAINLINE.md#执行位置) | [原文绑定与后验连续复制原规格](SPEC-N15-01_BINDING_COPY_V2_20261102.md) | 已按任务负结果收束，评分/归因修正见当前复审，原卡与确认封存留存 |
| [15.1原提案](../MAINLINE.md#执行位置) | [14.8绑定的历史原件](PLAN-N15-01_SPAN_BOUND_CONTINUOUS_20261102.md) | hash绑定原字节保留，已由rev2取代当前要求，不并行发布排期 |
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

完整清单与统一规则见[LEGACY_ASSET_INDEX](LEGACY_ASSET_INDEX.md)。新核心复用/适配/重新实现以及权重、数据和实际删除条件集中见该页§5；合同映射不表示旧候选仍是当前主模型。

## 3. 使用规则

- 当前工作包以总计划中链接的合同为准；引用本页第 2 节时只能作证据、基线或背景。
- 旧文档中的“当前/下一步/唯一动作”只解释其写作时点，不产生当前指令。
- 冻结 SPEC 与报告保持路径、字节和 Git 身份；本页不因收敛改写原规则。
- 新合同必须先在[MAINLINE](../MAINLINE.md)第 3 节登记阶段，才能成为“当前合同”。

## 4. 如何查具体实验

按 `PLAN-N1/N2/N5`、`SPEC`、`EXP`、`execution_closeout`、`APPROVAL` 命名族检索 `plans/reference`，再读对应合同/结案。机器结果看 `reports/README.md`；历史结构见 `plans/archive/README.md`。未完成步骤维护在总计划，证据和审批维护在 03，运营债务沿原债表；整理不改变原结论或授权。

| 当前责任 | 文档 | 范围 |
|---|---|---|
| N16原版A | [已裁规格](SPEC-N16-01_SELECTION_LOCKED_BINDING_DRAFT_20261103.md) | 原0a–0d已交付，原16.1.1在途 |
| 结案后条件候选 | [逐步选择路由](DESIGN-POST-N16_SELECTION_ROUTING_20261103.md) | 16.1.3据结果判断采用和编号 |
