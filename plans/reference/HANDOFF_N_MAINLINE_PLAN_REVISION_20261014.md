# N 主线调度修订交接（2026-10-14）

> **2026-10-15 最新调度**：08状态流48对/96查询完成，CPU155.078125秒；09补零位置48臂完成，CPU77.5秒，三臂严格命中仍均0/16。低成本诊断自主推进，当前唯一待决为10的核心证据/信用学习路线；非几分钟预算审批。见 [09裁决](EXP-N1-09_C2_padding_placement_adjudication_20261015.md) 与 [10路线评审](PLAN-N1-10_native_evidence_credit_route_review_20261015.md)。
本件用于接续用户“根据项目了解修订开发计划”的成果。动态队首只看 [03](../active/roadmap/03_CURRENT_EXECUTION.md)，调度看 [09](../active/roadmap/09_NEXT_MAINLINE_PLAN.md)，完整修订看 [PLAN-N-01](PLAN-N-01_development_revision_20261014.md)。先现读 Git HEAD 与工作区，不能把本件时点当作新实验身份。

## 已完成

- 新主线：C0 证据/合同→C1 持久载体→C2 生成消费→C3 信用→C4 净增益→C5 联合能力。
- [N1 实施附件](PLAN-N1-06_carrier_consumer_alignment_20261014.md)已解决设计层配置/恢复、生产消费互斥、生成轨迹来源及信用可测性关系；精确代码触点和新消费/信用数值门待其各自版本冻结。
- C1 批⑵按 owner 后续批准的≤75分钟 CPU 上限通过；96项／288轮 open/off 逐项一致，288 episode 独立恢复与旧档负对照通过。完整报告哈希与结果见 PLAN-N-01 §7、08 ㊵-875。
- [C2 校准预注册](PLAN-N1-07_C2_consumer_calibration_prereg_20261014.md)已冻结16项七臂零训练机制校准；仅为单器官可行性诊断，累计≤12分钟预算已批准并完成112臂，CPU尾端601.90625秒，消费者与32条支持保存/恢复通过；详见 [C2完整裁决](PLAN-N1-07_C2_calibration_result_adjudication_20261015.md)，不能代替三器官出口。
- 原工程队首已存 [历史快照](../archive/history/n_mainline_plan_revision_20261014/EXECUTION_QUEUE_BEFORE_REVISION.md)。G137/G115 残余未删：v5 为 54靶/53文件、15暴露、160未解析形式，旧v4不能与它直接比较。
- 冷归档四项原件已移除；复算前按 [恢复说明](../../outputs/README.md)取回。负结果、冻结合同与模型权重没有删除。
- 后续用户批准 C1 批⑴实现与批⑵零训练面，并将批⑵预算由≤30重定价为≤75分钟。批⑴/批⑵均已完成；当前唯一队首为 C2 本次累计CPU总上限是否由10重定价至12分钟，见 [03](../active/roadmap/03_CURRENT_EXECUTION.md)。

## 当前决策边界

PLAN-N1-06 r1 的 C1 规格已实现并按完整面通过。两臂均为96项／288生成／44,963步，`per_item`逐项相同、0失败；open checkpoint 经独立进程恢复288 episodes／1152维，缺失episode为空、反馈形状兼容、恢复后记录关闭；旧母档负对照通过。C1没有训练、默认行为变更或消费者接入。后续 C2 单器官完整校准已取得；model选择8/16、七臂严格命中均0/16，08/09已完成，10核心学习接口路线是当前方案决策。2026-10-15长期授权替代逐批权限等待，冻结范围/判据/测面和不可逆动作保留讨论。

不要直接跑旧 PLAN-N1-04 的训练臂：当前 fabric.step/observe/generate 为 no_grad，旧首步梯度门不可直接执行；“只重放首步就应零梯度”亦不成立。新版本先冻实际信用通路与负对照，参考可微通路不能冒充产品信用。

## 后置清单

- N6 与当前包无关的全库仪器统一/大重构保留积压，不能替代能力研究。
- B/G两臂80题回填、当前HEAD零并发整族汇总、目标提交根CI必需作业与manifest，在相应能力/采用出口关闭。
- N2/N3/N5依据主要失效选一个最小有界包，不重复跑已完成负实验；新增预算单独批。
- M7发布、M8规模化、多模态和默认采用按能力与质量前置后置。

2026-10-15 最新目标修订：按N主线持续推进并依实际情况调整计划，保持Taiji核心定义。11隔离逐位置参考工程已完成：CPU7秒，前向逐位相同的信用截断与零更新/优化器独立恢复通过；不代表原生信用或能力。全项目仍为C1通过、C2无严格收益、C3原生候选/训练及C4未完成；唯一下一步是原生写入/读取/学习归属合同。详见 [11工程裁决](EXP-N1-11_position_evidence_reference_adjudication_20261015.md)。

2026-10-15 连续推进补充：12原生Observation隔离候选接口落地，13候选自身两新进程恢复通过（CPU6.75秒）；候选/参考与仓库守卫27 passed。无默认替换、无能力训练或净增益结论。唯一下一步为分离课程/同信息三臂协议及无更新预检，保持N主线方向。详见 [13裁决](EXP-N1-13_native_candidate_restore_adjudication_20261015.md)。

2026-10-15 14预检完整：v1两处捷径已拒绝采用，v2随机实体/每类两观察，train/dev/holdout=96/32/32；CPU1.453125秒，12条train前向/backward、参数不变、optimizer0、holdout查询0。唯一建议为14三seed/三臂共3456更新、独立300秒CPU小试；新测面与训练范围准入仍须决定，尚未训练。见[14裁决](EXP-N1-14_curriculum_preflight_adjudication_20261015.md)。

2026-10-15 用户已批准14三seed/三臂3456更新、独立300秒CPU小试，并赋予同类低成本实验自主准入。14决策在此范围内关闭；15运行器已完成，正式训练前16项相关合同通过，readiness为零更新/零查询。当前唯一下一步为按[SPEC15](SPEC-N1-15_paired_runner_execution_contract_20261015.md)执行已冻完整包；未获能力/默认采用。

2026-10-15 15完整负结果：3456更新、576题与18次实际新进程恢复全齐，CPU81.796875秒<300；三seed/三臂holdout均0/32，全部自然停止。原门不放行，不据CE下降声称能力。唯一下一步为九模型×96训练项零更新拟合诊断，不新增dev/holdout查询。见[15裁决](EXP-N1-15_paired_pilot_adjudication_20261015.md)。

2026-10-15 17固定学习曲线完整：同架构/lr/Adam、单seed11逐位置隔离副本追加3072更新，CPU35.3125秒，480组train-only，原档/源码不变、两恢复通过。已见exact0→24→35→53→90/96，末端条件字节99.03%，不计未见或对照优势。静态课程审计发现train仅9答案类，dev的d/holdout的m/q/z无正目标监督；不删除困难字符，下一步按[18](PLAN-N1-18_fresh_content_curriculum_20261015.md)扩训练词值/字母覆盖及新独立留出面。见[17裁决](EXP-N1-17_train_only_learning_curve_adjudication_20261015.md)。

2026-10-15：18新课程准入完成，train/dev/holdout 192/32/32，96个不同训练目标词，正目标26字母最少15次；12项零更新预检通过，CPU1.578125秒，未查询dev/holdout。见[18裁决](EXP-N1-18_v3_curriculum_admission_20261015.md)。唯一后继为v3独立train-only充分性校准，先定学习剂量/成本再冻结新三臂配对。


19待执行：SPEC-N1-19_v3_train_sufficiency_calibration_20261015.md、scripts/training/run_taiji_n1_v3_train_calibration.py，6144更新/独立300秒CPU、0 dev/holdout；26项前置合同通过。无需重复请求同类低成本权限。


2026-10-15：[19裁决](EXP-N1-19_v3_train_sufficiency_adjudication_20261015.md)完整6144更新/960组train读数/两独立恢复，CPU64.328125秒；已见93/192，lookup2/revision1/transition44/unknown46（各48），训练充分性门未过，dev/holdout查询0。唯一后继为全部96词值train项的零更新内容寻址诊断，不直接九模型重跑。


20待执行：scripts/training/audit_taiji_n1_train_content_read.py，SPEC20已冻、29项前置合同通过；0更新/留出、60秒CPU，只诊断19终点。


2026-10-15：[20内容诊断裁决](EXP-N1-20_train_content_read_adjudication_20261015.md)完整，CPU2.703125秒，288自由/96辅助读数、0更新/dev/holdout，参数/源档/代码/manifest不变。正确来源词值替换96项0正确、58项输出变；干扰来源50项输出变，不能把敏感或注意力当绑定。唯一建议[21观察保真辅助学习](PLAN-N1-21_observation_fidelity_learning_review_20261015.md)，新增目标超出QA-only范围，需用户保留的方案决策；CPU同类授权已有，不再次请求小额资源权限。全项目C1过/C2零收益/C3内容绑定与未见保持/C4/三器官及其他N债仍欠。


2026-10-15：用户批准21观察保真辅助学习；已实现Taiji独立loss模块，不增参数/推理头，cut覆盖QA与辅助两支。按[SPEC22](SPEC-N1-22_observation_fidelity_execution_contract_20261015.md)先60秒CPU无更新工程预检和独立zero恢复，再自动执行6144更新/192train、独立300秒CPU校准，0 dev/holdout。32项前置合同通过。未来真实决策使用选项弹窗，不依赖决定的工作继续。全项目未见/保持/C4/三器官及其余N债仍欠。


2026-10-15：[22裁决](EXP-N1-22_observation_fidelity_adjudication_20261015.md)完整6144更新/960组/两恢复，CPU67.328125秒；QA121/192（16/13/46/46）、词值29/96，保真99.9666%，未达173及各类40门。0 dev/holdout；与19同初始化/材料/更新但非同CPU，单seed不证明优势。唯一后继[SPEC23](SPEC-N1-23_fidelity_content_read_diagnostic_20261015.md)复用20替换词复审内容寻址，零更新/留出，独立60秒CPU。全项目其余欠账仍在。


2026-10-15：[23裁决](EXP-N1-23_fidelity_content_read_adjudication_20261015.md)完整，CPU3.25秒、96词值train项/288自由/96gold-prefix/288保真，0更新/dev/holdout且全部守卫通过。原exact29/96，正确词替换0/96、输出全96项变，替换目标字节局部重构100%；干扰换词exact12/96。局部保真不等于绑定。唯一建议[24输出前两次内部重读](PLAN-N1-24_pre_emission_iterative_read_review_20261015.md)，新增读取语义，已用选择弹窗请用户决定；依赖步骤等待实际答复，证据/测试/提交继续。C1过/C2严格收益零/C3未见保持/C4/三器官及其余N债保留。


用户已在选择弹窗批准24，无需重复确认；57项相关验证通过，提交23证据后直接进入两hop候选实现和恢复预检。


2026-10-15：用户通过选择弹窗批准24，独立Taiji两hop候选及[SPEC25](SPEC-N1-25_iterative_read_execution_contract_20261015.md)运行器就绪，参数40337不变；中间hop无输出/来源写入，格式明确two-hop，旧one-hop不能静默恢复。31项前置合同通过，Ruff/B/SIM/Black通过。唯一队首先60秒CPU零更新工程预检/zero恢复，通过自动6144更新/独立300秒CPU train-only，0 dev/holdout。全项目尚未能力收官。



2026-10-15：[25校准](EXP-N1-25_iterative_read_adjudication_20261015.md)完整，6144更新/960组，CPU90.953125秒，zero/final独立恢复与全部守卫过；train115/192、词值26/96，未过充分性门，未优于22的121/192。相关62测过，0 dev/holdout。按原失败出口自动执行[SPEC26](SPEC-N1-26_iterative_content_read_diagnostic_20261015.md)同材料替换诊断，不加训练/机制；全项目能力与CI债保留。


2026-10-15：[26内容复审](EXP-N1-26_iterative_content_read_adjudication_20261015.md)完整，CPU3.03125秒，原exact26/96、正确换词0/96、干扰换词保留7/96，正确替换局部字节100%，0更新/dev/holdout，守卫全过。24两hop未显示绑定收益，保留25失败；唯一建议[27地址干预诊断](PLAN-N1-27_address_intervention_diagnostic_review_20261015.md)，已弹出选择，实际答复前不执行新干预。全项目C1过/C2零收益/C3未见保持/C4/三器官及其余N面与CI债未齐。

2026-10-15：用户已在选择弹窗批准27，队首转为其零训练地址干预实施/原生控制验证，不再等待决定；原576序列/60秒CPU/零留出合同保持。

2026-10-15：27已弹窗批准；离线地址干预运行器完成，6项工程合同通过（2.68秒），覆盖单/双hop原生逐位一致、源mask和one-hot、boundary原生、gold previous与留出拒绝。即按冻结576条序列/CPU≤60秒执行，不训练、不更改候选。


2026-10-15：[27地址诊断](EXP-N1-27_address_intervention_adjudication_20261015.md)完整，576条gold-prefix、CPU4.5秒、0更新/自由输出/dev/holdout，原生控制/全部守卫过。22单hop六字节exact原生29/来源11/位置43，25双hop26/16/79（含原生boundary仅48）；oracle不是能力，也不证明定位唯一根因。68相关测过。唯一建议[28学习型相对地址状态](PLAN-N1-28_learned_relative_address_review_20261015.md)，新增132参数及回答内地址分布，已弹窗请求，实际答复前不实施。全项目C1过/C2严格收益零/C3未见保持/C4/三器官及其余N与CI债未齐。


2026-10-15：用户通过选择弹窗批准28，Taiji隔离相对地址候选/29运行器就绪，新增132参数总40469、旧层初始化同25；7工程测通过（4.56秒），源内质量/梯度、首步及零门、问题隔离、teacher/free轨迹、信用和Adam恢复过。唯一队首[SPEC29](SPEC-N1-29_relative_read_execution_contract_20261015.md)先60秒CPU零更新独立预检，通过自动6144更新/960组/300秒CPU train-only；0 dev/holdout，原门及全项目能力/CI债保留。


2026-10-15：[29相对地址校准](EXP-N1-29_relative_read_adjudication_20261015.md)完整6144更新/960组，CPU108秒，zero/final恢复及全部守卫过，train190/192、四类48/47/48/47，充分性门通过；0 dev/holdout、75相关测过。只是已见单seed，不代表未见/保持/C4或三器官过。唯一建议[30三seed三臂小试](PLAN-N1-30_relative_paired_review_20261015.md)55,296更新/独立CPU1200秒，九模型与充分性/恢复冻结后首次v3 holdout，沿14原行为门；已弹窗请求测面准入，实际答复前不启动。全项目其余N/根CI及保持债不变。

2026-10-15：用户在选择弹窗批准30，实施冻结三seed三臂运行器；获准55,296更新/独立1200秒CPU，仅前置齐备后首次v3留出，dev不查、原门不变。
