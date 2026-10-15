# SPEC-N1-31 · 已批准30九模型运行合同（2026-10-15）

用户已通过选择弹窗批准[30](PLAN-N1-30_relative_paired_review_20261015.md)。独立运行器默认只读；`--preflight`先三seed×三臂首train零更新信用/参数检查及一次zero独立恢复（独立CPU≤60秒），成功且代码identity一致后`--execute-approved`才执行。未动29冻结实现。

正式九fresh模型按seed11/12/13、各position/terminal/cut固定6144更新，合55,296；参数/目标/数据/臂语义、train/holdout/行为门完全沿30。每seed初始摘要相同，seed11旧层还对25 zero；每模型更新前zero与末尾final独立worker恢复，正式共18。签名复用29含参数/Adam/RNG/来源、首/第二步logits与地址、辅助logits，完整协议另版，不能和29混用。

最终train192×9＝1728组（每组teacher+自由），九模型训练及18恢复都齐、三个position均≥173/192且各类≥40/48，才冻结九final档SHA/参数/代码/协议并写`pre_holdout_freeze.json`，之后完整holdout32×9＝288组。dev始终0，teacher只作副读数。缺前置拒绝模型访问；任何position不足则完整记录九训练结果并判training-insufficient，0 holdout、不补训。训练充分性失败不删seed/臂。

正式累计CPU≤1200秒含父与18worker，单线程，触边/错误保存partial、计时不完整拒判、不重跑不缩分母。所有报告/档exclusive，不覆盖旧文件。全原始train/holdout输出、标签及teacher数、自然boundary/cap、每步loss、初末摘要/恢复签名留存。每seed≥16/32、每类≥2/8、position三seed总分别比terminal/cut多≥12才候选推进；不代替保持/三器官或默认采用。

运行前10相关工程测试通过（3.90秒），包含三臂/恢复/训练分母前置、holdout/dev拒绝、双控制/逐类别行为门、模型与Adam续接。相关CI债及全项目能力债保留。

正式训练启动前复核发现协议继承了29单模型的seed/arm/updates/milestones/holdout_queries元数据，虽然实际执行分支已按九模型/留出门实现，披露合同仍冲突。因此移除这五个单模型字段，明确holdout_rows_if_sufficient=288；新增回归断言。原零更新预检报告保留为v1过程证据，不作为修正后源码的准入。修正后独占v2预检报告/目录，累计计入v1的6.78125秒CPU，工程总上限仍60秒。此前正式训练/留出均0；无配方、判据或测面变化，不重跑正式批次。


2026-10-15：31协议元数据修正冻结9d200a42c，v2预检passed，累计CPU14.0625秒含v1的6.78125秒，零更新/留出；正式九模型已启动。首个seed11 position终点190/192，独立恢复完成，累计CPU97.67秒。78相关合同通过（7.13秒）；尚未取得九模型对照或v3留出结果。


2026-10-15：[31裁决](EXP-N1-31_relative_paired_adjudication_20261015.md)九模型55,296更新、1728组train、18独立恢复完整，CPU939.796875秒，全部工程/源档守卫过；position190/180/180，但seed12/13 transition38/39<40，status=training-insufficient，can_advance=false。holdout/dev均未查询，报告留出0不是0/32成绩。31按原合同封存失败，不在其内补训。唯一建议[32另版九模型统一续接3072](PLAN-N1-32_fixed_paired_continuation_review_20261015.md)，额外27,648更新/独立600秒CPU，原门不变，已弹窗待决定。78相关测试过；C1工程过/C2原零收益/C3跨seed充分与未见保持/C4/三器官和其他N/根CI债仍在。
