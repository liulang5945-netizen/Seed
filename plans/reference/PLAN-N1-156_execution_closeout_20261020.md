# PLAN-N1-156 执行结案

**日期：2026-10-20**  
**状态：完整结束；仅为train-only描述性归因**  
**计划：** [PLAN-N1-156](PLAN-N1-156_plan155a_train_score_itemwise_attribution_20261020.md)  
**报告：** [逐题配对归因](../../reports/taiji_n1_plan156_plan155a_train_score_itemwise_attribution_20261020.json)

## 结果

对N1-155A已消费的7,488条评分完成逐题配对：5,760条原训练门评分、1,728条长度评分；36个长度×seed×kind分层与54个训练门配对分层均覆盖。progress及N1-155A绑定来源哈希复核一致。CPU2.859375/60秒，0模型/检查点加载、0推理、0查询、0参数更新、0 dev/holdout/confirmation，`can_promote=false`。

seed12 control的48条transition在original和rename条件都为30/48；对应candidate均为48/48，每组有18条candidate-only正确、0条control-only正确。四个factorial条件下candidate也均48/48，control分别29–31/48，反向错转对为0。seed12 unknown原训练题candidate为47/48、control40/48，7条仅candidate正确、0条仅control正确；rename为48/48对41/48，同样有7条仅candidate正确。以上均为同一已消费训练评分集的逐题差异，不能据此声称新任务泛化或内部机制因果关系。

长度评分中的unknown在8/16/32行，candidate与control合计分别为67/72对66/72、69/72对69/72、71/72对71/72。8→16、16→32的逐题变化未显示candidate在各seed上稳定占优；所以本批不能确认“unknown短上下文再平衡”带来稳健的长度特异提升。第一个停止门仍是N1-155原冻结门：seed12 uniform control transition不足，fresh face没有生成或查询。

## 后续唯一队首

依据用户授予的执行计划批准权，已批准[PLAN-N1-157](PLAN-N1-157_seed128_candidate_vs_parent_fresh_diagnostic_20261020.md)：首次生成并消费未用过的seed128 fresh face，比较三个N1-155A candidate final与各自N1-150 `length_mixed`父模型，共2,880条查询。此为描述性候选对父模型诊断，不调整原训练门、不晋级，也不把缺失的N1-155 uniform control改写为通过。
